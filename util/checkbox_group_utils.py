"""
Geometric checkbox-GROUP detection (template registration / Pass 1).

The OCR-anchored detector (`detect_checkboxes_visual`) finds individual
selection boxes for the semantic layer, but it never groups the options of a
single question (Male / Female ; Salaried / Self-Employed...).  This module
groups them into first-class `checkbox_group` template fields so extraction
can return the actual list of checked option texts instead of N booleans.

Architecture (adaptive geometry + OCR text, deliberately NOT fixed-pixel):

   Stage 1 - box candidates.  Two morphological passes over the binary image
   (closed external contours, plus CCOMP interior holes for rings that touch
   gridlines/underlines).  Every candidate is gated with a set of rules
   anchored to the PAGE'S OWN TEXT HEIGHT instead of absolute pixels, so the
   same detector handles a 12px museum-form box and a 67px trade-form box:

       * text-height scale: box side in [0.7x, 2.2x] the median text height,
       * squareness: aspect ratio in [0.8, 1.2] (text entry boxes are far
         wider than tall and are rejected by this alone),
       * rim classifier: a genuine selection box is a dark closed frame with a
         comparatively empty centre (`_ring_density` + `_classify_window`),
       * inner-text veto: a box that contains printed words is a table cell /
         boxed label, not a selection box,
       * grid rejection: a box sharing >= 2 edges with long ruling lines
         (morphological open, length >= 3 x max box side) is part of a ruled
         grid.  A lone checkbox merely sitting on an underline shares one edge
         and is kept.

   Stage 2 - option binding: from each checkbox project a horizontal ray left
   and right and bind the nearest OCR element whose vertical center sits on
   the checkbox's Y-baseline (tolerance from text height, reach capped at
   3 x box width).  Prefers right-side text (the option label), falls back
   left.  Leading selection marks are stripped from the bound label.

   Stage 3 - clustering: a strict "same printed line" band gates the
   page-relative horizontal reach (0.3 x page width), so a row of long-labelled
   options ("Disability ... includes: Loss of limbs / Blindness") never fuses
   into the question BELOW it.  Stacked option lines (Male / Female, a 2x2
   grid) merge through a relaxed x-overlap band, except two stacked boxes that
   carry the SAME bound label - repeated "Yes" under distinct sub-questions are
   separate questions, never one group's options.

The parent label is NOT chosen here: `build_template_fields` runs the shared
`_label_for_region` against each macro-box, the same rule that labels every
other template field (searches Left then Up).

Group schema (pixel space of the processed image):

    {
        "id": "cbg_000",
        "bbox": [x1, y1, x2, y2],          # macro-box
        "options": [
            {"text": str, "bbox": [x1,y1,x2,y2]},   # the checkbox bbox
        ],
    }
"""

import cv2
import numpy as np

from util.image_utils import _load_gray, _threshold_gray
from util.checkbox_utils import (
    _classify_window,
    strip_leading_option_mark,
)
from util.geometry_utils import _intersection_over_area

# Box glyphs an OCR engine reports as the first character of an option label
# when a form prints its selection box as a character rather than as drawn ink.
_BOX_GLYPHS = ("☐", "☑", "☒", "□", "▢", "◻", "❑", "❒")


def _leads_with_box_glyph(text):
    return bool(text) and text[0] in _BOX_GLYPHS


def _inside_any(bbox, boxes):
    x1, y1, x2, y2 = bbox
    return any(
        x1 >= bx1 and y1 >= by1 and x2 <= bx2 and y2 <= by2
        for bx1, by1, bx2, by2 in boxes
    )
from util.line_utils import detect_line_segments, merge_collinear_horizontal


def _ring_density(bin_img, x1, y1, x2, y2, band_frac=0.16):
    """Outer-band vs central ink density for a candidate box window.

    `_classify_window` samples a 1-px rim, which is fine for large boxes but
    the rim of a small printed box (e.g. 12x12) occupies most of the window and
    shows up as "interior ink", so a fixed 1-px inner test rejects it.  Here the
    band thickness scales with the box so the test means the same thing for any
    box size: a box is a rim-only shape (dark frame, empty middle).
    """
    h, w = bin_img.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    win = bin_img[y1:y2, x1:x2]
    hh, ww = win.shape[:2]
    if hh < 4 or ww < 4:
        return None
    band = max(1, int(round(min(hh, ww) * band_frac)))
    outer = np.zeros_like(win, dtype=bool)
    outer[:band, :] = True
    outer[-band:, :] = True
    outer[:, :band] = True
    outer[:, -band:] = True
    center = np.zeros_like(win, dtype=bool)
    center[band:-band, band:-band] = True
    outer_dark = float(np.mean(win[outer] > 0))
    center_dark = float(np.mean(win[center] > 0)) if center.any() else 0.0
    return outer_dark, center_dark


def _ring_side_darkness(bin_img, x1, y1, x2, y2, band_frac=0.16):
    """Dark fraction of each of the four scaled border bands plus the centre.

    `_classify_window` inspects a strict 1-px rim, which only works when the
    candidate window hugs the ring exactly.  A ring whose box was derived from
    its interior HOLE (`_checkbox_candidates` pass 2) is necessarily a little
    larger than the ring itself, so that 1-px rim falls in blank margin and the
    ring looks "missing".  Here the band thickness scales with the window (same
    geometry as `_ring_density`), so a genuine ring sitting anywhere near the
    border still darkens every band, while a lone ruling line / text edge only
    darkens the one or two bands it crosses.

    Returns (top, bottom, left, right, center) in [0,1] or None when the window
    is too small to measure.
    """
    h, w = bin_img.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    win = bin_img[y1:y2, x1:x2]
    hh, ww = win.shape[:2]
    if hh < 6 or ww < 6:
        return None
    band = max(1, int(round(min(hh, ww) * band_frac)))
    top = float(np.mean(win[:band, :] > 0))
    bot = float(np.mean(win[-band:, :] > 0))
    left = float(np.mean(win[:, :band] > 0))
    right = float(np.mean(win[:, -band:] > 0))
    inner = win[band:-band, band:-band]
    center = float(np.mean(inner > 0)) if inner.size else 0.0
    return top, bot, left, right, center


# ---------------------------------------------------------------------------
# Stage 1 - adaptive checkbox contours
# ---------------------------------------------------------------------------

def _checkbox_scale(median_text_h):
    """Expected selection-box side range (min, max) anchored to text height.

    A real drawn box is roughly one text line tall.  Anchoring the accepted
    size to the page's OWN median text height (0.7x..2.2x) replaces the older
    fixed-pixel `MIN_REAL_SCALE` (which killed small-box forms such as the
    12px museum form while doing nothing for large-type forms) without
    admitting printed letter glyphs: those are ~0.3x-0.5x text height.
    """
    mh = float(median_text_h or 20.0)
    if mh < 4.0:
        mh = 20.0
    low = max(10.0, round(0.7 * mh))
    high = max(round(2.2 * mh), low + 2)
    return low, high


def _grid_line_masks(bin_img, min_len):
    """Long-ruling masks for the grid-rejection gate.

    MORPH_OPEN with a kernel of `min_len` keeps only dark runs at least that
    long.  It answers "is this box edge shared with a grid line longer than
    3 x box side?" - the line must be a real ruling, never the checkbox's own
    tiny ring (a ring is far shorter than the kernel).
    """
    hh, ww = bin_img.shape[:2]
    h_len = max(int(min_len), 12)
    v_len = max(int(min_len), 12)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min(h_len, ww), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min(v_len, hh)))
    hmask = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, h_kernel)
    vmask = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, v_kernel)
    return hmask, vmask


def _edge_line_fraction(mask, bbox, edge):
    """Dark fraction along one box edge (sampling the 2 boundary rows/cols to
    absorb sub-pixel placement of the box vs the ruling)."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = mask.shape[:2]
    xa, xb = max(x1, 0), min(x2, w)
    ya, yb = max(y1, 0), min(y2, h)
    if xb <= xa or yb <= ya:
        return 0.0
    fracs = []
    if edge == "top":
        for yy in (y1, y1 + 1):
            if ya <= yy < yb:
                fracs.append(float(np.mean(mask[yy, xa:xb] > 0)))
    elif edge == "bottom":
        for yy in (y2 - 1, y2):
            if ya <= yy < yb:
                fracs.append(float(np.mean(mask[yy, xa:xb] > 0)))
    elif edge == "left":
        for xx in (x1, x1 + 1):
            if xa <= xx < xb:
                fracs.append(float(np.mean(mask[ya:yb, xx] > 0)))
    elif edge == "right":
        for xx in (x2 - 1, x2):
            if xa <= xx < xb:
                fracs.append(float(np.mean(mask[ya:yb, xx] > 0)))
    return max(fracs) if fracs else 0.0


def _is_grid_cell(box, hmask, vmask):
    """Grid rejection: a box sharing >= 3 edges with long ruling lines is a
    table cell enclosed by a grid, not a selection box.

    A 2-edge rule is too aggressive - real forms print option boxes inside
    bordered option rows (top + bottom rulings around each label row), so a
    genuine checkbox in that layout shares exactly two edges.  Squareness +
    the size anchor (<= 2.2x text height) already discard rectangular table
    cells; this gate only kills the rare fully-enclosed square cell that slips
    past every other veto."""
    edges = sum(
        _edge_line_fraction(hmask, box, e) >= 0.4 for e in ("top", "bottom")
    )
    edges += sum(
        _edge_line_fraction(vmask, box, e) >= 0.4 for e in ("left", "right")
    )
    return edges >= 3


def _has_inner_word(box, elements):
    """Inner-text veto: an OCR text whose bbox is entirely inside the box means
    the box is a table cell / boxed label, not a selection box.  Single glyphs
    that could be a handwritten mark (X, tick) are not treated as words."""
    x1, y1, x2, y2 = [float(v) for v in box]
    for el in elements or []:
        eb = el.get("bbox")
        if not eb or len(eb) != 4:
            continue
        if len(str(el.get("text") or "").strip()) < 3:
            continue
        if eb[0] >= x1 and eb[1] >= y1 and eb[2] <= x2 and eb[3] <= y2:
            return True
    return False


def _is_selection_box(bin_img, x, y, w, h):
    """Rim/centre classifier + centre-fill state gate for a candidate window.

    A valid selection box is a closed dark frame with a comparatively empty
    middle: 1-px perimeter almost fully dark (`_classify_window` edge_dark),
    the scaled band (`_ring_density`) dark on the rim and light in the centre.
    The centre-fill test is intentionally strict: a genuine empty ring's centre
    is nearly blank on a blank form, while printed glyphs / solid graphics
    whose counter mimics a ring still fill ~25% of the middle (the density of
    printed text).  Requiring `center_dark < 0.20` (measured on real blank-form
    rings: ~0.00, on the header-glyph counter that slipped through: 0.25)
    rejects boxed text and graphic counters while keeping every thin ring.

    The 1-px rim test only fires for a window that hugs the ring exactly.  A
    window derived from a hole contour also hugs it (the hole contour traces the
    ring's inner edge), which is why the hole pass tries the un-padded bbox
    first; a window padded by a guessed rim thickness lands in blank margin on
    every side, where neither this rim test nor the scaled side-bands can
    recover the ring.
    Returns True/False.
    """
    r = _ring_density(bin_img, x, y, x + w, y + h)
    if r is None:
        return False
    outer_dark, center_dark = r
    rc = _classify_window(bin_img, x, y, x + w, y + h)
    if rc is None:
        return False
    edge_dark, _ = rc
    if not (center_dark < 0.20 and center_dark <= 0.6 * outer_dark):
        return False
    if edge_dark >= 0.65:
        return True
    bands = _ring_side_darkness(bin_img, x, y, x + w, y + h)
    if bands is None:
        return False
    if min(bands[:4]) >= 0.25:
        return True
    # A ROUND selection mark (radio button, or a hand-drawn/printed circle) has
    # no straight sides, so the 1-px rim test above scores near zero: the rim
    # only crosses the window border near the four axis points instead of
    # running along its edges.  `edge_dark` for such a shape lands around 0.18
    # and it is rejected as "not a frame", even though it is exactly as valid a
    # selection mark as a square.
    #
    # A circle is distinguished from a straight-edged frame by its CORNERS: on a
    # square all four corners are dark; on a circle they are blank.  Measure the
    # corners of the scaled band window and accept when they are light while
    # every side band stays dark, i.e. the ink runs edge-to-edge through the
    # middle of each side but stops short at the corners.
    return _is_round_ring(bin_img, x, y, w, h, bands)


def _is_round_ring(bin_img, x1, y1, w, h, bands=None):
    """True when the window holds a closed ROUND selection mark.

    Checks that (a) the middle of all four sides is dark, so the shape really
    does enclose the window, (b) the four corners are light, which is what
    separates a circle from a square/rectangle frame, and (c) the interior is
    comparatively empty so a filled blob or printed glyph cannot qualify.
    """
    ih, iw = bin_img.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(iw, x1 + w), min(ih, y1 + h)
    win = bin_img[y1:y2, x1:x2]
    hh, ww = win.shape[:2]
    if hh < 8 or ww < 8:
        return False
    if bands is None:
        bands = _ring_side_darkness(bin_img, x1, y1, x2, y2)
        if bands is None:
            return False
    top, bot, left, right = bands[:4]
    # Every side's midpoint must be inked, otherwise nothing is enclosed.  A
    # circle only clips ~20% of each band (its widest point is a single point
    # at the middle), whereas a square inks the whole band, so the round test
    # admits a lower floor than the square rim test above.
    if min(top, bot, left, right) < 0.15:
        return False
    band = max(1, int(round(min(hh, ww) * 0.16)))
    corner = max(2, int(round(min(hh, ww) * 0.18)))
    cs = [
        win[:corner, :corner],           # top-left
        win[:corner, -corner:],          # top-right
        win[-corner:, :corner],          # bottom-left
        win[-corner:, -corner:],         # bottom-right
    ]
    if any(float(np.mean(c > 0)) > 0.30 for c in cs):
        return False  # dark corners -> a square frame, not a circle
    # Reject a diagonal/graphic shape: the centre band must stay light.
    cy0, cy1 = hh // 2 - max(1, hh // 8), hh // 2 + max(1, hh // 8)
    cx0, cx1 = ww // 2 - max(1, ww // 8), ww // 2 + max(1, ww // 8)
    if float(np.mean(win[cy0:cy1, cx0:cx1] > 0)) >= 0.30:
        return False
    # Confirm the ink really forms a closed ring rather than four disconnected
    # side marks, by its RADIAL profile: a circle puts its ink in one
    # mid-radius annulus while leaving both the middle and the outside clear.
    # (A flood-fill test was tried first and is unreliable here: a thin rim
    # often touches the window edge, so the fill leaks in and reports no hole.)
    cy, cx = (hh - 1) / 2.0, (ww - 1) / 2.0
    rad = np.hypot(
        np.arange(ww, dtype=np.float32)[None, :] - cx,
        np.arange(hh, dtype=np.float32)[:, None] - cy,
    )
    R = max(min(ww, hh) / 2.0, 1.0)
    ink = win
    inner = rad < 0.55 * R
    outer = rad > 1.10 * R
    band_in = (rad >= 0.75 * R) & (rad < 1.10 * R)
    if not (inner.any() and outer.any() and band_in.any()):
        return False
    if float(np.mean(ink[inner])) > 0.06:
        return False
    if float(np.mean(ink[outer])) > 0.06:
        return False
    # The annulus must be genuinely inked all the way round, not on one arc.
    return float(np.mean(ink[band_in])) >= 0.10


def _checkbox_candidates(
    gray,
    elements=None,
    median_text_h=None,
    exclude_bboxes=None,
):
    """Detect small square selection boxes from pixel geometry.

    Two complementary passes (union, deduped on proximity):

      1. contour pass - morphological close + RETR_EXTERNAL, then gate with a
         text-height-anchored size/aspect range plus the rim/centre, inner-text
         and grid-rejection classifiers;
      2. hole pass    - RETR_CCOMP ring+hole pairs.  A selection box rim that
         TOUCHES surrounding gridlines merges with the grid into one giant
         connected contour after the close, so the external pass cannot isolate
         it.  The interior HOLE of such a ring is still a distinct contour, so
         rings touching gridlines survive here.

    Both passes keep only windows whose side fits the page's text-height scale,
    whose aspect is square-ish, and whose frame/centre genuinely looks like a
    selection box.  Returns a list of
    {"bbox", "width", "height", "center"} candidates with center outside every
    exclusion bbox.

      3. glyph pass    - some forms print the box as a CHARACTER ("U+2610
    BALLOT BOX", "U+2611", "U+2612", "U+25A1") that the OCR engine reads as
    part of the option's own text ("<box> Male").  There is no drawn square in
    the pixels for the contour passes to find, so those options are invisible
    to pure geometry.  Here each glyph-led element yields a candidate sized
    from the text's own height and positioned where the glyph sits.
    """
    bin_img = _threshold_gray(gray)
    low, high = _checkbox_scale(median_text_h)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    # A page frame / dark scan border can become one giant connected component
    # that encloses the entire page.  With RETR_EXTERNAL that frame is then the
    # only external contour and every interior selection box is returned as a
    # hole (i.e. hidden).  Sever the outermost few pixels so a frame can no
    # longer enclose the page; selection boxes never sit that close to the edge.
    h, w = closed.shape[:2]
    m = max(2, int(round(min(h, w) * 0.006)))
    closed = closed.copy()
    closed[:m, :] = 0
    closed[-m:, :] = 0
    closed[:, :m] = 0
    closed[:, -m:] = 0

    # Grid ruling masks: a line is "long" when it exceeds 3x the largest box
    # we would accept, guaranteed longer than any candidate box's own side.
    hmask, vmask = _grid_line_masks(bin_img, min_len=3.0 * high)

    exclude = exclude_bboxes or []
    cands = []

    def _accept(x, y, cw, ch):
        """Applies every non-geometry gate once we know the position/size."""
        side = min(cw, ch)
        big = max(cw, ch)
        if side < low or big > high:
            return False
        aspect = cw / float(ch) if ch else 0.0
        # A selection box is roughly square; real forms print option boxes
        # that are a bit wider than tall (slightly rectangular), so the band
        # is a little wider than a strict square. Text entry boxes and ruling
        # segments are FAR wider than tall (aspect >> 1.4) and are still
        # rejected by this alone.
        if not (0.75 <= aspect <= 1.4):
            return False
        bbox = [float(x), float(y), float(x + cw), float(y + ch)]
        cx = x + cw / 2.0
        cy = y + ch / 2.0
        if any(ex[0] <= cx <= ex[2] and ex[1] <= cy <= ex[3] for ex in exclude):
            return False
        if not _is_selection_box(bin_img, int(x), int(y), int(cw), int(ch)):
            return False
        if _has_inner_word(bbox, elements):
            return False
        if _is_grid_cell(bbox, hmask, vmask):
            return False
        return True

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if not _accept(x, y, cw, ch):
            continue
        cands.append({
            "bbox": [float(x), float(y), float(x + cw), float(y + ch)],
            "width": float(cw),
            "height": float(ch),
            "center": [x + cw / 2.0, y + ch / 2.0],
        })

    # ---- Pass 1b - isolated ring components (no morphological close) ----
    # Pass 1 works on `closed`, a 3x3 morphological CLOSE of the whole page.
    # That close is what lets a rim that merely TOUCHES a rule or a neighbouring
    # glyph fuse into one blob, and it is also why a selection box printed in
    # light ink can vanish: the close dilates the faint 1px rim into its
    # neighbour, so RETR_EXTERNAL never returns a standalone square.  The rim is
    # still a clean connected component in the UNCLOSED binary, so probe
    # components directly there and gate them with the same checks.  This
    # recovers forms whose boxes are drawn but too faint or too close to other
    # ink to survive the close.
    cc2, lab2, st2, _ce2 = cv2.connectedComponentsWithStats(
        (bin_img > 0).astype(np.uint8), 8
    )
    for i in range(1, cc2):
        x, y, cw, ch, area = (int(v) for v in st2[i][:5])
        if cw * ch > 4 * high * high:
            continue
        if not _accept(x, y, cw, ch):
            continue
        cand = {
            "bbox": [float(x), float(y), float(x + cw), float(y + ch)],
            "width": float(cw),
            "height": float(ch),
            "center": [x + cw / 2.0, y + ch / 2.0],
        }
        if not _same_candidate(cand, cands):
            cands.append(cand)

    # ---- Pass 2 - hole-based rings (survive gridline / underline contact) ---
    # A closed selection box whose rim merges with a long rule shares one giant
    # external contour, so the contour pass cannot bound it.  Its interior is
    # still a CCOMP hole though.  Instead of gating the giant parent (whose
    # bbox breaks the size/aspect gates) we re-derive the box from the HOLE:
    # expand the hole bbox by an estimated ring thickness, then gate that
    # window exactly like the contour pass.
    cc_cnts, cc_hier = cv2.findContours(bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if cc_hier is not None:
        cc_hier = cc_hier[0]
        for i, hole in enumerate(cc_cnts):
            if cc_hier[i][3] < 0:
                continue  # only holes (children of a ring)
            hx, hy, hw2, hh2 = cv2.boundingRect(hole)
            if hw2 < 5 or hh2 < 5:
                continue
            # A CCOMP hole contour traces the ring's INNER edge, so its bbox is
            # already the true ring window: a 1-px boundary sample lands on the
            # rim and reads fully dark.  Padding it by a guessed rim thickness
            # can push the WHOLE window out into blank margin, where BOTH the
            # 1-px rim test and the scaled side-band test read light and the
            # ring is lost.  That is what erased every checkbox on forms whose
            # rules/underlines fuse all rings into one giant connected
            # component (so the contour pass cannot bound them) - e.g. the
            # mobile-capture dealer form, where the one surviving ring window
            # went 62x62 with edge_dark 0.0 and bands 0.19 while the 50x50 true
            # window scored edge_dark 1.0.  Try the true window first and keep
            # the padded one only as a fallback for rings the hole contour
            # under-reports.
            ring = max(3, min(8, int(round(min(hw2, hh2) * 0.12))))
            for pad in (0, ring):
                x, y = hx - pad, hy - pad
                cw, ch = hw2 + 2 * pad, hh2 + 2 * pad
                if not _accept(x, y, cw, ch):
                    continue
                # Dedupe against the contour-pass candidates (same ring twice).
                if not any(
                    abs(c["center"][0] - (x + cw / 2.0)) <= max(2.0, 0.4 * min(c["width"], float(cw)))
                    and abs(c["center"][1] - (y + ch / 2.0)) <= max(2.0, 0.4 * min(c["height"], float(ch)))
                    for c in cands
                ):
                    cands.append({
                        "bbox": [float(x), float(y), float(x + cw), float(y + ch)],
                        "width": float(cw),
                        "height": float(ch),
                        "center": [x + cw / 2.0, y + ch / 2.0],
                    })
                break

    for gc in _glyph_candidates(elements, exclude_bboxes or []):
        if not _same_candidate(gc, cands):
            cands.append(gc)

    return _dedupe_candidates(cands)


def _dedupe_candidates(cands):
    """Collapse passes that rediscovered the same printed box.

    The contour pass, the unclosed-component pass and the glyph pass can all
    propose one physical box (the glyph pass when the box is printed as a
    character the OCR text already carries).  Every copy would otherwise bind
    the same option label again, so the group reports `Male, Male`.  Kept when
    copies coincide, or when one is contained in another and at least half its
    size -- a genuinely nested smaller box next to a larger one is real.
    """
    kept = []
    for cand in cands:
        if _same_candidate(cand, kept):
            # Prefer the richer geometry when the copies differ slightly.
            for i, c in enumerate(kept):
                if _same_candidate(cand, [c]):
                    if (
                        not cand.get("from_glyph")
                        and c.get("from_glyph")
                    ):
                        kept[i] = cand
                    break
            continue
        kept.append(cand)
    return kept


def _same_candidate(cand, existing):
    """True when `cand` repeats a box the passes above already proposed.

    Several passes legitimately rediscover the SAME printed box (the contour
    pass and the unclosed-component pass both see a clean standalone rim), and
    a box drawn as a character is also inside the text it leads.  Keeping both
    copies would bind the same option label twice and duplicate it in the
    group.  Identical bboxes are collapsed outright; a box that merely sits
    inside another one is treated as the same box only when it is at least half
    the size, since a genuinely nested small box next to a large one is real.
    """
    for c in existing:
        cb = c["bbox"]
        overlap_w = min(cand["bbox"][2], cb[2]) - max(cand["bbox"][0], cb[0])
        overlap_h = min(cand["bbox"][3], cb[3]) - max(cand["bbox"][1], cb[1])
        if overlap_w <= 0 or overlap_h <= 0:
            continue
        contained = (
            cand["bbox"][0] >= cb[0] - 1
            and cand["bbox"][1] >= cb[1] - 1
            and cand["bbox"][2] <= cb[2] + 1
            and cand["bbox"][3] <= cb[3] + 1
        )
        identical = all(
            abs(a - b) <= 1 for a, b in zip(cand["bbox"], cb)
        )
        if identical:
            return True
        if contained and min(cand["width"], cand["height"]) >= 0.5 * min(
            c["width"], c["height"]
        ):
            return True
        # A glyph-derived box is positioned from the OCR text bbox, which
        # carries padding and so lands a few px off the drawn rim it
        # represents.  Treat near-coincident centres as one box rather than
        # requiring exact containment.
        ccx = (cand["bbox"][0] + cand["bbox"][2]) / 2.0
        ccy = (cand["bbox"][1] + cand["bbox"][3]) / 2.0
        ecx = (cb[0] + cb[2]) / 2.0
        ecy = (cb[1] + cb[3]) / 2.0
        near = 0.6 * max(min(c["width"], c["height"]),
                         min(cand["width"], cand["height"]))
        if abs(ccx - ecx) <= near and abs(ccy - ecy) <= near:
            return True
    return False


def _glyph_candidates(elements, exclude_bboxes):
    """Selection boxes printed as a CHARACTER and read as leading label text.

    A form that renders its box as "☐" leaves no square outline in the pixels,
    so the contour passes cannot see it; the option only exists as the glyph at
    the start of the OCR text.  This reconstructs the box geometry from the
    text's own metrics: the glyph occupies roughly the first character's width
    at the top of the text line, and a ballot box is square and about one text
    height tall.
    """
    out = []
    for el in elements or []:
        eb = el.get("bbox")
        if not eb or len(eb) != 4:
            continue
        text = str(el.get("text") or "").strip()
        if not text or text[0] not in _BOX_GLYPHS:
            continue
        rest = text[1:].strip()
        # A lone box glyph with no label is a bare checkbox; it still counts,
        # but a glyph that is the ENTIRE text is more likely OCR noise.
        if not rest:
            continue
        x1, y1, x2, y2 = (float(v) for v in eb)
        side = min(y2 - y1, max(x2 - x1, 1.0))
        # The glyph sits at the very start of the line, before the label.
        gx1 = x1
        gx2 = min(x1 + side, x2)
        cand = {
            "bbox": [gx1, y1, gx2, y1 + side],
            "width": gx2 - gx1,
            "height": side,
            "center": [(gx1 + gx2) / 2.0, y1 + side / 2.0],
            "from_glyph": True,
        }
        if any(
            _intersection_over_area(cand["bbox"], box) > 0.25
            for box in exclude_bboxes or []
        ):
            continue
        out.append(cand)
    return out


# ---------------------------------------------------------------------------
# Stage 2 - bind each checkbox to its nearest same-row label
# ---------------------------------------------------------------------------

def _ink_embedded_in_text(box, text_bbox, median_text_h):
    """True when `box` is a letter-shaped blob SWALLOWED by a longer word.

    Small closed contours sometimes sit inside a word's bounding box: a
    slashed "0" in "Officer", the counter of a "D" in "Date".  They survive
    every shape gate - they really are small, closed and roughly square - and
    then bind to that word as if it were an option.

    The discriminator is the LEFT inset.  A genuine option box printed as
    part of a merged "glyph + label" element starts at, or a hair left of,
    that element's box, because the box IS the start of the text run.  A blob
    that is a letter is surrounded: the element starts well to its left.  So
    require a generous left inset before calling the ink part of the word.
    """
    if not text_bbox:
        return False
    inset = box[0] - text_bbox[0]
    return inset > max(2.0 * median_text_h, 12.0)


def _horizontal_edge_gap(box, text_bbox):
    """Distance between the box edge and text bbox edge; 0 when overlapping."""
    if text_bbox[2] <= box[0]:
        return box[0] - text_bbox[2]
    if text_bbox[0] >= box[2]:
        return text_bbox[0] - box[2]
    return 0.0


def _bind_options(checkboxes, elements, median_text_h):
    """For each checkbox keep the nearest OCR element on its Y-baseline.

    Side preference matters: an option's box is usually to the LEFT of its
    label, but the same row can also carry the question label ("Manner of
    Acquisition:  box Bequeathed box Exchanged").  OCR often includes the box
    glyph in the option's bbox, so the option text overlaps the box while the
    question label sits cleanly to the left.  We therefore prefer a text
    element that extends to the RIGHT of the box (the option label), and only
    fall back to a left-side element (box drawn right of its label, e.g.
    Gender: "Male box") when no right-side text exists.

    Returns a list of bound pairs:
        {"checkbox": {...}, "element": {...}, "text": cleaned, "dist": float}
    """
    row_tol = max(0.5 * median_text_h, 8.0)
    pairs = []
    for cb in checkboxes:
        cx, cy = cb["center"]
        max_search = max(3 * cb["width"], 40.0)
        best = None
        best_d = float("inf")
        best_side = -1  # 1 = right (preferred), 0 = left
        for el in elements:
            eb = el.get("bbox")
            if not eb or len(eb) != 4:
                continue
            text = str(el.get("text") or "").strip()
            if not text:
                continue
            ecy = (eb[1] + eb[3]) / 2.0
            if abs(ecy - cy) > row_tol:
                continue
            if eb[2] > cx:
                side = 1
                d = max(0.0, eb[0] - cb["bbox"][2])
            elif eb[0] < cx:
                side = 0
                d = max(0.0, cb["bbox"][0] - eb[2])
            else:
                continue
            if d > max_search:
                continue
            if (side > best_side) or (side == best_side and d < best_d):
                best_side, best_d, best = side, d, el
        if best is not None and _ink_embedded_in_text(cb["bbox"], best["bbox"], median_text_h):
            continue
        if best is not None:
            cleaned, _changed = strip_leading_option_mark(best["text"])
            pairs.append({
                "checkbox": cb,
                "element": best,
                "text": (cleaned or "").strip(),
                "dist": best_d,
            })
    return pairs


# ---------------------------------------------------------------------------
# Stage 3 - spatial clustering into option groups
# ---------------------------------------------------------------------------

def _cluster_options(pairs, median_text_h, page_width, rules=None):
    """Union-find over bound pairs.

    Two different kinds of adjacency are judged with two different vertical
    tolerances, because they answer different questions:

        * SAME PRINTED LINE (horizontal reach): two boxes whose centers sit on
          the same text baseline belong to one question when their box gap is
          within a page-relative reach (0.3 x page width).  "Same line" is a
          STRICT band (0.5 x max(text height, checkout height)): anything more
          than a baseline jitter apart is a different row.  A too-loose band
          here fused this form's "disability" row into the Yes/No table below
          it (64px apart, both inside an old 2x band).

        * STACKED LINES (vertical reach): options of one question are often
          printed on consecutive lines (Male / Female, or a 2x2 grid).  Those
          boxes x-overlap and may be several text lines apart, so the band is
          relaxed (2.0 x max(text height, checkbox height)).  The ONE guard
          that keeps sub-questions apart: two stacked boxes that carry the
          SAME bound label are not options of one question but the same-column
          cells of different rows ("Yes" repeated under five different
          sub-questions) - do NOT merge them.  Options with distinct labels
          (Male/Female, Salaried/Commission Based) merge normally.
    """
    if not pairs:
        return []
    mh = max(float(median_text_h), 10.0)
    cbh = float(np.median([p["checkbox"]["height"] for p in pairs])) or mh
    row_band = 2.0 * max(mh, cbh)
    same_row = 0.5 * max(mh, cbh)
    # How far apart two boxes may sit and still be options of the same question
    # when they share a row.  A page-relative cap (previously 0.3 x width) is
    # the wrong yardstick: it makes the answer depend on the SCAN RESOLUTION
    # rather than the layout, so a sparsely spread option row on a wide page got
    # torn into one group per option ("long: [a]" and "long: [b]" instead of
    # "long: [a, b]"), while the same layout on a narrow page stayed intact.
    #
    # What actually separates "options of one question" from "options of two
    # different questions printed side by side" is the gap RELATIVE to the
    # other options on that row.  A question's options are laid out on a roughly
    # even pitch, so a wide gap is fine as long as the boxes around it sit at
    # comparable spacing.  Scale the reach by the row's own median pitch with a
    # generous multiple, and keep a page-relative floor for sparse pages.
    _same_row_pitches = []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            a_box = pairs[i]["checkbox"]["bbox"]
            b_box = pairs[j]["checkbox"]["bbox"]
            if abs(pairs[i]["checkbox"]["center"][1]
                   - pairs[j]["checkbox"]["center"][1]) > same_row:
                continue
            gap = _horizontal_edge_gap(a_box, b_box)
            if gap > 0:
                _same_row_pitches.append(gap)
    if _same_row_pitches:
        h_reach = max(
            4.0 * float(np.median(_same_row_pitches)),
            8.0 * cbh,
            0.3 * float(page_width),
        )
    else:
        h_reach = max(8.0 * cbh, 0.3 * float(page_width))

    parent = list(range(len(pairs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            a_box = pairs[i]["checkbox"]["bbox"]
            b_box = pairs[j]["checkbox"]["bbox"]
            vgap = abs(pairs[i]["checkbox"]["center"][1]
                       - pairs[j]["checkbox"]["center"][1])
            xov = min(a_box[2], b_box[2]) - max(a_box[0], b_box[0])
            if xov > 0:
                # Stacked option lines of one question, but never two rows
                # whose bound labels are textually identical (repeated "Yes"
                # under distinct sub-questions).
                if pairs[i]["text"] and pairs[i]["text"] == pairs[j]["text"]:
                    continue
                # ...and never two rows the form has ruled apart.  Each option
                # row of a boxed section ("Driving License [ ] No [ ] Yes" and
                # "Marital Status [ ] Single [ ] Married") is closed by its own
                # printed rule, so the rule between them means two questions.
                # Without this the vertically-stacked, x-overlapping boxes fuse
                # and the merged group borrows one question's label.
                upper, lower = (
                    (a_box, b_box) if a_box[1] <= b_box[1] else (b_box, a_box)
                )
                if _printed_rule_between(upper, lower, rules):
                    continue
                if vgap <= row_band:
                    union(i, j)
                continue
            if vgap <= same_row and _horizontal_edge_gap(a_box, b_box) <= h_reach:
                union(i, j)

    clusters = {}
    for idx, p in enumerate(pairs):
        clusters.setdefault(find(idx), []).append(p)
    return list(clusters.values())


def _group_macro_bbox(group):
    coords = []
    for p in group:
        coords.append(p["checkbox"]["bbox"])
        coords.append(p["element"]["bbox"])
    return [
        min(c[0] for c in coords),
        min(c[1] for c in coords),
        max(c[2] for c in coords),
        max(c[3] for c in coords),
    ]


def _printed_rule_between(upper, lower, rules, tol=6.0, cover=0.6):
    """True when a printed horizontal rule separates two stacked option rows.

    A form that draws each question inside its own ruled band - "Driving
    License  [ ] No  [ ] Yes" in one box, "Marital Status  [ ] Single  [ ]
    Married" in the box below - prints a full-width rule BETWEEN the two rows.
    That rule is the form telling us these are two separate questions, so the
    rows must not be fused into one option group (which would hand the merged
    group whichever label is nearest and lose a real field).

    The rule must lie in the vertical gap between the two clusters and span
    most of the columns their checkboxes share, so a short tick, a leader line
    or an unrelated rule elsewhere on the page never splits a genuine wrapped
    question.
    """
    if not rules or len(upper) != 4 or len(lower) != 4:
        return False
    ux1, uy1, ux2, uy2 = upper
    lx1, ly1, lx2, ly2 = lower
    top = min(uy2, ly2)
    bottom = max(uy1, ly1)
    if bottom - top <= tol:
        return False  # rows touch; no rule between them
    shared_x1 = max(ux1, lx1)
    shared_x2 = min(ux2, lx2)
    shared_w = shared_x2 - shared_x1
    if shared_w <= 0:
        return False
    for r in rules:
        y = r["y"]
        if not (top + tol < y < bottom - tol):
            continue
        span = min(r["x2"], shared_x2) - max(r["x1"], shared_x1)
        if span >= cover * shared_w:
            return True
    return False


def _merge_same_question_rows(clusters, wrap_band, rules=None):
    """Merge stacked option rows that belong to ONE question.

    `_cluster_options` only fuses vertically-adjacent boxes when a pair of
    boxes x-OVERLAPS (Male / Female, a 2x2 grid).  A question whose options
    wrap onto a second printed line without that overlap - "Applicant is:
    [ ] Individual [ ] Corporation [ ] Limited Liability Company [ ]
    Partnership" / "[ ] Limited Liability Partnership [ ] Limited Partnership
    [ ] Other (specify)" - would otherwise stay split into two groups, and the
    lower row would grab a neighbouring option's text as its label.

    Two clusters merge when their checkbox boxes are vertically adjacent (gap
    within ONE printed line pitch - the rows of a wrapped question are
    consecutive; a field a full row further down is a different question
    entirely), their boxes' horizontal spans overlap, and their option labels
    are NOT the same set.  The identical-set guard keeps repeated "Yes"/"No"
    columns of an option table as separate sub-questions (each row carries the
    same two labels); a label equality there means distinct questions, never a
    wrapped continuation.

    Overlap is measured on the CHECKBOX boxes alone, never the group macro bbox:
    option LABELS are wide, so two side-by-side questions of the same column
    (e.g. a single box next to a long label, above a Yes/No table) x-overlap at
    the macro level even though their boxes are columns apart.
    """
    def _extent(group):
        xs = [p["checkbox"]["bbox"][0] for p in group]
        xe = [p["checkbox"]["bbox"][2] for p in group]
        ys = [p["checkbox"]["bbox"][1] for p in group]
        ye = [p["checkbox"]["bbox"][3] for p in group]
        return min(xs), min(ys), max(xe), max(ye)

    clusters = sorted(clusters, key=lambda c: (_extent(c)[1], _extent(c)[0]))
    changed = True
    while changed:
        changed = False
        merged = []
        i = 0
        while i < len(clusters):
            cur = clusters[i]
            j = i + 1
            while j < len(clusters):
                a = _extent(cur)
                b = _extent(clusters[j])
                # NOTE: this is an OVERLAP measure, not a separation distance.
                # When two clusters are vertically separated it goes negative
                # and only the "rows touch" test below rejects the pair; the
                # wrap band is what decides whether non-touching rows are
                # consecutive lines of one question.
                gap = max(a[1], b[1]) - min(a[3], b[3])
                if gap > wrap_band:
                    break  # sorted by top: every later cluster is farther off
                if min(a[2], b[2]) - max(a[0], b[0]) <= 0:
                    j += 1  # boxes are columns apart => different questions
                    continue
                set_a = {p["text"] for p in cur}
                set_b = {p["text"] for p in clusters[j]}
                if set_a and set_b and set_a == set_b:
                    break  # same repeated options => distinct sub-questions
                # A printed rule between the two rows means the form boxed them
                # as separate questions, not as one wrapped option line.
                if _printed_rule_between(a, b, rules):
                    break
                cur = cur + clusters[j]
                clusters.pop(j)
                changed = True
            merged.append(cur)
            i += 1
        clusters = merged
    return clusters


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_checkbox_groups(
    image_or_gray,
    elements=None,
    median_text_h=None,
    exclude_bboxes=None,
    min_options=1,
):
    """Detect checkbox option groups on a form image.

    Returns list of group dicts (see module docstring).  Runs every stage;
    an empty list means no checkbox groups were found.

    `min_options` drops clusters with too few bound options; it defaults to
    1 so genuine single checkboxes (e.g. "check this box if you were eligible
    last year") survive the strict veto gates above.
    """
    gray = _load_gray(image_or_gray)
    elements = elements or []
    if median_text_h is None:
        heights = [e.get("height") for e in elements if e.get("height")]
        median_text_h = float(np.median(heights)) if heights else 20.0

    candidates = _checkbox_candidates(
        gray,
        elements=elements,
        median_text_h=median_text_h,
        exclude_bboxes=exclude_bboxes,
    )
    if not candidates:
        return []

    # A lone tiny decoy (e.g. a decorative printed square) can sit at the exact
    # lower size bound and pass every veto gate.  When the page yields enough
    # candidates, reject any box far smaller than the page's OWN median
    # checkbox - a genuine option box on this page would never be roughly half
    # the size of its siblings.
    if len(candidates) >= 3:
        sides = sorted(
            min(c["width"], c["height"])
            for c in candidates
        )
        med = sides[len(sides) // 2]
        # The intent was to drop lone decoys far smaller than their siblings.
        # A 0.55x cut does that, but it also silently deletes every real option
        # box on a form that mixes sizes: a page whose questions use a large
        # box and whose last question uses a small one lost the whole small
        # question (its 41px boxes fell just under 0.55 x 75 = 41.25, and with
        # them that question's options and label).  Reject only boxes that are
        # BOTH far smaller than the median AND isolated - a small box that
        # shares its row with another small box is a real option, not a decoy.
        keep_min = max(12.0, 0.55 * med)
        kept = []
        for c in candidates:
            side = min(c["width"], c["height"])
            if side >= keep_min:
                kept.append(c)
                continue
            # Keep a sub-threshold box when another sub-threshold box sits on
            # the same printed row: a question's options are all one size.
            cy = c["center"][1]
            companions = sum(
                1
                for o in candidates
                if o is not c
                and min(o["width"], o["height"]) < keep_min
                and abs(o["center"][1] - cy) <= 0.75 * max(c["height"], o["height"])
            )
            if companions:
                kept.append(c)
        candidates = kept
        if not candidates:
            return []

    pairs = _bind_options(candidates, elements, median_text_h)
    if not pairs:
        return []
    page_width = gray.shape[1]
    # Printed rules: a rule drawn between two stacked option rows means the form
    # boxed them as separate questions (see `_printed_rule_between`).
    rules = merge_collinear_horizontal(
        detect_line_segments(gray, orientation="horizontal", min_length_ratio=0.20)
    )
    clusters = _cluster_options(pairs, median_text_h, page_width, rules=rules)
    if min_options > 1:
        clusters = [c for c in clusters if len(c) >= min_options]

    # Merge option rows of one question that print on consecutive lines without
    # any pair of x-overlapping boxes (a wrapped option line under the same
    # question label).  The band is ONE printed line pitch: the rows of a
    # wrapped question are consecutive, so a band beyond that would swallow
    # unrelated fields printed a row further down.
    mh = max(float(median_text_h), 10.0)
    # The band must express ONE PRINTED LINE PITCH - the vertical step between
    # an option and the next line of that SAME question.  Neither page-wide
    # median works: `median_text_h` is the median over ALL OCR elements, and on
    # a form whose biggest elements are its own selection boxes both that and
    # the median box height come out at the box size, far above the real pitch.
    # A band that wide then swallows rows of two DIFFERENT questions a blank
    # line apart (an "important" yes/no/other row and the "long" a/b row below
    # it), fusing them so one label ends up describing both.
    #
    # Use the SMALL box on the page as the unit instead.  A checkbox is drawn on
    # its option line and is no taller than the line, so the smallest box is a
    # tight, outlier-free proxy for the line pitch: on a form of uniformly sized
    # boxes it IS the pitch, and on a page mixing large and small boxes it
    # tracks the option text rather than the largest graphic.  One pitch and a
    # bit is a consecutive line; two pitches is a different question.
    sides = [
        min(p["checkbox"]["width"], p["checkbox"]["height"]) for p in pairs
    ]
    unit = float(np.min(sides)) if sides else mh
    cbh = float(np.median(sides)) if sides else mh
    wrap_band = max(1.5 * unit, min(0.9 * cbh, 1.5 * mh))
    clusters = _merge_same_question_rows(clusters, wrap_band, rules=rules)

    groups = []
    for i, cluster in enumerate(sorted(clusters, key=lambda g: _group_macro_bbox(g)[1])):
        macro = _group_macro_bbox(cluster)
        # Reading order: row (bucketed by text height) then left-to-right.
        row_h = max(float(median_text_h), 10.0)
        options = [
            {
                "text": p["text"],
                "bbox": [round(v, 2) for v in p["checkbox"]["bbox"]],
            }
            for p in sorted(
                cluster,
                key=lambda p: (
                    round(p["checkbox"]["bbox"][1] / row_h),
                    p["checkbox"]["bbox"][0],
                ),
            )
        ]
        groups.append({
            "id": f"cbg_{i:03d}",
            "bbox": [round(v, 2) for v in macro],
            "options": options,
        })
    return groups