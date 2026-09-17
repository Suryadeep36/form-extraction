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
    hole-derived window is deliberately a hair larger than the ring it came
    from, so its 1-px rim lands in blank margin; the scaled side-bands catch
    that case instead (all four bands dark for a ring, never for a lone line).
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
    return min(bands[:4]) >= 0.25


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
        if not (0.8 <= aspect <= 1.2):
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
            ring = max(3, min(8, int(round(min(hw2, hh2) * 0.12))))
            x, y = hx - ring, hy - ring
            cw, ch = hw2 + 2 * ring, hh2 + 2 * ring
            if not _accept(x, y, cw, ch):
                continue
            # Dedupe against the contour-pass candidates (same ring twice).
            if any(
                abs(c["center"][0] - (x + cw / 2.0)) <= max(2.0, 0.4 * min(c["width"], float(cw)))
                and abs(c["center"][1] - (y + ch / 2.0)) <= max(2.0, 0.4 * min(c["height"], float(ch)))
                for c in cands
            ):
                continue
            cands.append({
                "bbox": [float(x), float(y), float(x + cw), float(y + ch)],
                "width": float(cw),
                "height": float(ch),
                "center": [x + cw / 2.0, y + ch / 2.0],
            })

    return cands


# ---------------------------------------------------------------------------
# Stage 2 - bind each checkbox to its nearest same-row label
# ---------------------------------------------------------------------------

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

def _cluster_options(pairs, median_text_h, page_width):
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
    h_reach = 0.3 * float(page_width)

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


def _merge_same_question_rows(clusters, wrap_band):
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
        keep_min = max(12.0, 0.55 * med)
        candidates = [
            c for c in candidates
            if min(c["width"], c["height"]) >= keep_min
        ]
        if not candidates:
            return []

    pairs = _bind_options(candidates, elements, median_text_h)
    if not pairs:
        return []
    page_width = gray.shape[1]
    clusters = _cluster_options(pairs, median_text_h, page_width)
    if min_options > 1:
        clusters = [c for c in clusters if len(c) >= min_options]

    # Merge option rows of one question that print on consecutive lines without
    # any pair of x-overlapping boxes (a wrapped option line under the same
    # question label).  The band is ONE printed line pitch: the rows of a
    # wrapped question are consecutive, so a band beyond that would swallow
    # unrelated fields printed a row further down.
    mh = max(float(median_text_h), 10.0)
    cbh = float(np.median([p["checkbox"]["height"] for p in pairs])) or mh
    wrap_band = 1.2 * max(mh, cbh)
    clusters = _merge_same_question_rows(clusters, wrap_band)

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