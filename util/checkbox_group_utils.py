"""
Geometric checkbox-GROUP detection (template registration / Pass 1).

The OCR-anchored detector (`detect_checkboxes_visual`) finds individual
selection boxes for the semantic layer, but it never groups the options of a
single question (Male / Female ; Salaried / Self-Employed...).  This module
groups them into first-class `checkbox_group` template fields so extraction
can return the actual list of checked option texts instead of N booleans.

Pipeline (pure geometry + OCR text):

    1. Contour detection: closed small boxes/ovals in the binary image kept
       only when their bounding area is in [200, 1600] px^2 and their aspect
       ratio in [0.6, 1.5] (a square-ish selection box; long text input
       fields and large photo boxes are filtered out by area / aspect).  A
       morphological close solidifies thin/printed rings so each ring yields
       exactly one external contour, then the window is gated with the shared
       box-rim/ink classifier to reject stray glyphs.
    2. Option binding: from each checkbox project a horizontal ray left and
       right and bind the nearest OCR element whose vertical center sits on
       the checkbox's Y-baseline.  The distance is measured between the box
       edge and the text bbox edge (overlapping text = distance 0) and is
       capped at 3 * checkbox width.
    3. Spatial clustering: two bound options belong to one group when they
       sit in the same row band (|dy| <= 1.5 * max(text_height, checkbox
       height)) and within 3 * option width horizontally, or when their
       checkboxes share a column vertically within that row band (stacked
       options).  The non-overlapping same-row reach is scaled to the option
       extent (checkbox->label width) because a question row spreads far
       wider than 3 raw checkbox widths.  The group's macro-bbox is the union
       of all its checkboxes + bound labels.

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


# ---------------------------------------------------------------------------
# Stage 1 - geometric checkbox contours
# ---------------------------------------------------------------------------

# A real drawn selection box (vs. a print glyph like o/a) is at least this
# large.  Kept large enough to stay clear of book-print form3 noise (<=18px)
# while admitting mobile_form's 52px table-row boxes.
MIN_REAL_SCALE = 24

def _checkbox_candidates(
    gray,
    min_area=90,
    max_area=6400,
    min_aspect=0.6,
    max_aspect=1.5,
    exclude_bboxes=None,
):
    """Detect small closed square/oval selection boxes from pixel geometry.

    Two complementary passes (union, deduped on proximity):

      1. contour pass   - morphological close + RETR_EXTERNAL, then gate with
         the shared box-rim/ink classifier;
      2. hole pass      - RETR_CCOMP ring+hole pairs.  A selection box rim
         that TOUCHES surrounding gridlines merges with the grid into one
         giant connected contour after the close, so the external pass cannot
         isolate it.  The interior HOLE of such a ring is still a distinct
         contour, so rings touching gridlines survive here.

    Both passes keep only windows whose bounding area is in
    [min_area, max_area] px^2, aspect in [min_aspect, max_aspect] and whose
    frame/centre genuinely looks like a selection box (edge_dark >= 0.65,
    banded centre nearly empty).  Returns a list of
    {"bbox", "width", "height", "center"} candidates with center outside every
    exclusion bbox.
    """
    bin_img = _threshold_gray(gray)
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
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    exclude = exclude_bboxes or []
    cands = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if not (min_area <= area <= max_area):
            continue
        aspect = w / float(h)
        if not (min_aspect <= aspect <= max_aspect):
            continue
        bbox = [float(x), float(y), float(x + w), float(y + h)]
        cx = x + w / 2.0
        cy = y + h / 2.0
        if any(ex[0] <= cx <= ex[2] and ex[1] <= cy <= ex[3] for ex in exclude):
            continue
        r = _ring_density(bin_img, int(x), int(y), int(x + w), int(y + h))
        if r is None:
            continue
        outer_dark, center_dark = r
        rc = _classify_window(bin_img, int(x), int(y), int(x + w), int(y + h))
        if rc is None:
            continue
        edge_dark, _ = rc
        # Registration runs on an EMPTY form: a genuine selection box is a
        # closed rim.  Its 1-px bounding perimeter is therefore almost fully
        # dark (edge_dark high) even for tiny boxes; text glyphs / underscores
        # only darken part of the perimeter, so this is the main discriminator.
        # The banded centre test then rejects solid blobs and filled glyphs
        # whose middle is as inky as their frame.
        if not (
            edge_dark >= 0.65
            and center_dark < 0.30
            and center_dark <= 0.6 * outer_dark
        ):
            continue
        cands.append({
            "bbox": bbox,
            "width": float(w),
            "height": float(h),
            "center": [cx, cy],
        })

    # ---- Pass 2 - hole-based rings (survive gridline contact) ------------
    # A closed selection box whose rim merges with table gridlines shares one
    # giant external contour, but its interior is still a CCOMP hole.  The
    # parent of that hole is the ring; gate it exactly like the contour pass.
    cc_cnts, cc_hier = cv2.findContours(bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if cc_hier is not None:
        cc_hier = cc_hier[0]
        for i, child in enumerate(cc_cnts):
            parent = cc_hier[i][3]
            if parent < 0:
                continue  # only holes (children); their parent is a ring
            x, y, w, h = cv2.boundingRect(cc_cnts[parent])
            area = w * h
            if not (min_area <= area <= max_area):
                continue
            aspect = w / float(h) if h else 0.0
            if not (min_aspect <= aspect <= max_aspect):
                continue
            ring_area = cv2.contourArea(cc_cnts[parent])
            if ring_area <= 0:
                continue
            ratio = cv2.contourArea(child) / ring_area
            # Hollow selection box (thin/medium frame -> big interior hole).
            # A solid dot or a text-glyph counter fills most of its window, so
            # the hole/parent ratio is much smaller.
            if not (0.35 <= ratio <= 0.97):
                continue
            bbox = [float(x), float(y), float(x + w), float(y + h)]
            cx = x + w / 2.0
            cy = y + h / 2.0
            # A real-scale ring survives the close/grid merge specifically
            # inside table areas (mobile_form), so let it through there; the
            # contour pass above still honours the exclusion, and small text
            # glyphs inside tables (form3) stay excluded.
            in_table = any(
                ex[0] <= cx <= ex[2] and ex[1] <= cy <= ex[3] for ex in exclude
            )
            if in_table and min(w, h) < MIN_REAL_SCALE:
                continue
            r = _ring_density(bin_img, int(x), int(y), int(x + w), int(y + h))
            if r is None:
                continue
            outer_dark, center_dark = r
            rc = _classify_window(bin_img, int(x), int(y), int(x + w), int(y + h))
            if rc is None:
                continue
            edge_dark, _ = rc
            if not (
                edge_dark >= 0.65
                and center_dark < 0.30
                and center_dark <= 0.6 * outer_dark
            ):
                continue
            # Dedupe against the contour-pass candidates (same ring found twice).
            if any(
                abs(c["center"][0] - cx) <= max(2.0, 0.4 * min(c["width"], float(w)))
                and abs(c["center"][1] - cy) <= max(2.0, 0.4 * min(c["height"], float(h)))
                for c in cands
            ):
                continue
            cands.append({
                "bbox": bbox,
                "width": float(w),
                "height": float(h),
                "center": [cx, cy],
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

def _cluster_options(pairs, median_text_h):
    """Union-find over bound pairs.

    Two options join when they are on the same ROW band and close enough
    horizontally, or in the same COLUMN over a small vertical gap:

        * row band: |cy_a - cy_b| <= row_band, where
          row_band = 2 * max(median_text_h, median_checkbox_height)
          (option rows can be ~1.5 checkbox heights apart);
        * horizontal reach: gap between the two checkboxes <=
          3 * median option width, where an option's width is its label's
          right edge minus its checkbox's left edge (a question row spreads
          far wider than 3 raw checkbox widths);
        * stacked: the checkboxes overlap in x AND the vertical gap is within
          the row band (e.g. Male / Female on two lines).

    Gating the horizontal rule to the same row band is what stops two
    different questions that share an x-column (Gender above Employment) from
    fusing into one giant macro-box.
    """
    if not pairs:
        return []
    mh = max(float(median_text_h), 10.0)
    cbh = float(np.median([p["checkbox"]["height"] for p in pairs])) or mh
    row_band = 2.0 * max(mh, cbh)

    widths = []
    for p in pairs:
        cb = p["checkbox"]["bbox"]
        eb = p["element"]["bbox"]
        widths.append(abs(max(cb[2], eb[2]) - min(cb[0], eb[0])))
    opt_w = float(np.median(widths)) if widths else 100.0
    h_reach = 3.0 * opt_w

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
            vgap = abs(pairs[i]["checkbox"]["center"][1] - pairs[j]["checkbox"]["center"][1])
            if vgap > row_band:
                continue
            xov = min(a_box[2], b_box[2]) - max(a_box[0], b_box[0])
            if xov > 0:
                union(i, j)
                continue
            if _horizontal_edge_gap(a_box, b_box) <= h_reach:
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_checkbox_groups(
    image_or_gray,
    elements=None,
    median_text_h=None,
    exclude_bboxes=None,
    min_options=2,
):
    """Detect checkbox option groups on an (empty) form image.

    Returns list of group dicts (see module docstring).  Runs every stage;
    an empty list means no checkbox groups were found.

    `min_options` drops clusters with too few bound options.  Real option
    sets have at least two choices; a lone rim bound to nearby text is far
    more often a stray glyph inside a label ("Full Name :", "Zip Code",
    a section heading) than a genuine single-option question.
    """
    gray = _load_gray(image_or_gray)
    elements = elements or []
    if median_text_h is None:
        heights = [e.get("height") for e in elements if e.get("height")]
        median_text_h = float(np.median(heights)) if heights else 20.0

    candidates = _checkbox_candidates(gray, exclude_bboxes=exclude_bboxes)
    if not candidates:
        return []
    pairs = _bind_options(candidates, elements, median_text_h)
    if not pairs:
        return []
    clusters = _cluster_options(pairs, median_text_h)
    if min_options > 1:
        clusters = [c for c in clusters if len(c) >= min_options]

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