"""
Generic form-field detection and label/value association.

The concepts here are deliberately distinct from OCR text segmentation:

    * OCR text segmentation - the OCR engine's notion of a "line".
    * FORM FIELD segregation - input regions defined by lines, boxes or blank
                               gaps, independent of how the OCR grouped text.

A printed line or box is geometric evidence that the *user* is meant to write
there. The OCR engine may merge several such regions into one element; this
module re-segments text using the detected input regions from geometry.

No document-specific rules exist here. The algorithms only use generic visual
structure: horizontal lines, small boxes, blank gaps, alignment and distance.
"""

import re
import numpy as np
import cv2

import util.config as config
from util.image_utils import _load_gray, _threshold_gray
from util.line_utils import (
    detect_line_segments,
    merge_collinear_horizontal,
    merge_collinear_vertical,
)
from util.geometry_utils import (
    _iou,
    _bbox_center,
    _bbox_area,
    _bbox_intersection,
    _intersection_over_area,
    _point_in_bbox,
    _expand_bbox,
    _union_bbox,
)


# ---------------------------------------------------------------------------
# Input-region detection
# ---------------------------------------------------------------------------

def _is_field_continuation(all_segments, seg, median_text_h):
    """
    True when a near-full-width underline is the second (continuation) row of
    a field: a shorter underline sits directly above it within a couple of text
    heights, overlapping most of the full-width line's length.  These multi-row
    fields ("General nature of business ____ / ______________") must keep their
    second line as an input region instead of being dropped as a page rule.
    """
    mh = max(float(median_text_h or 0), 1.0)
    gap_max = 2.2 * mh
    seg_len = max(seg["length"], 1.0)
    for other in all_segments:
        if other is seg:
            continue
        gap = seg["y"] - other["y"]
        if gap < 4 or gap > gap_max:
            continue
        xov = min(seg["x2"], other["x2"]) - max(seg["x1"], other["x1"])
        if xov >= 0.5 * seg_len:
            return True
    return False


def detect_input_regions(image_or_gray, elements=None, table_bboxes=None, checkboxes=None, checkbox_group_bboxes=None):
    """
    Detect input regions from visual evidence: underlines, small boxes and
    blank gaps between printed labels.

    Returns a list of:
        {
            "id": "field_region_000",
            "kind": "underline" | "box" | "blank",
            "bbox": [x1, y1, x2, y2],
            "source": str,
            "confidence": float,
            "elements": [...]   # OCR elements overlapping the region
        }
    """
    gray = _load_gray(image_or_gray)
    h, w = gray.shape

    table_bboxes = table_bboxes or []
    checkboxes = checkboxes or []
    checkbox_group_bboxes = checkbox_group_bboxes or []

    if elements:
        heights = [e["height"] for e in elements if e.get("height")]
        median_text_h = float(np.median(heights)) if heights else 18.0
    else:
        median_text_h = 18.0

    pad_y = max(10, int(median_text_h * 0.8))

    raw = []

    # ---- 1. underline fields ---------------------------------------------
    # One underline = one input key. Do NOT fuse nearby underlines into a
    # single region: a field such as "Period From ___ to ___" is printed as
    # two separate underlines and must produce two keys ("Period From" and
    # "to"). Only merge segments that are effectively the same line (their
    # x-ranges touch/overlap) so a faint/dashed line still yields one region.
    h_segments = detect_line_segments(
        gray,
        orientation="horizontal",
        min_length_ratio=config.FIELD_LINE_MIN_WIDTH_RATIO,
        max_thickness=20,
        kernel_scale=0.015,
        connect_gap_ratio=0.015,
    )
    merged = _dedup_underline_segments(h_segments, y_tol=6, elements=elements)

    for seg in merged:
        bbox = [seg["x1"], seg["y"] - pad_y, seg["x2"], seg["y"] + pad_y]

        if _inside_any_table(bbox, table_bboxes, overlap_ratio=0.45):
            continue

                # A field underline is typically not a full-width page rule. Only skip
        # segments that span essentially the whole page AND start at the left
        # margin; a long field underline (e.g. the second address row, which
        # starts right of its label) must survive. Don't skip a line that is
        # the continuation row of a labelled field sitting just above it.
        if (
            seg["length"] > w * 0.75
            and seg["x1"] < w * 0.12
            and seg["x2"] > w * 0.85
            and not _is_field_continuation(merged, seg, median_text_h)
        ):
            continue

        raw.append(
            {
                "kind": "underline",
                "bbox": [round(v, 2) for v in bbox],
                "confidence": 0.8,
            }
        )

    # ---- 2. box fields ----------------------------------------------------
    box_regions = []
    for box in _detect_box_regions(gray, elements):
        if _inside_any_table(box["bbox"], table_bboxes, overlap_ratio=0.45):
            continue
        if any(_iou(box["bbox"], cb["bbox"]) > 0.4 for cb in checkboxes):
            continue
        if any(_intersection_over_area(box["bbox"], gb) > 0.25 for gb in checkbox_group_bboxes):
            continue
        entry = {
            "kind": "box",
            "bbox": box["bbox"],
            "confidence": 0.7,
        }
        cap = _caption_for_unlabelled_box(box["bbox"], elements, median_text_h)
        if cap:
            entry["label"] = cap["text"]
            entry["label_bbox"] = cap["bbox"]
        box_regions.append(entry)

    # A box that merely frames a stacked column of inner boxes (a tall
    # "a / b / c" frame drawn as ONE outer box with interior dividers) is a
    # container, not a value region: drop it so the inner cells stand alone.
    # Its children (≥2) share its left/right edges and tile its full height.
    box_tol = max(4.0, 0.2 * median_text_h)

    def _is_box_container(box, all_boxes):
        bb = box["bbox"]
        area = max((bb[2] - bb[0]) * (bb[3] - bb[1]), 1.0)
        covered = 0.0
        children = 0
        for other in all_boxes:
            if other is box:
                continue
            ob = other["bbox"]
            # Only STRICTLY smaller boxes can be interior cells of this box.
            # An enclosing frame is bigger than the cell, so counting it as
            # the cell's child would make every inner cell of a divided
            # "a/b/c" frame look like a container and get dropped wholesale.
            if (ob[2] - ob[0]) * (ob[3] - ob[1]) >= area:
                continue
            if not (abs(ob[0] - bb[0]) <= box_tol and abs(ob[2] - bb[2]) <= box_tol):
                continue
            cx1 = max(ob[0], bb[0])
            cy1 = max(ob[1], bb[1])
            cx2 = min(ob[2], bb[2])
            cy2 = min(ob[3], bb[3])
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            covered += (cx2 - cx1) * (cy2 - cy1)
            children += 1
        return children >= 2 and covered >= 0.95 * area

    box_regions = [b for b in box_regions if not _is_box_container(b, box_regions)]
    raw.extend(box_regions)

    # ---- 2b. caption-under-rule fields (line or box above a small name) -----
    # A value rule whose field name is printed as a SMALL caption directly
    # beneath it ("______ / address").  These rules are faint/dashed or
    # full-width page lines that the underline pass drops, so they currently
    # register no field at all; the caption is their signature.
    for cap in _detect_caption_fields(gray, elements, median_text_h):
        # No _inside_any_table gate here (unlike every other pass): the ML
        # table model over-flags whole caption rows as zero-cell "tables"
        # (e.g. [60,186,1610,565] on the veteran form), which would swallow
        # every real caption field. The caption itself - a SMALL printed name
        # under a bare writing rule - is the stronger signature.
        raw.append(cap)

    # ---- 3. grid cells (bordered single-cell fields) -----------------------
    for cell in _detect_grid_cells(
        gray,
        elements=elements,
        median_text_h=median_text_h,
        table_bboxes=table_bboxes,
        checkbox_group_bboxes=checkbox_group_bboxes,
    ):
        raw.append(
            {
                "kind": "grid_cell",
                "bbox": cell["bbox"],
                "confidence": cell["confidence"],
                "grid_label": cell["grid_label"],
                "grid_label_bbox": cell["grid_label_bbox"],
                "grid_continuations": cell.get("grid_continuations"),
            }
        )

    # ---- 4. blank gaps between labels -------------------------------------
    if elements:
        for gap in _detect_blank_gaps(elements, gray):
            if _inside_any_table(gap["bbox"], table_bboxes, overlap_ratio=0.45):
                continue
            if any(_iou(gap["bbox"], cb["bbox"]) > 0.4 for cb in checkboxes):
                continue
            if any(_intersection_over_area(gap["bbox"], gb) > 0.25 for gb in checkbox_group_bboxes):
                continue
            raw.append(
                {
                    "kind": "blank",
                    "bbox": gap["bbox"],
                    "confidence": 0.45,
                }
            )

# ---- absorb underline/caption lines into their enclosing box ----------
    # A printed box's borders are horizontal lines too, so the underline pass
    # (and the caption-rule pass) register the box's TOP and BOTTOM edges --
    # and any caption rule printed inside it -- as separate `underline`
    # regions.  A real input BOX is the authoritative value region: one box
    # must yield one field, not "box + 2 underlines".  Absorb every line whose
    # band rides inside a detected box (its border or an inner caption rule)
    # and shares most of the box's width.  No label is transferred here: the
    # border line's caption is dropped with the line; the box gets its label
    # from the surrounding text via template label-association ("__/__"
    # captions, text above / to-the-left, same-row band rules).  Multi-row
    # "____ / ____" fields are unaffected: they have no enclosing box.
    if box_regions:
        absorb_tol = max(4.0, 0.2 * median_text_h)
        dropped_ids = set()
        for region in raw:
            if region.get("kind") != "underline":
                continue
            rb = region["bbox"]
            line_cy = (rb[1] + rb[3]) / 2.0
            rw = max(rb[2] - rb[0], 1.0)
            for box in box_regions:
                bb = box["bbox"]
                if not (bb[1] - absorb_tol <= line_cy <= bb[3] + absorb_tol):
                    continue
                xov = min(rb[2], bb[2]) - max(rb[0], bb[0])
                if xov <= 0:
                    continue
                if xov < 0.7 * rw or xov < 0.5 * (bb[2] - bb[0]):
                    continue
                dropped_ids.add(id(region))
                break
        if dropped_ids:
            raw = [r for r in raw if id(r) not in dropped_ids]

    # ---- deduplicate overlapping regions ----------------------------------
    ordered = sorted(raw, key=lambda r: r["confidence"], reverse=True)
    deduped = []
    for region in ordered:
        if any(_iou(region["bbox"], kept["bbox"]) > 0.55 for kept in deduped):
            continue
        deduped.append(region)

    deduped.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

    # Reading order: top-to-bottom, left-to-right *within* the same printed
    # row. Adjacent underlines of one row (e.g. "Period From ___ to ___")
    # can be a few px apart vertically, so group rows by a small tolerance
    # and sort each row's regions by x.
    row_tol = max(8.0, 0.5 * median_text_h)
    rows = []
    for region in deduped:
        top = region["bbox"][1]
        if rows and top - rows[-1]["top"] <= row_tol:
            rows[-1]["items"].append(region)
        else:
            rows.append({"top": top, "items": [region]})
    ordered = []
    for row in rows:
        ordered.extend(sorted(row["items"], key=lambda r: r["bbox"][0]))
    deduped = ordered

    regions = []
    for i, region in enumerate(deduped):
        entry = {
            "id": f"field_region_{i:03d}",
            "kind": region["kind"],
            "bbox": region["bbox"],
            "source": "cv",
            "confidence": region["confidence"],
        }
        if region.get("kind") == "grid_cell":
            entry["grid_label"] = region.get("grid_label")
            entry["grid_label_bbox"] = region.get("grid_label_bbox")
            entry["grid_continuations"] = region.get("grid_continuations")
        if region.get("label"):
            entry["label"] = region["label"]
            entry["label_bbox"] = region.get("label_bbox") or region["bbox"]
        regions.append(entry)

    return regions


def _box_contains_text(bbox, elements):
    """True when a real word's center falls inside the box. A photo/logo frame
    carries printed words inside ("Paste your passport photograph here") while
    a genuine input box on an empty form is blank.  Lone decorative glyphs
    printed inside a value box as part of the prompt (a currency symbol "\u20ac"
    in "Market Value [\u20ac]____", a date slash "//") are NOT disqualifying:
    a REAL input box may legitimately carry such pre-printed marks, so only a
    word of at least two letters can mark a box as text-bearing."""
    if not elements:
        return False
    bx1, by1, bx2, by2 = bbox
    for el in elements:
        text = (el.get("text") or "").strip()
        if len(text) < 2 or not re.search(r"[A-Za-z]{2}", text):
            continue
        eb = el.get("bbox")
        if not eb or len(eb) != 4:
            continue
        cx = (eb[0] + eb[2]) / 2.0
        cy = (eb[1] + eb[3]) / 2.0
        if bx1 <= cx <= bx2 and by1 <= cy <= by2:
            return True
    return False


def _detect_box_regions(gray, elements=None, min_ratio=0.004, max_ratio=0.22):
    """
    Detect small closed rectangles that are candidate input boxes
    (e.g. "[______]") rather than large table borders.
    """
    h, w = gray.shape
    binary = _threshold_gray(gray)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw < 18 or ch < 12:
            continue

        area_ratio = (cw * ch) / (w * h)
        if area_ratio < min_ratio or area_ratio > max_ratio:
            continue

        # A full-width field box (its blank writing area spans most of the
        # page, e.g. "(e) The class(es)..." on form 123) is a genuine input
        # region.  Only reject near-full-page rectangles (a box that is both
        # wider than the page half AND tall enough to be a whole section).
        if cw > w * 0.5 and ch > 0.4 * h:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

        # Rectangularity: contour area close to bounding-rect area and fill
        # roughly a hollow box (closed outline, low interior ink).
        contour_area = cv2.contourArea(contour)
        rect_area = cw * ch
        if rect_area <= 0:
            continue
        rect_ratio = contour_area / rect_area
        if rect_ratio < 0.4:
            continue  # too sparse to be an outline

        # A box whose borders are printed as thin/faint rules (field boxes on
        # a light-blue template) often has a sparse contour whose polygon
        # approximation collapses to 2 points.  When the enclosed area still
        # nearly fills its bounding rect (a clean ring) it IS a box; only the
        # polygon gate can be omitted for such rings.
        if len(approx) < 4 and rect_ratio < 0.85:
            continue

        interior = binary[y + 2 : y + ch - 2, x + 2 : x + cw - 2]
        if interior.size == 0:
            continue
        interior_dark = float(np.mean(interior > 0))

        # Hollow box: outline exists but interior is mostly empty.
        if interior_dark > 0.6:
            continue

        # Photo-frame / illustration boxes carry printed instructions inside;
        # genuine input boxes are empty. Drop any box that already has OCR
        # text so it cannot hijack a nearby label (e.g. the GENDER photo box).
        if _box_contains_text([x, y, x + cw, y + ch], elements):
            continue

        boxes.append(
            {
                "bbox": [x, y, x + cw, y + ch],
                "confidence": 0.7,
            }
        )

    # Deduplicate
    deduped = []
    for box in sorted(boxes, key=lambda b: b["bbox"][2] * b["bbox"][3]):
        if any(_iou(box["bbox"], kept["bbox"]) > 0.5 for kept in deduped):
            continue
        deduped.append(box)

    return deduped[:80]


def _cell_top_text(cell, elements, median_text_h):
    """
    Find the printed text that sits inside a grid cell.

    Elements are sorted top-to-bottom then left-to-right; the TOP line inside
    the cell is the cell's printed label ("CONTACT PERSON NAME:" at the top of
    a bordered rectangle with a blank writing area beneath it).  Returns
    {"text", "bbox"} or None when the cell contains no OCR text.
    """
    L, T, R, B = cell
    if not elements:
        return None
    inside = []
    for el in elements:
        eb = el.get("bbox")
        if not eb or len(eb) != 4:
            continue
        cx = (eb[0] + eb[2]) / 2.0
        cy = (eb[1] + eb[3]) / 2.0
        if L <= cx <= R and T <= cy <= B:
            inside.append(el)
    if not inside:
        return None
    mh = max(float(median_text_h or 0), 1.0)
    inside.sort(key=lambda e: (e["center"][1], e["center"][0]))
    top = inside[0]
    topsy = top["center"][1]
    row_tol = max(8.0, 0.5 * mh)
    line = [top]
    for el in inside[1:]:
        if el["center"][1] - topsy <= row_tol:
            line.append(el)
        else:
            break
    line.sort(key=lambda e: e["bbox"][0])
    text = " ".join((el.get("text") or "").strip() for el in line).strip()
    if not text:
        return None
    return {
        "text": text,
        "bbox": [
            min(e["bbox"][0] for e in line),
            min(e["bbox"][1] for e in line),
            max(e["bbox"][2] for e in line),
            max(e["bbox"][3] for e in line),
        ],
    }


def _detect_grid_cells(
    gray,
    elements=None,
    median_text_h=18.0,
    table_bboxes=None,
    checkbox_group_bboxes=None,
    row_rule_ratio=0.30,
    col_span_ratio=0.7,
    min_blank_ratio=0.7,
):
    """
    Detect "cell-based" / grid-layout input fields.

    In a grid form the printed label and the blank writing area share ONE
    bordered rectangle: the label is printed at the top of the cell and the
    user writes in the space below it.  These cells are rarely a single closed
    contour (the borders are thin), so the cell grid is rebuilt from the
    form's line skeleton:

        1. full-width horizontal rules -> row bands (each band is a row of
           cells), and
        2. vertical rules that span a band -> column dividers subdividing it.

    A band segment becomes a cell field only when an OCR element (the label)
    sits inside its top area with blank space beneath it (the writing area),
    so section-title strips, blank separator rows, paragraphs and pure table
    regions never become fields.

    Returns a list of:
        {
            "bbox": [x1, y1, x2, y2],      # the FULL cell
            "grid_label": str,              # top line inside the cell
            "grid_label_bbox": [..],        # label bbox
            "confidence": float,
        }
    """
    h, w = gray.shape
    mh = max(float(median_text_h or 0), 1.0)
    elements = elements or []
    table_bboxes = table_bboxes or []
    checkbox_group_bboxes = checkbox_group_bboxes or []

    # Row rules: horizontal segments spanning most of the page.
    h_segments = detect_line_segments(
        gray,
        orientation="horizontal",
        min_length_ratio=row_rule_ratio,
        max_thickness=20,
        kernel_scale=0.015,
        connect_gap_ratio=0.015,
    )
    h_segments = merge_collinear_horizontal(
        h_segments, y_tol=6, gap_tol_ratio=0.02, max_gap_px=60
    )
    row_rules = sorted(
        {
            round(s["y"], 1)
            for s in h_segments
            if s["length"] >= row_rule_ratio * w
        }
    )

    v_segments = detect_line_segments(
        gray,
        orientation="vertical",
        min_length_ratio=0.015,
        max_thickness=12,
        kernel_scale=0.015,
        connect_gap_ratio=0.015,
    )
    v_segments = merge_collinear_vertical(
        v_segments, x_tol=6, gap_tol_ratio=0.05, max_gap_px=40
    )

    if len(row_rules) < 2:
        return []

    # Typical cell height is 1-2 text rows; anything several rows taller is a
    # section/privacy box, not a single labelled input cell.
    gaps = [
        b - a for a, b in zip(row_rules, row_rules[1:]) if b - a >= 1.6 * mh
    ]
    median_gap = float(np.median(gaps)) if gaps else 0.0
    max_cell_h = max(2.5 * median_gap, 3.0 * mh, 60.0)

    # Per-band column dividers (top -> sorted spanning x values). Cached so the
    # continuation pass below can re-check bands without recomputation.
    band_walls = {}
    for top, bottom in zip(row_rules, row_rules[1:]):
        if bottom - top < 1.6 * mh:
            band_walls[top] = []
            continue
        walls = []
        for s in v_segments:
            span = min(bottom, s["y2"]) - max(top, s["y1"])
            if span >= max(min(col_span_ratio * (bottom - top), 30.0), 20.0):
                walls.append(s["x"])
        band_walls[top] = sorted(
            {round(x, 1) for x in walls if 0 < x < w}
        )

    cells = []
    for top, bottom in zip(row_rules, row_rules[1:]):
        band_h = bottom - top
        if band_h < 1.6 * mh or band_h > max_cell_h:
            continue

        walls = band_walls[top]
        if len(walls) < 2:
            continue

        xs = [walls[0]] + walls[1:-1] + [walls[-1]]
        for L, R in zip(xs, xs[1:]):
            if R - L < 60:
                continue

            cell = [L, top, R, bottom]
            if _inside_any_table(cell, table_bboxes, overlap_ratio=0.45):
                continue
            if any(
                _intersection_over_area(cell, gb) > 0.25
                for gb in checkbox_group_bboxes
            ):
                continue

            top_text = _cell_top_text(cell, elements, mh)
            if not top_text or len(top_text["text"]) < 2:
                continue

            # A full-row heading (covers most of the cell width) is a section
            # title, not a top-left cell label.
            label_w = top_text["bbox"][2] - top_text["bbox"][0]
            if label_w >= 0.8 * (R - L):
                continue

            # The blank writing area beneath the label.
            if bottom - top_text["bbox"][3] < min_blank_ratio * mh:
                continue

            cells.append(
                {
                    "bbox": cell,
                    "grid_label": top_text["text"],
                    "grid_label_bbox": top_text["bbox"],
                    "confidence": 0.35,
                }
            )

    # Continuation rows: an empty full-height band directly below a labelled
    # cell is that cell's extra writing line, exactly as a second blank
    # underline is fused into a single-line field. A multi-line field such as
    # "ADDRESS:" has its label printed at the top of the first row and a blank
    # second row below it; both rows belong to the same value. Half-height gaps
    # (section spacers) and bands that carry printed text stop the chain.
    cont_min_h = max(1.6 * mh, 0.75 * median_gap)
    band_list = list(zip(row_rules, row_rules[1:]))
    top_index = {top: i for i, top in enumerate(row_rules)}
    for cell in cells:
        ctop = cell["bbox"][1]
        left, right = cell["bbox"][0], cell["bbox"][2]
        conts = []
        for top2, bottom2 in band_list[top_index.get(ctop, 0) + 1:]:
            band_h = bottom2 - top2
            if band_h < cont_min_h:
                break
            if _band_contains_text(top2, bottom2, elements):
                break
            walls2 = band_walls.get(top2, [])
            if len(walls2) < 2:
                break
            # A continuation must span the SAME columns as the parent cell (for
            # a full-width cell that means no interior divider below it).
            if walls2[0] == left and walls2[-1] == right:
                conts.append([left, top2, right, bottom2])
            else:
                break
        if conts:
            cell["grid_continuations"] = conts

    return cells


def _band_contains_text(top, bottom, elements):
    """True when any OCR element's center lies inside the horizontal band."""
    if not elements:
        return False
    for el in elements:
        eb = el.get("bbox")
        if not eb or len(eb) != 4:
            continue
        cy = el.get("center", [None, None])[1]
        if cy is None:
            cy = (eb[1] + eb[3]) / 2.0
        if top <= cy <= bottom:
            return True
    return False


def _detect_blank_gaps(elements, gray, gap_ratio=0.012):
    """
    Turn large blank gaps between printed labels on the same row into
    implicit input regions.

    Generic rule: when two text elements sit on the same row and the gap
    between them is both large and free of ink, the gap is an input region.
    This reproduces behaviours like "Date: ____ Place: ____" even when the
    blanks are too faint for line detection.
    """
    h, w = gray.shape
    rows = []
    for el in sorted(elements, key=lambda e: (e["center"][1], e["center"][0])):

        matched = False
        for row in rows:
            if abs(el["center"][1] - row["cy"]) <= 16:
                row["items"].append(el)
                matched = True
                break
        if not matched:
            rows.append({"cy": el["center"][1], "items": [el]})

    gaps = []
    min_gap = max(40, int(w * gap_ratio))

    for row in rows:
        items = sorted(row["items"], key=lambda e: e["bbox"][0])
        for a, b in zip(items, items[1:]):
            gap = b["bbox"][0] - a["bbox"][2]
            if gap < min_gap:
                continue
            if gap > w * 0.45:
                continue  # too large to be a field

            x1 = int(round(a["bbox"][2]))
            x2 = int(round(b["bbox"][0]))
            y1 = int(round(min(a["bbox"][1], b["bbox"][1]) - 4))
            y2 = int(round(max(a["bbox"][3], b["bbox"][3]) + 4))

            band = gray[max(0, y1) : y2, max(0, x1) : x2]
            if band.size == 0:
                continue
            band_binary = _threshold_gray(band)
            ink = float(np.mean(band_binary > 0))

            if ink > 0.05:
                # A short blank often carries a faint printed underline that
                # line detection missed (e.g. "Technical subjects: ____").
                # Tolerate ink shaped like a thin horizontal rule (a few
                # consecutive inked rows covering most of the gap width) but
                # reject real printed text, which spans many rows and blobs.
                longest_run = 0
                run = 0
                for row_ink in (band_binary > 0).sum(axis=1):
                    run = run + 1 if row_ink > 0 else 0
                    longest_run = max(longest_run, run)
                col_ink = (band_binary > 0).sum(axis=0)
                coverage = float((col_ink > 0).mean())
                # The gap should be visually empty (or carry a faint rule only).
                if ink > 0.25 or longest_run > 5 or coverage < 0.6:
                    continue

            gaps.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": 0.5,
                }
            )

    return gaps


def _inside_any_table(bbox, table_bboxes, overlap_ratio=0.5):
    for table_bbox in table_bboxes:
        inter = _bbox_intersection(bbox, table_bbox)
        if inter is None:
            continue
        inter_area = (inter[2] - inter[0]) * (inter[3] - inter[1])
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if bbox_area <= 0:
            continue
        if inter_area / bbox_area > overlap_ratio:
            return True
    return False


def _gap_between_lines_has_text(line_a, line_b, elements, y_tol):
    """
    True when a *non-separator* OCR element occupies the gap between two
    collinear horizontal segments. Keeps adjacent underline fields apart so
    "Total Marks ___ Out of ___" becomes two keys. Text that is only '/'
    or '-' is a date-field separator ("___ / ___ / ____") and does NOT block.
    """
    if not elements:
        return False
    left, right = sorted([line_a, line_b], key=lambda s: s["x1"])
    gx1, gx2 = left["x2"], right["x1"]
    if gx1 >= gx2:
        return False  # lines already overlap; no real gap
    gy = left["y"]
    band = max(12.0, 2.5 * y_tol)
    y_lo, y_hi = gy - band, gy + band
    for el in elements:
        eb = el.get("bbox")
        if not eb or len(eb) != 4:
            continue
        cy = el.get("center", [None, None])[1]
        if cy is None:
            cy = (eb[1] + eb[3]) / 2.0
        if not (y_lo <= cy <= y_hi):
            continue
        if eb[2] <= gx1 + 2.0 or eb[0] >= gx2 - 2.0:
            continue
        text = (el.get("text") or "").strip()
        if not text:
            continue
        if re.fullmatch(r"[\\/‑–—\-\u2212]+", text):
            continue  # date-field separator, merge across it
        return True
    return False


def _dedup_underline_segments(segments, y_tol=6, gap_tol=4.0, elements=None):
    """
    Merge horizontal line segments that are effectively the same printed
    underline: same row (y within y_tol) whose x-ranges touch or overlap
    (gap <= gap_tol). Segments separated by a real gap keep their own region
    so "Period From ___ to ___" yields two distinct keys/underlines. OCR text
    sitting in the gap (a printed label such as "Out of") is also a barrier;
    a bare "/" or "-" separator is not.
    """
    segs = sorted(segments, key=lambda s: (s["y"], s["x1"]))
    merged = []
    for seg in segs:
        placed = False
        for kept in merged:
            if abs(kept["y"] - seg["y"]) > y_tol:
                continue
            if seg["x1"] <= kept["x2"] + gap_tol and kept["x1"] <= seg["x2"] + gap_tol:
                if _gap_between_lines_has_text(kept, seg, elements, y_tol):
                    continue  # a label sits between them -> keep separate
                kept["x1"] = min(kept["x1"], seg["x1"])
                kept["x2"] = max(kept["x2"], seg["x2"])
                kept["length"] = kept["x2"] - kept["x1"]
                kept["y"] = (kept["y"] + seg["y"]) / 2.0
                placed = True
                break
        if not placed:
            merged.append(dict(seg))
    for s in merged:
        s["length"] = s["x2"] - s["x1"]
    return merged


# ---------------------------------------------------------------------------
# Caption-under-rule fields
# ---------------------------------------------------------------------------
# Some forms print the field's name as a SMALL caption directly UNDER the
# value rule instead of beside it:
#
#     _______________
#     address
#
# The caption is the distinctive signal: its text is noticeably smaller than
# the page's median text height, it hugs the rule's bottom edge, and it
# overlaps the rule's x-range.  The rule itself is often faint / 1px / dashed
# or a full-width line that the underline pass drops as a page rule, so it is
# invisible to the normal input-region detection and produces no field at all.

def _caption_candidates_below(bbox, elements, median_text_h, ink=None):
    """OCR elements directly beneath `bbox` that read as a field caption.

    A caption is text printed BELOW the value rule: its top sits just under
    the rule's bottom edge (within a small gap) and its body is SMALLER than
    the page's median text height ("relatively smaller than everything
    else").  It must also overlap the rule's x-range and carry a real word.

    `ink` (optional): mapping {id(element): (ink_top, ink_height)} giving the
    TRUE ink extent of each element measured from the image.  The light dashed
    writing rules of a sky-blue template bleed into the OCR quads above the
    text, inflating the reported box (a 17px "Age" caption arrives as a 33px
    quad whose top rides ON the rule).  When provided, that inflated box
    height is replaced by the real inked height and the box's top may sit on
    - or a little above - the rule.

    Returns the matching elements as
    [{"text", "bbox", "height"}, ...] sorted left to right.
    """
    mh = max(float(median_text_h or 0), 1.0)
    small_max = 0.9 * mh
    lo = bbox[3] + 1.0
    hi = bbox[3] + max(8.0, 0.6 * mh)
    caps = []
    for el in elements or []:
        eb = el.get("bbox")
        if not eb or len(eb) != 4:
            continue
        text = (el.get("text") or "").strip()
        if len(text) < 2:
            continue
        if not re.search(r"[A-Za-z0-9\u00C0-\u024F]", text):
            continue
        if ink is not None and id(el) in ink and ink[id(el)]:
            ink_top, h = ink[id(el)]
            if ink_top > hi or ink_top < bbox[3] - mh:
                continue
        else:
            h = eb[3] - eb[1]
            if not (0.5 <= h <= small_max):
                continue
            if not (lo <= eb[1] <= hi):
                continue
        if not (0.5 <= h <= small_max):
            continue
        xov = min(eb[2], bbox[2]) - max(eb[0], bbox[0])
        if xov <= 0 or xov < (eb[2] - eb[0]) * 0.3:
            continue
        caps.append({"text": text, "bbox": eb, "height": h})
    caps.sort(key=lambda c: c["bbox"][0])
    return caps


def _caption_ink_height(gray, eb, line_y, mh, thr=205):
    """True inked height of a caption printed beneath a writing rule.

    Scans the grayscale image inside the element's OWN vertical extent and
    reports the tallest contiguous run of inked rows (tiny gaps of 1-3px are
    tolerated for descenders/ascenders).  Starting at `line_y + 1` excludes
    the rule itself, which bleeds into the top of the OCR quad of a caption
    printed just beneath it.

    Bounding the scan by the element's own box top is what keeps distant
    elements from inheriting a near neighbour's ink: an element far below the
    line has no overlap between its own rows and the capped window, so it
    returns None and falls back to the (rejecting) OCR-path height test.

    Returns (ink_top, ink_height) or None when no ink is found.
    """
    h, w = gray.shape
    x1 = max(0, int(eb[0]))
    x2 = min(w - 1, int(eb[2]))
    lo = min(h - 1, int(max(eb[1], line_y) + 1))
    hi = min(h - 1, int(min(eb[3], line_y + 2 * mh)))
    if x2 <= x1 or hi <= lo:
        return None
    rows = []
    for y in range(lo, hi + 1):
        cnt = int(np.count_nonzero(gray[y, x1:x2 + 1] <= thr))
        if cnt >= 2:
            rows.append(y)
    if not rows:
        return None
    spans = []
    s = rows[0]
    p = rows[0]
    for r in rows[1:]:
        if r - p <= 3:
            p = r
            continue
        spans.append((s, p))
        s = p = r
    spans.append((s, p))
    best = max(spans, key=lambda sp: sp[1] - sp[0])
    return best[0], best[1] - best[0] + 1


def _scan_writing_rules(gray, thr=250, min_piece=8, bridge=12, min_len=80):
    """Scan grayscale for printed horizontal writing rules.

    The sky-blue template prints its field rules as very light, micro-dashed
    lines (29px pieces on 1-2px gaps) that the morphological kernel of
    `detect_line_segments` either eats or shatters into arbitrary chunks.  So
    the rules are read directly from the pixel runs: dark runs are merged when
    separated by a small gap (dashes of one printed rule) and a span becomes a
    rule when it is long enough AND thin (printed text occupies many rows, a
    writing rule only one or two).

    Returns [{"y": float, "x1": float, "x2": float}, ...] with each printed
    field's rule kept as its OWN span (no fusion across field gaps).
    """
    h, w = gray.shape
    out = []
    for y in range(h):
        row = gray[y]
        dark = np.where(row <= thr)[0]
        if dark.size == 0:
            continue
        # runs of continuous dark pixels
        runs = []
        s = int(dark[0])
        p = int(dark[0])
        for i in dark[1:]:
            if i - p > 1:
                if p - s + 1 >= min_piece:
                    runs.append((s, p))
                s = int(i)
            p = int(i)
        if p - s + 1 >= min_piece:
            runs.append((s, p))
        if not runs:
            continue
        # merge nearby runs (dashes of one rule)
        spans = []
        cur = runs[0]
        for a, b in runs[1:]:
            if a - cur[1] <= bridge:
                cur = (cur[0], b)
            else:
                spans.append(cur)
                cur = (a, b)
        spans.append(cur)
        for x1, x2 in spans:
            if x2 - x1 + 1 < min_len:
                continue
            # thinness: a writing rule is tall-in-height-unthick - only 1-3 of
            # the neighbouring rows have ANY ink across the span, whereas text
            # glyphs paint every row they cross.
            thick = 0
            for yy in range(max(0, y - 2), min(h, y + 3)):
                if np.any(gray[yy, x1:x2 + 1] <= thr):
                    thick += 1
            if thick <= 3:
                out.append({"y": float(y), "x1": float(x1), "x2": float(x2)})
    return out


def _detect_caption_fields(gray, elements, median_text_h):
    """Detect caption-under-line fields from horizontal rules.

    Lines are scanned with a short minimum length so faint / dashed rules
    survive (the underline pass needs longer segments and misses them), then
    colinear dashes are merged into printed rows.  A row only becomes a field
    when a small caption sits directly beneath it:

      * one caption  -> one field spanning the whole row ("______ / address");
      * many captions -> each caption gets its own field over the dashes
        sitting above its cell ("Branch of Service | Disability Rating | ...").

    Guards keep three decoys out:

      * a caption that is the FIRST text of a printed BOX whose top edge is the
        rule ("Property Tax", "Form 50-135" sitting right under a box border)
        is not a field caption;
      * a rule that UNDERLINES content printed above it (a footer page rule)
        is not a caption rule - the writing line of a caption field is bare;
      * rows whose length comes only from a couple of short decorative stubs.

    Returns list of input-region dicts (kind "underline") carrying the caption
    as their pre-validated label:
        {"kind", "bbox", "label", "label_bbox", "source", "confidence"}
    """
    h, w = gray.shape
    mh = max(float(median_text_h or 0), 1.0)

    # Faint / micro-dashed rules are the whole point here: the sky-blue
    # template prints its writing rules as very light dashes (29px pieces on
    # 1-2px gaps) that the morphological kernel of `detect_line_segments`
    # either eats entirely or shatters into arbitrary 78px chunks.  So the
    # rules are scanned directly from the pixel runs instead, which keeps each
    # printed field's rule as ONE span (the four branch of service /
    # disability rating / age / serial number rules arrive as four separate
    # spans instead of being fused by the dashed re-merge).
    rules = _scan_writing_rules(
        gray,
        thr=250,
        min_piece=15,
        bridge=10,
        min_len=max(60.0, 2.0 * mh),
    )
    if not rules:
        return []

    # Closed rectangles ("boxes").  A caption that is merely the first text of
    # a printed box whose top edge is our "rule" (a header label like
    # "Property Tax" under its box border) must not be read as a field caption.
    binary = _threshold_gray(gray)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects = [
        [float(x), float(y), float(x + cw), float(y + ch)]
        for c in contours
        for x, y, cw, ch in [cv2.boundingRect(c)]
        if cw >= 0.40 * mh and ch >= 0.40 * mh
    ]

    def box_header_line(line_y, line_x1, line_x2, cap):
        """True when `cap` is the first text inside a closed box whose top
        edge coincides with - or rides a few px above - the detected rule.

        The tolerance is deliberate: the veteran header box has TWO top rules
        ("Property Tax" under y87.5, "Form 50-135" under y95) so the caption's
        reporting line sits below the box's strict top border.  A real caption
        field's writing line is bare, so the box must contain the caption.

        The detected row is NOT required to fit inside the box: the line
        scanner inflates morphologically noisy rules beyond their true extent
        (a synthetic 100-400 box border arrives as x84-420), so only the
        caption's centre is tested.
        """
        ccx = (cap["bbox"][0] + cap["bbox"][2]) / 2.0
        ccy = (cap["bbox"][1] + cap["bbox"][3]) / 2.0
        top_lo = line_y - 0.4 * mh
        for rx1, ry1, rx2, ry2 in rects:
            if not (top_lo <= ry1 <= line_y + 2):
                continue
            if rx1 <= ccx <= rx2 and ry1 <= ccy <= ry2:
                return True
        return False

    def text_above(line_y, line_x1, line_x2):
        """True when printed text rides a rule that closes off content above
        (a footer/page rule or an interior header border) instead of opening a
        writing line below.

        Checks the element BBOX against the band just above the rule; on the
        veteran header the "Property Tax" label sits partly ABOVE and partly
        BELOW its decorative rule, so a center-point test misses it while an
        overlap test catches it.  A genuine writing rule has a bare band above
        within its span."""
        lo = line_y - 1.2 * mh
        hi = line_y - 2.0
        for el in elements or []:
            eb = el.get("bbox")
            if not eb or len(eb) != 4:
                continue
            if eb[3] <= lo or eb[1] >= hi:
                continue
            if min(eb[2], line_x2) - max(eb[0], line_x1) > 0:
                return True
        return False

    pad = max(8.0, 0.8 * mh)
    regions = []

    # The tiny writing rules on the sky-blue template are printed as very
    # light dashes that the OCR quads swallow, so the caption's reported box
    # height is inflated (an "Age" glyph of ~17px arrives as a 33px quad whose
    # top rides ON the rule).  The smallness test below therefore uses the
    # TRUE inked height measured from the image instead of the OCR box height.
    def ink_map_for(line_y):
        imap = {}
        for el in elements or []:
            eb = el.get("bbox")
            if not eb or len(eb) != 4:
                continue
            v = _caption_ink_height(gray, eb, line_y, mh)
            imap[id(el)] = (v[0], v[1]) if v else None
        return imap

    # Group same printed row (within a few px).  Each writing rule keeps its
    # true x-extent; the four branch-of-service / disability-rating / age /
    # serial-number rules stay SEPARATE spans (the old scanner fused them into
    # one row and handed a single caption the whole width).
    line_ys = []
    for r in sorted(rules, key=lambda s: s["y"]):
        placed = False
        for ly in line_ys:
            if abs(ly - r["y"]) <= 4:
                placed = True
                break
        if not placed:
            line_ys.append(r["y"])
    for ly in line_ys:
        row_rules = [r for r in rules if abs(r["y"] - ly) <= 4]
        row_rules.sort(key=lambda r: r["x1"])
        row_x1, row_x2 = row_rules[0]["x1"], row_rules[-1]["x2"]
        caps = _caption_candidates_below(
            [row_x1, ly, row_x2, ly],
            elements,
            mh,
            ink=ink_map_for(ly),
        )
        if not caps:
            continue
        if text_above(ly, row_x1, row_x2):
            continue
        if len(caps) == 1:
            # one caption -> one field over the whole writing line
            cap = caps[0]
            if box_header_line(ly, row_x1, row_x2, cap):
                continue
            regions.append({
                "kind": "underline",
                "bbox": [row_x1, ly - pad, row_x2, ly + pad],
                "label": cap["text"],
                "label_bbox": cap["bbox"],
                "source": "caption",
                "confidence": 0.9,
            })
            continue
        # many captions -> each claims the rule(s) overlapping its x-range
        for cap in caps:
            hit = [
                r for r in row_rules
                if min(r["x2"], cap["bbox"][2]) - max(r["x1"], cap["bbox"][0]) > 0
            ]
            if hit:
                bx1 = min(r["x1"] for r in hit)
                bx2 = max(r["x2"] for r in hit)
            else:
                bx1, bx2 = cap["bbox"][0], cap["bbox"][2]
            if box_header_line(ly, bx1, bx2, cap):
                continue
            regions.append({
                "kind": "underline",
                "bbox": [bx1, ly - pad, bx2, ly + pad],
                "label": cap["text"],
                "label_bbox": cap["bbox"],
                "source": "caption",
                "confidence": 0.9,
            })

    return regions


def _caption_for_unlabelled_box(box_bbox, elements, median_text_h):
    """Attach a below-caption label to a box that has no label anywhere.

    A value box printed above its field name ("[____]  /  address") looks the
    same as the underline variant.  Only boxes that currently carry NO label
    in the standard above / left zones are re-labelled with the small caption
    beneath them, so labelled boxes keep their existing behaviour.
    """
    caps = _caption_candidates_below(box_bbox, elements, median_text_h)
    if not caps:
        return None
    mh = max(float(median_text_h or 0), 1.0)
    x1, y1, x2, y2 = box_bbox
    tol = max(8.0, 0.4 * mh)
    for el in elements or []:
        eb = el.get("bbox")
        if not eb or len(eb) != 4:
            continue
        if el.get("is_value"):
            continue
        text = (el.get("text") or "").strip()
        if len(text) < 2:
            continue
        if not re.search(r"[A-Za-z0-9\u00C0-\u024F]", text):
            continue
        cx = (eb[0] + eb[2]) / 2.0
        cy = (eb[1] + eb[3]) / 2.0
        above = (y1 - 1.5 * mh <= cy <= y2) and (
            min(eb[2], x2) - max(eb[0], x1) > 0
        )
        left = eb[3] <= y2 + tol and eb[2] <= x1 + tol
        if above or left:
            return None
    return caps[0]


# ---------------------------------------------------------------------------
# Compound OCR splitting
# ---------------------------------------------------------------------------

def _token_boxes(element):
    """
    Return ordered per-token boxes for an OCR element.

    Prefers real word-level boxes from PaddleOCR (`element["words"]`) and
    falls back to a proportional estimate over the element bbox when they are
    not available.

    Returns (tokens, use_real):
        tokens: list of {"text": str, "bbox": [x1,y1,x2,y2]}
        use_real: True when tokens come from word-level boxes.
    """
    words = element.get("words")
    if words:
        tokens = []
        for word in words:
            text = str(word.get("text", "")).strip()
            if not text:
                continue
            wb = word.get("bbox")
            if not wb or len(wb) != 4:
                continue
            tokens.append({"text": text, "bbox": [float(v) for v in wb]})
        if tokens:
            return tokens, True

    # ---- proportional fallback -----------------------------------
    text = element.get("text", "")
    parts = text.split(" ")
    parts = [p for p in parts if p.strip()]

    if not parts:
        return [], False

    x1, y1, x2, y2 = element["bbox"]
    total_chars = sum(len(p) for p in parts)
    if total_chars == 0:
        return [], False

    width = x2 - x1
    tokens = []
    cursor = x1

    for part in parts:
        frac = len(part) / total_chars
        px2 = cursor + width * frac
        tokens.append(
            {
                "text": part,
                "bbox": [cursor, y1, px2, y2],
            }
        )
        cursor = px2

    return tokens, False


def _region_line_cy(region):
    """Vertical centre of an input region (its implied printed line)."""
    bbox = region["bbox"]
    return (bbox[1] + bbox[3]) / 2.0


def _value_baseline_band(region):
    """How far a value token's baseline may sit from the region's line."""
    rb = region["bbox"]
    rh = rb[3] - rb[1]
    return max(8.0, 0.25 * rh)


def _value_claims_region(el, region):
    """
    True when an OCR element looks like a handwritten/typed *value* written on
    this region's printed line: it overlaps the line horizontally (its center
    falls inside the region's x-range) and its baseline (box bottom) sits on
    the line. Tokens from the rows above/below whose centres drift into a
    region's padded box (e.g. a title word hanging over a field, or the first
    words of the following label line) do not qualify.

    Printed form text is never a value: a colon-terminated label, any line
    whose baseline sits clearly above the printed line (labels print above the
    underline, handwritten values sit ON it), and full-line headings /
    instructions that span most of the region's width.
    """
    text = (el.get("text") or "").strip()
    if text.endswith(":"):
        return False
    rb = region["bbox"]
    tcx, _ = _bbox_center(el["bbox"])
    if not (rb[0] <= tcx <= rb[2]):
        return False
    line_cy = _region_line_cy(region)
    if el["bbox"][3] < line_cy - 2:
        return False
    region_w = max(rb[2] - rb[0], 1.0)
    if len(text.split()) >= 3 and (el["bbox"][2] - el["bbox"][0]) >= 0.6 * region_w:
        return False
    return abs(el["bbox"][3] - line_cy) <= _value_baseline_band(region)


def _token_region_window(token, region):
    """
    Window in which a token's center would make it a *value* of the region.

    Returns the window or None. Three gates decide:
      * printed form text is never a value: a colon-terminated token, any
        token whose baseline sits clearly above the printed line (labels print
        above the underline, handwritten values sit ON it), and full-line
        headings / instructions are all rejected up front;
      * horizontal: token center inside the region's x-range,
      * vertical:   token *baseline* (box bottom) within a small band of the
                    region's printed line. The baseline is what a value sits
                    on, so text from the row above/below that merely drifts
                    into the region's padded box is never claimed.
    """
    rb = region["bbox"]
    text = (token.get("text") or "").strip()
    if text.endswith(":"):
        return None
    line_cy = _region_line_cy(region)
    if token["bbox"][3] < line_cy - 2:
        return None
    if len(text.split()) >= 3 and (token["bbox"][2] - token["bbox"][0]) >= 0.6 * max(rb[2] - rb[0], 1.0):
        return None
    rh = rb[3] - rb[1]
    band = _value_baseline_band(region)
    window = [rb[0], line_cy - 1.5 * rh, rb[2], line_cy + band]
    tcx, tcy = _bbox_center(token["bbox"])
    if not _point_in_bbox([tcx, tcy], window):
        return None
    if abs(token["bbox"][3] - line_cy) > band:
        return None
    return window


def split_compound_ocr_element(element, input_regions, min_region_overlap=0.25):
    """
    Split one OCR element into value + printed-text sub-elements using the
    detected input regions.

    Tokens whose center falls inside an input region's value window are
    grouped into a *value* sub-element for that region. Remaining tokens are
    grouped into plain text sub-elements by horizontal proximity (these act as
    labels or connective text such as "to", "No. of days").

    A token belongs to the region whose printed line is *vertically nearest*
    so a value sitting between two consecutive lines is assigned to its own
    line, not the one above/below.

    Returns:
        list of sub-elements. When no assignment is possible it returns
        [element] unchanged so the caller can fall back to the raw element.
    """
    tokens, _ = _token_boxes(element)

    if not tokens or not input_regions:
        return [element]

    elements_are_region_text = False
    for token in tokens:
        for region in input_regions:
            if _token_region_window(token, region) is not None:
                elements_are_region_text = True
                break
        if elements_are_region_text:
            break

    if not elements_are_region_text:
        return [element]

    # ---- assign tokens to regions (guard above, nearest line wins) ----
    assigned = {}  # token index -> region id
    for i, token in enumerate(tokens):
        tcx, tcy = _bbox_center(token["bbox"])
        best_region = None
        # Score: (vertical distance to the line, prefer a line below the
        # token, -containment). Underlines sit under their text, so when two
        # lines are equidistant the one below the writing wins.
        best_score = (float("inf"), 1, -1.0)
        for region in input_regions:
            window = _token_region_window(token, region)
            if window is None:
                continue
            overlap = _intersection_over_area(token["bbox"], window)
            if overlap < min_region_overlap:
                continue
            line_cy = _region_line_cy(region)
            score = (
                abs(tcy - line_cy),
                0 if line_cy >= tcy else 1,
                -overlap,
            )
            if score < best_score:
                best_score = score
                best_region = region

        if best_region is not None:
            assigned[i] = best_region["id"]

    if not assigned:
        return [element]

    region_tokens = {}
    plain_tokens = []

    for i, token in enumerate(tokens):
        region_id = assigned.get(i)
        if region_id is not None:
            region_tokens.setdefault(region_id, []).append((i, token))
        else:
            plain_tokens.append((i, token))

    sub_elements = []
    parent_id = element.get("id", "t000")
    parent_conf = element.get("confidence")
    parent_text = element.get("text", "")

    # Values (one per region, preserving region x-order).
    region_order = {
        rid: idx for idx, rid in enumerate(r["id"] for r in input_regions)
    }
    for region_id, token_list in sorted(
        region_tokens.items(), key=lambda kv: region_order.get(kv[0], 9999)
    ):
        token_list.sort(key=lambda it: it[0])
        bbox = _union_bbox([t["bbox"] for _, t in token_list])
        text = " ".join(t["text"] for _, t in token_list).strip()
        sub_elements.append(
            {
                "id": f"{parent_id}_v{_region_suffix(region_id)}",
                "type": "text",
                "text": text,
                "bbox": [round(v, 2) for v in bbox],
                "center": [round(v, 2) for v in _bbox_center(bbox)],
                "width": round(bbox[2] - bbox[0], 2),
                "height": round(bbox[3] - bbox[1], 2),
                "confidence": parent_conf,
                "is_value": True,
                "region_id": region_id,
                "parent_id": parent_id,
                "parent_text": parent_text,
            }
        )

    # Plain text groups.
    plain_tokens.sort(key=lambda it: it[0])
    i = 0
    group_idx = 0
    while i < len(plain_tokens):
        group = [plain_tokens[i]]
        j = i
        while j + 1 < len(plain_tokens):
            current = plain_tokens[j][1]
            nxt = plain_tokens[j + 1][1]
            gap = nxt["bbox"][0] - current["bbox"][2]
            tw = current["bbox"][2] - current["bbox"][0]
            if gap > max(12.0, tw * 0.8):
                break
            group.append(plain_tokens[j + 1])
            j += 1

        bbox = _union_bbox([t["bbox"] for _, t in group])
        text = " ".join(t["text"] for _, t in group).strip()
        if text:
            sub_elements.append(
                {
                    "id": f"{parent_id}_s{group_idx}",
                    "type": "text",
                    "text": text,
                    "bbox": [round(v, 2) for v in bbox],
                    "center": [round(v, 2) for v in _bbox_center(bbox)],
                    "width": round(bbox[2] - bbox[0], 2),
                    "height": round(bbox[3] - bbox[1], 2),
                    "confidence": parent_conf,
                    "is_value": False,
                    "parent_id": parent_id,
                    "parent_text": parent_text,
                }
            )
            group_idx += 1
        i = j + 1

    sub_elements.sort(key=lambda e: (e["bbox"][1], e["bbox"][0]))

    return sub_elements


def _region_suffix(region_id):
    try:
        return region_id.split("_")[-1]
    except (IndexError, AttributeError):
        return region_id


def assign_ocr_to_input_regions(elements, input_regions):
    """
    For every input region, find the OCR elements that overlap it (its
    `value`) and the printed text to its left (its `label`).

    Returns a list updated in place: each input region dict gains
    `overlap_elements` and `label_elements`.
    """
    regions = [dict(r) for r in input_regions]

    heights = [e.get("height") for e in elements if e.get("height")]
    median_text_h = float(np.median(heights)) if heights else 20.0

    for region in regions:
        bbox = region["bbox"]
        x1, y1, x2, y2 = bbox
        region_id = region["id"]

        overlap_elements = []
        for el in elements:
            # Value sub-elements created by the compound splitter belong to
            # exactly one region.
            if el.get("is_value"):
                if el.get("region_id") == region_id:
                    overlap_elements.append(el)
                continue

            # A value is OCR text written *on* the region's printed line: it
            # overlaps the line horizontally and its baseline sits on it.
            # Handwritten text floats at most a few px off the line, so
            # label words of the row above/below (title text, next row's
            # label) are never captured.
            if _value_claims_region(el, region):
                overlap_elements.append(el)

        # Elements already claimed as a value belong to at least one region;
        # later regions must not treat them as labels.
        for el in overlap_elements:
            if not el.get("is_value") and not el.get("region_id"):
                el["is_value"] = True

        # Labels: printed text immediately to the left of the region, aligned
        # on the region's line. We rank candidates by *vertical* distance to
        # the region's line first so a label from the row above/below is never
        # preferred over the region's own label, then prefer the closest end.
        # A tight cluster keeps multi-word labels ("Period From") together
        # while a value from a neighbouring field is never pulled in.
        label_elements = []
        for el in elements:
            if el.get("is_value"):
                continue
            if el["bbox"][2] > x1 + 0.3 * median_text_h:
                continue  # must end essentially before the region
            if el["bbox"][2] < x1 - 200:
                continue

            v_top, v_bottom = y1 - (y2 - y1) * 0.5, y2 + (y2 - y1) * 0.5
            if not (el["center"][1] >= v_top and el["center"][1] <= v_bottom):
                continue

            label_elements.append(el)

        line_cy = (y1 + y2) / 2.0
        label_elements.sort(
            key=lambda e: (
                abs(e["center"][1] - line_cy),
                abs(e["bbox"][2] - x1),
            )
        )

        if label_elements:
            best = label_elements[0]
            cluster = [best]
            for el in label_elements[1:]:
                same_row = (
                    abs(el["center"][1] - best["center"][1])
                    <= max(8.0, 0.6 * median_text_h)
                )
                if same_row and best["bbox"][0] - el["bbox"][2] <= 100:
                    cluster.append(el)
                else:
                    break
            label_elements = cluster

        region["overlap_elements"] = overlap_elements
        region["label_elements"] = label_elements

    return regions


def _repair_date_range(text):
    """
    Restore slashes dropped by OCR in dates.

    PaddleOCR occasionally reads a '/' of a date as a '1' (printed
    "29 / 8 / 26" -> OCR "2918 / 26"). Two unambiguous cases are repaired:

      * a full range "A to B" that ends in "<digits> / <year>" where the
        trailing digit run admits exactly one valid day/month split;
      * a standalone date "dd1m / yy" whose middle slash was read as a '1'.

    Everything else is returned unchanged.
    """

    def _valid(dd, mm):
        try:
            d, mo = int(dd), int(mm)
        except ValueError:
            return False
        return 1 <= d <= 31 and 1 <= mo <= 12

    if not text:
        return text

    # Case 1: "... to <run> / <year>" with an unambiguous day/month split.
    m = re.fullmatch(
        r"\s*(.+?)\s+\bto\b\s+([0-9]{3,4})\s*/\s*([0-9]{2,4})\s*",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        left, run, year = m.group(1), m.group(2), m.group(3)
        # The left side must itself look like a date (keeps the rule scoped
        # to actual date ranges, e.g. "... to 2918 / 26").
        if re.search(r"\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}", left):
            variants = []
            if len(run) == 4 and run[2] == "1" and _valid(run[:2], run[3]):
                variants.append((run[:2], run[3]))  # slash read as '1'
            for ln in (1, 2):
                if 1 <= len(run) - ln <= 2 and _valid(run[:ln], run[ln:]):
                    v = (run[:ln], run[ln:])
                    if v not in variants:
                        variants.append(v)
            if len(variants) == 1:
                dd, mm = variants[0]
                return f"{left} to {int(dd)} / {int(mm)} / {year}"

    # Case 2: standalone date whose middle slash was read as a '1'
    # (printed "29 / 8 / 26" -> "2918 / 26") - e.g. the "to" key of a
    # From/To underline pair.
    m = re.fullmatch(
        r"\s*([0-9]{2})1([0-9])\s*/\s*([0-9]{2,4})\s*",
        text,
        flags=re.IGNORECASE,
    )
    if m and _valid(m.group(1), m.group(2)):
        return f"{int(m.group(1))} / {int(m.group(2))} / {m.group(3)}"

    return text


def build_form_fields(elements, input_regions):
    """
    Assemble first-class `field` objects from input regions + OCR elements.

    Returns a list of:
        {
            "id": "field_000",
            "label": str,
            "label_element_ids": [...],
            "label_bbox": [..],
            "value": str | None,
            "value_element_ids": [...],
            "value_bbox": [..] | None,
            "input_region": {...},
            "confidence": float,
            "uncertain": bool,
        }
    """
    regions = assign_ocr_to_input_regions(elements, input_regions)

    fields = []

    for i, region in enumerate(regions):
        region_id = region["id"]
        overlap = list(region["overlap_elements"])
        labels = list(region["label_elements"])

        # Sub-elements produced by the compound splitter are the value of
        # their region; raw elements just use their bbox overlap.
        value_elements = []
        for e in overlap:
            if e.get("is_value"):
                if e.get("region_id") == region_id:
                    value_elements.append(e)
            else:
                value_elements.append(e)

        if value_elements:
            value_text = " ".join(e["text"] for e in sorted(
                value_elements, key=lambda e: (e["bbox"][1], e["bbox"][0])
            )).strip()
            value_text = _repair_date_range(text=value_text)
            value_bbox = _union_bbox([e["bbox"] for e in value_elements])
            value_ids = [e["id"] for e in value_elements]
            uncertain = False
        else:
            value_text = None
            value_bbox = None
            value_ids = []
            uncertain = True

        if labels:
            label_text = " ".join(
                e["text"] for e in sorted(labels, key=lambda e: e["bbox"][0])
            ).strip()
            label_ids = [e["id"] for e in labels]
            label_bbox = _union_bbox([e["bbox"] for e in labels])
        else:
            label_text = None
            label_ids = []
            label_bbox = None

        # ---- generic colon rule --------------------------------------
        # A printed OCR line of the form "label: value" has its authoritative
        # label/value boundary at the colon, regardless of how OCR token boxes
        # happened to cluster around the input regions (e.g. the single OCR
        # line "Reason for Leave: Personal" must produce label "Reason for
        # Leave", value "Personal" - never label "Reason for" with a value
        # that still contains the printed "Leave :"). Applied only when the
        # field's value comes from a single such printed line.
        if value_elements:
            parent_texts = {
                e.get("parent_text") for e in value_elements if e.get("parent_text")
            }
            if len(parent_texts) == 1:
                ptext = next(iter(parent_texts))
                if ":" in ptext:
                    parts = [p.strip() for p in ptext.split(":", 1)]
                    label_text = parts[0] or label_text
                    if not parts[1]:
                        value_text, uncertain = None, True
                    else:
                        value_text = parts[1]

        confs = [
            e.get("confidence") for e in value_elements + labels if e.get("confidence") is not None
        ]
        confidence = round(sum(confs) / len(confs), 4) if confs else None

        fields.append(
            {
                "id": f"field_{i:03d}",
                "label": label_text,
                "label_element_ids": label_ids,
                "label_bbox": label_bbox,
                "value": value_text,
                "value_element_ids": value_ids,
                "value_bbox": value_bbox,
                "input_region": {
                    "id": region["id"],
                    "kind": region["kind"],
                    "bbox": region["bbox"],
                    "confidence": region["confidence"],
                },
                "confidence": confidence,
                "uncertain": uncertain,
            }
        )

    return fields


def split_compound_elements(elements, input_regions):
    """
    High-level entry point: split every OCR element that spans multiple input
    regions into value + text sub-elements.

    Returns the new element list (original elements replaced by their
    sub-elements only when a split actually happened).
    """
    output = []
    replaced_ids = set()

    for element in elements:
        sub = split_compound_ocr_element(element, input_regions)
        if len(sub) > 1:
            replaced_ids.add(element["id"])

        # Give value sub-elements a stable id and merge back into the list.
        for piece in sub:
            output.append(piece)

    print(f"[FIELDS] Compound OCR elements split: {len(replaced_ids)}")
    return output