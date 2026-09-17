"""
Pure helpers for the two-pass template pipeline.

Pass 1 (registration) turns an empty form into a list of field descriptors
(value regions + labels) and table structures (cell grids).  Pass 2
(extraction) reads a filled form that has been warped into the template
coordinate space and recovers each field's value / each table cell's text.

Everything here is domain-agnostic: no form-specific labels, positions or
keywords are hardcoded.  Value regions on an empty form are blank, so any
OCR text that lands inside a field's value region on the warped filled form
IS that field's value; checkboxes are decided by pixel density instead.
"""

import re
import math
from difflib import SequenceMatcher
import numpy as np

# ---------------------------------------------------------------------------
# Value / label helpers
# ---------------------------------------------------------------------------


def _norm_label(text):
    """Normalize a label for fuzzy matching (lowercase, collapse whitespace,
    drop trailing/leading punctuation and box-drawing debris)."""
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text)
    t = t.strip(" :;:.,-_|*")
    return t.lower()


def _label_is_field_label(text):
    """Heuristic used during registration: a field label is short, printable
    text that does not itself look like a full value/instruction/heading."""
    if not text:
        return False
    t = text.strip()
    if len(t) < 2:
        return False
    # Multi-sentence instructions / headings are not single value labels.
    # A long line that still *ends* in ":" is a label-headed field
    # (e.g. "Name of Board from which qualifying Examination Std. 12th
    # (Science) passed:") and must stay a valid label.
    if t.count(".") > 2 or (
        ":" in t and len(t) > 40 and not t.rstrip().endswith(":")
    ):
        return False
    if t.lower() == "for office use only":
        return False
    return True


def _find_upward_label(region, elements, median_text_h):
    """
    When the text immediately left of a region is rejected (e.g. a
    parenthetical instruction such as "(Gujarat Board...)"), look straight up
    (within 1.5 * median_text_h) for the primary label on the line above
    ("Name of Board from which..."). Returns {"text", "bbox"} or None.
    """
    if not elements:
        return None
    rb = region["bbox"]
    x1, y1, x2, y2 = rb
    region_w = max(x2 - x1, 1.0)
    lo_y = y1 - 1.5 * median_text_h
    hi_y = y1
    candidates = []
    for el in elements:
        if el.get("is_value"):
            continue
        eb = el.get("bbox")
        if not eb or len(eb) != 4:
            continue
        cy = el.get("center", [None, None])[1]
        if cy is None:
            cy = (eb[1] + eb[3]) / 2.0
        if not (lo_y <= cy <= hi_y):
            continue
        # Share roughly the same horizontal extent as the region.
        overlap = min(eb[2], x2) - max(eb[0], x1)
        if overlap <= 0:
            continue
        if overlap < (eb[2] - eb[0]) * 0.5 and overlap < region_w * 0.5:
            continue
        text = (el.get("text") or "").strip()
        if not text or not _label_is_field_label(text):
            continue
        candidates.append(el)
    if not candidates:
        return None
    candidates.sort(key=lambda e: abs(e["center"][1] - y1))
    best = candidates[0]
    return {"text": best.get("text", "").strip(), "bbox": best["bbox"]}


def _region_gap(rb, eb):
    """Closest-axis gap between a region box and a label box (0 on any axis
    where the boxes overlap)."""
    return (
        max(0.0, rb[0] - eb[2], eb[0] - rb[2]),
        max(0.0, rb[1] - eb[3], eb[1] - rb[3]),
    )


def _is_label_row_member(el, elements, median_text_h):
    """
    True when some OTHER printable element (>= 2 chars) sits on the same
    baseline as `el` (within +-0.6 * median_text_h).

    An isolated single word (watermark junk like "AHA") is not part of a row
    of labels and cannot sponsor a colon-less single-word field; a header-row
    label like "Age"/"Gender" that shares its baseline with "Full Name :",
    "Date of Birth :" etc. can.
    """
    if not elements:
        return False
    eb = el.get("bbox")
    if not eb or len(eb) != 4:
        return False
    cy = el.get("center", [None, None])[1]
    if cy is None:
        cy = (eb[1] + eb[3]) / 2.0
    band = max(8.0, 0.6 * median_text_h)
    for other in elements:
        if other is el or other.get("is_value"):
            continue
        if len((other.get("text") or "").strip()) < 2:
            continue
        ob = other.get("bbox")
        if not ob or len(ob) != 4:
            continue
        ocy = other.get("center", [None, None])[1]
        if ocy is None:
            ocy = (ob[1] + ob[3]) / 2.0
        if abs(ocy - cy) <= band:
            return True
    return False


def _find_straddle_prefix(region, elements, median_text_h, tol):
    """
    Recover a label prefix from a glued OCR element whose box crosses the
    region's left edge and vertically overlaps it ("Date of Birth://" ->
    "Date of Birth").  Only word boxes that END before the region are kept, so
    the trailing "/" separators are dropped.  Returns
    {"text", "bbox", "parent"} (nearest to the region's line) or None.
    """
    rb = region["bbox"]
    x1, y1, x2, y2 = rb
    mh = max(float(median_text_h or 0), 1.0)
    lo = y1 - mh
    hi = y2 + mh
    line_cy = (y1 + y2) / 2.0
    best = None
    best_d = None
    for el in elements:
        if el.get("is_value"):
            continue
        eb = el.get("bbox")
        if not eb or len(eb) != 4:
            continue
        if not (eb[0] < x1 < eb[2]):
            continue
        if not (eb[1] < y2 and eb[3] > y1 and eb[3] <= y2):
            continue
        cy = el.get("center", [None, None])[1]
        if cy is None:
            cy = (eb[1] + eb[3]) / 2.0
        if not (lo <= cy <= hi):
            continue
        all_words = [
            w for w in (el.get("words") or [])
            if w.get("bbox") and len(w["bbox"]) == 4
        ]
        kept = [w for w in all_words if w["bbox"][2] <= x1 + tol]
        if not kept:
            continue
        if len(kept) == len(all_words):
            # Every word lands before the region: keep the original text so
            # OCR formatting ("No. of days") is preserved, not re-spaced.
            wtext = (el.get("text") or "").strip()
        else:
            wtext = " ".join(
                w.get("text", "").strip() for w in kept if w.get("text")
            ).strip()
        wtext = re.sub(r"\s+", " ", wtext)
        if len(wtext) < 2 or re.fullmatch(r"[\\/‑–—\-\u2212]+", wtext):
            continue
        bbox = [
            min(w["bbox"][0] for w in kept),
            min(w["bbox"][1] for w in kept),
            max(w["bbox"][2] for w in kept),
            max(w["bbox"][3] for w in kept),
        ]
        d = abs(cy - line_cy)
        if best is None or d < best_d:
            best = {"text": wtext, "bbox": bbox, "parent": el}
            best_d = d
    return best


def _closest_above(rb, elements, median_text_h):
    """
    Nearest element printed on the line directly above the region (cy within
    [y1 - 1.5*h, y1 + 0.5*h], x-overlapping) that is a genuine field label and
    not a parenthetical instruction. Returns {"text", "bbox"} or None.
    """
    x1, y1, x2, y2 = rb
    mh = max(float(median_text_h or 0), 1.0)
    lo = y1 - 1.5 * mh
    hi = y1 + 0.5 * mh
    best = None
    best_d = None
    for el in elements:
        if el.get("is_value"):
            continue
        eb = el.get("bbox")
        if not eb or len(eb) != 4:
            continue
        t = (el.get("text") or "").strip()
        if len(t) < 2:
            continue
        if t.lstrip().startswith("(") and len(t) > 12:
            continue
        cy = el.get("center", [None, None])[1]
        if cy is None:
            cy = (eb[1] + eb[3]) / 2.0
        if not (lo <= cy <= hi):
            continue
        if min(eb[2], x2) - max(eb[0], x1) <= 0:
            continue
        if not _label_is_field_label(t):
            continue
        dx, dy = _region_gap(rb, eb)
        d = math.hypot(dx, dy)
        if best is None or d < best_d:
            best = {"text": t, "bbox": eb}
            best_d = d
    return best


def _label_for_region(region, elements, median_text_h):
    """
    Find the printed label for an input region by minimum distance.

    Every plausible candidate is collected and scored by Euclidean distance
    from the closest point of its box to the region box:

      1. Straddle prefix: a glued OCR element ("Date of Birth://") crossing
         the region's left edge and overlapping it vertically yields its
         word-level prefix ("Date of Birth") as a candidate.
      2. Above / column labels: elements printed just above and overlapping
         the region's x-range (cy within [y1 - 1.5*h, y1 + 0.5*h]) -- this
         band also catches labels printed on/over the value line, and makes a
         close column header beat neighbouring-column text.
      3. Same-row left labels: elements whose bottom ends near the region's
         top (<= y1 + h) and that end at-or-before the region's left edge
         (with a small tolerance for short connectives like "to").

    The minimum-distance candidate wins, then post-filters enforce the label
    rules: a parenthetical instruction is replaced by the nearest above label,
    an isolated colon-less single word is rejected unless it rides a real row
    of labels, and a full-line heading printed over the region is not a label.

    Returns {"text": str, "bbox": [x1,y1,x2,y2]} or None.
    """
    rb = region["bbox"]
    x1, y1, x2, y2 = rb
    mh = max(float(median_text_h or 0), 1.0)
    tol = max(8.0, 0.4 * mh)
    above_lo = y1 - 1.5 * mh
    above_hi = y1 + 0.5 * mh
    # Distance cap: a label cannot be arbitrarily far. Stops a wide photo box
    # from hijacking a label several fields away. Cap at 200px so it can never
    # loosen beyond the old guarantee on large text.
    max_dist = min(15 * mh, 200)

    straddle = _find_straddle_prefix(region, elements, mh, tol)
    straddle_par = id(straddle["parent"]) if straddle else None

    candidates = []
    for el in elements:
        # Templates are registered from EMPTY forms, so any printed text that
        # merely lies *inside* a region's value window (e.g. a prompt printed
        # inside the field box, "Zip Code") is still a label, not a value.
        # `assign_ocr_to_input_regions` may have mis-tagged it as `is_value`
        # for geometry reasons. Only such elements are re-admitted, and only
        # when they are normal text height hugging the region's top edge --
        # a huge watermark blob overlapping a region must never become a label.
        is_val = bool(el.get("is_value"))
        eb = el.get("bbox")
        if not eb or len(eb) != 4:
            continue
        if is_val:
            bh = eb[3] - eb[1]
            if bh > 1.2 * mh or not (y1 - mh <= eb[3] <= y1 + mh):
                continue
        text = (el.get("text") or "").strip()
        if len(text) < 2:
            continue
        cy = el.get("center", [None, None])[1]
        if cy is None:
            cy = (eb[1] + eb[3]) / 2.0

        # 1. Straddling glue: the raw element is consumed by its prefix.
        if id(el) == straddle_par:
            dx, dy = _region_gap(rb, straddle["bbox"])
            candidates.append({
                "text": straddle["text"],
                "bbox": straddle["bbox"],
                "el": el,
                "dist": math.hypot(dx, dy),
                "above": False,
            })
            continue

        # 2. Above / column label.
        if above_lo <= cy <= above_hi and min(eb[2], x2) - max(eb[0], x1) > 0:
            dx, dy = _region_gap(rb, eb)
            candidates.append({
                "text": text, "bbox": eb, "el": el,
                "dist": math.hypot(dx, dy), "above": True,
            })
            continue

        # 3. Same-row label sitting to the left.
        if cy >= y1 - 1.5 * mh and eb[3] <= y1 + mh and eb[2] <= x1 + tol:
            if eb[0] < x2:
                dx, dy = _region_gap(rb, eb)
                d = math.hypot(dx, dy)
                if d <= max_dist:
                    candidates.append({
                        "text": text, "bbox": eb, "el": el,
                        "dist": d, "above": False,
                    })

    if not candidates:
        return None

    # Minimum distance first; an exact row/left label breaks distance ties.
    candidates.sort(key=lambda c: (c["dist"], c["above"]))
    best = candidates[0]
    text = best["text"]
    bbox = best["bbox"]

    # Parenthetical instructions ("(Gujarat Board ...):") are not labels; the
    # nearest non-instruction label above is the real one.
    if text.lstrip().startswith("(") and len(text) > 12:
        primary = _closest_above(rb, elements, mh)
        if primary:
            return {"text": primary["text"], "bbox": primary["bbox"]}
        return None

    # An isolated colon-less single word ("AHA", stray header noise) is not a
    # label unless it rides a real row of printed labels. The exception is a
    # checkbox group: a lone word printed directly above/left of a cluster of
    # selection boxes ("Gender", "Sex") IS its question label.
    if ":" not in text and len(text.split()) == 1:
        if region.get("kind") != "checkbox_group":
            if not _is_label_row_member(best["el"], elements, mh):
                return None

    # A full-line heading printed over the value region (spans >= 72% of the
    # region width and genuinely overlaps it) is not a label. A same-row left
    # label only ever overlaps by the tiny `tol` slop, so it survives.
    lcy = (bbox[1] + bbox[3]) / 2.0
    region_w = max(x2 - x1, 1.0)
    if y1 <= lcy <= y2:
        xov = min(bbox[2], x2) - max(bbox[0], x1)
        if xov >= 0.3 * region_w and (bbox[2] - bbox[0]) >= 0.72 * region_w:
            return None

    # A printed box needs its label directly attached: sitting above/left and
    # *horizontally overlapping* the box, or on its own row. A distant label
    # floating above the box's top edge (e.g. a header instruction grabbing a
    # decorative top box) is not a field label.
    if region.get("kind") == "box":
        xov = min(bbox[2], x2) - max(bbox[0], x1)
        if xov <= 0 and bbox[3] < y1:
            return None

    return {"text": text, "bbox": bbox}


def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def build_template_fields(doc_rep, image_width, image_height):
    """
    Convert a document representation (from an EMPTY form) into template
    fields.

    Labels come from the input regions + OCR elements directly (see
    `_label_for_region`) rather than from the `fields` label association, so
    glassy short labels like "to" that OCR pushes a few px past a region's
    left edge still become their own field instead of being glued to the
    previous label.  Adjacent underlines of one row ("Period From + to +
    No. of days") therefore each produce a separate template field.

    Each returned field:
        {
            "label": original label text,
            "label_norm": normalized label text (matching key),
            "label_bbox": normalized [x1,y1,x2,y2],
            "value_bbox": normalized [x1,y1,x2,y2] (the blank value region),
            "kind": "underline" | "box" | "blank",
            "confidence": float or None,
        }
    The value region of a field whose connected region is a checkbox is left
    with its generic kind; checkbox decision happens at extraction time via
    IoU with the detected checkboxes.
    """
    elements = doc_rep.get("elements", [])
    heights = [e.get("height") for e in elements if e.get("height")]
    median_text_h = float(np.median(heights)) if heights else 20.0

    # First pass: label every region; regions with no usable label become
    # orphans for the vertical-grouping sweep below.
    entries = []  # [(region, label_info)] — labelled fields
    orphans = []  # regions that got no usable label
    for region in doc_rep.get("input_regions", []):
        rb = region.get("bbox")
        if not rb or len(rb) != 4:
            continue
        label_info = _label_for_region(region, elements, median_text_h)
        if label_info:
            label = label_info["text"]
            if _label_is_field_label(label):
                entries.append((region, label_info))
                continue
            # A label was found but rejected by the heuristic (e.g. a long
            # printed instruction). Not a true orphan: drop it instead of
            # absorbing the row into a neighbouring field.
            continue
        # True orphan: no label text at all to the region's left.
        orphans.append(region)

    # Vertical region grouping: an orphan underline sitting directly below a
    # labelled underline that shares its x-range belongs to that field (e.g.
    # the second line of a two-line "Address:" box). Record each adopted line
    # so the field can keep one value box per row; then stretch the parent's
    # value_bbox down over the orphan and drop the orphan.
    adopted_rows = {}
    for orphan in orphans:
        ob = orphan.get("bbox")
        if not ob or len(ob) != 4:
            continue
        if orphan.get("kind") != "underline":
            continue
        # A true adoptable orphan is a blank line: it contains no printed
        # text of its own. If text overlaps it (e.g. a long printed heading
        # spanning its width that merely rejected the label heuristics),
        # keep it out of the sweep so it can't sponge a label from above.
        orphan_w = max(ob[2] - ob[0], 1.0)
        if any(
            (eb := el.get("bbox"))
            and eb[1] < ob[3]
            and eb[3] > ob[1]
            and min(eb[2], ob[2]) - max(eb[0], ob[0]) >= orphan_w * 0.3
            for el in elements
        ):
            continue
        for region, _label in entries:
            rb = region["bbox"]
            if region.get("kind") != "underline":
                continue
            gap = ob[1] - rb[3]
            if gap <= 0 or gap > 2 * median_text_h:
                continue
            if min(ob[2], rb[2]) > max(ob[0], rb[0]):
                adopted_rows.setdefault(id(region), []).append((rb[:], ob[:]))
                rb[3] = ob[3]
                break

    out = []
    # Priority: a printed underline or box is the authoritative value region
    # for a label.  A whitespace "blank" only becomes a field when NO
    # underline/box exists for that same label.  So collect the labels already
    # claimed by a structured (underline/box) region and drop any matching
    # blank below.
    structured_labels = {
        _norm_label(label_info["text"])
        for region, label_info in entries
        if (region.get("kind") or "blank") in ("underline", "box")
    }
    for region, label_info in entries:
        rb = region.get("bbox")
        if not rb or len(rb) != 4:
            continue

        kind = region.get("kind") or "blank"
        rb = list(rb)

        # Gap-blank regions only become fields when their label carries a ":"
        # ("Total Marks Obtained:", "Technical subjects:" — the label cluster
        # may also be "Total Marks Obtained: Out of"). A blank without any ":"
        # is layout whitespace (e.g. the gap between the printed "Signature of
        # Applicant" and "Date" labels), not a writable field, so drop it.
        if kind == "blank":
            if ":" not in label_info["text"]:
                continue
            # Never let a blank shadow a real underline/box for the same label.
            if _norm_label(label_info["text"]) in structured_labels:
                continue

        # One value box per physical line. A label on its own row above the
        # underline (e.g. "Name :" / "______") is NOT an extra value line: the
        # value area is the underline(s). Only when a field's value really
        # spans several printed rows (an underline plus stacked orphans) do we
        # keep one box per row.
        boxes = [rb]
        if kind == "underline":
            parents = adopted_rows.get(id(region))
            if parents:
                # The merged band = parent underline + adopted orphans; restore
                # one box per row.
                boxes = [list(parents[0][0])]
                boxes.extend(list(o) for _, o in parents)
            # Enlarge each line vertically (handwriting stands on the line),
            # clamping against OTHER fields' regions so a box never covers or
            # is covered by a neighbouring row. Padding below is kept tiny so a
            # box's window does not reach the next row's printed label.
            for b in boxes:
                bh = max(b[3] - b[1], 6.0)
                pad_up = min(0.9 * median_text_h, 0.6 * bh)
                pad_down = min(0.2 * median_text_h, 0.15 * bh)
                above = [r["bbox"][3] for r, _ in entries
                         if r is not region and r["bbox"][3] <= b[1] + 2.0]
                below = [r["bbox"][1] for r, _ in entries
                         if r is not region and r["bbox"][1] >= b[3] - 2.0]
                if above:
                    pad_up = min(pad_up, max(0.0, b[1] - max(above)))
                if below:
                    pad_down = min(pad_down, max(0.0, min(below) - b[3]))
                b[1] -= pad_up
                b[3] += pad_down

        norm_bbox = [
            min(b[0] for b in boxes) / image_width,
            min(b[1] for b in boxes) / image_height,
            max(b[2] for b in boxes) / image_width,
            max(b[3] for b in boxes) / image_height,
        ]
        norm_boxes = []
        for b in boxes:
            norm_boxes.append([
                b[0] / image_width,
                b[1] / image_height,
                b[2] / image_width,
                b[3] / image_height,
            ])
        lb = label_info["bbox"]
        norm_label = [
            lb[0] / image_width,
            lb[1] / image_height,
            lb[2] / image_width,
            lb[3] / image_height,
        ]
        out.append({
            "label": label_info["text"],
            "label_norm": _norm_label(label_info["text"]),
            "label_bbox": [round(v, 5) for v in norm_label],
            "value_bbox": [round(v, 5) for v in norm_bbox],
            "value_bboxes": [[round(v, 5) for v in nb] for nb in norm_boxes],
            "kind": region.get("kind") or "blank",
            "confidence": region.get("confidence"),
        })

    # Checkbox option groups (geometric clusters): each group is itself a
    # first-class field whose value is the list of checked option texts.  The
    # parent label comes from the shared `_label_for_region` on the group's
    # macro-box (searches Left then Up exactly like any other field).
    groups = doc_rep.get("checkbox_groups") or []
    unnamed_idx = 0
    for group in groups:
        gb = group.get("bbox")
        if not gb or len(gb) != 4:
            continue
        label_info = _label_for_region(
            {"bbox": gb, "kind": "checkbox_group"}, elements, median_text_h
        )
        label = ""
        label_bbox = None
        if label_info and _label_is_field_label(label_info["text"]):
            label = label_info["text"]
            label_bbox = [
                label_info["bbox"][0] / image_width,
                label_info["bbox"][1] / image_height,
                label_info["bbox"][2] / image_width,
                label_info["bbox"][3] / image_height,
            ]
        # A checkbox group with no printed question label (e.g. a bare row of
        # mutually-exclusive boxes on a form) is still a real field: number it
        # so it is not silently dropped and has a usable name downstream.
        if not label:
            unnamed_idx += 1
            label = f"Checkbox Group {unnamed_idx}"
        options = []
        for opt in group.get("options") or []:
            ob = opt.get("bbox")
            if not ob or len(ob) != 4:
                continue
            options.append({
                "text": opt.get("text") or "",
                "bbox": [
                    round(ob[0] / image_width, 5),
                    round(ob[1] / image_height, 5),
                    round(ob[2] / image_width, 5),
                    round(ob[3] / image_height, 5),
                ],
            })
        if not options:
            continue
        out.append({
            "label": label,
            "label_norm": _norm_label(label),
            "label_bbox": [round(v, 5) for v in label_bbox] if label_bbox else None,
            "value_bbox": [
                round(gb[0] / image_width, 5),
                round(gb[1] / image_height, 5),
                round(gb[2] / image_width, 5),
                round(gb[3] / image_height, 5),
            ],
            "value_bboxes": [[
                round(gb[0] / image_width, 5),
                round(gb[1] / image_height, 5),
                round(gb[2] / image_width, 5),
                round(gb[3] / image_height, 5),
            ]],
            "kind": "checkbox_group",
            "options": options,
            "confidence": None,
        })

    # Overlapping duplicate regions (an underline plus a low-confidence blank
    # gap covering the same physical spot) share the label we just derived;
    # keep the most reliable structure (underline > box > blank, then
    # confidence).  Non-overlapping same-label fields (e.g. a second "to" on
    # a different line) are kept.
    kind_rank = {"box": 3, "underline": 2, "blank": 1, "checkbox_group": 2}
    ordered = sorted(
        out,
        key=lambda f: (kind_rank.get(f["kind"], 0), f["confidence"] or 0.0),
        reverse=True,
    )
    result = []
    for field in ordered:
        dup = any(
            kept["label_norm"] == field["label_norm"]
            and _iou(field["value_bbox"], kept["value_bbox"]) > 0.25
            for kept in result
        )
        if not dup:
            result.append(field)

    result.sort(key=lambda f: (f["value_bbox"][1], f["value_bbox"][0]))

    # Static background text found INSIDE a field's value area on the blank
    # form (watermarks such as "Q123RF", printed decorations).  It is neither
    # a label nor a value: it never changes on the filled form, so extraction
    # must not read it back as a value.  Stored per field's area; extraction
    # excludes any token overlapping these boxes.
    label_boxes = [f["label_bbox"] for f in result if f.get("label_bbox")]
    noise = []
    for e in elements:
        eb = e.get("bbox")
        if not eb or len(eb) != 4:
            continue
        if not (e.get("text") or "").strip():
            continue
        ecx = (eb[0] + eb[2]) / 2.0 / image_width
        ecy = (eb[1] + eb[3]) / 2.0 / image_height
        in_value = any(
            f["value_bbox"][0] <= ecx <= f["value_bbox"][2]
            and f["value_bbox"][1] <= ecy <= f["value_bbox"][3]
            for f in result
        )
        if not in_value:
            continue
        is_label = any(
            lb[0] <= ecx <= lb[2] and lb[1] <= ecy <= lb[3] for lb in label_boxes
        )
        if is_label:
            continue
        noise.append([
            round(eb[0] / image_width, 5),
            round(eb[1] / image_height, 5),
            round(eb[2] / image_width, 5),
            round(eb[3] / image_height, 5),
        ])
    doc_rep["static_noise"] = noise

    return result


def build_template_tables(doc_rep, image_width, image_height):
    """
    Convert the empty form's detected tables into template table structures.

    Tables were detected and cell-griddified by the CV + pretrained-model
    pipeline (`doc_rep["tables"]`, built by `finalize_tables`).  The template
    keeps each cell's grid address, span and normalized bbox plus the static
    text the empty form already prints in the cell (headers and labels).  At
    extraction time the same template cell grid is projected onto the aligned
    filled form and its cells refilled with whatever OCR text landed inside.

    Each table:
        {
            "id", "bbox", "n_rows", "n_cols", "n_cells",
            "structure_source": "cv" | "model" | None,
            "cells": [
                {"row", "column", "row_span", "column_span",
                 "bbox": normalized, "text": static text or None}
            ],
        }
    """
    tables = []
    for t in doc_rep.get("tables", []):
        cells = []
        n_rows = 0
        n_cols = 0
        for c in t.get("cells", []):
            cb = c.get("bbox")
            if not cb or len(cb) != 4:
                continue
            row_span = int(c.get("row_span") or 1)
            col_span = int(c.get("column_span") or 1)
            row = c.get("row")
            col = c.get("column")
            if row is not None:
                n_rows = max(n_rows, int(row) + row_span)
            if col is not None:
                n_cols = max(n_cols, int(col) + col_span)
            cells.append({
                "row": row,
                "column": col,
                "row_span": row_span,
                "column_span": col_span,
                "bbox": [
                    round(cb[0] / image_width, 5),
                    round(cb[1] / image_height, 5),
                    round(cb[2] / image_width, 5),
                    round(cb[3] / image_height, 5),
                ],
                "text": (c.get("text") or "").strip() or None,
            })

        tb = t.get("bbox")
        tables.append({
            "id": t.get("id"),
            "bbox": [
                round(tb[0] / image_width, 5),
                round(tb[1] / image_height, 5),
                round(tb[2] / image_width, 5),
                round(tb[3] / image_height, 5),
            ] if tb and len(tb) == 4 else None,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_cells": len(cells),
            "structure_source": t.get("structure_source"),
            "cells": cells,
        })
    return tables


# ---------------------------------------------------------------------------
# Anchor-based value extraction
# ---------------------------------------------------------------------------

def _anchor_match_rate(text_a, text_b):
    """
    Similarity between two label strings (0..1) used for anchor matching.

    Combines token overlap (Jaccard on normalized tokens) with character
    similarity so OCR artifacts don't kill the match: "No . ofdays" (merged)
    vs "No. of days" scores low on tokens (2 vs 3 tokens -> 1/3) but high on
    characters (~0.83), while glued labels such as "Period From 29 / 8 / 26"
    stay well below the threshold on both.
    """
    if not text_a or not text_b:
        return 0.0
    na = _norm_label(text_a)
    nb = _norm_label(text_b)
    ta = set(na.split())
    tb = set(nb.split())
    token_score = len(ta & tb) / max(len(ta), len(tb)) if ta and tb else 0.0
    if not na or not nb:
        return 0.0
    char_score = SequenceMatcher(None, na, nb).ratio()
    return max(token_score, char_score)


def find_label_anchor(label_text, elements):
    """
    Find the OCR element on a (warped) filled form whose text best matches a
    template label. Returns the element dict or None.

    Matching is token-overlap based so OCR parens/periods that get glued or
    dropped don't break the anchor (e.g. "Mobile No . :" -> "Mobile No.").
    """
    if not elements or not label_text:
        return None
    best, best_score = None, 0.0
    target = _norm_label(label_text)
    for e in elements:
        t = e.get("text") or ""
        score = _anchor_match_rate(t, target)
        if score > best_score:
            best, best_score = e, score
    if best_score < 0.6:
        return None
    return best