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


def _label_for_region(region, elements, median_text_h):
    """
    Find the printed label for an input region.

    Order of fallback:
      1. A label on the region's own row, sitting to the LEFT and ending
         at-or-before the region's left edge (small tolerance for short
         connectives like "to" that OCR shoves a few px past the start).
      2. A *straddling* text element whose box crosses the region's left
         edge -- e.g. OCR glues "Date of Birth://" into one box. Its
         word-level boxes isolate the real label prefix ("Date of Birth")
         from the trailing "/" separators.
      3. A primary label on the line directly ABOVE the region (within
         1.5 * median_text_h), when the row itself has no left label
         (e.g. "Address:" printed above its value line, or a multiline
         label such as "Name of Board from which..." / "(Gujarat Board:)").

    Returns {"text": str, "bbox": [x1,y1,x2,y2]} or None.
    """
    rb = region["bbox"]
    x1, y1, x2, y2 = rb
    line_cy = (y1 + y2) / 2.0
    tol = max(8.0, 0.4 * median_text_h)
    row_lo = y1 - (y2 - y1) * 0.5
    row_hi = y2 + (y2 - y1) * 0.5

    candidates = []
    # Distance cap: a label cannot be arbitrarily far to the left. This stops
    # a wide photo box from hijacking a label several fields away. Cap at
    # 200px so it can never loosen beyond the old guarantee on large text.
    max_dist = min(15 * median_text_h, 200)
    for el in elements:
        if el.get("is_value"):
            continue
        eb = el.get("bbox")
        if not eb or len(eb) != 4:
            continue
        # A single stray punct glyph (":", ")", ".") is never a label.
        if len((el.get("text") or "").strip()) < 2:
            continue
        cy = el.get("center", [None, None])[1]
        if cy is None:
            cy = (eb[1] + eb[3]) / 2.0
        if not (row_lo <= cy <= row_hi):
            continue
        if eb[0] >= x2:
            continue
        if eb[2] > x1 + tol:
            continue
        if eb[2] < x1 - max_dist:
            continue
        candidates.append(el)

    if not candidates:
        # 2. Straddling text element: its box crosses the region's left edge.
        # Use the word boxes that end before the region to recover the label
        # prefix ("Date of Birth" out of a glued "Date of Birth://").
        straddle = []
        for el in elements:
            if el.get("is_value"):
                continue
            eb = el.get("bbox")
            if not eb or len(eb) != 4:
                continue
            cy = el.get("center", [None, None])[1]
            if cy is None:
                cy = (eb[1] + eb[3]) / 2.0
            if not (row_lo <= cy <= row_hi):
                continue
            if not (eb[0] < x1 < eb[2]):
                continue
            words = el.get("words") or []
            kept = [
                w for w in words
                if w.get("bbox") and len(w["bbox"]) == 4 and w["bbox"][2] <= x1 + tol
            ]
            if not kept:
                continue
            wtext = " ".join(
                w.get("text", "").strip() for w in kept if w.get("text")
            ).strip()
            wtext = re.sub(r"\s+", " ", wtext)
            if len(wtext) < 2 or re.fullmatch(r"[\\/‑–—\-\u2212]+", wtext):
                continue
            straddle.append(
                {
                    "text": wtext,
                    "bbox": [
                        min(w["bbox"][0] for w in kept),
                        min(w["bbox"][1] for w in kept),
                        max(w["bbox"][2] for w in kept),
                        max(w["bbox"][3] for w in kept),
                    ],
                    "cy": cy,
                }
            )
        if straddle:
            straddle.sort(key=lambda s: abs(s["cy"] - line_cy))
            return {"text": straddle[0]["text"], "bbox": straddle[0]["bbox"]}

        # 3. Primary label on the line directly above the region. Only trust
        # colon-terminated headers here: the row has no left label at all, so
        # without a ":" we would inherit stray watermark/header noise.
        primary = _find_upward_label(region, elements, median_text_h)
        if primary and ":" in primary["text"]:
            return primary
        return None

    # Closest to the region's line first, then the label that ends nearest
    # the region's start edge.
    candidates.sort(
        key=lambda e: (
            abs(e["center"][1] - line_cy),
            abs(e["bbox"][2] - x1),
        )
    )
    best = candidates[0]
    cluster = [best]
    for el in candidates[1:]:
        same_row = (
            abs(el["center"][1] - best["center"][1]) <= max(8.0, 0.6 * median_text_h)
        )
        if same_row and best["bbox"][0] - el["bbox"][2] <= 100:
            cluster.append(el)
        else:
            break

    cluster.sort(key=lambda e: e["bbox"][0])
    text = " ".join(e.get("text", "").strip() for e in cluster).strip()
    if not text:
        return None
    bbox = [
        min(e["bbox"][0] for e in cluster),
        min(e["bbox"][1] for e in cluster),
        max(e["bbox"][2] for e in cluster),
        max(e["bbox"][3] for e in cluster),
    ]

    # The text right of the underline is often just a parenthetical
    # instruction ("(Gujarat Board...)") while the real label sits on the line
    # above ("Name of Board from which..."). Recover it rather than failing.
    looks_instruction = text.lstrip().startswith("(") and len(text) > 12
    if (not _label_is_field_label(text)) or looks_instruction:
        primary = _find_upward_label(region, elements, median_text_h)
        if primary:
            if looks_instruction:
                # Parenthetical instruction: take the primary alone so the
                # label stays clean ("Name of Board from which...", not the
                # glued "Name of Board ... (Gujarat Board ...):").
                text = primary["text"]
            else:
                text = f"{primary['text']} {text}".strip()
            bbox = [
                min(primary["bbox"][0], bbox[0]),
                min(primary["bbox"][1], bbox[1]),
                max(primary["bbox"][2], bbox[2]),
                max(primary["bbox"][3], bbox[3]),
            ]

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
    for region, label_info in entries:
        rb = region.get("bbox")
        if not rb or len(rb) != 4:
            continue

        kind = region.get("kind") or "blank"
        rb = list(rb)

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

    # Overlapping duplicate regions (an underline plus a low-confidence blank
    # gap covering the same physical spot) share the label we just derived;
    # keep the most reliable structure (underline > box > blank, then
    # confidence).  Non-overlapping same-label fields (e.g. a second "to" on
    # a different line) are kept.
    kind_rank = {"box": 3, "underline": 2, "blank": 1}
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