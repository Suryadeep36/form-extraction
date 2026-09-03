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
    if t.count(".") > 2 or ":" in t and len(t) > 40:
        return False
    if t.lower() == "for office use only":
        return False
    return True


def _label_for_region(region, elements, median_text_h):
    """
    Find the printed label immediately to the left of an input region.

    A label must sit on the region's own row band, begin before the region's
    right edge, and end at-or-before the region's left edge (with a small
    tolerance so a short connective label such as "to" that OCR places a few
    px past the region start is still claimed -- the exact case that breaks if
    the gate is `end <= x1 + 0.3*text_h`).

    Returns {"text": str, "bbox": [x1,y1,x2,y2]} or None.
    """
    rb = region["bbox"]
    x1, y1, x2, y2 = rb
    line_cy = (y1 + y2) / 2.0
    tol = max(8.0, 0.4 * median_text_h)

    candidates = []
    for el in elements:
        if el.get("is_value"):
            continue
        eb = el.get("bbox")
        if not eb or len(eb) != 4:
            continue
        if not (y1 - (y2 - y1) * 0.5 <= el["center"][1] <= y2 + (y2 - y1) * 0.5):
            continue
        if eb[0] >= x2:
            continue
        if eb[2] > x1 + tol:
            continue
        if eb[2] < x1 - 200:
            continue
        candidates.append(el)

    if not candidates:
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

    out = []
    for region in doc_rep.get("input_regions", []):
        rb = region.get("bbox")
        if not rb or len(rb) != 4:
            continue
        label_info = _label_for_region(region, elements, median_text_h)
        if not label_info:
            continue
        label = label_info["text"]
        if not _label_is_field_label(label):
            continue

        norm_bbox = [
            rb[0] / image_width,
            rb[1] / image_height,
            rb[2] / image_width,
            rb[3] / image_height,
        ]
        lb = label_info["bbox"]
        norm_label = [
            lb[0] / image_width,
            lb[1] / image_height,
            lb[2] / image_width,
            lb[3] / image_height,
        ]
        out.append({
            "label": label,
            "label_norm": _norm_label(label),
            "label_bbox": [round(v, 5) for v in norm_label],
            "value_bbox": [round(v, 5) for v in norm_bbox],
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