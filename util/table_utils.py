"""
CV-based table detection and grid structure reconstruction.

A table here is a first-class structure: a region of the page that contains
intersecting horizontal + vertical line structures (bordered tables), or a
strong rectangular border enclosing cell-like content.

The detection is multi-stage so it can be reasoned about and tuned
independently:

    A. horizontal segments
    B. vertical segments
    C. collinear merge
    D. connected line-graph components   -> candidate table regions
    E. cell grid reconstruction           -> rows / columns / cells
    F. scoring
    G. duplicate removal
    H. multiple tables returned

This module contains NO document-specific rules. It only knows about lines,
grids and rectangles.
"""

import numpy as np
import cv2

import util.config as config
from util.image_utils import _load_gray
from util.line_utils import (
    detect_line_segments,
    merge_collinear_horizontal,
    merge_collinear_vertical,
    segments_to_mask,
    segment_intersections,
)
from util.geometry_utils import (
    _iou,
    _bbox_center,
    _bbox_area,
    _point_in_bbox,
)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _merge_duplicate_candidates(candidates, iou_threshold=0.75):
    """
    Remove near-duplicate candidates produced by overlapping connected
    components, keeping the highest-confidence one.
    """
    ordered = sorted(candidates, key=lambda c: c["confidence"], reverse=True)
    deduped = []
    for cand in ordered:
        dup = False
        for kept in deduped:
            if _iou(cand["bbox"], kept["bbox"]) > iou_threshold:
                dup = True
                break
        if not dup:
            deduped.append(cand)
    return deduped


def detect_table_candidates(
    image_or_gray,
    elements=None,
    min_area_ratio=None,
    min_width_ratio=None,
    min_height_ratio=None,
):
    """
    Detect table candidates using connected line components.

    Returns:
        list of {
            "bbox": [x1, y1, x2, y2],
            "confidence": float,
            "source": "cv",
            "sources": ["cv"],
            "horizontals": [...],  # merged h segments fully inside bbox
            "verticals": [...],    # merged v segments fully inside bbox
            "n_horizontal": int,
            "n_vertical": int,
            "n_intersections": int,
            "cells": [...],        # see build_cv_structure
            "rows": [...],         # y boundaries
            "columns": [...],      # x boundaries
        }
    """
    gray = _load_gray(image_or_gray)
    h, w = gray.shape

    min_area_ratio = min_area_ratio or config.MIN_TABLE_AREA_RATIO
    min_width_ratio = min_width_ratio or config.TABLE_LINE_MIN_WIDTH_RATIO
    min_height_ratio = min_height_ratio or config.TABLE_LINE_MIN_HEIGHT_RATIO

    # ---- stage A / B: raw segments --------------------------------
    h_segments = detect_line_segments(
        gray, orientation="horizontal", min_length_ratio=min_width_ratio
    )
    v_segments = detect_line_segments(
        gray, orientation="vertical", min_length_ratio=min_height_ratio
    )

    # ---- stage C: merge collinear ----------------------------------
    horizontals = merge_collinear_horizontal(
        h_segments, y_tol=8, max_gap_px=int(w * 0.02) + 20
    )
    verticals = merge_collinear_vertical(
        v_segments, x_tol=8, max_gap_px=int(h * 0.02) + 20
    )

    if len(horizontals) < 2 or len(verticals) < 2:
        return []

    # ---- stage D: connected line components ------------------------
    hmask = segments_to_mask(horizontals, (h, w), "horizontal", thickness=3)
    vmask = segments_to_mask(verticals, (h, w), "vertical", thickness=3)
    grid = cv2.add(hmask, vmask)

    # Bridge small gaps in broken borders without gluing neighbouring
    # tables together: modest kernel, then re-check with real segments.
    bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    grid_closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, bridge)

    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(
        grid_closed, connectivity=8
    )

    min_area = w * h * min_area_ratio
    min_w = w * 0.15
    min_h = h * 0.05

    raw_candidates = []

    for idx in range(1, n_comp):
        x, y, cw, ch, area = stats[idx]
        if area < min_area:
            continue
        if cw < min_w or ch < min_h:
            continue

        bbox = [int(x), int(y), int(x + cw), int(y + ch)]

        inside_h = [
            seg for seg in horizontals if _segment_within(seg, bbox, orientation="h")
        ]
        inside_v = [
            seg for seg in verticals if _segment_within(seg, bbox, orientation="v")
        ]

        if len(inside_h) < 2 or len(inside_v) < 2:
            continue

        intersects = segment_intersections(inside_h, inside_v)

        candidate = {
            "bbox": bbox,
            "confidence": 0.0,
            "source": "cv",
            "sources": ["cv"],
            "horizontals": inside_h,
            "verticals": inside_v,
            "n_horizontal": len(inside_h),
            "n_vertical": len(inside_v),
            "n_intersections": len(intersects),
            "cells": [],
            "rows": [],
            "columns": [],
            "structure_source": "cv",
        }

        _score_candidate(candidate, gray, elements)
        raw_candidates.append(candidate)

    deduped = _merge_duplicate_candidates(raw_candidates)

    if not deduped:
        return []

    # ---- stage E: grid structure for survivors ----------------------
    final = []
    for cand in deduped:
        rows, columns = _grid_boundaries(cand)
        cand["rows"] = rows
        cand["columns"] = columns
        cand["cells"] = build_cv_structure(
            cand["bbox"], rows, columns, cand["verticals"], cand["horizontals"]
        )
        final.append(cand)

    final.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))
    return final


def _segment_within(seg, bbox, orientation):
    x1, y1, x2, y2 = bbox
    if orientation == "h":
        return (
            seg["y"] >= y1
            and seg["y"] <= y2
            and seg["x1"] >= x1 - 3
            and seg["x2"] <= x2 + 3
        )
    return (
        seg["x"] >= x1
        and seg["x"] <= x2
        and seg["y1"] >= y1 - 3
        and seg["y2"] <= y2 + 3
    )


def _grid_boundaries(candidate):
    """
    Derive row (y) and column (x) boundaries for a table from its internal
    line segments.
    """
    bbox = candidate["bbox"]
    x1, y1, x2, y2 = bbox

    rows = sorted({round(seg["y"]) for seg in candidate["horizontals"]})

    # Keep only rows whose horizontal segment spans a meaningful width and
    # touches at least one vertical line (so titles across cells count).
    valid_rows = []
    for y in rows:
        segs = [s for s in candidate["horizontals"] if abs(s["y"] - y) <= 3]
        if not segs:
            continue
        width = max(s["x2"] for s in segs) - min(s["x1"] for s in segs)
        if width < (x2 - x1) * 0.35:
            continue
        valid_rows.append(y)

    columns = sorted({round(seg["x"]) for seg in candidate["verticals"]})
    valid_cols = []
    for x in columns:
        segs = [s for s in candidate["verticals"] if abs(s["x"] - x) <= 3]
        if not segs:
            continue
        height = max(s["y2"] for s in segs) - min(s["y1"] for s in segs)
        if height < (y2 - y1) * 0.25:
            continue
        valid_cols.append(x)

    # The outer border must be part of the grid when present.
    rows = [y1] + [y for y in valid_rows if y1 < y < y2] + [y2]
    cols = [x1] + [x for x in valid_cols if x1 < x < x2] + [x2]

    # De-duplicate near-identical rows/cols.
    rows = _unique_sorted(rows, tol=4)
    cols = _unique_sorted(cols, tol=4)
    return rows, cols


def _unique_sorted(values, tol=4):
    out = []
    for v in sorted(values):
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


def build_cv_structure(bbox, rows, columns, verticals, horizontals):
    """
    Build a cell grid from row/column boundaries.

    Merged cells are detected heuristically: when an internal vertical line
    is missing at a column boundary between two row bands, the two cells are
    merged horizontally (same detection applies to missing horizontal lines).

    Each cell:
        {
            "id", "row", "column", "row_span", "column_span", "bbox"
        }
    """
    x1, y1, x2, y2 = bbox

    if len(rows) < 2 or len(columns) < 2:
        return []

    # Map each internal boundary to the vertical/horizontal segments that run
    # across each band, so we can decide whether an edge actually exists.
    cells = []
    cell_index = 0

    for r in range(len(rows) - 1):
        top, bottom = rows[r], rows[r + 1]

        # Compute column spans for this row band.
        col_spans = []
        j = 0
        while j < len(columns) - 1:
            left = columns[j]
            k = j
            # Extend while no vertical line separates the next column in this band.
            while k < len(columns) - 2 and not _vertical_edge_in_band(
                columns[k + 1], top, bottom, verticals, tol=4
            ):
                k += 1
            right = columns[k + 1]
            col_spans.append((left, right, k - j + 1))
            j = k + 1

        for (left, right, col_span) in col_spans:
            col_index = columns.index(left)

            # Row span: extend downward while no horizontal line below.
            row_span = 1
            rr = r
            while rr < len(rows) - 2 and not _horizontal_edge_in_span(
                rows[rr + 1], left, right, horizontals, tol=4
            ):
                rr += 1
                row_span += 1

            cells.append(
                {
                    "id": f"cell_{cell_index:03d}",
                    "row": r,
                    "column": col_index,
                    "row_span": row_span,
                    "column_span": col_span,
                    "bbox": [
                        max(x1, left),
                        max(y1, top),
                        min(x2, right),
                        min(y2, bottom),
                    ],
                }
            )
            cell_index += 1

    return cells


def _vertical_edge_in_band(x, top, bottom, verticals, tol=4):
    for seg in verticals:
        if abs(seg["x"] - x) <= tol and seg["y1"] <= top + tol and seg["y2"] >= bottom - tol:
            return True
    return False


def _horizontal_edge_in_span(y, left, right, horizontals, tol=4):
    for seg in horizontals:
        if abs(seg["y"] - y) <= tol and seg["x1"] <= left + tol and seg["x2"] >= right - tol:
            return True
    return False


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_candidate(candidate, gray, elements=None):
    """
    Heuristic table confidence in [0, 1]. The score combines several pieces of
    *independent* visual evidence:

      * line_strength      - how many horizontal+vertical segments support the
                             region, normalized by a reasonable minimum.
      * intersection_density - cross-points of h×v segments fill the region.
      * rectangularity     - how well the structure resembles a rectangle.
      * border_completeness - dark ink along the four border edges.
      * ocr_density        - how much OCR content sits inside the region.

    The weights and normalization are heuristic but documented here so the
    meaning of the number is reproducible. There is no "calibration" claim.
    """
    bbox = candidate["bbox"]
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        candidate["confidence"] = 0.0
        return

    n_h = candidate["n_horizontal"]
    n_v = candidate["n_vertical"]
    n_x = candidate["n_intersections"]

    # A useful table has a minimum of ~3 rows and ~2 cols of evidence.
    line_strength = min(1.0, (n_h + n_v) / 8.0)

    max_x = max(1, (n_h - 1) * (n_v - 1))
    intersection_density = min(1.0, n_x / max_x)

    area = _bbox_area(bbox)
    mask_area = w * h
    rectangularity = min(1.0, area / max(1.0, mask_area * 0.85))

    border = _border_completeness(gray, bbox)

    ocr_density = 0.0
    if elements:
        inside = [
            e for e in elements if _point_in_bbox(_bbox_center(e["bbox"]), bbox)
        ]
        # Saturated at a modest count; a table rarely needs >40 elements to be
        # considered text-rich.
        ocr_density = min(1.0, len(inside) / 12.0)

    confidence = (
        0.35 * line_strength
        + 0.15 * intersection_density
        + 0.15 * rectangularity
        + 0.25 * border
        + 0.10 * ocr_density
    )

    candidate["confidence"] = round(float(min(1.0, max(0.0, confidence))), 4)


def _border_completeness(gray, bbox, sample=40):
    """
    Fraction of the four border edges that contain dark (ink) pixels.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = gray.shape
    if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h:
        return 0.5

    region = gray[y1:y2, x1:x2]
    _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    rh, rw = binary.shape
    if rh < 4 or rw < 4:
        return 0.0

    dirty = 0
    total = 0

    top = binary[0, :]
    bottom = binary[rh - 1, :]
    left = binary[:, 0]
    right = binary[:, rw - 1]

    for edge, length in ((top, rw), (bottom, rw), (left, rh), (right, rh)):
        indices = np.linspace(0, length - 1, min(sample, length)).astype(int)
        vals = edge.reshape(-1)[indices]
        dark = float(np.mean(vals > 60))
        dirty += dark
        total += 1

    return dirty / total if total else 0.0