"""
Table orchestration: CV detection + pretrained-model detection + fusion +
structure recognition + OCR-to-cell assignment.

Tables are first-class structures in the document representation. They are
kept separate from ordinary `regions` and from `fields`.
"""

import numpy as np
import cv2

import util.config as config
from util.geometry_utils import (
    _iou,
    _bbox_center,
    _bbox_intersection,
    _intersection_over_area,
    _point_in_bbox,
)
from util.table_utils import (
    detect_table_candidates,
    build_cv_structure,
)
from service.table_model_service import (
    get_table_model_service,
)


# ---------------------------------------------------------------------------
# Candidate fusion
# ---------------------------------------------------------------------------

def _fuse_candidates(cv_candidates, model_candidates, image_shape):
    """
    Merge CV + model detections into a single list of table candidates.

    Rules (documented in the project README as well):
      1. Strong geometric overlap   -> one table, sources union.
      2. Model-only detection       -> accepted as-is.
      3. CV-only detection          -> accepted with its own confidence.
      4. Two nearby tables are NOT merged just because they are close.
      5. One detector's box covering two clearly separate tables is split
         using the other detector's structure.
    """
    h, w = image_shape[:2]

    def _label_source(item):
        return "model" if item.get("source") == "model" else "cv"

    all_items = [dict(c) for c in cv_candidates]
    all_items.extend(dict(c) for c in model_candidates)

    # Sort best-first so confident detections anchor merges.
    all_items.sort(key=lambda it: it.get("confidence", 0.0), reverse=True)

    fused = []

    def _contains_or_overlaps(a, b, iou_threshold):
        iou = _iou(a, b)
        if iou >= iou_threshold:
            return True
        inter = _bbox_intersection(a, b)
        if inter is None:
            return False
        inter_area = (inter[2] - inter[0]) * (inter[3] - inter[1])
        small = min(_area(a), _area(b))
        return small > 0 and inter_area / small >= 0.85

    def _area(bbox):
        x1, y1, x2, y2 = bbox
        return max(1.0, (x2 - x1) * (y2 - y1))

    for item in all_items:
        bbox = list(item["bbox"])
        src = item.get("source", "cv")

        merged_into = None

        for kept in fused:
            if _contains_or_overlaps(bbox, kept["bbox"], config.TABLE_FUSION_IOU_THRESHOLD):
                # Model overrides CV when geometric confidence is comparable.
                if src == "model" and item.get("confidence", 0) >= kept["confidence"]:
                    kept["bbox"] = bbox
                kept["sources"] = list(
                    dict.fromkeys(kept["sources"] + [src])
                )
                kept["confidence"] = round(
                    max(kept["confidence"], item.get("confidence", 0.0)), 4
                )
                merged_into = kept
                break

        if merged_into is not None:
            continue

        # ---- Case 5: split a big box containing multiple tables ----------
        if src == "model":
            split = _try_split_multi_table(item, fused, image_shape)
            if split:
                continue

        fused.append(
            {
                "bbox": bbox,
                "confidence": round(item.get("confidence", 0.0), 4),
                "source": src,
                "sources": [src],
            }
        )

    # Filter CV-only tables below the CV confidence floor.
    filtered = []
    for table in fused:
        if "cv" in table["sources"] and "model" not in table["sources"]:
            if table["confidence"] < config.TABLE_CV_MIN_CONFIDENCE:
                continue
        filtered.append(table)

    return filtered


def _try_split_multi_table(item, fused, image_shape):
    """
    If `item` (model detection) roughly contains two already-kept CV tables
    that are far apart, split it: keep both CV tables instead of the big box.
    """
    containees = [
        kept
        for kept in fused
        if "cv" in kept["sources"]
        and "model" not in kept["sources"]
        and _iou(item["bbox"], kept["bbox"]) > 0.8
    ]

    if len(containees) < 2:
        return False

    centers = [_bbox_center(c["bbox"]) for c in containees]
    ys = sorted(c[1] for c in centers)
    if ys[-1] - ys[0] > (item["bbox"][3] - item["bbox"][1]) * 0.5:
        # Two tables stacked vertically inside one model box.
        for kept in containees:
            if "cv" in kept["sources"]:
                kept["sources"] = list(
                    dict.fromkeys(kept["sources"] + ["model"])
                )
        return True
    return False


# ---------------------------------------------------------------------------
# Structure + OCR assignment
# ---------------------------------------------------------------------------

def _assign_ocr_to_cells(elements, cells, table_bbox):
    """
    Deterministic OCR-to-cell mapping.

    Each cell receives:
        "text"             - joined text
        "ocr_element_ids"  - ids of assigned elements
        "ocr_word_ids"     - word ids within assigned elements

    Order of preference per element:
        1. element center inside exactly one cell
        2. intersection-over-area with a cell >= threshold
        3. word-level boxes spread across multiple cells (compound header
           texts spanning merged cells)
    """
    x1, y1, x2, y2 = table_bbox
    cell_assignments = {}

    word_text = {}  # (element_id, word_index) -> word text for reconstruction

    for element in elements:
        center = _bbox_center(element["bbox"])

        # Prefer words when the element spans several cells.
        words = element.get("words")
        if words:
            resolved = []
            for wi, word in enumerate(words):
                wb = word.get("bbox")
                if not wb or len(wb) != 4:
                    continue
                wcenter = _bbox_center([float(v) for v in wb])
                target = _cell_for_point(cells, wcenter)
                if target is not None:
                    resolved.append((wi, target))
                elif _point_in_bbox(wcenter, table_bbox):
                    target = _cell_for_ioa(cells, [float(v) for v in wb])
                    if target is not None:
                        resolved.append((wi, target))
            if resolved and len(resolved) > 1:
                for wi, cell_id in resolved:
                    cell_assignments.setdefault(cell_id, []).append(
                        (element["id"], wi)
                    )
                    word_text[(element["id"], wi)] = str(
                        words[wi].get("text", "")
                    ).strip()
                continue

        target = _cell_for_point(cells, center)
        if target is not None:
            cell_assignments.setdefault(target, []).append(
                (element["id"], None)
            )
            continue

        target = _cell_for_ioa(cells, element["bbox"])
        if target is not None:
            cell_assignments.setdefault(target, []).append(
                (element["id"], None)
            )

    for cell in cells:
        cell["text"] = None
        cell["ocr_element_ids"] = []
        cell["ocr_word_ids"] = []
        entries = cell_assignments.get(cell["id"], [])
        if not entries:
            continue

        element_ids = []
        texts = []
        word_ids = []
        for element_id, word_index in entries:
            element_ids.append(element_id)
            if word_index is not None:
                word_ids.append(f"{element_id}:{word_index}")
                wt = word_text.get((element_id, word_index))
                if wt:
                    texts.append(wt)
            else:
                element = next((e for e in elements if e["id"] == element_id), None)
                if element:
                    texts.append(element.get("text", ""))

        cell["text"] = " ".join(t for t in texts if t).strip() or None
        cell["ocr_element_ids"] = list(dict.fromkeys(element_ids))
        cell["ocr_word_ids"] = word_ids


def _cell_for_point(cells, point):
    matches = [c for c in cells if _point_in_bbox(point, c["bbox"])]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]["id"]
    # Overlapping (merged) cells: take the smallest.
    best = min(matches, key=lambda c: _area(c["bbox"]))
    return best["id"]


def _cell_for_ioa(cells, bbox):
    best_id = None
    best_ioa = 0.0
    for cell in cells:
        ioa = _intersection_over_area(bbox, cell["bbox"])
        if ioa > best_ioa:
            best_ioa = ioa
            best_id = cell["id"]
    return best_id if best_ioa >= 0.5 else None


def _area(bbox):
    x1, y1, x2, y2 = bbox
    return max(1.0, (x2 - x1) * (y2 - y1))


def _offset_bboxes(cells, dx, dy):
    out = []
    for cell in cells:
        bbox = cell["bbox"]
        out.append({**cell, "bbox": [bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy]})
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def detect_and_fuse_tables(image, elements):
    """
    Stage 1: detect tables (CV + optional model), fuse, return lightweight
    candidates. Structure recognition / OCR assignment happen in stage 2 so
    the field layer can exclude table regions first.

    Returns:
        (fused, cv_candidates)
            fused: [{bbox, confidence, source, sources}]
            cv_candidates: raw CV candidates with grid structure (cells rows).
                These are needed later for the CV structure-recognition
                fallback.
    """
    height, width = image.shape[:2]

    # ---- CV detection ----------------------------------------------------
    print("[TABLE-CV] Detecting tables...")
    cv_candidates = detect_table_candidates(image, elements=elements)
    print(f"[TABLE-CV] Candidates: {len(cv_candidates)}")
    print(cv_candidates)
    # ---- Model detection -------------------------------------------------
    model_candidates = []
    model_service = get_table_model_service()
    if config.TABLE_MODEL_ENABLED:
        print("[TABLE-MODEL] Detecting tables...")
        model_candidates = model_service.detect_tables(image)
        print(f"[TABLE-MODEL] Candidates: {len(model_candidates)}")
        print(model_candidates)
    # ---- Fusion -----------------------------------------------------------
    print("[TABLE-FUSION] Merging candidates...")
    fused = _fuse_candidates(cv_candidates, model_candidates, (height, width))
    print(f"[TABLE-FUSION] Final tables: {len(fused)}")
    print(fused)

    for table in fused:
        table["bbox"] = _clamp_bbox(table["bbox"], width, height)

    return fused, cv_candidates


def finalize_tables(fused, doc_rep, image):
    """
    Stage 2: for each fused candidate run structure recognition (model-first,
    CV fallback) and assign OCR text to cells.
    """
    height, width = image.shape[:2]
    elements = doc_rep.get("elements", [])
    model_service = get_table_model_service()

    tables = []
    for i, table in enumerate(fused):
        bbox = [float(v) for v in table["bbox"]]

        cells = []
        structure_source = None

        # Prefer the CV grid when one exists: it is rebuilt directly from the
        # drawn lines, so a fully-bordered table gets its exact rows/columns.
        # Learned (model) structure is used when there is no CV grid (weak or
        # fragmented lines) - that is the case the model is best at.
        cv_table = next(
            (c for c in (doc_rep.get("_cv_tables") or []) if _iou(bbox, c["bbox"]) > 0.5),
            None,
        )
        if cv_table and cv_table.get("cells"):
            cells = cv_table["cells"]
            structure_source = "cv"
            # The cells are defined over the CV candidate's grid extent.
            bbox = [float(v) for v in cv_table["bbox"]]
            print(
                f"[TABLE-STRUCTURE] table_{i:03d} cells: {len(cells)} "
                f"(source=cv)"
            )
        elif model_service.available:
            wired = _looks_wired(
                image[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
            )
            model_cells, n_rows, n_cols = model_service.recognize_structure(
                image[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])],
                wired=wired,
            )
            if model_cells:
                model_cells = _offset_bboxes(model_cells, bbox[0], bbox[1])
                # SLANeXt occasionally over-extends beyond the detected table
                # and produces phantom rows that swallow unrelated text below
                # (e.g. a "FOR OFFICE USE ONLY" strip). Drop cells whose
                # centre falls outside the table bbox.
                model_cells = _clip_phantom_cells(model_cells, bbox)
                cells = model_cells
                structure_source = "model"
                print(
                    f"[TABLE-STRUCTURE] table_{i:03d} cells: {len(cells)} "
                    f"(rows={n_rows}, source=model)"
                )

        _assign_ocr_to_cells(elements, cells, bbox)

        tables.append(
            {
                "id": f"table_{i:03d}",
                "bbox": [round(v, 2) for v in bbox],
                "confidence": table.get("confidence"),
                "source": table.get("source"),
                "sources": table.get("sources", []),
                "cells": cells,
                "structure_source": structure_source,
                "n_cells": len(cells),
            }
        )

    print(f"[OCR-CELL] Tables processed: {len(tables)}")
    return tables


def _clamp_bbox(bbox, width, height):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(x1, width))
    y1 = max(0.0, min(y1, height))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return [x1, y1, x2, y2]


def _cell_bbox_center(cell):
    bbox = cell["bbox"]
    return [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]


def _clip_phantom_cells(cells, table_bbox):
    """
    Drop cells whose centre lies outside the detected table bbox.

    Learned structure networks sometimes emit grid rows that extend far past
    the detected table (phantom rows that then swallow unrelated text below
    the table, e.g. a "FOR OFFICE USE ONLY" strip). This keeps the structure
    inside the region that was actually detected as a table.
    """
    x1, y1, x2, y2 = table_bbox
    tol_y = max(8.0, (y2 - y1) * 0.06)
    tol_x = max(8.0, (x2 - x1) * 0.06)

    clipped = []
    for cell in cells:
        cx, cy = _cell_bbox_center(cell)
        if x1 - tol_x <= cx <= x2 + tol_x and y1 - tol_y <= cy <= y2 + tol_y:
            clipped.append(cell)
    return clipped


def process_tables(doc_rep, image):
    """
    Legacy convenience entry point (single-shot): detect/fuse/finalize.
    """
    fused, cv_candidates = detect_and_fuse_tables(image, doc_rep.get("elements", []))
    doc_rep["_cv_tables"] = cv_candidates
    return finalize_tables(fused, doc_rep, image)


def _looks_wired(crop):
    """
    Decide whether a table crop is bordered ("wired") or borderless by
    measuring the amount of straight-line ink present.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    h, w = gray.shape
    if h < 20 or w < 20:
        return True

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h_kernel_len = max(15, int(w * 0.04))
    v_kernel_len = max(15, int(h * 0.04))
    h_lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    )
    v_lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    )

    h_ratio = cv2.countNonZero(h_lines) / max(1.0, h * w)
    v_ratio = cv2.countNonZero(v_lines) / max(1.0, h * w)

    return (h_ratio > 0.004 and v_ratio > 0.004) or (h_ratio + v_ratio > 0.012)