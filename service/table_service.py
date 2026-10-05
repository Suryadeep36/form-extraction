"""
Table orchestration: CV detection + pretrained-model detection + fusion +
structure recognition + OCR-to-cell assignment.

Tables are first-class structures in the document representation. They are
kept separate from ordinary `regions` and from `fields`.
"""

import numpy as np
import cv2
import os
import re

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
from util.line_utils import detect_line_segments, merge_collinear_horizontal
from service.table_model_service import (
    get_table_model_service,
)
from service.detection_service import detect_vertical_lines


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
    # 1. Find the absolute highest physical line drawn by OpenCV
    if not cells:
        return
    highest_drawn_line = min(c["bbox"][1] for c in cells if "bbox" in c)

    # 2. Filter out any OCR elements whose center sits above the highest drawn line
    filtered_elements = []
    for el in elements:
        _, y_center = _bbox_center(el["bbox"])
        # If the word's center is below the table's top line, keep it
        if y_center >= highest_drawn_line:
            filtered_elements.append(el)
            
    # Swap out the elements list for the rest of the function
    elements = filtered_elements

    if cells and all("bbox" in c for c in cells):
        strict_x1 = min(c["bbox"][0] for c in cells)
        strict_y1 = min(c["bbox"][1] for c in cells)
        strict_x2 = max(c["bbox"][2] for c in cells)
        strict_y2 = max(c["bbox"][3] for c in cells)
        table_bbox = [strict_x1, strict_y1, strict_x2, strict_y2]
    
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
    height, width = image.shape[:2]

    # ---- CV detection (runs as baseline truth for structural cross-referencing)
    cv_candidates = detect_table_candidates(image, elements=elements)

    # ---- Page rules, shared by FILTER 4 in both branches -------------------
    horizontals = merge_collinear_horizontal(
        detect_line_segments(image, orientation="horizontal", min_length_ratio=0.20)
    )
    
    # ---- Model detection -------------------------------------------
    model_candidates = []
    model_consulted = False
    if config.TABLE_MODEL_ENABLED:
        model_service = get_table_model_service()
        model_candidates = model_service.detect_tables(image)
        # True only when the detector actually ran (loaded successfully).
        # "No tables found" from a working model is authoritative: a plain
        # box layout (rect on top, two squares in the middle, rect at the
        # bottom) must NOT be turned into a table that swallows the input
        # fields inside those boxes.
        model_consulted = model_service.available and not model_service.load_error
        
    final_candidates = []
    
    if model_candidates:
        print("[TABLE] ML Model detected candidates. Validating...")
        for mod in model_candidates:
            bbox = mod["bbox"]
            box_height = bbox[3] - bbox[1]
            box_width = bbox[2] - bbox[0]
            
            # FILTER 1: Aggressive Height Threshold
            # A real table with a header and at least one data row needs vertical space.
            # A single-line input box will almost always be less than 80-90px.
            if box_height < max(90, height * 0.05):
                print(f"[TABLE FILTER] Dropped ML fake table (too short, height={box_height}): {bbox}")
                continue
                
            # FILTER 2: CV Cross-reference for Single Cells
            # A single drawn box has exactly 4 intersections (the corners).
            # A real table (even a simple 1x2 grid) will have at least 6 intersections.
            cv_match = next((c for c in cv_candidates if _iou(bbox, c["bbox"]) > 0.5), None)
            if cv_match and cv_match.get("n_intersections", 0) <= 4:
                print(f"[TABLE FILTER] Dropped ML fake table (Single cell / <= 4 intersections): {bbox}")
                continue

            # FILTER 2b: Single-column row stacks are field groups, not tables.
            # A region whose CV grid has no interior vertical dividers (only the
            # two outer borders) and several stacked rows is a stack of form
            # fields (label + underline per row, e.g. SIGNATURE / Address /
            # Date / Telephone), not a table. Keeping it would swallow those
            # fields as empty cells.
            if cv_match and _is_single_column_stack(cv_match):
                print(f"[TABLE FILTER] Dropped ML fake table (single-column field stack): {bbox}")
                continue

            # FILTER 3: Field-layout lookalike
            # Whole-page layouts of form fields (labels + underlines/boxes)
            # are routinely misread as a single giant "table". A genuine
            # table's region holds few ":"-terminated field labels and a real
            # grid of several vertical rules; a form layout packs many colon
            # labels and little-to-no vertical structure. Rejecting it here
            # keeps the real input fields exposed instead of swallowing them
            # as (empty) table cells.
            # NOTE: cv_match.n_vertical is too permissive (it uses a lower
            # min_height_ratio=0.10 and counts short box edges). We use
            # detect_vertical_lines (min_height_ratio=0.15) which only
            # counts lines spanning ≥15% of the image — true grid rules.
            field_labels = [
                el for el in (elements or [])
                if (el.get("text") or "").strip().endswith(":")
                and bbox[1] - 60 <= (el["bbox"][1] + el["bbox"][3]) / 2 <= bbox[3]
            ]
            long_verticals = [
                v for v in detect_vertical_lines(image)
                if bbox[0] - 2 <= v["x"] <= bbox[2] + 2
            ]
            if len(field_labels) >= 3 and len(long_verticals) <= 3:
                print(
                    f"[TABLE FILTER] Dropped layout-as-table "
                    f"(field_labels={len(field_labels)}, verticals={len(long_verticals)}): {bbox}"
                )
                continue

            # FILTER 4: labelled form rows are not a data table.
            # A blank data-entry table prints its column headers once and
            # leaves the rest of its bands empty; a form section prints a
            # label in EVERY band.  Keeping the latter hides the section's real
            # input fields inside empty table cells.
            is_stack, why = _is_label_value_stack(bbox, horizontals, elements)
            if is_stack:
                print(
                    f"[TABLE FILTER] Dropped ML label/value stack "
                    f"({why}): {bbox}"
                )
                continue

            # If it survives the filters, it is a legitimate table
            final_candidates.append({
                "bbox": _clamp_bbox(bbox, width, height),
                "confidence": round(mod.get("confidence", 0.99), 4),
                "source": "model",
                "sources": ["model"],
            })
            
# CV fallback is a safety net ONLY for when the model could not be
    # consulted (disabled / failed to load). When the model DID run and
    # found no table, do not manufacture one from CV: box layouts are
    # routinely misread as grids and their fields would be hidden.
    if not final_candidates and not model_consulted:
        print("[TABLE] Model unavailable. Falling back to CV.")
        for cv in cv_candidates:
            bbox = cv["bbox"]
            
            # Filter fake CV tables
            if cv.get("n_intersections", 0) == 0:
                continue
            if (bbox[3] - bbox[1]) < max(60, height * 0.03):
                continue
            # FILTER 2b (single-column field stack), same as the model branch:
            # a stack of field underlines must not become a table.
            if _is_single_column_stack(cv):
                print(f"[TABLE FILTER] Dropped CV fake table (single-column field stack): {bbox}")
                continue
            # FILTER 3 (field-layout lookalike), same as the model branch:
            # whole-page layouts of form fields (labels + underlines/boxes)
            # are routinely misread as a single giant "table". Rejecting it
            # here keeps the real input fields exposed instead of swallowing
            # them as (empty) table cells.
            field_labels = [
                el for el in (elements or [])
                if (el.get("text") or "").strip().endswith(":")
                and bbox[1] - 60 <= (el["bbox"][1] + el["bbox"][3]) / 2 <= bbox[3]
            ]
            long_verticals = [
                v for v in detect_vertical_lines(image)
                if bbox[0] - 2 <= v["x"] <= bbox[2] + 2
            ]
            if len(field_labels) >= 3 and len(long_verticals) <= 3:
                print(
                    f"[TABLE FILTER] Dropped layout-as-table "
                    f"(field_labels={len(field_labels)}, verticals={len(long_verticals)}): {bbox}"
                )
                continue
            
            # FILTER 4 (labelled form rows), same as the model branch.
            is_stack, why = _is_label_value_stack(bbox, horizontals, elements)
            if is_stack:
                print(
                    f"[TABLE FILTER] Dropped CV label/value stack "
                    f"({why}): {bbox}"
                )
                continue

            final_candidates.append({
                "bbox": _clamp_bbox(bbox, width, height),
                "confidence": round(cv.get("confidence", 0.5), 4),
                "source": "cv",
                "sources": ["cv"],
            })

    return final_candidates, cv_candidates

# def detect_and_fuse_tables(image, elements):
#     """
#     Stage 1: detect tables (CV + optional model), fuse, return lightweight
#     candidates. Structure recognition / OCR assignment happen in stage 2 so
#     the field layer can exclude table regions first.

#     Returns:
#         (fused, cv_candidates)
#             fused: [{bbox, confidence, source, sources}]
#             cv_candidates: raw CV candidates with grid structure (cells rows).
#                 These are needed later for the CV structure-recognition
#                 fallback.
#     """
#     height, width = image.shape[:2]

#     # ---- CV detection ----------------------------------------------------
#     print("[TABLE-CV] Detecting tables...")
#     cv_candidates = detect_table_candidates(image, elements=elements)
#     print(f"[TABLE-CV] Candidates: {len(cv_candidates)}")
#     print(cv_candidates)
#     # ---- Model detection -------------------------------------------------
#     model_candidates = []
#     model_service = get_table_model_service()
#     if config.TABLE_MODEL_ENABLED:
#         print("[TABLE-MODEL] Detecting tables...")
#         model_candidates = model_service.detect_tables(image)
#         print(f"[TABLE-MODEL] Candidates: {len(model_candidates)}")
#         print(model_candidates)

    # print("\n" + "="*40)
    # print("🧨 DEBUG: RAW TABLE DETECTION OUTPUT")
    # print("="*40)
    
    # if cv_candidates:
    #     cv_box = [int(x) for x in cv_candidates[0]['bbox']]
    #     cv_rows = max((c.get('row', 0) for c in cv_candidates[0].get('cells', [])), default=0) + 1
    #     cv_cols = max((c.get('column', 0) for c in cv_candidates[0].get('cells', [])), default=0) + 1
    #     print(f"[CV MODEL]     BBox: {cv_box}")
    #     print(f"[CV MODEL]     Grid: {cv_rows} rows x {cv_cols} cols")
    # else:
    #     print("[CV MODEL]     No table detected.")

    # if model_candidates:
    #     mod_box = [int(x) for x in model_candidates[0]['bbox']]
    #     print(f"[ML MODEL]     BBox: {mod_box} (Confidence: {model_candidates[0].get('confidence')})")
    # else:
    #     print("[ML MODEL]     No table detected.")
    # print("="*40 + "\n")

    # debug_img = image.copy()
    
    # # Draw CV bounding box in BLUE
    # if cv_candidates:
    #     x1, y1, x2, y2 = [int(v) for v in cv_candidates[0]["bbox"]]
    #     cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    #     cv2.putText(debug_img, "CV (BLUE)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # # Draw ML bounding box in RED
    # if model_candidates:
    #     x1, y1, x2, y2 = [int(v) for v in model_candidates[0]["bbox"]]
    #     cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    #     cv2.putText(debug_img, "ML (RED)", (x2 - 150, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # # Save to current directory
    # cv2.imwrite("debug_table_boxes.jpg", debug_img)
    # print("🧨 DEBUG: Saved 'debug_table_boxes.jpg'. Open it to see the bounding boxes!")


    # ---- Fusion -----------------------------------------------------------
    # print("[TABLE-FUSION] Merging candidates...")
    # fused = _fuse_candidates(cv_candidates, model_candidates, (height, width))
    # print(f"[TABLE-FUSION] Final tables: {len(fused)}")
    # print(fused)

    # for table in fused:
    #     table["bbox"] = _clamp_bbox(table["bbox"], width, height)

    # return fused, cv_candidates


# def finalize_tables(fused, doc_rep, image):
#     """
#     Stage 2: for each fused candidate run structure recognition and assign OCR
#     text to cells.

#     Structure authority is the pretrained model: its SLANeXt topology is what
#     decides how many rows/columns the grid really has (and which cells span).
#     OpenCV is used only for precise cell geometry - the drawn line positions
#     that map OCR text into the cells.  Models are trusted for the *structural
#     description* of a table; CV (the drawn grid) is what locates the values.

#     When the model is unavailable the CV grid is used as-is so the system
#     still works on machines without the model stack installed.
#     """
#     height, width = image.shape[:2]
#     elements = doc_rep.get("elements", [])
#     model_service = get_table_model_service()

#     tables = []
#     for i, table in enumerate(fused):
#         bbox = [float(v) for v in table["bbox"]]

#         cells = []
#         structure_source = None

#         cv_table = next(
#             (c for c in (doc_rep.get("_cv_tables") or []) if _iou(bbox, c["bbox"]) > 0.1),
#             None,
#         )

#         # Model topology: the grid dimensions (rows x columns) are taken from
#         # the pretrained structure model.  It is trusted to say how many rows
#         # and columns the table really has - which fixes CV over-extending the
#         # grid into footer text below the last drawn row.
#         model_rows = None
#         model_cols = None
#         if model_service.available:
#             wired = _looks_wired(
#                 image[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
#             )
#             model_cells, n_rows, n_cols = model_service.recognize_structure(
#                 image[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])],
#                 wired=wired,
#             )
#             if model_cells:
#                 model_rows, model_cols = n_rows, n_cols

#         # Prefer the drawn CV grid for geometry, then clip its extent to the
#         # model-decided row/column count so phantom rows/cols added by CV
#         # beyond the real grid (footer text, border gutters) do not survive.
#         if cv_table and cv_table.get("cells"):
#             cells = _clip_cv_cells(
#                 cv_table["cells"], model_rows, model_cols
#             )
#             structure_source = "cv"
#             # The cells are defined over the CV candidate's grid extent.
#             bbox = [float(v) for v in cv_table["bbox"]]
#             print(
#                 f"[TABLE-STRUCTURE] table_{i:03d} cells: {len(cells)} "
#                 f"(source=cv rows={model_rows} cols={model_cols})"
#             )
#         elif model_service.available and model_rows is not None:
#             if model_cells:
#                 model_cells = _offset_bboxes(model_cells, bbox[0], bbox[1])
#                 # SLANeXt occasionally over-extends beyond the detected table
#                 # and produces phantom rows that swallow unrelated text below
#                 # (e.g. a "FOR OFFICE USE ONLY" strip). Drop cells whose
#                 # centre falls outside the table bbox.
#                 model_cells = _clip_phantom_cells(model_cells, bbox)
#                 cells = model_cells
#                 structure_source = "model"
#                 print(
#                     f"[TABLE-STRUCTURE] table_{i:03d} cells: {len(cells)} "
#                     f"(rows={model_rows}, source=model)"
#                 )

#         _assign_ocr_to_cells(elements, cells, bbox)

#         tables.append(
#             {
#                 "id": f"table_{i:03d}",
#                 "bbox": [round(v, 2) for v in bbox],
#                 "confidence": table.get("confidence"),
#                 "source": table.get("source"),
#                 "sources": table.get("sources", []),
#                 "cells": cells,
#                 "structure_source": structure_source,
#                 "n_cells": len(cells),
#             }
#         )

#     print(f"[OCR-CELL] Tables processed: {len(tables)}")
#     return tables

def _header_column_clusters(elements, row_bbox, median_text_h):
    """
    Group the printed text of a table's header row into one x-interval per
    printed column.

    Two labels belong to the same column when their x-ranges touch or overlap,
    which is what a header wrapped over two printed lines looks like
    ("Year of" over "Graduate").  Labels printed side by side never overlap, so
    they stay separate.
    """
    x1, y1, x2, y2 = [float(v) for v in row_bbox]
    band = 0.5 * max(float(median_text_h or 0), 10.0)
    spans = []
    for el in elements:
        b = el.get("bbox")
        if not b or len(b) != 4:
            continue
        text = (el.get("text") or "").strip()
        if len(text) < 2 or not re.search(r"[A-Za-z0-9\u00C0-\u024F]", text):
            continue
        cy = (b[1] + b[3]) / 2.0
        if not (y1 - band <= cy <= y2 + band):
            continue
        if b[2] <= x1 + 1 or b[0] >= x2 - 1:
            continue
        # Cluster against the label's own extent, NOT the drawn cell's: the
        # point of this pass is that a drawn cell spans several labels.
        spans.append([float(b[0]), float(b[2])])
    if not spans:
        return []
    spans.sort()
    clusters = []
    for a, b in spans:
        if clusters and a <= clusters[-1][1]:
            clusters[-1][1] = max(clusters[-1][1], b)
        else:
            clusters.append([a, b])
    return clusters


def _refine_columns_from_header(cells, elements, bbox, median_text_h):
    """
    Re-derive a table's column boundaries from its header text.

    Many printed forms rule their grid for FILING, not for the header: the
    vertical rules land wherever the boxes below need them and routinely cut
    straight through a header label, so a five-column header reads back as one
    cell with all five labels concatenated ("Year of Degree / Course University
    / Institute Grade City Graduate").  The header's own label spacing is the
    only faithful statement of the column layout, so prefer it.

    This only ever SPLITS -- if the header does not clearly subdivide the drawn
    grid the ruled structure is left untouched, so a table whose rules already
    match its header is unaffected.
    """
    if not cells:
        return cells
    rows = {}
    for c in cells:
        rows.setdefault(c.get("row", 0), []).append(c)
    if len(rows) < 2:
        return cells  # a single band is a field block, not a column grid
    n_cols = max((c.get("column", 0) for c in cells), default=0) + 1
    tx1, _ty1, tx2, _ty2 = [float(v) for v in bbox]
    # Collect header labels across the WHOLE table width, not inside the first
    # drawn cell.  A merged cell spans several labels by construction, so
    # scoping the search to one cell's x-range would only ever see its own
    # label and never find a split.
    top_row = min(rows)
    y_lo = min(c["bbox"][1] for c in rows[top_row])
    y_hi = max(c["bbox"][3] for c in rows[top_row])
    clusters = _header_column_clusters(
        elements, [tx1, y_lo, tx2, y_hi], median_text_h
    )
    if len(clusters) <= n_cols:
        return cells
    # Every gap must be a real column gutter.  A phrase printed continuously
    # ("Skill & Training Achievement(s)") arrives as one wide block, and a form
    # with no clear gutter anywhere must not be chopped up.
    min_gap = 0.9 * max(float(median_text_h or 0), 10.0)
    if any(
        clusters[i + 1][0] - clusters[i][1] < min_gap
        for i in range(len(clusters) - 1)
    ):
        return cells

    tx1, _ty1, tx2, _ty2 = [float(v) for v in bbox]
    # Split at the gutter MIDPOINT, but never inside a label or outside the
    # table's own extent.
    bounds = [tx1]
    for i in range(len(clusters) - 1):
        mid = (clusters[i][1] + clusters[i + 1][0]) / 2.0
        bounds.append(min(max(mid, clusters[i][1] + 1.0), clusters[i + 1][0] - 1.0))
    bounds.append(tx2)

    # Preserve every drawn ROW extent separately: rows of the same grid can be
    # different heights, and reusing whichever cell happened to be first would
    # flatten them all onto one band.
    row_span = {}
    for r, cs in rows.items():
        row_span[r] = [
            min(c["bbox"][1] for c in cs),
            max(c["bbox"][3] for c in cs),
        ]

    refined = []
    for r in sorted(rows):
        ry1, ry2 = row_span[r]
        for ci in range(len(clusters)):
            refined.append({
                # Carry the original cell's identity: OCR assignment keys off
                # `id`, and a header-derived cell replaces one drawn cell.
                "id": rows[r][min(ci, len(rows[r]) - 1)]["id"],
                "row": r,
                "column": ci,
                "bbox": [bounds[ci], ry1, bounds[ci + 1], ry2],
            })
    for n, c in enumerate(refined):
        c["id"] = f"{c['id']}_h{n}"
    return refined


def _page_layout_score(cells, bbox, width, height):
    """
    FILTER 5 - reject a candidate that is really the whole page's field layout.

    A form printed as a stack of full-width labelled rows (PATIENT DETAILS:
    Name / Date of Birth / Address / Phone ...) rules horizontal lines across
    the page for each field and a box or two around the page edge.  The grid
    detector reads that as one enormous "table" spanning the entire sheet, and
    the field rows disappear inside its cells.

    Two signals, both required, so a genuine large table survives:

      * the candidate covers most of the SHEET -- a data table is a region of
        a form, never the page itself;
      * most rows are a single undivided cell -- real grid rows are divided by
        the interior vertical rules into the shared column set.

    Returns the share of undivided rows when the candidate is page-sized (a
    truthy value meaning "reject"), else 0.0.  Exposed as a score so the bbox
    stage can apply the SAME test: stage 1 has no grid cells, so it falls back
    to the candidate's own aspect/extent and the number of ruled bands.
    """
    if not cells:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if width <= 0 or height <= 0:
        return 0.0
    area = ((x2 - x1) * (y2 - y1)) / float(width * height)
    if area < 0.75:
        return 0.0
    rows = {}
    for c in cells:
        rows.setdefault(c.get("row", 0), []).append(c)
    if len(rows) < 4:
        return 0.0
    undivided = sum(1 for cs in rows.values() if len(cs) == 1)
    share = undivided / len(rows)
    return share if share >= 0.6 else 0.0


def _is_page_layout(cells, bbox, width, height):
    return bool(_page_layout_score(cells, bbox, width, height))


def _looks_like_page_layout_bbox(bbox, horizontals, width, height, min_bands=4):
    """
    Stage-1 (bbox-only) form of the same FILTER 5.

    `table_bboxes` are consumed by field detection to suppress input regions,
    so a page-sized false table that survives stage 1 does not merely render as
    a wrong table -- it silently deletes every real field inside it. This must
    therefore reject the bbox BEFORE it becomes an exclusion box.

    With no grid available, a page-sized candidate is judged by its ruled band
    count: a real data table's internal rules divide it into rows whose heights
    match its text, while a stack of full-width form-field rows (each field's
    own underline plus the surrounding box edges) yields many thin, evenly
    stacked bands.
    """
    if width <= 0 or height <= 0:
        return False
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if ((x2 - x1) * (y2 - y1)) / float(width * height) < 0.75:
        return False
    ys = sorted({
        float(h["y"]) for h in (horizontals or [])
        if y1 - 4 <= h["y"] <= y2 + 4
    })
    if len(ys) < min_bands:
        return False
    bands = [(b - a) for a, b in zip(ys[:-1], ys[1:])]
    if not bands:
        return False
    band_h = (y2 - y1) / len(bands)
    median_band = float(np.median(bands))
    # Rows thin relative to the page, and no one of them is a tall data row.
    thin = sum(1 for b in bands if b < 0.06 * height)
    # A real data table's rows are the tallest thing on the page (a ruled row
    # tall enough to write in); a stack of field underlines is uniformly
    # thin. Compare the band's median against a single text line -- anything
    # near or below one line cannot be a data row.
    line = 0.035 * height
    return thin >= 0.6 * len(bands) and median_band <= max(2.5 * line, line)


def finalize_tables(fused, doc_rep, image):
    """
    Stage 2: Hybrid Extraction.
    Uses the Machine Learning bounding box (PicoDet) for the outer table limits, 
    but uses OpenCV for the internal grid structure, clamping the CV cells 
    strictly to the ML boundaries to cut off floating text and footers.
    """
    height, width = image.shape[:2]
    elements = doc_rep.get("elements", [])

    tables = []
    for i, table in enumerate(fused):
        # 1. This bbox is the tight bounding box from the ML detection model
        ml_bbox = [float(v) for v in table["bbox"]]
        cells = []
        structure_source = None

        # 2. Find the OpenCV grid that overlaps with this ML table (IoU > 0.1)
        cv_table = next(
            (c for c in (doc_rep.get("_cv_tables") or []) if _iou(ml_bbox, c["bbox"]) > 0.1),
            None,
        )

        # 3. Apply the OpenCV Grid, but clamp it to the ML Bounding Box
        if cv_table and cv_table.get("cells"):
            raw_cv_cells = cv_table["cells"]
            mx1, my1, mx2, my2 = ml_bbox
            
            for cell in raw_cv_cells:
                if "bbox" not in cell:
                    continue
                    
                cx1, cy1, cx2, cy2 = cell["bbox"]
                
                # If the CV cell is completely outside the ML box (e.g. phantom footer rows), discard it
                if cy2 <= my1 or cy1 >= my2 or cx2 <= mx1 or cx1 >= mx2:
                    continue
                    
                # THE GUILLOTINE: Clamp the CV cell's coordinates to the ML boundary.
                # This mathematically forces the top row to start at 543 instead of 515.
                cell["bbox"] = [
                    max(cx1, mx1),
                    max(cy1, my1),
                    min(cx2, mx2),
                    min(cy2, my2)
                ]
                cells.append(cell)

            # 3b. Drop the thin top/bottom sliver rows the clamp can leave
            # behind (they are the phantom null rows) and renumber the grid.
            cells = _drop_edge_slivers(cells)
            structure_source = "cv"

            print(f"[TABLE-STRUCTURE] table_{i:03d} cells: {len(cells)} (source=hybrid_ml_cv)")

        # 4. Prefer the header's own label spacing for the column layout.
        heights = [
            (e["bbox"][3] - e["bbox"][1]) for e in elements
            if e.get("bbox") and len(e["bbox"]) == 4
            and (e["bbox"][3] - e["bbox"][1]) > 0
        ]
        median_text_h = float(np.median(heights)) if heights else 20.0
        before_cols = max((c.get("column", 0) for c in cells), default=0) + 1
        cells = _refine_columns_from_header(cells, elements, ml_bbox, median_text_h)
        after_cols = max((c.get("column", 0) for c in cells), default=0) + 1
        if after_cols != before_cols:
            print(
                f"[TABLE-HEADER-COLS] table_{i:03d} cols {before_cols} -> "
                f"{after_cols} from header labels"
            )

        # 5. FILTER 5: whole-page field layout masquerading as one table.
        # Runs after structure is built because it reads the grid's row shape.
        if _is_page_layout(cells, ml_bbox, width, height):
            print(
                f"[TABLE FILTER] Dropped page-layout-as-table "
                f"(page-sized, mostly undivided rows): {ml_bbox}"
            )
            continue

        # 6. Assign OCR strictly within this clamped geometry
        _assign_ocr_to_cells(elements, cells, ml_bbox)

        tables.append(
            {
                "id": f"table_{i:03d}",
                "bbox": [round(v, 2) for v in ml_bbox],
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


def _is_single_column_stack(cv_candidate):
    """
    Detect a vertical stack of form-field rows that LOOKS like a bordered
    grid but is really one field under each label (Signature / Address /
    Date / Telephone ...).

    Signals: the CV grid has no interior vertical dividers (at most the two
    outer border lines => `n_vertical <= 2`) but at least three stacked rows.
    A genuine table carries at least one interior column divider; a form
    section with only label rows is a field group, not a table.
    """
    n_vertical = cv_candidate.get("n_vertical", 0) or 0
    n_horizontal = cv_candidate.get("n_horizontal", 0) or 0
    return n_vertical <= 2 and n_horizontal >= 3


def _band_text_occupancy(bbox, horizontals, elements, tol=4):
    """How many rule-bounded row bands of `bbox` carry printed text.

    Returns (text_bearing, total).  Bands are the spans BETWEEN consecutive
    horizontal rules that fall inside the bbox; the slivers above the first and
    below the last rule are ignored because a candidate bbox is not always
    rule-aligned.
    """
    ys = sorted({
        float(h["y"]) for h in (horizontals or [])
        if bbox[1] - tol <= h["y"] <= bbox[3] + tol
    })
    if len(ys) < 2:
        return 0, 0
    bearing = 0
    for y1, y2 in zip(ys[:-1], ys[1:]):
        for el in elements or []:
            if el.get("type") not in (None, "text") or not el.get("bbox"):
                continue
            cy = (el["bbox"][1] + el["bbox"][3]) / 2.0
            if y1 - 6 <= cy <= y2 + 6:
                bearing += 1
                break
    return bearing, len(ys) - 1


def _is_label_value_stack(bbox, horizontals, elements):
    """FILTER 4 - reject a "table" that is really a stack of labelled form rows.

    A blank data-entry table prints its column HEADERS once and then leaves the
    remaining row bands empty for handwriting, so its rules enclose at least
    one completely blank band.  A form section built from labelled rows
    (PERSONAL INFORMATION: Full Name / Nationality, Address, Phone / Email /
    DoB, Driving License, Marital Status) prints a label in EVERY band, and so
    does a single option row (Full-Time / Part-Time / Contract).

    Neither is tabular data, and keeping them turns every input field in the
    section into an empty table cell, which is what hides the form's real
    fields.  Reject both shapes:

      * fewer than two rule-bounded bands - a lone option row has no data row;
      * three or more bands and NOT ONE of them blank - every band is a
        printed label, i.e. a field stack.

    `horizontals` is the page's detected horizontal rules (see
    `util.line_utils.detect_line_segments` / `merge_collinear_horizontal`).
    """
    bearing, total = _band_text_occupancy(bbox, horizontals, elements)
    if total < 2:
        return True, f"no data row (bands={total})"
    if total >= 3 and bearing == total:
        return True, f"every band is a printed label ({bearing}/{total})"
    return False, f"bands {bearing}/{total} carry text"


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


def _drop_edge_slivers(cells):
    """
    Remove ultra-thin leading/trailing row bands left over by the ML-bbox clamp.

    When a table's outer border line is detected a hair inside the model's
    bounding box, the clamp smears the first and last grid rows into thin
    slivers (a few px tall) at the top and bottom edges.  These produce the
    extra null rows seen as a "9x6 instead of 7x6" table.  Any band that is
    much shorter than a genuine row is a clamp artifact, not a real row; we
    drop only the slivers at the leading/trailing edges (interior thin bands,
    e.g. partial sub-header lines, are kept) and renumber the surviving rows
    to a clean 0-based, contiguous grid.

    Surviving cells keep their geometry; only their `row` index (and any
    `row_span`) is recomputed from the remaining row bands.
    """
    if not cells:
        return cells

    bands = {}
    for cell in cells:
        b = cell.get("bbox")
        if not b or len(b) != 4:
            continue
        key = (round(b[1], 1), round(b[3], 1))
        bands.setdefault(key, []).append(cell)

    ordered = sorted(((y1, y2, cs) for (y1, y2), cs in bands.items()), key=lambda t: (t[0], t[1]))
    if not ordered:
        return cells

    heights = [y2 - y1 for (y1, y2, _) in ordered]
    median = float(np.median(heights)) if heights else 0.0
    if median <= 0:
        return cells

    tol = 0.3 * median

    # Drop slivers hugging the very top edge.
    while len(ordered) > 1 and (ordered[0][1] - ordered[0][0]) < tol:
        ordered.pop(0)
    # Drop slivers hugging the very bottom edge.
    while len(ordered) > 1 and (ordered[-1][1] - ordered[-1][0]) < tol:
        ordered.pop()

    if len(ordered) < 2:
        return cells

    # New row index for each surviving cell = position of the band its top
    # aligns with, in the cleaned (0-based) row order.
    row_index = {id(c): r for r, (y1, y2, cs) in enumerate(ordered) for c in cs}

    out = []
    for cell in cells:
        b = cell.get("bbox")
        if not b or len(b) != 4 or id(cell) not in row_index:
            continue
        result = dict(cell)
        result["row"] = row_index[id(cell)]
        out.append(result)
    return out


def _clip_cv_cells(cells, model_rows, model_cols):
    """
    Trim a CV-built cell grid down to the model-decided dimensions.

    The drawn-grid (CV) structure is geometrically precise but can add phantom
    rows below the last real row - e.g. a long vertical line or the border
    reaching into footer text ("Kindly sanction..."), or extra gutter columns
    at the edges.  The pretrained structure model is trusted on how many rows
    and columns the table truly has, so any CV cell living on or beyond those
    bounds is dropped.  When the model gave no dimensions the grid is kept
    untouched.

    Row/column indices are the logical grid addresses emitted by
    `build_cv_structure` (0-based), so clipping is a plain index comparison.
    """
    if model_rows is None and model_cols is None:
        return cells
    if model_rows is None:
        model_rows = model_cols
    if model_cols is None:
        model_cols = model_rows

    keep = []
    for cell in cells:
        row = cell.get("row")
        column = cell.get("column")
        row_span = int(cell.get("row_span") or 1)
        col_span = int(cell.get("column_span") or 1)
        if row is not None and row_span and row + row_span > model_rows:
            continue
        if column is not None and col_span and column + col_span > model_cols:
            continue
        keep.append(cell)
    return keep


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