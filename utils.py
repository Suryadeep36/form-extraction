import re
from functools import lru_cache

import cv2
import numpy as np
from paddleocr import PaddleOCR
import json
from groq import Groq
import os


# ---------------------------------------------------------------------------
# Cached models
#
# The OCR engine and the orientation classifier are expensive to construct,
# so we build them once and reuse them across requests.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_ocr_engine():
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        lang="en",
    )


@lru_cache(maxsize=1)
def _get_orientation_model():
    from paddlex import create_model

    return create_model("PP-LCNet_x1_0_doc_ori")


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def _load_image(image_or_path):
    if isinstance(image_or_path, np.ndarray):
        return image_or_path
    img = cv2.imread(image_or_path)
    if img is None:
        raise FileNotFoundError(image_or_path)
    return img


def _load_gray(image_or_path):
    img = _load_image(image_or_path)
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _norm(bbox, image_width, image_height):
    x1, y1, x2, y2 = bbox
    return [
        round(x1 / image_width, 4),
        round(y1 / image_height, 4),
        round(x2 / image_width, 4),
        round(y2 / image_height, 4),
    ]


def _union_bbox(bboxes):
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return [x1, y1, x2, y2]


def _point_in_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


# ---------------------------------------------------------------------------
# Stage 1 - Document preprocessing
#
# Orientation correction + deskew so that OCR and computer-vision coordinates
# all live in the same upright coordinate space.
# ---------------------------------------------------------------------------

def _rotate_image(image, angle):
    if abs(angle) < 1e-6:
        return image

    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(mat[0, 0])
    sin = abs(mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    mat[0, 2] += (new_w - w) / 2
    mat[1, 2] += (new_h - h) / 2

    return cv2.warpAffine(
        image,
        mat,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderValue=(255, 255, 255),
    )


def _estimate_skew(image_or_path):
    gray = _load_gray(image_or_path)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 100:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    return angle


def preprocess_document(image_or_path, output_path=None):
    """
    Correct document orientation and deskew.

    Returns:
        {
            "image": ndarray,
            "path": path of the corrected image (or None),
            "orientation_corrected": int (0, 90, 180, 270),
            "deskew_angle": float
        }
    """
    image = _load_image(image_or_path)

    orientation = 0
    try:
        result = list(_get_orientation_model().predict(image))
        label = str(result[0]["label_names"][0])
        orientation = int(label)
        k = orientation // 90
        if k:
            image = np.rot90(image, k=k)
    except Exception as e:
        print(f"Orientation classification failed: {e}")

    skew = _estimate_skew(image)
    if abs(skew) >= 0.75:
        image = _rotate_image(image, skew)

    info = {
        "image": image,
        "path": output_path,
        "orientation_corrected": orientation,
        "deskew_angle": round(skew, 3),
    }

    if output_path:
        cv2.imwrite(output_path, image)

    return info


# ---------------------------------------------------------------------------
# Stage 2 - OCR
#
# Text + bounding boxes + confidence. Every element is identified by an
# immutable ID (t000, t001, ...) that the rest of the pipeline operates on.
# ---------------------------------------------------------------------------

def get_ocr_data(image_or_path):
    ocr = _get_ocr_engine()
    result = ocr.predict(image_or_path)

    if not result:
        return []

    page = result[0]

    if hasattr(page, "keys"):
        data = page.get("res", page)
    else:
        return []

    dt_polys = data.get("dt_polys", [])
    rec_texts = data.get("rec_texts", [])
    rec_scores = data.get("rec_scores", [])

    output = []

    for i, poly in enumerate(dt_polys):

        if i >= len(rec_texts):
            continue

        text = str(rec_texts[i]).strip()

        if not text:
            continue

        points = [
            [float(p[0]), float(p[1])]
            for p in poly
        ]

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)

        score = None

        if i < len(rec_scores):
            score = float(rec_scores[i])

        output.append({
            "id": f"t{i:03d}",
            "type": "text",
            "text": text,

            "bbox": [
                round(x1, 2),
                round(y1, 2),
                round(x2, 2),
                round(y2, 2)
            ],

            "center": [
                round((x1 + x2) / 2, 2),
                round((y1 + y2) / 2, 2)
            ],

            "width": round(x2 - x1, 2),
            "height": round(y2 - y1, 2),

            "confidence": score
        })

    return output


# ---------------------------------------------------------------------------
# Stage 3 - Layout analysis (CV)
#
# CV determines WHERE things are: lines, regions, checkboxes.
# ---------------------------------------------------------------------------

def _threshold_gray(gray):
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5, 
    )

def detect_horizontal_lines(image_or_path, min_width_ratio=0.20):
    img = _load_gray(image_or_path)
    h, w = img.shape

    binary = _threshold_gray(img)

    # 1. Use a much smaller kernel to survive gaps (e.g., 3% of width)
    kernel_length = max(15, int(w * 0.03)) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    # 2. Extract horizontal elements
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3. Aggressively dilate horizontally to bridge the gaps in noisy lines
    horizontal = cv2.dilate(horizontal, np.ones((2, 25), np.uint8), iterations=1)

    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)

        # 4. Enforce the strict 20% length rule HERE, on the reconstructed contour
        if width < w * min_width_ratio:
            continue

        # 5. Increased height tolerance for slight document skews
        if height > 40: 
            continue

        lines.append({
            "y": y + height // 2,
            "x1": x,
            "x2": x + width,
            "length": width
        })

    lines.sort(key=lambda x: x["y"])

    merged = []
    for line in lines:
        if not merged:
            merged.append(line)
            continue

        previous = merged[-1]
        if abs(line["y"] - previous["y"]) <= 10:
            if line["length"] > previous["length"]:
                merged[-1] = line
        else:
            merged.append(line)

    return merged


def detect_vertical_lines(image_or_path, min_height_ratio=0.15):
    img = _load_gray(image_or_path)
    h, w = img.shape

    binary = _threshold_gray(img)

    # 1. Small kernel for survival
    kernel_length = max(15, int(h * 0.03)) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))

    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # 2. Aggressively dilate vertically to bridge gaps
    vertical = cv2.dilate(vertical, np.ones((25, 2), np.uint8), iterations=1)

    contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)

        # 3. Enforce the strict height rule HERE
        if height < h * min_height_ratio:
            continue

        # 4. Increased width tolerance for skew
        if width > 40: 
            continue

        lines.append({
            "x": x + width // 2,
            "y1": y,
            "y2": y + height,
            "length": height
        })

    lines.sort(key=lambda x: x["x"])

    merged = []
    for line in lines:
        if not merged:
            merged.append(line)
            continue

        previous = merged[-1]
        if abs(line["x"] - previous["x"]) <= 10:
            if line["length"] > previous["length"]:
                merged[-1] = line
        else:
            merged.append(line)

    return merged


def _iou(bbox_a, bbox_b):
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    return inter / float(area_a + area_b - inter)


def detect_rectangular_regions(image_or_path, min_area_ratio=0.008):
    """
    Detect rectangular (bordered) regions in a scanned form.
    Uses grid fusion to merge cells into a master table boundary.
    """
    img = _load_gray(image_or_path)
    h, w = img.shape

    binary = _threshold_gray(img)

    # 1. Use smaller survival kernels (3% of dimensions)
    h_kernel_len = max(15, int(w * 0.03))
    v_kernel_len = max(15, int(h * 0.03))

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Aggressively dilate to bridge gaps in the lines
    horizontal = cv2.dilate(horizontal, np.ones((2, 25), np.uint8), iterations=1)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    
    # Aggressively dilate to bridge gaps in the lines
    vertical = cv2.dilate(vertical, np.ones((25, 2), np.uint8), iterations=1)

    # 2. Combine into a grid
    table_structure = cv2.add(horizontal, vertical)

    # 3. THE FUSION STEP: Fuse the grid lines into solid blocks
    # This prevents OpenCV from seeing individual cells.
    fuse_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    table_structure = cv2.morphologyEx(table_structure, cv2.MORPH_CLOSE, fuse_kernel)

    # 4. Use RETR_EXTERNAL to only get the outer perimeter of the fused block
    contours, _ = cv2.findContours(
        table_structure,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    regions = []
    min_area = w * h * min_area_ratio

    for contour in contours:
        x, y, rw, rh = cv2.boundingRect(contour)
        area = rw * rh

        if area < min_area:
            continue

        if rw < w * 0.12: # Must be at least 12% of image width
            continue

        if rh < 25:
            continue

        regions.append({
            "bbox": [x, y, x + rw, y + rh],
            "x": x,
            "y": y,
            "width": rw,
            "height": rh,
            "area": area
        })

    regions.sort(key=lambda r: r["area"], reverse=True)

    # 5. Remove near-duplicate / nested look-alike detections
    deduped = []
    for region in regions:
        is_duplicate = False
        for kept in deduped:
            if _iou(region["bbox"], kept["bbox"]) > 0.75:
                is_duplicate = True
                break
        if not is_duplicate:
            deduped.append(region)

    return deduped


def detect_checkboxes(image_or_path, min_size=15, max_size=60):
    """
    CV-first checkbox detection.

    Detects small square contours (checkbox outlines) and determines the
    checked state from the amount of ink inside the box.

    Returns:
        List of:
        {
            "id": "c000",
            "bbox": [x1, y1, x2, y2],
            "center": [cx, cy],
            "width": int,
            "height": int,
            "state": "checked" | "unchecked" | "uncertain",
            "confidence": float
        }
    """
    gray = _load_gray(image_or_path)
    binary = _threshold_gray(gray)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    raw = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        if w < min_size or h < min_size:
            continue

        if w > max_size or h > max_size:
            continue

        aspect = w / float(h)
        if not (0.72 <= aspect <= 1.38):
            continue

        area = cv2.contourArea(contour)
        rect_area = w * h
        fill = area / rect_area if rect_area > 0 else 1.0

        if fill < 0.3 or fill > 0.9:
            continue

        # Ring vs interior ink: a checkbox outline is dark at the border.
        x1 = max(0, x + 2)
        y1 = max(0, y + 2)
        x2 = min(gray.shape[1], x + w - 2)
        y2 = min(gray.shape[0], y + h - 2)

        interior = binary[y1:y2, x1:x2]
        if interior.size == 0:
            continue

        interior_dark = float(np.mean(interior > 0))

        raw.append({
            "bbox": [x, y, x + w, y + h],
            "width": w,
            "height": h,
            "interior_dark": interior_dark,
        })

    # Merge near-duplicate detections of the same box.
    merged = []

    for box in sorted(raw, key=lambda b: b["width"] * b["height"], reverse=True):

        duplicate = False

        for kept in merged:
            if _iou(box["bbox"], kept["bbox"]) > 0.5:
                duplicate = True
                break

        if not duplicate:
            merged.append(box)

    checkboxes = []

    for i, box in enumerate(merged):

        x1, y1, x2, y2 = box["bbox"]
        interior_dark = box["interior_dark"]

        if interior_dark >= 0.10:
            state = "checked"
            confidence = min(0.98, 0.6 + interior_dark * 2)
        elif interior_dark >= 0.045:
            state = "uncertain"
            confidence = 0.5
        else:
            state = "unchecked"
            confidence = 0.85

        checkboxes.append({
            "id": f"c{i:03d}",
            "bbox": [x1, y1, x2, y2],
            "center": [
                round((x1 + x2) / 2, 2),
                round((y1 + y2) / 2, 2)
            ],
            "width": box["width"],
            "height": box["height"],
            "state": state,
            "confidence": round(confidence, 3),
        })

    return checkboxes


# ---------------------------------------------------------------------------
# Stage 4 - Row grouping
# ---------------------------------------------------------------------------

def group_into_rows(items, y_tolerance=12):
    items = sorted(
        items,
        key=lambda x: x["center"][1]
    )

    rows = []

    for item in items:

        cy = item["center"][1]

        matched_row = None

        for row in rows:
            if abs(cy - row["center_y"]) <= y_tolerance:
                matched_row = row
                break

        if matched_row:
            matched_row["items"].append(item)
            matched_row["center_y"] = np.mean([
                x["center"][1]
                for x in matched_row["items"]
            ])
        else:
            rows.append({
                "center_y": cy,
                "items": [item]
            })

    for row in rows:
        row["items"].sort(
            key=lambda x: x["center"][0]
        )

    rows.sort(
        key=lambda x: x["center_y"]
    )

    return rows


# ---------------------------------------------------------------------------
# Stage 5 - Unified document representation
# ---------------------------------------------------------------------------

def build_document_representation(image, image_path=None):
    """
    Run OCR + layout analysis and assemble the internal document graph.

    Returns:
        {
            "image_path": str | None,
            "image_width": int,
            "image_height": int,
            "elements": [...],   # text elements (t000, ...)
            "regions": [...],    # r000, ...
            "checkboxes": [...], # c000, ...
            "lines": {
                "horizontal": [...],
                "vertical": [...]
            }
        }
    """
    if isinstance(image, np.ndarray):
        image_height, image_width = image.shape[:2]
    else:
        loaded = _load_image(image)
        image_height, image_width = loaded.shape[:2]
        image = loaded

    print("Running PaddleOCR...")
    raw_elements = get_ocr_data(image)
    print(f"OCR items: {len(raw_elements)}")

    print("Normalizing compound OCR elements (Pass 1)...")
    elements = normalize_ocr_elements(raw_elements)
    print(f"Normalized OCR items: {len(elements)}")

    print("Detecting horizontal lines...")
    horizontal_lines = detect_horizontal_lines(image)
    print(f"Horizontal lines: {len(horizontal_lines)}")

    print("Detecting vertical lines...")
    vertical_lines = detect_vertical_lines(image)
    print(f"Vertical lines: {len(vertical_lines)}")

    print("Detecting rectangular regions...")
    raw_regions = detect_rectangular_regions(image)
    regions = [
        {
            "id": f"r{i:03d}",
            "bbox": r["bbox"],
            "x": r["x"],
            "y": r["y"],
            "width": r["width"],
            "height": r["height"],
            "area": r["area"],
        }
        for i, r in enumerate(raw_regions)
    ]
    print(f"Regions: {len(regions)}")

    print("Detecting checkboxes...")
    checkboxes = detect_checkboxes(image)
    print(f"Checkboxes: {len(checkboxes)}")

    return {
        "image_path": image_path,
        "image_width": image_width,
        "image_height": image_height,
        "elements": elements,
        "regions": regions,
        "checkboxes": checkboxes,
        "lines": {
            "horizontal": horizontal_lines,
            "vertical": vertical_lines,
        },
    }


# ---------------------------------------------------------------------------
# Stage 6 - Candidate field builder (geometry-based)
#
# Deterministically score label <-> value candidate pairs. The LLM later
# chooses among these candidates (or adds its own), always by ID.
# ---------------------------------------------------------------------------

_LABEL_SUFFIX_RE = re.compile(r"[:\uFF1A?？]+$")
_NUMERIC_RE = re.compile(r"^[\d\s,.:/\-\u00B0%°]+$")


def _is_label_like(element):
    text = element["text"].strip()

    if not text:
        return False

    if _LABEL_SUFFIX_RE.search(text):
        return True

    if len(text) > 45:
        return False

    if _NUMERIC_RE.match(text):
        return False

    return True


def _relationship_score(label, value, regions, image_width, image_height):
    lx1, ly1, lx2, ly2 = label["bbox"]
    vx1, vy1, vx2, vy2 = value["bbox"]
    lc = label["center"]
    vc = value["center"]

    score = 0.0

    # 1. Vertical overlap (same row)
    inter = max(0, min(ly2, vy2) - max(ly1, vy1))
    lh = max(1, ly2 - ly1)
    vh = max(1, vy2 - vy1)
    v_overlap = inter / min(lh, vh)

    if v_overlap > 0.5:
        score += 3.0 * v_overlap

    dx = vc[0] - lc[0]
    dy = vc[1] - lc[1]

    # 2. Value to the right on the same row
    if v_overlap > 0.5 and vx1 >= lx2 - 8 and dx > 0:
        distance_ratio = dx / image_width
        score += 2.0 * max(0.0, 1.0 - distance_ratio * 4.0)

    # 3. Value directly below the label (left aligned)
    elif dy > 0 and vy1 >= ly2 - 4:
        distance_ratio = (vy1 - ly2) / max(1, image_height)
        lw = max(1, lx2 - lx1)
        vw = max(1, vx2 - vx1)
        align = max(0.0, 1.0 - abs(vx1 - lx1) / max(lw, vw))
        score += 2.0 * max(0.0, 1.0 - distance_ratio * 12.0) * (0.5 + 0.5 * align)

    # 4. Shared region containment
    for region in regions:
        rx1, ry1, rx2, ry2 = region["bbox"]
        if (_point_in_bbox(lc, region["bbox"])
                and _point_in_bbox(vc, region["bbox"])):
            score += 1.0
            break

    return score


def generate_field_candidates(
    doc_rep,
    max_candidates=5,
    min_score=1.2
):
    elements = doc_rep["elements"]
    regions = doc_rep["regions"]
    image_width = doc_rep["image_width"]
    image_height = doc_rep["image_height"]

    candidates = []

    for label in elements:

        if not _is_label_like(label):
            continue

        scored = []

        for value in elements:

            if value["id"] == label["id"]:
                continue

            if _is_label_like(value) and (
                    abs(value["center"][1] - label["center"][1]) <= 8
            ):
                # Two label-like elements on the same row: unlikely a pair.
                continue

            score = _relationship_score(
                label,
                value,
                regions,
                image_width,
                image_height
            )

            if score >= min_score:
                scored.append({
                    "value_id": value["id"],
                    "score": round(score, 3),
                })

        scored.sort(key=lambda x: x["score"], reverse=True)

        if scored:
            candidates.append({
                "label_id": label["id"],
                "candidates": scored[:max_candidates],
            })

    return candidates


# ---------------------------------------------------------------------------
# Stage 7 - LLM prompt (structural interpretation, IDs only)
# ---------------------------------------------------------------------------

def _format_elements_for_prompt(doc_rep):
    lines = []

    for element in doc_rep["elements"]:
        bbox = _norm(
            element["bbox"],
            doc_rep["image_width"],
            doc_rep["image_height"]
        )
        confidence = (
            "unknown" if element["confidence"] is None
            else f"{element['confidence']:.3f}"
        )
        lines.append(
            f'{element["id"]} "{element["text"]}" '
            f'bbox=({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}) '
            f'conf={confidence}'
        )

    return "\n".join(lines)


def _format_regions_for_prompt(doc_rep):
    lines = []

    for region in doc_rep["regions"]:
        bbox = _norm(
            region["bbox"],
            doc_rep["image_width"],
            doc_rep["image_height"]
        )
        lines.append(
            f'{region["id"]} bbox=({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}) '
            f'size={region["width"]}x{region["height"]}'
        )

    return "\n".join(lines)


def _format_checkboxes_for_prompt(doc_rep):
    lines = []

    for checkbox in doc_rep["checkboxes"]:
        bbox = _norm(
            checkbox["bbox"],
            doc_rep["image_width"],
            doc_rep["image_height"]
        )
        lines.append(
            f'{checkbox["id"]} state="{checkbox["state"]}" '
            f'conf={checkbox["confidence"]} '
            f'bbox=({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})'
        )

    return "\n".join(lines)


def _format_candidates_for_prompt(candidates):
    lines = []

    for candidate in candidates:
        pairs = ", ".join(
            f"{c['value_id']}({c['score']})"
            for c in candidate["candidates"]
        )
        lines.append(f'{candidate["label_id"]} -> {pairs}')

    return "\n".join(lines)


def build_structure_prompt(doc_rep, candidates):
    elements_repr = _format_elements_for_prompt(doc_rep)
    regions_repr = _format_regions_for_prompt(doc_rep)
    checkboxes_repr = _format_checkboxes_for_prompt(doc_rep)
    candidates_repr = _format_candidates_for_prompt(candidates)

    return f"""
You are a document understanding system. You are given OCR text, bounding
boxes, detected regions, checkboxes and candidate relationships from a
physical form. The form may belong to ANY domain.

YOUR ROLE
=========
Decide WHAT things mean and HOW they relate. Computer vision has already
decided WHERE things are. You must NEVER invent or reproduce coordinates.
You must ONLY reference the IDs that are provided to you.

INPUT DATA
==========

Document size: {doc_rep["image_width"]} x {doc_rep["image_height"]} pixels.
Coordinates below are normalized to [0,1]. (0,0) is the top-left, (1,1) is
the bottom-right.

OCR ELEMENTS (id "text" bbox=(x1,y1,x2,y2) conf=confidence)
-----------------------------------------------------------
{elements_repr}

REGIONS (bordered boxes detected by computer vision)
---------------------------------------------------
{regions_repr}

CHECKBOXES (cv-determined state)
-------------------------------
{checkboxes_repr}

CANDIDATE LABEL->VALUE RELATIONSHIPS (geometric scoring)
-------------------------------------------------------
{candidates_repr}

TASK
====
1. SECTIONS
   Group elements and regions into logical sections. Each section must have
   a meaningful name. Assign every meaningful element to exactly one section
   by listing its ID in "element_ids". A section may use "region_ids" for
   bordered regions it covers (optional).

   Use "region_ids" whenever a detected region clearly belongs to a section
   (e.g. a bordered box, a table box, a field group). This gives the section
   an exact bounding box. Do not force a region into a section if it does
   not belong there.

2. RELATIONSHIPS
   Decide relationships between elements using text content, spatial layout
   and the candidate list. Supported types:

   - "label_value":
       label_id: the label element (single id).
       value_ids: one or more elements that form the value.
     A value is usually immediately to the right of its label on the same
     row, or directly below it. Use the geometric candidates as a guide but
     do not blindly trust them.
     If the label element ALREADY contains the answer (e.g. "AGE: 21"),
     still create a label_value relationship, but you may leave value_ids
     empty -- the system will split the text.
     Do NOT confuse instructions, headings, or unrelated text with values.
     IMPORTANT: value_ids must contain ONLY the text that is the actual
     value. Never include neighboring option labels, instructions, or other
     fields' content.

   - "question":
       question_id: the question text element (single id).
       answer_id: the element containing the chosen answer/option (or null
                  if no answer is visible).
       option_ids: the elements that are the possible options for this
                   question (e.g. "YES ...", "NO ...").
     Use this when a label is a QUESTION whose possible answers are nearby
     options (often paired with checkboxes). Do NOT use "label_value" for
     such questions, and never stuff the options into a label_value value.

   - "checkbox_option":
       checkbox_id: a checkbox element.
       label_id: the text element that is the label/option for this checkbox
                 (usually immediately to the right of the checkbox).
       checked: the selected state (true/false). If the cv state is
                "uncertain", decide from the ink or leave as the cv value.
     Never attach instructional text to a checkbox label.

   - "table":
       region_id: a region that is a table.
       has_header: true if the first row is a header row.

3. UNASSIGNED TEXT
   If some element is truly irrelevant (watermark, page number, logo text),
   list its ID in "unassigned_text_ids". Prefer assigning everything.

RULES
=====
- Only use IDs that exist in the input. Never invent IDs.
- Never output coordinates, bounding boxes, or pixel values.
- Never invent text that was not observed.
- A heading is a section title, not a field label, unless the structure
  indicates otherwise.
- Do not treat every line or region boundary as a semantic section.
- Preserve table structure: elements inside a table region stay in the table.
- Preserve handwriting as observed; do not silently correct it.
- If a field is present but empty, still create the label_value relationship
  with empty value_ids.

OUTPUT FORMAT (JSON only)
=========================
{{
  "document_type": "short description or null",
  "sections": [
    {{
      "name": "Section name",
      "region_ids": ["r000"],
      "element_ids": ["t000", "t001"]
    }}
  ],
  "relationships": [
    {{
      "type": "label_value",
      "label_id": "t000",
      "value_ids": ["t001"]
    }},
    {{
      "type": "checkbox_option",
      "checkbox_id": "c000",
      "label_id": "t002",
      "checked": true
    }},
    {{
      "type": "question",
      "question_id": "t004",
      "answer_id": "t005",
      "option_ids": ["t005", "t006"]
    }},
    {{
      "type": "table",
      "region_id": "r001",
      "has_header": true
    }}
  ],
  "unassigned_text_ids": []
}}
"""


# ---------------------------------------------------------------------------
# Stage 8 - Deterministic resolver
#
# Maps LLM-returned IDs back onto the actual geometry produced by OCR/CV.
# ---------------------------------------------------------------------------

def _infer_value_type(text):
    stripped = text.strip()

    if re.fullmatch(r"\d{1,4}[-/]\d{1,2}[-/]\d{2,4}", stripped):
        return "date"

    if re.fullmatch(r"\d{1,2}:\d{2}(\s?(AM|PM))?", stripped, re.IGNORECASE):
        return "time"

    if re.fullmatch(r"[+-]?\d[\d,]*([.,]\d+)?%?", stripped):
        return "number"

    if re.fullmatch(
        r"[+-]?\d[\d,]*([.,]\d+)?\s*(kg|cm|m|km|mi|mph|ml|l|hr|min|g|mg|USD|\$|°C|F)",
        stripped,
        re.IGNORECASE,
    ):
        return "measurement"

    return "text"


def _avg_confidence(confidences):
    values = [c for c in confidences if c is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _element_map(doc_rep):
    return {e["id"]: e for e in doc_rep["elements"]}


def _region_map(doc_rep):
    return {r["id"]: r for r in doc_rep["regions"]}


def _checkbox_map(doc_rep):
    return {c["id"]: c for c in doc_rep["checkboxes"]}


def _elements_in_bbox(doc_rep, bbox):
    return [
        e for e in doc_rep["elements"]
        if _point_in_bbox(e["center"], bbox)
    ]


def _resolve_sections(llm_data, doc_rep):
    elements = _element_map(doc_rep)
    regions = _region_map(doc_rep)

    sections = []

    raw_sections = llm_data.get("sections") or []

    for section in raw_sections:

        name = str(section.get("name") or "").strip() or "Section"

        region_bboxes = []
        for rid in section.get("region_ids") or []:
            region = regions.get(rid)
            if region:
                region_bboxes.append(region["bbox"])

        section_element_ids = []
        for eid in section.get("element_ids") or []:
            if eid in elements and eid not in section_element_ids:
                section_element_ids.append(eid)

        # Elements whose center lies inside one of the section's regions
        # automatically belong to the section.
        for region in regions.values():
            if region["bbox"] in region_bboxes:
                for e in _elements_in_bbox(doc_rep, region["bbox"]):
                    if e["id"] not in section_element_ids:
                        section_element_ids.append(e["id"])

        sections.append({
            "name": name,
            "region_ids": list(section.get("region_ids") or []),
            "element_ids": section_element_ids,
            "region_bboxes": region_bboxes,
        })

    return sections


def _find_section_for_element(sections, element_id, elements):
    element = elements.get(element_id)
    if not element:
        return 0

    for index, section in enumerate(sections):
        for bbox in section["region_bboxes"]:
            if _point_in_bbox(element["center"], bbox):
                return index

    # Fall back to a section that explicitly lists the element.
    for index, section in enumerate(sections):
        if element_id in section["element_ids"]:
            return index

    return -1


def _build_table(doc_rep, region_id, has_header):
    region = _region_map(doc_rep).get(region_id)
    if not region:
        return None

    items = _elements_in_bbox(doc_rep, region["bbox"])
    if not items:
        return None

    rows = group_into_rows(items, y_tolerance=12)

    # Cluster columns by x-center gaps.
    row_cells = []

    for row in rows:
        cells = []
        current_cell = [row["items"][0]]

        for item in row["items"][1:]:
            prev = current_cell[-1]
            gap = item["center"][0] - (
                prev["bbox"][2] + prev["bbox"][0]
            ) / 2
            if gap > 0 and gap > prev["width"] * 0.5:
                cells.append(current_cell)
                current_cell = [item]
            else:
                current_cell.append(item)

        cells.append(current_cell)
        row_cells.append(cells)

    width = doc_rep["image_width"]
    height = doc_rep["image_height"]

    normalized_cells = []

    for cells in row_cells:
        normalized_row = []
        for cell in cells:
            bbox = _union_bbox([e["bbox"] for e in cell])
            normalized_row.append({
                "value": " ".join(e["text"] for e in cell).strip(),
                "element_ids": [e["id"] for e in cell],
                "bbox": _norm(bbox, width, height),
                "bbox_pixels": bbox,
                "confidence": _avg_confidence([e["confidence"] for e in cell]),
            })
        normalized_cells.append(normalized_row)

    if not normalized_cells:
        return None

    if has_header and len(normalized_cells) > 1:
        headers = [cell["value"] for cell in normalized_cells[0]]
        rows_data = normalized_cells[1:]
    else:
        headers = []
        rows_data = normalized_cells

    return {
        "title": None,
        "headers": headers,
        "rows": rows_data,
        "region_id": region_id,
        "bbox": _norm(region["bbox"], width, height),
        "bbox_pixels": region["bbox"],
        "confidence": None,
        "uncertain": False,
        "reason": None,
    }


def _split_inline_label_value(label_element):
    text = label_element["text"]
    match = re.split(r"\s*[:\uFF1A]\s*", text, maxsplit=1)
    if len(match) == 2 and match[1].strip():
        return match[0].strip(), match[1].strip()
    return text, None


def resolve_structure(llm_data, doc_rep, candidates):
    elements = _element_map(doc_rep)
    checkboxes = _checkbox_map(doc_rep)
    regions = _region_map(doc_rep)
    width = doc_rep["image_width"]
    height = doc_rep["image_height"]

    sections = _resolve_sections(llm_data, doc_rep)

    # Guard: if the LLM produced no sections, create a single catch-all.
    if not sections:
        sections = [{
            "name": "Document",
            "region_ids": [],
            "element_ids": [e["id"] for e in doc_rep["elements"]],
            "region_bboxes": [],
        }]

    # Section metadata (bbox, consumed elements).
    section_outputs = []
    section_consumed = [[] for _ in sections]

    for index, section in enumerate(sections):
        bboxes = list(section["region_bboxes"])
        for eid in section["element_ids"]:
            element = elements.get(eid)
            if element:
                bboxes.append(element["bbox"])

        if bboxes:
            bbox_pixels = _union_bbox(bboxes)
        else:
            bbox_pixels = None

        section_outputs.append({
            "name": section["name"],
            "region_ids": section["region_ids"],
            "element_ids": list(section["element_ids"]),
            "bbox_pixels": bbox_pixels,
            "bbox": _norm(bbox_pixels, width, height) if bbox_pixels else None,
            "fields": [],
            "questions": [],
            "checkboxes": [],
            "tables": [],
            "other_elements": [],
            "uncertain": False,
            "reason": None,
        })

    def section_for_element(eid, create=True):
        index = _find_section_for_element(sections, eid, elements)
        if index >= 0:
            return index
        return None

    relationships = llm_data.get("relationships") or []
    review_reasons = []

    # ----- label_value relationships -> fields -----
    for relation in relationships:

        rtype = relation.get("type")

        if rtype == "label_value":
            label_id = relation.get("label_id")
            value_ids = relation.get("value_ids") or []

            if label_id not in elements:
                review_reasons.append(
                    f"LLM referenced unknown label id {label_id!r}"
                )
                continue

            label_element = elements[label_id]
            valid_value_ids = [
                vid for vid in value_ids if vid in elements
            ]
            invalid_value_ids = [
                vid for vid in value_ids if vid not in elements
            ]
            if invalid_value_ids:
                review_reasons.append(
                    f"LLM referenced unknown value ids {invalid_value_ids}"
                )

            value_elements = [elements[vid] for vid in valid_value_ids]
            value_elements.sort(
                key=lambda e: (e["center"][1], e["center"][0])
            )

            label_text = label_element["text"]

            if value_elements:
                value_text = " ".join(
                    e["text"] for e in value_elements
                ).strip()
                value_bbox = _union_bbox([e["bbox"] for e in value_elements])
            else:
                value_text = None
                value_bbox = label_element["bbox"]

            inline_value = None
            if not value_elements:
                label_text, inline_value = _split_inline_label_value(
                    label_element
                )
                if inline_value is not None:
                    value_text = inline_value
                    value_bbox = label_element["bbox"]
                    valid_value_ids = [label_element["id"]]

            confidence = _avg_confidence(
                [label_element["confidence"]]
                + [e["confidence"] for e in value_elements]
            )

            index = section_for_element(label_id)
            field = {
                "label": label_text,
                "value": value_text,
                "value_type": _infer_value_type(value_text) if value_text else None,
                "confidence": confidence,
                "uncertain": value_text is None,
                "reason": "value not found" if value_text is None else None,
                "label_element_ids": [label_element["id"]],
                "value_element_ids": valid_value_ids,
                "label_bbox": _norm(
                    label_element["bbox"], width, height
                ),
                "label_bbox_pixels": label_element["bbox"],
                "value_bbox": _norm(value_bbox, width, height),
                "value_bbox_pixels": value_bbox,
            }

            if index is None:
                section_outputs.append({
                    "name": "Unassigned",
                    "region_ids": [],
                    "element_ids": [],
                    "bbox_pixels": _union_bbox(
                        [label_element["bbox"]] + [e["bbox"] for e in value_elements]
                    ) if value_elements else label_element["bbox"],
                    "bbox": None,
                    "fields": [],
                    "questions": [],
                    "checkboxes": [],
                    "tables": [],
                    "other_elements": [],
                    "uncertain": True,
                    "reason": "no section assigned",
                })
                index = len(section_outputs) - 1
                section_consumed.append([])

            section_outputs[index]["fields"].append(field)
            section_consumed[index].extend([label_element["id"]] + valid_value_ids)

        elif rtype == "question":
            question_id = relation.get("question_id")
            answer_id = relation.get("answer_id")
            option_ids = relation.get("option_ids") or []

            if question_id not in elements:
                review_reasons.append(
                    f"LLM referenced unknown question id {question_id!r}"
                )
                continue

            question_element = elements[question_id]
            answer_element = elements.get(answer_id)
            option_elements = [
                elements[oid] for oid in option_ids if oid in elements
            ]

            options = []
            for option in option_elements:
                options.append({
                    "label": option["text"],
                    "selected": option["id"] == answer_id,
                    "confidence": option["confidence"],
                    "element_id": option["id"],
                    "bbox": _norm(option["bbox"], width, height),
                    "bbox_pixels": option["bbox"],
                })

            index = section_for_element(question_id)

            entry = {
                "question": question_element["text"],
                "answer": (
                    answer_element["text"] if answer_element else None
                ),
                "options": options,
                "confidence": _avg_confidence(
                    [question_element["confidence"]]
                    + [e["confidence"] for e in option_elements]
                ),
                "uncertain": answer_element is None,
                "reason": (
                    None if answer_element
                    else "no visible selected option"
                ),
                "question_element_ids": [question_element["id"]],
                "answer_element_ids": (
                    [answer_element["id"]] if answer_element else []
                ),
            }

            if index is None:
                index = 0

            section_outputs[index]["questions"].append(entry)
            section_consumed[index].append(question_element["id"])
            section_consumed[index].extend([o["id"] for o in option_elements])
            if answer_element:
                section_consumed[index].append(answer_element["id"])

        elif rtype == "checkbox_option":
            checkbox_id = relation.get("checkbox_id")
            label_id = relation.get("label_id")

            if checkbox_id not in checkboxes:
                review_reasons.append(
                    f"LLM referenced unknown checkbox id {checkbox_id!r}"
                )
                continue

            checkbox = checkboxes[checkbox_id]

            if label_id and label_id in elements:
                label_element = elements[label_id]
                label_text = label_element["text"]
                label_index = section_for_element(label_id)
            else:
                label_element = None
                label_text = None
                label_index = None

            if label_index is None:
                # Associate checkbox to a section by its own position.
                label_index = None
                for index, section in enumerate(sections):
                    for bbox in section["region_bboxes"]:
                        if _point_in_bbox(checkbox["center"], bbox):
                            label_index = index
                            break
                    if label_index is not None:
                        break

            checked = relation.get("checked")
            if checked is None:
                checked = checkbox["state"] == "checked"
            elif checkbox["state"] == "uncertain":
                pass  # trust the LLM decision when cv is uncertain

            if label_element:
                section_consumed[label_index].append(label_element["id"])

            entry = {
                "label": label_text,
                "checked": bool(checked),
                "instruction": None,
                "confidence": checkbox["confidence"],
                "uncertain": checkbox["state"] == "uncertain",
                "reason": None,
                "checkbox_id": checkbox_id,
                "label_element_ids": (
                    [label_element["id"]] if label_element else []
                ),
                "bbox": _norm(checkbox["bbox"], width, height),
                "bbox_pixels": checkbox["bbox"],
            }

            if label_index is not None and label_index < len(section_outputs):
                section_outputs[label_index]["checkboxes"].append(entry)
            else:
                section_outputs[0]["checkboxes"].append(entry)

        elif rtype == "table":
            table = _build_table(
                doc_rep,
                relation.get("region_id"),
                bool(relation.get("has_header", True)),
            )

            if table is None:
                review_reasons.append(
                    f"Could not build table for region "
                    f"{relation.get('region_id')!r}"
                )
                continue

            region_id = relation.get("region_id")
            index = None

            for section_index, section in enumerate(sections):
                if region_id in section["region_ids"]:
                    index = section_index
                    break

            if index is None:
                for section_index, section in enumerate(sections):
                    region = regions.get(region_id)
                    if region and any(
                        _point_in_bbox(
                            region["center"] if "center" in region
                            else [(region["bbox"][0] + region["bbox"][2]) / 2,
                                  (region["bbox"][1] + region["bbox"][3]) / 2],
                            bbox,
                        )
                        for bbox in section["region_bboxes"]
                    ):
                        index = section_index
                        break

            if index is None:
                index = 0

            for row in table["rows"]:
                for cell in row:
                    section_consumed[index].extend(cell["element_ids"])

            section_outputs[index]["tables"].append(table)

    # ----- other elements referenced by sections but not in fields -----
    for index, section in enumerate(sections):
        other = []
        for eid in section["element_ids"]:
            element = elements.get(eid)
            if element and eid not in section_consumed[index]:
                other.append({
                    "text": element["text"],
                    "element_id": eid,
                    "bbox": _norm(element["bbox"], width, height),
                    "bbox_pixels": element["bbox"],
                })
        section_outputs[index]["other_elements"] = other

    # ----- unassigned text -----
    referenced = set()
    for index in range(len(section_outputs)):
        referenced.update(section_consumed[index])

    unassigned = []
    for element in doc_rep["elements"]:
        if element["id"] in referenced:
            continue
        if any(
            element["id"] in section["element_ids"]
            for section in sections
        ):
            continue
        unassigned.append(element)

    llm_unassigned = llm_data.get("unassigned_text_ids") or []
    for eid in llm_unassigned:
        element = elements.get(eid)
        if element and element["id"] not in referenced:
            if element not in unassigned:
                unassigned.append(element)

    unassigned_list = [
        {
            "text": e["text"],
            "element_id": e["id"],
            "bbox": _norm(e["bbox"], width, height),
            "bbox_pixels": e["bbox"],
            "confidence": e["confidence"],
        }
        for e in unassigned
    ]

    result = {
        "document_type": llm_data.get("document_type"),
        "sections": section_outputs,
        "unassigned_text": unassigned_list,
    }

    return result, review_reasons


# ---------------------------------------------------------------------------
# Stage 9 - Validation
# ---------------------------------------------------------------------------

def validate_extraction(result, doc_rep, review_reasons):
    review_reasons = list(review_reasons)

    confidences = []

    for section in result["sections"]:
        for field in section["fields"]:
            if field["confidence"] is not None:
                confidences.append(field["confidence"])
            if field["uncertain"]:
                review_reasons.append(
                    f"Field {field['label']!r} has no reliable value"
                )

        for checkbox in section["checkboxes"]:
            if checkbox["confidence"] is not None:
                confidences.append(checkbox["confidence"])

    for element in doc_rep["elements"]:
        if element["confidence"] is not None and element["confidence"] < 0.5:
            review_reasons.append(
                f"Low OCR confidence for {element['text']!r}"
            )

    if result["unassigned_text"]:
        review_reasons.append(
            f"{len(result['unassigned_text'])} text element(s) unassigned"
        )

    if confidences:
        document_confidence = round(
            sum(confidences) / len(confidences), 4
        )
    else:
        document_confidence = None

    result["document_confidence"] = document_confidence
    result["requires_human_review"] = bool(review_reasons)
    result["review_reasons"] = review_reasons

    return result


# ---------------------------------------------------------------------------
# Convenience: full pipeline
# ---------------------------------------------------------------------------

def analyze_document(image_or_path, image_path=None):
    """
    End-to-end: preprocess -> OCR -> layout -> candidates.
    Returns (doc_rep, candidates). The LLM stage lives in main.py.
    """
    info = preprocess_document(image_or_path)

    doc_rep = build_document_representation(
        info["image"],
        image_path=image_path,
    )
    doc_rep["preprocessing"] = {
        "orientation_corrected": info["orientation_corrected"],
        "deskew_angle": info["deskew_angle"],
    }

    candidates = generate_field_candidates(doc_rep)

    return doc_rep, candidates

def _interpolate_sub_bbox(full_text: str, sub_text: str, bbox: list) -> list:
    """
    Calculates the proportional bounding box for a sub-string based on character counts.
    """
    start_idx = full_text.find(sub_text)
    
    # Fallback: if the LLM hallucinated or changed the text slightly, return the original box
    if start_idx == -1:
        return bbox 

    end_idx = start_idx + len(sub_text)
    total_chars = len(full_text)
    
    if total_chars == 0:
        return bbox

    x1, y1, x2, y2 = bbox
    total_width = x2 - x1

    new_x1 = x1 + (total_width * (start_idx / total_chars))
    new_x2 = x1 + (total_width * (end_idx / total_chars))

    return [round(new_x1, 2), y1, round(new_x2, 2), y2]

import json

def run_pass_1_llm_splitter(compound_texts: list) -> dict:
    """
    Sends compound strings to a fast LLM (e.g., Groq Llama 3 8B) to split them.
    """
    if not compound_texts:
        return {}

    # Format the input list as a JSON string so the LLM parses it easily
    input_json_str = json.dumps(compound_texts, indent=2)

    prompt = f"""
    I am providing a list of strings detected by an OCR engine. 
    Some strings contain multiple form fields merged together (e.g., "City: NY State: NY").
    Split these compound strings into individual logical fields.
    If a string is just one field, return it as a single item in the array.
    
    Return ONLY a valid JSON object where the key is the original string, and the value is an array of the split strings.
    Do not alter the spelling or casing of the text.

    Input Strings:
    {input_json_str}
    """
    
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise RuntimeError(
            "GROQ_API_KEY environment variable is not set."
            )

        groq_client = Groq(
            api_key=groq_api_key
        )

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an OCR data cleaning utility. "
                        "You output strict JSON objects without any markdown formatting or conversational text."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            response_format={
                "type": "json_object"
            }
        )
        
        content = response.choices[0].message.content
        parsed_dict = json.loads(content)
        
        return parsed_dict

    except json.JSONDecodeError as e:
        print(f"JSON Parsing Error in Pass 1: {e}")
        return {}
    except Exception as e:
        print(f"API Error in Pass 1: {e}")
        return {}


def normalize_ocr_elements(elements: list) -> list:
    """
    Orchestrates the splitting of compound elements and reconstructs 
    the geometry so the rest of the pipeline functions normally.
    """
    compound_candidates = []
    
    # 1. Filter elements that might be compound to save LLM tokens
    # (e.g., look for multiple colons or long strings)
    for el in elements:
        text = el.get("text", "")
        if text.count(":") > 1 or (":" in text and len(text.split()) > 4):
            compound_candidates.append(text)

    # If nothing looks compound, skip the LLM call entirely
    if not compound_candidates:
        return elements

    # 2. Get the semantic splits from the fast LLM
    try:
        split_map = run_pass_1_llm_splitter(compound_candidates)
    except Exception as e:
        print(f"Pass 1 Splitter failed: {e}. Falling back to raw OCR.")
        return elements

    normalized_elements = []
    
    # 3. Rebuild the elements array with new visual coordinates
    for el in elements:
        text = el.get("text", "")
        
        # If the LLM successfully split this specific text into multiple parts
        if text in split_map and isinstance(split_map[text], list) and len(split_map[text]) > 1:
            
            sub_texts = split_map[text]
            for i, sub_text in enumerate(sub_texts):
                new_bbox = _interpolate_sub_bbox(text, sub_text, el["bbox"])
                new_x1, y1, new_x2, y2 = new_bbox
                
                # Create synthetic elements that look identical to PaddleOCR output
                normalized_elements.append({
                    "id": f"{el['id']}_s{i}", # e.g., t005_s0, t005_s1
                    "type": "text",
                    "text": sub_text.strip(),
                    "bbox": new_bbox,
                    "center": [
                        round((new_x1 + new_x2) / 2, 2),
                        round((y1 + y2) / 2, 2)
                    ],
                    "width": round(new_x2 - new_x1, 2),
                    "height": round(y2 - y1, 2),
                    "confidence": el.get("confidence")
                })
        else:
            # Leave non-compound elements exactly as they are
            normalized_elements.append(el)

    return normalized_elements