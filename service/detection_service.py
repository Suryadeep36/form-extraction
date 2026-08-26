import cv2
import numpy as np

from util.image_utils import (
    _load_gray,
    _threshold_gray
)

from util.geometry_utils import (
    _iou
)

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
        table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    regions = []
    min_area = w * h * min_area_ratio

    for contour in contours:
        x, y, rw, rh = cv2.boundingRect(contour)
        area = rw * rh

        if area < min_area:
            continue

        if rw < w * 0.12:  # Must be at least 12% of image width
            continue

        if rh < 25:
            continue

        regions.append(
            {
                "bbox": [x, y, x + rw, y + rh],
                "x": x,
                "y": y,
                "width": rw,
                "height": rh,
                "area": area,
            }
        )

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

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

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

        raw.append(
            {
                "bbox": [x, y, x + w, y + h],
                "width": w,
                "height": h,
                "interior_dark": interior_dark,
            }
        )

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

        checkboxes.append(
            {
                "id": f"c{i:03d}",
                "bbox": [x1, y1, x2, y2],
                "center": [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)],
                "width": box["width"],
                "height": box["height"],
                "state": state,
                "confidence": round(confidence, 3),
            }
        )

    return checkboxes

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

    contours, _ = cv2.findContours(
        horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)

        # 4. Enforce the strict 20% length rule HERE, on the reconstructed contour
        if width < w * min_width_ratio:
            continue

        # 5. Increased height tolerance for slight document skews
        if height > 40:
            continue

        lines.append({"y": y + height // 2, "x1": x, "x2": x + width, "length": width})

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

        lines.append({"x": x + width // 2, "y1": y, "y2": y + height, "length": height})

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
