import cv2
import numpy as np

from util.image_utils import (
    _load_gray,
    _threshold_gray
)

from util.geometry_utils import (
    _iou
)

from util.line_utils import (
    detect_line_segments,
    merge_collinear_horizontal,
    merge_collinear_vertical,
)

from util.checkbox_utils import (
    detect_checkboxes_visual,
    strip_leading_option_mark,
    checkbox_from_leading_mark,
    merge_checkbox_candidates,
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
    """
    Detect horizontal line segments.

    Returns:
        list of {"y": float, "x1": float, "x2": float, "length": float}
        sorted top-to-bottom. Merged across small vertical offsets so the
        caller sees one entry per visual line.
    """
    segments = detect_line_segments(
        image_or_path,
        orientation="horizontal",
        min_length_ratio=min_width_ratio,
    )
    merged = merge_collinear_horizontal(segments)

    lines = []
    for seg in merged:
        lines.append(
            {
                "y": round(seg["y"], 2),
                "x1": round(seg["x1"], 2),
                "x2": round(seg["x2"], 2),
                "length": round(seg["length"], 2),
            }
        )

    lines.sort(key=lambda x: x["y"])

    deduped = []
    for line in lines:
        if not deduped:
            deduped.append(line)
            continue
        previous = deduped[-1]
        if abs(line["y"] - previous["y"]) <= 6 and line["length"] == previous["length"]:
            continue
        deduped.append(line)

    return deduped


def detect_vertical_lines(image_or_path, min_height_ratio=0.15):
    """
    Detect vertical line segments.

    Returns:
        list of {"x": float, "y1": float, "y2": float, "length": float}
        sorted left-to-right.
    """
    segments = detect_line_segments(
        image_or_path,
        orientation="vertical",
        min_length_ratio=min_height_ratio,
    )
    merged = merge_collinear_vertical(segments)

    lines = []
    for seg in merged:
        lines.append(
            {
                "x": round(seg["x"], 2),
                "y1": round(seg["y1"], 2),
                "y2": round(seg["y2"], 2),
                "length": round(seg["length"], 2),
            }
        )

    lines.sort(key=lambda x: x["x"])
    return lines


def detect_checkboxes_ocr_anchored(image_or_path, raw_elements=None):
    """
    OCR-anchored checkbox detection.

    Combines two complementary signals:
      1. a conservative pixel detector (catches clear printed box rings and
         unambiguous marks; may under-detect on degraded scans), and
      2. the OCR leading-mark signal (when the engine glues an X/tick/underscore
         onto the front of an option label it is a reliable `checked` signal).

    Each candidate is associated with the OCR element it belongs to.
    """
    pixel = detect_checkboxes_visual(image_or_path, ocr_elements=raw_elements)
    marks = []
    for element in raw_elements or []:
        if element.get("type") != "text":
            continue
        c = checkbox_from_leading_mark(element, image=image_or_path)
        if c:
            marks.append(c)
    merged = merge_checkbox_candidates(pixel, marks)
    # Keep only candidates the system is confident are *selected*. The pixel
    # detector's unchecked/uncertain rings are unreliable on degraded scans and
    # only add noise downstream; the OCR-mark signal is the trusted indicator.
    kept = [c for c in merged if c.get("state") == "checked" or c.get("source") == "ocr_mark"]
    kept.sort(key=lambda c: (c["center"][1], c["center"][0]))
    for i, c in enumerate(kept):
        c["id"] = f"c{i:03d}"
    return kept


def detect_checkboxes_with_marks(image_or_path, raw_elements=None):
    """
    Combine plain box detection with the OCR-anchored detector, deduped, and
    enrich each with the leading option mark (if any) so callers can decide
    whether a label's leading glyph is a real selection mark.
    """
    plain = {cb["bbox"][0]: cb for cb in detect_checkboxes(image_or_path)}
    anchored = detect_checkboxes_ocr_anchored(image_or_path, raw_elements)

    merged = []
    used = set()
    for cb in anchored:
        merged.append(cb)
        used.add((cb["bbox"][0], cb["bbox"][1]))

    for cb in plain.values():
        if (cb["bbox"][0], cb["bbox"][1]) in used:
            continue
        cb.setdefault("element_id", None)
        cb.setdefault("word", None)
        cb.setdefault("associated_text", None)
        merged.append(cb)

    merged.sort(key=lambda c: (c["center"][1], c["center"][0]))
    for i, c in enumerate(merged):
        c["id"] = f"c{i:03d}"
    return merged
