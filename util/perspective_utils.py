"""
Document-boundary detection and perspective correction.

This module is intentionally independent from OCR so that either a scanned or
photographed page can be normalized before the text/geometric layers run.

Coordinate contract
-------------------
After a perspective correction the returned image defines a NEW coordinate
system. All OCR / CV / field / table coordinates produced downstream are
expressed in this corrected space. The transformation metadata is stored in
the preprocessing section of the document representation, and
`original_to_processed` / `processed_to_original` helpers are provided to map
between the two systems explicitly.
"""

import numpy as np
import cv2

from util.geometry_utils import (
    _cv2_perspective_transform_impl as _cv2_perspective_transform,
)


def _order_corners(corners):
    """
    Order 4 corners consistently as:
        [top-left, top-right, bottom-right, bottom-left]

    Strategy: the top-left and bottom-right corners lie on one diagonal, and
    the other two on the other. We classify by total (x + y)-order.
    """
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    if corners.shape != (4, 2):
        raise ValueError("corner detection must return exactly 4 points")

    s = corners.sum(axis=1)
    diff = np.diff(corners, axis=1).ravel()

    top_left = corners[np.argmin(s)]
    bottom_right = corners[np.argmax(s)]
    top_right = corners[np.argmin(diff)]
    bottom_left = corners[np.argmax(diff)]

    return [top_left, top_right, bottom_right, bottom_left]


def _quadrilateral_corners(mask):
    """
    Return the 4 ordered corners of the largest convex quadrilateral found in
    `mask`, or None if no suitable quadrilateral exists.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    # Largest contour by area.
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area <= 0:
        return None

    # In case the contour is the whole (already ~rectangular) frame.
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) >= 4:
        polygon = approx.reshape(-1, 2)
        if len(polygon) > 4:
            hull = cv2.convexHull(polygon)
            if hull.shape[0] >= 4:
                polygon = hull.reshape(-1, 2)
            polygon = cv2.approxPolyDP(
                polygon, 0.02 * cv2.arcLength(polygon, True), True
            ).reshape(-1, 2)

        if len(polygon) == 4:
            return _order_corners(polygon)

        # More than 4 sides: keep the 4 extreme corners of the hull.
        ext = np.array(
            [
                polygon[np.argmin(polygon[:, 0] + polygon[:, 1])],
                polygon[np.argmax(polygon[:, 0] - polygon[:, 1])],
                polygon[np.argmax(polygon[:, 0] + polygon[:, 1])],
                polygon[np.argmin(polygon[:, 0] - polygon[:, 1])],
            ]
        )
        return _order_corners(_order_corners(ext) if len(ext) == 4 else ext)

    return None


def _mask_area_ratio(mask):
    total = mask.shape[0] * mask.shape[1]
    if total == 0:
        return 0.0
    return float(cv2.countNonZero(np.asarray(mask > 0, dtype=np.uint8))) / total


def detect_document_corners(gray):
    """
    Detect the 4 page corners of a photographed / skewed document page.

    Returns:
        (corners, confidence)
            corners: 4x2 array [tl, tr, br, bl] in `gray` coordinates or None.
            confidence: float in [0, 1] describing how confident we are that
                the returned quadrilateral really is the page boundary.

    The method combines a thresholded background/foreground separation with a
    contour-based quadrilateral search. It tolerates shadows and imperfect
    corners by falling back to the 4 extreme hull points.
    """
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarize twice (adaptive + Otsu) and combine so both clean scans and
    # noisy photographs have a chance.
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 15
    )

    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    combined = cv2.bitwise_and(adaptive, otsu)

    # Fill interior holes so a closed rectangle yields a solid blob.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    corners = _quadrilateral_corners(closed)

    if corners is None:
        # Second attempt: grab the dominant cross-lines and intersect them.
        corners = _corner_lines_fallback(gray)

    if corners is None:
        return None, 0.0

    corners = np.asarray(corners, dtype=np.float32)

    # Confidence: how much of the mask area the quad covers + shape regularness.
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)

    area_ratio = _mask_area_ratio(mask)
    quad_area = cv2.contourArea(corners)
    frame_area = h * w

    # Area of the quad relative to the whole frame: a page usually occupies
    # a large majority of the image.
    size_factor = min(1.0, quad_area / max(1.0, frame_area * 0.5))

    # Rectangularity: compare quad area to its axis-aligned bounding box.
    x1, y1 = np.min(corners, axis=0)
    x2, y2 = np.max(corners, axis=0)
    bbox_area = max(1.0, (x2 - x1) * (y2 - y1))
    rectangularity = min(1.0, quad_area / bbox_area)

    confidence = 0.5 * size_factor + 0.3 * rectangularity + 0.2 * min(
        1.0, area_ratio
    )

    # A page boundary that covers less than 55% and is heavily non-rectangular
    # is likely noise.
    if size_factor < 0.5 or rectangularity < 0.75:
        return None, 0.0

    return corners, round(float(min(1.0, confidence)), 4)


def _corner_lines_fallback(gray):
    """
    Last-resort corner detection: find the top-most / bottom-most / left-most
    / right-most long straight lines and take their intersections.
    """
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=int(min(w, h) * 0.15),
        minLineLength=int(min(w, h) * 0.5),
        maxLineGap=int(min(w, h) * 0.12),
    )

    if lines is None or len(lines) < 4:
        return None

    segments = [l[0] for l in lines]

    def slope(seg):
        x1, y1, x2, y2 = seg
        dx = x2 - x1
        if abs(dx) < 1e-6:
            return float("inf") if y2 >= y1 else float("-inf")
        return (y2 - y1) / dx

    horizontals = [
        seg for seg in segments if abs(slope(seg)) < 0.3 and abs(seg[1] - seg[3]) < min(w, h) * 0.05
    ]
    verticals = [
        seg for seg in segments if abs(slope(seg)) > 2.0
    ]

    if not horizontals or not verticals:
        return None

    top = min(horizontals, key=lambda s: (s[1] + s[3]) / 2)
    bottom = max(horizontals, key=lambda s: (s[1] + s[3]) / 2)
    left = min(verticals, key=lambda s: min(s[0], s[2]))
    right = max(verticals, key=lambda s: max(s[0], s[2]))

    corners = [
        _line_intersection(top, left),
        _line_intersection(top, right),
        _line_intersection(bottom, right),
        _line_intersection(bottom, left),
    ]

    if any(c is None for c in corners):
        return None

    return _order_corners(corners)


def _line_intersection(seg_a, seg_b):
    """Intersection of two infinite lines defined by segments, or None."""
    x1, y1, x2, y2 = seg_a
    x3, y3, x4, y4 = seg_b

    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-9:
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
    return [px, py]


def correct_perspective(image):
    """
    Apply perspective correction when the page corners can be detected with
    sufficient confidence.

    Returns:
        (corrected_image, info dict)

    info:
        {
            "perspective_correction": {
                "applied": bool,
                "confidence": float,
                "corners": [[x,y], ...] or None,   # tl, tr, br, bl (original)
                "target_size": [w, h],
                "transform_matrix": 3x3 list,
            }
        }

    When confidence is low the original image is returned and `applied` is
    False.
    """
    import numpy as np

    import util.config as config

    h, w = image.shape[:2]

    info = {
        "perspective_correction": {
            "applied": False,
            "confidence": 0.0,
            "corners": None,
            "target_size": [w, h],
            "transform_matrix": None,
            "reason": "disabled or not applicable",
        }
    }

    if not config.PERSPECTIVE_CORRECTION_ENABLED:
        info["perspective_correction"]["reason"] = "disabled by configuration"
        return image, info

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    try:
        corners, confidence = detect_document_corners(gray)
    except Exception as e:
        print(f"[PREPROCESS] Perspective correction failed: {e}")
        info["perspective_correction"]["reason"] = "corner detection error"
        return image, info

    if corners is None or confidence < config.PERSPECTIVE_MIN_CONFIDENCE:
        info["perspective_correction"]["reason"] = (
            "low-confidence or no corners"
            + (f" (confidence={confidence})" if corners is not None else "")
        )
        return image, info

    tl, tr, br, bl = [np.asarray(pts, dtype=np.float32) for pts in corners]

    def _dist(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    width_a = _dist(br, bl)
    width_b = _dist(tr, tl)
    height_a = _dist(tr, br)
    height_b = _dist(tl, bl)

    max_width = max(int(round(width_a)), int(round(width_b)))
    max_height = max(int(round(height_a)), int(round(height_b)))

    if max_width <= 0 or max_height <= 0:
        info["perspective_correction"]["reason"] = "degenerate corner geometry"
        return image, info

    target = np.float32(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ]
    )

    src = np.float32([tl, tr, br, bl])
    matrix = cv2.getPerspectiveTransform(src, target)

    corrected = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    info["perspective_correction"] = {
        "applied": True,
        "confidence": confidence,
        "corners": [c.tolist() for c in corners],
        "target_size": [max_width, max_height],
        "transform_matrix": matrix.tolist(),
        "reason": "detected document boundary",
    }

    print(f"[PREPROCESS] Perspective correction: applied (conf={confidence})")

    return corrected, info


def original_to_processed(bbox, preprocessing, image_size=None):
    """
    Map a [x1, y1, x2, y2] bbox from original-image coordinates into the
    processed (corrected) coordinate system.

    Returns the input bbox unchanged when no perspective transform was applied.
    """
    pc = preprocessing.get("perspective_correction") or {}
    matrix = pc.get("transform_matrix")

    if not pc.get("applied") or not matrix:
        return list(bbox)

    from util.geometry_utils import _transform_bbox

    matrix_np = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    target_size = pc.get("target_size")
    return _transform_bbox(bbox, matrix_np, target_size)


def processed_to_original(bbox, preprocessing, image_size=None):
    """
    Map a [x1, y1, x2, y2] bbox from processed coordinates back into the
    original-image coordinate system.
    """
    pc = preprocessing.get("perspective_correction") or {}
    matrix = pc.get("transform_matrix")

    if not pc.get("applied") or not matrix:
        return list(bbox)

    inv = np.linalg.inv(np.asarray(matrix, dtype=np.float64).reshape(3, 3))

    from util.geometry_utils import _transform_bbox

    return _transform_bbox(bbox, inv, image_size)