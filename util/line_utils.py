"""
Low-level line-segment detection and merging.

The existing `service/detection_service.py` performs morphological line
extraction but returns only bounding-box approximations. This module keeps a
strict separation:

    util           -> generic, dependency-free helpers
    service        -> documents decisions (thresholds, formatting)

These helpers return *segments* (real endpoints) so higher layers (table
detection, field detection) can reason about collinearity, intersections and
broken lines instead of guessing from a bounding rect.
"""

import cv2
import numpy as np

from util.image_utils import _load_gray, _threshold_gray


def detect_line_segments(
    image_or_gray,
    orientation="horizontal",
    min_length_ratio=0.10,
    max_thickness=30,
    kernel_scale=0.02,
    connect_gap_ratio=0.02,
):
    """
    Detect straight line segments of a given orientation.

    Args:
        image_or_gray: BGR/gray image or path.
        orientation: "horizontal" | "vertical".
        min_length_ratio: minimum segment length as a fraction of the
            corresponding image dimension.
        max_thickness: maximum perpendicular thickness of a detected segment.
        kernel_scale: morphological kernel length as a fraction of the image
            dimension (survival kernel for broken/faded lines).
        connect_gap_ratio: dilation used to bridge gaps in broken lines.

    Returns:
        list of segments:
            horizontal -> {"x1": float, "x2": float, "y": float,
                           "length": float}
            vertical   -> {"y1": float, "y2": float, "x": float,
                           "length": float}
    """
    gray = _load_gray(image_or_gray)
    h, w = gray.shape

    min_length = max(10, int(min_length_ratio * (w if orientation == "horizontal" else h)))

    if orientation == "horizontal":
        kernel_len = max(15, int(w * kernel_scale))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        connect = np.ones((2, max(15, int(w * connect_gap_ratio))), np.uint8)
    else:
        kernel_len = max(15, int(h * kernel_scale))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
        connect = np.ones((max(15, int(h * connect_gap_ratio)), 2), np.uint8)

    binary = _threshold_gray(gray)

    lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    lines_img = cv2.dilate(lines_img, connect, iterations=1)

    contours, _ = cv2.findContours(
        lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    segments = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)

        if orientation == "horizontal":
            if cw < min_length or ch > max_thickness:
                continue
            segments.append(
                {"x1": float(x), "x2": float(x + cw), "y": float(y + ch / 2), "length": float(cw)}
            )
        else:
            if ch < min_length or cw > max_thickness:
                continue
            segments.append(
                {"x1": float(x + cw / 2), "x2": float(x), "y1": float(y), "y2": float(y + ch), "x": float(x + cw / 2), "length": float(ch)}
            )

    return segments


def merge_collinear_horizontal(segments, y_tol=8, gap_tol_ratio=0.03, max_gap_px=80):
    """
    Merge collinear, overlapping or near-touching horizontal segments into
    longer ones.

    y_tol: maximum vertical separation (in px) for two segments to be merged.
    The merged segment keeps the widest horizontal extent.
    """
    if not segments:
        return []

    merged = []
    for seg in sorted(segments, key=lambda s: s["x1"]):
        placed = False
        for m in merged:
            if abs(m["y"] - seg["y"]) <= y_tol:
                gap = max(0.0, seg["x1"] - m["x2"])
                allowed_gap = max(max_gap_px, (m["length"] + seg["length"]) * gap_tol_ratio)
                if seg["x1"] <= m["x2"] + allowed_gap:
                    m["x1"] = min(m["x1"], seg["x1"])
                    m["x2"] = max(m["x2"], seg["x2"])
                    m["length"] = m["x2"] - m["x1"]
                    m["y"] = round((m["y"] + seg["y"]) / 2, 2)
                    placed = True
                    break
        if not placed:
            merged.append(
                {"x1": seg["x1"], "x2": seg["x2"], "y": seg["y"], "length": seg["length"]}
            )

    merged.sort(key=lambda s: s["y"])
    return merged


def merge_collinear_vertical(segments, x_tol=8, gap_tol_ratio=0.03, max_gap_px=80):
    """
    Merge collinear, overlapping or near-touching vertical segments.
    """
    if not segments:
        return []

    merged = []
    for seg in sorted(segments, key=lambda s: s["y1"]):
        placed = False
        for m in merged:
            if abs(m["x"] - seg["x"]) <= x_tol:
                gap = max(0.0, seg["y1"] - m["y2"])
                allowed_gap = max(max_gap_px, (m["length"] + seg["length"]) * gap_tol_ratio)
                if seg["y1"] <= m["y2"] + allowed_gap:
                    m["y1"] = min(m["y1"], seg["y1"])
                    m["y2"] = max(m["y2"], seg["y2"])
                    m["length"] = m["y2"] - m["y1"]
                    m["x"] = round((m["x"] + seg["x"]) / 2, 2)
                    placed = True
                    break
        if not placed:
            merged.append(
                {"y1": seg["y1"], "y2": seg["y2"], "x": seg["x"], "length": seg["length"]}
            )

    merged.sort(key=lambda s: s["x"])
    return merged


def segments_to_mask(segments, shape, orientation, thickness=1):
    """
    Draw line segments into a binary mask.

    Returns a uint8 mask of the same height/width as `shape`.
    """
    mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)
    if orientation == "horizontal":
        for seg in segments:
            y = int(round(seg["y"]))
            cv2.line(mask, (int(seg["x1"]), y), (int(seg["x2"]), y), 255, thickness)
    else:
        for seg in segments:
            x = int(round(seg["x"]))
            cv2.line(mask, (x, int(seg["y1"])), (x, int(seg["y2"])), 255, thickness)
    return mask


def segment_intersections(horizontals, verticals):
    """
    Compute intersections between horizontal and vertical segments.

    Returns a list of {"x": float, "y": float, "h": segment, "v": segment}.
    """
    intersections = []
    for hseg in horizontals:
        hy = hseg["y"]
        for vseg in verticals:
            vx = vseg["x"]
            if (
                hseg["x1"] - 2 <= vx <= hseg["x2"] + 2
                and vseg["y1"] - 2 <= hy <= vseg["y2"] + 2
            ):
                intersections.append({"x": vx, "y": hy, "h": hseg, "v": vseg})
    return intersections