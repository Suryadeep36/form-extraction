def _bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def _bbox_intersection(bbox_a, bbox_b):
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _horizontal_overlap(bbox_a, bbox_b):
    ax1, _, ax2, _ = bbox_a
    bx1, _, bx2, _ = bbox_b
    ox1 = max(ax1, bx1)
    ox2 = min(ax2, bx2)
    return max(0.0, ox2 - ox1)


def _vertical_overlap(bbox_a, bbox_b):
    _, ay1, _, ay2 = bbox_a
    _, by1, _, by2 = bbox_b
    oy1 = max(ay1, by1)
    oy2 = min(ay2, by2)
    return max(0.0, oy2 - oy1)


def _intersection_over_area(bbox_a, bbox_b):
    """
    Ratio of the intersection area to the area of bbox_a.
    """
    inter = _bbox_intersection(bbox_a, bbox_b)
    if inter is None:
        return 0.0
    area_a = _bbox_area(bbox_a)
    if area_a <= 0:
        return 0.0
    return _bbox_area(inter) / area_a


def _bbox_contains(outer, inner):
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2


def _iou(bbox_a, bbox_b):
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    inter = _bbox_intersection(bbox_a, bbox_b)
    if inter is None:
        return 0.0
    inter_area = (inter[2] - inter[0]) * (inter[3] - inter[1])

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    if area_a + area_b - inter_area <= 0:
        return 0.0

    return inter_area / float(area_a + area_b - inter_area)


def _point_in_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

def _point_contained_in_bbox(point, bbox):
    return _point_in_bbox(point, bbox)


def _norm(bbox, image_width, image_height):
    x1, y1, x2, y2 = bbox
    return [
        round(x1 / image_width, 4),
        round(y1 / image_height, 4),
        round(x2 / image_width, 4),
        round(y2 / image_height, 4),
    ]


def _norm_h(bbox, image_width, image_height):
    """Alias of _norm kept for readability in call sites."""
    return _norm(bbox, image_width, image_height)


def _union_bbox(bboxes):
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return [x1, y1, x2, y2]

def _expand_bbox(bbox, pad_x, pad_y):
    x1, y1, x2, y2 = bbox
    return [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y]

def _translate_bbox(bbox, dx, dy):
    x1, y1, x2, y2 = bbox
    return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

def _transform_bbox(bbox, matrix, dst_size):
    """
    Warp a [x1, y1, x2, y2] bbox through a 3x3 homography matrix.

    Args:
        bbox: [x1, y1, x2, y2] in source coordinates.
        matrix: 3x3 homography from source to destination.
        dst_size: [width, height] of the destination image. Coordinates are
                  clamped into this range when provided.

    Returns:
        [x1', y1', x2', y2'] in destination coordinates.
    """
    import numpy as np

    x1, y1, x2, y2 = bbox
    src = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
    )
    dst = cv2_perspective_transform(src, matrix)

    nx1 = float(np.min(dst[:, 0]))
    ny1 = float(np.min(dst[:, 1]))
    nx2 = float(np.max(dst[:, 0]))
    ny2 = float(np.max(dst[:, 1]))

    if dst_size:
        width, height = dst_size
        nx1 = max(0.0, min(nx1, width))
        ny1 = max(0.0, min(ny1, height))
        nx2 = max(0.0, min(nx2, width))
        ny2 = max(0.0, min(ny2, height))

    return [nx1, ny1, nx2, ny2]


def cv2_perspective_transform(points, matrix):
    """
    Thin wrapper around cv2.perspectiveTransform that also covers numpy-array
    inputs of the shape (N, 2) which cv2 does not accept directly.
    """
    import numpy as np

    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(-1, 2)
    if points.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) points, got shape {points.shape}")

    return _cv2_perspective_transform_impl(points, matrix)


def _cv2_perspective_transform_impl(points, matrix):
    import cv2

    reshaped = points.reshape(1, -1, 2)
    transformed = cv2.perspectiveTransform(reshaped, matrix)
    return transformed.reshape(-1, 2)


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