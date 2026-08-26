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

def _point_in_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

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