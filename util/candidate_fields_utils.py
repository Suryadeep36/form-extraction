import re

from util.geometry_utils import (
    _point_in_bbox
)

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
        if _point_in_bbox(lc, region["bbox"]) and _point_in_bbox(vc, region["bbox"]):
            score += 1.0
            break

    return score