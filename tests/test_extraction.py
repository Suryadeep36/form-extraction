"""
Unit tests for _extract_one's anti-bleed behaviour.

The per-line value_bboxes contract says each underline box wraps exactly ONE
physical writing row. These tests pin the two regressions:

  1. A single-line field ("Full Name :" / "______") whose window slides over
     the printed header of the NEXT section ("ACPC Details :") must return
     only the field's handwriting, not the header.
  2. A multi-line field (Address with two stacked underlines) keeps one line
     per box and joins them in reading order.
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TABLE_MODEL_ENABLED"] = "false"

import util.config as config
importlib.reload(config)

from service.template_service import _extract_one

FAILURES = []


def check(name, condition, detail=""):
    status = "ok" if condition else "FAIL"
    print(f"  [{status}] {name}" + ("" if condition else f" -- {detail}"))
    if not condition:
        FAILURES.append(name)
    return condition


W, H = 1096, 1600


def _norm(px):
    return [px[0] / W, px[1] / H, px[2] / W, px[3] / H]


def _workspace(elements, labels_px=()):
    return {
        "image": None,
        "element_px": elements,
        "checkboxes": [],
        "global_labels_px": list(labels_px),
    }


def test_single_line_drops_adjacent_header_line():
    # Full Name underline [90,382,806,420] padded as build_template_fields
    # would: pad_up ~21.6, pad_down ~4.8 -> box [90,360.4,806,424.8].
    px = [90, 360.4, 806, 424.8]
    box = _norm(px)
    field = {
        "label": "Full Name of Applicant (as per Std.X):",
        "label_bbox": _norm([98, 346, 569, 369]),
        "value_bbox": box,
        "value_bboxes": [box],
        "value_bbox_px": px,
        "value_boxes_px": [px],
        "kind": "underline",
    }
    elements = [
        # Handwriting on the underline -> center ~403.
        {"bbox": [200, 392, 260, 414], "text": "John", "confidence": 0.95},
        # Printed header of the NEXT section, right below the underline.
        {"bbox": [107, 415, 241, 441], "text": "ACPC", "confidence": 0.99},
    ]
    res = _extract_one(field, _workspace(elements), W, H)
    check("header not merged into value", res["value"] == "John", res)


def test_multiline_keeps_one_line_per_box():
    # Address: two stacked underlines -> one box per row (padded).
    box1_px = [90, 1216, 761, 1278]
    box2_px = [90, 1268, 761, 1330]
    box1 = _norm(box1_px)
    box2 = _norm(box2_px)
    field = {
        "label": "Address:",
        "label_bbox": _norm([94, 1197, 181, 1221]),
        "value_bbox": _norm([90, 1216, 761, 1330]),
        "value_bboxes": [box1, box2],
        "value_bbox_px": [90, 1216, 761, 1330],
        "value_boxes_px": [box1_px, box2_px],
        "kind": "underline",
    }
    elements = [
        {"bbox": [200, 1238, 280, 1254], "text": "Street", "confidence": 0.95},
        {"bbox": [200, 1290, 260, 1306], "text": "City", "confidence": 0.95},
        # A leaked token from the row below the second underline; x is inside
        # the box so geometry alone cannot reject it. The per-line contract
        # must drop it because it is not the line nearest the second box.
        {"bbox": [700, 1325, 712, 1341], "text": "CC", "confidence": 0.9},
    ]
    res = _extract_one(field, _workspace(elements), W, H)
    check("multi-line value joined in order", res["value"] == "Street City", res)


def main():
    test_single_line_drops_adjacent_header_line()
    test_multiline_keeps_one_line_per_box()

    if FAILURES:
        print(f"FAILURES ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)


if __name__ == "__main__":
    main()