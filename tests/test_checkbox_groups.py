"""
Checkbox-group pipeline tests (synthetic geometry).

Blind-tests the geometric stages without OCR/Paddle dependency:

    1. detect_checkbox_groups on a drawn image + handcrafted OCR elements:
       contour filtering (area/aspect gates), nearest-label binding, spatial
       clustering into a Gender (stacked) and an Employment Type (2x2) group;
       large boxes / long underlines are rejected.
    2. build_template_fields emits kind="checkbox_group" fields labelled via
       _label_for_region on the macro-box.
    3. _extract_checkbox_group on the blank and a filled variant returns the
       checked option texts.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from util.checkbox_group_utils import detect_checkbox_groups
from util.template_utils import build_template_fields
from service.template_service import _extract_checkbox_group

W, H = 800, 900


def _ring(img, box, thickness=1):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thickness)


def _el(text, bbox):
    x1, y1, x2, y2 = bbox
    return {
        "id": "",
        "type": "text",
        "text": text,
        "bbox": [float(x1), float(y1), float(x2), float(y2)],
        "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
        "width": x2 - x1,
        "height": float(y2 - y1),
        "confidence": 0.9,
        "words": [],
    }


def _blank_image():
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    # -- Gender group (stacked rows, right-hand column) --
    _ring(img, (436, 190, 462, 216))   # Male box
    _ring(img, (436, 222, 462, 248))   # Female box
    # -- Employment Type group (2x2, left columns) --
    _ring(img, (44, 570, 70, 596))     # Salaried
    _ring(img, (300, 570, 326, 596))   # Self-Employed
    _ring(img, (44, 606, 70, 632))     # Commission Based
    _ring(img, (300, 606, 326, 632))   # Others
    # -- Decoys: a large photo box and a long underline both rejected by
    #    the area / aspect gates --
    _ring(img, (600, 60, 780, 150))
    cv2.rectangle(img, (600, 400, 740, 430), (0, 0, 0), 2)
    return img


def _elements():
    return [
        _el("Gender", (430, 150, 560, 180)),
        _el("Male", (470, 193, 540, 215)),
        _el("Female", (470, 225, 540, 247)),
        _el("Employment Type", (120, 530, 260, 562)),
        _el("Salaried", (78, 573, 160, 595)),
        _el("Self-Employed", (334, 573, 470, 595)),
        _el("Commission Based", (78, 609, 230, 631)),
        _el("Others", (334, 609, 420, 631)),
    ]


def test_contour_filtering_and_binding_and_clustering():
    img = _blank_image()
    groups = detect_checkbox_groups(img, _elements())

    assert len(groups) == 2, f"expected 2 groups, got {len(groups)}: {groups}"

    by_opts = {frozenset(o["text"] for o in g["options"]): g for g in groups}
    assert frozenset({"Male", "Female"}) in by_opts, list(by_opts)
    assert frozenset({"Salaried", "Self-Employed", "Commission Based", "Others"}) in by_opts, \
        list(by_opts)

    gender = by_opts[frozenset({"Male", "Female"})]
    # Macro-box spans both stacked rows and lives only in its own column.
    assert gender["bbox"][1] <= 216 and gender["bbox"][3] >= 225, gender["bbox"]
    assert gender["bbox"][0] >= 400, gender["bbox"]

    emp = by_opts[frozenset({"Salaried", "Self-Employed", "Commission Based", "Others"})]
    # Group spans two stacked option rows (checkbox y 570..596 and 606..632).
    assert emp["bbox"][3] - emp["bbox"][1] > 50, emp["bbox"]
    # The decoy photo box / underline were rejected: no macro-box intrudes.
    for g in (gender, emp):
        assert not (g["bbox"][0] < 750 and g["bbox"][2] > 620
                    and g["bbox"][1] < 145 and g["bbox"][3] > 65), g["bbox"]


def test_template_fields_emit_checkbox_groups():
    img = _blank_image()
    elements = _elements()
    groups = detect_checkbox_groups(img, elements)
    doc_rep = {
        "elements": elements,
        "input_regions": [],
        "checkbox_groups": groups,
    }
    fields = build_template_fields(doc_rep, W, H)

    cb_fields = [f for f in fields if f["kind"] == "checkbox_group"]
    assert len(cb_fields) == 2, [f["kind"] for f in fields]

    by_label = {f["label_norm"]: f for f in cb_fields}
    assert "gender" in by_label, by_label
    assert "employment type" in by_label, by_label

    gender = by_label["gender"]
    assert {o["text"] for o in gender["options"]} == {"Male", "Female"}
    for o in gender["options"]:
        assert all(0.0 <= v <= 1.0 for v in o["bbox"]), o
    emp = by_label["employment type"]
    assert len(emp["options"]) == 4, emp["options"]

    # No decoy field inside the checkbox area: the macro-box is not a
    # writable "box"/"blank" input region at the same spot.
    assert all(f["value_bbox"][0] < 500 for f in fields), [f["value_bbox"] for f in fields]


def test_extraction_blank_and_filled():
    img_blank = _blank_image()
    elements = _elements()
    groups = detect_checkbox_groups(img_blank, elements)
    fields = build_template_fields(
        {"elements": elements, "input_regions": [], "checkbox_groups": groups}, W, H
    )
    gen = next(f for f in fields if f["label_norm"] == "gender")
    emp = next(f for f in fields if f["label_norm"] == "employment type")

    ws = {"image": img_blank}
    r = _extract_checkbox_group(dict(gen), ws, W, H)
    assert r["value"] == [], r
    assert all(not o["checked"] for o in r["options"] if isinstance(o.get("checked"), bool))

    # Filled variant: X on Male (Gender), filled block on Commission Based.
    img_filled = img_blank.copy()
    mx1, my1, mx2, my2 = 436, 190, 462, 216
    cv2.line(img_filled, (mx1, my1), (mx2, my2), (0, 0, 0), 3)
    cv2.line(img_filled, (mx1, my2), (mx2, my1), (0, 0, 0), 3)
    cv2.rectangle(img_filled, (47, 609), (67, 629), (0, 0, 0), -1)

    ws_f = {"image": img_filled}
    rg = _extract_checkbox_group(dict(gen), ws_f, W, H)
    rg = {o["text"]: o for o in rg["options"]}
    assert rg["Male"]["checked"] is True, rg
    assert rg["Female"]["checked"] is False, rg
    assert _extract_checkbox_group(dict(gen), ws_f, W, H)["value"] == ["Male"]

    re_ = _extract_checkbox_group(dict(emp), ws_f, W, H)
    opts = {o["text"]: o for o in re_["options"]}
    assert opts["Commission Based"]["checked"] is True, opts
    assert opts["Salaried"]["checked"] is False, opts
    assert re_["value"] == ["Commission Based"], re_["value"]


def test_repeated_label_rows_and_adjacent_questions():
    """Regression for the veteran Form 50-135 clustering bugs.

    Three Yes/No rows share the SAME two columns and the SAME "Yes"/"No"
    labels (a repeated-option table of separate sub-questions): each row must
    stay its own [Yes, No] group - never one 6-option mega-group.  A
    single-checkbox question row printed just above the table (60px away, well
    inside the old relaxed 2x band) must NOT fuse into the table either.
    """

    def qbox(img, y1, x):
        _ring(img, (x, y1, x + 36, y1 + 36))

    img = np.full((H, W, 3), 255, dtype=np.uint8)
    # Question row A (single): a 36px box next to a long option label.
    qbox(img, 55, 250)
    # Three Yes/No rows B, B2, B3 (pitch 60, identical box columns/labels).
    for y in (115, 175, 235):
        qbox(img, y, 500)
        qbox(img, y, 600)

    elements = [
        _el("Loss of one or more limbs", (294, 58, 520, 80)),
        _el("Are you the surviving spouse?", (70, 118, 300, 140)),
        _el("Yes", (560, 120, 600, 142)),
        _el("No", (660, 120, 690, 142)),
        _el("Are you a surviving child?", (70, 178, 290, 200)),
        _el("Yes", (560, 180, 600, 202)),
        _el("No", (660, 180, 690, 202)),
        _el("Unmarried?", (70, 238, 240, 260)),
        _el("Yes", (560, 240, 600, 262)),
        _el("No", (660, 240, 690, 262)),
    ]

    groups = detect_checkbox_groups(img, elements)
    opt_sets = sorted(frozenset(o["text"] for o in g["options"]) for g in groups)

    # One single-option group (row A) plus three independent [Yes, No] groups.
    assert opt_sets == [
        frozenset({"Loss of one or more limbs"}),
        frozenset({"Yes", "No"}),
        frozenset({"Yes", "No"}),
        frozenset({"Yes", "No"}),
    ], [sorted(s) for s in opt_sets]

    for g in groups:
        texts = [o["text"] for o in g["options"]]
        assert texts.count("Yes") <= 1 and texts.count("No") <= 1, (g["bbox"], texts)


def main():
    test_contour_filtering_and_binding_and_clustering()
    test_template_fields_emit_checkbox_groups()
    test_extraction_blank_and_filled()
    test_repeated_label_rows_and_adjacent_questions()
    print("  [ok] checkbox-group detection / template / extraction")


if __name__ == "__main__":
    main()