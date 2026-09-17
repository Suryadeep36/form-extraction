"""
Grid / cell-based form regression tests.

A "cell-based" field is a bordered rectangle whose label is printed INSIDE
the cell at the top-left and whose blank writing area is the space below it
("CONTACT PERSON NAME:" over an empty box). These fields are recovered by
`_detect_grid_cells` (line-skeleton grid reconstruction) and must:

  1. register with the FULL cell as the value region (not the label strip),
  2. keep labels the generic heuristic would reject -- single words without a
     colon ("NAME", "ADDRESS") and multi-dot labels ("SSN NO. OR TAXPAYER ID
     NO."),
  3. never duplicate a checkbox option group's question label, and
  4. never become title/instruction bands (they carry no blank writing space).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from util.form_utils import _detect_grid_cells
from util.template_utils import build_template_fields

FAILURES = []


def check(name, condition, detail=""):
    status = "ok" if condition else "FAIL"
    print(f"  [{status}] {name}" + ("" if condition else f" -- {detail}"))
    if not condition:
        FAILURES.append(name)
    return condition


def _el(text, bbox, height=20.0):
    return {
        "text": text,
        "bbox": list(bbox),
        "center": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
        "height": height,
    }


def _synthetic_grid_image():
    """A 3-cell grid: two side-by-side cells on row 1 and one full-width cell
    on row 2, enclosed by outer walls that span all bands (like a real form's
    outer frame)."""
    w, h = 1700, 2200
    img = np.full((h, w), 255, dtype=np.uint8)

    def hline(y):
        cv2.line(img, (87, y), (1613, y), 0, 2)

    def vline(x, y1=80, y2=400):
        cv2.line(img, (x, y1), (x, y2), 0, 2)

    for y in (100, 200, 250, 350):
        hline(y)
    vline(100, 80, 400)
    vline(900, 80, 400)
    vline(500, 100, 200)
    return img


def test_grid_cell_detection():
    img = _synthetic_grid_image()
    elements = [
        _el("CONTACT PERSON NAME:", [110, 120, 320, 138]),
        _el("TELEPHONE NUMBER:", [520, 120, 690, 138]),
        _el("ADDRESS:", [112, 258, 200, 276]),
    ]
    cells = _detect_grid_cells(
        img,
        elements=elements,
        median_text_h=18.0,
        table_bboxes=[],
        checkbox_group_bboxes=[],
    )
    check("3 grid cells detected", len(cells) == 3, f"cells={[[c['bbox'], c['grid_label']] for c in cells]}")

    by_label = {c["grid_label"]: c for c in cells}
    check("left cell labelled CONTACT PERSON NAME",
          "CONTACT PERSON NAME:" in by_label, list(by_label))
    check("right cell labelled TELEPHONE NUMBER",
          "TELEPHONE NUMBER:" in by_label)
    check("full-width cell labelled ADDRESS", "ADDRESS:" in by_label)

    contact = by_label["CONTACT PERSON NAME:"]
    bx1, by1, bx2, by2 = contact["bbox"]
    # Wall coordinates land on the line centres (within +/-2 px of the drawn
    # edges); the point is that the value box is the WHOLE cell, not the label
    # strip's height.
    check("cell bbox is the FULL cell (not the label strip)",
          abs(bx1 - 100) <= 2 and abs(by1 - 100) <= 2
          and abs(bx2 - 500) <= 2 and abs(by2 - 200) <= 2,
          contact["bbox"])
    check("label bbox is the interior top line",
          contact["grid_label_bbox"] == [110, 120, 320, 138],
          contact["grid_label_bbox"])


def test_no_cell_for_title_band():
    """Bands that are NOT writable fields must not become fields: a tight
    section-title strip (title fills the band, no blank writing space below)
    and a full-width instruction/heading band."""
    w, h = 1700, 2200
    img = np.full((h, w), 255, dtype=np.uint8)

    def hline(y):
        cv2.line(img, (87, y), (1613, y), 0, 2)

    def vline(x, y1, y2):
        cv2.line(img, (x, y1), (x, y2), 0, 2)

    # Tight title strip [100, 133].
    hline(100)
    hline(133)
    # Real full-width cell [180, 280].
    hline(180)
    hline(280)
    # Full-width heading band [340, 400] whose text spans the whole width.
    hline(340)
    hline(400)
    vline(100, 80, 400)
    vline(900, 80, 400)

    elements = [
        _el("AGENCY INFORMATION", [120, 111, 500, 129]),
        _el("CONTACT PERSON NAME:", [110, 198, 320, 216]),
        _el("THIS LINE IS PRINTED INSTRUCTIONS, NOT A FIELD LABEL", [101, 358, 901, 376]),
    ]
    cells = _detect_grid_cells(
        img, elements=elements, median_text_h=18.0,
        table_bboxes=[], checkbox_group_bboxes=[],
    )
    labels = [c["grid_label"] for c in cells]
    check("title strip band skipped", "AGENCY INFORMATION" not in labels, labels)
    check("full-width instruction band skipped",
          "THIS LINE IS PRINTED INSTRUCTIONS, NOT A FIELD LABEL" not in labels, labels)
    check("real cell kept", "CONTACT PERSON NAME:" in labels, labels)


def test_grid_cell_field_registration():
    w, h = 1700, 2200
    elements = [
        _el("CONTACT PERSON NAME:", [110, 120, 320, 138]),
        _el("NAME", [111, 403, 200, 421]),
        _el("SSN NO. OR TAXPAYER ID NO.", [520, 403, 800, 421]),
    ]
    regions = [
        {
            "id": "r0", "kind": "grid_cell",
            "bbox": [100, 100, 500, 200], "confidence": 0.35,
            "grid_label": "CONTACT PERSON NAME:",
            "grid_label_bbox": [110, 120, 320, 138],
        },
        {
            "id": "r1", "kind": "grid_cell",
            "bbox": [100, 390, 480, 490], "confidence": 0.35,
            "grid_label": "NAME", "grid_label_bbox": [111, 403, 200, 421],
        },
        {
            "id": "r2", "kind": "grid_cell",
            "bbox": [480, 390, 900, 490], "confidence": 0.35,
            "grid_label": "SSN NO. OR TAXPAYER ID NO.",
            "grid_label_bbox": [520, 403, 800, 421],
        },
    ]
    fields = build_template_fields(
        {"elements": elements, "input_regions": regions}, w, h
    )
    labels = [f["label"] for f in fields]
    check("cell field CONTACT PERSON NAME registered",
          "CONTACT PERSON NAME:" in labels, labels)
    check("colon-less single word NAME registered", "NAME" in labels, labels)
    check("multi-dot SSN label registered",
          "SSN NO. OR TAXPAYER ID NO." in labels, labels)

    contact = next(f for f in fields if f["label"] == "CONTACT PERSON NAME:")
    check("value_bbox is the FULL cell", contact["value_bbox"] == [
        round(100 / w, 5), round(100 / h, 5),
        round(500 / w, 5), round(200 / h, 5),
    ], contact["value_bbox"])


def test_grid_cell_does_not_shadow_checkbox_group_label():
    w, h = 1700, 2200
    elements = [
        _el("TYPE OF ACCOUNT:", [105, 505, 260, 525]),
        _el("CHECKING", [300, 505, 420, 525]),
        _el("SAVINGS", [455, 505, 555, 525]),
    ]
    regions = [
        {
            "id": "r0", "kind": "grid_cell",
            "bbox": [100, 500, 380, 560], "confidence": 0.35,
            "grid_label": "TYPE OF ACCOUNT:",
            "grid_label_bbox": [105, 505, 260, 525],
        },
    ]
    doc_rep = {
        "elements": elements,
        "input_regions": regions,
        "checkbox_groups": [
            {
                "bbox": [100, 500, 600, 560],
                "options": [
                    {"text": "CHECKING", "bbox": [300, 505, 420, 525]},
                    {"text": "SAVINGS", "bbox": [455, 505, 555, 525]},
                ],
            }
        ],
    }
    fields = build_template_fields(doc_rep, w, h)
    kinds = [(f["kind"], f["label"]) for f in fields]
    check("grid cell dropped when a checkbox group owns the label",
          all(k != "grid_cell" for k, _ in kinds), kinds)
    check("checkbox group field retained",
          any(k == "checkbox_group" and v == "TYPE OF ACCOUNT:" for k, v in kinds),
          kinds)


def main():
    test_grid_cell_detection()
    test_no_cell_for_title_band()
    test_grid_cell_field_registration()
    test_grid_cell_does_not_shadow_checkbox_group_label()
    print()
    if FAILURES:
        print(f"FAILURES ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("ALL GRID-CELL TESTS PASSED")


if __name__ == "__main__":
    main()