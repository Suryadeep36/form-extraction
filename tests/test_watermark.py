"""
Watermark / oversized-graphic regression tests.

A blank form is registered into a template only when its value regions are
empty.  A faint company watermark printed in the background ("Q123RF") can sit
inside a field's writing area and be misread as printed content, wrongly
discarding the field (the application-form Address line was lost this way).
The empty-form guard must ignore elements far taller than the page's own text
height - such an oversized element can never be printed content of a single
writing line - while still rejecting real-sized printed text.

Run with:
    ./venv/bin/python tests/test_watermark.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.template_utils import build_template_fields

FAILURES = []

W, H = 1269, 714


def check(name, condition, detail=""):
    status = "ok" if condition else "FAIL"
    print(f"  [{status}] {name}" + ("" if condition else f" -- {detail}"))
    if not condition:
        FAILURES.append(name)
    return condition


def _el(text, bbox, height=28.0):
    return {
        "text": text,
        "bbox": list(bbox),
        "center": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
        "height": height,
    }


def _doc_rep(elements):
    """Address-like layout: a labelled underline plus an adopted second line."""
    return {
        "elements": elements,
        "input_regions": [
            {"bbox": [23.0, 271.0, 883.0, 315.0], "kind": "underline"},
            {"bbox": [23.0, 328.5, 883.0, 372.5], "kind": "underline"},
        ],
        "checkbox_groups": [],
    }


def _baseline_elements():
    return [
        _el("Address :", [29, 261, 142, 289], 28.0),
        _el("Zip Code", [969, 470, 1065, 502], 28.0),
        _el("City", [400, 470, 445, 502], 28.0),
    ]


def test_oversized_watermark_keeps_field():
    # Watermark 'Q123RF' is 92px tall (3.3x the page's 28px text) and overlaps
    # the second address line; it must NOT discard the Address field.
    elements = _baseline_elements() + [
        _el("Q123RF", [388, 320, 484, 412], 92.0),
    ]
    doc_rep = _doc_rep(elements)
    fields = build_template_fields(doc_rep, W, H)

    addresses = [
        f for f in fields
        if "address" in (f.get("label") or "").lower()
    ]
    check("Address field kept despite watermark", len(addresses) == 1,
          [f.get("label") for f in fields])
    if addresses:
        # Both address lines are its value region.
        vb = addresses[0]["value_bbox"]
        check("Address spans both lines", vb[3] - vb[1] > 0.14,
              vb)


def test_normal_sized_text_still_discards():
    # A normal-height printed text inside the writing area is real content on
    # a "blank" form: the field must still be discarded (guard intact).
    elements = _baseline_elements() + [
        _el("Q123RF", [388, 330, 484, 358], 28.0),
    ]
    doc_rep = _doc_rep(elements)
    fields = build_template_fields(doc_rep, W, H)

    addresses = [
        f for f in fields
        if "address" in (f.get("label") or "").lower()
    ]
    check("normal-sized text still discards field", len(addresses) == 0,
          [f.get("label") for f in fields])


def test_no_watermark_keeps_field():
    elements = _baseline_elements()
    fields = build_template_fields(_doc_rep(elements), W, H)
    addresses = [
        f for f in fields
        if "address" in (f.get("label") or "").lower()
    ]
    check("blank Address field registered", len(addresses) == 1,
          [f.get("label") for f in fields])


def main():
    test_no_watermark_keeps_field()
    test_oversized_watermark_keeps_field()
    test_normal_sized_text_still_discards()
    print()
    if FAILURES:
        print(f"FAILURES ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("ALL WATERMARK TESTS PASSED")


if __name__ == "__main__":
    main()