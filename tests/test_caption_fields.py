"""
Caption-under-rule field detection.

A writing rule (underline/box) with a SMALL field-name caption printed
directly BENEATH it should register as a labelled template field.  Header
rules that simply sit above a printed label ("Form 50-135" under its box
border) or a footer rule that underlines content must not become captions.

Also covers:
  * box variant: a plain box + small caption below → label attached.
  * existing-label guard: a box that already has a left/above label is
    not re-labelled by the caption beneath.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from util.form_utils import (
    _caption_candidates_below,
    _detect_caption_fields,
    _caption_for_unlabelled_box,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "ok" if condition else "FAIL"
    print(f"  [{status}] {name}" + ("" if condition else f" -- {detail}"))
    if not condition:
        FAILURES.append(name)
    return condition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MH = 27.0
SMALL_MAX = 0.9 * MH
IMG_H, IMG_W = 2200, 1700


def _empty_gray():
    return np.full((IMG_H, IMG_W), 255, dtype=np.uint8)


def _draw_dashed_line(gray, y, x1, x2, dash_len=80, gap=40, thickness=2):
    x = x1
    while x < x2:
        ex = min(x + dash_len, x2)
        cv2.line(gray, (x, y), (ex, y), 50, thickness)
        x = ex + gap


def _draw_solid_line(gray, y, x1, x2, thickness=2):
    cv2.line(gray, (x1, y), (x2, y), 50, thickness)


def _draw_rect_outline(gray, x1, y1, x2, y2, thickness=2):
    cv2.rectangle(gray, (x1, y1), (x2, y2), 50, thickness)


def _el(text, x1, y1, x2, y2):
    return {
        "text": text,
        "bbox": [float(x1), float(y1), float(x2), float(y2)],
        "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
        "height": float(y2 - y1),
    }


# ---------------------------------------------------------------------------
# _caption_candidates_below
# ---------------------------------------------------------------------------

def test_small_text_below_rule_is_captured():
    mh = 27.0
    rule = [100, 500, 1200, 500]
    el = _el("Address", 200, 515, 400, 535)
    caps = _caption_candidates_below(rule, [el], mh)
    check("small text captured", len(caps) == 1, repr(caps))
    if caps:
        check("caption text correct", caps[0]["text"] == "Address", caps[0])


def test_large_text_below_rule_rejected():
    mh = 27.0
    rule = [100, 500, 1200, 500]
    el = _el("Full Name of the Applicant here", 200, 515, 400, 548)
    caps = _caption_candidates_below(rule, [el], mh)
    check("large text rejected", len(caps) == 0, repr(caps))


def test_text_far_below_rule_rejected():
    mh = 27.0
    rule = [100, 500, 1200, 500]
    el = _el("Address", 200, 550, 400, 570)
    caps = _caption_candidates_below(rule, [el], mh)
    check("far text rejected", len(caps) == 0, repr(caps))


# ---------------------------------------------------------------------------
# _detect_caption_fields
# ---------------------------------------------------------------------------

def test_single_caption_under_dashed_rule():
    gray = _empty_gray()
    _draw_dashed_line(gray, y=600, x1=100, x2=1200)
    elements = [_el("Legal Description (if known)", 200, 615, 400, 635)]
    regions = _detect_caption_fields(gray, elements, MH)
    check("one region found", len(regions) == 1, [len(regions)] + [
        (r.get("label"), [round(v, 1) for v in r["bbox"]]) for r in regions
    ])
    if regions:
        r = regions[0]
        check("label correct", r["label"] == "Legal Description (if known)", r.get("label"))
        check("kind is underline", r["kind"] == "underline", r.get("kind"))
        pad = max(8.0, 0.8 * MH)
        check("bbox hugs the rule", abs(r["bbox"][1] - (600 - pad)) <= 2.0, r["bbox"])
        check("label_bbox set", r.get("label_bbox") is not None, r.get("label_bbox"))


def test_two_captions_under_dashed_rule_produce_two_fields():
    gray = _empty_gray()
    _draw_dashed_line(gray, y=980, x1=189, x2=1518)
    elements = [
        _el("Branch of Service", 200, 995, 400, 1015),
        _el("Serial Number", 1200, 995, 1400, 1015),
    ]
    regions = _detect_caption_fields(gray, elements, MH)
    check("two regions found", len(regions) == 2, [len(regions)])
    labels = sorted(r["label"] for r in regions)
    check("labels are Branch/Serial", labels == ["Branch of Service", "Serial Number"], labels)
    for r in regions:
        check(f"bbox width < half row ({r['label']})", r["bbox"][2] - r["bbox"][0] < 700, r["bbox"])


def test_footer_line_with_text_above_rejected():
    """A footer rule whose span is filled by content above it must not become a
    caption field — even though a small 'label' text sits below it."""
    gray = _empty_gray()
    _draw_solid_line(gray, y=2121, x1=64, x2=1638)
    elements = [
        _el("Page 2 \u2022 50-135", 64, 2100, 300, 2120),
        _el("notes", 100, 2130, 200, 2148),
    ]
    regions = _detect_caption_fields(gray, elements, MH)
    check("footer line rejected", len(regions) == 0, [len(regions)])


def test_header_decoy_box_border_line_rejected():
    """A short box-border rule in the form header with 'Form 50-135' printed
    directly below it and 'Property Tax' riding above must not become a
    caption field (bbox-overlap text_above guard)."""
    gray = _empty_gray()
    _draw_solid_line(gray, y=100, x1=93, x2=273)
    elements = [
        _el("Property Tax", 70, 85, 180, 110),
        _el("Form 50-135", 110, 115, 250, 135),
    ]
    regions = _detect_caption_fields(gray, elements, MH)
    check("header decoy rejected", len(regions) == 0, [len(regions)])


def test_header_box_header_line_guard():
    """A caption that is the first text inside a closed box whose top edge
    coincides with the detected rule must be rejected."""
    gray = _empty_gray()
    _draw_rect_outline(gray, 100, 100, 400, 200, thickness=2)
    _draw_solid_line(gray, y=100, x1=100, x2=400)
    elements = [
        _el("Form 50-135", 110, 115, 250, 135),
    ]
    regions = _detect_caption_fields(gray, elements, MH)
    check("box-header decoy rejected", len(regions) == 0, [len(regions)])


# ---------------------------------------------------------------------------
# _caption_for_unlabelled_box
# ---------------------------------------------------------------------------

def test_unlabelled_box_gets_caption():
    gray = _empty_gray()
    _draw_rect_outline(gray, 100, 100, 400, 200)
    box = [100.0, 100.0, 400.0, 200.0]
    elements = [_el("Date", 120, 215, 250, 235)]
    cap = _caption_for_unlabelled_box(box, elements, MH)
    check("caption found", cap is not None, repr(cap))
    if cap:
        check("caption text == 'Date'", cap["text"] == "Date", cap["text"])


def test_box_with_above_label_not_relabelled():
    """A box that already has a printed label directly above or to the left
    must NOT be re-labelled by the caption beneath it."""
    box = [200.0, 200.0, 500.0, 300.0]
    elements = [
        _el("Question:", 200, 170, 320, 195),
        _el("hint", 220, 315, 300, 335),
    ]
    cap = _caption_for_unlabelled_box(box, elements, MH)
    check("existing label respected", cap is None, repr(cap))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    test_small_text_below_rule_is_captured()
    test_large_text_below_rule_rejected()
    test_text_far_below_rule_rejected()
    test_single_caption_under_dashed_rule()
    test_two_captions_under_dashed_rule_produce_two_fields()
    test_footer_line_with_text_above_rejected()
    test_header_decoy_box_border_line_rejected()
    test_header_box_header_line_guard()
    test_unlabelled_box_gets_caption()
    test_box_with_above_label_not_relabelled()

    if FAILURES:
        print(f"FAILURES ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("  [ok] caption-under-rule detection")


if __name__ == "__main__":
    main()
