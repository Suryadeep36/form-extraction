"""
Focused registration-time label fixes.

Covers the three edge cases from the GUJCET/GTU blanks:
  1. Date of Birth: OCR glues "Date of Birth://" into one box that straddles
     the slashed value underline -> the word-level prefix isolates the
     "Date of Birth" label.
  2. Name of Board: the line sits below "(Gujarat Board / CBSE/ISCE / OTHER
     Board):" while the real primary label is the line above; label found
     upward, parenthetical instruction dropped.
  3. Address: a value underline whose row has no left label at all picks up
     the "Address:" header directly above.

Also pins the orphan-adoption guard: a blank underline row below a labelled
field is absorbed into that field's value_bbox, but an orphan that overlaps
printed text (a heading row, not a blank) must NOT be absorbed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.template_utils import _label_for_region, build_template_fields

FAILURES = []


def check(name, condition, detail=""):
    status = "ok" if condition else "FAIL"
    print(f"  [{status}] {name}" + ("" if condition else f" -- {detail}"))
    if not condition:
        FAILURES.append(name)
    return condition


def _run(regions, elements, w=1096, h=1600):
    doc_rep = {"elements": elements, "input_regions": regions}
    return build_template_fields(doc_rep, w, h)


def test_dob_slashed_line_label_uses_word_prefix():
    elements = [
        {
            "text": "Date of Birth://",
            "bbox": [498, 725, 834, 750],
            "center": [666, 737.5],
            "words": [
                {"text": "Date", "bbox": [509, 726, 543, 749]},
                {"text": " ", "bbox": [547, 726, 551, 749]},
                {"text": "of", "bbox": [559, 726, 570, 749]},
                {"text": " ", "bbox": [574, 726, 578, 749]},
                {"text": "Birth", "bbox": [582, 726, 624, 749]},
                {"text": "://", "bbox": [631, 726, 822, 749]},
            ],
        }
    ]
    region = {"bbox": [728, 730, 995, 766], "kind": "underline"}
    li = _label_for_region(region, elements, 23.5)
    check("DOB label != None", li is not None, repr(li))
    if li:
        check("DOB label is 'Date of Birth'", li["text"] == "Date of Birth", repr(li["text"]))


def test_board_multiline_label_recovered_from_above():
    elements = [
        {
            "text": "Name of Board from which qualifying Examination Std. 12th (Science) passed:",
            "bbox": [129, 631, 899, 663],
            "center": [514, 647],
        },
        {
            "text": "(Gujarat Board / CBSE/ISCE / OTHER Board):",
            "bbox": [134, 665, 572, 691],
            "center": [353, 678],
        },
    ]
    region = {"bbox": [569, 668, 998, 704], "kind": "underline"}
    li = _label_for_region(region, elements, 23.5)
    want = "Name of Board from which qualifying Examination Std. 12th (Science) passed:"
    check("Board label != None", li is not None, repr(li))
    if li:
        check("Board label is the primary line", li["text"] == want, repr(li["text"]))


def test_address_label_found_directly_above():
    elements = [
        {"text": "Address:", "bbox": [94, 1197, 181, 1221], "center": [137, 1209]}
    ]
    region = {"bbox": [90, 1238, 761, 1274], "kind": "underline"}
    li = _label_for_region(region, elements, 23.5)
    check("Address label != None", li is not None, repr(li))
    if li:
        check("Address label == 'Address:'", li["text"] == "Address:", repr(li["text"]))


def test_header_noise_not_taken_as_upward_label():
    # Watermark junk ('AHA' / 'or o.') has no ':' -> must not become a label.
    elements = [
        {"text": "AHA", "bbox": [385, 168, 420, 192], "center": [402, 180]}
    ]
    region = {"bbox": [317, 201, 935, 237], "kind": "underline"}
    li = _label_for_region(region, elements, 23.5)
    check("no-colon noise is not a label", li is None, repr(li))


def test_stacked_orphan_underline_adopted_into_address_field():
    elements = [
        {"text": "Address:", "bbox": [94, 1197, 181, 1221], "center": [137, 1209]}
    ]
    regions = [
        {"bbox": [90, 1238, 761, 1274], "kind": "underline"},
        {"bbox": [90, 1290, 761, 1326], "kind": "underline"},
    ]
    fields = _run(regions, elements)
    check("stacked pair -> one field", len(fields) == 1, [f["label"] for f in fields])
    if fields:
        f = fields[0]
        check("orphan adopted label", f["label"] == "Address:", repr(f["label"]))
        # value_bbox stretches over BOTH lines (y2 at the second line).
        y1, y2 = f["value_bbox"][1], f["value_bbox"][3]
        check("value covers both lines", abs((y2 - y1) * 1600 - (1326 - 1238)) < 2, f["value_bbox"])


def test_orphan_over_headline_text_not_adopted():
    # "Qualifying Examination Marks..." underline is wider than its label and
    # overlaps printed text -> it is a heading, not the second address line,
    # so "Std . 12th Seat No" must stay single-line.
    elements = [
        {"text": "Std . 12th Seat No", "bbox": [142, 730, 294, 751], "center": [218, 740.5]},
        {"text": "Qualifving Examination Marks ( Std . 12th ):", "bbox": [141, 789, 516, 809], "center": [328, 799]},
    ]
    regions = [
        {"bbox": [294, 734, 507, 770], "kind": "underline"},
        {"bbox": [127, 793, 528, 829], "kind": "underline"},
    ]
    fields = _run(regions, elements)
    check("seat-no stays single field", len(fields) == 1, [f["label"] for f in fields])
    if fields:
        f = fields[0]
        y2 = f["value_bbox"][3]
        check("seat-no not stretched over QE line", abs((y2 - 0.4821875)) < 0.01, f["value_bbox"])


def main():
    test_dob_slashed_line_label_uses_word_prefix()
    test_board_multiline_label_recovered_from_above()
    test_address_label_found_directly_above()
    test_header_noise_not_taken_as_upward_label()
    test_stacked_orphan_underline_adopted_into_address_field()
    test_orphan_over_headline_text_not_adopted()

    if FAILURES:
        print(f"FAILURES ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("  [ok] label edge-case fixes")


if __name__ == "__main__":
    main()