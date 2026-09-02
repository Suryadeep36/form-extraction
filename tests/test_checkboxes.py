"""
Unit tests for the checkbox/mark and option-instruction helpers.

These exercise the deterministic text helpers only (no OCR / table model), so
they are fast and reproducible.  Run with:

    ./venv/bin/python tests/run_all.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.checkbox_utils import (
    strip_leading_option_mark,
    checkbox_from_leading_mark,
)
from util.structure_utils import (
    _split_option_instruction,
    _split_multiselect_question,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "ok" if condition else "FAIL"
    print(f"  [{status}] {name}" + ("" if condition else f" -- {detail}"))
    if not condition:
        FAILURES.append(name)
    return condition


def test_strip_leading_option_mark():
    print("test_strip_leading_option_mark")
    # OCR glues a leading X onto an option label.
    stripped, changed = strip_leading_option_mark("XPartially Boatable")
    check("XPartially Boatable -> Partially Boatable", stripped == "Partially Boatable" and changed)
    check("changed flag set", changed is True)

    # Genuine spaced 'X Latitude North' / 'X Longitude West' are preserved.
    s, c = strip_leading_option_mark("X Latitude North")
    check("spaced 'X Latitude North' preserved", s == "X Latitude North" and not c)
    s, c = strip_leading_option_mark("X Longitude West")
    check("spaced 'X Longitude West' preserved", s == "X Longitude West" and not c)

    # Symbol marks are stripped as well.
    s, c = strip_leading_option_mark("_NO IfNO, check one below")
    check("_NO stripped", s == "NO IfNO, check one below" and c)
    s, c = strip_leading_option_mark("\u25a1Dry - Visited")
    check("square glyph stripped", s == "Dry - Visited" and c)
    s, c = strip_leading_option_mark("\u2713Checkeditem")
    check("check glyph stripped", s == "Checkeditem" and c)

    # Unmarked labels are untouched.
    s, c = strip_leading_option_mark("Wadeable")
    check("plain label untouched", s == "Wadeable" and not c)


def test_checkbox_from_leading_mark():
    print("test_checkbox_from_leading_mark")
    c = checkbox_from_leading_mark({"id": "t028", "text": "XPartially Boatable", "bbox": [10, 20, 100, 40]})
    check("leading-X yields checked candidate", c is not None and c.get("state") == "checked")
    if c:
        check("candidate tied to element id", c.get("element_id") == "t028")
        check("candidate label is stripped", c.get("associated_text") == "Partially Boatable")
        check("candidate source is ocr_mark", c.get("source") == "ocr_mark")

    # Spaced 'X Latitude North' must NOT become a checked option.
    c = checkbox_from_leading_mark({"id": "t011", "text": "X Latitude North", "bbox": [10, 20, 100, 40]})
    check("spaced latitude label not a checkbox", c is None, str(c))

    # No mark -> None.
    c = checkbox_from_leading_mark({"id": "t030", "text": "Boatable", "bbox": [10, 20, 100, 40]})
    check("unmarked label yields no candidate", c is None)


def test_split_option_instruction():
    print("test_split_option_instruction")
    check("glued IfNO splits", _split_option_instruction("NO IfNO, check one below") ==
          ("NO", "IfNO, check one below"))
    check("spaced If YES splits", _split_option_instruction("YES If YES, check one below") ==
          ("YES", "If YES, check one below"))
    check("plain label no instruction", _split_option_instruction("Partially Boatable") ==
          ("Partially Boatable", None))
    check("multiword no instruction", _split_option_instruction("Dry - Visited") ==
          ("Dry - Visited", None))


def test_split_multiselect_question():
    print("test_split_multiselect_question")
    q, opts = _split_multiselect_question(
        "Stream/River verified by (x all that apply):GPSlocal contactsignsroads  topo map"
    )
    check("multi-select marker recognized", q is not None, str(q))
    check("question lead extracted", q == "Stream/River verified by (x all that apply)", str(q))
    check("options segmented", opts == ["GPS", "local contact", "signs", "roads", "topo map"],
          str(opts))

    # Non-multi-select text is skipped safely.
    check("ordinary text skipped", _split_multiselect_question("Did You Sample This Site?") == (None, []))
    check("short field skipped", _split_multiselect_question("RIVER ID") == (None, []))


def main():
    tests = [
        test_strip_leading_option_mark,
        test_checkbox_from_leading_mark,
        test_split_option_instruction,
        test_split_multiselect_question,
    ]
    for test in tests:
        try:
            test()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            FAILURES.append(test.__name__)
            print(f"  [ERROR] {test.__name__}: {exc}")

    print()
    if FAILURES:
        print(f"FAILURES ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("ALL CHECKBOX TESTS PASSED")


if __name__ == "__main__":
    main()
