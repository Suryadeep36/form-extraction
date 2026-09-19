"""
Geometry regression tests.

Run with:
    ./venv/bin/python tests/run_all.py [--quick]

These tests exercise the deterministic CV path (table model stack disabled) so
they are reproducible without downloading pretrained table weights. They
verify the *geometry* dares to do: perspective, fields, tables, and the
label/value split of a classic "Period From ___ To ___" layout without any
form-specific text rules.
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TABLE_MODEL_ENABLED"] = "false"

import util.config as config
importlib.reload(config)

from synth import (
    make_leave_application,
    make_wired_table,
    make_weak_header_table,
    default_table_texts,
)

from service.document_service import analyze_document
from service.structure_service import resolve_structure

import test_checkboxes as checkboxes
import test_templates as templates
import test_label_fixes as label_fixes
import test_extraction as extraction
import test_grid_cells as grid_cells
import test_watermark as watermark

FAILURES = []


def check(name, condition, detail=""):
    status = "ok" if condition else "FAIL"
    print(f"  [{status}] {name}" + ("" if condition else f" -- {detail}"))
    if not condition:
        FAILURES.append(name)
    return condition


def _analyze(image, name="synthetic"):
    return analyze_document(image, image_path=name)


# ---------------------------------------------------------------------------
# Reference leave-application form
# ---------------------------------------------------------------------------

def test_leave_form_fields():
    print("test_leave_form_fields")
    image = make_leave_application()
    doc_rep, _ = _analyze(image, "leave_application")

    fields = doc_rep["fields"]

    def find(needle):
        return [f for f in fields if (f["label"] or "").lower().count(needle)]

    from_fields = find("from")
    check("'From' field detected", bool(from_fields), f"labels={[(f['label'], f['value']) for f in fields]}")
    check(
        "'From' value is 01-06-2026",
        any("01-06-2026" in (f["value"] or "") for f in from_fields),
        [(f["label"], f["value"]) for f in from_fields],
    )

    to_fields = find("to")
    check("'To' field detected", bool(to_fields))
    check(
        "'To' value is 05-06-2026",
        any(f["value"] and "05-06-2026" in f["value"] for f in to_fields),
        [(f["label"], f["value"]) for f in to_fields],
    )

    days_fields = find("leave days")
    check("'Leave Days' field detected", bool(days_fields))
    check(
        "'Leave Days' value is 5",
        any(f["value"] and f["value"].strip() == "5" for f in days_fields),
        [(f["label"], f["value"]) for f in days_fields],
    )

    merged = [
        f for f in fields
        if "from" in (f["label"] or "").lower() and "to" in (f["label"] or "").lower()
    ]
    check("From/To stay separate fields (no hardcoded split)", not merged,
          [(f["label"], f["value"]) for f in merged])

    check("No field's value glues From+To together", not any(
        f["value"] and "01-06-2026" in f["value"] and "05-06-2026" in f["value"]
        for f in fields
    ))


def test_leave_form_resolver_backfill():
    print("test_leave_form_resolver_backfill")
    image = make_leave_application()
    doc_rep, candidates = _analyze(image, "leave_application")

    # Sparse LLM answer: no label_value relationships at all. The resolver
    # must surface the geometry fields in the section anyway.
    llm = {
        "document_type": "leave application",
        "sections": [
            {
                "name": "Main Form",
                "region_ids": [],
                "element_ids": [e["id"] for e in doc_rep["elements"]],
            }
        ],
        "relationships": [],
        "unassigned_text_ids": [],
    }

    result, reasons = resolve_structure(llm, doc_rep, candidates)
    fields = [f for s in result["sections"] for f in s["fields"]]
    labels = " ".join((f["label"] or "").lower() for f in fields)
    check("backfill keeps From/To/Leave labels", all(
        key in labels for key in ("from", "to", "leave")
    ), labels)

    values = [f["value"] for f in fields if f["value"]]
    check("backfill keeps the From date value", any("01-06-2026" in v for v in values), values)
    check("backfill keeps the To date value", any("05-06-2026" in v for v in values), values)

    with_regions = [f for f in fields if f.get("input_region")]
    check("backfilled fields carry input regions", len(with_regions) >= 3,
          f"only {len(with_regions)} had regions")


def test_leave_form_perspective_invariance():
    print("test_leave_form_perspective_invariance")
    import cv2
    import numpy as np
    image = make_leave_application()

    doc_rep, _ = _analyze(image, "leave_application")
    check("perspective metadata present", "perspective_correction" in doc_rep["preprocessing"])


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def test_wired_table_structure():
    print("test_wired_table_structure")
    image = make_wired_table(
        cols=3, data_rows=3, header=("Item", "Qty", "Price"),
        cell_texts=default_table_texts(3, 3),
    )
    doc_rep, _ = _analyze(image, "wired_table")

    check("table detected", len(doc_rep["tables"]) >= 1)
    if not doc_rep["tables"]:
        return
    table = doc_rep["tables"][0]
    # header + 3 data rows across 3 columns = 12 cells in a full-wire grid.
    check("full grid cell count", table["n_cells"] >= 12, f"cells={table['n_cells']}")

    texts = " ".join((c.get("text") or "") for c in table["cells"])
    for needle in ("Item", "Qty", "Price", "Widget"):
        check(f"cell OCR contains {needle!r}", needle in texts, texts)

    spans = [(c["row_span"], c["column_span"]) for c in table["cells"]]
    check("no accidental merges in a full grid", all(s == (1, 1) for s in spans), spans)


def test_merged_header_table():
    print("test_merged_header_table")
    image = make_weak_header_table()
    doc_rep, _ = _analyze(image, "merged_header")

    check("merged-header table detected", len(doc_rep["tables"]) >= 1)
    if not doc_rep["tables"]:
        return
    table = doc_rep["tables"][0]

    header_cells = [c for c in table["cells"] if c.get("row") == 0]
    check("header row has a merged cell", any(
        c["column_span"] >= 2 for c in header_cells
    ), f"header spans={[c['column_span'] for c in header_cells]}")

    texts = " ".join((c.get("text") or "") for c in table["cells"])
    check("header text preserved", "Details" in texts, texts)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        test_leave_form_fields,
        test_leave_form_resolver_backfill,
        test_leave_form_perspective_invariance,
        test_wired_table_structure,
        test_merged_header_table,
        checkboxes.test_strip_leading_option_mark,
        checkboxes.test_checkbox_from_leading_mark,
        checkboxes.test_split_option_instruction,
        checkboxes.test_split_multiselect_question,
        templates.test_template_register_extract,
        label_fixes.main,
        extraction.main,
        grid_cells.main,
        watermark.main,
    ]
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        tests = [test_leave_form_fields, test_wired_table_structure]

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
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()