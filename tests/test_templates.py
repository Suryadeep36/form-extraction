"""
Two-pass template pipeline tests (real form pair).

Uses the committed reference pair in the repo root: empty.jpeg is the blank
Leave Application template, this.jpeg is a filled scan of the same layout.
Runs register_template -> extract_filled and asserts the KV pairs resolve
to the known hand-written values.
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TABLE_MODEL_ENABLED"] = "false"

import util.config as config
importlib.reload(config)

from service.template_service import register_template, extract_filled

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMPTY = os.path.join(ROOT, "empty.jpeg")
FILLED = os.path.join(ROOT, "this.jpeg")
ALSO_FILLED = os.path.join(ROOT, "that.jpeg")

TEMPLATE_ID = None


def _kv(fields):
    return {f["label"]: f["value"] for f in fields if f["value"]}


def _has(label, kv):
    return any(label in k.lower() for k in kv)


def test_template_register_extract():
    """Register the empty form once, then extract both fillings by alignment."""
    global TEMPLATE_ID
    assert os.path.exists(EMPTY), f"missing {EMPTY}"
    assert os.path.exists(FILLED), f"missing {FILLED}"

    template = register_template(EMPTY, name="Leave Application")
    TEMPLATE_ID = template["template_id"]

    fields = template["fields"]
    assert len(fields) >= 10, f"expected >=10 template fields, got {len(fields)}"
    labels = " ".join((f.get("label") or "").lower() for f in fields)
    assert "name" in labels and "mobile" in labels and "department" in labels, labels

    # The "Period From ___ to ___ No. of days" row must yield THREE template
    # fields (a separate "to" and "No. of days" key, not a glued "From").
    assert "to" in labels, f"'to' field missing from Period row -> {labels}"
    assert "no. of days" in labels, f"'No. of days' field missing -> {labels}"

    # The empty form's arrangement table must be captured as a template table.
    tables = template["tables"]
    assert tables, "template captured no tables"
    main = max(tables, key=lambda t: t["n_cells"])
    assert main["n_rows"] >= 3 and main["n_cols"] >= 4, (
        f"expected a real cell grid, got {main['n_rows']}x{main['n_cols']}"
    )
    assert main["structure_source"] in ("cv", "model", None), main["structure_source"]

    labels_seen = set()
    for image in (FILLED, ALSO_FILLED):
        result = extract_filled(template, image)
        labels_seen.add(result["alignment"]["method"])
        kv = _kv(result["fields"])
        assert set(labels_seen) <= {"homography", "similarity", "identity"}

        assert _has("date", kv) and _has("period", kv), f"{image}: missing anchors -> {kv}"
        assert _has("mobile", kv), f"{image}: mobile missing -> {kv}"
        assert _has("name", kv), f"{image}: name missing -> {kv}"

        # Every template table comes back with its cells filled from OCR text.
        assert result["tables"], f"{image}: extraction returned no tables"
        table = result["tables"][0]
        assert len(table["cells"]) >= main["n_cells"], table["cells"]
        all_text = " ".join(c["text"] or "" for c in table["cells"]).lower()
        assert "class" in all_text and "signature" in all_text, (
            f"{image}: table headers not preserved -> {all_text[:200]}"
        )
        if os.path.basename(image) == "this.jpeg":
            assert "load" in all_text, (
                f"{image}: handwritten table value not captured -> {all_text[:200]}"
            )


def main():
    test_template_register_extract()
    print("  [ok] template register/extract on real empty/filled pair")


if __name__ == "__main__":
    main()