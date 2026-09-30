"""Table-detection diagnostic for a form image."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.image_utils import _load_image  # noqa: E402
from service.document_service import build_document_representation  # noqa: E402

path = sys.argv[1]
img = _load_image(path)
print(f"image size: {img.shape[1]} x {img.shape[0]}")
doc_rep = build_document_representation(img)

print("\n=== TABLES ===")
print(f"count: {len(doc_rep.get('tables') or [])}")
for i, t in enumerate(doc_rep.get("tables") or []):
    b = [round(v) for v in t["bbox"]]
    cells = t.get("cells")
    print(f"  [{i}] bbox={b} w={b[2]-b[0]} h={b[3]-b[1]}"
          f" rows={t.get('rows')} cols={t.get('cols')}"
          f" n_cells={len(cells) if cells is not None else None}"
          f" keys={sorted(t.keys())}")

print("\n=== FIELDS ===")
fields = doc_rep.get("fields") or []
print(f"count: {len(fields)}")
kinds = {}
for f in fields:
    kinds[f.get("kind") or f.get("type") or "?"] = kinds.get(
        f.get("kind") or f.get("type") or "?", 0) + 1
print(f"kinds: {kinds}")

print("\n=== raw table candidate counters ===")
for key in ("table_candidates", "cv_candidates", "fused_tables", "table_bboxes",
            "horizontal_lines", "vertical_lines", "rectangular_regions"):
    v = doc_rep.get(key)
    print(f"  {key}: {len(v) if isinstance(v, list) else v}")
