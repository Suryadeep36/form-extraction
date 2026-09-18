import os
os.environ["TABLE_MODEL_ENABLED"] = "true"

import json
from util.form_utils import detect_input_regions
from service.ocr_service import get_ocr_data

path = "extract-form-forms/veteran_form.png"
elements = get_ocr_data(path)
regions = detect_input_regions(path, elements)

caps = [r for r in regions if r.get("source") == "caption"]
noncaps = [r for r in regions if r.get("source") != "caption"]
print(f"total regions={len(regions)} caption={len(caps)}")
print("== caption fields ==")
for r in caps:
    b = [round(v, 1) for v in r["bbox"]]
    lb = [round(v, 1) for v in r.get("label_bbox", [])]
    print(f"  {r.get('label','')[:40]:40s} kw={r['kind']:9s} bbox={b} label_bbox={lb}")
print("== non-caption underline/box ==")
for r in noncaps:
    if r["kind"] in ("underline", "box"):
        b = [round(v, 1) for v in r["bbox"]]
        print(f"  {r.get('label','')[:40]:40s} kw={r['kind']:9s} bbox={b}")