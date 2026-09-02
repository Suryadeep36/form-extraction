import sys
import json

from service.document_service import build_document_representation

image = sys.argv[1] if len(sys.argv) > 1 else "form.jpeg"

doc_rep = build_document_representation(image)

print(f"\n=== {image} ===")
print(f"size={doc_rep['image_width']}x{doc_rep['image_height']}")
print(f"elements={len(doc_rep['elements'])} regions={len(doc_rep['regions'])} checkboxes={len(doc_rep['checkboxes'])}")

print("\n-- ELEMENTS --")
for e in doc_rep["elements"]:
    print(
        f"{e['id']:12s} {e['text']!r:45s} bbox={[round(x,4) for x in e['bbox']]}"
        + (f" is_value={e.get('is_value')}" if e.get("is_value") else "")
    )

print("\n-- REGIONS --")
for r in doc_rep["regions"]:
    print(f"{r['id']:6s} bbox={[round(x,4) for x in r['bbox']]}")

print("\n-- FIELDS --")
for f in doc_rep.get("fields", []):
    print(json.dumps(f, default=str))

print("\n-- TABLES --")
for t in doc_rep.get("tables", []):
    print(
        f"bbox={[round(x,2) for x in t['bbox']]} conf={t.get('confidence')} "
        f"source={t.get('structure_source')} cells={len(t.get('cells') or [])}"
    )
    for c in t.get("cells") or []:
        print(
            f"  {c['id']} r{c['row']}c{c['column']} span={c.get('row_span')}x{c.get('column_span')} "
            f"bbox={[round(x,2) for x in c['bbox']]} text={c.get('text')!r}"
        )