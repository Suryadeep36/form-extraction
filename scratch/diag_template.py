"""End-to-end: register mobile_form.png as a template, inspect the fields."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.image_utils import _load_image  # noqa: E402
from service.document_service import build_document_representation  # noqa: E402
from util.template_utils import build_template_fields  # noqa: E402

path = sys.argv[1]
doc_rep = build_document_representation(_load_image(path))
print(f"checkboxes={len(doc_rep['checkboxes'])}  groups={len(doc_rep['checkbox_groups'])}")

ih, iw = _load_image(path).shape[:2]
fields = build_template_fields(doc_rep, iw, ih)
cbs = [f for f in fields if f.get("kind") == "checkbox_group"]
print(f"total template fields={len(fields)}  checkbox_group fields={len(cbs)}")
for f in cbs:
    print(f"  label={f.get('label')!r}")
    print(f"    options={[o.get('text') for o in f.get('options') or []]}")
