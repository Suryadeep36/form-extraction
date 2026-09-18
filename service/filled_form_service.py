"""
File-backed persistence for FILLED (extracted) forms.

Every filled form run against a template is saved here as a JSON record plus
its aligned/warped image, so it never needs to be re-uploaded + re-extracted
after a refresh: it can be listed and reopened just like the empty templates
are (which live under uploads/templates/).

Layout on disk (uploads/filled_forms/):
    {id}.json   -> metadata + full extraction (image stored as a file ref)
    {id}.jpg    -> the aligned/warped image shown in the UI overlay
"""

import os
import json
import uuid
import base64
import datetime

from util.config import UPLOAD_DIR
from service.template_service import extract_filled, _json_safe

FILLED_DIR = os.path.join(UPLOAD_DIR, "filled_forms")


def _dir():
    os.makedirs(FILLED_DIR, exist_ok=True)
    return FILLED_DIR


def _json_path(filled_id):
    return os.path.join(_dir(), f"{filled_id}.json")


def _image_path(filled_id):
    return os.path.join(_dir(), f"{filled_id}.jpg")


def _mime_of(data_url):
    if not isinstance(data_url, str) or "," not in data_url:
        return "image/jpeg"
    prefix = data_url.split(",", 1)[0]
    return prefix[5:].split(";", 1)[0] or "image/jpeg"


def save_filled_form(template, image_path, source_filename=None):
    """Extract a FILLED form against `template` and persist the result.

    Returns the record dict as delivered to the API client (the warped image
    is included inline as a data URL for immediate display).  On disk the
    image is stored as a separate JPEG; the JSON file keeps a file reference
    instead of the base64 blob.
    """
    extraction = extract_filled(template, image_path)
    filled_id = str(uuid.uuid4())

    warped = extraction.get("warped_image_data_url")
    warped_mime = _mime_of(warped) if warped else "image/jpeg"
    stored_image = None
    if warped:
        try:
            payload = warped.split(",", 1)[1]
            with open(_image_path(filled_id), "wb") as f:
                f.write(base64.b64decode(payload))
            stored_image = f"{filled_id}.jpg"
        except Exception as e:
            print(f"[FILLED] warp image save failed: {e}")
            stored_image = None

    record = {
        "id": filled_id,
        "template_id": template.get("id"),
        "template_name": template.get("name"),
        "source_filename": source_filename or os.path.basename(image_path),
        "created_at": datetime.datetime.utcnow().isoformat(),
        "warped_image_file": stored_image,
        "warped_image_mime": warped_mime,
        "extraction": {
            k: v
            for k, v in extraction.items()
            if k != "warped_image_data_url"
        },
    }

    with open(_json_path(filled_id), "w") as f:
        json.dump(_json_safe(record), f, indent=2)

    # Client-facing record carries the warped image inline.
    if warped:
        extraction = dict(extraction)
        extraction["warped_image_data_url"] = warped
    record["extraction"] = extraction
    print(
        f"[FILLED] saved {filled_id} for {template.get('id')} "
        f"({len(extraction.get('fields') or [])} fields)"
    )
    return record


def list_filled_forms(template_id=None):
    """Summaries (no image payload) for all saved filled forms."""
    if not os.path.isdir(_dir()):
        return []
    out = []
    for fname in sorted(os.listdir(_dir()), reverse=True):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(_dir(), fname)) as f:
                rec = json.load(f)
            if template_id and rec.get("template_id") != template_id:
                continue
            extraction = rec.get("extraction") or {}
            out.append({
                "id": rec.get("id"),
                "template_id": rec.get("template_id"),
                "template_name": rec.get("template_name"),
                "source_filename": rec.get("source_filename"),
                "created_at": rec.get("created_at"),
                "field_count": len(extraction.get("fields") or []),
                "table_count": len(extraction.get("tables") or []),
                "has_image": bool(rec.get("warped_image_file")),
            })
        except Exception as e:
            print(f"[FILLED] skip {fname}: {e}")
    return out


def load_filled_form(filled_id, include_image=True):
    """Full record for one filled form, optionally with the warped image data
    URL reconstructed from the stored JPEG."""
    path = _json_path(filled_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        record = json.load(f)
    extraction = record.get("extraction") or {}

    if include_image and record.get("warped_image_file"):
        ipath = os.path.join(_dir(), record["warped_image_file"])
        if os.path.exists(ipath):
            with open(ipath, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            mime = record.get("warped_image_mime") or "image/jpeg"
            extraction = dict(extraction)
            extraction["warped_image_data_url"] = f"data:{mime};base64,{data}"

    record["extraction"] = extraction
    return record


def delete_filled_form(filled_id):
    removed = False
    for path in (_json_path(filled_id), _image_path(filled_id)):
        if os.path.exists(path):
            os.remove(path)
            removed = True
    return removed