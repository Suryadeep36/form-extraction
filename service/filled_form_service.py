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
from util.config import UPLOAD_DIR
from service.template_service import extract_filled, _json_safe
from service.s3_service import upload_to_s3, download_from_s3, list_from_s3, delete_from_s3

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


def save_filled_form(user_id, template, image_path, source_filename=None):
    """Extract a FILLED form against `template` and persist the result to S3."""
    extraction = extract_filled(template, image_path)
    filled_id = str(uuid.uuid4())

    warped = extraction.get("warped_image_data_url")
    warped_mime = _mime_of(warped) if warped else "image/jpeg"
    stored_image = None
    if warped:
        try:
            payload = warped.split(",", 1)[1]
            upload_to_s3(user_id, "filled_forms", f"{filled_id}.jpg", base64.b64decode(payload), warped_mime)
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

    record_json = json.dumps(_json_safe(record), indent=2).encode('utf-8')
    upload_to_s3(user_id, "filled_forms", f"{filled_id}.json", record_json, "application/json")

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


def list_filled_forms(user_id, template_id=None):
    """Summaries (no image payload) for all saved filled forms in S3."""
    keys = list_from_s3(user_id, "filled_forms")
    out = []
    # Reverse sort gives approximate newest-first if using UUIDs, but created_at is better
    for key in sorted(keys, reverse=True):
        if not key.endswith(".json"):
            continue
        try:
            file_id = key.split("/")[-1]
            meta_bytes = download_from_s3(user_id, "filled_forms", file_id)
            if not meta_bytes: continue
            
            rec = json.loads(meta_bytes)
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
            print(f"[FILLED] skip {key}: {e}")
    # Sort out by created_at explicitly
    out.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return out


def load_filled_form(user_id, filled_id, include_image=True):
    """Full record for one filled form, optionally with the warped image data
    URL reconstructed from the stored JPEG in S3."""
    meta_bytes = download_from_s3(user_id, "filled_forms", f"{filled_id}.json")
    if not meta_bytes:
        return None
        
    record = json.loads(meta_bytes)
    extraction = record.get("extraction") or {}

    if include_image and record.get("warped_image_file"):
        img_bytes = download_from_s3(user_id, "filled_forms", record["warped_image_file"])
        if img_bytes:
            data = base64.b64encode(img_bytes).decode()
            mime = record.get("warped_image_mime") or "image/jpeg"
            extraction = dict(extraction)
            extraction["warped_image_data_url"] = f"data:{mime};base64,{data}"

    record["extraction"] = extraction
    return record


def delete_filled_form(user_id, filled_id):
    removed_json = delete_from_s3(user_id, "filled_forms", f"{filled_id}.json")
    removed_img = delete_from_s3(user_id, "filled_forms", f"{filled_id}.jpg")
    return removed_json or removed_img