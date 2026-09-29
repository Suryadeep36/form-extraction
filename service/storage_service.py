"""
S3 persistence for uploaded unstructured documents and their extraction output.

Files are stored on S3.
- `{doc_id}.json` stores metadata and full response
- `{doc_id}_img.jpg` stores the perspective-corrected image

All helpers return builtin types only so the FastAPI
layer can serialize them directly.
"""

import os
import json
import uuid
import base64
from datetime import datetime

from service.s3_service import upload_to_s3, download_from_s3, list_from_s3, delete_from_s3


def init_db() -> bool:
    """No-op. Left here to satisfy lifespan requirements."""
    print("[STORAGE] Using S3 for all document storage. No DB needed.")
    return True


def storage_enabled() -> bool:
    """Always return True since S3 is our only persistent store."""
    return True


def _decode_data_url(data_url):
    """Decode a base64 data URL (e.g. 'data:image/jpeg;base64,....') to bytes."""
    if not data_url:
        raise ValueError("Missing processed image data URL")
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def save_document(user_id, filename, mime_type, image_data_url, doc_rep, response_body):
    """Persist a corrected document image + its extraction output to S3."""
    data = (response_body or {}).get("data") or {}
    document_type = data.get("document_type") if isinstance(data, dict) else None

    doc_id = str(uuid.uuid4())
    stored_img_name = f"{doc_id}_img.jpg"
    
    image_bytes = _decode_data_url(image_data_url)
    upload_to_s3(user_id, "documents", stored_img_name, image_bytes, mime_type)

    record = {
        "id": doc_id,
        "original_filename": filename,
        "file_path": stored_img_name,
        "mime_type": mime_type,
        "document_type": document_type,
        "status": "completed",
        "image_width": doc_rep.get("image_width"),
        "image_height": doc_rep.get("image_height"),
        "full_response": response_body or {},
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    
    record_json = json.dumps(_json_safe(record), indent=2).encode('utf-8')
    upload_to_s3(user_id, "documents", f"{doc_id}.json", record_json, "application/json")
    
    return doc_id


def list_documents(user_id, limit=200):
    """Summaries ordered newest-first. Never returns stored response bodies."""
    keys = list_from_s3(user_id, "documents")
    out = []
    
    for key in keys:
        if not key.endswith(".json"):
            continue
        try:
            file_id = key.split("/")[-1]
            meta_bytes = download_from_s3(user_id, "documents", file_id)
            if not meta_bytes:
                continue
                
            rec = json.loads(meta_bytes)
            out.append({
                "id": rec.get("id"),
                "original_filename": rec.get("original_filename"),
                "mime_type": rec.get("mime_type"),
                "document_type": rec.get("document_type"),
                "status": rec.get("status"),
                "image_width": rec.get("image_width"),
                "image_height": rec.get("image_height"),
                "created_at": rec.get("created_at"),
                "updated_at": rec.get("updated_at"),
            })
        except Exception as e:
            print(f"[STORAGE] skip {key}: {e}")
            
    out.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return out[:limit]


def get_document(user_id, doc_id):
    """Full record including the stored response body, or None."""
    meta_bytes = download_from_s3(user_id, "documents", f"{doc_id}.json")
    if not meta_bytes:
        return None
        
    return json.loads(meta_bytes)


def get_document_file(user_id, doc_id):
    """Bytes + mime type of the stored file, for serving to the UI."""
    record = get_document(user_id, doc_id)
    if not record or not record.get("file_path"):
        return None
        
    stored_name = record["file_path"]
    image_bytes = download_from_s3(user_id, "documents", stored_name)
    if not image_bytes:
        return None

    return {"mime_type": record["mime_type"], "image": image_bytes}


def delete_document(user_id, doc_id):
    """Delete a document row. Returns True when a row was actually removed.
    The stored file is also unlinked."""
    record = get_document(user_id, doc_id)
    if not record:
        return False
        
    stored_name = record.get("file_path")
    if stored_name:
        delete_from_s3(user_id, "documents", stored_name)
        
    delete_from_s3(user_id, "documents", f"{doc_id}.json")
    return True