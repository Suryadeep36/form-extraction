"""
Persistence for uploaded unstructured documents and their extraction output.

Which backend runs is decided by the single `STORAGE_MODE` switch (see
service/blob_store.py):

    cloud -> S3 only. `{doc_id}.json` holds the metadata + full response and
             `{doc_id}_img.jpg` holds the perspective-corrected image. No
             relational database is used at all.

    local -> a local SQL database (sqlite or PostgreSQL via
             LOCAL_DATABASE_URL) holds one `documents` row per upload, and the
             corrected image is written to LOCAL_STORAGE_DIR/documents/.

Both modes store the same record shape and every helper returns builtin types
only, so the FastAPI layer can serialize them directly.
"""

import json
import uuid
import base64
from datetime import datetime

from service.blob_store import LOCAL, get_mode, put_blob, get_blob, delete_blob, list_blobs
from service import db_service

DOCUMENTS_PREFIX = "documents"

# `created_at`/`updated_at` are ISO-8601 UTC strings rather than timestamp
# columns: they sort lexicographically, so ORDER BY works identically on sqlite
# and PostgreSQL without either backend needing a cast.
DOCUMENTS_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS documents (
    id                TEXT PRIMARY KEY,
    user_id           TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    file_path         TEXT,
    mime_type         TEXT,
    document_type     TEXT,
    status            TEXT NOT NULL DEFAULT 'processing',
    image_width       INTEGER,
    image_height      INTEGER,
    full_response     TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS documents_user_created_idx
    ON documents (user_id, created_at DESC);
"""

DOCUMENTS_SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS documents (
    id                uuid PRIMARY KEY,
    user_id           text NOT NULL,
    original_filename text NOT NULL,
    file_path         text,
    mime_type         text,
    document_type     text,
    status            text NOT NULL DEFAULT 'processing',
    image_width       integer,
    image_height      integer,
    full_response     jsonb NOT NULL,
    created_at        text NOT NULL,
    updated_at        text NOT NULL
);

CREATE INDEX IF NOT EXISTS documents_user_created_idx
    ON documents (user_id, created_at DESC);
"""

_SUMMARY_COLUMNS = (
    "id, original_filename, mime_type, document_type, status, "
    "image_width, image_height, created_at, updated_at"
)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def init_db() -> bool:
    """Prepare persistence for the active mode."""
    if get_mode() == LOCAL:
        if not db_service.init():
            return False
        return _create_documents_table()
    print("[STORAGE] Using S3 for all document storage. No DB needed.")
    return True


def storage_enabled() -> bool:
    """Whether document persistence is usable in the active mode."""
    if get_mode() == LOCAL:
        return db_service.enabled()
    return True


def _create_documents_table() -> bool:
    schema = (
        DOCUMENTS_SCHEMA_POSTGRES
        if db_service.backend() == db_service.POSTGRES
        else DOCUMENTS_SCHEMA_SQLITE
    )
    try:
        with db_service.connect() as conn:
            for statement in filter(None, (s.strip() for s in schema.split(";"))):
                conn.execute(statement)
        return True
    except Exception as e:
        print(f"[STORAGE] schema creation failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


def _as_dict(row):
    """Normalize a DB row (dict from psycopg, sqlite3.Row) to a plain dict."""
    if row is None:
        return None
    if isinstance(row, dict):
        return dict(row)
    return {key: row[key] for key in row.keys()}


def _as_json(value):
    """`full_response` comes back parsed on jsonb and as text on sqlite."""
    if value is None or isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


# ---------------------------------------------------------------------------
# Cloud (S3) implementations - unchanged behaviour
# ---------------------------------------------------------------------------

def _save_document_s3(user_id, filename, mime_type, image_data_url, doc_rep, response_body):
    data = (response_body or {}).get("data") or {}
    document_type = data.get("document_type") if isinstance(data, dict) else None

    doc_id = str(uuid.uuid4())
    stored_img_name = f"{doc_id}_img.jpg"

    image_bytes = _decode_data_url(image_data_url)
    put_blob(user_id, DOCUMENTS_PREFIX, stored_img_name, image_bytes, mime_type)

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
        "updated_at": datetime.utcnow().isoformat(),
    }

    record_json = json.dumps(_json_safe(record), indent=2).encode("utf-8")
    put_blob(user_id, DOCUMENTS_PREFIX, f"{doc_id}.json", record_json, "application/json")

    return doc_id


def _list_documents_s3(user_id, limit=200):
    keys = list_blobs(user_id, DOCUMENTS_PREFIX)
    out = []

    for key in keys:
        if not key.endswith(".json"):
            continue
        try:
            file_id = key.split("/")[-1]
            meta_bytes = get_blob(user_id, DOCUMENTS_PREFIX, file_id)
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


# ---------------------------------------------------------------------------
# Local (SQL database + disk) implementations
# ---------------------------------------------------------------------------

def _save_document_local(user_id, filename, mime_type, image_data_url, doc_rep, response_body):
    data = (response_body or {}).get("data") or {}
    document_type = data.get("document_type") if isinstance(data, dict) else None

    doc_id = str(uuid.uuid4())
    stored_img_name = f"{doc_id}_img.jpg"

    image_bytes = _decode_data_url(image_data_url)
    put_blob(user_id, DOCUMENTS_PREFIX, stored_img_name, image_bytes, mime_type)

    now = datetime.utcnow().isoformat()
    full_response = _json_safe(response_body or {})

    with db_service.connect() as conn:
        conn.execute(
            """
            INSERT INTO documents (
                id, user_id, original_filename, file_path, mime_type,
                document_type, status, image_width, image_height,
                full_response, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                doc_id,
                user_id,
                filename,
                stored_img_name,
                mime_type,
                document_type,
                "completed",
                doc_rep.get("image_width"),
                doc_rep.get("image_height"),
                json.dumps(full_response),
                now,
                now,
            ),
        )

    return doc_id


def _list_documents_local(user_id, limit=200):
    with db_service.connect() as conn:
        rows = conn.execute(
            f"""
            SELECT {_SUMMARY_COLUMNS}
            FROM documents
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        ).fetchall()

    return [_as_dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_document(user_id, filename, mime_type, image_data_url, doc_rep, response_body):
    """Persist a corrected document image + its extraction output.

    `image_data_url` is a base64 data URL of the perspective-corrected image
    (the coordinate system all overlays are normalized against). Returns the
    new document id.
    """
    if get_mode() == LOCAL:
        return _save_document_local(
            user_id, filename, mime_type, image_data_url, doc_rep, response_body
        )
    return _save_document_s3(
        user_id, filename, mime_type, image_data_url, doc_rep, response_body
    )


def list_documents(user_id, limit=200):
    """Summaries ordered newest-first. Never returns stored response bodies."""
    if get_mode() == LOCAL:
        return _list_documents_local(user_id, limit)
    return _list_documents_s3(user_id, limit)


def get_document(user_id, doc_id):
    """Full record including the stored response body, or None."""
    if get_mode() == LOCAL:
        with db_service.connect() as conn:
            row = conn.execute(
                f"""
                SELECT {_SUMMARY_COLUMNS}, file_path, full_response
                FROM documents
                WHERE user_id = %s AND id = %s
                """,
                (user_id, doc_id),
            ).fetchone()
        record = _as_dict(row)
        if record is None:
            return None
        record["full_response"] = _as_json(record["full_response"])
        return record

    meta_bytes = get_blob(user_id, DOCUMENTS_PREFIX, f"{doc_id}.json")
    if not meta_bytes:
        return None
    return json.loads(meta_bytes)


def get_document_file(user_id, doc_id):
    """Bytes + mime type of the stored file, for serving to the UI."""
    if get_mode() == LOCAL:
        with db_service.connect() as conn:
            row = conn.execute(
                "SELECT file_path, mime_type FROM documents "
                "WHERE user_id = %s AND id = %s",
                (user_id, doc_id),
            ).fetchone()
        record = _as_dict(row)
        if not record or not record.get("file_path"):
            return None

        image_bytes = get_blob(user_id, DOCUMENTS_PREFIX, record["file_path"])
        if not image_bytes:
            return None
        return {"mime_type": record["mime_type"], "image": image_bytes}

    record = get_document(user_id, doc_id)
    if not record or not record.get("file_path"):
        return None

    image_bytes = get_blob(user_id, DOCUMENTS_PREFIX, record["file_path"])
    if not image_bytes:
        return None

    return {"mime_type": record["mime_type"], "image": image_bytes}


def delete_document(user_id, doc_id):
    """Delete a document row. Returns True when a row was actually removed.
    The stored file is also removed."""
    if get_mode() == LOCAL:
        with db_service.connect() as conn:
            row = conn.execute(
                "SELECT file_path FROM documents WHERE user_id = %s AND id = %s",
                (user_id, doc_id),
            ).fetchone()
            record = _as_dict(row)
            if record is None:
                return False

            conn.execute(
                "DELETE FROM documents WHERE user_id = %s AND id = %s",
                (user_id, doc_id),
            )

        if record.get("file_path"):
            delete_blob(user_id, DOCUMENTS_PREFIX, record["file_path"])
        return True

    record = get_document(user_id, doc_id)
    if not record:
        return False

    stored_name = record.get("file_path")
    if stored_name:
        delete_blob(user_id, DOCUMENTS_PREFIX, stored_name)

    delete_blob(user_id, DOCUMENTS_PREFIX, f"{doc_id}.json")
    return True
