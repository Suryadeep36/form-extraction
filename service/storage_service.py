"""
PostgreSQL persistence for uploaded documents and their extraction output.

Files are stored on disk under config.UPLOAD_DIR (e.g. `uploads/`) while the
database keeps one `documents` row per upload with everything needed to replay
the live viewer: metadata (filename, mime type, document type, status, image
dimensions) plus the complete /extract-document response body as JSONB.

All helpers return builtin types only (no psycopg objects) so the FastAPI
layer can serialize them directly.
"""

import os
import re
import uuid
import base64

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from util.config import DATABASE_URL, UPLOAD_DIR


def _connect():
    """Open a psycopg3 connection that returns dict-style rows."""
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)

DOCUMENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id                uuid PRIMARY KEY,
    original_filename text NOT NULL,
    file_path         text,
    mime_type         text,
    document_type     text,
    status            text NOT NULL DEFAULT 'processing',
    image_width       integer,
    image_height      integer,
    full_response     jsonb NOT NULL,
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_created_at_idx
    ON documents (created_at DESC);
"""


def storage_enabled() -> bool:
    """Whether persistence is usable (DATABASE_URL configured)."""
    return bool(DATABASE_URL)


def init_db() -> bool:
    """Create the documents table."""
    if not DATABASE_URL:
        print("[STORAGE] DATABASE_URL not set - document storage disabled.")
        return False
    try:
        with _connect() as conn:
            conn.execute(DOCUMENTS_SCHEMA)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        print("[STORAGE] Database ready.")
        return True
    except Exception as e:  # pragma: no cover - defensive startup path
        print(f"[STORAGE] init failed: {e}")
        return False


def _decode_data_url(data_url):
    """Decode a base64 data URL (e.g. 'data:image/jpeg;base64,....') to bytes."""
    if not data_url:
        raise ValueError("Missing processed image data URL")
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)


def _abs_path(file_path):
    """Resolve a stored file path (may be absolute or relative to CWD)."""
    if not file_path:
        return None
    if os.path.isabs(file_path):
        return file_path
    return os.path.normpath(os.path.abspath(file_path))


def _summarize(row) -> dict:
    return {
        "id": str(row["id"]),
        "original_filename": row["original_filename"],
        "mime_type": row["mime_type"],
        "document_type": row["document_type"],
        "status": row["status"],
        "image_width": row["image_width"],
        "image_height": row["image_height"],
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
    }


def save_document(filename, mime_type, image_data_url, doc_rep, response_body):
    """Persist a corrected document image + its extraction output.

    `image_data_url` is a base64 data URL of the perspective-corrected image
    (the coordinate system all overlays are normalized against). It is decoded
    and written to uploads/; a `documents` row stores the full extraction
    response. Returns the new document id.
    """
    data = (response_body or {}).get("data") or {}
    document_type = data.get("document_type") if isinstance(data, dict) else None

    doc_id = str(uuid.uuid4())
    ext = os.path.splitext(filename or "")[1]
    ext = re.sub(r"[^.\w]", "", ext).lower()[:12] or ".jpg"
    stored_name = f"{uuid.uuid4().hex}{ext}"
    rel_path = os.path.join(UPLOAD_DIR, stored_name)
    abs_path = os.path.abspath(rel_path)

    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "wb") as fh:
        fh.write(_decode_data_url(image_data_url))

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO documents (
                id, original_filename, file_path, mime_type, document_type,
                status, image_width, image_height, full_response
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(doc_id),
                filename,
                rel_path,
                mime_type,
                document_type,
                "completed",
                doc_rep.get("image_width"),
                doc_rep.get("image_height"),
                Jsonb(response_body or {}),
            ),
        )
    return str(doc_id)


def list_documents(limit=200):
    """Summaries ordered newest-first. Never returns stored response bodies."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, original_filename, mime_type, document_type, status,
                   image_width, image_height, created_at, updated_at
            FROM documents
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
    return [_summarize(row) for row in rows]


def get_document(doc_id):
    """Full record including the stored response body, or None."""
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, original_filename, mime_type, document_type, status,
                   image_width, image_height, full_response, created_at,
                   updated_at
            FROM documents
            WHERE id = %s
            """,
            (doc_id,),
        ).fetchone()

    if row is None:
        return None

    return {
        **_summarize(row),
        "full_response": row["full_response"],
    }


def get_document_file(doc_id):
    """Bytes + mime type of the stored file, for serving to the UI."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT file_path, mime_type FROM documents WHERE id = %s",
            (doc_id,),
        ).fetchone()

    if row is None:
        return None

    path = _abs_path(row["file_path"])
    if not path or not os.path.exists(path):
        return None

    with open(path, "rb") as fh:
        return {"mime_type": row["mime_type"], "image": fh.read()}


def delete_document(doc_id):
    """Delete a document row. Returns True when a row was actually removed.
    The stored file is also unlinked."""
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM documents WHERE id = %s RETURNING file_path",
            (doc_id,),
        )
        row = cur.fetchone()

    if row is None:
        return False

    path = _abs_path(row["file_path"])
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
    return True