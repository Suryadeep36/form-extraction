"""
The storage switch.

Every blob in the app (documents, templates, filled forms) goes through the four
functions below. Which backend actually runs is decided by one value,
`STORAGE_MODE`, so switching between cloud and local is a single env change
with no other code touched:

    STORAGE_MODE=cloud   -> S3 via boto3 (service/s3_service.py)
    STORAGE_MODE=local   -> files on disk (service/local_blob_store.py)

Both backends implement the same interface and the same `(user_id, prefix,
file_id)` namespacing, and `list_blobs` returns keys in the same shape S3 does,
so callers behave identically in either mode.

Backends are imported lazily: local mode never needs boto3, and cloud mode
never needs a writable local tree.
"""

import os

from util.config import STORAGE_MODE

CLOUD = "cloud"
LOCAL = "local"

# Programmatic override, set by set_mode(). None means "use the env value".
_override = None


class StorageModeError(RuntimeError):
    """Raised when STORAGE_MODE is not one of the supported values."""


def _normalize(mode):
    value = (mode or "").strip().lower()
    if value not in (CLOUD, LOCAL):
        raise StorageModeError(
            f"STORAGE_MODE must be '{CLOUD}' or '{LOCAL}', got {mode!r}"
        )
    return value


def get_mode() -> str:
    """The active mode: the override if set, else the configured value."""
    return _override or _normalize(STORAGE_MODE)


def set_mode(mode):
    """Switch backends at runtime. Pass None to fall back to the env value."""
    global _override
    _override = _normalize(mode) if mode is not None else None
    return get_mode()


def _backend():
    """Return the module implementing the active mode."""
    if get_mode() == LOCAL:
        from service import local_blob_store
        return local_blob_store
    from service import s3_service
    return s3_service


# ---------------------------------------------------------------------------
# Backend operations
# ---------------------------------------------------------------------------

def put_blob(user_id, prefix, file_id, file_content, content_type="application/octet-stream"):
    """Store bytes under `{user_id}/{prefix}/{file_id}`."""
    if get_mode() == LOCAL:
        return _backend().put_blob(user_id, prefix, file_id, file_content, content_type)
    from service.s3_service import upload_to_s3
    return upload_to_s3(user_id, prefix, file_id, file_content, content_type)


def get_blob(user_id, prefix, file_id):
    """Fetch bytes, or b"" when absent."""
    if get_mode() == LOCAL:
        return _backend().get_blob(user_id, prefix, file_id)
    from service.s3_service import download_from_s3
    return download_from_s3(user_id, prefix, file_id)


def delete_blob(user_id, prefix, file_id):
    """Delete one file. False when it was not there."""
    if get_mode() == LOCAL:
        return _backend().delete_blob(user_id, prefix, file_id)
    from service.s3_service import delete_from_s3
    return delete_from_s3(user_id, prefix, file_id)


def list_blobs(user_id, prefix):
    """Keys under `{user_id}/{prefix}/`."""
    if get_mode() == LOCAL:
        return _backend().list_blobs(user_id, prefix)
    from service.s3_service import list_from_s3
    return list_from_s3(user_id, prefix)


def describe() -> dict:
    """Storage wiring for the health check, so you can confirm the switch."""
    mode = get_mode()
    if mode == LOCAL:
        from service import local_blob_store
        blobs = local_blob_store.describe()
    else:
        from service.s3_service import AWS_REGION, AWS_S3_BUCKET
        blobs = {
            "backend": "aws-s3",
            "bucket": AWS_S3_BUCKET or None,
            "region": AWS_REGION,
        }

    from service import db_service
    return {
        "mode": mode,
        "blobs": blobs,
        "database": db_service.describe(),
    }


def log_status() -> str:
    """One-line startup banner describing where things are being written."""
    info = describe()
    db = info["database"]
    db_text = db.get("backend") or "disabled"
    if db.get("location"):
        db_text = f"{db_text} ({db['location']})"
    return f"[STORAGE] mode={info['mode']} blobs={info['blobs']['backend']} db={db_text}"
