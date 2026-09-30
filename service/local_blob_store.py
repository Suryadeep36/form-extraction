"""
Filesystem backend for the blob store, used when STORAGE_MODE=local.

Mirrors the four operations exposed by `service.s3_service` so the two
backends are interchangeable:

    put_blob(user_id, prefix, file_id, content, content_type) -> bool
    get_blob(user_id, prefix, file_id)                       -> bytes
    delete_blob(user_id, prefix, file_id)                    -> bool
    list_blobs(user_id, prefix)                              -> list[str]

Layout, matching the pre-cloud file-backed version so existing data under
`uploads/` stays readable:

    <LOCAL_STORAGE_DIR>/<prefix>/<file_id>

With LOCAL_STORAGE_USER_SCOPE=true each user gets its own subtree, which
matches the S3 key layout:

    <LOCAL_STORAGE_DIR>/<user_id>/<prefix>/<file_id>

`list_blobs` returns keys in the same shape S3 returns (relative to the
user root), so callers can keep doing `key.split("/")[-1]` in either mode.
"""

import os
import re

from util.config import LOCAL_STORAGE_DIR, LOCAL_STORAGE_USER_SCOPE, local_dir

# Segments are used to build real paths here (unlike S3 keys), so anything
# that could escape the storage root is rejected outright.
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MAX_SEGMENT = 128


class UnsafePathError(ValueError):
    """Raised when a user/prefix/file id cannot be used as a path segment."""


def _safe_segment(value, label):
    """Validate one path segment, or raise. Guards against traversal."""
    if not isinstance(value, str) or not value:
        raise UnsafePathError(f"Invalid {label}: must be a non-empty string")
    if len(value) > _MAX_SEGMENT:
        raise UnsafePathError(f"Invalid {label}: longer than {_MAX_SEGMENT} chars")
    if not _SAFE_SEGMENT.match(value):
        raise UnsafePathError(f"Invalid {label}: {value!r}")
    return value


def _prefix(user_id, prefix):
    """Directory holding `prefix` for this user, created on demand."""
    _safe_segment(prefix, "prefix")
    if LOCAL_STORAGE_USER_SCOPE:
        return local_dir(_safe_segment(user_id, "user_id"), prefix)
    return local_dir(prefix)


def _path(user_id, prefix, file_id):
    return os.path.join(_prefix(user_id, prefix), _safe_segment(file_id, "file_id"))


def _key(user_id, prefix, file_id):
    """Key in the same shape S3 hands back."""
    if LOCAL_STORAGE_USER_SCOPE:
        return f"{user_id}/{prefix}/{file_id}"
    return f"{prefix}/{file_id}"


def put_blob(user_id, prefix, file_id, file_content, content_type="application/octet-stream"):
    """Write bytes to disk. Content type is ignored (no metadata sidecar)."""
    try:
        path = _path(user_id, prefix, file_id)
    except UnsafePathError as e:
        print(f"[LOCAL-STORE] put rejected: {e}")
        return False

    try:
        with open(path, "wb") as fh:
            fh.write(file_content or b"")
        return True
    except OSError as e:
        print(f"[LOCAL-STORE] write failed: {e}")
        return False


def get_blob(user_id, prefix, file_id):
    """Read bytes back, or b"" when absent/unreadable."""
    try:
        path = _path(user_id, prefix, file_id)
    except UnsafePathError as e:
        print(f"[LOCAL-STORE] get rejected: {e}")
        return b""

    try:
        with open(path, "rb") as fh:
            return fh.read()
    except FileNotFoundError:
        return b""
    except OSError as e:
        print(f"[LOCAL-STORE] read failed: {e}")
        return b""


def delete_blob(user_id, prefix, file_id):
    """Delete one file. Returns False when it was not there."""
    try:
        path = _path(user_id, prefix, file_id)
    except UnsafePathError as e:
        print(f"[LOCAL-STORE] delete rejected: {e}")
        return False

    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False
    except OSError as e:
        print(f"[LOCAL-STORE] delete failed: {e}")
        return False


def list_blobs(user_id, prefix):
    """Keys for every file in this user's `prefix`, sorted by name."""
    try:
        directory = _prefix(user_id, prefix)
    except UnsafePathError as e:
        print(f"[LOCAL-STORE] list rejected: {e}")
        return []

    try:
        names = sorted(os.listdir(directory))
    except OSError as e:
        print(f"[LOCAL-STORE] list failed: {e}")
        return []

    return [_key(user_id, prefix, name) for name in names if os.path.isfile(os.path.join(directory, name))]


def describe():
    """Human-readable backend description for the health check."""
    return {
        "backend": "local-filesystem",
        "root": os.path.abspath(LOCAL_STORAGE_DIR),
        "user_scoped": LOCAL_STORAGE_USER_SCOPE,
    }
