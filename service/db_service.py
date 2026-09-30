"""
Relational database access, driven by the same STORAGE_MODE switch.

    STORAGE_MODE=cloud -> no DB. Documents are stored as JSON objects in S3 by
                          service/storage_service.py, exactly as before.
    STORAGE_MODE=local -> a local SQL database, selected by LOCAL_DATABASE_URL:

        sqlite:///data/form_extract.db   zero-dependency default
        postgresql://user:pass@host/db  local Postgres

`connect()` yields a connection whose `execute()` works the same on both
backends and returns a cursor with `fetchone()` / `fetchall()`. Parameter style
is `%s` everywhere; it is translated to sqlite's `?` internally, so the SQL in
service/storage_service.py is written once.
"""

import os
import sqlite3
from contextlib import contextmanager

from util.config import LOCAL_DATABASE_URL, resolve_sqlite_path
from service.blob_store import LOCAL, get_mode

SQLITE = "sqlite"
POSTGRES = "postgresql"
NONE = "none"


def _dsn():
    return (LOCAL_DATABASE_URL or "").strip()


def backend() -> str:
    """Which SQL backend the configured DSN selects, or NONE when unset."""
    if get_mode() != LOCAL:
        return NONE
    dsn = _dsn()
    if dsn.startswith("sqlite:"):
        return SQLITE
    if dsn.startswith(("postgres://", "postgresql://")):
        return POSTGRES
    return NONE


def _sqlite_path() -> str:
    """Filesystem path from a `sqlite:///relative/path.db` style DSN."""
    _, _, tail = _dsn().partition("sqlite:")
    # Drop the authority section: "sqlite:///data/x.db" -> "/data/x.db".
    path = tail[3:] if tail.startswith("///") else tail.lstrip("/")
    if not path:
        path = "data/form_extract.db"
    return resolve_sqlite_path(path)


def enabled() -> bool:
    """Whether a usable database is configured for the active mode."""
    return backend() != NONE


def init() -> bool:
    """Prepare the database (create the file / verify connectivity)."""
    target = backend()
    if target == NONE:
        print("[DB] No database configured for this mode (cloud mode is S3-only).")
        return False

    try:
        if target == SQLITE:
            path = _sqlite_path()
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with connect() as conn:
                conn.execute("SELECT 1")
            print(f"[DB] SQLite ready at {path}")
            return True

        with connect() as conn:
            conn.execute("SELECT 1")
        print("[DB] PostgreSQL ready.")
        return True
    except Exception as e:
        print(f"[DB] init failed: {e}")
        return False


@contextmanager
def connect():
    """Yield a connection with a portable `execute(sql, params)` API."""
    target = backend()

    if target == SQLITE:
        conn = sqlite3.connect(_sqlite_path())
        try:
            conn.row_factory = sqlite3.Row
            yield _SQLiteConn(conn)
            conn.commit()
        finally:
            conn.close()
        return

    if target == POSTGRES:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as e:
            raise RuntimeError(
                "LOCAL_DATABASE_URL points at PostgreSQL but psycopg is not "
                "installed. Run `pip install psycopg` or set "
                "LOCAL_DATABASE_URL=sqlite:///data/form_extract.db"
            ) from e

        with psycopg.connect(_dsn(), row_factory=dict_row) as conn:
            yield conn
            conn.commit()
        return

    raise RuntimeError(
        "No database configured. Set STORAGE_MODE=local with a "
        "LOCAL_DATABASE_URL, or STORAGE_MODE=cloud to use S3."
    )


class _SQLiteConn:
    """Adapter giving sqlite3 a psycopg-shaped execute()."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, params=()):
        if params is None:
            params = ()
        elif not isinstance(params, (list, tuple)):
            params = (params,)
        return self._conn.execute(_to_sqlite(sql), params)


def _to_sqlite(sql: str) -> str:
    """Rewrite `%s` placeholders to sqlite's `?`.

    Only ever applied to SQL written inside this codebase, which uses `%s` and
    no literal percent signs, so a plain replace is safe here.
    """
    return sql.replace("%s", "?")


def describe() -> dict:
    """Database wiring for the health check."""
    target = backend()
    if target == SQLITE:
        location = _sqlite_path()
    elif backend == POSTGRES:
        location = _dsn().split("@", 1)[-1]
    else:
        location = None

    return {
        "backend": target,
        "location": location,
        "enabled": target != NONE,
    }
