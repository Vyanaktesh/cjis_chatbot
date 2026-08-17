"""
Thin psycopg2 connection helper shared by every repository module.

Deliberately not an ORM: Phase 1-2 stay on plain SQL so the schema (in
db/migrations/) remains the single source of truth and every query is easy
to read/audit — appropriate for a project that may run on tightly
controlled government infrastructure.
"""

from contextlib import contextmanager

import psycopg2
import psycopg2.extras

from app.core.config import get_settings


@contextmanager
def get_conn():
    """Yields a psycopg2 connection; commits on success, rolls back on error."""
    settings = get_settings()
    conn = psycopg2.connect(settings.postgres_dsn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def get_dict_cursor(conn):
    """Yields a cursor that returns rows as dicts instead of tuples."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield cur
    finally:
        cur.close()
