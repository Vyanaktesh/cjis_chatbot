"""
Thin psycopg2 connection helper shared by every repository module.

Deliberately not an ORM: Phase 1-2 stay on plain SQL so the schema (in
db/migrations/) remains the single source of truth and every query is easy
to read/audit — appropriate for a project that may run on tightly
controlled government infrastructure.

Connections come from a lazily-created, thread-safe pool: a FastAPI service
runs sync endpoints in a worker threadpool, so a new connect() per request
would exhaust Postgres `max_connections` and add auth latency under load.
The pool caps concurrent connections and reuses them. `connect_timeout`
bounds how long a dead DB can block a request; a server-side
`statement_timeout` bounds a runaway query.
"""

import threading
import time
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from psycopg2.pool import PoolError, ThreadedConnectionPool

from app.core.config import get_settings

_pool: ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:  # double-checked under the lock
                settings = get_settings()
                _pool = ThreadedConnectionPool(
                    minconn=settings.postgres_pool_min,
                    maxconn=settings.postgres_pool_max,
                    dsn=settings.postgres_dsn,
                    connect_timeout=settings.postgres_connect_timeout,
                    options=f"-c statement_timeout={int(settings.postgres_statement_timeout_ms)}",
                )
    return _pool


def _borrow(pool: ThreadedConnectionPool) -> "psycopg2.extensions.connection":
    """psycopg2's pool raises PoolError immediately when every connection is
    checked out. Retry within a bounded window so a brief burst above the
    pool size waits for a connection to free up rather than failing hard."""
    timeout = get_settings().postgres_pool_acquire_timeout
    deadline = time.monotonic() + timeout
    while True:
        try:
            return pool.getconn()
        except PoolError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


@contextmanager
def get_conn():
    """Borrows a pooled psycopg2 connection; commits on success, rolls back
    on error, and always returns the connection to the pool. A connection
    whose rollback fails is treated as broken and closed rather than
    returned, so a poisoned connection can't be handed to the next caller."""
    pool = _get_pool()
    conn = _borrow(pool)
    broken = False
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            broken = True
        raise
    finally:
        pool.putconn(conn, close=broken)


@contextmanager
def get_dict_cursor(conn):
    """Yields a cursor that returns rows as dicts instead of tuples."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield cur
    finally:
        cur.close()
