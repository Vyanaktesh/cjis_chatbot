"""A pooled connection the server has dropped (database restart, idle timeout)
must not fail the next request. Needs a reachable Postgres; skipped otherwise.
Only the test's own connection is terminated."""
import os

import pytest

psycopg2 = pytest.importorskip("psycopg2")


@pytest.fixture()
def pool_env():
    params = dict(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        user=os.environ.get("POSTGRES_USER", "rag_admin"),
        password=os.environ.get("POSTGRES_PASSWORD", "change_me_dev_only"),
    )
    try:
        admin = psycopg2.connect(dbname="postgres", connect_timeout=3, **params)
    except Exception as exc:
        pytest.skip(f"Postgres not reachable: {exc}")
    admin.autocommit = True

    previous = os.environ.get("POSTGRES_DB")
    os.environ["POSTGRES_DB"] = "postgres"
    from app.core.config import get_settings

    get_settings.cache_clear()
    import app.db.connection as dbc

    dbc._pool = None
    yield admin, dbc
    if dbc._pool is not None:
        dbc._pool.closeall()
        dbc._pool = None
    if previous is None:
        os.environ.pop("POSTGRES_DB", None)
    else:
        os.environ["POSTGRES_DB"] = previous
    get_settings.cache_clear()
    admin.close()


def test_a_connection_dropped_by_the_server_is_replaced_transparently(pool_env):
    admin, dbc = pool_env

    with dbc.get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT pg_backend_pid();")
        victim_pid = cur.fetchone()[0]

    # Simulate a database restart / dropped connection: kill that pooled session server-side.
    with admin.cursor() as cur:
        cur.execute("SELECT pg_terminate_backend(%s);", (victim_pid,))

    with dbc.get_conn() as conn, conn.cursor() as cur:  # used to raise OperationalError
        cur.execute("SELECT 1;")
        assert cur.fetchone()[0] == 1
