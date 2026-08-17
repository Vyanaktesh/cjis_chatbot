#!/usr/bin/env python3
"""
Phase 1 verification script.

Connects to both Postgres and Qdrant using the same config the rest of the
app will use, runs a trivial round-trip against each, and prints a clear
OK/FAIL per service. Exits non-zero if anything fails, so it can be used
in CI or a pre-flight check later.

Usage:
    python scripts/health_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402


def check_postgres(settings) -> bool:
    import psycopg2

    try:
        conn = psycopg2.connect(settings.postgres_dsn, connect_timeout=5)
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()

            # Report which of our Phase 1 tables already exist, if migrations
            # have been run — purely informational, not a failure condition.
            cur.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name;
                """
            )
            tables = [row[0] for row in cur.fetchall()]
        conn.close()
        print(f"[Postgres] OK - connected to '{settings.postgres_db}' "
              f"at {settings.postgres_host}:{settings.postgres_port}")
        if tables:
            print(f"[Postgres] tables present: {', '.join(tables)}")
        else:
            print("[Postgres] no tables yet — run scripts/migrate.py")
        return True
    except Exception as exc:
        print(f"[Postgres] FAIL - {exc}")
        return False


def check_qdrant(settings) -> bool:
    from qdrant_client import QdrantClient

    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_http_port,
            grpc_port=settings.qdrant_grpc_port,
            api_key=settings.qdrant_api_key or None,
            timeout=5,
        )
        collections = client.get_collections()
        print(f"[Qdrant]   OK - connected at "
              f"{settings.qdrant_host}:{settings.qdrant_http_port}")
        names = [c.name for c in collections.collections]
        print(f"[Qdrant]   collections present: {names or '(none yet)'}")
        return True
    except Exception as exc:
        print(f"[Qdrant]   FAIL - {exc}")
        return False


def main() -> int:
    settings = get_settings()
    print(f"Environment: {settings.app_env}\n")

    pg_ok = check_postgres(settings)
    qdrant_ok = check_qdrant(settings)

    print()
    if pg_ok and qdrant_ok:
        print("ALL OK")
        return 0

    print("ONE OR MORE CHECKS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
