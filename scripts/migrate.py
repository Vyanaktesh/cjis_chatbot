#!/usr/bin/env python3
"""
Minimal, dependency-light migration runner.

Applies every .sql file in db/migrations/ (in filename order) that hasn't
been applied yet, tracking progress in a `schema_migrations` table. Kept
intentionally simple for Phase 1 — if the project outgrows this, swapping
in Alembic later is a drop-in replacement since migrations already live as
plain, ordered SQL files.

Usage:
    python scripts/migrate.py
"""

import sys
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "db" / "migrations"


def ensure_migrations_table(cur):
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            filename    TEXT PRIMARY KEY,
            applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )


def already_applied(cur) -> set[str]:
    cur.execute("SELECT filename FROM schema_migrations;")
    return {row[0] for row in cur.fetchall()}


def main() -> int:
    settings = get_settings()

    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not migration_files:
        print(f"No migration files found in {MIGRATIONS_DIR}")
        return 1

    try:
        conn = psycopg2.connect(settings.postgres_dsn)
    except Exception as exc:
        print(f"FAILED to connect to Postgres: {exc}")
        return 1

    conn.autocommit = False
    applied_count = 0
    try:
        with conn.cursor() as cur:
            ensure_migrations_table(cur)
            conn.commit()

            applied = already_applied(cur)

            for path in migration_files:
                if path.name in applied:
                    print(f"skip   {path.name} (already applied)")
                    continue

                sql = path.read_text()
                print(f"apply  {path.name} ...")
                try:
                    cur.execute(sql)
                    cur.execute(
                        "INSERT INTO schema_migrations (filename) VALUES (%s);",
                        (path.name,),
                    )
                    conn.commit()
                    applied_count += 1
                    print(f"OK     {path.name}")
                except Exception as exc:
                    conn.rollback()
                    print(f"FAILED {path.name}: {exc}")
                    return 1
    finally:
        conn.close()

    print(f"\nDone. {applied_count} migration(s) applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
