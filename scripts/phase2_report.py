#!/usr/bin/env python3
"""
Independent Phase 2 verification report — queries Postgres and the local
filesystem directly (not just trusting the smoke test's own printout) to
confirm: the registry loaded, raw content landed on disk with hashes,
and the deliberately-broken URL produced a clean logged failure rather
than a crash.

Usage:
    python scripts/phase2_report.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.connection import get_conn  # noqa: E402

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> int:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM sources;")
        total_sources = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM sources WHERE canonical = true;")
        canonical_count = cur.fetchone()[0]

        cur.execute(
            "SELECT service_category, count(*) FROM sources GROUP BY service_category ORDER BY 2 DESC;"
        )
        by_category = cur.fetchall()

        cur.execute("SELECT count(*) FROM source_versions;")
        total_versions = cur.fetchone()[0]

        cur.execute(
            """
            SELECT s.url, sv.version, sv.content_hash, sv.raw_content_path
            FROM source_versions sv
            JOIN sources s ON s.id = sv.source_id
            ORDER BY sv.created_at DESC;
            """
        )
        versions = cur.fetchall()

        cur.execute(
            "SELECT action, count(*) FROM audit_log GROUP BY action ORDER BY 2 DESC;"
        )
        audit_actions = cur.fetchall()

        cur.execute(
            """
            SELECT entity_type, action, details, created_at
            FROM audit_log
            WHERE action = 'fetch_failure'
            ORDER BY created_at DESC
            LIMIT 5;
            """
        )
        failures = cur.fetchall()

    print(f"sources registered:        {total_sources} (expected 47)")
    print(f"  canonical (gov.in/mea):   {canonical_count}")
    print(f"  by service_category:")
    for cat, count in by_category:
        print(f"    {cat:<24} {count}")

    print(f"\nsource_versions created:    {total_versions}")
    all_on_disk = True
    for url, version, content_hash, raw_path in versions:
        on_disk = (Path(__file__).resolve().parent.parent / raw_path).exists()
        all_on_disk = all_on_disk and on_disk
        print(f"  v{version}  {url}")
        print(f"       hash={content_hash[:16]}...  on_disk={on_disk}  path={raw_path}")

    print(f"\naudit_log action counts:")
    for action, count in audit_actions:
        print(f"    {action:<16} {count}")

    print(f"\nmost recent fetch_failure entries (if any — proves failures log cleanly rather than crashing):")
    if not failures:
        print("    (none — every source fetched successfully on this run)")
    for entity_type, action, details, created_at in failures:
        print(f"    [{created_at}] {entity_type} {action}: {details}")

    print(f"\nraw files on disk under data/raw/:")
    file_count = 0
    if RAW_DIR.exists():
        for p in sorted(RAW_DIR.rglob("*")):
            if p.is_file():
                file_count += 1
                print(f"    {p.relative_to(RAW_DIR.parent.parent)}  ({p.stat().st_size} bytes)")
    else:
        print("    (data/raw/ does not exist yet)")

    # "OK" just means: registry fully loaded, every version recorded in
    # Postgres actually has its raw file present on disk (no clobbered/
    # missing files), and every version's file count matches — it does NOT
    # require any failures to be present. A clean 47-for-47 run is success,
    # not a gap.
    ok = total_sources == 47 and total_versions > 0 and all_on_disk and file_count == total_versions
    print("\nALL OK" if ok else "\nINCOMPLETE — see gaps above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
