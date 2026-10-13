#!/usr/bin/env python3
"""
Take one source out of the knowledge base, reversibly.

For the source with the given URL this:
  1. marks the source inactive, so fetch_all_sources / embed_and_index_all
     skip it from now on, and
  2. rejects every one of its chunks that isn't already superseded, through
     the same app.review.service.decide_chunk() kb_admin uses -- so Postgres
     and Qdrant both get review_status='rejected' and each decision is
     audit-logged with the actor and reason.

Nothing is deleted. Retrieval only returns approved chunks, so a rejected
chunk stops being used immediately; re-approving it in kb_admin (and setting
the source active again) brings it back.

Also remove the URL from db/seed/source_registry.json, otherwise the next
load_source_registry.py run would mark the source active again.

Usage:
    python scripts/retire_source.py --url https://example.gov.in/page --reason "why" --dry-run
    python scripts/retire_source.py --url https://example.gov.in/page --reason "why"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.connection import get_conn  # noqa: E402
from app.db.sources_repo import set_active  # noqa: E402
from app.review.service import ChunkNotFound, decide_chunk  # noqa: E402
from app.vectorstore.qdrant_store import get_qdrant_client  # noqa: E402

ACTOR = "retire_source"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="Exact source URL, as stored in sources.url.")
    parser.add_argument("--reason", required=True, help="Recorded in the audit log for every rejected chunk.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change, but change nothing.")
    args = parser.parse_args()

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, active FROM sources WHERE url = %s;", (args.url,))
            row = cur.fetchone()
            if row is None:
                print(f"No source with url {args.url!r}.")
                return 1
            source_id, active = row[0], row[1]
            cur.execute(
                "SELECT id, chunk_index, review_status FROM chunks "
                "WHERE source_id = %s AND review_status <> 'superseded' ORDER BY chunk_index;",
                (str(source_id),),
            )
            chunks = cur.fetchall()

    print(f"Source {source_id}  active={active}")
    print(f"{len(chunks)} chunk(s) to reject:")
    for chunk_id, index, status in chunks:
        print(f"  #{index}  {chunk_id}  {status}")

    if args.dry_run:
        print("Dry run: no changes made.")
        return 0

    with get_conn() as conn:
        set_active(conn, source_id, False)
    print("Source marked inactive.")

    qc = get_qdrant_client()
    rejected = failed = 0
    for chunk_id, index, _status in chunks:
        try:
            # One transaction per chunk, as in bulk_approve_demo.py: a failure
            # loses only that chunk, and Postgres/Qdrant roll back together.
            with get_conn() as conn:
                decide_chunk(conn, qc, chunk_id, "rejected", actor=ACTOR, reason=args.reason)
            rejected += 1
        except ChunkNotFound:
            failed += 1
            print(f"  skip #{index} {chunk_id}: superseded or gone")
        except Exception as exc:  # noqa: BLE001 -- report and keep going
            failed += 1
            print(f"  FAILED #{index} {chunk_id}: {exc}")

    print(f"Done. Rejected {rejected}, skipped/failed {failed}.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
