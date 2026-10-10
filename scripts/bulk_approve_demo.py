#!/usr/bin/env python3
"""
DEMO SHORTCUT -- bulk-approve every pending chunk so the chatbot has
answerable content without standing up the separate kb_admin review service.

This is NOT editorial review. It approves everything currently in
review_status='pending_review', bypassing the human read-and-decide step that
kb_admin exists for. Use it only to seed a demo; never treat its output as
reviewed content.

It calls the exact same app.review.service.decide_chunk() that kb_admin calls,
so Postgres and Qdrant stay in sync (status written to both, every decision
audit-logged) just as a real approval would.

Each chunk is approved in its OWN short-lived get_conn() transaction. get_conn()
commits only at the end of its `with` block, so one giant transaction over the
whole batch would roll back every prior approval too if any single chunk failed
partway through. Per-chunk transactions mean a failure loses only that chunk.

Usage:
    python scripts/bulk_approve_demo.py
    python scripts/bulk_approve_demo.py --service-category passport
    python scripts/bulk_approve_demo.py --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.chunks_repo import list_pending  # noqa: E402
from app.db.connection import get_conn  # noqa: E402
from app.review.service import ChunkNotFound, decide_chunk  # noqa: E402
from app.vectorstore.qdrant_store import get_qdrant_client  # noqa: E402

ACTOR = "bulk_approve_demo"
PAGE = 200


def collect_pending_ids(service_category):
    """Snapshot all pending chunk ids up front, in one read connection, so we
    iterate a fixed list rather than a queue that shrinks as we approve."""
    ids = []
    offset = 0
    with get_conn() as conn:
        while True:
            rows, total = list_pending(
                conn, service_category=service_category, limit=PAGE, offset=offset
            )
            if not rows:
                break
            ids.extend(row["id"] for row in rows)
            offset += len(rows)
            if offset >= total:
                break
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--service-category",
        default=None,
        help="Only approve chunks in this service category (e.g. passport, oci, visa).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List how many chunks would be approved, but change nothing.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("DEMO SHORTCUT: bulk-approving pending chunks. This is NOT editorial")
    print("review -- it approves everything pending. Use for demos only.")
    print("=" * 70)

    scope = args.service_category or "ALL categories"
    print(f"Scope: {scope}")

    ids = collect_pending_ids(args.service_category)
    print(f"Found {len(ids)} pending chunk(s).")

    if not ids:
        print("Nothing to approve.")
        return 0

    if args.dry_run:
        print("Dry run: no changes made.")
        return 0

    qc = get_qdrant_client()
    approved = 0
    failed = 0
    for i, chunk_id in enumerate(ids, 1):
        try:
            # Fresh transaction per chunk (see module docstring).
            with get_conn() as conn:
                decide_chunk(conn, qc, chunk_id, "approved", actor=ACTOR)
            approved += 1
        except ChunkNotFound:
            # Superseded or vanished between the snapshot and now -- skip it.
            failed += 1
            print(f"  [{i}/{len(ids)}] skip {chunk_id}: no longer approvable")
            continue
        except Exception as exc:  # noqa: BLE001 -- report and keep going
            failed += 1
            print(f"  [{i}/{len(ids)}] FAILED {chunk_id}: {exc}")
            continue
        if approved % 25 == 0 or i == len(ids):
            print(f"  approved {approved}/{len(ids)}...")

    print("-" * 70)
    print(f"Done. Approved {approved}, skipped/failed {failed}.")
    print("Reminder: this was a demo shortcut, not real review.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
