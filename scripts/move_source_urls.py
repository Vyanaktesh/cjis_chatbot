#!/usr/bin/env python3
"""
Point existing sources at a new URL for the SAME document, keeping their
history, versions and review decisions.

Used when a site moves a file without changing it -- e.g. VFS Global now
serves the one-pager checklists from www.vfsglobal.com and refuses
automated fetches on services.vfsglobal.com. Updates sources.url, every
chunk's source_url in Postgres, and the source_url in each chunk's Qdrant
payload (that is the link the chat's source chip opens), Postgres first,
Qdrant last, in one transaction per source.

Usage:
    python scripts/move_source_urls.py moves.json        # [[old_url, new_url], ...]
    python scripts/move_source_urls.py moves.json --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.audit_repo import log_event  # noqa: E402
from app.db.connection import get_conn  # noqa: E402
from app.vectorstore.qdrant_store import get_qdrant_client, set_payload_fields  # noqa: E402

ACTOR = "move_source_urls"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("moves", type=Path, help="JSON file: [[old_url, new_url], ...]")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    moves = json.loads(args.moves.read_text(encoding="utf-8"))
    if isinstance(moves, dict):
        moves = moves["renames"]
    qc = get_qdrant_client()
    moved = skipped = 0
    for old, new in moves:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM sources WHERE url = %s;", (old,))
                row = cur.fetchone()
                cur.execute("SELECT 1 FROM sources WHERE url = %s;", (new,))
                clash = cur.fetchone()
                if row is None or clash is not None:
                    reason = "not found" if row is None else "new URL already registered"
                    print(f"  skip ({reason}): {old}")
                    skipped += 1
                    continue
                source_id = row[0]
                cur.execute("SELECT qdrant_point_id FROM chunks WHERE source_id = %s AND qdrant_point_id IS NOT NULL;",
                            (str(source_id),))
                point_ids = [str(r[0]) for r in cur.fetchall()]
                print(f"  {old}\n    -> {new}  ({len(point_ids)} chunk(s))")
                if args.dry_run:
                    continue
                cur.execute("UPDATE sources SET url = %s WHERE id = %s;", (new, str(source_id)))
                cur.execute("UPDATE chunks SET source_url = %s WHERE source_id = %s;", (new, str(source_id)))
            log_event(conn, entity_type="source", entity_id=source_id, action="url_moved", actor=ACTOR,
                      details={"from": old, "to": new})
            if point_ids:
                set_payload_fields(qc, point_ids, {"source_url": new})
            moved += 1
    print(f"Done. Moved {moved}, skipped {skipped}." + (" (dry run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
