#!/usr/bin/env python3
"""
Loads db/seed/source_registry.json's `sources` array into the Postgres
`sources` table via the CRUD in app/db/sources_repo.py. Upserts on URL, so
it's safe to re-run any time the registry file is updated.

Entries under the registry's `not_yet_located` / `pending_from_office`
sections are informational gaps, not fetchable sources (no URL) — they are
intentionally NOT loaded here. They're preserved in the Claude project doc
and in db/seed/source_registry.json's sibling metadata for humans to track.

Usage:
    python scripts/load_source_registry.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.connection import get_conn  # noqa: E402
from app.db.sources_repo import upsert_source  # noqa: E402

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "db" / "seed" / "source_registry.json"


def main() -> int:
    data = json.loads(REGISTRY_PATH.read_text())
    entries = data["sources"]

    created_or_updated = 0
    with get_conn() as conn:
        for entry in entries:
            applicant_variant = (
                [entry["applicant_variant"]] if entry.get("applicant_variant") else []
            )
            upsert_source(
                conn,
                url=entry["url"],
                source_type=entry["file_type"],
                service_category=entry["service_category"],
                canonical=entry["canonical"],
                jurisdiction=["all"],
                applicant_variant=applicant_variant,
                active=True,
                notes=entry.get("notes"),
                title=entry.get("title"),
                source_group=entry.get("group"),
            )
            created_or_updated += 1
            print(f"  [{entry['id']:>2}] upserted: {entry['title']}")

    print(f"\nDone. {created_or_updated} source(s) upserted from {REGISTRY_PATH.name}.")
    print(
        f"({len(data.get('not_yet_located', {}).get('items', []))} not_yet_located and "
        f"{len(data.get('pending_from_office', {}).get('items', []))} pending_from_office "
        "items were skipped — no URL to register.)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
