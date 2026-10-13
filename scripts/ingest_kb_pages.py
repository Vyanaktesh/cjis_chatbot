#!/usr/bin/env python3
"""
Load the Consulate-supplied pages under frontend/public/kb/ into the
knowledge base.

These pages have no upstream website to fetch -- they are Word documents the
Consulate sent by email, converted with scripts/docx_to_kb_html.py and served
by our own nginx at https://<site>/kb/... . So instead of fetching, this
reads each page from disk, and otherwise does what fetch_all_sources.py +
embed_and_index_all.py do for a fetched page:

  1. upserts the source row from db/seed/source_registry.json (every entry
     whose URL path starts with /kb/),
  2. records a new source_versions row only when the page's bytes changed,
     keeping the raw copy under data/raw/<source_id>/v<N>/ like the fetcher,
  3. runs app.ingestion.pipeline.embed_and_index_version (extract -> chunk
     -> embed -> Postgres + Qdrant, superseding an older version's chunks),
  4. with --approve, approves the new chunks through decide_chunk(), the
     same call kb_admin makes, so both stores and the audit log agree.

Re-running with unchanged pages does nothing except (with --approve)
approving anything still pending.

Usage (inside the backend container, which has the embedding model):
    python scripts/ingest_kb_pages.py --dry-run
    python scripts/ingest_kb_pages.py --approve --reason "Supplied by CGI Atlanta, Oct 2026"

    --registry / --pages-root point elsewhere when the image predates the
    files, e.g. after `docker cp`-ing them into the container.
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.db.audit_repo import log_event  # noqa: E402
from app.db.connection import get_conn  # noqa: E402
from app.db.source_versions_repo import create_version, get_latest_version  # noqa: E402
from app.db.sources_repo import upsert_source  # noqa: E402
from app.ingestion.pipeline import REPO_ROOT as PIPELINE_ROOT  # noqa: E402
from app.ingestion.pipeline import embed_and_index_version  # noqa: E402
from app.review.service import ChunkNotFound, decide_chunk  # noqa: E402
from app.vectorstore.qdrant_store import get_qdrant_client  # noqa: E402

ACTOR = "ingest_kb_pages"
RAW_DIR = PIPELINE_ROOT / "data" / "raw"


def kb_entries(registry_path: Path) -> list[dict]:
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    return [e for e in data["sources"] if urlparse(e["url"]).path.startswith("/kb/")]


def pending_chunk_ids(conn, source_id) -> list:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM chunks WHERE source_id = %s AND review_status = 'pending_review' ORDER BY chunk_index;",
            (str(source_id),),
        )
        return [row[0] for row in cur.fetchall()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path, default=REPO_ROOT / "db" / "seed" / "source_registry.json")
    parser.add_argument("--pages-root", type=Path, default=REPO_ROOT / "frontend" / "public",
                        help="Directory the /kb/... URL paths are resolved against.")
    parser.add_argument("--approve", action="store_true", help="Approve the pages' pending chunks.")
    parser.add_argument("--reason", default="Document supplied by the Consulate",
                        help="Recorded in the audit log for each approval.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change, but change nothing.")
    args = parser.parse_args()

    entries = kb_entries(args.registry)
    if not entries:
        print(f"No /kb/ entries in {args.registry}.")
        return 1

    pages = []
    for entry in entries:
        path = args.pages_root / urlparse(entry["url"]).path.lstrip("/")
        if not path.is_file():
            print(f"MISSING {path} for {entry['url']}")
            return 1
        pages.append((entry, path, path.read_bytes()))

    print(f"{len(pages)} page(s):")
    for entry, path, content in pages:
        print(f"  [{entry['id']}] {entry['title']}  ({len(content)} bytes)")
    if args.dry_run:
        print("Dry run: no changes made.")
        return 0

    qc = get_qdrant_client()
    embedder = None
    failed = 0
    for entry, path, content in pages:
        content_hash = hashlib.sha256(content).hexdigest()
        with get_conn() as conn:
            source = upsert_source(
                conn,
                url=entry["url"],
                source_type=entry["file_type"],
                service_category=entry["service_category"],
                canonical=entry["canonical"],
                jurisdiction=["all"],
                applicant_variant=[entry["applicant_variant"]] if entry.get("applicant_variant") else [],
                active=True,
                notes=entry.get("notes"),
                title=entry.get("title"),
                source_group=entry.get("group"),
            )
            latest = get_latest_version(conn, source.id)
            if latest is not None and latest.content_hash == content_hash:
                print(f"  unchanged: {entry['title']} (v{latest.version})")
            else:
                if embedder is None:
                    # Imported lazily: loading BGE-M3 takes a while and a lot
                    # of memory, and isn't needed when nothing changed.
                    from app.embedding.bge_m3 import BgeM3Embedder

                    t0 = time.time()
                    print("  loading embedding model...")
                    embedder = BgeM3Embedder(batch_size=12)
                    print(f"  model ready in {time.time() - t0:.1f}s")
                next_version = (latest.version + 1) if latest else 1
                raw_path = RAW_DIR / str(source.id) / f"v{next_version}" / path.name
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_bytes(content)
                version = create_version(
                    conn,
                    source_id=source.id,
                    content_hash=content_hash,
                    retrieval_date=datetime.now(timezone.utc),
                    raw_content_path=str(raw_path.relative_to(PIPELINE_ROOT)),
                    fetch_status="success",
                )
                log_event(
                    conn,
                    entity_type="source_version",
                    entity_id=version.id,
                    action="local_document_loaded",
                    actor=ACTOR,
                    details={"url": entry["url"], "content_hash": content_hash, "version": version.version},
                )
                result = embed_and_index_version(conn, qc, embedder, source, version)
                print(
                    f"  indexed: {entry['title']} v{version.version}: "
                    f"{result['chunks_created']} chunk(s), {result['chunks_superseded']} superseded"
                )
            to_approve = pending_chunk_ids(conn, source.id) if args.approve else []

        approved = 0
        for chunk_id in to_approve:
            try:
                # Own transaction per chunk, as in bulk_approve_demo.py.
                with get_conn() as conn:
                    decide_chunk(conn, qc, chunk_id, "approved", actor=ACTOR, reason=args.reason)
                approved += 1
            except ChunkNotFound:
                failed += 1
                print(f"    skip {chunk_id}: no longer approvable")
            except Exception as exc:  # noqa: BLE001 -- report and keep going
                failed += 1
                print(f"    FAILED {chunk_id}: {exc}")
        if to_approve:
            print(f"    approved {approved}/{len(to_approve)}")

    print(f"Done. {failed} failure(s).")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
