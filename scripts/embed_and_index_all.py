#!/usr/bin/env python3
"""
Phase 4: embed every source's chunks (BGE-M3 dense + sparse) and index
them — Qdrant gets the vectors + full metadata payload, Postgres gets the
same metadata as the system of record for the review/versioning workflow.
Every chunk lands with review_status='pending_review' — nothing is
retrievable until Phase 5's admin review approves it.

As of Phase 5 this is a thin CLI wrapper around
`app/ingestion/pipeline.embed_and_index_version`, the same function the
Phase 5 manual-upload API endpoint calls for a single document — so a
document processed either way goes through identical extract/chunk/embed/
index/supersede logic.

Re-runs are idempotent: each chunk's ID is derived deterministically from
(source_version_id, chunk_index), so re-embedding the same content updates
the same Qdrant point / Postgres row rather than duplicating it.

Usage:
    python scripts/embed_and_index_all.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.core.logging_config import configure_logging, get_logger  # noqa: E402
from app.db.connection import get_conn  # noqa: E402
from app.db.source_versions_repo import get_latest_version  # noqa: E402
from app.db.sources_repo import list_sources  # noqa: E402
from app.embedding.bge_m3 import MODEL_NAME, BgeM3Embedder  # noqa: E402
from app.ingestion.pipeline import chunk_id_for, extract_and_chunk, index_records  # noqa: E402
from app.vectorstore.qdrant_store import ensure_collection, get_qdrant_client  # noqa: E402

logger = get_logger(__name__)


def main() -> int:
    configure_logging(get_settings().log_level)

    with get_conn() as conn:
        sources = list_sources(conn, active_only=True)
        source_versions = {s.id: get_latest_version(conn, s.id) for s in sources}

    to_process = [(s, source_versions[s.id]) for s in sources if source_versions.get(s.id) is not None]
    print(f"{len(sources)} active sources, {len(to_process)} with a fetched version to embed.\n")

    print("Extracting + chunking each source...")
    # (source, version, records) per source — extraction/chunking is fast
    # (no model involved), so doing it per-source up front costs nothing,
    # and it's what lets us batch every chunk from every source through a
    # SINGLE embed() call below instead of one call per source.
    per_source: list[tuple] = []
    for source, version in to_process:
        try:
            records = extract_and_chunk(source, version)
        except Exception as exc:
            print(f"  SKIP {source.title}: extraction failed: {exc}")
            continue
        for r in records:
            r["id"] = chunk_id_for(version.id, r["chunk_index"])
        per_source.append((source, version, records))

    all_records = [r for _, _, records in per_source for r in records]
    print(f"{len(all_records)} total chunks to embed.\n")

    print(f"Loading {MODEL_NAME} (downloads on first run, cached after)...")
    t0 = time.time()
    embedder = BgeM3Embedder(batch_size=12)
    print(f"Model ready in {time.time() - t0:.1f}s.\n")

    print(f"Embedding {len(all_records)} chunks (dense + sparse) in one batch...")
    t0 = time.time()
    try:
        all_embeddings = embedder.embed([r["chunk_text"] for r in all_records])
    except Exception as exc:
        # The per-source extract (above) and index (below) loops are each
        # guarded; without this, one embedding failure would discard the
        # whole run after all extraction work is already done.
        print(f"ERROR: embedding failed, aborting run: {exc}")
        return 1
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed / max(1, len(all_records)):.2f}s/chunk).\n")

    client = get_qdrant_client()
    ensure_collection(client)

    total_chunks = 0
    total_superseded = 0
    sources_with_chunks = 0
    t0 = time.time()
    cursor = 0
    with get_conn() as conn:
        for source, version, records in per_source:
            n = len(records)
            embeddings_slice = all_embeddings[cursor : cursor + n]
            cursor += n
            try:
                summary = index_records(conn, client, source, version, records, embeddings_slice)
            except Exception as exc:
                print(f"  SKIP {source.title}: indexing failed: {exc}")
                continue
            if summary["chunks_created"] > 0:
                sources_with_chunks += 1
            total_chunks += summary["chunks_created"]
            total_superseded += summary["chunks_superseded"]
    elapsed = time.time() - t0

    print(f"Indexed in {elapsed:.1f}s.\n")
    print("=== Summary ===")
    print(f"sources with chunks: {sources_with_chunks}")
    print(f"total chunks embedded + indexed: {total_chunks}")
    if total_superseded:
        print(f"chunks superseded by a newer version: {total_superseded}")
    print(f"embedding model: {MODEL_NAME}")
    print("all new chunks inserted with review_status='pending_review' (nothing retrievable until Phase 5 approval)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
