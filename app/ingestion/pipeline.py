"""
Shared "extract -> chunk -> tag metadata -> embed -> index" pipeline.

This is the same logic Phase 4's `scripts/embed_and_index_all.py` used
inline, pulled out so Phase 5's manual-upload API endpoint can run a single
document through the exact same steps instead of a second, subtly
different copy — the same lesson Phase 2 already applied to fetch/persist
logic (`app/fetcher/orchestrate.py`).

It also owns a piece of the ingestion pipeline that Phase 4 hadn't
implemented yet: the project brief's "content changes: don't overwrite,
mark old chunks superseded, insert new version, re-review" rule. Every
call to `embed_and_index_version` embeds/indexes exactly one
(source, source_version) pair, then marks any chunk still attached to an
OLDER version of that same source as `review_status='superseded'` in BOTH
Postgres and Qdrant — it is no longer the current content and shouldn't be
retrieved, approved, or rejected as if it still were.
"""

import uuid
from pathlib import Path
from typing import Any

from app.chunking.chunker import chunk_blocks
from app.chunking.tag_metadata import build_chunk_records
from app.db.audit_repo import log_event
from app.db.chunks_repo import list_by_source_excluding_version, mark_superseded, upsert_chunk
from app.db.source_versions_repo import SourceVersion
from app.db.sources_repo import Source
from app.embedding.bge_m3 import MODEL_NAME, BgeM3Embedder
from app.extraction.html_extractor import extract_html
from app.extraction.pdf_extractor import extract_pdf
from app.vectorstore.qdrant_store import build_point, ensure_collection, set_payload_fields, upsert_points

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def chunk_id_for(source_version_id, chunk_index: int) -> uuid.UUID:
    """Deterministic chunk ID, shared as both the Postgres PK and the Qdrant
    point ID — makes re-running ingestion for the same version idempotent."""
    return uuid.uuid5(uuid.NAMESPACE_URL, f"source_version:{source_version_id}:chunk:{chunk_index}")


def extract_and_chunk(source: Source, version: SourceVersion) -> list[dict[str, Any]]:
    raw_path = REPO_ROOT / version.raw_content_path
    raw_bytes = raw_path.read_bytes()
    blocks = extract_pdf(raw_bytes) if source.source_type == "pdf" else extract_html(raw_bytes)
    chunks = chunk_blocks(blocks)
    return build_chunk_records(chunks, source, version)


def to_qdrant_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": str(record["source_id"]),
        "source_version_id": str(record["source_version_id"]),
        "chunk_index": record["chunk_index"],
        "source_url": record["source_url"],
        "retrieval_date": record["retrieval_date"].isoformat() if record["retrieval_date"] else None,
        "source_last_modified": (
            record["source_last_modified"].isoformat() if record["source_last_modified"] else None
        ),
        "service_category": record["service_category"],
        "canonical": record["canonical"],
        "jurisdiction": record["jurisdiction"],
        "applicant_variant": record["applicant_variant"],
        "review_status": record["review_status"],
        "version": record["version"],
        "content_hash": record["content_hash"],
        "chunk_text": record["chunk_text"],
        "heading_trail": record["_heading_trail"],
        "used_ocr": record["_used_ocr"],
    }


def index_records(
    conn,
    qdrant_client,
    source: Source,
    version: SourceVersion,
    records: list[dict[str, Any]],
    embeddings,
) -> dict[str, Any]:
    """
    Second half of the pipeline: given chunk records that are ALREADY
    embedded (dense + sparse vectors attached, one per record in the same
    order), upserts them into Postgres + Qdrant and supersedes any chunks
    left over from older versions of the same source. Split out from
    `embed_and_index_version` so a caller processing many sources at once
    (the bulk CLI script) can embed everything in a single batched model
    call — much faster on CPU than one `embed()` call per source — while
    still reusing this exact indexing/supersede logic per source.

    Deliberately does every Postgres write FIRST and every Qdrant write
    LAST. Qdrant isn't part of the caller's Postgres transaction, so a
    failure partway through can't be rolled back the same way — this
    ordering was chosen after a real bug reproduced exactly that: an
    error in the (Postgres-only) supersede step, hit *after* Qdrant's
    upsert had already run, left two Qdrant points with no matching
    Postgres row once the transaction rolled back. Doing all Postgres
    work first means a Postgres-side failure now happens before Qdrant is
    touched at all, so the rollback leaves nothing orphaned. A failure in
    the Qdrant step itself is still possible and isn't fully solved here —
    true cross-store atomicity would need an outbox/2PC pattern, out of
    scope for this phase — but the specific failure mode that actually
    occurred is closed off.
    """
    for record in records:
        db_record = dict(record)
        db_record["embedding_model"] = MODEL_NAME
        db_record["qdrant_point_id"] = record["id"]
        db_record["used_ocr"] = record["_used_ocr"]
        upsert_chunk(conn, db_record)

    log_event(
        conn,
        entity_type="source_version",
        entity_id=version.id,
        action="chunks_embedded",
        details={"source_id": str(source.id), "version": version.version, "chunk_count": len(records)},
    )

    # Figure out (in Postgres) what needs superseding before touching
    # Qdrant at all.
    stale = list_by_source_excluding_version(conn, source.id, version.id)
    to_supersede = [row for row in stale if row["review_status"] != "superseded"]
    stale_ids = [row["id"] for row in to_supersede]
    if stale_ids:
        mark_superseded(conn, stale_ids)
        log_event(
            conn,
            entity_type="source",
            entity_id=source.id,
            action="chunks_superseded",
            details={
                "new_version": version.version,
                "superseded_chunk_ids": [str(i) for i in stale_ids],
            },
        )

    # All Postgres writes are done — now the Qdrant side.
    ensure_collection(qdrant_client)

    points = []
    for record, emb in zip(records, embeddings):
        payload = to_qdrant_payload(record)
        points.append(build_point(record["id"], emb.dense, emb.sparse_indices, emb.sparse_values, payload))
    upsert_points(qdrant_client, points)

    if stale_ids:
        stale_qdrant_ids = [row["qdrant_point_id"] for row in to_supersede if row["qdrant_point_id"]]
        set_payload_fields(qdrant_client, stale_qdrant_ids, {"review_status": "superseded"})

    return {
        "source_id": str(source.id),
        "version": version.version,
        "chunks_created": len(records),
        "chunks_superseded": len(stale_ids),
    }


def embed_and_index_version(
    conn,
    qdrant_client,
    embedder: BgeM3Embedder,
    source: Source,
    version: SourceVersion,
) -> dict[str, Any]:
    """
    Extracts, chunks, embeds, and indexes ONE (source, version) pair —
    the single-document path used by the Phase 5 manual-upload API
    endpoint, where embedding one document at a time is the natural shape
    (there's only ever one document per upload request). For bulk
    ingestion across many sources, prefer batching all sources' texts
    through one `embedder.embed()` call and calling `index_records`
    directly per source (see `scripts/embed_and_index_all.py`) — far
    fewer, larger model calls, which matters a lot on CPU-only hardware.
    """
    records = extract_and_chunk(source, version)
    for r in records:
        r["id"] = chunk_id_for(version.id, r["chunk_index"])

    texts = [r["chunk_text"] for r in records]
    embeddings = embedder.embed(texts) if texts else []

    return index_records(conn, qdrant_client, source, version, records, embeddings)
