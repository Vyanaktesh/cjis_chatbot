"""
"tag metadata" step of the ingestion pipeline: takes chunker output plus
the Source/SourceVersion rows it came from, and produces plain dicts
shaped exactly like the `chunks` table's columns (db/migrations/0001).
Phase 4 will take these dicts, add embedding_model/qdrant_point_id once
BGE-M3 embeds them, and insert/upsert for real. This module doesn't touch
the DB itself — it's pure data shaping, reused as-is by Phase 4.
"""

import hashlib
from typing import Any

from app.chunking.chunker import Chunk
from app.db.source_versions_repo import SourceVersion
from app.db.sources_repo import Source


def build_chunk_records(
    chunks: list[Chunk],
    source: Source,
    source_version: SourceVersion,
) -> list[dict[str, Any]]:
    records = []
    for chunk in chunks:
        content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
        records.append(
            {
                "source_id": source.id,
                "source_version_id": source_version.id,
                "chunk_index": chunk.chunk_index,
                "chunk_text": chunk.text,
                "content_hash": content_hash,
                "source_url": source.url,
                "retrieval_date": source_version.retrieval_date,
                "source_last_modified": source_version.source_last_modified,
                "service_category": source.service_category,
                "canonical": source.canonical,
                "jurisdiction": source.jurisdiction,
                "applicant_variant": source.applicant_variant,
                "review_status": "pending_review",
                "version": source_version.version,
                # informational, not schema columns — useful for the Phase 3
                # verify step and for debugging chunk boundaries later
                "_heading_trail": chunk.heading_trail,
                "_block_types": chunk.block_types,
                "_page_numbers": chunk.page_numbers,
                "_used_ocr": chunk.used_ocr,
            }
        )
    return records
