"""
Phase 6: hybrid dense+sparse retrieval over the APPROVED chunk set.

This is the first phase where `review_status` actually gates what an end
user can see. Every search here is hard-filtered to
`review_status='approved'` — not a caller-adjustable option — because
nothing pending/rejected/superseded is meant to be retrievable per the
project brief; Phase 5's review workflow is the only thing that ever
moves a chunk into 'approved'.

Uses Qdrant's native Query API (Prefetch + FusionQuery/RRF) via
`app.vectorstore.qdrant_store.hybrid_search` — see that function's
docstring for why filtering happens inside each leg's search rather than
as a post-filter, and why RRF (rank-based) rather than a raw score
average (dense cosine similarity and sparse lexical scores aren't on
comparable scales).
"""

from typing import Any, Optional

from app.embedding.bge_m3 import BgeM3Embedder
from app.vectorstore.qdrant_store import build_filter, get_qdrant_client, hybrid_search

APPROVED = "approved"


def _point_to_result(point, rank: int) -> dict[str, Any]:
    p = point.payload
    return {
        "rank": rank,
        "score": point.score,
        "chunk_id": str(point.id),
        "source_id": p.get("source_id"),
        "source_url": p.get("source_url"),
        "service_category": p.get("service_category"),
        "canonical": p.get("canonical"),
        "jurisdiction": p.get("jurisdiction"),
        "applicant_variant": p.get("applicant_variant"),
        "version": p.get("version"),
        "retrieval_date": p.get("retrieval_date"),
        "heading_trail": p.get("heading_trail"),
        "used_ocr": p.get("used_ocr"),
        "chunk_text": p.get("chunk_text"),
    }


def search(
    query: str,
    *,
    limit: int = 8,
    prefetch_limit: int = 50,
    service_category: Optional[str] = None,
    canonical: Optional[bool] = None,
    source_id: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    applicant_variant: Optional[str] = None,
    embedder: Optional[BgeM3Embedder] = None,
    client=None,
) -> list[dict[str, Any]]:
    """Embed `query`, run hybrid dense+sparse search restricted to approved
    chunks (plus any of the optional metadata filters), and return a plain
    list of result dicts ordered by fused rank."""
    embedder = embedder or BgeM3Embedder(batch_size=1)
    client = client or get_qdrant_client()

    query_filter = build_filter(
        service_category=service_category,
        canonical=canonical,
        review_status=APPROVED,  # always enforced — never caller-overridable
        source_id=source_id,
        jurisdiction=jurisdiction,
        applicant_variant=applicant_variant,
    )

    [query_embedding] = embedder.embed([query])
    points = hybrid_search(
        client,
        query_dense=query_embedding.dense,
        query_sparse_indices=query_embedding.sparse_indices,
        query_sparse_values=query_embedding.sparse_values,
        limit=limit,
        prefetch_limit=prefetch_limit,
        query_filter=query_filter,
    )
    return [_point_to_result(p, i + 1) for i, p in enumerate(points)]
