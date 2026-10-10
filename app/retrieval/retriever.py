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

from app.core.config import get_settings
from app.embedding.bge_m3 import BgeM3Embedder, EmbeddingResult
from app.retrieval.reranker import Reranker
from app.vectorstore.qdrant_store import (
    COLLECTION_NAME,
    build_filter,
    dense_search,
    get_qdrant_client,
    hybrid_search,
)

APPROVED = "approved"


def has_approved_content(client=None) -> bool:
    """True if at least one approved chunk exists. Lets the caller tell "this
    question is off-topic" apart from "nothing has been approved yet" (both
    score 0 in probe_relevance). If the check itself fails, assume content
    exists so a hiccup never changes which message the visitor sees."""
    try:
        client = client or get_qdrant_client()
        count = client.count(
            collection_name=COLLECTION_NAME,
            count_filter=build_filter(review_status=APPROVED),
            exact=False,
        ).count
        return count > 0
    except Exception:  # noqa: BLE001 -- best-effort check, see docstring
        return True


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


def probe_relevance(
    query: str,
    *,
    service_category: Optional[str] = None,
    canonical: Optional[bool] = None,
    source_id: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    applicant_variant: Optional[str] = None,
    embedder: Optional[BgeM3Embedder] = None,
    client=None,
    query_embedding: Optional[EmbeddingResult] = None,
) -> float:
    """Cheap off-topic guardrail check: raw dense-only cosine similarity of
    `query` against the single closest approved chunk, restricted to the
    same filters `search()` would use. Deliberately NOT hybrid+RRF (see
    Settings.retrieval_min_relevance's docstring for why RRF's rank-based
    score can't be thresholded) and deliberately top-1 only, since this
    exists purely to decide whether to bother with real retrieval + a
    generation call at all, not to return usable results.

    `query_embedding`: pass a precomputed embedding (e.g. from
    app.generation.service, which needs it again for search() right after)
    to skip re-embedding the same query text -- BGE-M3 inference isn't
    free, and there's no reason to pay for it twice in one turn."""
    client = client or get_qdrant_client()

    query_filter = build_filter(
        service_category=service_category,
        canonical=canonical,
        review_status=APPROVED,
        source_id=source_id,
        jurisdiction=jurisdiction,
        applicant_variant=applicant_variant,
    )

    if query_embedding is None:
        embedder = embedder or BgeM3Embedder(batch_size=1)
        [query_embedding] = embedder.embed([query])
    points = dense_search(client, query_embedding.dense, limit=1, query_filter=query_filter)
    return points[0].score if points else 0.0


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
    reranker: Optional[Reranker] = None,
    query_embedding: Optional[EmbeddingResult] = None,
) -> list[dict[str, Any]]:
    """Embed `query`, run hybrid dense+sparse search restricted to approved
    chunks (plus any of the optional metadata filters), and return a plain
    list of result dicts ordered by relevance.

    `query_embedding`: pass a precomputed embedding to skip re-embedding --
    see probe_relevance()'s docstring for why this matters (this is the
    other half of the same pair of calls per turn).

    When `settings.retrieval_rerank` is on, the initial hybrid search pulls
    `retrieval_rerank_candidates` results instead of just `limit`, and a
    cross-encoder (see app/retrieval/reranker.py) re-scores that shortlist
    against the actual query text before cutting down to `limit` — added
    because hybrid dense+sparse fusion alone was shown (via kb_admin's
    eval golden set, built from real citizen queries) to often rank the
    genuinely correct chunk too low for short, keyword-style real queries.

    Defaults OFF (see Settings.retrieval_rerank): the cross-encoder is too
    slow on CPU-only hardware for the live chat path. kb_admin's own .env
    turns it on specifically for offline eval runs, where the extra
    latency is an acceptable trade for the accuracy signal.
    """
    client = client or get_qdrant_client()
    settings = get_settings()

    query_filter = build_filter(
        service_category=service_category,
        canonical=canonical,
        review_status=APPROVED,  # always enforced — never caller-overridable
        source_id=source_id,
        jurisdiction=jurisdiction,
        applicant_variant=applicant_variant,
    )

    if query_embedding is None:
        embedder = embedder or BgeM3Embedder(batch_size=1)
        [query_embedding] = embedder.embed([query])
    search_limit = max(limit, settings.retrieval_rerank_candidates) if settings.retrieval_rerank else limit
    points = hybrid_search(
        client,
        query_dense=query_embedding.dense,
        query_sparse_indices=query_embedding.sparse_indices,
        query_sparse_values=query_embedding.sparse_values,
        limit=search_limit,
        prefetch_limit=prefetch_limit,
        query_filter=query_filter,
    )
    results = [_point_to_result(p, i + 1) for i, p in enumerate(points)]

    if not settings.retrieval_rerank or not results:
        return results[:limit]

    reranker = reranker or Reranker()
    return reranker.rerank(query, results, limit=limit)
