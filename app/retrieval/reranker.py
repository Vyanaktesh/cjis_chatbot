"""
Cross-encoder reranking, added after real-world eval data (see the
consulate-kb-admin evaluation module's golden set, built from actual
citizen queries) showed a specific, repeated failure mode: hybrid
dense+sparse retrieval alone often ranks the genuinely correct chunk too
low (or misses it in the top-k entirely) for short, terse, keyword-style
real queries like "I ALREADY HAVE OCI" or "Police Clearance Certificate" —
phrases with little semantic content for a bi-encoder to match against,
even when the correct chunk exists and is well-formed.

Same BGE family as the embedding model (BAAI/bge-reranker-v2-m3), same
FlagEmbedding package already required for app.embedding.bge_m3 — no new
dependency. A cross-encoder scores the query against each candidate
chunk's actual text directly (rather than comparing two independently
computed vectors), which is slower per-pair but far more precise; run only
over a modest shortlist (retriever.py's rerank_candidates setting) rather
than the whole collection, so the cost stays bounded.

CPU-only per the project's modest/no-GPU-hardware requirement, same as
BgeM3Embedder. Model pulled once from Hugging Face Hub on first use and
cached after (~1.1GB) -- no third-party API call, stays self-hostable.
"""

import sys
import types

from app.core.logging_config import get_logger

logger = get_logger(__name__)

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


def _ensure_datasets_importable() -> None:
    """Same FlagEmbedding-imports-`datasets`-unconditionally workaround as
    app.embedding.bge_m3's helper of the same name (see that module's
    docstring for the full explanation) -- duplicated rather than imported
    across modules since it's a private compatibility shim, not a shared
    public helper."""
    if "datasets" in sys.modules:
        return
    try:
        import datasets  # noqa: F401
    except Exception as exc:
        logger.warning(
            f"real 'datasets' package unavailable ({exc!r}); substituting a stub "
            "(see app.embedding.bge_m3._ensure_datasets_importable for why this is safe)."
        )
        sys.modules["datasets"] = types.ModuleType("datasets")


class Reranker:
    _model = None  # class-level: loading is expensive, share it across instances in one process

    def __init__(self):
        if Reranker._model is None:
            _ensure_datasets_importable()
            from FlagEmbedding import FlagReranker

            logger.info(f"loading {RERANKER_MODEL_NAME} (first call in this process only)...")
            Reranker._model = FlagReranker(RERANKER_MODEL_NAME, use_fp16=False, devices=["cpu"])
        self._model = Reranker._model

    def rerank(self, query: str, candidates: list[dict], *, limit: int) -> list[dict]:
        """Re-scores `candidates` (each a retrieval result dict with a
        `chunk_text` field, as produced by retriever.py's _point_to_result)
        against `query` with the cross-encoder, and returns the top `limit`
        re-ranked, with `rank` and `score` overwritten to reflect the new
        order — the original fused hybrid-search rank/score are kept under
        `prerank_rank`/`prerank_score` so a caller can still see what the
        first-pass retrieval produced."""
        if not candidates:
            return []

        pairs = [(query, c["chunk_text"]) for c in candidates]
        scores = self._model.compute_score(pairs)
        if isinstance(scores, float):  # a single-pair call returns a bare float, not a list
            scores = [scores]

        rescored = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["prerank_rank"] = item["rank"]
            item["prerank_score"] = item["score"]
            item["score"] = float(score)
            rescored.append(item)

        rescored.sort(key=lambda c: c["score"], reverse=True)
        top = rescored[:limit]
        for i, item in enumerate(top):
            item["rank"] = i + 1
        return top
