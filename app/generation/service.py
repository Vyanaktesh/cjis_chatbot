"""
Phase 7: ties Phase 6's retrieval (approved-only, hybrid dense+sparse)
together with Qwen3 generation into one "answer a question" call.

Deliberately short-circuits before ever calling the model when retrieval
finds nothing — an empty-context prompt is asking the model to correctly
decline on its own, which is a weaker guarantee than just not generating
at all when there's genuinely nothing approved to answer from.
"""

from typing import Any, Optional

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.generation.prompt import build_rag_messages, extract_cited_indices, strip_thinking
from app.retrieval.retriever import probe_relevance
from app.retrieval.retriever import search as retrieval_search

logger = get_logger(__name__)

GENERATION_UNAVAILABLE_ANSWER = (
    "The answer generation service is temporarily unavailable (it may be "
    "rate-limited or briefly down). Please wait a moment and try again."
)


def _default_generator() -> Any:
    """Picks the generation backend from config (Phase 7 follow-up):
    "gemini" opts into Google's API for speed (see gemini_backend.py's
    docstring for the trade-off); anything else (including unset) keeps the
    original self-hosted Qwen3/llama.cpp backend. Imported lazily inside
    each branch so a machine that only has one backend's deps installed
    (e.g. no google-genai) doesn't fail to import this module at all."""
    settings = get_settings()
    if settings.generation_backend == "gemini":
        from app.generation.gemini_backend import GeminiGenerator

        return GeminiGenerator()
    from app.generation.qwen_backend import QwenGenerator

    return QwenGenerator()

NO_CONTEXT_ANSWER = (
    "I don't have approved information covering that yet. Please contact "
    "the consulate directly, or check back once this topic has been reviewed."
)

# Distinct from NO_CONTEXT_ANSWER: that message implies the topic is
# consulate-related but not yet reviewed/approved, which would be
# misleading for a question that was never in scope to begin with (e.g.
# "what is the capital of France?"). Kept short and polite throughout —
# this is a government-run assistant, so a curt or robotic decline reads
# poorly even when the underlying answer is just "no".
OUT_OF_SCOPE_ANSWER = (
    "Thank you for your message. This looks like it's outside what I can "
    "help with here, as I'm only able to answer questions about Indian "
    "passport, OCI, and visa services. Please feel free to ask me about "
    "those, or contact the consulate directly for anything else."
)


def _retrieval_query(query: str, history: Optional[list[dict[str, str]]]) -> str:
    """Phase 8: cheap follow-up handling for retrieval. A real query
    rewrite would mean an extra LLM call before we even start retrieval —
    on this CPU-only setup that's another ~30s+ tacked onto every single
    turn, which isn't worth it. Instead: if there was a previous user
    message, prepend just that one to the current query text before
    embedding, so a follow-up like "what about for a child?" after an OCI
    question still pulls in "OCI" as retrieval context. Only the most
    recent prior user turn is used (not the whole history) to avoid
    dragging retrieval off-topic as a conversation wanders."""
    if not history:
        return query
    for message in reversed(history):
        if message.get("role") == "user" and message.get("content"):
            return f"{message['content']} {query}"
    return query


def answer_question(
    query: str,
    *,
    limit: int = 6,
    service_category: Optional[str] = None,
    canonical: Optional[bool] = None,
    source_id: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    applicant_variant: Optional[str] = None,
    history: Optional[list[dict[str, str]]] = None,
    generator: Optional[Any] = None,
) -> dict[str, Any]:
    search_query = _retrieval_query(query, history)
    settings = get_settings()

    relevance = probe_relevance(
        search_query,
        service_category=service_category,
        canonical=canonical,
        source_id=source_id,
        jurisdiction=jurisdiction,
        applicant_variant=applicant_variant,
    )
    if relevance < settings.retrieval_min_relevance:
        logger.info(f"declined as out of scope (relevance={relevance:.3f}): {query!r}")
        return {
            "query": query,
            "answer": OUT_OF_SCOPE_ANSWER,
            "grounded": False,
            "citations": [],
            "retrieved_count": 0,
        }

    chunks = retrieval_search(
        search_query,
        limit=limit,
        service_category=service_category,
        canonical=canonical,
        source_id=source_id,
        jurisdiction=jurisdiction,
        applicant_variant=applicant_variant,
    )

    if not chunks:
        return {
            "query": query,
            "answer": NO_CONTEXT_ANSWER,
            "grounded": False,
            "citations": [],
            "retrieved_count": 0,
        }

    messages = build_rag_messages(query, chunks, history)
    generator = generator or _default_generator()
    try:
        raw = generator.generate(messages)
    except Exception as exc:
        # Deliberately broad: whichever backend is active (Gemini rate
        # limits/API errors after retries -- see gemini_backend.py's
        # GenerationUnavailable -- or any other unexpected failure from
        # either backend), a generation failure should degrade to an
        # honest message like NO_CONTEXT_ANSWER above, not an unhandled
        # 500 that leaves the widget showing nothing at all.
        logger.warning(f"generation backend failed: {exc}")
        return {
            "query": query,
            "answer": GENERATION_UNAVAILABLE_ANSWER,
            "grounded": False,
            "citations": [],
            "retrieved_count": len(chunks),
            "generation_error": str(exc),
        }
    answer = strip_thinking(raw)

    cited = extract_cited_indices(answer)
    citations = []
    for n in cited:
        if 1 <= n <= len(chunks):
            c = chunks[n - 1]
            citations.append(
                {
                    "index": n,
                    "chunk_id": c["chunk_id"],
                    "source_url": c["source_url"],
                    "service_category": c["service_category"],
                    "canonical": c["canonical"],
                }
            )

    return {
        "query": query,
        "answer": answer,
        "grounded": True,
        "citations": citations,
        "retrieved_count": len(chunks),
        "uncited_sources_retrieved": len(chunks) - len(citations),
    }
