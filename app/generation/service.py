"""
Phase 7: ties Phase 6's retrieval (approved-only, hybrid dense+sparse)
together with Qwen3 generation into one "answer a question" call.

Deliberately short-circuits before ever calling the model when retrieval
finds nothing — an empty-context prompt is asking the model to correctly
decline on its own, which is a weaker guarantee than just not generating
at all when there's genuinely nothing approved to answer from.
"""

import re
from functools import lru_cache
from typing import Any, Optional

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.db.connection import get_conn
from app.embedding.bge_m3 import BgeM3Embedder
from app.generation.guardrails import screen_query
from app.generation.prompt import (
    MORE_DETAILS_OFFER,
    build_rag_messages,
    build_voice_messages,
    ensure_more_details_offer,
    extract_cited_indices,
    first_sentences,
    is_more_details_request,
    normalize_citation_groups,
    strip_citation_markers,
    strip_for_speech,
    strip_thinking,
)
from app.retrieval.retriever import has_approved_content, pinned_chunks, probe_relevance
from app.retrieval.retriever import search as retrieval_search

logger = get_logger(__name__)

# Short, friendly replies for greetings and thanks. These never reach retrieval
# or the model: "hello" and "thanks" score too low against consular content, so
# they used to be refused as "outside what I can help with", which reads as rude.
# `smalltalk: True` in the response tells the widget not to offer the "raise a
# query with the consulate" form for them.
_SMALLTALK = [
    (
        re.compile(
            r"^(?:hi+|hello+|hey+|heya|hiya|namaste|namaskar|greetings|"
            r"good (?:morning|afternoon|evening))(?: (?:there|dost|team|all))?[ !.,]*$"
        ),
        "Hello! I can help with questions about Indian passport, OCI and visa services. What would you like to know?",
    ),
    (
        re.compile(
            r"^(?:thanks|thank you|thank u|thankyou|thx|ty|many thanks|"
            r"thanks a lot|thank you so much|thanks so much|dhanyavad(?:am)?|shukriya)[ !.,]*$"
        ),
        "You're welcome! Let me know if there is anything else I can help with.",
    ),
    (
        re.compile(r"^(?:bye|goodbye|good bye|see you|see ya|ok bye|okay bye|take care)[ !.,]*$"),
        "Goodbye! Come back any time you have a question about passport, OCI or visa services.",
    ),
    (
        re.compile(r"^(?:ok|okay|k|got it|cool|great|nice|alright|fine|perfect|understood)[ !.,]*$"),
        "Glad to help! Ask me anything about passport, OCI or visa services.",
    ),
]


def _smalltalk_reply(query: str) -> Optional[str]:
    normalized = re.sub(r"\s+", " ", query.strip().lower())
    for pattern, reply in _SMALLTALK:
        if pattern.match(normalized):
            return reply
    return None


@lru_cache(maxsize=4)
def _domain_keyword_re(keywords: str) -> Optional["re.Pattern[str]"]:
    words = [w.strip() for w in keywords.split(",") if w.strip()]
    if not words:
        return None
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE)


def _mentions_domain_keyword(query: str) -> bool:
    pattern = _domain_keyword_re(get_settings().retrieval_domain_keywords)
    return bool(pattern and pattern.search(query))

# The model is told (prompt.py SYSTEM_PROMPT, rule 3) to say "I don't have
# approved information covering that" when the sources don't answer the
# question. Matching that phrase lets the backend treat such a reply as a
# decline rather than an answer.
_DECLINE_RE = re.compile(
    r"\b(?:do not|don['’]t|dont) have (?:any )?approved information\b",
    re.IGNORECASE,
)

GENERATION_UNAVAILABLE_ANSWER = (
    "Sorry, the service is temporarily down. Please wait a moment and try "
    "again."
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
    # The last TWO earlier questions, so a chain of short follow-ups ("what
    # documents for OCI?" -> "and for a child?" -> "is it the same for my
    # wife?") keeps the original topic. Replies that only accept the offer of
    # more details ("yes") carry no topic and are skipped.
    earlier = [
        m["content"]
        for m in history
        if m.get("role") == "user" and m.get("content") and not is_more_details_request(m["content"])
    ][-2:]
    return " ".join([*earlier, query]) if earlier else query


_CITATION_MARK_RE = re.compile(r"\[(\d+)\]")
_ADJACENT_DUP_RE = re.compile(r"(\[\d+\])(?:\1)+")


def _dedupe_citations_by_source(answer: str, chunks: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Several retrieved chunks usually come from the SAME source document, so an
    answer can cite [1][2][3][4] that all point to one URL -- showing four
    identical source chips is noise. Collapse to ONE citation per unique source:
    give each source a single number (in order of first citation), rewrite the
    bracket markers in the answer to match, and return one citation per source.
    Markers that pointed at the same source then become adjacent duplicates
    (e.g. [1][1][1]), which are collapsed to a single [1]. Markers pointing at an
    out-of-range chunk number are dropped."""
    source_to_new: dict[str, int] = {}
    old_to_new: dict[int, int] = {}
    citations: list[dict[str, Any]] = []
    for old in extract_cited_indices(answer):
        if not (1 <= old <= len(chunks)):
            continue
        c = chunks[old - 1]
        url = c["source_url"]
        if url not in source_to_new:
            source_to_new[url] = len(source_to_new) + 1
            citations.append(
                {
                    "index": source_to_new[url],
                    "chunk_id": c["chunk_id"],
                    "source_url": url,
                    "service_category": c["service_category"],
                    "canonical": c["canonical"],
                }
            )
        old_to_new[old] = source_to_new[url]

    def _remap(m: "re.Match[str]") -> str:
        n = int(m.group(1))
        return f"[{old_to_new[n]}]" if n in old_to_new else ""

    new_answer = _CITATION_MARK_RE.sub(_remap, answer)
    new_answer = _ADJACENT_DUP_RE.sub(r"\1", new_answer)
    return new_answer, citations


_WORD_RE = re.compile(r"[a-z0-9]{4,}")
_ATTRIBUTION_STOPWORDS = frozenset(
    "that this with from have will must your they their been were which when where what should would could "
    "also only into about after before other than then them these those such more most each does done "
    "applicant applicants application applications please".split()
)
# A bullet counts as taken from a chunk when at least this share of its content
# words appear in that chunk.
_ATTRIBUTION_MIN_OVERLAP = 0.6


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _ATTRIBUTION_STOPWORDS}


def _attribute_uncited(answer: str, chunks: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Fallback for an answer the model wrote WITHOUT any [n] markers. Each
    bullet is matched to the retrieved chunk that contains most of its content
    words; a marker is added only where that overlap is high, and only when most
    of the substantive bullets are matched. Otherwise the answer is returned
    unchanged (and stays ungrounded). The model only ever sees these chunks, so
    this recovers its sources; it never lends a source to unsupported text."""
    chunk_words = [_content_words(c.get("chunk_text") or "") for c in chunks]
    lines = answer.splitlines()
    substantive = matched = 0
    for i, line in enumerate(lines):
        words = _content_words(line)
        if len(words) < 4 or MORE_DETAILS_OFFER in line:
            continue
        substantive += 1
        best, best_idx = 0.0, -1
        for j, cw in enumerate(chunk_words):
            score = len(words & cw) / len(words)
            if score > best:
                best, best_idx = score, j
        if best >= _ATTRIBUTION_MIN_OVERLAP:
            matched += 1
            lines[i] = f"{line.rstrip()} [{best_idx + 1}]"
    if not substantive or matched * 2 < substantive:
        return answer, []
    return _dedupe_citations_by_source("\n".join(lines), chunks)


def _attach_source_titles(citations: list[dict[str, Any]]) -> None:
    """Adds each citation's source title (from the sources registry) as
    `source_title`, for the widget's source chips. URLs alone make poor labels:
    the Consulate's pages are /eoial_pages/MjY1-style codes. Best-effort: if
    the lookup fails, the chips fall back to a label made from the URL."""
    if not citations:
        return
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT url, title FROM sources WHERE url = ANY(%s);",
                ([c["source_url"] for c in citations],),
            )
            titles = dict(cur.fetchall())
    except Exception as exc:  # noqa: BLE001 -- see docstring
        logger.warning(f"source title lookup failed: {exc}")
        return
    for c in citations:
        if titles.get(c["source_url"]):
            c["source_title"] = titles[c["source_url"]]


def answer_question(
    query: str,
    *,
    mode: str = "text",
    generator: Optional[Any] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Public entry point. Runs the grounded pipeline unchanged, then — only in
    voice mode — attaches a `spoken_text` field (a short summary of the already
    grounded answer, safe to read aloud). mode="text" (the default) returns
    exactly what it always did, so existing callers are unaffected."""
    result = _answer_core(query, generator=generator, **kwargs)
    if mode == "voice":
        _attach_spoken_text(result, generator)
    return result


def _attach_spoken_text(result: dict[str, Any], generator: Optional[Any]) -> None:
    """Adds result["spoken_text"]. For a grounded answer it condenses the
    written answer into a spoken reply with a second (cheap) model call; that
    call only summarises already-grounded text, so it adds no grounding risk,
    and it falls back to the cleaned full answer if it fails. Short canned
    replies (declines, guardrails, smalltalk, outages) are just sanitised."""
    answer = result.get("answer", "")
    if result.get("grounded"):
        try:
            gen = generator or _default_generator()
            raw = gen.generate(build_voice_messages(result.get("query", ""), answer))
            spoken = strip_for_speech(strip_thinking(raw))
            if not spoken:
                spoken = strip_for_speech(answer)
        except Exception as exc:  # noqa: BLE001 -- logged; fall back to the cleaned answer
            logger.warning(f"voice summary generation failed: {exc}")
            spoken = strip_for_speech(answer)
    else:
        # Declines / canned replies: sanitise, and keep it brief when the model
        # wrote a long decline (the full text still shows on screen).
        spoken = first_sentences(strip_for_speech(answer), max_words=55)
    result["spoken_text"] = spoken


def _answer_core(
    query: str,
    *,
    limit: Optional[int] = None,
    service_category: Optional[str] = None,
    canonical: Optional[bool] = None,
    source_id: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    applicant_variant: Optional[str] = None,
    history: Optional[list[dict[str, str]]] = None,
    generator: Optional[Any] = None,
) -> dict[str, Any]:
    smalltalk = _smalltalk_reply(query)
    if smalltalk is not None:
        return {
            "query": query,
            "answer": smalltalk,
            "grounded": False,
            "citations": [],
            "retrieved_count": 0,
            "smalltalk": True,
        }

    # Input guardrails (prompt-injection / jailbreak, and self-harm safety).
    # These run before any retrieval or model call and answer directly. The
    # query text is never logged -- only the guardrail kind -- since it can
    # contain sensitive content.
    guard = screen_query(query)
    if guard is not None:
        reply, kind = guard
        logger.info(f"guardrail triggered (kind={kind}, chars={len(query)})")
        return {
            "query": query,
            "answer": reply,
            "grounded": False,
            "citations": [],
            "retrieved_count": 0,
            # Tells the widget not to offer human escalation for these.
            "guardrail": kind,
        }

    search_query = _retrieval_query(query, history)
    settings = get_settings()
    if limit is None:
        limit = settings.retrieval_chat_limit

    # Embedded once here and handed to both probe_relevance() and
    # retrieval_search() below -- they used to each embed search_query
    # independently, paying BGE-M3 inference cost twice per turn for
    # identical input. Small (~0.5-1s) but free to eliminate.
    # With history, the new message is also embedded on its own (one batch,
    # one model call) for the follow-up check below.
    standalone = search_query != query
    embeddings = BgeM3Embedder(batch_size=2).embed([search_query, query] if standalone else [search_query])
    query_embedding = embeddings[0]
    filters = dict(
        service_category=service_category,
        canonical=canonical,
        source_id=source_id,
        jurisdiction=jurisdiction,
        applicant_variant=applicant_variant,
    )

    relevance = probe_relevance(search_query, query_embedding=query_embedding, **filters)
    if _mentions_domain_keyword(query) or is_more_details_request(query):
        on_topic = True
    elif standalone:
        # See Settings.retrieval_followup_min_relevance: the earlier question
        # alone must not carry an unrelated new message through.
        own = probe_relevance(query, query_embedding=embeddings[1], **filters)
        on_topic = relevance >= settings.retrieval_min_relevance and own >= settings.retrieval_followup_min_relevance
    else:
        on_topic = relevance >= settings.retrieval_min_relevance
    if not on_topic:
        # The question text is deliberately not logged: visitors can type
        # personal details (names, passport numbers) into it.
        logger.info(f"declined as out of scope (relevance={relevance:.3f}, chars={len(query)})")
        if not has_approved_content():
            # Nothing is approved yet, so *every* question scores 0. Saying "this
            # is outside what I can help with" would be wrong; say it hasn't been
            # reviewed yet instead.
            return {
                "query": query,
                "answer": NO_CONTEXT_ANSWER,
                "grounded": False,
                "citations": [],
                "retrieved_count": 0,
            }
        return {
            "query": query,
            "answer": OUT_OF_SCOPE_ANSWER,
            "grounded": False,
            "citations": [],
            "retrieved_count": 0,
            # Tells the widget not to offer "pass this along to the consulate":
            # an off-topic question is not something for consular staff.
            "out_of_scope": True,
        }

    chunks = retrieval_search(
        search_query,
        limit=limit,
        service_category=service_category,
        canonical=canonical,
        source_id=source_id,
        jurisdiction=jurisdiction,
        applicant_variant=applicant_variant,
        query_embedding=query_embedding,
    )

    if not chunks:
        return {
            "query": query,
            "answer": NO_CONTEXT_ANSWER,
            "grounded": False,
            "citations": [],
            "retrieved_count": 0,
        }

    # Add the Consulate's contact page (see Settings.retrieval_pinned_source_urls)
    # so a question the retrieved sources can't fully answer can still end with
    # the right, cited contact. Skipped when the caller narrowed the search
    # to a category/source, which asks for exactly those sources.
    pinned_urls = [u.strip() for u in settings.retrieval_pinned_source_urls.split(",") if u.strip()]
    if pinned_urls and not (service_category or source_id or canonical is not None):
        have = {c["chunk_id"] for c in chunks}
        chunks = chunks + [c for c in pinned_chunks(pinned_urls) if c["chunk_id"] not in have]

    messages = build_rag_messages(query, chunks, history, full_answer=bool(history) and is_more_details_request(query))
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
        # The detail goes to the log only. It used to be returned to the
        # browser as `generation_error`, which leaked internals (hostnames,
        # account names, provider messages); the widget only needs to know that
        # it happened, so the value is a fixed code.
        logger.warning(f"generation backend failed: {exc}")
        return {
            "query": query,
            "answer": GENERATION_UNAVAILABLE_ANSWER,
            "grounded": False,
            "citations": [],
            "retrieved_count": len(chunks),
            "generation_error": "generation_unavailable",
        }
    answer = normalize_citation_groups(strip_thinking(raw))

    # Collapse [1][2][3]... that all point to the same source document into one
    # citation, renumbering the answer's bracket markers to match.
    answer, citations = _dedupe_citations_by_source(answer, chunks)

    # "grounded" means the answer is backed by sources it actually cites. The
    # widget uses it to decide whether to offer the "raise a query with the
    # consulate" form, so it must be False when the model itself says it has no
    # approved information (SYSTEM_PROMPT rule 3), when it cites nothing, or when
    # it returns nothing at all. Previously this was always True once the
    # model produced any text, so a model-written "I don't have approved
    # information" reply never offered the escalation.
    declined = _DECLINE_RE.search(answer) is not None
    if not citations and not declined and answer.strip():
        # The model answered but left out the [n] markers it was asked for.
        answer, citations = _attribute_uncited(answer, chunks)
    grounded = bool(citations) and not declined
    if declined:
        citations = []  # no source chips under an answer that says nothing was found

    # The [n] markers have done their job (picking the citations above); the
    # visitor sees the sources as chips after the answer instead of numbers in
    # the text. A short answer ends with the offer of more details (prompt rule
    # 6c); the full answer given after "yes" does not repeat it (rule 11).
    _attach_source_titles(citations)
    answer = strip_citation_markers(answer)
    if grounded and not is_more_details_request(query):
        answer = ensure_more_details_offer(answer)
    else:
        # A refusal ("I don't have approved information...") must not offer
        # more details -- the model sometimes adds the line anyway -- and the
        # full answer after "yes" does not repeat it.
        answer = answer.replace(MORE_DETAILS_OFFER, "").rstrip()

    return {
        "query": query,
        "answer": answer,
        "grounded": grounded,
        "citations": citations,
        "retrieved_count": len(chunks),
        "uncited_sources_retrieved": len(chunks) - len(citations),
    }
