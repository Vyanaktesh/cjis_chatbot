"""
RAG prompt construction: turns a user question + retrieved (already
approved-only, per Phase 6) chunks into a strict, grounded chat prompt.

This is a government consular chatbot — the generation step must never
answer from the model's own background knowledge, never invent document
requirements, fees, or eligibility rules, and must say plainly when the
retrieved context doesn't cover the question rather than guess. The
system prompt below is deliberately blunt about this rather than relying
on the model's general instruction-following.
"""

import re
from typing import Any, Optional

SYSTEM_PROMPT = """You are an assistant answering questions about Indian consular services (passport, OCI, visa, and related services) STRICTLY using the numbered source excerpts provided in the user's latest message.

Rules, no exceptions:
1. Answer ONLY using information present in the numbered sources given in the CURRENT message. Do not use any outside knowledge, even if you believe it to be true.
2. Every factual claim you make (a document requirement, a fee, a timeline, an eligibility rule) MUST be immediately followed by the bracket number(s) of the source(s) it came from, like this [1] or [2][3].
3. If the sources do not contain enough information to answer the question, say so plainly — for example: "I don't have approved information covering that. Please contact the consulate directly." Do not guess, extrapolate, or fill gaps with general knowledge.
4. If sources disagree with each other, say so explicitly and cite both, rather than picking one silently.
5. Never invent a source number that wasn't given to you.
6. Be concise and direct. This is a practical service-information lookup, not an essay.
6a. Never use an em dash (—) or en dash (–) in your response. Use a period, comma, or regular hyphen instead.
7. Earlier turns in this conversation (if any) may be shown to you before the current message — use them ONLY to understand what the user is referring to (e.g. resolving "what about for a child" after an OCI question). Their citation numbers were relative to that earlier turn's own source list and do NOT apply now. Every claim in your new answer must still be justified by, and cited to, the numbered sources in the CURRENT message only."""


def _render_sources(chunks: list[dict[str, Any]]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        heading = " > ".join(chunk.get("heading_trail") or [])
        header = f"[{i}] Source: {chunk['source_url']}"
        if heading:
            header += f"\nSection: {heading}"
        blocks.append(f"{header}\n{chunk['chunk_text']}")
    return "\n\n".join(blocks)


# --- Phase 8: bounded sliding-window conversation memory ---
#
# Full unbounded history would eventually overflow the model's 4096-token
# context (and, well before that, blow the CPU-only generation latency
# budget even further). Standard practice for a resource-constrained RAG
# chatbot is a *bounded* window: keep the most recent N turns, and trim
# further by an approximate character/token budget so a couple of unusually
# long messages can't silently push everything else out. Trimming always
# drops the OLDEST turns first, keeping the most recent ones intact.
HISTORY_MAX_TURNS = 5  # a "turn" = one user message + one assistant reply
HISTORY_CHAR_BUDGET = 2400  # ~600 tokens at a ~4 chars/token rule of thumb


def _trim_history(history: Optional[list[dict[str, str]]]) -> list[dict[str, str]]:
    """Keeps at most the last HISTORY_MAX_TURNS turns from `history`
    (a chronological list of {"role": "user"|"assistant", "content": str}),
    then further trims from the oldest end until the total content length
    fits HISTORY_CHAR_BUDGET. Returns [] if history is empty/None — the
    no-history behavior is unchanged from before this feature existed."""
    if not history:
        return []

    # Keep only well-formed messages, most-recent-first for windowing.
    cleaned = [
        m for m in history if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    windowed = cleaned[-(HISTORY_MAX_TURNS * 2) :]

    total = sum(len(m["content"]) for m in windowed)
    while windowed and total > HISTORY_CHAR_BUDGET:
        dropped = windowed.pop(0)
        total -= len(dropped["content"])

    return windowed


def build_rag_messages(
    query: str,
    chunks: list[dict[str, Any]],
    history: Optional[list[dict[str, str]]] = None,
) -> list[dict[str, str]]:
    """Builds the chat messages for a grounded RAG answer: a system prompt,
    a bounded window of prior conversation turns (see _trim_history), and
    the current turn's sources + question. Caller is expected to have
    already checked `chunks` is non-empty — see
    app/generation/service.py, which short-circuits before ever calling
    the model when retrieval finds nothing, rather than trusting the
    model to correctly decline on empty context."""
    sources_block = _render_sources(chunks)
    user_content = (
        f"Numbered sources:\n\n{sources_block}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the numbered sources above, with bracket citations. /no_think"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(_trim_history(history))
    messages.append({"role": "user", "content": user_content})
    return messages


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_CITATION_RE = re.compile(r"\[(\d+)\]")
_DASH_RE = re.compile(r"\s*[—–]\s*")

_RANGE_WORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}


def _is_range(left: str, right: str) -> bool:
    """A dash between two numbers/times/weekdays/months is a RANGE
    ("10 – 15 days", "$25 – $50", "9 am – 5 pm", "Monday – Friday"), not a
    clause break. Turning a range into a comma changes the meaning of a fee,
    a processing time or office hours, so ranges must be handled separately."""
    left_tokens = left.split()[-2:]
    right_tokens = right.split()[:2]
    if any(c.isdigit() for t in left_tokens for c in t) and any(
        c.isdigit() for t in right_tokens for c in t
    ):
        return True
    left_word = re.sub(r"\W+", "", left_tokens[-1]).lower() if left_tokens else ""
    right_word = re.sub(r"\W+", "", right_tokens[0]).lower() if right_tokens else ""
    return left_word in _RANGE_WORDS and right_word in _RANGE_WORDS


def _strip_dashes(text: str) -> str:
    """Rule 6a in SYSTEM_PROMPT asks the model not to use em/en dashes, but
    LLMs reach for them out of habit regardless of instruction; this is a
    deterministic safety net. A dash used as a clause break ("word — word")
    becomes ", ". A dash used as a RANGE keeps its meaning: with spaces it
    becomes " to " ("10 – 15 days" -> "10 to 15 days"), without spaces a plain
    hyphen ("5–10" -> "5-10")."""

    def replace(match: "re.Match[str]") -> str:
        left = match.string[: match.start()]
        right = match.string[match.end():]
        if _is_range(left, right):
            return " to " if len(match.group(0)) > 1 else "-"
        return ", "

    return _DASH_RE.sub(replace, text)


def strip_thinking(raw_text: str) -> str:
    """Qwen3 wraps chain-of-thought in <think>...</think>; /no_think should
    make this an empty block, but strip it defensively either way rather
    than assume the suffix always works across model/runtime versions.
    Also normalizes away em/en dashes (see _strip_dashes) since the model
    doesn't reliably follow the SYSTEM_PROMPT instruction against them."""
    return _strip_dashes(_THINK_BLOCK_RE.sub("", raw_text).strip())


def extract_cited_indices(answer_text: str) -> list[int]:
    """Which [n] source numbers the model actually cited, deduplicated,
    in first-appearance order — used to report only the sources actually
    referenced rather than everything that was retrieved."""
    seen = []
    for match in _CITATION_RE.finditer(answer_text):
        n = int(match.group(1))
        if n not in seen:
            seen.append(n)
    return seen
