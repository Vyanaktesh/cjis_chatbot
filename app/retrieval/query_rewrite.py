"""
Query rewriting for retrieval.

Visitors write the way people talk: "My son is 7 and his Indian passport was
lost at the hotel when we went to Disney World in July, how do I get a new
one?". Searching with all of that pulls in documents that match the story
(children, travel, July) instead of the one that matches the request (re-issue
of a lost passport for a minor), and the right checklist drops down the list
or out of it. Measured on 32 real-style questions: the correct source was the
top result for 24 asked plainly but only 17 asked with that kind of detail,
and for 2 of the detailed ones it fell out of the top 8 altogether.

So before searching, the generation model rewrites the message into a short
search query that keeps what decides the answer (the service, what the visitor
wants, who it is for, conditions such as lost or urgent) and drops the rest.
The search then runs with BOTH the visitor's own words and the rewrite, and the
two rankings are fused with equal weight (see retriever.fuse_results and
Settings.retrieval_rewrite_weight), so the original's best results always stay
in: a poor rewrite can add candidates but never push out what the visitor's
own words found. The rewrite is used only for searching: the answer is written
to the visitor's own message, and the off-topic check is unchanged.

It also resolves follow-ups ("and for my son?") from the earlier questions in
the conversation, which the plain concatenation in service._retrieval_query
only approximates.

Best-effort throughout: any failure, an empty or odd reply, or a reply slower
than Settings.query_rewrite_timeout_seconds means None, and the caller
searches with the original text exactly as before.
"""

import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

REWRITE_SYSTEM_PROMPT = """You turn a visitor's message to the help chatbot of the Consulate General of India, Atlanta into a short search query for the consulate's knowledge base (passport, OCI, visa, surrender of Indian passport / renunciation, emergency certificate, attestation, power of attorney, life certificate, birth and death registration, police clearance, consulate office information).

Keep:
- the service asked about and what the visitor wants to do or know (apply, renew, lost, damaged, change a detail, documents, fee, processing time, status, eligibility);
- who the application is for when it changes the rules: child / minor, adult, foreign spouse, US citizen of Indian origin, OCI holder;
- conditions that matter: lost, damaged, expired passport, no valid visa status, urgent / Tatkal, emergency.
Drop: back-story, names, cities, dates, feelings, greetings, and family members unless the application is for that person.
Use the consulate's terms: "passport re-issue" for renewing an Indian passport or changing details on it (name, address); "OCI card"; "surrender certificate"; "emergency certificate".
If earlier questions are given, the new message may be a follow-up: make the query stand alone (for "and for my son?" after an OCI question, write the OCI query for a minor).
Do not answer the question, do not add facts, and do not add things the visitor did not ask about (no "fee" or "documents" unless they asked about fees or documents). Output only the query, one line, under 20 words."""

_MAX_QUERY_CHARS = 200
_PREFIX_RE = re.compile(r"^(?:search\s+)?query\s*:\s*", re.IGNORECASE)

# A small pool: each rewrite is one short API call, and running it on a thread
# lets the caller embed and check the original message at the same time.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="query-rewrite")


def build_rewrite_messages(query: str, earlier_questions: Optional[list[str]] = None) -> list[dict[str, str]]:
    user = query
    if earlier_questions:
        listed = "\n".join(f"- {q}" for q in earlier_questions)
        user = f"Earlier questions in this conversation:\n{listed}\n\nNew message: {query}"
    return [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def clean_rewrite(raw: Optional[str]) -> Optional[str]:
    """The model's reply as a usable query, or None. Takes the first non-empty
    line, drops a "Query:" label and surrounding quotes, and rejects anything
    empty or implausibly long (a model that answered instead of rewriting)."""
    if not raw:
        return None
    line = next((ln.strip() for ln in raw.strip().splitlines() if ln.strip()), "")
    line = _PREFIX_RE.sub("", line).strip().strip('"\'`').strip()
    if not line or len(line) > _MAX_QUERY_CHARS:
        return None
    return line


def _rewrite(generator: Any, query: str, earlier_questions: Optional[list[str]]) -> Optional[str]:
    started = time.perf_counter()
    raw = generator.generate(build_rewrite_messages(query, earlier_questions), max_tokens=60, temperature=0.0)
    rewritten = clean_rewrite(raw)
    # The rewrite is not logged: it can still carry personal details.
    logger.info(f"query rewrite took {time.perf_counter() - started:.2f}s (ok={rewritten is not None})")
    return rewritten


def rewrite_active() -> bool:
    """Whether questions get rewritten. Only with the Gemini backend: on the
    self-hosted Qwen3/llama.cpp backend every model call takes tens of seconds
    on CPU, far more than a rewrite is worth, so that path searches with the
    original text as before."""
    settings = get_settings()
    return settings.query_rewrite_enabled and settings.generation_backend == "gemini"


def start_rewrite(generator: Any, query: str, earlier_questions: Optional[list[str]] = None) -> Optional[Future]:
    """Starts the rewrite in the background and returns its future, or None
    when rewriting is off (see rewrite_active). Collect it with
    wait_for_rewrite()."""
    if not rewrite_active():
        return None
    try:
        return _executor.submit(_rewrite, generator, query, earlier_questions)
    except Exception as exc:  # noqa: BLE001 -- best-effort, see module docstring
        logger.warning(f"query rewrite could not start: {exc}")
        return None


def wait_for_rewrite(future: Optional[Future]) -> Optional[str]:
    """The rewritten query, or None on failure or after the time budget (the
    search then uses the original text). A late reply is simply ignored."""
    if future is None:
        return None
    try:
        return future.result(timeout=get_settings().query_rewrite_timeout_seconds)
    except Exception as exc:  # noqa: BLE001 -- timeout or model error: fall back
        logger.warning(f"query rewrite unavailable, searching with the original text ({type(exc).__name__})")
        return None
