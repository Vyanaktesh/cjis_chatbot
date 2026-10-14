"""
Records each chat question and its answer in Postgres (`chat_logs`), so the
team can see what visitors ask and which questions the knowledge base can't
answer yet (see scripts/chat_log_report.py).

- Personal details are masked first (app/core/redact.py). The question is
  masked in full; in the answer only the visitor's own details are removed,
  so the Consulate's public helplines and emails stay readable.
- No IP address is stored.
- Rows older than Settings.chat_log_retention_days (60) are deleted, checked at
  most once an hour per process.
- Runs after the reply has been sent (a FastAPI background task) and never
  raises: a logging failure must not affect a visitor's answer.
"""

import threading
import time
from typing import Any, Optional

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.core.redact import find_personal_details, mask_personal_details

logger = get_logger(__name__)

_PURGE_EVERY_SECONDS = 3600
_last_purge = 0.0
_purge_lock = threading.Lock()


def outcome_of(result: dict[str, Any]) -> str:
    if result.get("smalltalk"):
        return "smalltalk"
    if result.get("guardrail"):
        return "guardrail"
    if result.get("out_of_scope"):
        return "out_of_scope"
    if result.get("generation_error"):
        return "error"
    return "answered" if result.get("grounded") else "declined"


def record_chat(
    *,
    question: str,
    result: dict[str, Any],
    session_id: Optional[str],
    mode: str,
    is_followup: bool,
    latency_ms: int,
    search_query: Optional[str] = None,
) -> None:
    settings = get_settings()
    if not settings.chat_log_enabled:
        return
    try:
        from app.db.chat_log_repo import insert_chat_log
        from app.db.connection import get_conn

        details = find_personal_details(question)
        if search_query:
            details += find_personal_details(search_query)
        with get_conn() as conn:
            insert_chat_log(
                conn,
                session_id=session_id,
                mode=mode,
                is_followup=is_followup,
                question=mask_personal_details(question),
                search_query=mask_personal_details(search_query, also=details) if search_query else None,
                answer=mask_personal_details(result.get("answer") or "", also=details),
                outcome=outcome_of(result),
                source_urls=[c["source_url"] for c in result.get("citations") or []],
                retrieved_count=result.get("retrieved_count"),
                latency_ms=latency_ms,
            )
            _purge_old_rows(conn, settings.chat_log_retention_days)
    except Exception as exc:  # noqa: BLE001 -- see module docstring
        logger.warning(f"chat log not written: {type(exc).__name__}: {exc}")


def _purge_old_rows(conn, days: int) -> None:
    global _last_purge
    with _purge_lock:
        if time.monotonic() - _last_purge < _PURGE_EVERY_SECONDS and _last_purge:
            return
        _last_purge = time.monotonic()
    from app.db.chat_log_repo import delete_older_than

    deleted = delete_older_than(conn, days)
    if deleted:
        logger.info(f"chat log retention: deleted {deleted} row(s) older than {days} days")
