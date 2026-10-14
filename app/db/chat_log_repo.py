"""
Writer for `chat_logs` (db/migrations/0005_chat_logs.sql): one row per question
asked in the chat widget, with personal details already masked by the caller
(app/core/chat_logging.py), plus the retention clean-up.
"""

from typing import Optional


def insert_chat_log(
    conn,
    *,
    session_id: Optional[str],
    mode: str,
    is_followup: bool,
    question: str,
    search_query: Optional[str],
    answer: Optional[str],
    outcome: str,
    source_urls: list[str],
    retrieved_count: Optional[int],
    latency_ms: Optional[int],
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chat_logs (session_id, mode, is_followup, question, search_query,
                                   answer, outcome, source_urls, retrieved_count, latency_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """,
            (session_id, mode, is_followup, question, search_query, answer, outcome,
             source_urls, retrieved_count, latency_ms),
        )


def delete_older_than(conn, days: int) -> int:
    """Deletes rows older than `days` days; returns how many."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chat_logs WHERE created_at < now() - make_interval(days => %s);", (days,))
        return cur.rowcount
