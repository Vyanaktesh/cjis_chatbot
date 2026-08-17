"""
Append-only writer for `audit_log` — every fetch attempt (success AND
failure), plus later-phase events (extraction, embedding, approve/reject),
gets a row here. This is what satisfies Phase 2's "log every attempt"
requirement in a way that's queryable, not just stdout text.
"""

from typing import Optional
from uuid import UUID

import psycopg2.extras


def log_event(
    conn,
    *,
    entity_type: str,
    entity_id: Optional[UUID],
    action: str,
    actor: str = "system",
    details: Optional[dict] = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO audit_log (entity_type, entity_id, action, actor, details)
            VALUES (%s, %s, %s, %s, %s);
            """,
            (
                entity_type,
                str(entity_id) if entity_id else None,
                action,
                actor,
                psycopg2.extras.Json(details) if details is not None else None,
            ),
        )


def list_recent(conn, limit: int = 20) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT * FROM audit_log ORDER BY created_at DESC LIMIT %s;",
            (limit,),
        )
        return cur.fetchall()
