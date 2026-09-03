"""
The actual approve/reject operation for Phase 5's admin review API.

Updates review_status in BOTH Postgres (the system of record) and Qdrant
(whose payload copy is used for retrieval-time filtering) so the two
stores can't quietly drift apart the way `used_ocr` briefly did in Phase
4 — same fix, applied proactively this time instead of after the fact.
Every decision is also written to `audit_log` with an actor, so there's a
durable record of who approved/rejected what chunk and when (and why, if a
reason was given).
"""

from typing import Optional
from uuid import UUID

from app.db.audit_repo import log_event
from app.db.chunks_repo import get_chunk, set_review_status
from app.vectorstore.qdrant_store import set_payload_fields

VALID_DECISIONS = {"approved", "rejected"}


class ChunkNotFound(Exception):
    pass


def decide_chunk(
    conn,
    qdrant_client,
    chunk_id: UUID,
    decision: str,
    actor: str,
    reason: Optional[str] = None,
) -> dict:
    if decision not in VALID_DECISIONS:
        raise ValueError(f"decision must be one of {sorted(VALID_DECISIONS)}, got {decision!r}")

    chunk = get_chunk(conn, chunk_id)
    if chunk is None:
        raise ChunkNotFound(str(chunk_id))

    # Do BOTH Postgres writes first, then update Qdrant last. If the Qdrant
    # update raises, the exception propagates and get_conn rolls the whole
    # Postgres transaction back, so the two stores revert together (both
    # keep the old status) rather than drifting. Ordering Qdrant in the
    # middle -- as this did before -- meant a failure in log_event AFTER the
    # Qdrant write left Qdrant updated but Postgres rolled back.
    set_review_status(conn, chunk_id, decision)
    log_event(
        conn,
        entity_type="chunk",
        entity_id=chunk_id,
        action=decision,
        actor=actor,
        details={"reason": reason} if reason else None,
    )
    if chunk.get("qdrant_point_id"):
        set_payload_fields(qdrant_client, [chunk["qdrant_point_id"]], {"review_status": decision})

    return {
        "chunk_id": str(chunk_id),
        "source_id": str(chunk["source_id"]),
        "previous_review_status": chunk["review_status"],
        "review_status": decision,
    }
