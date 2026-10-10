"""
CRUD for the `chunks` table — Postgres mirror of what's embedded and
upserted into Qdrant. Chunk IDs are derived deterministically (see
scripts/embed_and_index_all.py) from (source_version_id, chunk_index), so
re-running the embed step is idempotent: the same chunk gets the same ID
in both Postgres and Qdrant, and this upsert updates it in place rather
than duplicating it.
"""

from typing import Any, Optional
from uuid import UUID

from psycopg2.extras import RealDictCursor


def upsert_chunk(conn, record: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chunks (
                id, source_id, source_version_id, chunk_index, chunk_text, content_hash,
                source_url, retrieval_date, source_last_modified, service_category, canonical,
                jurisdiction, applicant_variant, review_status, version, embedding_model,
                qdrant_point_id, used_ocr
            ) VALUES (
                %(id)s, %(source_id)s, %(source_version_id)s, %(chunk_index)s, %(chunk_text)s,
                %(content_hash)s, %(source_url)s, %(retrieval_date)s, %(source_last_modified)s,
                %(service_category)s, %(canonical)s, %(jurisdiction)s, %(applicant_variant)s,
                %(review_status)s, %(version)s, %(embedding_model)s, %(qdrant_point_id)s, %(used_ocr)s
            )
            ON CONFLICT (id) DO UPDATE SET
                chunk_text = EXCLUDED.chunk_text,
                content_hash = EXCLUDED.content_hash,
                source_last_modified = EXCLUDED.source_last_modified,
                embedding_model = EXCLUDED.embedding_model,
                qdrant_point_id = EXCLUDED.qdrant_point_id,
                used_ocr = EXCLUDED.used_ocr,
                review_status = EXCLUDED.review_status,
                updated_at = now();
            """,
            {
                "id": str(record["id"]),
                "source_id": str(record["source_id"]),
                "source_version_id": str(record["source_version_id"]),
                "chunk_index": record["chunk_index"],
                "chunk_text": record["chunk_text"],
                "content_hash": record["content_hash"],
                "source_url": record["source_url"],
                "retrieval_date": record["retrieval_date"],
                "source_last_modified": record["source_last_modified"],
                "service_category": record["service_category"],
                "canonical": record["canonical"],
                "jurisdiction": record["jurisdiction"],
                "applicant_variant": record["applicant_variant"],
                "review_status": record["review_status"],
                "version": record["version"],
                "embedding_model": record["embedding_model"],
                "qdrant_point_id": str(record["qdrant_point_id"]),
                "used_ocr": record["used_ocr"],
            },
        )


def get_review_state(conn, chunk_ids: list) -> dict[str, dict]:
    """Returns {chunk_id: {review_status, content_hash}} for those of
    `chunk_ids` that already exist, locking the rows (FOR UPDATE) so a
    reviewer's approve/reject can't land between this read and the upsert
    that follows it in the same transaction."""
    if not chunk_ids:
        return {}
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT id, review_status, content_hash FROM chunks "
            "WHERE id = ANY(%s::uuid[]) FOR UPDATE;",
            ([str(i) for i in chunk_ids],),
        )
        return {str(row["id"]): row for row in cur.fetchall()}


def count_chunks(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks;")
        return cur.fetchone()[0]


def counts_by_category(conn) -> list[tuple[str, int]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT service_category, count(*) FROM chunks GROUP BY service_category ORDER BY 2 DESC;"
        )
        return cur.fetchall()


def set_review_status(conn, chunk_id: UUID, status: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE chunks SET review_status = %s WHERE id = %s;",
            (status, str(chunk_id)),
        )


def get_chunk(conn, chunk_id: UUID) -> Optional[dict]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM chunks WHERE id = %s;", (str(chunk_id),))
        return cur.fetchone()


def list_pending(
    conn,
    *,
    service_category: Optional[str] = None,
    source_id: Optional[UUID] = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """Returns (rows, total_count) for chunks with review_status='pending_review'."""
    where = ["review_status = 'pending_review'"]
    params: list[Any] = []
    if service_category is not None:
        where.append("service_category = %s")
        params.append(service_category)
    if source_id is not None:
        where.append("source_id = %s")
        params.append(str(source_id))
    where_clause = " AND ".join(where)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"SELECT count(*) AS n FROM chunks WHERE {where_clause};", params)
        total = cur.fetchone()["n"]

        cur.execute(
            f"""
            SELECT * FROM chunks
            WHERE {where_clause}
            ORDER BY source_id, chunk_index
            LIMIT %s OFFSET %s;
            """,
            params + [limit, offset],
        )
        rows = cur.fetchall()
    return rows, total


def list_by_source_version(conn, source_version_id: UUID) -> list[dict]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT * FROM chunks WHERE source_version_id = %s ORDER BY chunk_index;",
            (str(source_version_id),),
        )
        return cur.fetchall()


def list_by_source_excluding_version(conn, source_id: UUID, source_version_id: UUID) -> list[dict]:
    """All chunks for a source that do NOT belong to the given version — used
    to find what needs to be superseded once a new version has been embedded."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT * FROM chunks
            WHERE source_id = %s AND source_version_id != %s
            ORDER BY version, chunk_index;
            """,
            (str(source_id), str(source_version_id)),
        )
        return cur.fetchall()


def mark_superseded(conn, chunk_ids: list[UUID]) -> None:
    if not chunk_ids:
        return
    with conn.cursor() as cur:
        cur.execute(
            # explicit ::uuid[] cast — without it Postgres infers the
            # parameter as text[], which doesn't match the uuid `id`
            # column and raises "operator does not exist: uuid = text"
            "UPDATE chunks SET review_status = 'superseded' WHERE id = ANY(%s::uuid[]);",
            ([str(i) for i in chunk_ids],),
        )


def counts_by_source(conn) -> list[dict]:
    """Per-source chunk counts by review_status, plus latest version number —
    the overview list the admin review UI starts from."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                s.id AS source_id,
                s.title,
                s.url,
                s.service_category,
                s.canonical,
                s.source_group,
                max(c.version) AS latest_chunked_version,
                count(c.id) FILTER (WHERE c.review_status = 'pending_review') AS pending_count,
                count(c.id) FILTER (WHERE c.review_status = 'approved') AS approved_count,
                count(c.id) FILTER (WHERE c.review_status = 'rejected') AS rejected_count,
                count(c.id) FILTER (WHERE c.review_status = 'superseded') AS superseded_count,
                count(c.id) AS total_chunks
            FROM sources s
            LEFT JOIN chunks c ON c.source_id = s.id
            GROUP BY s.id, s.title, s.url, s.service_category, s.canonical, s.source_group
            ORDER BY s.source_group, s.title;
            """
        )
        return cur.fetchall()
