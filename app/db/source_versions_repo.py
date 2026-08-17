"""
CRUD for `source_versions` — one row per successfully fetched content
version of a source. Versions increment per source; a fetch that produces
the same content_hash as the current latest version is NOT a new version
(caller decides whether to record one).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID

from psycopg2.extras import RealDictCursor


@dataclass
class SourceVersion:
    id: UUID
    source_id: UUID
    version: int
    content_hash: str
    retrieval_date: datetime
    source_last_modified: Optional[datetime]
    raw_content_path: str
    extracted_text_path: Optional[str]
    fetch_status: str
    created_at: Optional[datetime] = None


def get_latest_version(conn, source_id: UUID) -> Optional[SourceVersion]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT * FROM source_versions
            WHERE source_id = %s
            ORDER BY version DESC
            LIMIT 1;
            """,
            (str(source_id),),
        )
        row = cur.fetchone()
    return SourceVersion(**row) if row else None


def create_version(
    conn,
    *,
    source_id: UUID,
    content_hash: str,
    retrieval_date: datetime,
    raw_content_path: str,
    source_last_modified: Optional[datetime] = None,
    extracted_text_path: Optional[str] = None,
    fetch_status: str = "success",
) -> SourceVersion:
    latest = get_latest_version(conn, source_id)
    next_version = (latest.version + 1) if latest else 1

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            INSERT INTO source_versions (
                source_id, version, content_hash, retrieval_date,
                source_last_modified, raw_content_path, extracted_text_path,
                fetch_status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *;
            """,
            (
                str(source_id), next_version, content_hash, retrieval_date,
                source_last_modified, raw_content_path, extracted_text_path,
                fetch_status,
            ),
        )
        row = cur.fetchone()
    return SourceVersion(**row)


def list_versions(conn, source_id: UUID) -> list[SourceVersion]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT * FROM source_versions WHERE source_id = %s ORDER BY version;",
            (str(source_id),),
        )
        rows = cur.fetchall()
    return [SourceVersion(**row) for row in rows]
