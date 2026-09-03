"""
CRUD for the `sources` table — the registry of URLs to fetch.

Upserts key on `url` (the natural unique key), so re-running the loader
against the same registry file is always safe.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID

from psycopg2.extras import RealDictCursor


@dataclass
class Source:
    id: UUID
    url: str
    source_type: str
    service_category: str
    canonical: bool
    jurisdiction: list
    applicant_variant: list
    active: bool
    notes: Optional[str]
    title: Optional[str]
    source_group: Optional[str]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# Explicit column list (never `SELECT *`): a column added to the `sources`
# table must not be passed as an unexpected kwarg to Source(**row) and break
# every read path. A hardcoded constant, so interpolating it into SQL is safe.
_COLUMNS = (
    "id, url, source_type, service_category, canonical, jurisdiction, "
    "applicant_variant, active, notes, title, source_group, created_at, updated_at"
)


def upsert_source(
    conn,
    *,
    url: str,
    source_type: str,
    service_category: str,
    canonical: bool = False,
    jurisdiction: Optional[list] = None,
    applicant_variant: Optional[list] = None,
    active: bool = True,
    notes: Optional[str] = None,
    title: Optional[str] = None,
    source_group: Optional[str] = None,
) -> Source:
    jurisdiction = jurisdiction if jurisdiction is not None else ["all"]
    applicant_variant = applicant_variant if applicant_variant is not None else []

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            INSERT INTO sources (
                url, source_type, service_category, canonical,
                jurisdiction, applicant_variant, active, notes, title, source_group
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO UPDATE SET
                source_type = EXCLUDED.source_type,
                service_category = EXCLUDED.service_category,
                canonical = EXCLUDED.canonical,
                jurisdiction = EXCLUDED.jurisdiction,
                applicant_variant = EXCLUDED.applicant_variant,
                active = EXCLUDED.active,
                notes = EXCLUDED.notes,
                title = EXCLUDED.title,
                source_group = EXCLUDED.source_group
            RETURNING id, url, source_type, service_category, canonical,
                      jurisdiction, applicant_variant, active, notes, title, source_group;
            """,
            (
                url, source_type, service_category, canonical,
                jurisdiction, applicant_variant, active, notes, title, source_group,
            ),
        )
        row = cur.fetchone()
    return Source(**row)


def get_by_url(conn, url: str) -> Optional[Source]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"SELECT {_COLUMNS} FROM sources WHERE url = %s;", (url,))
        row = cur.fetchone()
    return Source(**row) if row else None


def get_by_id(conn, source_id: UUID) -> Optional[Source]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"SELECT {_COLUMNS} FROM sources WHERE id = %s;", (str(source_id),))
        row = cur.fetchone()
    return Source(**row) if row else None


def list_sources(conn, active_only: bool = True) -> list[Source]:
    query = f"SELECT {_COLUMNS} FROM sources"
    if active_only:
        query += " WHERE active = true"
    query += " ORDER BY source_group, title;"
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return [Source(**row) for row in rows]


def set_active(conn, source_id: UUID, active: bool) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE sources SET active = %s WHERE id = %s;",
            (active, str(source_id)),
        )
