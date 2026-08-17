"""
Phase 5: "view diff against previous version".

Compares a source's two most recent fetched/embedded versions at two
levels: a human-readable unified diff of the whole document's chunk text
(for a reviewer to actually read what changed), and a chunk-level summary
(added/removed chunk indices, by content_hash membership rather than raw
index alignment — chunk boundaries can shift between versions even when
most content is unchanged, so hash-set comparison is the honest way to
tell "genuinely new/removed text" from "same text, renumbered").
"""

import difflib
from uuid import UUID

from app.db.chunks_repo import list_by_source_version
from app.db.source_versions_repo import list_versions
from app.db.sources_repo import get_by_id


class NoDiffAvailable(Exception):
    """Raised when a source has fewer than 2 fetched versions — nothing to diff yet."""


def diff_latest_versions(conn, source_id: UUID) -> dict:
    source = get_by_id(conn, source_id)
    if source is None:
        raise ValueError(f"no such source: {source_id}")

    versions = list_versions(conn, source_id)
    if len(versions) < 2:
        raise NoDiffAvailable(
            f"source {source_id} has {len(versions)} fetched version(s) — need at least 2 to diff"
        )

    prev_version, latest_version = versions[-2], versions[-1]

    prev_chunks = list_by_source_version(conn, prev_version.id)
    latest_chunks = list_by_source_version(conn, latest_version.id)

    prev_text_lines = "\n\n".join(c["chunk_text"] for c in prev_chunks).splitlines()
    latest_text_lines = "\n\n".join(c["chunk_text"] for c in latest_chunks).splitlines()

    unified = list(
        difflib.unified_diff(
            prev_text_lines,
            latest_text_lines,
            fromfile=f"v{prev_version.version}",
            tofile=f"v{latest_version.version}",
            lineterm="",
        )
    )

    prev_by_hash = {c["content_hash"]: c for c in prev_chunks}
    latest_by_hash = {c["content_hash"]: c for c in latest_chunks}

    added = [c["chunk_index"] for c in latest_chunks if c["content_hash"] not in prev_by_hash]
    removed = [c["chunk_index"] for c in prev_chunks if c["content_hash"] not in latest_by_hash]
    unchanged_count = len(set(prev_by_hash) & set(latest_by_hash))

    return {
        "source_id": str(source_id),
        "title": source.title,
        "url": source.url,
        "previous_version": prev_version.version,
        "latest_version": latest_version.version,
        "previous_chunk_count": len(prev_chunks),
        "latest_chunk_count": len(latest_chunks),
        "unchanged_chunk_count": unchanged_count,
        "added_chunk_indices": sorted(added),
        "removed_chunk_indices": sorted(removed),
        "unified_diff": "\n".join(unified),
    }
