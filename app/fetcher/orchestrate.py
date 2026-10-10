"""
Shared "fetch one URL, then persist the outcome" logic — used by both the
Phase 2 smoke test (a handful of sources) and the full registry fetch (all
47). Kept in one place so the two never drift out of sync on how versions,
raw files, and audit_log rows get written.
"""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from uuid import UUID

from app.db.audit_repo import log_event
from app.db.source_versions_repo import create_version, get_latest_version
from app.extraction.html_extractor import extract_html
from app.fetcher.playwright_fetcher import fetch_one

RAW_STORAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


def _extracted_text_fingerprint(content: bytes) -> str:
    """Hash of the page's extracted TEXT (what actually becomes chunks), not of
    its raw markup."""
    text = "\n".join(block.text for block in extract_html(content))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def same_extracted_text(
    source_type: str,
    previous_raw_path: str,
    new_content: bytes,
    *,
    base_dir: Optional[Path] = None,
) -> bool:
    """True when an HTML page's text is identical to the stored previous
    version even though its raw bytes differ.

    The fetcher saves the *rendered* page, which usually differs on every
    fetch (timestamps, tokens, visitor counters, ad slots) without any real
    change to the content. Comparing raw hashes made each of those a "new
    version", which supersedes the approved chunks and leaves the chatbot
    with nothing approved for that source until someone reviews them again.
    PDFs are compared by their bytes, as before."""
    if source_type != "html":
        return False
    try:
        base = base_dir or RAW_STORAGE_DIR.parent.parent
        previous = (base / previous_raw_path).read_bytes()
        return _extracted_text_fingerprint(previous) == _extracted_text_fingerprint(new_content)
    except Exception:  # noqa: BLE001 -- if the old file is missing/unreadable, treat the page as changed
        return False


def _local_filename(url: str, source_type: str) -> str:
    name = Path(urlparse(url).path).name or "index"
    if source_type == "html" and not name.endswith(".html"):
        name = (name or "index") + ".html"
    return name


def _make_attempt_callback(conn, source_id: Optional[UUID]):
    def _cb(attempt_number: int, success: bool, error: Optional[str]):
        log_event(
            conn,
            entity_type="source",
            entity_id=source_id,
            action="fetch_attempt",
            details={"attempt": attempt_number, "success": success, "error": error},
        )
    return _cb


def fetch_and_persist(
    context,
    conn,
    settings,
    *,
    url: str,
    source_type: str,
    source_id: Optional[UUID] = None,
) -> dict:
    """
    Fetches `url` and, on success for a registered source (source_id set),
    persists raw content + a source_versions row — unless the content_hash
    matches the existing latest version, in which case it logs
    'fetch_no_change' instead of creating a duplicate version. Every
    attempt and outcome is written to audit_log regardless of source_id,
    so ad-hoc/unregistered test URLs still get a full audit trail.
    """
    attempt_cb = _make_attempt_callback(conn, source_id)
    result = fetch_one(
        context,
        url=url,
        source_type=source_type,
        user_agent=settings.fetcher_user_agent,
        attempt_callback=attempt_cb,
    )

    outcome = {
        "url": url,
        "success": result.success,
        "attempts": result.attempts,
        "http_status": result.http_status,
        "content_hash": result.content_hash,
        "bytes": len(result.content_bytes) if result.content_bytes else 0,
        "error": result.error,
        "skipped_reason": result.skipped_reason,
        "used_request_fallback": result.used_request_fallback,
        "raw_path": None,
        "version": None,
        "unchanged": False,
    }

    if not result.success:
        log_event(
            conn,
            entity_type="source",
            entity_id=source_id,
            action="fetch_failure",
            details={
                "url": url,
                "error": result.error,
                "skipped_reason": result.skipped_reason,
                "attempts": result.attempts,
            },
        )
        return outcome

    if source_id is None:
        # ad-hoc/unregistered URL (e.g. the smoke test's broken-URL case) —
        # nothing to version, the fetch outcome itself is the point.
        return outcome

    latest = get_latest_version(conn, source_id)
    identical_bytes = latest is not None and latest.content_hash == result.content_hash
    identical_text = (
        latest is not None
        and not identical_bytes
        and same_extracted_text(source_type, latest.raw_content_path, result.content_bytes)
    )
    if identical_bytes or identical_text:
        outcome.update(raw_path=latest.raw_content_path, version=latest.version, unchanged=True)
        log_event(
            conn,
            entity_type="source",
            entity_id=source_id,
            action="fetch_no_change",
            details={
                "url": url,
                "content_hash": result.content_hash,
                "existing_version": latest.version,
                "used_request_fallback": result.used_request_fallback,
                "reason": "identical_bytes" if identical_bytes else "identical_extracted_text",
            },
        )
        return outcome

    # Each version gets its OWN directory (v1, v2, ...) — otherwise a
    # re-fetch that produces a new version silently overwrites the
    # previous version's file on disk, even though source_versions still
    # has a row (and a raw_content_path) pointing at the now-clobbered
    # path. next_version must match what create_version() below will
    # independently compute (it re-derives the same value the same way,
    # and this script runs strictly sequentially per source, so there's
    # no race between the two computations).
    next_version = (latest.version + 1) if latest else 1
    version_dir = RAW_STORAGE_DIR / str(source_id) / f"v{next_version}"
    version_dir.mkdir(parents=True, exist_ok=True)
    raw_path = version_dir / _local_filename(url, source_type)
    raw_path.write_bytes(result.content_bytes)

    version = create_version(
        conn,
        source_id=source_id,
        content_hash=result.content_hash,
        retrieval_date=datetime.now(timezone.utc),
        raw_content_path=str(raw_path.relative_to(RAW_STORAGE_DIR.parent.parent)),
        source_last_modified=result.source_last_modified,
        fetch_status="success",
    )
    # Explicit raise, not assert: assertions are stripped under `python -O`,
    # which would let a concurrent-write version mismatch silently persist a
    # row pointing at the wrong raw file instead of failing loudly.
    if version.version != next_version:
        raise RuntimeError(
            f"version numbering mismatch for source {source_id}: computed {next_version}, "
            f"DB assigned {version.version} — likely a concurrent write to the same source"
        )
    outcome.update(raw_path=str(raw_path), version=version.version)

    log_event(
        conn,
        entity_type="source_version",
        entity_id=version.id,
        action="fetch_success",
        details={
            "url": url,
            "content_hash": result.content_hash,
            "version": version.version,
            "used_request_fallback": result.used_request_fallback,
        },
    )
    return outcome
