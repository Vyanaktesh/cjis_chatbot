#!/usr/bin/env python3
"""
Phase 3 verification script.

Runs the extractor + section-aware chunker against every source that has
been fetched (Phase 2's fetch_all_sources.py — should be all 47), using
each source's LATEST version. Writes full chunk dumps to
data/chunks_preview/<source_id>.json for manual inspection, prints a
per-source summary, and specifically prints one real checklist's chunk
boundaries so it's easy to eyeball that no requirement got split
mid-item.

Usage:
    python scripts/run_extraction_test.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking.chunker import chunk_blocks, estimate_tokens  # noqa: E402
from app.chunking.tag_metadata import build_chunk_records  # noqa: E402
from app.core.logging_config import configure_logging  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.db.connection import get_conn  # noqa: E402
from app.db.source_versions_repo import get_latest_version  # noqa: E402
from app.db.sources_repo import list_sources  # noqa: E402
from app.extraction.html_extractor import extract_html  # noqa: E402
from app.extraction.pdf_extractor import extract_pdf  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PREVIEW_DIR = REPO_ROOT / "data" / "chunks_preview"

# A real checklist to print in full for manual "no split mid-requirement"
# inspection — the most common OCI applicant profile per the registry.
SPOTLIGHT_URL = "https://services.vfsglobal.com/one-pager/india/united-states-of-america/oci-services/pdf/Foreign-national-previously-Indian-passport-holder-(adult)-2025.pdf"


def _serialize_record(r: dict) -> dict:
    out = dict(r)
    out["source_id"] = str(out["source_id"])
    out["source_version_id"] = str(out["source_version_id"])
    out["retrieval_date"] = out["retrieval_date"].isoformat() if out["retrieval_date"] else None
    out["source_last_modified"] = (
        out["source_last_modified"].isoformat() if out["source_last_modified"] else None
    )
    return out


def process_one(source, version) -> list[dict]:
    raw_path = REPO_ROOT / version.raw_content_path
    raw_bytes = raw_path.read_bytes()

    if source.source_type == "pdf":
        blocks = extract_pdf(raw_bytes)
    else:
        blocks = extract_html(raw_bytes)

    chunks = chunk_blocks(blocks)
    return build_chunk_records(chunks, source, version), blocks


def main() -> int:
    configure_logging(get_settings().log_level)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    with get_conn() as conn:
        sources = list_sources(conn, active_only=True)
        source_versions = {s.id: get_latest_version(conn, s.id) for s in sources}

    print(f"{len(sources)} active sources; extracting + chunking each at its latest version...\n")

    summary_rows = []
    spotlight_records = None
    zero_chunk_sources = []

    for source in sources:
        version = source_versions.get(source.id)
        if version is None:
            print(f"SKIP (no fetched version yet): {source.title or source.url}")
            zero_chunk_sources.append(source)
            continue

        try:
            records, blocks = process_one(source, version)
        except Exception as exc:
            print(f"FAIL  {source.title or source.url}: {exc}")
            zero_chunk_sources.append(source)
            continue

        token_counts = [estimate_tokens(r["chunk_text"]) for r in records]
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        block_type_hist = {}
        for b in blocks:
            block_type_hist[b.type.value] = block_type_hist.get(b.type.value, 0) + 1

        summary_rows.append(
            {
                "title": source.title,
                "url": source.url,
                "source_type": source.source_type,
                "num_blocks": len(blocks),
                "num_chunks": len(records),
                "avg_tokens": round(avg_tokens, 1),
                "block_types": block_type_hist,
            }
        )

        if len(records) == 0:
            zero_chunk_sources.append(source)

        out_path = PREVIEW_DIR / f"{source.id}.json"
        out_path.write_text(
            json.dumps([_serialize_record(r) for r in records], indent=2), encoding="utf-8"
        )

        if source.url == SPOTLIGHT_URL:
            spotlight_records = records

        print(
            f"[{source.source_type:>4}] {len(blocks):>4} blocks -> {len(records):>3} chunks "
            f"(avg {avg_tokens:.0f} tok)  {source.title}"
        )

    print("\n=== Summary ===")
    total_chunks = sum(r["num_chunks"] for r in summary_rows)
    print(f"sources processed: {len(summary_rows)} / {len(sources)}")
    print(f"total chunks:       {total_chunks}")
    print(f"zero-chunk sources: {len(zero_chunk_sources)}")
    for s in zero_chunk_sources:
        print(f"  - {s.title or s.url}")

    if spotlight_records:
        print("\n=== Spotlight: full chunk boundaries for a real checklist ===")
        print(f"URL: {SPOTLIGHT_URL}")
        for r in spotlight_records:
            print(f"\n--- chunk {r['chunk_index']} ({r['_block_types']}) ---")
            print(r["chunk_text"])
    else:
        print("\nWARNING: spotlight URL not found among processed sources.")

    print(f"\nFull per-source chunk dumps written to: {PREVIEW_DIR.relative_to(REPO_ROOT)}/")

    return 0 if not zero_chunk_sources else 1


if __name__ == "__main__":
    raise SystemExit(main())
