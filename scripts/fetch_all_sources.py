#!/usr/bin/env python3
"""
Fetches every active source in the registry (all 47), not just the Phase 2
smoke-test subset. This is what Phase 3 (extraction + chunking) needs to
run against — chunking has to happen on the real fetched content for every
source, not just the handful used to prove the fetcher works.

Each source gets its own DB transaction and is fully isolated: one bad URL
logs a clean failure and the run moves on to the next source rather than
stopping. A short pause between sources keeps this polite towards the real
VFS/gov.in servers being hit.

Usage:
    python scripts/fetch_all_sources.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.core.logging_config import configure_logging, get_logger  # noqa: E402
from app.db.connection import get_conn  # noqa: E402
from app.db.sources_repo import list_sources  # noqa: E402
from app.fetcher.orchestrate import fetch_and_persist  # noqa: E402
from app.fetcher.playwright_fetcher import browser_session  # noqa: E402

DELAY_BETWEEN_SOURCES_SECONDS = 1.5

logger = get_logger(__name__)


def main() -> int:
    configure_logging(get_settings().log_level)
    settings = get_settings()

    with get_conn() as conn:
        sources = list_sources(conn, active_only=True)

    print(f"Fetching {len(sources)} active source(s)...\n")

    results = []
    with browser_session(settings.fetcher_user_agent) as context:
        for i, source in enumerate(sources, start=1):
            label = source.title or source.url
            print(f"[{i}/{len(sources)}] {label} ({source.source_type}, {source.service_category})")
            with get_conn() as conn:
                outcome = fetch_and_persist(
                    context,
                    conn,
                    settings,
                    url=source.url,
                    source_type=source.source_type,
                    source_id=source.id,
                )
            outcome["title"] = label
            results.append(outcome)

            if outcome["success"]:
                tag = "unchanged" if outcome["unchanged"] else f"v{outcome['version']}"
                print(f"    OK  ({tag}, {outcome['bytes']} bytes, {outcome['attempts']} attempt(s))")
            else:
                print(f"    FAIL  {outcome['error']}")

            if i < len(sources):
                time.sleep(DELAY_BETWEEN_SOURCES_SECONDS)

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    new_versions = [r for r in successes if not r["unchanged"]]
    unchanged = [r for r in successes if r["unchanged"]]

    print("\n=== Summary ===")
    print(f"total sources:     {len(results)}")
    print(f"succeeded:         {len(successes)}  ({len(new_versions)} new/changed, {len(unchanged)} unchanged)")
    print(f"failed:            {len(failures)}")

    if failures:
        print("\nFailed sources (need investigation before Phase 3 can chunk them):")
        for r in failures:
            print(f"  - {r['title']}: {r['url']}")
            print(f"      error: {r['error']}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
