#!/usr/bin/env python3
"""
Phase 2 verification script.

1. Ensures the full 47-source registry is loaded into Postgres (idempotent
   upsert — see load_source_registry.py).
2. Live-fetches a small, representative subset through the real Playwright
   fetcher: one VFS PDF, one VFS HTML page, two gov.in HTML pages on
   different domains, plus one deliberately broken URL that isn't a
   registered source at all.
3. Writes: raw content to disk, a source_versions row per successful
   fetch, and audit_log rows for every attempt and outcome (success or
   failure) — nothing crashes the run, including the broken URL.

For fetching the FULL registry (all 47), see scripts/fetch_all_sources.py
instead — this script is deliberately a small, fast subset.

Usage:
    python scripts/run_fetch_smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.core.logging_config import configure_logging, get_logger  # noqa: E402
from app.db.connection import get_conn  # noqa: E402
from app.db.sources_repo import get_by_url  # noqa: E402
from app.fetcher.orchestrate import fetch_and_persist  # noqa: E402
from app.fetcher.playwright_fetcher import browser_session  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_source_registry import main as load_registry  # noqa: E402

# A representative mix from the real registry: a VFS PDF, a VFS HTML page,
# and two gov.in HTML pages on two different government domains (one of
# them the "needs a headless browser" style status portal).
SMOKE_TEST_URLS = [
    "https://services.vfsglobal.com/one-pager/india/united-states-of-america/oci-services/pdf/Foreign-national-previously-Indian-passport-holder-(adult)-2025.pdf",
    "https://services.vfsglobal.com/one-pager/india/united-states-of-america/oci-services/",
    "https://indiainatlanta.gov.in/",
    "https://ociservices.gov.in/onlineOCI/statusEnqury",
]

# Not a registered source on purpose — proves the failure path logs
# cleanly instead of crashing the run.
DELIBERATELY_BROKEN_URL = "https://services.vfsglobal.com/one-pager/india/united-states-of-america/oci-services/pdf/this-file-does-not-exist-404.pdf"

logger = get_logger(__name__)


def run_registered(context, settings, url: str) -> dict:
    with get_conn() as conn:
        source = get_by_url(conn, url)
        if source is None:
            raise RuntimeError(f"{url} is not registered — run load_source_registry.py first")
        return fetch_and_persist(
            context, conn, settings, url=url, source_type=source.source_type, source_id=source.id
        )


def run_unregistered(context, settings, url: str) -> dict:
    with get_conn() as conn:
        # broken test URL: treated as a PDF, matches the real entry it's based on
        return fetch_and_persist(context, conn, settings, url=url, source_type="pdf", source_id=None)


def main() -> int:
    configure_logging(get_settings().log_level)

    print("=== Step 1: loading full source registry (idempotent) ===")
    load_registry()

    print("\n=== Step 2: live fetch smoke test ===")
    settings = get_settings()
    results = []

    with browser_session(settings.fetcher_user_agent) as context:
        for url in SMOKE_TEST_URLS:
            print(f"\nfetching (registered source): {url}")
            results.append(run_registered(context, settings, url))

        print(f"\nfetching (deliberately broken, unregistered): {DELIBERATELY_BROKEN_URL}")
        results.append(run_unregistered(context, settings, DELIBERATELY_BROKEN_URL))

    print("\n=== Summary ===")
    for r in results:
        status = "OK" if r["success"] else "FAIL"
        print(
            f"[{status}] {r['url']}\n"
            f"        attempts={r['attempts']} http_status={r['http_status']} "
            f"bytes={r['bytes']} hash={ (r['content_hash'] or '')[:12] }\n"
            f"        raw_path={r['raw_path']} error={r['error']} skipped={r['skipped_reason']}"
        )

    successes = sum(1 for r in results if r["success"])
    failures = len(results) - successes
    print(f"\n{successes} succeeded, {failures} failed (1 failure expected — the broken URL).")

    broken_result = next(r for r in results if r["url"] == DELIBERATELY_BROKEN_URL)
    if broken_result["success"]:
        print("\nWARNING: the deliberately-broken URL unexpectedly succeeded — pick a different one.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
