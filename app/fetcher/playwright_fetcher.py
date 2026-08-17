"""
Playwright-based fetcher — used uniformly for every source (HTML and PDF),
per the project brief, rather than a plain HTTP client for the "easy"
sources and a browser only for the hard ones. Rationale: consistent
handling means one code path to trust, and some gov.in domains reject
plain requests outright regardless of content type.

Two sub-APIs are used depending on content type:
  - HTML: full page navigation (`page.goto`) so JS-rendered content is
    captured, then `page.content()` for the rendered DOM.
  - PDF: Playwright's APIRequestContext (`context.request.get`), which
    still goes through the same browser-grade network stack/TLS fingerprint
    and custom headers, but returns raw bytes directly rather than trying
    to coax a rendered PDF viewer into giving up its bytes.

This module only fetches and hashes RAW content. Extracting text (and
re-hashing the extracted region, as the ingestion pipeline is ultimately
meant to do) is Phase 3's job — see db/migrations/0001_init_schema.sql's
comment on source_versions.content_hash.
"""

import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

from playwright.sync_api import sync_playwright

from app.core.logging_config import get_logger
from app.fetcher.retry import RetryAttempt, retry_with_backoff
from app.fetcher.robots import can_fetch

logger = get_logger(__name__)

DEFAULT_TIMEOUT_MS = 30_000


def _proxy_config_from_env() -> Optional[dict]:
    """
    Playwright's request-context API (used for PDFs) picks up the standard
    HTTPS_PROXY/HTTP_PROXY env vars automatically, but a real Chromium page
    (`page.goto`, used for HTML) does NOT — it needs the proxy passed
    explicitly at launch time. Some government network environments sit
    behind a mandatory egress proxy, so this is a real deployment concern,
    not just a dev-sandbox quirk. Bypass list (NO_PROXY) is forwarded too.
    """
    server = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
    )
    if not server:
        return None
    config = {"server": server}
    bypass = os.environ.get("NO_PROXY") or os.environ.get("no_proxy")
    if bypass:
        config["bypass"] = bypass
    return config


@dataclass
class FetchResult:
    success: bool
    url: str
    final_url: Optional[str] = None
    http_status: Optional[int] = None
    content_bytes: Optional[bytes] = None
    content_hash: Optional[str] = None
    source_last_modified: Optional[datetime] = None
    attempts: int = 0
    error: Optional[str] = None
    skipped_reason: Optional[str] = None
    used_request_fallback: bool = False


@contextmanager
def browser_session(user_agent: str):
    """Launches one headless Chromium browser + context for a whole run."""
    proxy = _proxy_config_from_env()
    if proxy:
        logger.info(
            "launching Chromium with egress proxy",
            extra={"fields": {"proxy_server": proxy["server"]}},
        )
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, proxy=proxy)
        context = browser.new_context(user_agent=user_agent, proxy=proxy)
        try:
            yield context
        finally:
            context.close()
            browser.close()


def _parse_last_modified(headers: dict) -> Optional[datetime]:
    raw = headers.get("last-modified")
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw)
    except Exception:
        return None


def _fetch_pdf(context, url: str, timeout_ms: int):
    resp = context.request.get(url, timeout=timeout_ms)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status} fetching PDF {url}")
    body = resp.body()
    return body, resp.status, dict(resp.headers), url, False


def _fetch_html_via_page(context, url: str, timeout_ms: int):
    page = context.new_page()
    try:
        resp = page.goto(url, wait_until="load", timeout=timeout_ms)
        if resp is None:
            raise RuntimeError(f"No response navigating to {url}")
        if not resp.ok:
            raise RuntimeError(f"HTTP {resp.status} fetching {url}")
        html = page.content()
        headers = dict(resp.headers)
        final_url = page.url
        return html.encode("utf-8"), resp.status, headers, final_url
    finally:
        page.close()


def _fetch_html(context, url: str, timeout_ms: int):
    """
    Primary path: full page navigation (`page.goto`), so JS-rendered
    content is captured — this is the path real deployments need for any
    source that renders content client-side.

    Fallback: if the browser-level navigation fails for network reasons
    (rather than a genuine HTTP error status), retry via Playwright's
    request-context API instead. This matters in network environments
    where a full browser's connection is blocked/reset but simple
    request/response traffic on the same host isn't — that's exactly what
    this project's own dev sandbox does, and similar egress restrictions
    are plausible on locked-down government infrastructure too. The
    fallback loses JS execution, so it's a degraded — not equivalent —
    result, and is logged clearly as such.
    """
    try:
        return (*_fetch_html_via_page(context, url, timeout_ms), False)
    except Exception as page_exc:
        logger.warning(
            "page.goto failed, falling back to request-context fetch (no JS execution)",
            extra={"fields": {"url": url, "page_error": str(page_exc)}},
        )
        resp = context.request.get(url, timeout=timeout_ms)
        if not resp.ok:
            raise RuntimeError(
                f"HTTP {resp.status} fetching {url} (request-context fallback, "
                f"after page.goto also failed: {page_exc})"
            ) from page_exc
        body = resp.body()
        return body, resp.status, dict(resp.headers), url, True


def fetch_one(
    context,
    *,
    url: str,
    source_type: str,
    user_agent: str,
    max_attempts: int = 3,
    base_delay_seconds: float = 1.0,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    check_robots: bool = True,
    attempt_callback: Optional[callable] = None,
) -> FetchResult:
    """
    Fetches a single URL with retries + exponential backoff. Never raises —
    failures (including a robots.txt disallow) come back as a FetchResult
    with success=False and a populated `error` / `skipped_reason`, so a
    caller looping over many sources can't have one bad URL crash the run.

    `attempt_callback(attempt_number, success, error_message)`, if given, is
    invoked after every single attempt (not just the final outcome) — the
    caller can use this to write a durable per-attempt audit_log row.
    """
    if check_robots:
        allowed, reason = can_fetch(url, user_agent, request_context=context.request)
        if not allowed:
            logger.warning(
                "fetch skipped: robots.txt disallows this user-agent",
                extra={"fields": {"url": url, "reason": reason}},
            )
            return FetchResult(success=False, url=url, skipped_reason="robots_disallowed")

    attempts_used = 0

    def _do_fetch():
        nonlocal attempts_used
        attempts_used += 1
        try:
            if source_type == "pdf":
                result = _fetch_pdf(context, url, timeout_ms)
            else:
                result = _fetch_html(context, url, timeout_ms)
        except Exception as exc:
            if attempt_callback:
                attempt_callback(attempts_used, False, str(exc))
            raise
        if attempt_callback:
            attempt_callback(attempts_used, True, None)
        return result

    def _on_attempt_failed(attempt: RetryAttempt):
        logger.warning(
            "fetch attempt failed",
            extra={
                "fields": {
                    "url": url,
                    "attempt": attempt.attempt_number,
                    "error": str(attempt.exception),
                }
            },
        )

    try:
        content_bytes, status, headers, final_url, used_fallback = retry_with_backoff(
            _do_fetch,
            max_attempts=max_attempts,
            base_delay_seconds=base_delay_seconds,
            on_attempt_failed=_on_attempt_failed,
        )
    except Exception as exc:
        logger.error(
            "fetch failed after all retries",
            extra={"fields": {"url": url, "attempts": attempts_used, "error": str(exc)}},
        )
        return FetchResult(
            success=False,
            url=url,
            attempts=attempts_used,
            error=str(exc),
        )

    content_hash = hashlib.sha256(content_bytes).hexdigest()
    logger.info(
        "fetch succeeded",
        extra={
            "fields": {
                "url": url,
                "attempts": attempts_used,
                "http_status": status,
                "content_hash": content_hash,
                "bytes": len(content_bytes),
                "used_request_fallback": used_fallback,
            }
        },
    )
    return FetchResult(
        success=True,
        url=url,
        final_url=final_url,
        http_status=status,
        content_bytes=content_bytes,
        content_hash=content_hash,
        source_last_modified=_parse_last_modified(headers),
        attempts=attempts_used,
        used_request_fallback=used_fallback,
    )
