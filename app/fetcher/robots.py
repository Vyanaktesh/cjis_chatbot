"""
robots.txt checking.

Tries a plain HTTP fetch first (fast, no browser needed). Some gov.in
domains block plain requests entirely (per the project brief) — if the
plain fetch fails, falls back to fetching robots.txt through the same
Playwright request context the fetcher uses for real pages. If robots.txt
is unreachable by either method, the standard convention applies: treat
the site as allowing everything, and log that the check was inconclusive.
"""

import urllib.request
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from app.core.logging_config import get_logger

logger = get_logger(__name__)


def _robots_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/robots.txt"


def _fetch_plain(url: str, user_agent: str, timeout: float = 8.0) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            if resp.status >= 400:
                return None
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def _fetch_via_playwright(url: str, request_context, timeout_ms: float = 8000) -> str | None:
    try:
        resp = request_context.get(url, timeout=timeout_ms)
        if not resp.ok:
            return None
        return resp.text()
    except Exception:
        return None


def can_fetch(url: str, user_agent: str, request_context=None) -> tuple[bool, str]:
    """
    Returns (allowed, reason). `allowed` is True whenever robots.txt is
    missing/unreachable (standard "no robots.txt = allow all" convention)
    or when it explicitly permits `user_agent` on `url`.
    """
    robots_url = _robots_url(url)

    text = _fetch_plain(robots_url, user_agent)
    method = "plain_http"
    if text is None and request_context is not None:
        text = _fetch_via_playwright(robots_url, request_context)
        method = "playwright"

    if text is None:
        logger.info(
            "robots.txt unreachable, defaulting to allow",
            extra={"fields": {"robots_url": robots_url}},
        )
        return True, "robots_txt_unreachable_defaulting_allow"

    parser = RobotFileParser()
    parser.parse(text.splitlines())
    allowed = parser.can_fetch(user_agent, url)
    return allowed, f"checked_via_{method}"
