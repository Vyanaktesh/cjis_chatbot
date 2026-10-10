"""Login-gate unit tests (app/api/auth.py). These exercise the pure helpers
with explicit Settings, so they need no running services and no module reload."""
from starlette.requests import Request
from starlette.responses import Response

from app.api import auth
from app.core.config import Settings


def _request_with_cookie(cookie_header: str | None) -> Request:
    headers = [(b"cookie", cookie_header.encode())] if cookie_header else []
    return Request({"type": "http", "headers": headers})


def _settings(**over) -> Settings:
    base = dict(_env_file=None, session_secret="unit-test-secret-value", session_cookie_name="cjis_session")
    base.update(over)
    return Settings(**base)


def test_gate_is_off_when_no_password_is_set():
    s = _settings(login_password=None)
    assert auth.login_required(s) is False
    # With the gate off every request is treated as authenticated.
    assert auth.is_authenticated(_request_with_cookie(None), s) is True


def test_gate_on_requires_a_valid_cookie():
    s = _settings(login_password="secret")
    assert auth.login_required(s) is True
    # No cookie -> not authenticated.
    assert auth.is_authenticated(_request_with_cookie(None), s) is False


def test_password_compare_is_exact():
    s = _settings(login_password="correct-horse")
    assert auth.password_matches("correct-horse", s) is True
    assert auth.password_matches("wrong", s) is False


def test_issued_cookie_round_trips_as_authenticated():
    s = _settings(login_password="secret")
    resp = Response()
    auth.issue_session_cookie(resp, s)
    set_cookie = resp.headers.get("set-cookie")
    assert set_cookie and "cjis_session=" in set_cookie
    assert "httponly" in set_cookie.lower()
    assert "samesite=lax" in set_cookie.lower()

    # Extract the cookie value and present it on a fresh request.
    token = set_cookie.split("cjis_session=", 1)[1].split(";", 1)[0]
    assert auth.is_authenticated(_request_with_cookie(f"cjis_session={token}"), s) is True


def test_a_tampered_or_foreign_cookie_is_rejected():
    s = _settings(login_password="secret")
    # Signed with a different secret -> bad signature under ours.
    other = _settings(login_password="secret", session_secret="a-totally-different-secret")
    resp = Response()
    auth.issue_session_cookie(resp, other)
    token = resp.headers["set-cookie"].split("cjis_session=", 1)[1].split(";", 1)[0]
    assert auth.is_authenticated(_request_with_cookie(f"cjis_session={token}"), s) is False
    # Outright garbage is rejected too.
    assert auth.is_authenticated(_request_with_cookie("cjis_session=not-a-real-token"), s) is False


def test_expired_cookie_is_rejected():
    s = _settings(login_password="secret", session_ttl_seconds=0)
    resp = Response()
    auth.issue_session_cookie(resp, s)
    token = resp.headers["set-cookie"].split("cjis_session=", 1)[1].split(";", 1)[0]
    # max_age=0 means anything older than "now" is expired immediately.
    import time

    time.sleep(1)
    assert auth.is_authenticated(_request_with_cookie(f"cjis_session={token}"), s) is False
