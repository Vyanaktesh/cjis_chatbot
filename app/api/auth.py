"""
Shared-password login gate for the public API.

This is deliberately NOT a user-account system: there is one password
(LOGIN_PASSWORD) and a signed, httponly session cookie, enough to keep a
customer demo from being wide open without building real identity. When
LOGIN_PASSWORD is empty (the local-dev default) the gate is a no-op so
nothing has to be set up to run the app locally.

The cookie carries a signed, timestamped token (itsdangerous) -- the
server stays stateless, there is no session store to keep or expire. The
signature proves the cookie was issued by this server with the current
SESSION_SECRET; the timestamp lets us reject one older than the TTL.
"""

import hmac

from fastapi import Depends, HTTPException, Request, Response
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from app.core.config import Settings, get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Value stored inside the signed cookie. It carries no identity (there is
# only one shared password); the signature is what matters.
_COOKIE_PAYLOAD = "authenticated"
_SALT = "cjis-session-v1"


def _serializer(settings: Settings) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(settings.session_secret, salt=_SALT)


def login_required(settings: Settings) -> bool:
    """The gate is active only when a password is configured."""
    return bool(settings.login_password)


def password_matches(candidate: str, settings: Settings) -> bool:
    # Constant-time compare so a wrong password can't be timed character by
    # character.
    expected = settings.login_password or ""
    return hmac.compare_digest(candidate, expected)


def issue_session_cookie(response: Response, settings: Settings) -> None:
    token = _serializer(settings).dumps(_COOKIE_PAYLOAD)
    response.set_cookie(
        key=settings.session_cookie_name,
        value=token,
        max_age=settings.session_ttl_seconds,
        httponly=True,
        samesite="lax",
        secure=settings.session_cookie_secure,
        path="/",
    )


def clear_session_cookie(response: Response, settings: Settings) -> None:
    response.delete_cookie(
        key=settings.session_cookie_name,
        path="/",
        httponly=True,
        samesite="lax",
        secure=settings.session_cookie_secure,
    )


def is_authenticated(request: Request, settings: Settings) -> bool:
    """True when the gate is off, or a valid, unexpired session cookie is
    present."""
    if not login_required(settings):
        return True
    token = request.cookies.get(settings.session_cookie_name)
    if not token:
        return False
    try:
        _serializer(settings).loads(token, max_age=settings.session_ttl_seconds)
        return True
    except SignatureExpired:
        return False
    except BadSignature:
        # Wrong/forged signature, or SESSION_SECRET rotated since it was issued.
        return False


def require_login(request: Request, settings: Settings = Depends(get_settings)) -> None:
    """FastAPI dependency: 401 unless the request carries a valid session
    (or the gate is disabled). A no-op when LOGIN_PASSWORD is unset."""
    if not is_authenticated(request, settings):
        raise HTTPException(status_code=401, detail="Authentication required.")
