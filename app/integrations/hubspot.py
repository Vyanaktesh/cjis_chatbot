"""
HubSpot escalation: when the chatbot can't ground an answer, the widget
offers the citizen a short form (name, city, state, email, question) that
lands in HubSpot as a real Contact + Ticket, so a consulate staff member
sees it in their normal ticket queue and can reply from there -- see
app/api/main.py's POST /support/ticket, which is the only caller of this
module.
"""

import json
import re
import threading
import time
from typing import Any, Optional

import httpx

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

HUBSPOT_API_BASE = "https://api.hubapi.com"
# Per HubSpot call. A submission makes up to four calls in a row, so the
# worst case is roughly four times this; the widget's own timeout is set above
# that so a slow-but-successful submission isn't reported as a failure.
REQUEST_TIMEOUT = 10.0

# HubSpot-defined association type id for "ticket to contact (primary)" --
# a fixed constant on HubSpot's side, not something this portal configures.
TICKET_TO_CONTACT_ASSOCIATION_TYPE_ID = 16


class HubSpotError(Exception):
    """Raised for any failure creating the Contact/Ticket -- missing token,
    HubSpot rejecting the request, an unexpected reply, or a network failure.
    Caught in app/api/main.py so a HubSpot outage degrades to a clear message
    instead of a raw 500 leaking API details to the browser."""


class HubSpotConflict(HubSpotError):
    """HubSpot answered 409 (the thing being created already exists)."""

    def __init__(self, message: str, detail: str):
        super().__init__(message)
        self.detail = detail


def _auth_header() -> dict[str, str]:
    settings = get_settings()
    if not settings.hubspot_access_token:
        raise HubSpotError(
            "HUBSPOT_ACCESS_TOKEN is not set in .env. Create a private app in "
            "HubSpot (Settings > Integrations > Private Apps) with "
            "crm.objects.contacts.write and crm.objects.tickets.write scopes, "
            "then add its access token to .env."
        )
    return {"Authorization": f"Bearer {settings.hubspot_access_token}"}


def _headers() -> dict[str, str]:
    return {**_auth_header(), "Content-Type": "application/json"}


def _error_detail(response: httpx.Response) -> str:
    """HubSpot's error message if the reply is JSON, otherwise a short slice
    of whatever came back (a gateway can answer with an HTML error page)."""
    try:
        body = response.json()
        if isinstance(body, dict) and body.get("message"):
            return str(body["message"])
    except ValueError:
        pass
    return (response.text or "")[:200] or f"HTTP {response.status_code}"


def _json(response: httpx.Response, where: str) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError as exc:
        raise HubSpotError(f"HubSpot returned a non-JSON reply ({where})") from exc
    if not isinstance(body, dict):
        raise HubSpotError(f"HubSpot returned an unexpected reply ({where})")
    return body


def _require_id(body: dict[str, Any], where: str) -> str:
    value = body.get("id")
    if not value:
        raise HubSpotError(f"HubSpot reply had no id ({where})")
    return str(value)


def _post(client: httpx.Client, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        resp = client.post(f"{HUBSPOT_API_BASE}{path}", json=payload, headers=_headers())
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = _error_detail(exc.response)
        message = f"HubSpot API error ({exc.response.status_code}) on {path}: {detail}"
        if exc.response.status_code == 409:
            raise HubSpotConflict(message, detail) from exc
        raise HubSpotError(message) from exc
    except httpx.RequestError as exc:
        raise HubSpotError(f"Could not reach HubSpot ({path}): {exc}") from exc
    return _json(resp, path)


def _upload_file(client: httpx.Client, *, filename: str, content: bytes, content_type: Optional[str]) -> tuple[str, str]:
    """Uploads via HubSpot's Files API (POST /files/v3/files) and returns
    (file id, file url). Requires the `files` scope on the access token,
    separate from the contacts/tickets scopes the rest of this module needs --
    see Settings.hubspot_citizen_pipeline_id's docstring.

    The photo is uploaded the moment it is submitted -- before anyone has
    reviewed it -- so its visibility comes from Settings.hubspot_photo_access
    (default: reachable by link but not indexed by search engines) rather than
    being hard-coded to fully public + indexable."""
    access = get_settings().hubspot_photo_access
    try:
        resp = client.post(
            f"{HUBSPOT_API_BASE}/files/v3/files",
            headers=_auth_header(),
            files={"file": (filename, content, content_type or "application/octet-stream")},
            data={
                "fileName": filename,
                "folderPath": "/chatbot-uploads/citizen-corner",
                "options": json.dumps({"access": access}),
            },
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HubSpotError(
            f"HubSpot file upload failed ({exc.response.status_code}): {_error_detail(exc.response)}"
        ) from exc
    except httpx.RequestError as exc:
        raise HubSpotError(f"Could not reach HubSpot (file upload): {exc}") from exc
    body = _json(resp, "file upload")
    url = body.get("url") or body.get("defaultHostingUrl")
    if not url:
        raise HubSpotError("HubSpot file upload reply had no url")
    return _require_id(body, "file upload"), url


def _delete_file(client: httpx.Client, file_id: str) -> None:
    """Best-effort cleanup of an uploaded photo whose ticket could not be
    created, so it isn't left behind in the portal with nothing pointing at it."""
    try:
        client.delete(f"{HUBSPOT_API_BASE}/files/v3/files/{file_id}", headers=_auth_header())
    except Exception as exc:  # noqa: BLE001 -- cleanup must never mask the original error
        logger.warning(f"could not delete orphaned HubSpot file {file_id}: {exc}")


_EXISTING_ID_RE = re.compile(r"Existing ID:\s*(\d+)", re.IGNORECASE)


def _find_or_create_contact(client: httpx.Client, *, email: str, name: str, city: str = "", state: str = "") -> str:
    """Returns the HubSpot contact id for `email`, creating the contact if it
    doesn't exist yet.

    An existing contact is left exactly as it is. This used to be a batch
    *upsert*, which also overwrote the existing contact's name/city/state --
    so anyone typing someone else's email address into the form could rewrite
    that person's record. Creating, and on HubSpot's 409 "already exists" reusing
    the id it reports, only needs the write scope and never modifies anyone.
    If HubSpot's 409 doesn't include an id, fall back to the old upsert so
    escalation keeps working."""
    first_name, _, last_name = name.strip().partition(" ")
    properties = {
        "email": email,
        "firstname": first_name,
        "lastname": last_name,
        "city": city,
        "state": state,
    }
    try:
        body = _post(client, "/crm/v3/objects/contacts", {"properties": properties})
        return _require_id(body, "contact create")
    except HubSpotConflict as conflict:
        match = _EXISTING_ID_RE.search(conflict.detail)
        if match:
            return match.group(1)
    body = _post(
        client,
        "/crm/v3/objects/contacts/batch/upsert",
        {"inputs": [{"id": email, "idProperty": "email", "properties": properties}]},
    )
    results = body.get("results")
    if not isinstance(results, list) or not results:
        raise HubSpotError("HubSpot contact upsert returned no results")
    return _require_id(results[0], "contact upsert")


def _create_ticket(client: httpx.Client, *, contact_id: str, subject: str, content: str) -> str:
    settings = get_settings()
    body = _post(
        client,
        "/crm/v3/objects/tickets",
        {
            "properties": {
                "subject": subject,
                "content": content,
                "hs_pipeline": settings.hubspot_ticket_pipeline_id,
                "hs_pipeline_stage": settings.hubspot_ticket_stage_id,
            },
            "associations": [
                {
                    "to": {"id": contact_id},
                    "types": [
                        {
                            "associationCategory": "HUBSPOT_DEFINED",
                            "associationTypeId": TICKET_TO_CONTACT_ASSOCIATION_TYPE_ID,
                        }
                    ],
                }
            ],
        },
    )
    return _require_id(body, "ticket create")


# A visitor whose first attempt timed out in the browser, then pressed "Try
# again", would otherwise create the same ticket twice. Remember successful
# submissions briefly and hand back the original ticket id for an identical
# repeat. In-memory and per-process: enough for the retry case, not a ledger.
_RECENT_TTL_SECONDS = 180.0
_recent_lock = threading.Lock()
_recent: dict[tuple, tuple[float, str]] = {}


def _recent_get(key: tuple) -> Optional[str]:
    now = time.monotonic()
    with _recent_lock:
        for stale in [k for k, (t, _) in _recent.items() if now - t > _RECENT_TTL_SECONDS]:
            del _recent[stale]
        hit = _recent.get(key)
        return hit[1] if hit else None


def _recent_put(key: tuple, ticket_id: str) -> None:
    with _recent_lock:
        _recent[key] = (time.monotonic(), ticket_id)


def create_support_ticket(*, name: str, email: str, city: str, state: str, message: str) -> str:
    """Finds or creates the Contact, creates a Ticket associated to it,
    returns the new ticket's HubSpot object id. Raises HubSpotError on any
    failure -- caller decides how to surface that to the citizen."""
    key = ("support", email.strip().lower(), message.strip())
    already = _recent_get(key)
    if already:
        return already
    subject = f"Chatbot escalation: {message[:80]}" + ("..." if len(message) > 80 else "")
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        contact_id = _find_or_create_contact(client, email=email, name=name, city=city, state=state)
        ticket_id = _create_ticket(client, contact_id=contact_id, subject=subject, content=message)
    _recent_put(key, ticket_id)
    return ticket_id


def _create_citizen_ticket(
    client: httpx.Client,
    *,
    contact_id: str,
    title: str,
    content: str,
    anonymous: bool,
    submission_type: str,
    photo_url: Optional[str],
) -> str:
    settings = get_settings()
    if not settings.hubspot_citizen_pipeline_id or not settings.hubspot_citizen_stage_pending_id:
        raise HubSpotError(
            "Citizen Corner isn't fully set up on the HubSpot side yet: "
            "HUBSPOT_CITIZEN_PIPELINE_ID / HUBSPOT_CITIZEN_STAGE_PENDING_ID "
            "are not set in .env. Create the 'Citizen Corner' ticket "
            "pipeline (Pending Review / Approved / Rejected stages) plus "
            "the is_anonumous and photo_url custom ticket properties in "
            "HubSpot, then add the pipeline + pending-stage ids to .env."
        )
    properties: dict[str, Any] = {
        "subject": title,
        "content": content,
        "hs_pipeline": settings.hubspot_citizen_pipeline_id,
        "hs_pipeline_stage": settings.hubspot_citizen_stage_pending_id,
        # Custom ticket properties -- must already exist in HubSpot (see
        # HubSpotError message above).
        # Property internal name is genuinely "is_anonumous" (missing the
        # "y") -- created manually on the HubSpot side before this module
        # knew it needed the property, and internal names can't be edited
        # after creation there (only the display label), so this is
        # matched to what actually exists rather than the correct
        # spelling. submission_type below, by contrast, was created via
        # this same Files/Tickets-scoped token calling POST
        # /crm/v3/properties/tickets directly -- correctly spelled, no
        # HubSpot UI step required.
        "is_anonumous": "true" if anonymous else "false",
        "submission_type": submission_type,
    }
    if photo_url:
        properties["photo_url"] = photo_url
    body = _post(
        client,
        "/crm/v3/objects/tickets",
        {
            "properties": properties,
            "associations": [
                {
                    "to": {"id": contact_id},
                    "types": [
                        {
                            "associationCategory": "HUBSPOT_DEFINED",
                            "associationTypeId": TICKET_TO_CONTACT_ASSOCIATION_TYPE_ID,
                        }
                    ],
                }
            ],
        },
    )
    return _require_id(body, "citizen ticket create")


# Mirrors the "submission_type" enumeration property's option values
# (created via POST /crm/v3/properties/tickets -- see
# _create_citizen_ticket's comment). Kept here as the single source of
# truth; app/api/main.py validates incoming requests against this same
# set before ever calling create_citizen_submission.
CITIZEN_SUBMISSION_TYPES = ("testimonial", "experience", "feedback", "photo")


def create_citizen_submission(
    *,
    name: str,
    email: str,
    title: str,
    content: str,
    anonymous: bool,
    submission_type: str,
    photo_bytes: Optional[bytes] = None,
    photo_filename: Optional[str] = None,
    photo_content_type: Optional[str] = None,
) -> str:
    """Citizen Corner (PRD FR-5.1-5.8): finds or creates the Contact,
    optionally uploads a photo, creates a Ticket in the dedicated Citizen
    Corner pipeline's Pending Review stage associated to that Contact, and
    returns the new ticket's HubSpot id (used as the citizen-facing
    reference number -- see app/api/main.py). name/email are always
    collected regardless of `anonymous`: that flag only controls public
    attribution once approved (FR-5.6), not whether we can send the
    acknowledgement (FR-5.4) or follow up internally. `submission_type`
    must be one of CITIZEN_SUBMISSION_TYPES -- without it, staff reviewing
    the pipeline saw only a title/content with no way to tell a
    testimonial apart from a complaint at a glance. Raises HubSpotError on
    any failure, including the pipeline/properties not being set up yet --
    caller decides how to surface that to the citizen. If the photo was
    already uploaded when a later step fails, it is deleted again."""
    key = ("citizen", email.strip().lower(), title.strip(), content.strip(), submission_type, bool(photo_bytes))
    already = _recent_get(key)
    if already:
        return already
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        contact_id = _find_or_create_contact(client, email=email, name=name)
        photo_url = None
        file_id = None
        if photo_bytes and photo_filename:
            file_id, photo_url = _upload_file(
                client,
                filename=photo_filename,
                content=photo_bytes,
                content_type=photo_content_type,
            )
        try:
            ticket_id = _create_citizen_ticket(
                client,
                contact_id=contact_id,
                title=title,
                content=content,
                anonymous=anonymous,
                submission_type=submission_type,
                photo_url=photo_url,
            )
        except Exception:
            if file_id:
                _delete_file(client, file_id)
            raise
    _recent_put(key, ticket_id)
    return ticket_id
