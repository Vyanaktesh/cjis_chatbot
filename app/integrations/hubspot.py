"""
HubSpot escalation: when the chatbot can't ground an answer, the widget
offers the citizen a short form (name, city, state, email, question) that
lands in HubSpot as a real Contact + Ticket, so a consulate staff member
sees it in their normal ticket queue and can reply from there -- see
app/api/main.py's POST /support/ticket, which is the only caller of this
module.
"""

from typing import Any, Optional

import httpx

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

HUBSPOT_API_BASE = "https://api.hubapi.com"
REQUEST_TIMEOUT = 15.0

# HubSpot-defined association type id for "ticket to contact (primary)" --
# a fixed constant on HubSpot's side, not something this portal configures.
TICKET_TO_CONTACT_ASSOCIATION_TYPE_ID = 16


class HubSpotError(Exception):
    """Raised for any failure creating the Contact/Ticket -- missing token,
    HubSpot rejecting the request, or a network failure. Caught in
    app/api/main.py so a HubSpot outage degrades to a clear message instead
    of a raw 500 leaking API details to the browser."""


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


def _post(client: httpx.Client, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        resp = client.post(f"{HUBSPOT_API_BASE}{path}", json=payload, headers=_headers())
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("message", exc.response.text) if exc.response.content else str(exc)
        raise HubSpotError(f"HubSpot API error ({exc.response.status_code}) on {path}: {detail}") from exc
    except httpx.RequestError as exc:
        raise HubSpotError(f"Could not reach HubSpot ({path}): {exc}") from exc
    return resp.json()


def _upload_file(client: httpx.Client, *, filename: str, content: bytes, content_type: Optional[str]) -> str:
    """Uploads via HubSpot's Files API (POST /files/v3/files) and returns
    the resulting file's public URL. PUBLIC_INDEXABLE because a rejected
    Files API access level (e.g. PRIVATE) would need signed URLs to ever
    display -- unnecessary complexity for photos a citizen is voluntarily
    submitting for potential publication anyway. Requires the `files`
    scope on the access token, separate from the contacts/tickets scopes
    the rest of this module needs -- see Settings.hubspot_citizen_pipeline_id's
    docstring."""
    try:
        resp = client.post(
            f"{HUBSPOT_API_BASE}/files/v3/files",
            headers=_auth_header(),
            files={"file": (filename, content, content_type or "application/octet-stream")},
            data={
                "fileName": filename,
                "folderPath": "/chatbot-uploads/citizen-corner",
                "options": '{"access": "PUBLIC_INDEXABLE"}',
            },
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("message", exc.response.text) if exc.response.content else str(exc)
        raise HubSpotError(f"HubSpot file upload failed ({exc.response.status_code}): {detail}") from exc
    except httpx.RequestError as exc:
        raise HubSpotError(f"Could not reach HubSpot (file upload): {exc}") from exc
    body = resp.json()
    return body.get("url") or body["defaultHostingUrl"]


def _upsert_contact(client: httpx.Client, *, email: str, name: str, city: str = "", state: str = "") -> str:
    """Create-or-update in one call via HubSpot's batch upsert endpoint,
    keyed on email -- avoids the plain create endpoint's 409-on-duplicate
    dance entirely (the same citizen messaging again just updates their
    existing Contact rather than erroring or creating a second one)."""
    first_name, _, last_name = name.strip().partition(" ")
    body = _post(
        client,
        "/crm/v3/objects/contacts/batch/upsert",
        {
            "inputs": [
                {
                    "id": email,
                    "idProperty": "email",
                    "properties": {
                        "email": email,
                        "firstname": first_name,
                        "lastname": last_name,
                        "city": city,
                        "state": state,
                    },
                }
            ]
        },
    )
    return body["results"][0]["id"]


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
    return body["id"]


def create_support_ticket(*, name: str, email: str, city: str, state: str, message: str) -> str:
    """Upserts the Contact, creates a Ticket associated to it, returns the
    new ticket's HubSpot object id. Raises HubSpotError on any failure --
    caller decides how to surface that to the citizen."""
    subject = f"Chatbot escalation: {message[:80]}" + ("..." if len(message) > 80 else "")
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        contact_id = _upsert_contact(client, email=email, name=name, city=city, state=state)
        return _create_ticket(client, contact_id=contact_id, subject=subject, content=message)


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
    return body["id"]


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
    """Citizen Corner (PRD FR-5.1-5.8): upserts the Contact, optionally
    uploads a photo, creates a Ticket in the dedicated Citizen Corner
    pipeline's Pending Review stage associated to that Contact, and
    returns the new ticket's HubSpot id (used as the citizen-facing
    reference number -- see app/api/main.py). name/email are always
    collected regardless of `anonymous`: that flag only controls public
    attribution once approved (FR-5.6), not whether we can send the
    acknowledgement (FR-5.4) or follow up internally. `submission_type`
    must be one of CITIZEN_SUBMISSION_TYPES -- without it, staff reviewing
    the pipeline saw only a title/content with no way to tell a
    testimonial apart from a complaint at a glance. Raises HubSpotError on
    any failure, including the pipeline/properties not being set up yet --
    caller decides how to surface that to the citizen."""
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        contact_id = _upsert_contact(client, email=email, name=name)
        photo_url = None
        if photo_bytes and photo_filename:
            photo_url = _upload_file(
                client,
                filename=photo_filename,
                content=photo_bytes,
                content_type=photo_content_type,
            )
        return _create_citizen_ticket(
            client,
            contact_id=contact_id,
            title=title,
            content=content,
            anonymous=anonymous,
            submission_type=submission_type,
            photo_url=photo_url,
        )
