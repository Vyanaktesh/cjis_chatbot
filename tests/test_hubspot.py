import types

import httpx
import pytest

from app.core.config import Settings
from app.integrations import hubspot

CONTACTS = ("POST", "/crm/v3/objects/contacts")
UPSERT = ("POST", "/crm/v3/objects/contacts/batch/upsert")
TICKETS = ("POST", "/crm/v3/objects/tickets")
FILES = ("POST", "/files/v3/files")


@pytest.fixture()
def hs(monkeypatch):
    hubspot._recent.clear()
    settings = Settings(
        _env_file=None,
        hubspot_access_token="test-token",
        hubspot_citizen_pipeline_id="9",
        hubspot_citizen_stage_pending_id="8",
    )
    monkeypatch.setattr(hubspot, "get_settings", lambda: settings)

    calls: list[tuple[str, str]] = []
    bodies: dict[tuple[str, str], bytes] = {}
    handlers: dict[tuple[str, str], object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        calls.append(key)
        bodies[key] = request.content
        fn = handlers.get(key)
        if fn is None and request.method == "DELETE":
            return httpx.Response(204)
        if fn is None:
            return httpx.Response(500, text=f"unexpected call {key}")
        return fn(request)

    real_client = httpx.Client
    monkeypatch.setattr(hubspot.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    return types.SimpleNamespace(calls=calls, bodies=bodies, handlers=handlers)


def ok(body, status=200):
    return lambda request: httpx.Response(status, json=body)


def support(**over):
    args = dict(name="Asha Rao", email="asha@example.com", city="Atlanta", state="GA", message="How long does OCI take?")
    return hubspot.create_support_ticket(**{**args, **over})


def test_a_new_contact_is_created_not_upserted(hs):
    hs.handlers[CONTACTS] = ok({"id": "101"}, 201)
    hs.handlers[TICKETS] = ok({"id": "555"}, 201)
    assert support() == "555"
    assert UPSERT not in hs.calls


def test_an_existing_contact_is_reused_and_never_modified(hs):
    hs.handlers[CONTACTS] = ok({"message": "Contact already exists. Existing ID: 777"}, 409)
    hs.handlers[TICKETS] = ok({"id": "556"}, 201)
    assert support() == "556"
    assert UPSERT not in hs.calls  # the old upsert overwrote the existing contact's name/city/state
    assert b'"id": "777"' in hs.bodies[TICKETS] or b'"id":"777"' in hs.bodies[TICKETS]


def test_a_conflict_without_an_id_falls_back_to_the_old_upsert(hs):
    hs.handlers[CONTACTS] = ok({"message": "Contact already exists."}, 409)
    hs.handlers[UPSERT] = ok({"results": [{"id": "303"}]})
    hs.handlers[TICKETS] = ok({"id": "557"}, 201)
    assert support() == "557"
    assert UPSERT in hs.calls


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(502, text="<html>Bad Gateway</html>"),  # gateway error page, not JSON
        httpx.Response(200, text="not json"),
        httpx.Response(201, json={}),  # success but no id
        httpx.Response(201, json=["unexpected"]),
    ],
)
def test_unexpected_replies_become_clean_hubspot_errors(hs, reply):
    hs.handlers[CONTACTS] = lambda request: reply
    with pytest.raises(hubspot.HubSpotError):
        support()


def test_an_identical_resubmission_returns_the_same_ticket_instead_of_a_duplicate(hs):
    hs.handlers[CONTACTS] = ok({"id": "101"}, 201)
    hs.handlers[TICKETS] = ok({"id": "600"}, 201)
    first = support()
    second = support()  # e.g. "Try again" after the browser timed out
    assert first == second == "600"
    assert hs.calls.count(TICKETS) == 1


def test_a_different_message_is_a_new_ticket(hs):
    hs.handlers[CONTACTS] = ok({"id": "101"}, 201)
    counter = iter(["1", "2"])
    hs.handlers[TICKETS] = lambda request: httpx.Response(201, json={"id": next(counter)})
    assert support(message="first question") != support(message="second question")


def citizen(**over):
    args = dict(
        name="Asha Rao", email="asha@example.com", title="Great service", content="Thank you",
        anonymous=False, submission_type="testimonial",
    )
    return hubspot.create_citizen_submission(**{**args, **over})


def test_photo_upload_uses_the_configured_non_indexable_access_level(hs):
    hs.handlers[CONTACTS] = ok({"id": "1"}, 201)
    hs.handlers[FILES] = ok({"id": "f1", "url": "https://files.example/photo.png"})
    hs.handlers[TICKETS] = ok({"id": "700"}, 201)
    assert citizen(photo_bytes=b"imagebytes", photo_filename="photo-abc.png", photo_content_type="image/png") == "700"
    assert b"PUBLIC_NOT_INDEXABLE" in hs.bodies[FILES]
    assert b"PUBLIC_INDEXABLE" not in hs.bodies[FILES].replace(b"PUBLIC_NOT_INDEXABLE", b"")
    assert b"https://files.example/photo.png" in hs.bodies[TICKETS]


def test_the_uploaded_photo_is_deleted_if_the_ticket_cannot_be_created(hs):
    hs.handlers[CONTACTS] = ok({"id": "1"}, 201)
    hs.handlers[FILES] = ok({"id": "f1", "url": "https://files.example/photo.png"})
    hs.handlers[TICKETS] = lambda request: httpx.Response(500, json={"message": "boom"})
    with pytest.raises(hubspot.HubSpotError):
        citizen(photo_bytes=b"imagebytes", photo_filename="photo-abc.png", photo_content_type="image/png")
    assert ("DELETE", "/files/v3/files/f1") in hs.calls
