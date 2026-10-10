import asyncio
import time

import httpx
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

import app.api.main as main


def _fresh_state(monkeypatch):
    main.limiter.reset()
    monkeypatch.setattr(main, "retrieval_search", lambda *a, **k: [])


def test_slow_citizen_submission_does_not_freeze_other_requests(monkeypatch):
    """The Citizen Corner handler is `async`; its blocking HubSpot call must run
    in a worker thread, or every other request waits for it."""
    _fresh_state(monkeypatch)

    def slow_hubspot(**kwargs):
        time.sleep(2)
        return "12345"

    monkeypatch.setattr(main.hubspot, "create_citizen_submission", slow_hubspot)

    async def scenario():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            async def submit():
                return await client.post(
                    "/citizen-corner/submit",
                    data={"name": "A", "email": "a@b.co", "title": "t", "content": "c", "submission_type": "feedback"},
                )

            async def other_request_during_submit():
                started = time.time()
                await asyncio.sleep(0.2)  # the submit is in flight by now
                response = await client.get("/search", params={"q": "passport"})
                return response.status_code, time.time() - started

            submitted, (status, elapsed) = await asyncio.gather(submit(), other_request_during_submit())
            return submitted.status_code, status, elapsed

    submit_status, other_status, elapsed = asyncio.run(scenario())
    assert submit_status == 200
    assert other_status == 200
    assert elapsed < 1.0, f"another request waited {elapsed:.1f}s behind a slow HubSpot call"


def _hit_search(app, forwarded_for: str, count: int) -> list[int]:
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return [
                (await client.get("/search", params={"q": "passport"}, headers={"X-Forwarded-For": forwarded_for(i)})).status_code
                for i in range(count)
            ]

    return asyncio.run(run())


def test_behind_a_trusted_proxy_each_visitor_gets_their_own_limit(monkeypatch):
    _fresh_state(monkeypatch)
    proxied = ProxyHeadersMiddleware(main.app, trusted_hosts="*")
    codes = _hit_search(proxied, lambda i: f"203.0.113.{i}", 65)  # 65 different visitors
    assert 429 not in codes


def test_behind_a_trusted_proxy_one_visitor_is_still_limited(monkeypatch):
    _fresh_state(monkeypatch)
    proxied = ProxyHeadersMiddleware(main.app, trusted_hosts="*")
    codes = _hit_search(proxied, lambda i: "203.0.113.7", 65)  # one visitor, 65 requests
    assert codes.index(429) + 1 == 61  # limit is 60/minute


def test_without_proxy_trust_all_visitors_share_one_limit(monkeypatch):
    """Documents the pitfall LOCAL_SETUP.md warns about: if uvicorn isn't told
    to trust the proxy, 65 different visitors exhaust a single shared bucket."""
    _fresh_state(monkeypatch)
    codes = _hit_search(main.app, lambda i: f"203.0.113.{i}", 65)
    assert codes.index(429) + 1 == 61
