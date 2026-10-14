"""Storing chat questions and answers (app/core/chat_logging.py)."""
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

import app.api.main as main
from app.core import chat_logging
from app.core.config import get_settings


@pytest.fixture
def stored(monkeypatch):
    """Logging on, with a fake database that records what would be written."""
    monkeypatch.setattr(get_settings(), "chat_log_enabled", True)
    rows, purges = [], []

    @contextmanager
    def fake_conn():
        yield object()

    import app.db.chat_log_repo as repo
    import app.db.connection as connection

    monkeypatch.setattr(connection, "get_conn", fake_conn)
    monkeypatch.setattr(repo, "insert_chat_log", lambda conn, **row: rows.append(row))
    monkeypatch.setattr(repo, "delete_older_than", lambda conn, days: purges.append(days) or 0)
    monkeypatch.setattr(chat_logging, "_last_purge", 0.0)
    return rows, purges


def record(**overrides):
    kwargs = dict(
        question="How do I apply for OCI?",
        result={"answer": "Thank you for asking.\n\n- Apply online.", "grounded": True,
                "citations": [{"source_url": "https://example.gov/oci"}], "retrieved_count": 8},
        session_id="s1",
        mode="text",
        is_followup=False,
        latency_ms=2300,
    )
    kwargs.update(overrides)
    chat_logging.record_chat(**kwargs)


@pytest.mark.parametrize(
    "result, outcome",
    [
        ({"grounded": True}, "answered"),
        ({"grounded": False}, "declined"),
        ({"grounded": False, "out_of_scope": True}, "out_of_scope"),
        ({"grounded": False, "smalltalk": True}, "smalltalk"),
        ({"grounded": False, "guardrail": "injection"}, "guardrail"),
        ({"grounded": False, "generation_error": "generation_unavailable"}, "error"),
    ],
)
def test_outcome(result, outcome):
    assert chat_logging.outcome_of(result) == outcome


def test_a_row_is_written_with_the_answer_and_sources(stored):
    rows, purges = stored
    record()
    [row] = rows
    assert row["question"] == "How do I apply for OCI?"
    assert row["outcome"] == "answered"
    assert row["source_urls"] == ["https://example.gov/oci"]
    assert row["answer"].startswith("Thank you for asking.")
    assert row["latency_ms"] == 2300 and row["mode"] == "text" and row["session_id"] == "s1"
    assert purges == [60]  # retention clean-up, default 60 days


def test_personal_details_are_masked_before_storing(stored):
    rows, _ = stored
    record(
        question="My full name: Ravi Kumar. File CANT0D7B4N24, what is the status?",
        result={"answer": "- Ravi Kumar, I can't see CANT0D7B4N24. Call +1 800 320 9693.", "grounded": True,
                "citations": []},
        search_query="OCI application status CANT0D7B4N24",
    )
    [row] = rows
    for text in (row["question"], row["answer"], row["search_query"]):
        assert "Ravi Kumar" not in text and "CANT0D7B4N24" not in text
    assert "+1 800 320 9693" in row["answer"]  # the Consulate's public helpline stays


def test_old_rows_are_purged_at_most_once_an_hour(stored):
    _, purges = stored
    record()
    record()
    assert purges == [60]


def test_a_database_failure_never_reaches_the_visitor(monkeypatch):
    monkeypatch.setattr(get_settings(), "chat_log_enabled", True)

    @contextmanager
    def broken():
        raise RuntimeError("database is down")
        yield  # pragma: no cover

    import app.db.connection as connection

    monkeypatch.setattr(connection, "get_conn", broken)
    record()  # logs a warning, does not raise


def test_nothing_is_written_when_switched_off(stored, monkeypatch):
    rows, _ = stored
    monkeypatch.setattr(get_settings(), "chat_log_enabled", False)
    record()
    assert rows == []


def test_chat_endpoint_logs_the_exchange_and_keeps_the_rewrite_private(monkeypatch):
    main.limiter.reset()
    logged = []
    monkeypatch.setattr(
        main,
        "answer_question",
        lambda q, **k: {"query": q, "answer": "- ok", "grounded": True, "citations": [], "retrieved_count": 1,
                        "rewritten_query": "oci application"},
    )
    monkeypatch.setattr(main, "record_chat", lambda **kw: logged.append(kw))
    client = TestClient(main.app)
    response = client.post("/chat", json={"message": "how do i apply for oci", "mode": "voice"})
    assert response.status_code == 200
    assert "rewritten_query" not in response.json()
    [entry] = logged
    assert entry["question"] == "how do i apply for oci"
    assert entry["search_query"] == "oci application"
    assert entry["mode"] == "voice" and entry["is_followup"] is False
    assert entry["session_id"] == response.json()["session_id"]
