"""Greetings, the on-topic keyword backup, the 'nothing approved yet' message,
and generic error codes in app/generation/service.py."""
import pytest

from app.generation import service

CHUNK = {
    "chunk_id": "c1",
    "source_url": "https://example.gov/oci",
    "service_category": "oci",
    "canonical": True,
    "heading_trail": [],
    "chunk_text": "OCI applications are submitted online.",
}


class FakeEmbedder:
    calls = 0

    def __init__(self, batch_size=1):
        pass

    def embed(self, texts):
        FakeEmbedder.calls += 1
        return [object() for _ in texts]


class Gen:
    def __init__(self, text="Submit online [1]."):
        self.text = text

    def generate(self, messages):
        if isinstance(self.text, Exception):
            raise self.text
        return self.text


@pytest.fixture(autouse=True)
def stubs(monkeypatch):
    FakeEmbedder.calls = 0
    monkeypatch.setattr(service, "BgeM3Embedder", FakeEmbedder)
    monkeypatch.setattr(service, "probe_relevance", lambda *a, **k: 0.9)
    monkeypatch.setattr(service, "retrieval_search", lambda *a, **k: [CHUNK])
    monkeypatch.setattr(service, "has_approved_content", lambda *a, **k: True)


@pytest.mark.parametrize(
    "message",
    ["hello", "Hi!", "hey there", "Namaste", "good morning", "thanks", "Thank you so much!", "thx", "ok", "bye"],
)
def test_greetings_and_thanks_get_a_friendly_reply_without_any_lookup(message):
    result = service.answer_question(message, generator=Gen(RuntimeError("model must not be called")))
    assert result["smalltalk"] is True
    assert result["grounded"] is False
    assert result["citations"] == []
    assert FakeEmbedder.calls == 0  # no embedding, no retrieval, no model call
    assert "outside what I can help with" not in result["answer"]


def test_a_real_question_that_starts_with_hello_is_not_smalltalk():
    result = service.answer_question("hello, how do I apply for an OCI card?", generator=Gen())
    assert "smalltalk" not in result
    assert result["grounded"] is True


def test_a_bare_domain_word_is_on_topic_even_when_the_score_is_just_under_the_floor(monkeypatch):
    monkeypatch.setattr(service, "probe_relevance", lambda *a, **k: 0.547)  # what a bare "OCI" scored
    result = service.answer_question("OCI", generator=Gen())
    assert result["grounded"] is True


def test_off_topic_questions_are_still_declined(monkeypatch):
    monkeypatch.setattr(service, "probe_relevance", lambda *a, **k: 0.46)
    result = service.answer_question("what is the capital of France", generator=Gen())
    assert result["answer"] == service.OUT_OF_SCOPE_ANSWER
    assert result["grounded"] is False


def test_when_nothing_is_approved_yet_it_says_so_instead_of_off_topic(monkeypatch):
    monkeypatch.setattr(service, "probe_relevance", lambda *a, **k: 0.0)
    monkeypatch.setattr(service, "has_approved_content", lambda *a, **k: False)
    result = service.answer_question("what is the capital of France", generator=Gen())
    assert result["answer"] == service.NO_CONTEXT_ANSWER


def test_generation_failures_return_a_fixed_code_not_the_exception_text():
    result = service.answer_question(
        "How do I renew my passport?",
        generator=Gen(RuntimeError("quota exceeded for project 12345 key AIzaSECRET")),
    )
    assert result["generation_error"] == "generation_unavailable"
    assert "AIzaSECRET" not in str(result)
