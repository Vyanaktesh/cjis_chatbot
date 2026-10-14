"""Phase 2 grounded voice mode: the spoken-text sanitiser, and that mode="voice"
attaches a clean spoken_text without changing the grounded pipeline."""
import pytest

from app.generation import service
from app.generation.prompt import VOICE_SYSTEM_PROMPT, strip_for_speech

CHUNK = {
    "chunk_id": "c1",
    "source_url": "https://services.vfsglobal.com/oci.pdf",
    "service_category": "oci",
    "canonical": True,
    "heading_trail": [],
    "chunk_text": "OCI applications are submitted online.",
}


class FakeEmbedder:
    def __init__(self, batch_size=1):
        pass

    def embed(self, texts):
        return [object() for _ in texts]


class VoiceGen:
    """Returns a grounded written answer for the RAG call, and a short spoken
    summary for the voice-condense call (distinguished by the system prompt)."""

    FULL = "To renew your passport, submit documents at VFS [1]. The fee is $100 [1] and it takes 10 days [1]."
    SPOKEN = "To renew your passport, submit your documents at VFS. The fee is 100 dollars and it takes about 10 days. Would you like more detail on any part?"

    def generate(self, messages):
        system = messages[0]["content"] if messages else ""
        return self.SPOKEN if system == VOICE_SYSTEM_PROMPT else self.FULL


@pytest.fixture(autouse=True)
def stubs(monkeypatch):
    monkeypatch.setattr(service, "BgeM3Embedder", FakeEmbedder)
    monkeypatch.setattr(service, "probe_relevance", lambda *a, **k: 0.9)
    monkeypatch.setattr(service, "retrieval_search", lambda *a, **k: [CHUNK])
    monkeypatch.setattr(service, "has_approved_content", lambda *a, **k: True)


# --- strip_for_speech ------------------------------------------------------

def test_strip_for_speech_removes_markers_urls_and_markdown():
    raw = "Fees are **$100** [1][2]. See https://vfs.example/oci.pdf for the form.\n- Step one\n- Step two [3]"
    out = strip_for_speech(raw)
    assert "[1]" not in out and "[2]" not in out and "[3]" not in out
    assert "http" not in out
    assert "**" not in out
    assert "\n" not in out
    assert "$100" in out  # the fee survives


def test_strip_for_speech_keeps_fees_and_deadlines():
    raw = "The fee is $275 and the deadline is 30 days [1]."
    out = strip_for_speech(raw)
    assert "$275" in out and "30 days" in out


# --- voice mode on answer_question ----------------------------------------

def test_voice_mode_attaches_clean_spoken_text_for_a_grounded_answer():
    result = service.answer_question("how do I renew my passport", mode="voice", generator=VoiceGen())
    assert result["grounded"] is True
    # the written answer carries its sources as chips, not as [n] markers in the text
    assert "[1]" not in result["answer"]
    assert len(result["citations"]) == 1
    # spoken_text is present, clean, and keeps the fee
    spoken = result["spoken_text"]
    assert "[1]" not in spoken and "http" not in spoken
    assert "100 dollars" in spoken


def test_text_mode_has_no_spoken_text():
    result = service.answer_question("how do I renew my passport", generator=VoiceGen())
    assert "spoken_text" not in result


def test_voice_mode_speaks_a_decline_without_a_model_call(monkeypatch):
    # Off-topic -> decline path: spoken_text is just the sanitised decline, and
    # the model is never called to produce it.
    monkeypatch.setattr(service, "probe_relevance", lambda *a, **k: 0.1)

    def boom(messages):
        raise AssertionError("the model must not be called for a decline")

    result = service.answer_question(
        "what is the capital of France",
        mode="voice",
        generator=type("G", (), {"generate": staticmethod(boom)})(),
    )
    assert result["grounded"] is False
    assert "spoken_text" in result
    assert result["spoken_text"] == strip_for_speech(result["answer"])
