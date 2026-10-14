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


HISTORY = [
    {"role": "user", "content": "How do I renew my passport?"},
    {"role": "assistant", "content": "- Apply through VFS."},
]


def _scores(combined, own):
    """probe_relevance stub: the earlier question + new message scores
    `combined`; the new message on its own scores `own`."""
    return lambda q, **k: combined if q.startswith("How do I renew") else own


def test_an_off_topic_message_after_a_passport_question_is_declined(monkeypatch):
    monkeypatch.setattr(service, "probe_relevance", _scores(0.80, 0.461))  # "capital of France" alone
    result = service.answer_question("What is the capital of France?", history=HISTORY, generator=Gen())
    assert result["answer"] == service.OUT_OF_SCOPE_ANSWER
    assert result["out_of_scope"] is True


def test_a_genuine_follow_up_still_goes_through(monkeypatch):
    monkeypatch.setattr(service, "probe_relevance", _scores(0.80, 0.527))  # "and for a child?" alone
    result = service.answer_question("and for a child?", history=HISTORY, generator=Gen())
    assert result["grounded"] is True


def test_accepting_more_details_always_goes_through(monkeypatch):
    monkeypatch.setattr(service, "probe_relevance", _scores(0.80, 0.30))
    result = service.answer_question("Yes, more details please", history=HISTORY, generator=Gen())
    assert result["grounded"] is True


def test_retrieval_context_keeps_the_last_two_real_questions():
    history = [
        {"role": "user", "content": "What documents do I need for a new OCI card?"},
        {"role": "assistant", "content": "- ..."},
        {"role": "user", "content": "yes"},
        {"role": "assistant", "content": "- ..."},
        {"role": "user", "content": "and for a child?"},
        {"role": "assistant", "content": "- ..."},
    ]
    assert service._retrieval_query("is it the same for my wife?", history) == (
        "What documents do I need for a new OCI card? and for a child? is it the same for my wife?"
    )
    assert service._retrieval_query("OCI fee?", None) == "OCI fee?"


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


# --- answers the model wrote without [n] markers ----------------------------

OCI_TIME_CHUNK = {
    **CHUNK,
    "chunk_id": "c2",
    "source_url": "https://example.gov/oci-faq",
    "chunk_text": "The normal processing time for a new OCI application based on Indian origin is "
    "5 to 6 weeks from the date of registration in the Consulate, subject to clearance "
    "from the Ministry of Home Affairs. OCI cards cannot be expedited.",
}


def test_an_uncited_answer_taken_from_a_chunk_still_gets_its_source(monkeypatch):
    monkeypatch.setattr(service, "retrieval_search", lambda *a, **k: [CHUNK, OCI_TIME_CHUNK])
    uncited = (
        "- The normal processing time for a new OCI application based on Indian origin is 5 to 6 weeks "
        "from the date of registration in the Consulate.\n"
        "- This processing time is subject to clearance from the Ministry of Home Affairs.\n"
        "- OCI cards cannot be expedited."
    )
    result = service.answer_question("how long does a new OCI take", generator=Gen(uncited))
    assert result["grounded"] is True
    assert [c["source_url"] for c in result["citations"]] == ["https://example.gov/oci-faq"]
    assert "[1]" not in result["answer"]  # markers still never shown


def test_an_uncited_answer_not_found_in_any_chunk_stays_ungrounded(monkeypatch):
    monkeypatch.setattr(service, "retrieval_search", lambda *a, **k: [CHUNK, OCI_TIME_CHUNK])
    invented = (
        "- Bangladeshi passports can be renewed at any Indian consulate within two weeks.\n"
        "- Bring your birth certificate and three photographs to the counter."
    )
    result = service.answer_question("renew bangladeshi passport", generator=Gen(invented))
    assert result["grounded"] is False
    assert result["citations"] == []


# --- notes on what the sources don't cover go last ---------------------------

ADDRESS_CHUNK = {**CHUNK, "chunk_id": "c3", "source_url": "https://example.gov/reissue"}


def test_a_leading_gap_note_moves_after_the_cited_content(monkeypatch):
    monkeypatch.setattr(service, "retrieval_search", lambda *a, **k: [ADDRESS_CHUNK])
    reply = (
        "- I don't have approved information covering whether you must wait for your passport to expire. "
        "Please contact the consulate directly.\n"
        "- To change your address, you apply for a passport re-issue [1].\n"
        "- Send a proof of your present address with the application [1]."
    )
    result = service.answer_question("can I change my address now or must I wait?", generator=Gen(reply))
    bullets = [l for l in result["answer"].splitlines() if l.startswith("- ")]
    assert bullets[0].startswith("- To change your address, you apply for a passport re-issue")
    assert "approved information" in bullets[-1]
    # A partial answer keeps its sources and is not treated as a decline.
    assert result["grounded"] is True
    assert [c["source_url"] for c in result["citations"]] == ["https://example.gov/reissue"]


def test_a_reply_with_nothing_cited_is_still_a_decline():
    reply = (
        "- I don't have approved information covering that. Please contact the consulate directly.\n"
        "- The consulate can help with questions like this."
    )
    result = service.answer_question("how long can I stay in India?", generator=Gen(reply))
    assert result["grounded"] is False
    assert result["citations"] == []
    assert "More details" not in result["answer"] and service.MORE_DETAILS_OFFER not in result["answer"]


def test_a_bullet_with_both_a_gap_note_and_a_cited_fact_stays_put():
    reply = (
        "- The sources do not state a waiting period, but a re-issue can be applied for to change the address [1].\n"
        "- Send a proof of address [1]."
    )
    assert service._gap_notes_last(reply) == reply


def test_gap_notes_keep_the_offer_line_at_the_end():
    reply = (
        "- The sources do not mention a waiting period.\n"
        "- Apply for a passport re-issue [1].\n\n"
        "Would you like more details on this?"
    )
    out = service._gap_notes_last(reply)
    assert out.splitlines()[0] == "- Apply for a passport re-issue [1]."
    assert out.splitlines()[1] == "- The sources do not mention a waiting period."
    assert out.rstrip().endswith("Would you like more details on this?")


def test_a_decline_with_only_a_cited_contact_is_still_a_decline():
    reply = (
        "- I don't have approved information covering where invitation letters should be mailed. "
        "Please contact the consulate directly.\n"
        "- You can contact the VFS Global Atlanta helpline at +1 800 320 9693 [1]."
    )
    result = service.answer_question("where do I mail my letter of invitation?", generator=Gen(reply))
    assert result["grounded"] is False  # the widget offers to raise it with the Consulate
    assert result["answer"].splitlines()[0].startswith("- I don't have approved information")


def test_a_cited_gap_note_still_goes_after_the_content():
    reply = (
        "- The provided sources do not specify the processing time for GEP verification [1].\n"
        "- Apply first on the US CBP Trusted Traveler Programs website [1].\n"
        "- Then submit your physical application to VFS Global [1]."
    )
    out = service._gap_notes_last(reply)
    assert out.splitlines()[0].startswith("- Apply first on the US CBP")
    assert out.splitlines()[-1].startswith("- The provided sources do not specify")


def test_a_sources_do_not_contain_reply_with_a_cited_contact_is_a_decline():
    reply = (
        "- The provided sources do not contain information regarding COVID-19 vaccine requirements [1].\n"
        "- For further assistance you may contact VFS Global at +1 800 320 9693 [1]."
    )
    result = service.answer_question("are there any covid vaccine requirements?", generator=Gen(reply))
    assert result["grounded"] is False
    assert result["citations"] == []
    assert service.MORE_DETAILS_OFFER not in result["answer"]
