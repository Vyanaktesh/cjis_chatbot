"""Answer format: short bulleted answers that end with an offer of more
details, citation markers removed from the text (sources stay as chips)."""

from app.generation.prompt import (
    MORE_DETAILS_OFFER,
    SHORT_ANSWER_REMINDER,
    SYSTEM_PROMPT,
    ensure_more_details_offer,
    is_more_details_request,
    strip_citation_markers,
)


def test_citation_markers_are_removed_from_the_text():
    text = "- Update online within three months [1][2].\n- The fee is US$25 [3], [4]. Apply at VFS [1]."
    assert strip_citation_markers(text) == "- Update online within three months.\n- The fee is US$25. Apply at VFS."


def test_grouped_citations_are_expanded_then_removed():
    from app.generation.prompt import normalize_citation_groups

    raw = "- Fee is US$25 [1, 2, 3, 4, 6].\n- Apply online [2-4]. Bring proof [1,5]."
    expanded = normalize_citation_groups(raw)
    assert expanded == "- Fee is US$25 [1][2][3][4][6].\n- Apply online [2][3][4]. Bring proof [1][5]."
    assert strip_citation_markers(expanded) == "- Fee is US$25.\n- Apply online. Bring proof."


def test_grouped_citations_count_as_cited_sources():
    from app.generation.service import _dedupe_citations_by_source

    chunks = [{"source_url": f"https://s/{i}", "chunk_id": str(i), "service_category": "oci", "canonical": True}
              for i in range(1, 7)]
    from app.generation.prompt import normalize_citation_groups

    _, citations = _dedupe_citations_by_source(normalize_citation_groups("Answer [1, 2, 3, 4, 6]."), chunks)
    assert [c["source_url"] for c in citations] == [f"https://s/{i}" for i in (1, 2, 3, 4, 6)]


class _Gen:
    def __init__(self, text):
        self.text = text

    def generate(self, messages):
        return self.text


def _answer(monkeypatch, query, model_text):
    from app.generation import service

    chunk = {"chunk_id": "c1", "source_url": "https://s/1", "service_category": "oci", "canonical": True,
             "chunk_text": "t", "heading_trail": []}
    monkeypatch.setattr(service, "BgeM3Embedder", lambda batch_size=1: type("E", (), {"embed": lambda self, t: [None]})())
    monkeypatch.setattr(service, "probe_relevance", lambda *a, **k: 0.9)
    monkeypatch.setattr(service, "retrieval_search", lambda *a, **k: [chunk])
    monkeypatch.setattr(service, "pinned_chunks", lambda urls: [])
    monkeypatch.setattr(service, "_attach_source_titles", lambda c: None)
    return service.answer_question(query, generator=_Gen(model_text))


def test_a_refusal_never_carries_the_more_details_offer(monkeypatch):
    out = _answer(monkeypatch, "Can I renew a Bangladeshi passport here?",
                  "I don't have approved information covering that. Please contact the consulate directly.\n\n"
                  + MORE_DETAILS_OFFER)
    assert out["grounded"] is False
    assert MORE_DETAILS_OFFER not in out["answer"]


def test_a_grounded_short_answer_gets_the_offer_and_no_markers(monkeypatch):
    out = _answer(monkeypatch, "OCI fee?", "- The fee is US$25 [1, 1].")
    assert out["grounded"] is True
    assert out["answer"] == "- The fee is US$25.\n\n" + MORE_DETAILS_OFFER


def test_square_brackets_that_are_not_citations_are_kept():
    assert strip_citation_markers("Select [Status Enquiry] on the portal.") == "Select [Status Enquiry] on the portal."


def test_offer_is_added_once():
    once = ensure_more_details_offer("- Point one.")
    assert once.endswith(MORE_DETAILS_OFFER)
    assert ensure_more_details_offer(once) == once


def test_accepting_the_offer_is_recognised():
    from app.generation.prompt import MORE_DETAILS_BUTTON_TEXT

    for q in ["yes", "Yes please", "yeah", "sure", "tell me more", "more details", "yes, thanks",
              MORE_DETAILS_BUTTON_TEXT, "show me more", "more info please"]:
        assert is_more_details_request(q), q
    for q in ["yes I need a passport for my child", "What documents do I need?", "no"]:
        assert not is_more_details_request(q), q


def test_system_prompt_asks_for_short_bulleted_answers_and_the_offer():
    assert "3 or 4 bullet points" in SYSTEM_PROMPT
    assert "150 words" in SYSTEM_PROMPT
    assert MORE_DETAILS_OFFER in SYSTEM_PROMPT


def test_prompt_asks_for_the_direct_answer_first():
    # "Can I change my address now or must I wait till it expires?" was
    # answered correctly but with "you don't need to wait" as the 3rd bullet.
    assert "the first bullet must open with it" in SYSTEM_PROMPT
    assert "first bullet opens with the answer if a source states it" in SHORT_ANSWER_REMINDER


def test_prompt_forbids_a_guessed_yes_or_no():
    # Pushed to lead with yes/no, the model once answered "No" to "Do I need
    # to apostille the POA if I have an e-Visa?", when the checklist requires
    # an apostille from non-Indian nationals.
    assert "do not guess a yes or no" in SYSTEM_PROMPT


def test_prompt_keeps_unanswerable_questions_as_short_declines():
    # "Do not decline, give the closest rule" leaked to every unanswerable
    # question: a vaccine question got tourist-visa checklist rules as filler.
    assert "Only if the sources say nothing at all about the service or situation asked about" in SYSTEM_PROMPT
    # ...but "can I do it now or must I wait?" is covered when the sources say how to do it.
    assert "is covered if the sources describe how to do it" in SYSTEM_PROMPT
    assert "Do not fill such a reply with facts about other services or topics" in SYSTEM_PROMPT
    assert "one or two bullets" in SYSTEM_PROMPT
    assert "with no facts about other services or topics and without the offer line" in SHORT_ANSWER_REMINDER
    assert "never a guessed yes or no" in SHORT_ANSWER_REMINDER


def test_prompt_prefers_a_rule_that_replaced_an_older_one():
    # The government OCI FAQ still describes re-issue at 20 and 50 and a police
    # report for a lost card; the Consulate's July 2026 notice replaced both.
    assert "that newer rule applies" in SYSTEM_PROMPT


def test_prompt_puts_the_service_and_first_step_first_for_what_to_do_questions():
    # "My son (7) lost his passport at Disney World, what should I do?" was
    # answered from the right checklist but opened with VFS online
    # registration, with the police report buried in the second bullet.
    assert "names the application or service the person needs and their first step" in SYSTEM_PROMPT
    assert "never first" in SYSTEM_PROMPT
    assert "names the application or service needed and the first step" in SHORT_ANSWER_REMINDER


def test_short_answers_open_with_a_thank_you_line_and_full_answers_do_not():
    # Requested Oct 2026: "Thank you for asking about applying for an OCI
    # card." before the bullets, in the same message.
    assert "thanks the person and names the topic" in SYSTEM_PROMPT
    assert "do not start with a thank-you line" in SYSTEM_PROMPT  # the full answer after "yes"
    assert "thanking the person and naming the topic" in SHORT_ANSWER_REMINDER
    from app.generation.prompt import FULL_ANSWER_REMINDER

    assert "without a thank-you line" in FULL_ANSWER_REMINDER
