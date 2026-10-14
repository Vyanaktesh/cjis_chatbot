"""Query rewriting before retrieval (app/retrieval/query_rewrite.py) and the
fused search in app/generation/service.py."""
import time

import pytest

from app.core.config import get_settings
from app.generation import service
from app.retrieval import query_rewrite
from app.retrieval.query_rewrite import REWRITE_SYSTEM_PROMPT, build_rewrite_messages, clean_rewrite
from app.retrieval.retriever import fuse_results


def chunk(cid, url=None):
    return {
        "chunk_id": cid,
        "source_url": url or f"https://example.gov/{cid}",
        "service_category": "passport",
        "canonical": True,
        "heading_trail": [],
        "chunk_text": f"text of {cid}",
    }


GENERIC = chunk("generic-reissue")
LOST_MINOR = chunk("lost-minor-checklist")
NOISY = "We went to Disney World in July and my son's Indian passport was lost in the hotel, he is 7. How do I get a new one?"
REWRITE = "lost Indian passport re-issue minor child"


class Gen:
    """Rewrites to REWRITE (or as told) and answers everything else; records
    what it was asked."""

    def __init__(self, rewrite=REWRITE, answer="Report the loss and apply for re-issue [1]."):
        self.rewrite, self.answer, self.calls = rewrite, answer, []

    def generate(self, messages, **kwargs):
        is_rewrite = messages[0]["content"] == REWRITE_SYSTEM_PROMPT
        self.calls.append(("rewrite" if is_rewrite else "answer", messages))
        if is_rewrite:
            if isinstance(self.rewrite, Exception):
                raise self.rewrite
            if callable(self.rewrite):
                return self.rewrite()
            return self.rewrite
        return self.answer


class FakeEmbedder:
    def __init__(self, batch_size=1):
        pass

    def embed(self, texts):
        return [object() for _ in texts]


@pytest.fixture
def searched(monkeypatch):
    """Rewriting on (Gemini backend), retrieval stubbed: the original wording
    finds the generic page first, the rewrite finds the lost-minor checklist."""
    settings = get_settings()
    monkeypatch.setattr(settings, "generation_backend", "gemini")
    monkeypatch.setattr(settings, "query_rewrite_enabled", True)
    monkeypatch.setattr(settings, "query_rewrite_timeout_seconds", 3.0)
    monkeypatch.setattr(service, "BgeM3Embedder", FakeEmbedder)
    monkeypatch.setattr(service, "probe_relevance", lambda *a, **k: 0.9)
    monkeypatch.setattr(service, "has_approved_content", lambda *a, **k: True)
    monkeypatch.setattr(service, "pinned_chunks", lambda *a, **k: [])
    queries = []

    def fake_search(q, **kwargs):
        queries.append(q)
        # The original wording ranks the lost-minor checklist 9th, outside an
        # 8-chunk answer; the rewrite ranks it first.
        if q == REWRITE:
            return [LOST_MINOR] + OTHERS[8:]
        return [GENERIC] + OTHERS[:7] + [LOST_MINOR]

    monkeypatch.setattr(service, "retrieval_search", fake_search)
    return queries


OTHERS = [chunk(f"other-{i}") for i in range(16)]


def chunks_given_to_model(gen):
    answer_messages = [m for kind, m in gen.calls if kind == "answer"][0]
    return answer_messages[-1]["content"]


# --- cleaning the model's reply ---------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("lost OCI card replacement", "lost OCI card replacement"),
        ('"passport re-issue change of address"', "passport re-issue change of address"),
        ("Query: OCI processing time Atlanta", "OCI processing time Atlanta"),
        ("\n\n  surrender certificate lost passport \nextra line", "surrender certificate lost passport"),
        ("", None),
        (None, None),
        ("   ", None),
        ("x" * 250, None),  # answered instead of rewriting
    ],
)
def test_clean_rewrite(raw, expected):
    assert clean_rewrite(raw) == expected


def test_earlier_questions_are_given_for_follow_ups():
    messages = build_rewrite_messages("and for my son?", ["Do I need to reissue my OCI card?"])
    assert messages[0]["content"] == REWRITE_SYSTEM_PROMPT
    assert "Do I need to reissue my OCI card?" in messages[1]["content"]
    assert "New message: and for my son?" in messages[1]["content"]


# --- fusion ------------------------------------------------------------------

def test_fusion_puts_what_both_wordings_found_first():
    a, b, c = chunk("a"), chunk("b"), chunk("c")
    fused = fuse_results([[a, b], [b, c]], [1.0, 1.0], limit=3)
    assert [r["chunk_id"] for r in fused] == ["b", "a", "c"]
    assert [r["rank"] for r in fused] == [1, 2, 3]


def test_fusion_weight_favours_the_heavier_list():
    a, b = chunk("a"), chunk("b")
    assert fuse_results([[a, b], [b, a]], [1.0, 2.0], limit=2)[0]["chunk_id"] == "b"


def test_each_wordings_top_results_survive_shared_middling_results():
    # "my passport expires next year... how long does re-issue take?": the
    # rewrite ranked the processing-time page first, but both wordings shared
    # many lower-ranked checklists, which out-scored it under plain fusion.
    checklists = [chunk(f"checklist-{i}") for i in range(10)]
    processing = [chunk(f"processing-time-{i}") for i in range(3)]
    original = checklists
    rewrite = processing + checklists
    plain = fuse_results([original, rewrite], [1.0, 1.0], limit=8)
    assert not any(r["chunk_id"].startswith("processing") for r in plain)  # the blind spot
    kept = fuse_results([original, rewrite], [1.0, 1.0], limit=8, keep_top=2)
    ids = [r["chunk_id"] for r in kept]
    assert {"processing-time-0", "processing-time-1", "checklist-0", "checklist-1"} <= set(ids)
    assert len(ids) == 8 and [r["rank"] for r in kept] == list(range(1, 9))


def test_a_poor_rewrite_cannot_push_out_the_originals_best_results():
    original = [chunk(f"o{i}") for i in range(16)]
    poor_rewrite = [chunk(f"w{i}") for i in range(16)]  # shares nothing
    fused = fuse_results([original, poor_rewrite], [1.0, get_settings().retrieval_rewrite_weight], limit=8)
    assert {"o0", "o1", "o2", "o3"} <= {r["chunk_id"] for r in fused}


# --- the pipeline ------------------------------------------------------------

def test_noisy_question_is_searched_with_both_wordings(searched):
    gen = Gen()
    result = service.answer_question(NOISY, generator=gen)
    assert searched == [NOISY, REWRITE]
    assert result["grounded"] is True
    given = chunks_given_to_model(gen)
    # The checklist only the rewrite ranked highly now reaches the model, and
    # the original wording's top result is still there.
    assert "text of lost-minor-checklist" in given
    assert "text of generic-reissue" in given


def test_the_answer_is_written_to_the_visitors_own_words(searched):
    gen = Gen()
    service.answer_question(NOISY, generator=gen)
    answer_messages = [m for kind, m in gen.calls if kind == "answer"][0]
    user_text = answer_messages[-1]["content"]
    assert "Disney World" in user_text  # the original question, not the rewrite


def test_a_failed_rewrite_falls_back_to_the_original_search(searched):
    result = service.answer_question(NOISY, generator=Gen(rewrite=RuntimeError("quota")))
    assert searched == [NOISY]
    assert result["grounded"] is True


def test_a_slow_rewrite_is_not_waited_for(searched, monkeypatch):
    monkeypatch.setattr(get_settings(), "query_rewrite_timeout_seconds", 0.05)

    def slow():
        time.sleep(0.5)
        return REWRITE

    started = time.perf_counter()
    service.answer_question(NOISY, generator=Gen(rewrite=slow))
    assert searched == [NOISY]
    assert time.perf_counter() - started < 0.45
    time.sleep(0.6)  # let the abandoned rewrite finish before logging shuts down


def test_an_unusable_rewrite_falls_back(searched):
    service.answer_question(NOISY, generator=Gen(rewrite=""))
    assert searched == [NOISY]


def test_a_more_details_reply_is_not_rewritten(searched):
    gen = Gen()
    history = [{"role": "user", "content": "How do I renew my passport?"}, {"role": "assistant", "content": "- Apply via VFS."}]
    service.answer_question("Yes, more details please", history=history, generator=gen)
    assert not [kind for kind, _ in gen.calls if kind == "rewrite"]


def test_follow_up_rewrite_sees_the_earlier_question(searched):
    gen = Gen()
    history = [{"role": "user", "content": "Do I need to reissue my OCI card?"}, {"role": "assistant", "content": "- No."}]
    service.answer_question("and for my son?", history=history, generator=gen)
    rewrite_messages = [m for kind, m in gen.calls if kind == "rewrite"][0]
    assert "Do I need to reissue my OCI card?" in rewrite_messages[1]["content"]


def test_no_rewrite_on_the_qwen_backend(searched, monkeypatch):
    monkeypatch.setattr(get_settings(), "generation_backend", "qwen")
    gen = Gen()
    service.answer_question(NOISY, generator=gen)
    assert searched == [NOISY]
    assert not [kind for kind, _ in gen.calls if kind == "rewrite"]


def test_rewriting_does_not_open_the_off_topic_gate(searched, monkeypatch):
    # The off-topic check still looks at the visitor's own words only.
    monkeypatch.setattr(service, "probe_relevance", lambda *a, **k: 0.2)
    result = service.answer_question("What is the capital of France?", generator=Gen(rewrite="passport re-issue"))
    assert result.get("out_of_scope") is True
    assert searched == []


def test_rewrite_switch_off(searched, monkeypatch):
    monkeypatch.setattr(get_settings(), "query_rewrite_enabled", False)
    service.answer_question(NOISY, generator=Gen())
    assert searched == [NOISY]
    assert query_rewrite.rewrite_active() is False
