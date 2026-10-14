"""Retrieval/guardrail tuning from the Oct 2026 test set: in-scope questions
that named none of the old domain words were declined as off-topic, and VFS
boilerplate crowded the Consulate's own pages out of the top results."""

import types

from app.core.config import Settings
from app.generation import service
from app.retrieval import retriever


def settings(**over):
    return Settings(_env_file=None, retrieval_rerank=False, **over)


def test_service_questions_without_the_old_keywords_are_in_scope(monkeypatch):
    monkeypatch.setattr(service, "get_settings", lambda: settings())
    for q in [
        "Who is considered an applicant for Power of Attorney?",
        "My I-130 and I-485 are pending, what should I submit as my valid status?",
        "When I try to schedule an appointment it says I already have one",
        "How long does GEP verification take?",
    ]:
        assert service._mentions_domain_keyword(q), q


def test_generic_words_do_not_make_a_question_in_scope(monkeypatch):
    monkeypatch.setattr(service, "get_settings", lambda: settings())
    assert not service._mentions_domain_keyword("How do I write a job application letter?")
    assert not service._mentions_domain_keyword("What is the capital of France?")


def fake_point(pid, score, canonical):
    return types.SimpleNamespace(id=pid, score=score, payload={"canonical": canonical, "source_url": pid, "chunk_text": pid})


def run_search(monkeypatch, points, **over):
    monkeypatch.setattr(retriever, "get_settings", lambda: settings(**over))
    seen = {}

    def fake_hybrid(client, **kw):
        seen["limit"] = kw["limit"]
        return points[: kw["limit"]]

    monkeypatch.setattr(retriever, "hybrid_search", fake_hybrid)
    emb = types.SimpleNamespace(dense=[0.0], sparse_indices=[], sparse_values=[])
    return retriever.search("q", limit=2, client=object(), query_embedding=emb), seen


def test_a_near_tied_official_chunk_is_promoted_into_the_results(monkeypatch):
    points = [fake_point("vfs-a", 0.50, False), fake_point("vfs-b", 0.48, False), fake_point("cgi", 0.46, True)]
    results, seen = run_search(monkeypatch, points, retrieval_canonical_boost=0.05)
    assert seen["limit"] == 6  # fetches a few extra so a near tie can be promoted
    assert [r["source_url"] for r in results] == ["cgi", "vfs-a"]
    assert [r["rank"] for r in results] == [1, 2]


def test_a_clearly_better_match_is_not_displaced_by_the_boost(monkeypatch):
    points = [fake_point("vfs-a", 0.75, False), fake_point("vfs-b", 0.67, False), fake_point("cgi", 0.20, True)]
    results, _ = run_search(monkeypatch, points, retrieval_canonical_boost=0.05)
    assert [r["source_url"] for r in results] == ["vfs-a", "vfs-b"]


def test_pinned_chunks_returns_only_approved_chunks_of_the_pinned_page_in_order():
    import uuid

    from qdrant_client import QdrantClient

    from app.vectorstore.qdrant_store import build_point, ensure_collection, upsert_points

    client = QdrantClient(":memory:")
    ensure_collection(client)

    def pt(url, idx, status):
        return build_point(uuid.uuid4(), [0.1] * 1024, [1], [0.5],
                           {"source_url": url, "chunk_index": idx, "review_status": status, "canonical": True,
                            "chunk_text": f"{url}#{idx}"})

    contact = "https://indiainatlanta.gov.in/eoial_pages/MzIy"
    upsert_points(client, [pt(contact, 1, "approved"), pt(contact, 0, "approved"), pt(contact, 2, "rejected"),
                           pt("https://other/page", 0, "approved")])
    got = retriever.pinned_chunks([contact], client=client)
    assert [c["chunk_text"] for c in got] == [f"{contact}#0", f"{contact}#1"]
    assert retriever.pinned_chunks([], client=client) == []


def test_pinned_chunks_failure_returns_nothing():
    class Broken:
        def scroll(self, **kw):
            raise ConnectionError("down")

    assert retriever.pinned_chunks(["https://x"], client=Broken()) == []


def test_obfuscated_consulate_emails_are_decoded_for_the_model():
    from app.generation.prompt import deobfuscate_emails

    assert deobfuscate_emails("email: miscservices[dot]atlanta[at]mea[dot]gov[dot]in]") == \
        "email: miscservices.atlanta@mea.gov.in]"
    assert deobfuscate_emails("visa[dot]atlanta[at]mea[dot]gov[dot] in") == "visa.atlanta@mea.gov.in"
    assert deobfuscate_emails("Email: VFS_VisaATL[at]vfsglobal[dot]com") == "Email: VFS_VisaATL@vfsglobal.com"
    assert deobfuscate_emails("Apply at the portal [1].") == "Apply at the portal [1]."


def test_boost_zero_keeps_fusion_order(monkeypatch):
    points = [fake_point("vfs-a", 0.50, False), fake_point("cgi", 0.49, True)]
    results, seen = run_search(monkeypatch, points, retrieval_canonical_boost=0.0)
    assert seen["limit"] == 2
    assert [r["source_url"] for r in results] == ["vfs-a", "cgi"]
