"""Citations must collapse to one per unique source document: an answer that
cites [1][2][3] where all three chunks share a source URL should show a single
source chip, with the answer's bracket markers renumbered to match."""
import os

os.environ.setdefault("WARMUP_ON_STARTUP", "false")

from app.generation.service import _dedupe_citations_by_source  # noqa: E402


def _chunk(cid, url, cat="oci"):
    return {"chunk_id": cid, "source_url": url, "service_category": cat, "canonical": True}


def test_same_source_collapses_to_one_citation():
    chunks = [
        _chunk("a", "https://vfs.example/oci.pdf"),
        _chunk("b", "https://vfs.example/oci.pdf"),
        _chunk("c", "https://vfs.example/oci.pdf"),
    ]
    answer = "Apply online [1][2][3]. Upload a photo [2] and sign [3]."
    new_answer, citations = _dedupe_citations_by_source(answer, chunks)

    assert len(citations) == 1
    assert citations[0]["index"] == 1
    assert citations[0]["source_url"] == "https://vfs.example/oci.pdf"
    # Adjacent duplicate markers collapse; repeated later refs become [1].
    assert "[1][2][3]" not in new_answer
    assert "[2]" not in new_answer and "[3]" not in new_answer
    assert "Apply online [1]." in new_answer


def test_distinct_sources_keep_separate_numbers():
    chunks = [
        _chunk("a", "https://vfs.example/oci.pdf"),
        _chunk("b", "https://mea.example/passport.pdf"),
        _chunk("c", "https://vfs.example/oci.pdf"),
    ]
    # chunk 1 and 3 share a source; chunk 2 is a different source.
    answer = "First [1]. Second [2]. Third [3]."
    new_answer, citations = _dedupe_citations_by_source(answer, chunks)

    assert len(citations) == 2
    urls = {c["source_url"]: c["index"] for c in citations}
    assert urls["https://vfs.example/oci.pdf"] == 1
    assert urls["https://mea.example/passport.pdf"] == 2
    # chunk 3 (same source as chunk 1) is renumbered to [1].
    assert "Third [1]." in new_answer
    assert "Second [2]." in new_answer


def test_out_of_range_markers_are_dropped():
    chunks = [_chunk("a", "https://vfs.example/oci.pdf")]
    answer = "Valid [1], bogus [9]."
    new_answer, citations = _dedupe_citations_by_source(answer, chunks)
    assert len(citations) == 1
    assert "[9]" not in new_answer
    assert "[1]" in new_answer
