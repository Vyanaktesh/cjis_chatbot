"""'grounded' drives whether the widget offers the "raise a query with the
consulate" form, so it must only be True for answers backed by cited sources."""
import pytest

from app.generation import service


class FakeEmbedder:
    def __init__(self, batch_size=1):
        pass

    def embed(self, texts):
        return [object() for _ in texts]


CHUNKS = [
    {
        "chunk_id": "c1",
        "source_url": "https://example.gov/oci",
        "service_category": "oci",
        "canonical": True,
        "heading_trail": [],
        "chunk_text": "OCI applications are submitted online.",
    },
    {
        "chunk_id": "c2",
        "source_url": "https://example.gov/passport",
        "service_category": "passport",
        "canonical": True,
        "heading_trail": [],
        "chunk_text": "Passport renewal is done at VFS.",
    },
]


class FakeGenerator:
    def __init__(self, text):
        self.text = text

    def generate(self, messages):
        return self.text


@pytest.fixture(autouse=True)
def stub_retrieval(monkeypatch):
    monkeypatch.setattr(service, "BgeM3Embedder", FakeEmbedder)
    monkeypatch.setattr(service, "probe_relevance", lambda *a, **k: 0.9)
    monkeypatch.setattr(service, "retrieval_search", lambda *a, **k: CHUNKS)


def ask(model_text):
    return service.answer_question("How long does renewal take?", generator=FakeGenerator(model_text))


def test_answer_with_a_real_citation_is_grounded():
    result = ask("Submit the form online [1].")
    assert result["grounded"] is True
    assert [c["index"] for c in result["citations"]] == [1]


def test_model_saying_it_has_no_approved_information_is_not_grounded():
    result = ask("I don't have approved information covering that. Please contact the consulate directly.")
    assert result["grounded"] is False  # so the widget offers the HubSpot form
    assert result["citations"] == []


def test_a_decline_is_not_grounded_even_if_it_cites_a_source():
    result = ask("I don't have approved information covering that [1].")
    assert result["grounded"] is False
    assert result["citations"] == []  # no source chips under "nothing found"


@pytest.mark.parametrize(
    "model_text",
    [
        "Processing usually takes a few weeks.",  # confident answer with no source behind it
        "Submit it online [7].",  # cites a source number that doesn't exist
        "",  # model returned nothing
    ],
)
def test_answers_without_a_valid_citation_are_not_grounded(model_text):
    assert ask(model_text)["grounded"] is False


def test_ordinary_use_of_the_words_does_not_look_like_a_decline():
    result = ask("If you do not have your old card, bring proof of identity [1].")
    assert result["grounded"] is True
    assert len(result["citations"]) == 1
