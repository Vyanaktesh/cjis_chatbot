from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.retrieval.retriever import has_approved_content
from app.vectorstore import qdrant_store
from app.vectorstore.qdrant_store import COLLECTION_NAME, build_point, ensure_collection, get_qdrant_client


def test_one_shared_client_is_reused():
    get_qdrant_client.cache_clear()
    assert get_qdrant_client() is get_qdrant_client()
    get_qdrant_client.cache_clear()


def point(status: str):
    import uuid

    return build_point(uuid.uuid4(), [0.1] * 1024, [1], [0.5], {"review_status": status})


def test_has_approved_content_reflects_what_is_actually_approved():
    client = QdrantClient(":memory:")
    ensure_collection(client)
    assert has_approved_content(client) is False  # empty collection

    qdrant_store.upsert_points(client, [point("pending_review"), point("rejected")])
    assert has_approved_content(client) is False  # nothing approved yet

    qdrant_store.upsert_points(client, [point("approved")])
    assert has_approved_content(client) is True


def test_has_approved_content_assumes_yes_if_the_check_itself_fails():
    class Broken:
        def count(self, **kwargs):
            raise ConnectionError("down")

    # A hiccup must not change which message a visitor sees.
    assert has_approved_content(Broken()) is True
