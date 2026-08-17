"""
Qdrant collection setup + upsert/search helpers for the `chunks`
collection: hybrid dense+sparse vectors per point, full metadata payload
so retrieval-time filtering (service_category, canonical, review_status,
jurisdiction, ...) works without a second round-trip to Postgres.
"""

from typing import Any, Optional
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import get_settings

COLLECTION_NAME = "chunks"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_DIM = 1024


def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_http_port,
        grpc_port=settings.qdrant_grpc_port,
        api_key=settings.qdrant_api_key or None,
        https=False,
    )


def ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: qmodels.VectorParams(size=DENSE_DIM, distance=qmodels.Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(),
        },
    )


def build_point(
    chunk_id: UUID,
    dense: list[float],
    sparse_indices: list[int],
    sparse_values: list[float],
    payload: dict[str, Any],
) -> qmodels.PointStruct:
    return qmodels.PointStruct(
        id=str(chunk_id),
        vector={
            DENSE_VECTOR_NAME: dense,
            SPARSE_VECTOR_NAME: qmodels.SparseVector(indices=sparse_indices, values=sparse_values),
        },
        payload=payload,
    )


def upsert_points(client: QdrantClient, points: list[qmodels.PointStruct]) -> None:
    if not points:
        return
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def build_filter(
    service_category: Optional[str] = None,
    canonical: Optional[bool] = None,
    review_status: Optional[str] = None,
    source_id: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    applicant_variant: Optional[str] = None,
) -> Optional[qmodels.Filter]:
    """`jurisdiction`/`applicant_variant` payload fields are arrays (e.g.
    `["all"]`, `["adult"]`); a `MatchValue` condition against an array
    payload field matches if the array CONTAINS that value, not if the
    whole array equals it — which is what we want ("this chunk applies to
    adults" should match a chunk tagged `["adult", "minor"]` too)."""
    conditions = []
    if service_category is not None:
        conditions.append(
            qmodels.FieldCondition(key="service_category", match=qmodels.MatchValue(value=service_category))
        )
    if canonical is not None:
        conditions.append(qmodels.FieldCondition(key="canonical", match=qmodels.MatchValue(value=canonical)))
    if review_status is not None:
        conditions.append(
            qmodels.FieldCondition(key="review_status", match=qmodels.MatchValue(value=review_status))
        )
    if source_id is not None:
        conditions.append(qmodels.FieldCondition(key="source_id", match=qmodels.MatchValue(value=source_id)))
    if jurisdiction is not None:
        conditions.append(qmodels.FieldCondition(key="jurisdiction", match=qmodels.MatchValue(value=jurisdiction)))
    if applicant_variant is not None:
        conditions.append(
            qmodels.FieldCondition(key="applicant_variant", match=qmodels.MatchValue(value=applicant_variant))
        )
    if not conditions:
        return None
    return qmodels.Filter(must=conditions)


def set_payload_fields(client: QdrantClient, point_ids: list, payload: dict[str, Any]) -> None:
    """Patches specific payload fields on existing points without touching
    their vectors or other payload fields — used to keep Qdrant's
    review_status in sync with Postgres after an approve/reject decision or
    a supersede-on-new-version event, without re-embedding anything."""
    if not point_ids:
        return
    client.set_payload(
        collection_name=COLLECTION_NAME,
        payload=payload,
        points=[str(p) for p in point_ids],
    )


def dense_search(
    client: QdrantClient,
    query_dense: list[float],
    *,
    limit: int = 5,
    query_filter: Optional[qmodels.Filter] = None,
):
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_dense,
        using=DENSE_VECTOR_NAME,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    ).points


def hybrid_search(
    client: QdrantClient,
    *,
    query_dense: list[float],
    query_sparse_indices: list[int],
    query_sparse_values: list[float],
    limit: int = 10,
    prefetch_limit: int = 50,
    query_filter: Optional[qmodels.Filter] = None,
):
    """
    Native Qdrant Query API hybrid search: runs the dense leg and the
    sparse leg as independent ANN searches (each already narrowed by
    `query_filter` — filtering happens INSIDE each leg's candidate search,
    not as a post-filter after fusion, so a tight filter can't starve one
    leg of candidates the other leg would have found), then fuses the two
    ranked lists with Reciprocal Rank Fusion (RRF). RRF combines by rank
    position rather than raw score, which sidesteps the fact that dense
    cosine similarity and sparse lexical scores live on totally different
    scales and can't be meaningfully averaged directly.
    """
    return client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            qmodels.Prefetch(
                query=query_dense,
                using=DENSE_VECTOR_NAME,
                filter=query_filter,
                limit=prefetch_limit,
            ),
            qmodels.Prefetch(
                query=qmodels.SparseVector(indices=query_sparse_indices, values=query_sparse_values),
                using=SPARSE_VECTOR_NAME,
                filter=query_filter,
                limit=prefetch_limit,
            ),
        ],
        query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
        limit=limit,
        with_payload=True,
    ).points
