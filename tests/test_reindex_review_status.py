"""Re-indexing must never change what a reviewer decided, and Postgres and
Qdrant must always agree. Needs a reachable Postgres (skipped otherwise); it
creates and drops its own throwaway database, so real data is never touched.
Qdrant is the in-memory client."""
import glob
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip("psycopg2")

DB_NAME = f"qa_reindex_{uuid.uuid4().hex[:8]}"
MIGRATIONS = sorted(glob.glob(str(Path(__file__).resolve().parent.parent / "db" / "migrations" / "*.sql")))


def _admin_dsn():
    return dict(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        user=os.environ.get("POSTGRES_USER", "rag_admin"),
        password=os.environ.get("POSTGRES_PASSWORD", "change_me_dev_only"),
    )


@pytest.fixture(scope="module")
def env():
    try:
        admin = psycopg2.connect(dbname="postgres", connect_timeout=3, **_admin_dsn())
    except Exception as exc:  # no database available -> skip, don't fail
        pytest.skip(f"Postgres not reachable: {exc}")
    admin.autocommit = True
    admin.cursor().execute(f"CREATE DATABASE {DB_NAME}")
    scratch = psycopg2.connect(dbname=DB_NAME, **_admin_dsn())
    scratch.autocommit = True
    for path in MIGRATIONS:
        scratch.cursor().execute(open(path).read())
    scratch.close()

    os.environ["POSTGRES_DB"] = DB_NAME
    from app.core.config import get_settings

    get_settings.cache_clear()
    import app.db.connection as dbc

    dbc._pool = None
    yield
    if dbc._pool is not None:
        dbc._pool.closeall()
        dbc._pool = None
    admin.cursor().execute(f"DROP DATABASE IF EXISTS {DB_NAME} WITH (FORCE)")
    admin.close()
    os.environ.pop("POSTGRES_DB", None)
    get_settings.cache_clear()


@pytest.fixture()
def world(env):
    from qdrant_client import QdrantClient

    from app.chunking.chunker import Chunk
    from app.chunking.tag_metadata import build_chunk_records
    from app.db.connection import get_conn
    from app.db.source_versions_repo import create_version
    from app.db.sources_repo import upsert_source
    from app.embedding.bge_m3 import EmbeddingResult
    from app.ingestion.pipeline import chunk_id_for, index_records
    from app.vectorstore.qdrant_store import COLLECTION_NAME, build_filter

    qc = QdrantClient(":memory:")
    with get_conn() as conn:
        src = upsert_source(
            conn, url=f"https://example.test/{uuid.uuid4().hex}", source_type="html",
            service_category="passport", title="Passport fees",
        )
        v1 = create_version(
            conn, source_id=src.id, content_hash="h1",
            retrieval_date=datetime.now(timezone.utc), raw_content_path="x",
        )

    class W:
        def index(self, version, texts):
            chunks = [Chunk(text=t, chunk_index=i) for i, t in enumerate(texts)]
            recs = build_chunk_records(chunks, src, version)
            for r in recs:
                r["id"] = chunk_id_for(version.id, r["chunk_index"])
            embs = [EmbeddingResult(dense=[0.1] * 1024, sparse_indices=[1], sparse_values=[0.5]) for _ in recs]
            with get_conn() as conn:
                index_records(conn, qc, src, version, recs, embs)

        def new_version(self, content_hash):
            with get_conn() as conn:
                return create_version(
                    conn, source_id=src.id, content_hash=content_hash,
                    retrieval_date=datetime.now(timezone.utc), raw_content_path="y",
                )

        def approve(self, version, index):
            from app.review.service import decide_chunk

            with get_conn() as conn:
                decide_chunk(conn, qc, chunk_id_for(version.id, index), "approved", actor="test")

        def pg(self):
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT chunk_text, review_status FROM chunks WHERE source_id = %s ORDER BY version, chunk_index",
                    (str(src.id),),
                )
                return [(t.split("\n\n")[-1], s) for t, s in cur.fetchall()]

        def served(self):
            """Chunk texts retrieval is allowed to serve (Qdrant approved filter)."""
            points, _ = qc.scroll(
                COLLECTION_NAME, scroll_filter=build_filter(review_status="approved"), limit=100, with_payload=True
            )
            return sorted(p.payload["chunk_text"].split("\n\n")[-1] for p in points)

        v1_ = v1

    return W()


def test_reindexing_unchanged_content_keeps_approval_in_both_stores(world):
    world.index(world.v1_, ["Fee is $100", "Processing 10 days"])
    world.approve(world.v1_, 0)
    world.index(world.v1_, ["Fee is $100", "Processing 10 days"])  # script re-run, nothing changed

    assert world.pg() == [("Fee is $100", "approved"), ("Processing 10 days", "pending_review")]
    assert world.served() == ["Fee is $100"]


def test_reindexing_changed_content_resets_approval_in_both_stores(world):
    world.index(world.v1_, ["Fee is $100", "Processing 10 days"])
    world.approve(world.v1_, 0)
    world.index(world.v1_, ["Fee is $250 (changed)", "Processing 10 days"])

    # the new, never-reviewed text must NOT carry the old approval
    assert world.pg() == [("Fee is $250 (changed)", "pending_review"), ("Processing 10 days", "pending_review")]
    assert world.served() == []


def test_a_superseded_chunk_cannot_be_approved_again(world):
    from app.review.service import ChunkNotFound, ChunkNotReviewable

    world.index(world.v1_, ["Fee is $100"])
    v2 = world.new_version("h2")
    world.index(v2, ["Fee is $300 (new fee)"])  # supersedes v1's chunk
    assert ("Fee is $100", "superseded") in world.pg()

    with pytest.raises(ChunkNotReviewable) as excinfo:
        world.approve(world.v1_, 0)  # a reviewer clicks approve on the OLD chunk

    assert isinstance(excinfo.value, ChunkNotFound)  # callers that only handle "not found" still cope
    assert ("Fee is $100", "superseded") in world.pg()
    assert world.served() == []  # the outdated text did not go live again


def test_superseded_chunks_stay_superseded_when_reindexed(world):
    world.index(world.v1_, ["Fee is $100"])
    world.approve(world.v1_, 0)
    v2 = world.new_version("h2")
    world.index(v2, ["Fee is $300 (new fee)"])  # supersedes v1's chunks
    assert ("Fee is $100", "superseded") in world.pg()

    world.index(world.v1_, ["Fee is $100"])  # old version re-indexed again
    assert ("Fee is $100", "superseded") in world.pg()
    assert world.served() == []
