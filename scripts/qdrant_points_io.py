#!/usr/bin/env python3
"""
Copy the `chunks` Qdrant collection between machines as plain data.

    python scripts/qdrant_points_io.py export /path/points.jsonl.gz
    python scripts/qdrant_points_io.py import /path/points.jsonl.gz --replace

Export writes every point (id, dense + sparse vectors, full payload) as one
JSON line, gzipped. Import recreates the collection with the app's own
ensure_collection() settings and upserts the points in batches, then checks
the point count matches the file.

Used instead of Qdrant's snapshot files, which came out unrestorable from a
Qdrant whose storage sits on a Windows bind mount (corrupt WAL index). Ship it
together with a pg_dump taken at the same moment, so the two stores agree.
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client.http import models as qmodels  # noqa: E402

from app.vectorstore.qdrant_store import (  # noqa: E402
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    ensure_collection,
    get_qdrant_client,
)

BATCH = 128


def export(path: Path) -> int:
    qc = get_qdrant_client()
    n, offset = 0, None
    with gzip.open(path, "wt", encoding="utf-8") as f:
        while True:
            points, offset = qc.scroll(COLLECTION_NAME, limit=256, offset=offset, with_payload=True, with_vectors=True)
            for p in points:
                sparse = p.vector[SPARSE_VECTOR_NAME]
                f.write(json.dumps({
                    "id": str(p.id),
                    "dense": p.vector[DENSE_VECTOR_NAME],
                    "sparse_indices": sparse.indices,
                    "sparse_values": sparse.values,
                    "payload": p.payload,
                }) + "\n")
                n += 1
            if offset is None:
                break
    print(f"exported {n} point(s) to {path}")
    return 0


def import_(path: Path, replace: bool) -> int:
    qc = get_qdrant_client()
    if replace and COLLECTION_NAME in [c.name for c in qc.get_collections().collections]:
        qc.delete_collection(COLLECTION_NAME)
        print(f"deleted existing collection {COLLECTION_NAME!r}")
    ensure_collection(qc)
    n, batch = 0, []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            batch.append(qmodels.PointStruct(
                id=d["id"],
                vector={
                    DENSE_VECTOR_NAME: d["dense"],
                    SPARSE_VECTOR_NAME: qmodels.SparseVector(indices=d["sparse_indices"], values=d["sparse_values"]),
                },
                payload=d["payload"],
            ))
            if len(batch) == BATCH:
                qc.upsert(COLLECTION_NAME, points=batch, wait=True)
                n += len(batch)
                batch = []
    if batch:
        qc.upsert(COLLECTION_NAME, points=batch, wait=True)
        n += len(batch)
    count = qc.count(COLLECTION_NAME, exact=True).count
    print(f"imported {n} point(s); collection now holds {count}")
    return 0 if count == n else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["export", "import"])
    parser.add_argument("path", type=Path)
    parser.add_argument("--replace", action="store_true", help="import: drop the existing collection first")
    args = parser.parse_args()
    return export(args.path) if args.action == "export" else import_(args.path, args.replace)


if __name__ == "__main__":
    raise SystemExit(main())
