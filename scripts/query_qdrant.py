#!/usr/bin/env python3
"""
Phase 4 verification script: raw similarity search directly against
Qdrant (not through any retrieval-layer code, which doesn't exist until
Phase 6) — proves the dense vectors are meaningful AND that metadata
filtering (service_category, canonical) actually narrows results the way
it should.

Usage:
    python scripts/query_qdrant.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.embedding.bge_m3 import BgeM3Embedder  # noqa: E402
from app.vectorstore.qdrant_store import build_filter, dense_search, get_qdrant_client  # noqa: E402


def show_results(label: str, results) -> None:
    print(f"\n--- {label} ({len(results)} result(s)) ---")
    for r in results:
        p = r.payload
        snippet = p["chunk_text"][:100].replace("\n", " ")
        print(
            f"  score={r.score:.4f}  category={p['service_category']:<22} canonical={p['canonical']!s:<5}  "
            f"{snippet}..."
        )


def main() -> int:
    embedder = BgeM3Embedder(batch_size=4)

    query = "What documents do I need to submit an OCI application as an adult?"
    print(f"Query: {query!r}\n")
    query_emb = embedder.embed([query])[0]

    client = get_qdrant_client()

    # 1. Unfiltered — should surface OCI checklist chunks near the top
    unfiltered = dense_search(client, query_emb.dense, limit=5)
    show_results("Unfiltered top 5", unfiltered)

    # 2. Filtered to service_category='oci' — every result must be OCI
    oci_filter = build_filter(service_category="oci")
    oci_results = dense_search(client, query_emb.dense, limit=5, query_filter=oci_filter)
    show_results("Filtered: service_category=oci", oci_results)
    oci_ok = all(r.payload["service_category"] == "oci" for r in oci_results) and len(oci_results) > 0

    # 3. Filtered to service_category='visa' (same query, unrelated category)
    #    — proves the filter actually excludes non-matching chunks, not
    #    just that matching ones happen to rank first.
    visa_filter = build_filter(service_category="visa")
    visa_results = dense_search(client, query_emb.dense, limit=5, query_filter=visa_filter)
    show_results("Filtered: service_category=visa (same OCI query)", visa_results)
    visa_ok = all(r.payload["service_category"] == "visa" for r in visa_results)

    # 4. Filtered to canonical=true — every result must be a gov.in/mea.gov.in source
    canonical_filter = build_filter(canonical=True)
    canonical_results = dense_search(client, query_emb.dense, limit=5, query_filter=canonical_filter)
    show_results("Filtered: canonical=true", canonical_results)
    canonical_ok = all(r.payload["canonical"] is True for r in canonical_results) and len(canonical_results) > 0

    # 5. Combined filter: canonical=true AND service_category=status_tracking
    combined_filter = build_filter(service_category="status_tracking", canonical=True)
    combined_results = dense_search(client, query_emb.dense, limit=5, query_filter=combined_filter)
    show_results("Filtered: service_category=status_tracking AND canonical=true", combined_results)
    combined_ok = all(
        r.payload["service_category"] == "status_tracking" and r.payload["canonical"] is True
        for r in combined_results
    )

    print("\n=== Checks ===")
    checks = {
        "unfiltered query returns results": len(unfiltered) > 0,
        "service_category=oci filter: all results are oci": oci_ok,
        "service_category=visa filter: all results are visa (query still about OCI)": visa_ok,
        "canonical=true filter: all results are canonical": canonical_ok,
        "combined filter: all results match both conditions": combined_ok,
    }
    all_pass = True
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        all_pass = all_pass and passed

    print("\nALL OK" if all_pass else "\nSOME CHECKS FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
