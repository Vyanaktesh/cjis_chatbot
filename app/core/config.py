"""
Central config loading for the whole project (all phases read from here).

Values come from environment variables, populated from a `.env` file in the
repo root during local dev (see .env.example). In containerized/prod
deployments, real environment variables can be set directly and .env is
not required.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_env: str = "development"
    log_level: str = "INFO"

    # --- PostgreSQL ---
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "rag_chatbot"
    postgres_user: str = "rag_admin"
    postgres_password: str = "change_me_dev_only"

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_http_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_api_key: str | None = None

    # --- Fetcher (used starting Phase 2, defined here so config is one place) ---
    fetcher_contact_email: str = ""
    fetcher_user_agent: str = "ConsulateRAGBot/0.1"

    # --- Retrieval reranking (added after kb_admin's eval golden set, built
    # from real citizen queries, showed short/keyword-style real queries
    # often rank the correct chunk too low from hybrid search alone — see
    # app/retrieval/reranker.py). Defaults OFF: BAAI/bge-reranker-v2-m3 is
    # too slow on CPU-only hardware for the live chat path (~2-8s per
    # candidate scored, measured live) — reintroducing the exact latency
    # problem that made this project switch generation to Gemini in the
    # first place. kb_admin's own .env explicitly turns this on for
    # offline eval runs (see consulate-kb-admin/.env), where slower
    # runtime is an acceptable trade for the accuracy signal; the live
    # chatbot's own .env should leave this False.
    retrieval_rerank: bool = False
    # How many candidates the initial hybrid search pulls before reranking
    # narrows down to the caller's actual `limit` — wide enough that a
    # correct-but-lower-ranked chunk has a real chance to be in the pool,
    # small enough that cross-encoder scoring (much slower per-item than
    # the bi-encoder search that precedes it) stays fast on CPU.
    retrieval_rerank_candidates: int = 30
    # Off-topic guardrail (e.g. "what is the capital of France?"): the RRF
    # fusion score hybrid_search returns is rank-based, not a calibrated
    # similarity, so it can't be thresholded meaningfully. Instead
    # app.retrieval.retriever.probe_relevance() runs one raw dense-only
    # top-1 lookup and compares BGE-M3 cosine similarity against this
    # floor; below it, the turn is declined before ever calling retrieval's
    # full hybrid search or the generation backend (saves latency and, on
    # the Gemini backend, quota). 0.55 was picked by probing this live
    # deployment's actual approved-chunk set: off-topic queries ("what is
    # a cat", "capital of France") scored 0.36-0.45 raw BGE-M3 cosine
    # similarity against their closest chunk, genuinely on-topic queries
    # ("how do I renew my passport", "OCI documents needed") scored
    # 0.70-0.73 -- a wide, clean gap, so 0.55 sits in the middle with
    # margin both directions. Not tuned against kb_admin's eval golden set
    # though -- re-probe with app.retrieval.retriever.probe_relevance() if
    # genuinely on-topic questions start getting declined, or off-topic
    # ones slip through, since the gap's exact location will shift as more
    # sources get approved into the chunk set.
    retrieval_min_relevance: float = 0.55

    # --- Generation (Phase 7) ---
    # Self-hosted, CPU-only GGUF model via llama.cpp — no third-party API.
    # Qwen3-4B (not the originally-specced 8B) is a deliberate, documented
    # trade-off for this build sandbox's ~2 CPU / ~8GB RAM ceiling; swapping
    # to Qwen3-8B (or larger) on real deployment hardware is just this one
    # path plus a re-download, no code change.
    generation_model_path: str = "data/models/Qwen3-4B-Q4_K_M.gguf"
    generation_n_ctx: int = 4096
    generation_n_threads: int = 2
    generation_max_tokens: int = 350

    # --- Generation backend selection (Phase 7 follow-up) ---
    # "qwen" (default): self-hosted via llama.cpp, nothing leaves this
    # machine. "gemini": routes generation to Google's API instead -- much
    # faster, but every question + retrieved source excerpt is sent to
    # Google for that turn. Opt-in only; see app/generation/gemini_backend.py.
    generation_backend: str = "qwen"

    # --- Gemini API (only used when generation_backend == "gemini") ---
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    # Separate token budget from generation_max_tokens above -- that value
    # (350) was deliberately kept low to bound worst-case *latency* on the
    # CPU-only Qwen path, which doesn't apply to Gemini. Gemini 2.5 Flash is
    # a "thinking" model: internal reasoning tokens are deducted from
    # max_output_tokens *before* the visible answer, so a low cap here can
    # silently truncate or empty out the real answer (a documented Gemini
    # 2.5 Flash gotcha, not specific to this project). thinking_budget=0
    # disables that reasoning step entirely -- unnecessary overhead for a
    # straightforward "cite from these sources" task anyway -- so the full
    # budget goes to the visible answer.
    gemini_max_output_tokens: int = 1024
    gemini_thinking_budget: int = 0

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Settings are cached — env is read once per process."""
    return Settings()
