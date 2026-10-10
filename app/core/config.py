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

    # --- CORS ---
    # Comma-separated list of allowed origins, or "*" for any origin.
    # Defaults to "*" so local dev (the Vite dev server on a random port)
    # keeps working out of the box, but this is the one setting that MUST
    # be locked down before a real deployment: with "*", any website can
    # call this API's write endpoints (/support/ticket,
    # /citizen-corner/submit) from a visitor's browser using their session.
    # Set to the real consulate site's origin(s) in production, e.g.
    # CORS_ALLOWED_ORIGINS=https://www.indiainatlanta.gov.in
    cors_allowed_origins: str = "*"

    # --- Rate limiting (slowapi/limits syntax: "<n>/<period>", e.g.
    # "20/minute") --- keyed on client IP. Defaults are generous enough for
    # normal use but bound the worst case: unlimited requests to /chat or
    # /generate could exhaust the Gemini free-tier quota or run up a paid
    # bill in minutes; unlimited /support/ticket or /citizen-corner/submit
    # calls could spam real tickets into HubSpot.
    rate_limit_generate: str = "20/minute"
    rate_limit_search: str = "60/minute"
    rate_limit_submit: str = "5/minute"
    # Where the rate-limit counters live. "memory://" is per-process; use a
    # shared store such as "redis://host:6379" when running multiple workers
    # (needs the `redis` package installed).
    rate_limit_storage_uri: str = "memory://"

    # --- PostgreSQL ---
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "rag_chatbot"
    postgres_user: str = "rag_admin"
    postgres_password: str = "change_me_dev_only"
    # Connection pool bounds + safety timeouts (see app/db/connection.py).
    # A per-request new connection exhausts Postgres under load; the pool
    # caps concurrent connections. connect_timeout bounds how long a dead
    # DB can block; statement_timeout bounds a runaway query.
    postgres_pool_min: int = 1
    postgres_pool_max: int = 20
    postgres_connect_timeout: int = 10
    postgres_statement_timeout_ms: int = 30000
    # How long to wait for a free pooled connection when all are in use
    # before giving up. psycopg2's pool raises immediately on exhaustion, so
    # get_conn retries within this window to turn a burst into a short wait
    # instead of a hard error.
    postgres_pool_acquire_timeout: float = 10.0

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_http_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_api_key: str | None = None
    # Default False keeps the local-docker setup working over plain HTTP.
    # MUST be set True for any remote/managed Qdrant, otherwise the API key
    # above is transmitted in cleartext. Also caps how long a slow/unreachable
    # Qdrant can block a request thread.
    qdrant_https: bool = False
    qdrant_timeout: float = 30.0

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
    # Words that mark a question as on-topic even when its embedding similarity
    # lands just under retrieval_min_relevance. Very short questions (a bare
    # "OCI" scored 0.547 against the 0.55 floor) carry little signal for the
    # embedding model, so an explicit domain-word check backs it up. Comma
    # separated, matched as whole words/phrases, case-insensitive.
    retrieval_domain_keywords: str = (
        "oci,pio,passport,visa,consulate,consular,renunciation,surrender,"
        "attestation,attest,apostille,vfs,police clearance,pcc,birth certificate,"
        "death certificate,emergency certificate,miscellaneous,overseas citizen"
    )

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
    # gemini-2.5-flash is retired for new API accounts (Google returns 404 and
    # points to the 3.x line), so default to a current model. Override via
    # GEMINI_MODEL in .env if your account needs a different one.
    gemini_model: str = "gemini-3.8-flash"
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
    # Per-call HTTP timeout. The SDK's default is no timeout at all, so one
    # stalled Google call would hold a worker thread forever and, with enough
    # of them, freeze the whole API. gemini_total_budget_seconds bounds a
    # request including its rate-limit retries.
    gemini_timeout_seconds: float = 20.0
    gemini_total_budget_seconds: float = 45.0

    # --- HubSpot escalation (Phase 8 follow-up) ---
    # When the chatbot can't ground an answer (see app/generation/service.py's
    # OUT_OF_SCOPE_ANSWER / NO_CONTEXT_ANSWER / GENERATION_UNAVAILABLE_ANSWER
    # -- all set grounded=False), the widget offers to escalate to a human:
    # a Contact is created/updated by email, then a Ticket is created and
    # associated to it, landing in a real HubSpot Service Hub queue a
    # consulate staff member can reply from. Server-side only -- this token
    # is never sent to the frontend. Get one from a HubSpot private app
    # (Settings > Integrations > Private Apps) with crm.objects.contacts.write
    # and crm.objects.tickets.write scopes.
    hubspot_access_token: str | None = None
    # HubSpot's default Support Pipeline ("0") and its first stage ("1") --
    # works out of the box on a fresh portal. Point these at a specific
    # pipeline/stage (Settings > Objects > Tickets > Pipelines) once one
    # exists for this chatbot specifically.
    hubspot_ticket_pipeline_id: str = "0"
    hubspot_ticket_stage_id: str = "1"

    # --- HubSpot Citizen Corner (testimonials/feedback/photos, PRD FR-5.x) ---
    # Separate dedicated pipeline from the escalation one above -- keeps
    # citizen feedback visually/operationally apart from real service
    # tickets on the HubSpot board. None until the pipeline + its "Pending
    # Review" stage + the "files" scope + two custom ticket properties
    # (is_anonumous, photo_url) are created on the HubSpot side (planned:
    # in person at the CJS office) -- see
    # app.integrations.hubspot.create_citizen_submission, which raises a
    # clear HubSpotError rather than a confusing failure if this is still
    # unset when a submission comes in.
    hubspot_citizen_pipeline_id: str | None = None
    hubspot_citizen_stage_pending_id: str | None = None
    # Who can open an uploaded Citizen Corner photo before it is reviewed.
    # Photos are uploaded the moment they are submitted, long before
    # moderation, so the default keeps them out of search engines
    # (PUBLIC_NOT_INDEXABLE). PRIVATE is stricter (portal users only) but the
    # review team then needs HubSpot's signed links to view them.
    hubspot_photo_access: str = "PUBLIC_NOT_INDEXABLE"

    # --- Login gate (shared-password access control) ---
    # A single shared password protecting the public API, NOT a user-account
    # system. When LOGIN_PASSWORD is empty (the default), the gate is a no-op
    # so local dev keeps working with zero setup. Set it for the demo/customer
    # deployment. SESSION_SECRET signs the session cookie (itsdangerous); its
    # default is a throwaway so cookies survive a single process only --
    # BOTH of these MUST be set to real values before any real deployment.
    login_password: str | None = None
    session_secret: str = "dev-only-insecure-session-secret-change-me"
    session_cookie_name: str = "cjis_session"
    session_ttl_seconds: int = 7 * 24 * 3600  # 7 days
    # Set True once the site is served over HTTPS so the cookie is only ever
    # sent over TLS. Left False by default so the cookie also works over plain
    # HTTP during an initial IP-only demo; flip it on for the real domain.
    session_cookie_secure: bool = False

    # --- Speech-to-text (self-hosted whisper.cpp via pywhispercpp) ---
    # Model size downloaded/loaded on first use. "base.en" is a good
    # accuracy/speed balance on CPU; "tiny.en" is lighter for constrained
    # hosts. English-only ".en" variants are smaller and faster than the
    # multilingual ones.
    whisper_model_size: str = "base.en"
    # Per-IP limit for the transcription endpoint and the largest audio upload
    # accepted (bytes). A few seconds of webm/opus is tiny; 15MB is generous.
    rate_limit_transcribe: str = "20/minute"
    max_audio_bytes: int = 15 * 1024 * 1024

    # --- Containerized frontend ---
    # Host port the nginx-served frontend listens on (docker-compose).
    frontend_port: int = 80

    # --- API process behaviour ---
    # Load the embedding model at startup (in the background) so the first
    # visitor doesn't wait for it. Turn off in tests or constrained setups.
    warmup_on_startup: bool = True
    # Reject requests that announce a body larger than this (bytes). Citizen
    # Corner photos are capped at 8MB; everything else is tiny.
    max_request_body_bytes: int = 10 * 1024 * 1024

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


def production_config_problems(settings: Settings) -> list[str]:
    """Unsafe settings that must not reach a real deployment. Empty list when
    APP_ENV isn't "production" (local dev keeps its convenient defaults) or
    when everything is acceptable."""
    if settings.app_env.strip().lower() != "production":
        return []
    problems = []
    if settings.cors_allowed_origins.strip() == "*":
        problems.append(
            "CORS_ALLOWED_ORIGINS is '*': any website could call the write endpoints "
            "from a visitor's browser. Set it to the real site origin(s)."
        )
    if settings.postgres_password == "change_me_dev_only":
        problems.append("POSTGRES_PASSWORD is still the development default.")
    if not settings.qdrant_api_key and settings.qdrant_host not in ("localhost", "127.0.0.1", "qdrant"):
        problems.append("QDRANT_API_KEY is empty for a non-local Qdrant host.")
    if settings.qdrant_api_key and not settings.qdrant_https and settings.qdrant_host not in ("localhost", "127.0.0.1", "qdrant"):
        problems.append("QDRANT_HTTPS is false, so the Qdrant API key travels in cleartext.")
    if not settings.login_password:
        problems.append("LOGIN_PASSWORD is empty, so the public API has no access control.")
    if settings.session_secret == "dev-only-insecure-session-secret-change-me":
        problems.append("SESSION_SECRET is still the development default; set a long random value.")
    return problems
