"""
Public retrieval + generation API.

Endpoints:
  GET  /health                              - liveness check (Postgres + Qdrant)
  GET  /search                               - Phase 6: hybrid retrieval, approved-only
  POST /generate                             - Phase 7: retrieval + grounded Qwen3 answer
  POST /chat                                 - Phase 8: frontend-facing chat endpoint
                                                (thin wrapper over /generate — see note below)

The admin-facing surface that used to live here — GET /sources, GET/POST
/review/*, and POST /upload — has moved to the separate kb_admin service
(port 8100, HTTP Basic auth required on every route). It reuses the exact
same app.review.* / app.ingestion.pipeline logic via kb_admin's sys.path
bridge rather than calling this service over HTTP. It was moved because
this app.py has no auth at all (see the CORS comment below) and those
routes let anyone who could reach this port write/approve/reject content —
fine for local dev, not something to expose alongside the public chat
endpoints. See consulate-kb-admin/kb_admin/api/documents.py and review.py.

Run with: uvicorn app.api.main:app --reload --port 8000

Phase 8 note on /chat and conversation memory: the server itself is fully
stateless between calls — there is no session store, no database table of
past turns, nothing to clean up or expire. The React widget is the sole
holder of the transcript, kept only in that browser tab's in-memory state
(assistant-ui's default runtime; not localStorage/sessionStorage/cookies —
a page refresh loses it entirely). On each call, the widget re-sends a
bounded sliding window of the last few turns as `history`, which
app/generation/prompt.py's `_trim_history` caps further (by turn count and
an approximate character budget) before it ever reaches the model — see
that module for why a bounded window, not full or zero history, was
chosen. This combination is what makes "session-scoped conversation, no
PII storage beyond the session" true by construction rather than by a
retention policy anyone has to trust.
"""

import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings, production_config_problems
from app.core.logging_config import configure_logging, get_logger
from app.db.connection import get_conn
from app.embedding.bge_m3 import BgeM3Embedder
from app.generation.service import GENERATION_UNAVAILABLE_ANSWER, answer_question
from app.integrations import hubspot
from app.integrations.images import ImageRejected, sanitize_image
from app.retrieval.retriever import search as retrieval_search
from app.vectorstore.qdrant_store import get_qdrant_client

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)

# Refuse to start a production deployment on unsafe defaults (wide-open CORS,
# the development database password, ...). Does nothing unless APP_ENV=production.
_config_problems = production_config_problems(settings)
if _config_problems:
    raise RuntimeError(
        "Refusing to start with unsafe production settings:\n- " + "\n- ".join(_config_problems)
    )


def _warm_up() -> None:
    try:
        BgeM3Embedder(batch_size=1)
        logger.info("embedding model loaded (warm-up)")
    except Exception:  # noqa: BLE001 -- the first request will simply load it instead
        logger.exception("embedding model warm-up failed")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Loading the embedding model takes ~15s the first time. Doing it in the
    # background at startup means the first visitor doesn't pay for it, while
    # the server still starts accepting requests immediately.
    if settings.warmup_on_startup:
        threading.Thread(target=_warm_up, name="embedding-warmup", daemon=True).start()
    yield


app = FastAPI(title="Consulate RAG Chatbot - Public API", version="0.9.0", lifespan=lifespan)

# Rate limiting, keyed on client IP (see Settings.rate_limit_* for why: an
# unbounded /generate or /chat could exhaust the Gemini quota or run up a
# bill in minutes, and unbounded /support/ticket or /citizen-corner/submit
# could spam real HubSpot tickets). Registered BEFORE CORSMiddleware below
# so CORS ends up the outermost layer -- verified this ordering is what
# makes a 429 response still carry CORS headers; without that, a rate
# limit hit shows the browser a generic network error instead of the
# actual "please slow down" message.
# The key is the client IP as the server sees it. Behind a reverse proxy / load
# balancer / CDN that is the PROXY's IP unless uvicorn is told which proxies to
# trust (--proxy-headers --forwarded-allow-ips=<proxy ip(s)>, never "*"), so
# every visitor would share one bucket. See "Running behind a reverse proxy" in
# LOCAL_SETUP.md. `rate_limit_storage_uri` points the counters at a shared store
# (e.g. redis://...) when running more than one worker; the default in-memory
# store is per-process, so N workers would allow N times the limit.
limiter = Limiter(key_func=get_remote_address, storage_uri=settings.rate_limit_storage_uri)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


class RequestGuardMiddleware:
    """Gives every request an id (returned as X-Request-ID and written to the
    log), logs one line per request with its status and duration, and rejects
    a request that announces a body bigger than `max_body_bytes` before the
    server reads it. The log line carries the path only: query strings can
    contain what a visitor typed (e.g. /search?q=...), which can be personal."""

    def __init__(self, app, max_body_bytes: int):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        status = {"code": 0}

        def log() -> None:
            logger.info(
                "request",
                extra={
                    "fields": {
                        "request_id": request_id,
                        "method": scope["method"],
                        "path": scope["path"],
                        "status": status["code"],
                        "duration_ms": round((time.perf_counter() - started) * 1000),
                    }
                },
            )

        length = dict(scope["headers"]).get(b"content-length", b"")
        if length.isdigit() and int(length) > self.max_body_bytes:
            status["code"] = 413
            response = JSONResponse(
                {"detail": "That upload is too large."},
                status_code=413,
                headers={"X-Request-ID": request_id},
            )
            await response(scope, receive, send)
            log()
            return

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                message = {**message, "headers": [*message.get("headers", []), (b"x-request-id", request_id.encode())]}
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            log()


app.add_middleware(RequestGuardMiddleware, max_body_bytes=settings.max_request_body_bytes)

# Phase 8: the React widget runs on its own dev-server origin (and, in
# production, potentially a different origin than the API). CORS_ALLOWED_ORIGINS
# defaults to "*" so local dev keeps working out of the box -- MUST be set to
# the real consulate site's origin(s) before a real deployment, since with
# "*" any website can call the write endpoints (/support/ticket,
# /citizen-corner/submit) from a visitor's own browser.
_cors_origins = (
    ["*"]
    if settings.cors_allowed_origins.strip() == "*"
    else [o.strip() for o in settings.cors_allowed_origins.split(",") if o.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    """200 when Postgres and Qdrant both answer, otherwise 503 naming which
    one(s) failed (a clean JSON reply, not a bare 500) so a load balancer or
    uptime monitor can act on it."""
    failing = []
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
    except Exception:  # noqa: BLE001 -- reported below, never raised to the caller
        logger.exception("health check: postgres unavailable")
        failing.append("postgres")
    try:
        get_qdrant_client().get_collections()
    except Exception:  # noqa: BLE001
        logger.exception("health check: qdrant unavailable")
        failing.append("qdrant")
    if failing:
        return JSONResponse(status_code=503, content={"status": "unavailable", "failing": failing})
    return {"status": "ok", "embedding_model_loaded": BgeM3Embedder._model is not None}


@app.get("/search")
@limiter.limit(settings.rate_limit_search)
def search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=500),
    limit: int = 8,
    service_category: Optional[str] = Query(None, max_length=100),
    canonical: Optional[bool] = None,
    source_id: Optional[str] = Query(None, max_length=100),
    jurisdiction: Optional[str] = Query(None, max_length=100),
    applicant_variant: Optional[str] = Query(None, max_length=100),
):
    """
    Phase 6: hybrid dense+sparse retrieval, hard-restricted to
    review_status='approved' regardless of any filter passed here — see
    app/retrieval/retriever.py. Nothing else is servable to an end user.
    """
    limit = max(1, min(limit, 50))
    results = retrieval_search(
        q,
        limit=limit,
        service_category=service_category,
        canonical=canonical,
        source_id=source_id,
        jurisdiction=jurisdiction,
        applicant_variant=applicant_variant,
    )
    return {"query": q, "count": len(results), "results": results}


# Limits on what a visitor (or a script pretending to be one) can send. Without
# them a single request could carry a 500,000-character message or a
# thousands-of-turns history -- cost on the paid model API and memory here.
MAX_MESSAGE_CHARS = 2000
MAX_HISTORY_TURNS = 20
MAX_HISTORY_CONTENT_CHARS = 4000


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def _clip(cls, v: str) -> str:
        # Earlier ANSWERS are re-sent as history and can be long; clip rather
        # than reject so a long answer never turns the next question into an
        # error. prompt.py trims history much further before it reaches the model.
        return v[:MAX_HISTORY_CONTENT_CHARS]


def _not_blank(v: str) -> str:
    v = v.strip()
    if not v:
        raise ValueError("must not be blank")
    return v


class GenerateRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    limit: int = 6
    service_category: Optional[str] = Field(None, max_length=100)
    canonical: Optional[bool] = None
    source_id: Optional[str] = Field(None, max_length=100)
    jurisdiction: Optional[str] = Field(None, max_length=100)
    applicant_variant: Optional[str] = Field(None, max_length=100)
    history: Optional[list[ChatTurn]] = Field(None, max_length=MAX_HISTORY_TURNS)

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, v: str) -> str:
        return _not_blank(v)


@app.post("/generate")
@limiter.limit(settings.rate_limit_generate)
def generate(request: Request, body: GenerateRequest):
    """
    Phase 7: retrieval (approved-only) + a grounded Qwen3 answer with
    bracket citations back to the retrieved chunks. If retrieval finds
    nothing, no model call is made at all — see
    app/generation/service.py's NO_CONTEXT_ANSWER.

    `history` (Phase 8) is optional: a bounded sliding window of prior
    {role, content} turns from this same browser session, used to make
    follow-up questions coherent — see app/generation/prompt.py's
    _trim_history and app/generation/service.py's _retrieval_query.
    """
    limit = max(1, min(body.limit, 20))
    history = [h.model_dump() for h in body.history] if body.history else None
    try:
        return answer_question(
            body.query,
            limit=limit,
            service_category=body.service_category,
            canonical=body.canonical,
            source_id=body.source_id,
            jurisdiction=body.jurisdiction,
            applicant_variant=body.applicant_variant,
            history=history,
        )
    except Exception as exc:  # noqa: BLE001 -- last-resort guard, logged below
        logger.exception("generate pipeline failed")
        # The exception text stays in the log. It used to be returned to the
        # browser, which leaked internals (hostnames, account names).
        return {
            "query": body.query,
            "answer": GENERATION_UNAVAILABLE_ANSWER,
            "grounded": False,
            "citations": [],
            "retrieved_count": 0,
            "generation_error": "pipeline_unavailable",
        }


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    session_id: Optional[str] = Field(None, max_length=64)
    history: Optional[list[ChatTurn]] = Field(None, max_length=MAX_HISTORY_TURNS)

    @field_validator("message")
    @classmethod
    def _message_not_blank(cls, v: str) -> str:
        return _not_blank(v)


@app.post("/chat")
@limiter.limit(settings.rate_limit_generate)
def chat(request: Request, body: ChatRequest):
    """
    Phase 8: frontend-facing chat endpoint. Thin wrapper over the same
    answer_question() pipeline as /generate, shaped for the React widget:
    - `session_id` is generated here if the client didn't send one yet
      (a random, non-identifying UUID — just enough to correlate a
      browser tab's messages if we ever want to, nothing PII).
    - No conversation state is kept server-side between calls. The
      client is the sole holder of the transcript (in the browser tab's
      memory only — lost on refresh) and re-sends a bounded window of
      recent turns as `history` on every call; see this module's
      docstring and app/generation/prompt.py's _trim_history for how that
      window is capped before it reaches the model.
    """
    session_id = body.session_id or str(uuid.uuid4())
    history = [h.model_dump() for h in body.history] if body.history else None
    result = _answer_or_degrade(body.message, history)
    return {"session_id": session_id, **result}


def _answer_or_degrade(query: str, history):
    """Runs the RAG pipeline, but never lets an infrastructure failure
    (Qdrant/Postgres/embedding-model down) surface as a raw 500 with a
    traceback to the browser. answer_question already degrades gracefully on
    a *generation* backend failure; this covers the retrieval/embedding
    steps that run before generation. `generation_error` is set so the
    widget treats it as a transient outage ("try again") rather than a
    genuine can't-answer that offers human escalation."""
    try:
        return answer_question(query, history=history)
    except Exception as exc:  # noqa: BLE001 -- last-resort guard, logged below
        logger.exception("chat/generate pipeline failed")
        return {
            "query": query,
            "answer": GENERATION_UNAVAILABLE_ANSWER,
            "grounded": False,
            "citations": [],
            "retrieved_count": 0,
            "generation_error": "pipeline_unavailable",
        }


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class SupportTicketRequest(BaseModel):
    name: str = Field(max_length=200)
    email: str = Field(max_length=320)
    city: str = Field(max_length=100)
    state: str = Field(max_length=100)
    message: str = Field(max_length=5000)
    session_id: Optional[str] = Field(None, max_length=64)

    @field_validator("name", "city", "state", "message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v

    @field_validator("email")
    @classmethod
    def _valid_email(cls, v: str) -> str:
        v = v.strip()
        if not _EMAIL_RE.match(v):
            raise ValueError("not a valid email address")
        return v


class SupportTicketResponse(BaseModel):
    ticket_id: str


@app.post("/support/ticket", response_model=SupportTicketResponse)
@limiter.limit(settings.rate_limit_submit)
def create_support_ticket(request: Request, body: SupportTicketRequest):
    """
    Escalation path for when the chatbot can't ground an answer (see
    app/generation/service.py's OUT_OF_SCOPE_ANSWER / NO_CONTEXT_ANSWER /
    GENERATION_UNAVAILABLE_ANSWER, all of which set grounded=False -- the
    widget offers this form specifically in that case). Creates/updates a
    HubSpot Contact by email and a Ticket associated to it -- see
    app/integrations/hubspot.py.
    """
    try:
        ticket_id = hubspot.create_support_ticket(
            name=body.name,
            email=body.email,
            city=body.city,
            state=body.state,
            message=body.message,
        )
    except hubspot.HubSpotError as exc:
        logger.warning(f"HubSpot ticket creation failed (session={body.session_id}): {exc}")
        raise HTTPException(
            status_code=502,
            detail="Could not submit your query right now. Please try again in a moment, or contact the consulate directly.",
        ) from exc
    return {"ticket_id": ticket_id}


# Citizen Corner (PRD FR-5.1-5.8): testimonials/feedback/photos, separate
# from the support-escalation ticket above. Placeholder limits below --
# not yet informed by "D8" (the actual consent/acceptance rules doc,
# not available at the time this was written); tighten/adjust once it is.
_MAX_PHOTO_BYTES = 8 * 1024 * 1024  # 8MB
_ALLOWED_PHOTO_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


class CitizenSubmissionResponse(BaseModel):
    ticket_id: str


@app.post("/citizen-corner/submit", response_model=CitizenSubmissionResponse)
@limiter.limit(settings.rate_limit_submit)
async def submit_citizen_corner(
    request: Request,
    name: str = Form(..., max_length=200),
    email: str = Form(..., max_length=320),
    title: str = Form(..., max_length=200),
    content: str = Form(..., max_length=5000),
    submission_type: str = Form(..., max_length=30),
    anonymous: bool = Form(False),
    photo: Optional[UploadFile] = File(None),
):
    """
    A citizen may share a testimonial, feedback, or photo at any point,
    independent of the Q&A/escalation flow above (FR-5.1). Nothing here
    is published directly -- see app/integrations/hubspot.py's
    create_citizen_submission for how this lands in HubSpot's moderation
    queue (FR-5.3), and that module's HubSpotError message for what still
    needs setting up there before this endpoint works end-to-end.
    """
    name = name.strip()
    email = email.strip()
    title = title.strip()
    content = content.strip()
    if not name or not email or not title or not content:
        raise HTTPException(status_code=422, detail="name, email, title, and content are all required.")
    if not _EMAIL_RE.match(email):
        raise HTTPException(status_code=422, detail="not a valid email address")
    if submission_type not in hubspot.CITIZEN_SUBMISSION_TYPES:
        allowed = ", ".join(hubspot.CITIZEN_SUBMISSION_TYPES)
        raise HTTPException(
            status_code=422,
            detail=f"submission_type must be one of: {allowed}.",
        )

    photo_bytes = None
    photo_filename = None
    photo_content_type = None
    if photo is not None and photo.filename:
        if photo.content_type not in _ALLOWED_PHOTO_CONTENT_TYPES:
            raise HTTPException(status_code=422, detail="Photo must be a JPEG, PNG, WEBP, or GIF image.")
        # Reject oversized uploads BEFORE buffering the whole file into
        # memory. Starlette populates UploadFile.size from the multipart
        # part headers when present; without this guard a malicious large
        # upload would be fully read into RAM only to be rejected below.
        if photo.size is not None and photo.size > _MAX_PHOTO_BYTES:
            raise HTTPException(status_code=422, detail="Photo must be under 8MB.")
        raw_photo = await photo.read()
        if len(raw_photo) > _MAX_PHOTO_BYTES:  # fallback when .size was unset
            raise HTTPException(status_code=422, detail="Photo must be under 8MB.")
        # The declared content type and filename come from the visitor's
        # browser and prove nothing. Decode the bytes as a real image,
        # strip location/camera metadata, and use a generated filename.
        try:
            photo_bytes, photo_content_type, photo_filename = await run_in_threadpool(sanitize_image, raw_photo)
        except ImageRejected:
            raise HTTPException(
                status_code=422,
                detail="That file doesn't look like a valid image. Please use a JPEG, PNG, WEBP, or GIF photo.",
            ) from None

    try:
        # hubspot.* makes blocking HTTP calls (up to three in a row). This
        # handler is `async` only because of `await photo.read()` above, so
        # the blocking call must be pushed to a worker thread -- run directly
        # it freezes the event loop and stalls every other request (chat,
        # health checks) for the whole HubSpot round trip.
        ticket_id = await run_in_threadpool(
            hubspot.create_citizen_submission,
            name=name,
            email=email,
            title=title,
            content=content,
            anonymous=anonymous,
            submission_type=submission_type,
            photo_bytes=photo_bytes,
            photo_filename=photo_filename,
            photo_content_type=photo_content_type,
        )
    except hubspot.HubSpotError as exc:
        logger.warning(f"Citizen Corner submission failed: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Could not submit your feedback right now. Please try again in a moment.",
        ) from exc
    return {"ticket_id": ticket_id}
