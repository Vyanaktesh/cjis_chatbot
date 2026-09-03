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
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.core.config import get_settings
from app.core.logging_config import configure_logging, get_logger
from app.db.connection import get_conn
from app.generation.service import GENERATION_UNAVAILABLE_ANSWER, answer_question
from app.integrations import hubspot
from app.retrieval.retriever import search as retrieval_search
from app.vectorstore.qdrant_store import get_qdrant_client

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)

app = FastAPI(title="Consulate RAG Chatbot — Public API", version="0.9.0")

# Rate limiting, keyed on client IP (see Settings.rate_limit_* for why: an
# unbounded /generate or /chat could exhaust the Gemini quota or run up a
# bill in minutes, and unbounded /support/ticket or /citizen-corner/submit
# could spam real HubSpot tickets). Registered BEFORE CORSMiddleware below
# so CORS ends up the outermost layer -- verified this ordering is what
# makes a 429 response still carry CORS headers; without that, a rate
# limit hit shows the browser a generic network error instead of the
# actual "please slow down" message.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

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
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
    client = get_qdrant_client()
    client.get_collections()
    return {"status": "ok"}


@app.get("/search")
@limiter.limit(settings.rate_limit_search)
def search(
    request: Request,
    q: str,
    limit: int = 8,
    service_category: Optional[str] = None,
    canonical: Optional[bool] = None,
    source_id: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    applicant_variant: Optional[str] = None,
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


class ChatTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class GenerateRequest(BaseModel):
    query: str
    limit: int = 6
    service_category: Optional[str] = None
    canonical: Optional[bool] = None
    source_id: Optional[str] = None
    jurisdiction: Optional[str] = None
    applicant_variant: Optional[str] = None
    history: Optional[list[ChatTurn]] = None


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
        return {
            "query": body.query,
            "answer": GENERATION_UNAVAILABLE_ANSWER,
            "grounded": False,
            "citations": [],
            "retrieved_count": 0,
            "generation_error": str(exc),
        }


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    history: Optional[list[ChatTurn]] = None


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
            "generation_error": str(exc),
        }


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class SupportTicketRequest(BaseModel):
    name: str
    email: str
    city: str
    state: str
    message: str
    session_id: Optional[str] = None

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
    name: str = Form(...),
    email: str = Form(...),
    title: str = Form(...),
    content: str = Form(...),
    submission_type: str = Form(...),
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
        photo_bytes = await photo.read()
        if len(photo_bytes) > _MAX_PHOTO_BYTES:  # fallback when .size was unset
            raise HTTPException(status_code=422, detail="Photo must be under 8MB.")
        photo_filename = photo.filename
        photo_content_type = photo.content_type

    try:
        ticket_id = hubspot.create_citizen_submission(
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
