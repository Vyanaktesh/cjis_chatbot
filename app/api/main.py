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

import uuid
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.logging_config import configure_logging
from app.db.connection import get_conn
from app.generation.service import answer_question
from app.retrieval.retriever import search as retrieval_search
from app.vectorstore.qdrant_store import get_qdrant_client

configure_logging(get_settings().log_level)

app = FastAPI(title="Consulate RAG Chatbot — Public API", version="0.9.0")

# Phase 8: the React widget runs on its own dev-server origin (and, in
# production, potentially a different origin than the API). Allowing any
# origin is fine here because this endpoint requires no auth and serves no
# PII back — it's the same publicly-answerable content /generate already
# serves. Tighten to a specific origin list before a real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
def search(
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
def generate(body: GenerateRequest):
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


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    history: Optional[list[ChatTurn]] = None


@app.post("/chat")
def chat(body: ChatRequest):
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
    result = answer_question(body.message, history=history)
    return {"session_id": session_id, **result}
