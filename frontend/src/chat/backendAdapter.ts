import type {
  ChatModelAdapter,
  ThreadAssistantMessagePart,
  ThreadMessage,
} from "@assistant-ui/react";
import { playReceive } from "./sound";

/**
 * Phase 8: bridges assistant-ui's runtime to our own FastAPI backend's
 * POST /chat endpoint (app/api/main.py). This is a plain request/response
 * call (not a streaming AsyncGenerator) because the underlying generation
 * step runs a self-hosted Qwen3 model on CPU only — there is no token
 * stream to relay, just one real answer that can take 30s-150s+ to arrive.
 * assistant-ui shows its "running" state for the whole span regardless;
 * see ChatWidget.tsx for the "still thinking" indicator built on top of
 * that, which matters a lot given how long a single turn can take here.
 */

// In production the frontend is served from the same origin as the API
// (nginx reverse-proxies /chat, /auth, ... to the backend -- see
// frontend/nginx.conf), so VITE_API_BASE_URL is built as "" and every call
// is a relative path. In local dev it defaults to the backend's dev port.
export const API_BASE =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "http://127.0.0.1:8000";

// Ticket/feedback submits are quick server-side (bounded by HubSpot's own
// timeout), so cap the client wait too -- otherwise a hung connection would
// leave the form spinner stuck forever with no error.
const SUBMIT_TIMEOUT_MS = 30000;

// A chat answer normally arrives within a few seconds, but retrieval plus
// generation can legitimately take longer under load. Past this, give up and
// tell the user instead of spinning forever (the request previously had no
// timeout at all, so a stalled server meant an endless "Looking into your
// question..." with nothing the user could do).
const CHAT_TIMEOUT_MS = 90000;

/** An error whose text is meant to be shown to the user as-is. assistant-ui
 * renders `String(error)`, which for a plain Error would add an "Error: "
 * prefix, so toString() returns just the message. */
class ChatError extends Error {
  override toString(): string {
    return this.message;
  }
}

const EMPTY_ANSWER_FALLBACK =
  "Sorry, I couldn't put an answer together for that. Could you try rephrasing your question?";

function messageForStatus(status: number): string {
  if (status === 429) {
    return "You're sending messages too quickly. Please wait a minute and try again.";
  }
  return `The consulate assistant couldn't be reached (HTTP ${status}). Please try again in a moment.`;
}

/** fetch() with an AbortController timeout, translating an abort into a
 * clear, user-displayable message. */
async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    // credentials: "include" so the login session cookie (see
    // app/api/auth.py) rides along on cross-origin dev requests too.
    return await fetch(url, {
      ...init,
      credentials: "include",
      signal: controller.signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error(
        "The request timed out. Please check your connection and try again.",
      );
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

// Generated once per page load, kept only in this module's memory — never
// written to localStorage/sessionStorage/cookies. Refreshing or closing the
// tab loses it, which is the whole point: "session-scoped conversation, no
// PII storage beyond the session" is true here by construction (there is
// nothing to expire or clean up) rather than by a retention policy anyone
// has to trust.
let sessionId: string | null = null;

// Called when the widget panel closes (see ChatWidget's onClosed) so the
// *next* open starts a genuinely new session id, matching the reset of the
// visible thread itself (see App.tsx's resetKey). The backend is fully
// stateless (app/api/main.py) and never looks this id up server-side --
// history is always sent explicitly in the request body -- so this has no
// functional effect on what the backend does with it, only on what id
// shows up in its logs for the next conversation.
export function resetSession() {
  sessionId = null;
}

// Mirrors app/generation/prompt.py's HISTORY_MAX_TURNS — no point sending
// the backend more turns than it will ever actually use.
const HISTORY_MAX_TURNS = 5;

function extractText(message: ThreadMessage): string {
  return message.content
    .map((part) => (part.type === "text" ? part.text : ""))
    .join("")
    .trim();
}

type BackendCitation = {
  index: number;
  chunk_id: string;
  source_url: string;
  service_category: string;
  canonical: boolean;
};

export type ChatResponse = {
  session_id: string;
  query: string;
  answer: string;
  grounded: boolean;
  citations: BackendCitation[];
  retrieved_count: number;
  // Present ONLY when the generation backend itself failed (service down),
  // set by app/generation/service.py. Lets the widget tell a technical
  // outage ("try again") apart from a genuine can't-answer (offer to
  // escalate to the consulate) -- both are un-grounded otherwise.
  generation_error?: string;
};

/**
 * Factory rather than a plain object so App.tsx can pass an `onResult`
 * callback: the widget needs `grounded` from every reply (to decide when
 * to offer a HubSpot support-ticket escalation -- see ChatWidget.tsx's
 * ticket flow) but assistant-ui's ChatModelAdapter only returns message
 * `content` parts into the thread, with no side-channel for the raw
 * backend response. This is that side-channel.
 */
export function createBackendAdapter(
  onResult?: (data: ChatResponse) => void,
): ChatModelAdapter {
  return {
    async run({ messages, abortSignal }) {
      const current = messages[messages.length - 1];
      const query = extractText(current);

      const history = messages
        .slice(0, -1)
        .filter((m) => m.role === "user" || m.role === "assistant")
        .slice(-(HISTORY_MAX_TURNS * 2))
        .map((m) => ({
          role: m.role as "user" | "assistant",
          content: extractText(m),
        }));

      // One controller serves both ways a request can be abandoned: the user
      // pressing Cancel (abortSignal from assistant-ui) and our own timeout.
      const controller = new AbortController();
      let timedOut = false;
      const timer = setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, CHAT_TIMEOUT_MS);
      const onUserAbort = () => controller.abort();
      if (abortSignal.aborted) controller.abort();
      else abortSignal.addEventListener("abort", onUserAbort);

      let data: ChatResponse;
      try {
        let res: Response;
        try {
          res = await fetch(`${API_BASE}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            credentials: "include",
            body: JSON.stringify({
              message: query,
              session_id: sessionId ?? undefined,
              history,
            }),
            signal: controller.signal,
          });
        } catch (err) {
          if (timedOut) {
            throw new ChatError(
              "This is taking longer than expected. Please try again in a moment.",
            );
          }
          if (abortSignal.aborted) throw err; // the user cancelled: not an error to display
          throw new ChatError(
            "Couldn't reach the assistant. Please check your connection and try again.",
          );
        }

        if (!res.ok) throw new ChatError(messageForStatus(res.status));

        try {
          data = (await res.json()) as ChatResponse;
        } catch {
          throw new ChatError(
            "The assistant sent a response that couldn't be read. Please try again.",
          );
        }
      } finally {
        clearTimeout(timer);
        abortSignal.removeEventListener("abort", onUserAbort);
      }

      sessionId = data.session_id ?? sessionId;
      onResult?.(data);

      // An empty/blank answer (e.g. the model returned nothing) would render
      // as an empty bubble, so show a fallback instead.
      const answer = data.answer?.trim() ? data.answer : EMPTY_ANSWER_FALLBACK;

      const content: ThreadAssistantMessagePart[] = [
        { type: "text", text: answer },
        ...(data.citations ?? []).map(
          (c): ThreadAssistantMessagePart => ({
            type: "source" as const,
            sourceType: "url" as const,
            id: c.chunk_id,
            url: c.source_url,
            title: `[${c.index}] ${(c.service_category ?? "source").toUpperCase()} source`,
          }),
        ),
      ];

      playReceive();
      return { content };
    },
  };
}

/* ---- Login gate (shared password; see app/api/auth.py) ----------------- */

export type AuthStatus = {
  login_required: boolean;
  authenticated: boolean;
};

/** GET /auth/me -- whether a password is configured at all, and whether this
 * browser is currently signed in. On any network error, assume the gate is
 * off so a backend hiccup can't lock the user out of a demo. */
export async function fetchAuthStatus(): Promise<AuthStatus> {
  try {
    const res = await fetchWithTimeout(
      `${API_BASE}/auth/me`,
      { method: "GET" },
      SUBMIT_TIMEOUT_MS,
    );
    if (!res.ok) return { login_required: false, authenticated: true };
    return (await res.json()) as AuthStatus;
  } catch {
    return { login_required: false, authenticated: true };
  }
}

/** POST /auth/login. Resolves true on success, false on a wrong password;
 * throws only on a network/server failure the caller should surface. */
export async function login(password: string): Promise<boolean> {
  const res = await fetchWithTimeout(
    `${API_BASE}/auth/login`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    },
    SUBMIT_TIMEOUT_MS,
  );
  if (res.status === 401) return false;
  if (!res.ok) {
    throw new Error(
      `Could not sign in (HTTP ${res.status}). Please try again in a moment.`,
    );
  }
  return true;
}

/** POSTs a recorded audio blob to /transcribe (self-hosted whisper.cpp) and
 * returns the transcribed text. Throws a user-displayable message on failure. */
export async function transcribeAudio(blob: Blob): Promise<string> {
  const form = new FormData();
  // Filename hints ffmpeg's container detection; the server re-detects anyway.
  form.set("audio", blob, "recording.webm");
  const res = await fetchWithTimeout(
    `${API_BASE}/transcribe`,
    { method: "POST", body: form },
    SUBMIT_TIMEOUT_MS,
  );
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(
      body?.detail ??
        "Could not transcribe that recording. Please try again or type your message.",
    );
  }
  const data = (await res.json()) as { text: string };
  return data.text ?? "";
}

export type SupportTicketInput = {
  name: string;
  email: string;
  city: string;
  state: string;
  message: string;
};

/** POSTs to app/api/main.py's /support/ticket -- see that endpoint's
 * docstring and app/integrations/hubspot.py for what happens server-side.
 * Throws with a user-displayable message on any failure. */
export async function submitSupportTicket(
  input: SupportTicketInput,
): Promise<{ ticketId: string }> {
  const res = await fetchWithTimeout(
    `${API_BASE}/support/ticket`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...input, session_id: sessionId ?? undefined }),
    },
    SUBMIT_TIMEOUT_MS,
  );

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(
      body?.detail ??
        "Could not submit your query right now. Please try again in a moment.",
    );
  }

  const data = (await res.json()) as { ticket_id: string };
  return { ticketId: data.ticket_id };
}

export type CitizenSubmissionType =
  | "testimonial"
  | "experience"
  | "feedback"
  | "photo";

export type CitizenSubmissionInput = {
  name: string;
  email: string;
  title: string;
  content: string;
  submissionType: CitizenSubmissionType;
  anonymous: boolean;
  photo?: File;
};

/** POSTs to app/api/main.py's /citizen-corner/submit -- multipart/form-data
 * (not JSON) since an optional photo file rides along. See that
 * endpoint's docstring and app/integrations/hubspot.py's
 * create_citizen_submission for what happens server-side. Throws with a
 * user-displayable message on any failure. */
export async function submitCitizenCorner(
  input: CitizenSubmissionInput,
): Promise<{ ticketId: string }> {
  const form = new FormData();
  form.set("name", input.name);
  form.set("email", input.email);
  form.set("title", input.title);
  form.set("content", input.content);
  form.set("submission_type", input.submissionType);
  form.set("anonymous", String(input.anonymous));
  if (input.photo) form.set("photo", input.photo);

  const res = await fetchWithTimeout(
    `${API_BASE}/citizen-corner/submit`,
    { method: "POST", body: form },
    SUBMIT_TIMEOUT_MS,
  );

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(
      body?.detail ??
        "Could not submit your feedback right now. Please try again in a moment.",
    );
  }

  const data = (await res.json()) as { ticket_id: string };
  return { ticketId: data.ticket_id };
}
