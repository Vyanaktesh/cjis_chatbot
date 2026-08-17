import type {
  ChatModelAdapter,
  ThreadAssistantMessagePart,
  ThreadMessage,
} from "@assistant-ui/react";

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

const API_BASE =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "http://127.0.0.1:8000";

// Generated once per page load, kept only in this module's memory — never
// written to localStorage/sessionStorage/cookies. Refreshing or closing the
// tab loses it, which is the whole point: "session-scoped conversation, no
// PII storage beyond the session" is true here by construction (there is
// nothing to expire or clean up) rather than by a retention policy anyone
// has to trust.
let sessionId: string | null = null;

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

type ChatResponse = {
  session_id: string;
  answer: string;
  grounded: boolean;
  citations: BackendCitation[];
  retrieved_count: number;
};

export const backendAdapter: ChatModelAdapter = {
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

    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: query,
        session_id: sessionId ?? undefined,
        history,
      }),
      signal: abortSignal,
    });

    if (!res.ok) {
      throw new Error(
        `The consulate assistant couldn't be reached (HTTP ${res.status}). Please try again in a moment.`,
      );
    }

    const data = (await res.json()) as ChatResponse;
    sessionId = data.session_id ?? sessionId;

    const content: ThreadAssistantMessagePart[] = [
      { type: "text", text: data.answer },
      ...data.citations.map(
        (c): ThreadAssistantMessagePart => ({
          type: "source" as const,
          sourceType: "url" as const,
          id: c.chunk_id,
          url: c.source_url,
          title: `[${c.index}] ${c.service_category.toUpperCase()} source`,
        }),
      ),
    ];

    return { content };
  },
};
