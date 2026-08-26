import { useEffect, useState, type ReactNode } from "react";
import {
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
} from "@assistant-ui/react";
import { ExternalLink, Maximize2, Minimize2, Send, Smile, X } from "lucide-react";
import { playSend } from "./sound";

const BOT_NAME = "DOST";
const BOT_TAGLINE = "Kahin Bhi, Kabhi Bhi";
const GREETING =
  "Hello, I am DOST! DOST HAI NA!! (Kahin Bhi, Kabhi Bhi) Your friend to help you. Got any questions?";

function BotAvatar({ size = 40, withStatus = false }: { size?: number; withStatus?: boolean }) {
  const iconSize = Math.round(size * 0.55);
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <div
        className="flex h-full w-full items-center justify-center rounded-full bg-gradient-to-br from-[var(--brand-blue)] to-[var(--brand-blue-dark)] shadow-sm ring-2 ring-white/70"
      >
        <Smile style={{ width: iconSize, height: iconSize }} className="text-white" />
      </div>
      {withStatus && (
        <span className="absolute -bottom-0.5 -right-0.5 flex h-3 w-3 items-center justify-center rounded-full bg-white">
          <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
        </span>
      )}
    </div>
  );
}

const CATEGORY_STYLES: Record<string, string> = {
  PASSPORT: "bg-[var(--brand-blue-light)] text-[var(--brand-blue)]",
  OCI: "bg-emerald-50 text-emerald-700",
  VISA: "bg-amber-50 text-amber-700",
};
const DEFAULT_CATEGORY_STYLE = "bg-slate-100 text-slate-600";

function SourceChip({ url, title }: { url?: string; title?: string }) {
  if (!url) return null;

  const match = title?.match(/^\[(\d+)\]\s*(\S+)/);
  const index = match?.[1];
  const category = match?.[2];

  let host = url;
  try {
    host = new URL(url).hostname.replace(/^www\./, "");
  } catch {
    // keep raw url as fallback label
  }

  return (
    <a
      href={url}
      target="_blank"
      rel="noreferrer"
      className="group flex max-w-full items-center gap-2 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-600 shadow-sm transition hover:-translate-y-px hover:border-[var(--brand-blue)]/40 hover:shadow"
      title={`${category ? `${category} · ` : ""}${host}`}
    >
      {index && (
        <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-[var(--brand-blue)] text-[10px] font-semibold text-white">
          {index}
        </span>
      )}
      {category && (
        <span
          className={`shrink-0 rounded-full px-1.5 py-0.5 text-[10px] font-semibold tracking-wide ${
            CATEGORY_STYLES[category] ?? DEFAULT_CATEGORY_STYLE
          }`}
        >
          {category}
        </span>
      )}
      <span className="truncate font-medium text-slate-700 group-hover:text-[var(--brand-blue)]">
        {host}
      </span>
      <ExternalLink className="ml-auto h-3 w-3 shrink-0 text-slate-400 group-hover:text-[var(--brand-blue)]" />
    </a>
  );
}

const INLINE_CITATION_RE = /\[(\d+)\]/g;

/**
 * The model is instructed to follow every factual claim with bracket
 * citations like "[5][6]" (see app/generation/prompt.py's SYSTEM_PROMPT) --
 * necessary for grounding, but raw "[5][6]" sitting in running prose reads
 * as clutter/typos to an end user rather than an intentional citation.
 * Rendered as small superscript badges instead, in the same brand-blue as
 * the numbered badges in SourceChip below, so the two visually read as one
 * citation system rather than two different things.
 */
function renderWithCitations(text: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  let lastIndex = 0;
  let key = 0;

  for (const match of text.matchAll(INLINE_CITATION_RE)) {
    const index = match.index ?? 0;
    if (index > lastIndex) nodes.push(text.slice(lastIndex, index));
    nodes.push(
      <sup
        key={`cite-${key++}`}
        className="mx-px inline-flex h-[14px] min-w-[14px] items-center justify-center rounded-full bg-[var(--brand-blue-light)] px-1 text-[9px] font-semibold leading-none text-[var(--brand-blue)]"
      >
        {match[1]}
      </sup>,
    );
    lastIndex = index + match[0].length;
  }
  if (lastIndex < text.length) nodes.push(text.slice(lastIndex));
  return nodes;
}

function UserMessage() {
  return (
    <MessagePrimitive.Root className="flex justify-end px-4 py-1.5">
      <div className="max-w-[85%] rounded-2xl rounded-br-md bg-[var(--brand-blue)] px-4 py-2.5 text-[14.5px] leading-relaxed text-white shadow-sm">
        <MessagePrimitive.Content>
          {({ part }) =>
            part.type === "text" ? (
              <span className="whitespace-pre-wrap">{part.text}</span>
            ) : null
          }
        </MessagePrimitive.Content>
      </div>
    </MessagePrimitive.Root>
  );
}

function AssistantMessage() {
  return (
    <MessagePrimitive.Root className="flex justify-start gap-2 px-4 py-1.5">
      {/* hasContent: false while a run is in flight and no parts have
          arrived yet — skip rendering an empty bubble; ThinkingIndicator
          (driven off ThreadPrimitive.If running) covers that state instead. */}
      <MessagePrimitive.If hasContent>
        <BotAvatar size={26} />
        <div className="max-w-[85%] rounded-2xl rounded-bl-md border border-slate-100 bg-white px-4 py-2.5 text-[14.5px] leading-relaxed text-slate-800 shadow-sm">
          <MessagePrimitive.Content>
            {({ part }) => {
              if (part.type === "text") {
                return (
                  <span className="whitespace-pre-wrap">
                    {renderWithCitations(part.text)}
                  </span>
                );
              }
              if (part.type === "source") {
                return (
                  <div className="citation-row">
                    <SourceChip url={part.url} title={part.title} />
                  </div>
                );
              }
              return null;
            }}
          </MessagePrimitive.Content>
        </div>
      </MessagePrimitive.If>
    </MessagePrimitive.Root>
  );
}

function ThreadWelcome() {
  return (
    <ThreadPrimitive.Empty>
      <div className="flex items-start gap-2 px-4 py-2">
        <BotAvatar size={26} />
        <div className="max-w-[85%] rounded-2xl rounded-bl-md border border-slate-100 bg-white px-4 py-2.5 text-[14.5px] leading-relaxed text-slate-800 shadow-sm">
          {GREETING}
        </div>
      </div>
    </ThreadPrimitive.Empty>
  );
}

function ElapsedSeconds() {
  // Mounted only while a run is in flight (see ThinkingIndicator below) --
  // ThreadPrimitive.If unmounts this on completion, so the counter always
  // starts fresh at 0 for each new question with no manual reset needed.
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, []);
  return <span>{seconds}s</span>;
}

function ThinkingIndicator() {
  // Deliberately backend-agnostic: no hardcoded "self-hosted, up to two
  // minutes" claim, since that was true for the original Qwen/llama.cpp
  // backend but goes stale (and reads as broken) whenever a faster backend
  // like Gemini is configured instead. A live elapsed-time counter stays
  // accurate regardless of which generation backend is active.
  return (
    <ThreadPrimitive.If running>
      <div className="flex items-center gap-2 px-4 py-1.5">
        <BotAvatar size={26} />
        <div className="flex items-center gap-2 rounded-2xl rounded-bl-md border border-slate-100 bg-white px-4 py-2.5 text-[13px] text-slate-500 shadow-sm">
          <span className="flex gap-1">
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-300 [animation-delay:-0.3s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-300 [animation-delay:-0.15s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-300" />
          </span>
          <span>
            Looking into your question&hellip; <ElapsedSeconds />
          </span>
        </div>
      </div>
    </ThreadPrimitive.If>
  );
}

function Composer() {
  return (
    <ComposerPrimitive.Root className="flex items-end gap-2 border-t border-slate-100 bg-white p-3">
      <ComposerPrimitive.Input
        placeholder="Ask me anything..."
        rows={1}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) playSend();
        }}
        className="consulate-scroll max-h-28 flex-1 resize-none rounded-2xl border border-slate-200 bg-slate-50 px-4 py-2.5 text-[14.5px] text-slate-800 outline-none placeholder:text-slate-400 focus:border-[var(--brand-blue)] focus:ring-1 focus:ring-[var(--brand-blue)]"
      />
      <ComposerPrimitive.Send asChild>
        <button
          type="submit"
          aria-label="Send message"
          onClick={() => playSend()}
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-[var(--brand-blue)] text-white transition hover:bg-[var(--brand-blue-dark)] disabled:opacity-40"
        >
          <Send className="h-4 w-4" />
        </button>
      </ComposerPrimitive.Send>
    </ComposerPrimitive.Root>
  );
}

function ChatPanel({ onClose }: { onClose: () => void }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className={`consulate-panel-enter flex max-w-[92vw] flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl transition-all duration-200 ease-out ${
        expanded
          ? "h-[720px] max-h-[88vh] w-[440px]"
          : "h-[600px] max-h-[80vh] w-[380px]"
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between bg-gradient-to-r from-[var(--brand-blue)] to-[var(--brand-blue-dark)] px-4 py-3.5">
        <div className="flex items-center gap-3">
          <BotAvatar size={40} withStatus />
          <div>
            <p className="text-[15px] font-semibold leading-tight text-white">
              {BOT_NAME} - {BOT_TAGLINE}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setExpanded((e) => !e)}
            aria-label={expanded ? "Shrink chat" : "Enlarge chat"}
            className="flex h-8 w-8 items-center justify-center rounded-full text-white/80 transition hover:bg-white/15 hover:text-white"
          >
            {expanded ? (
              <Minimize2 className="h-4 w-4" />
            ) : (
              <Maximize2 className="h-4 w-4" />
            )}
          </button>
          <button
            onClick={onClose}
            aria-label="Close chat"
            className="flex h-8 w-8 items-center justify-center rounded-full text-white/80 transition hover:bg-white/15 hover:text-white"
          >
            <X className="h-[18px] w-[18px]" />
          </button>
        </div>
      </div>

      {/* Thread */}
      <ThreadPrimitive.Root className="flex flex-1 flex-col overflow-hidden bg-slate-50">
        <ThreadPrimitive.Viewport className="consulate-scroll flex-1 space-y-1 overflow-y-auto py-3">
          <ThreadWelcome />
          <ThreadPrimitive.Messages
            components={{ UserMessage, AssistantMessage }}
          />
          <ThinkingIndicator />
        </ThreadPrimitive.Viewport>
        <Composer />
      </ThreadPrimitive.Root>
    </div>
  );
}

export function ChatWidget({ onClosed }: { onClosed?: () => void }) {
  const [open, setOpen] = useState(false);

  const handleClose = () => {
    setOpen(false);
    onClosed?.();
  };

  return (
    <div className="fixed bottom-5 right-5 z-50 flex flex-col items-end gap-3">
      {open && <ChatPanel onClose={handleClose} />}
      {!open && (
        <button
          onClick={() => setOpen(true)}
          aria-label="Open chat"
          className="relative flex h-14 w-14 items-center justify-center rounded-full bg-[var(--brand-blue)] text-white shadow-lg transition hover:scale-105 hover:bg-[var(--brand-blue-dark)]"
        >
          <span className="absolute inset-0 rounded-full bg-[var(--brand-blue)] opacity-30 animate-ping" />
          <Smile className="relative h-6 w-6" />
        </button>
      )}
    </div>
  );
}
