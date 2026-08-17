import { useEffect, useState } from "react";
import {
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
} from "@assistant-ui/react";
import { MessageCircle, Send, X } from "lucide-react";

const BOT_NAME = "Consulate Assistant";
const BOT_SUBTITLE = "Passport · OCI · Visa help";

function SourceChip({ url, title }: { url?: string; title?: string }) {
  if (!url) return null;
  return (
    <a
      href={url}
      target="_blank"
      rel="noreferrer"
      className="mt-1.5 inline-flex max-w-full items-center gap-1 truncate rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs font-medium text-[var(--brand-blue)] transition hover:bg-slate-50 hover:underline"
    >
      {title ?? "Source"}
    </a>
  );
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
    <MessagePrimitive.Root className="flex justify-start px-4 py-1.5">
      {/* hasContent: false while a run is in flight and no parts have
          arrived yet — skip rendering an empty bubble; ThinkingIndicator
          (driven off ThreadPrimitive.If running) covers that state instead. */}
      <MessagePrimitive.If hasContent>
        <div className="max-w-[88%] rounded-2xl rounded-bl-md border border-slate-100 bg-white px-4 py-2.5 text-[14.5px] leading-relaxed text-slate-800 shadow-sm">
          <MessagePrimitive.Content>
            {({ part }) => {
              if (part.type === "text") {
                return <span className="whitespace-pre-wrap">{part.text}</span>;
              }
              if (part.type === "source") {
                return (
                  <div className="block">
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
      <div className="flex h-full flex-col items-center justify-center gap-3 px-8 text-center">
        <div className="flex h-11 w-11 items-center justify-center rounded-full bg-[var(--brand-blue-light)]">
          <MessageCircle className="h-5 w-5 text-[var(--brand-blue)]" />
        </div>
        <p className="text-sm leading-relaxed text-slate-500">
          Ask about passport, OCI, or visa services. Answers are grounded
          only in reviewed consulate documents, with sources cited.
        </p>
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
      <div className="flex justify-start px-4 py-1.5">
        <div className="flex items-center gap-2 rounded-2xl rounded-bl-md border border-slate-100 bg-white px-4 py-2.5 text-[13px] text-slate-500 shadow-sm">
          <span className="flex gap-1">
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-300 [animation-delay:-0.3s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-300 [animation-delay:-0.15s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-300" />
          </span>
          <span>
            Looking into your question… <ElapsedSeconds />
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
        className="max-h-28 flex-1 resize-none rounded-full border border-slate-200 bg-slate-50 px-4 py-2.5 text-[14.5px] text-slate-800 outline-none placeholder:text-slate-400 focus:border-[var(--brand-blue)] focus:ring-1 focus:ring-[var(--brand-blue)]"
      />
      <ComposerPrimitive.Send asChild>
        <button
          type="submit"
          aria-label="Send message"
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-[var(--brand-blue)] text-white transition hover:bg-[var(--brand-blue-dark)] disabled:opacity-40"
        >
          <Send className="h-4 w-4" />
        </button>
      </ComposerPrimitive.Send>
    </ComposerPrimitive.Root>
  );
}

function ChatPanel({ onClose }: { onClose: () => void }) {
  return (
    <div className="flex h-[600px] max-h-[80vh] w-[380px] max-w-[92vw] flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between bg-[var(--brand-blue)] px-4 py-3.5">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-white/15">
            <MessageCircle className="h-5 w-5 text-white" />
          </div>
          <div>
            <p className="text-[15px] font-semibold leading-tight text-white">
              {BOT_NAME}
            </p>
            <p className="text-xs leading-tight text-white/75">{BOT_SUBTITLE}</p>
          </div>
        </div>
        <button
          onClick={onClose}
          aria-label="Close chat"
          className="flex h-8 w-8 items-center justify-center rounded-full text-white/80 transition hover:bg-white/15 hover:text-white"
        >
          <X className="h-[18px] w-[18px]" />
        </button>
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

export function ChatWidget() {
  const [open, setOpen] = useState(false);

  return (
    <div className="fixed bottom-5 right-5 z-50 flex flex-col items-end gap-3">
      {open && <ChatPanel onClose={() => setOpen(false)} />}
      {!open && (
        <button
          onClick={() => setOpen(true)}
          aria-label="Open chat"
          className="flex h-14 w-14 items-center justify-center rounded-full bg-[var(--brand-blue)] text-white shadow-lg transition hover:scale-105 hover:bg-[var(--brand-blue-dark)]"
        >
          <MessageCircle className="h-6 w-6" />
        </button>
      )}
    </div>
  );
}
