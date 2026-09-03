import { useEffect, useState, type ReactNode } from "react";
import {
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
} from "@assistant-ui/react";
import {
  ExternalLink,
  Heart,
  Maximize2,
  Minimize2,
  Moon,
  Send,
  Smile,
  Sun,
  X,
} from "lucide-react";
import type { ChatResponse } from "./backendAdapter";
import { playSend } from "./sound";
import { SupportTicketForm } from "./SupportTicketForm";
import { CitizenCornerForm } from "./CitizenCornerForm";

const BOT_NAME = "DOST";
const BOT_TAGLINE = "Kahin Bhi, Kabhi Bhi";
const GREETING =
  "Hello, I am DOST! DOST HAI NA!! (Kahin Bhi, Kabhi Bhi) Your friend to help you. Got any questions?";

/* ---- Theme (light/dark) ------------------------------------------------ */

type Theme = "light" | "dark";
const THEME_STORAGE_KEY = "dost-theme";

function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    // localStorage unavailable (private mode etc.) -- fall through to OS pref
  }
  if (
    typeof window !== "undefined" &&
    window.matchMedia?.("(prefers-color-scheme: dark)").matches
  ) {
    return "dark";
  }
  return "light";
}

/** Applies the theme by stamping an explicit class on <html> so the token
 * overrides in index.css win over the OS `prefers-color-scheme` default in
 * both directions; persists the choice for next visit. */
function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(initialTheme);

  useEffect(() => {
    const root = document.documentElement;
    root.classList.remove("cc-theme-light", "cc-theme-dark");
    root.classList.add(theme === "dark" ? "cc-theme-dark" : "cc-theme-light");
    try {
      localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // best-effort persistence only
    }
  }, [theme]);

  return [theme, () => setTheme((t) => (t === "dark" ? "light" : "dark"))];
}

/* ---- Small building blocks --------------------------------------------- */

function BotAvatar({
  size = 40,
  withStatus = false,
}: {
  size?: number;
  withStatus?: boolean;
}) {
  const iconSize = Math.round(size * 0.55);
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <div className="flex h-full w-full items-center justify-center rounded-full bg-gradient-to-br from-[var(--brand-blue)] to-[var(--brand-blue-dark)] shadow-sm ring-2 ring-white/70">
        <Smile
          style={{ width: iconSize, height: iconSize }}
          className="text-white"
        />
      </div>
      {withStatus && (
        <span className="absolute -bottom-0.5 -right-0.5 flex h-3 w-3 items-center justify-center rounded-full bg-[var(--cc-surface)]">
          <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
        </span>
      )}
    </div>
  );
}

/** Shared assistant-bubble shell: avatar + a themed rounded card, with the
 * mount animation. Every bot-side message (answer, offer, success, thanks)
 * goes through this so they're visually identical and animate consistently. */
function BotBubble({ children }: { children: ReactNode }) {
  return (
    <div className="consulate-msg-in flex items-start gap-2 px-4 py-1.5">
      <BotAvatar size={26} />
      <div className="min-w-0 max-w-[85%] break-words rounded-2xl rounded-bl-md border border-[var(--cc-border-soft)] bg-[var(--cc-surface)] px-4 py-2.5 text-[14.5px] leading-relaxed text-[var(--cc-text)] shadow-[var(--cc-shadow-sm)]">
        {children}
      </div>
    </div>
  );
}

const CATEGORY_STYLES: Record<string, string> = {
  PASSPORT: "bg-[var(--cc-accent-soft)] text-[var(--cc-accent-text)]",
  OCI: "bg-emerald-500/10 text-emerald-600",
  VISA: "bg-amber-500/10 text-amber-600",
};
const DEFAULT_CATEGORY_STYLE = "bg-[var(--cc-surface-2)] text-[var(--cc-text-soft)]";

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
      className="group flex max-w-full items-center gap-2 rounded-lg border border-[var(--cc-border)] bg-[var(--cc-surface)] px-2.5 py-1.5 text-xs text-[var(--cc-text-soft)] shadow-[var(--cc-shadow-sm)] transition-all duration-150 hover:-translate-y-px hover:border-[var(--cc-accent)]"
      title={`${category ? `${category} · ` : ""}${host}`}
    >
      {index && (
        <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-[var(--cc-accent)] text-[10px] font-semibold text-[var(--cc-accent-ink)]">
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
      <span className="min-w-0 flex-1 truncate font-medium text-[var(--cc-text)] group-hover:text-[var(--cc-accent-text)]">
        {host}
      </span>
      <ExternalLink className="ml-auto h-3 w-3 shrink-0 text-[var(--cc-text-faint)] group-hover:text-[var(--cc-accent-text)]" />
    </a>
  );
}

const INLINE_CITATION_RE = /\[(\d+)\]/g;

/**
 * The model follows every factual claim with bracket citations like
 * "[5][6]" (see app/generation/prompt.py's SYSTEM_PROMPT). Raw brackets in
 * running prose read as clutter, so they render as small superscript badges
 * matching the numbered badges in SourceChip -- one citation system, not two.
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
        className="mx-px inline-flex h-[14px] min-w-[14px] items-center justify-center rounded-full bg-[var(--cc-accent-soft)] px-1 text-[9px] font-semibold leading-none text-[var(--cc-accent-text)]"
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
    <MessagePrimitive.Root className="consulate-msg-in flex justify-end px-4 py-1.5">
      <div className="max-w-[85%] break-words rounded-2xl rounded-br-md bg-[var(--cc-accent)] px-4 py-2.5 text-[14.5px] leading-relaxed text-[var(--cc-accent-ink)] shadow-[var(--cc-shadow-sm)]">
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
    <MessagePrimitive.Root className="flex justify-start px-0 py-0">
      {/* hasContent stays false while a run is in flight before any part
          arrives -- skip the empty bubble; ThinkingIndicator covers that. */}
      <MessagePrimitive.If hasContent>
        <BotBubble>
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
        </BotBubble>
      </MessagePrimitive.If>
    </MessagePrimitive.Root>
  );
}

// Shown as tappable chips under the greeting (only while the thread is
// empty). `prompt` is the full question sent to the backend; the child text
// is the short label. send -> fires the question immediately on tap.
const STARTER_PROMPTS = [
  { label: "Renew passport", prompt: "How do I renew my passport?" },
  { label: "Apply for OCI", prompt: "How do I apply for an OCI card?" },
  { label: "Visa documents", prompt: "What documents do I need for a visa?" },
  { label: "Track application", prompt: "How do I check my application status?" },
  { label: "Attest a document", prompt: "How do I get a document attested?" },
];

function StarterChips() {
  return (
    <div className="flex flex-wrap gap-1.5 px-4 pl-[52px] pt-2">
      {STARTER_PROMPTS.map((p, i) => (
        <ThreadPrimitive.Suggestion
          key={p.label}
          prompt={p.prompt}
          send
          className="consulate-chip-in rounded-full border border-[var(--cc-border)] bg-[var(--cc-surface)] px-3 py-1.5 text-xs font-medium text-[var(--cc-text-soft)] shadow-[var(--cc-shadow-sm)] transition-all duration-150 hover:-translate-y-px hover:border-[var(--cc-accent)] hover:bg-[var(--cc-accent-soft)] hover:text-[var(--cc-accent-text)]"
          style={{ animationDelay: `${120 + i * 70}ms` }}
        >
          {p.label}
        </ThreadPrimitive.Suggestion>
      ))}
    </div>
  );
}

function ThreadWelcome() {
  return (
    <ThreadPrimitive.Empty>
      <BotBubble>
        <span className="text-[15px]">{GREETING}</span>
      </BotBubble>
      <StarterChips />
    </ThreadPrimitive.Empty>
  );
}

function ElapsedSeconds() {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, []);
  return <span>{seconds}s</span>;
}

function ThinkingIndicator() {
  return (
    <ThreadPrimitive.If running>
      <div className="consulate-msg-in flex items-center gap-2 px-4 py-1.5">
        <BotAvatar size={26} />
        <div className="flex items-center gap-2 rounded-2xl rounded-bl-md border border-[var(--cc-border-soft)] bg-[var(--cc-surface)] px-4 py-2.5 text-[13px] text-[var(--cc-text-soft)] shadow-[var(--cc-shadow-sm)]">
          <span className="flex gap-1">
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-[var(--cc-text-faint)] [animation-delay:-0.3s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-[var(--cc-text-faint)] [animation-delay:-0.15s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-[var(--cc-text-faint)]" />
          </span>
          <span>
            Looking into your question&hellip; <ElapsedSeconds />
          </span>
        </div>
      </div>
    </ThreadPrimitive.If>
  );
}

/* ---- Pill button helpers (shared styling) ------------------------------ */

function PrimaryPill({
  children,
  onClick,
}: {
  children: ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="rounded-full bg-[var(--cc-accent)] px-3 py-1.5 text-xs font-semibold text-[var(--cc-accent-ink)] transition-colors hover:bg-[var(--cc-accent-hover)]"
    >
      {children}
    </button>
  );
}

function GhostPill({
  children,
  onClick,
}: {
  children: ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="rounded-full px-3 py-1.5 text-xs font-medium text-[var(--cc-text-soft)] transition-colors hover:bg-[var(--cc-surface-2)]"
    >
      {children}
    </button>
  );
}

/* ---- Post-answer flows ------------------------------------------------- */

function ResolutionPrompt({
  onResolved,
  onNotResolved,
}: {
  onResolved: () => void;
  onNotResolved: () => void;
}) {
  return (
    <BotBubble>
      <p className="text-[13.5px]">Did this answer your question?</p>
      <div className="mt-2.5 flex gap-2">
        <PrimaryPill onClick={onResolved}>Yes, thanks</PrimaryPill>
        <GhostPill onClick={onNotResolved}>Not quite</GhostPill>
      </div>
    </BotBubble>
  );
}

function ThankYouMessage({ onCitizenCorner }: { onCitizenCorner: () => void }) {
  return (
    <BotBubble>
      <p className="text-[13.5px]">
        Thank you for visiting! If you'd like to share feedback on your
        experience, please visit Citizen Corner &mdash; we'd love to hear
        from you.
      </p>
      <button
        onClick={onCitizenCorner}
        className="mt-2.5 inline-flex items-center gap-1.5 rounded-full bg-[var(--cc-accent)] px-3 py-1.5 text-xs font-semibold text-[var(--cc-accent-ink)] transition-colors hover:bg-[var(--cc-accent-hover)]"
      >
        <Heart className="h-3.5 w-3.5" />
        Citizen Corner
      </button>
    </BotBubble>
  );
}

function TicketOfferBanner({
  onAccept,
  onDismiss,
}: {
  onAccept: () => void;
  onDismiss: () => void;
}) {
  return (
    <BotBubble>
      <p className="text-[13.5px]">
        No problem. Would you like me to pass this along to the consulate
        directly? Someone will follow up by email.
      </p>
      <div className="mt-2.5 flex gap-2">
        <PrimaryPill onClick={onAccept}>Raise a query</PrimaryPill>
        <GhostPill onClick={onDismiss}>No thanks</GhostPill>
      </div>
    </BotBubble>
  );
}

function TicketSuccessMessage() {
  return (
    <BotBubble>
      Your query has been submitted. Someone from the consulate will get back
      to you by email soon.
    </BotBubble>
  );
}

function CitizenSuccessMessage({
  referenceNumber,
}: {
  referenceNumber: string;
}) {
  return (
    <BotBubble>
      Thank you for sharing! Your reference number is{" "}
      <span className="font-semibold text-[var(--cc-accent-text)]">
        #{referenceNumber}
      </span>
      . It's in our review queue, and nothing is published until it's approved.
    </BotBubble>
  );
}

function QuickActionsBar({ onCitizenCorner }: { onCitizenCorner: () => void }) {
  return (
    <div className="flex flex-wrap gap-1.5 border-t border-[var(--cc-border-soft)] bg-[var(--cc-surface)] px-3 py-2">
      <button
        onClick={onCitizenCorner}
        className="flex items-center gap-1.5 rounded-full border border-[var(--cc-border)] px-3 py-1.5 text-xs font-medium text-[var(--cc-text-soft)] transition-all duration-150 hover:border-[var(--cc-accent)] hover:bg-[var(--cc-accent-soft)] hover:text-[var(--cc-accent-text)]"
      >
        <Heart className="h-3.5 w-3.5" />
        Citizen Corner
      </button>
    </div>
  );
}

function Composer() {
  return (
    <ComposerPrimitive.Root className="flex items-end gap-2 border-t border-[var(--cc-border-soft)] bg-[var(--cc-surface)] p-3">
      <ComposerPrimitive.Input
        placeholder="Ask me anything..."
        rows={1}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) playSend();
        }}
        className="consulate-scroll max-h-28 flex-1 resize-none rounded-2xl border border-[var(--cc-border)] bg-[var(--cc-surface-2)] px-4 py-2.5 text-[14.5px] text-[var(--cc-text)] outline-none transition-shadow placeholder:text-[var(--cc-text-faint)] focus:border-[var(--cc-accent)] focus:ring-2 focus:ring-[var(--cc-ring)]"
      />
      <ComposerPrimitive.Send asChild>
        <button
          type="submit"
          aria-label="Send message"
          onClick={() => playSend()}
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-[var(--cc-accent)] text-[var(--cc-accent-ink)] transition-all duration-150 hover:bg-[var(--cc-accent-hover)] active:scale-95 disabled:opacity-40"
        >
          <Send className="h-4 w-4" />
        </button>
      </ComposerPrimitive.Send>
    </ComposerPrimitive.Root>
  );
}

/* ---- Panel state machine ----------------------------------------------- */

// One post-answer flow, driven by each reply:
//   grounded reply    -> "resolution" (Did this answer your question?)
//     Yes             -> "thanked"    (closing message + Citizen Corner)
//     Not quite       -> "offer"      (escalate to a human ticket)
//   ungrounded reply  -> "offer" directly
//   offer -> Raise    -> "ticketForm" -> submitted -> "ticketSuccess"
type Flow =
  | "idle"
  | "resolution"
  | "thanked"
  | "offer"
  | "ticketForm"
  | "ticketSuccess";
type CitizenFlow = "hidden" | "form" | "success";

function ChatPanel({
  onClose,
  lastResult,
  theme,
  onToggleTheme,
}: {
  onClose: () => void;
  lastResult: ChatResponse | null;
  theme: Theme;
  onToggleTheme: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [flow, setFlow] = useState<Flow>("idle");
  const [seenResult, setSeenResult] = useState<ChatResponse | null>(null);
  const [citizenFlow, setCitizenFlow] = useState<CitizenFlow>("hidden");
  const [citizenReference, setCitizenReference] = useState<string | null>(null);

  // A fresh reply arrived: pick the post-answer flow.
  //  - grounded answer      -> ask whether it helped
  //  - service outage       -> no follow-up at all; the "try again" message
  //    stands on its own (escalating to a human ticket makes no sense for a
  //    transient technical error, only for a genuine can't-answer)
  //  - genuine can't-answer -> offer to escalate to the consulate
  // Protected state is the ticket form actively being filled in -- never
  // yank it out from under the user. Any lingering success/closing message
  // clears here, since the person has moved on to a new question.
  if (lastResult && lastResult !== seenResult) {
    setSeenResult(lastResult);
    if (flow !== "ticketForm") {
      if (lastResult.grounded) {
        setFlow("resolution");
      } else if (lastResult.generation_error) {
        setFlow("idle"); // service down -> just the "try again" message
      } else {
        setFlow("offer");
      }
    }
    if (citizenFlow === "success") {
      setCitizenFlow("hidden");
    }
  }

  const openCitizenCorner = () => setCitizenFlow("form");

  return (
    <div
      className={`consulate-panel-enter flex max-w-[92vw] flex-col overflow-hidden rounded-2xl border border-[var(--cc-border)] bg-[var(--cc-surface)] shadow-[var(--cc-shadow)] transition-all duration-200 ease-out ${
        expanded
          ? "h-[720px] max-h-[88vh] w-[440px]"
          : "h-[600px] max-h-[80vh] w-[380px]"
      }`}
    >
      {/* Header */}
      <div className="relative flex items-center justify-between overflow-hidden bg-gradient-to-br from-[var(--brand-blue)] to-[var(--brand-blue-dark)] px-4 py-3.5">
        {/* soft radial highlight so the band reads as designed, not flat */}
        <div
          className="pointer-events-none absolute inset-0 opacity-70"
          style={{
            background:
              "radial-gradient(120% 140% at 12% -20%, rgba(255,255,255,.22), transparent 55%)",
          }}
        />
        <div className="relative flex items-center gap-3">
          <BotAvatar size={40} withStatus />
          <div className="leading-tight">
            <p className="text-[15px] font-semibold text-white">{BOT_NAME}</p>
            <p className="text-[11.5px] font-medium tracking-wide text-white/70">
              {BOT_TAGLINE}
            </p>
          </div>
        </div>
        <div className="relative flex items-center gap-1">
          <button
            onClick={onToggleTheme}
            aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            className="flex h-8 w-8 items-center justify-center rounded-full text-white/80 transition-colors hover:bg-white/15 hover:text-white"
          >
            {theme === "dark" ? (
              <Sun className="h-4 w-4" />
            ) : (
              <Moon className="h-4 w-4" />
            )}
          </button>
          <button
            onClick={() => setExpanded((e) => !e)}
            aria-label={expanded ? "Shrink chat" : "Enlarge chat"}
            className="flex h-8 w-8 items-center justify-center rounded-full text-white/80 transition-colors hover:bg-white/15 hover:text-white"
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
            className="flex h-8 w-8 items-center justify-center rounded-full text-white/80 transition-colors hover:bg-white/15 hover:text-white"
          >
            <X className="h-[18px] w-[18px]" />
          </button>
        </div>
      </div>

      {/* Thread */}
      <ThreadPrimitive.Root className="flex flex-1 flex-col overflow-hidden bg-[var(--cc-bg)]">
        <ThreadPrimitive.Viewport className="consulate-scroll flex-1 space-y-1 overflow-y-auto py-3">
          <ThreadWelcome />
          <ThreadPrimitive.Messages
            components={{ UserMessage, AssistantMessage }}
          />
          <ThinkingIndicator />

          {flow === "resolution" && (
            <ResolutionPrompt
              onResolved={() => setFlow("thanked")}
              onNotResolved={() => setFlow("offer")}
            />
          )}
          {flow === "thanked" && (
            <ThankYouMessage onCitizenCorner={openCitizenCorner} />
          )}
          {flow === "offer" && (
            <TicketOfferBanner
              onAccept={() => setFlow("ticketForm")}
              onDismiss={() => setFlow("idle")}
            />
          )}
          {flow === "ticketForm" && (
            <div className="consulate-msg-in px-4 py-1.5">
              <SupportTicketForm
                defaultMessage={lastResult?.query ?? ""}
                onCancel={() => setFlow("idle")}
                onSubmitted={() => setFlow("ticketSuccess")}
              />
            </div>
          )}
          {flow === "ticketSuccess" && <TicketSuccessMessage />}

          {citizenFlow === "form" && (
            <div className="consulate-msg-in px-4 py-1.5">
              <CitizenCornerForm
                onCancel={() => setCitizenFlow("hidden")}
                onSubmitted={(referenceNumber) => {
                  setCitizenReference(referenceNumber);
                  setCitizenFlow("success");
                }}
              />
            </div>
          )}
          {citizenFlow === "success" && citizenReference && (
            <CitizenSuccessMessage referenceNumber={citizenReference} />
          )}
        </ThreadPrimitive.Viewport>
        <QuickActionsBar onCitizenCorner={openCitizenCorner} />
        <Composer />
      </ThreadPrimitive.Root>
    </div>
  );
}

export function ChatWidget({
  onClosed,
  lastResult,
}: {
  onClosed?: () => void;
  lastResult?: ChatResponse | null;
}) {
  const [open, setOpen] = useState(false);
  const [theme, toggleTheme] = useTheme();

  const handleClose = () => {
    setOpen(false);
    onClosed?.();
  };

  return (
    <div className="fixed bottom-5 right-5 z-50 flex flex-col items-end gap-3">
      {open && (
        <ChatPanel
          onClose={handleClose}
          lastResult={lastResult ?? null}
          theme={theme}
          onToggleTheme={toggleTheme}
        />
      )}
      {!open && (
        <button
          onClick={() => setOpen(true)}
          aria-label="Chat with DOST"
          className="consulate-float group flex items-center gap-2.5 rounded-full bg-gradient-to-br from-[var(--brand-blue)] to-[var(--brand-blue-dark)] py-2 pl-2 pr-4 text-white shadow-[var(--cc-shadow)] transition-all duration-200 hover:-translate-y-0.5 hover:shadow-xl"
        >
          <span className="relative flex h-10 w-10 items-center justify-center rounded-full bg-white/15 ring-1 ring-white/25">
            <span className="absolute inset-0 rounded-full bg-white/20 opacity-40 animate-ping" />
            <Smile className="relative h-5 w-5" />
          </span>
          <span className="flex flex-col items-start leading-tight">
            <span className="text-[13px] font-semibold">Chat with DOST</span>
            <span className="text-[10.5px] font-medium text-white/70">
              Passport · OCI · Visa
            </span>
          </span>
        </button>
      )}
    </div>
  );
}
