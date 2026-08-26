import { useState } from "react";
import { AssistantRuntimeProvider, useLocalRuntime } from "@assistant-ui/react";
import { backendAdapter, resetSession } from "./chat/backendAdapter";
import { ChatWidget } from "./chat/ChatWidget";

/**
 * Demo host page: a minimal placeholder standing in for the real consulate
 * website, so the widget can be seen in context (bottom-right corner,
 * floating over page content) exactly as it would once embedded there.
 * Only <ChatWidget /> (plus its AssistantRuntimeProvider wrapper) is the
 * actual Phase 8 deliverable — everything else on this page is a stand-in.
 */
function DemoPage() {
  return (
    <div className="min-h-screen bg-slate-100">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-full bg-[var(--brand-blue)]" />
            <div>
              <p className="text-sm font-semibold text-slate-800">
                Consulate General of India
              </p>
              <p className="text-xs text-slate-500">Demo site — Phase 8</p>
            </div>
          </div>
          <nav className="hidden gap-6 text-sm text-slate-600 sm:flex">
            <span>Passport</span>
            <span>OCI</span>
            <span>Visa</span>
            <span>Contact</span>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-16">
        <h1 className="text-3xl font-semibold text-slate-800">
          Consular services
        </h1>
        <p className="mt-3 max-w-2xl text-slate-600">
          This placeholder page stands in for the real consulate site. The
          chat widget in the bottom-right corner is the actual Phase 8
          deliverable — it talks to the FastAPI backend built in Phases
          1-7 (hybrid retrieval + grounded, self-hosted Qwen3 generation).
        </p>
      </main>
    </div>
  );
}

/**
 * useLocalRuntime keeps its thread state (all messages) for as long as the
 * component calling it stays mounted -- there's no "clear conversation"
 * method on a local runtime's thread list (verified against
 * @assistant-ui/core: switchToNewThread() throws "Method not implemented"
 * on LocalThreadListRuntimeCore). So clearing the conversation on close
 * means unmounting and remounting the whole runtime instead: bumping
 * `resetKey` changes this component's `key` in App below, which forces
 * React to throw away the old ChatRuntime (and the useLocalRuntime inside
 * it) and construct a brand new one -- a fresh, empty thread.
 */
function ChatRuntime({ onClosed }: { onClosed: () => void }) {
  const runtime = useLocalRuntime(backendAdapter);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ChatWidget onClosed={onClosed} />
    </AssistantRuntimeProvider>
  );
}

export default function App() {
  const [resetKey, setResetKey] = useState(0);

  return (
    <>
      <DemoPage />
      <ChatRuntime
        key={resetKey}
        onClosed={() => {
          resetSession();
          setResetKey((k) => k + 1);
        }}
      />
    </>
  );
}
