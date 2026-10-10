import { useEffect, useMemo, useState } from "react";
import { AssistantRuntimeProvider, useLocalRuntime } from "@assistant-ui/react";
import {
  createBackendAdapter,
  fetchAuthStatus,
  resetSession,
  type ChatResponse,
} from "./chat/backendAdapter";
import { ChatWidget } from "./chat/ChatWidget";
import { LoginScreen } from "./chat/LoginScreen";
import consulateLogo from "./assets/consulate-logo.jpg";

/**
 * Demo host page: a minimal placeholder standing in for the real consulate
 * website, so the widget can be seen in context (bottom-right corner,
 * floating over page content) exactly as it would once embedded there.
 * Only <ChatWidget /> (plus its AssistantRuntimeProvider wrapper) is the
 * actual Phase 8 deliverable — everything else on this page is a stand-in.
 */
function DemoPage() {
  return (
    <div className="min-h-screen bg-[var(--cc-bg)] transition-colors">
      <header className="border-b border-[var(--cc-border)] bg-[var(--cc-surface)]">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <img
              src={consulateLogo}
              alt="Consulate General of India, Atlanta"
              className="h-10 w-10 shrink-0 rounded-full object-contain"
            />
            <div className="leading-tight">
              <p className="text-sm font-semibold text-[var(--cc-text)]">
                Consulate General of India
              </p>
              <p className="text-[11.5px] font-medium tracking-wide text-[var(--cc-text-soft)]">
                Atlanta
              </p>
            </div>
          </div>
          <nav className="hidden gap-6 text-sm text-[var(--cc-text-soft)] sm:flex">
            <span>Passport</span>
            <span>OCI</span>
            <span>Visa</span>
            <span>Contact</span>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-16">
        <h1 className="text-3xl font-semibold text-[var(--cc-text)]">
          Consular services
        </h1>
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
  const [lastResult, setLastResult] = useState<ChatResponse | null>(null);
  // useMemo, not a fresh function every render: useLocalRuntime should keep
  // the same adapter identity across re-renders (setLastResult below is one),
  // not tear down and recreate the underlying runtime each time.
  const adapter = useMemo(() => createBackendAdapter(setLastResult), []);
  const runtime = useLocalRuntime(adapter);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ChatWidget onClosed={onClosed} lastResult={lastResult} />
    </AssistantRuntimeProvider>
  );
}

export default function App() {
  const [resetKey, setResetKey] = useState(0);
  // null = still checking /auth/me; false = password needed, not signed in;
  // true = authenticated (or the gate is off entirely).
  const [authed, setAuthed] = useState<boolean | null>(null);

  useEffect(() => {
    let active = true;
    fetchAuthStatus().then((status) => {
      if (active) setAuthed(!status.login_required || status.authenticated);
    });
    return () => {
      active = false;
    };
  }, []);

  if (authed === null) {
    // Brief blank state while the auth check is in flight.
    return <div className="min-h-screen bg-[var(--cc-bg)]" />;
  }

  if (!authed) {
    return <LoginScreen onAuthenticated={() => setAuthed(true)} />;
  }

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
