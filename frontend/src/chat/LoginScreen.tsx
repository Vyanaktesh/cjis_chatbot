import { useState, type FormEvent } from "react";
import { Loader2, Lock } from "lucide-react";
import { login } from "./backendAdapter";

/** Shown instead of the chat widget when the backend reports that a password
 * is required and this browser is not signed in yet (see App.tsx and
 * app/api/auth.py). On success it calls onAuthenticated so the app swaps to
 * the chat. */
export function LoginScreen({
  onAuthenticated,
}: {
  onAuthenticated: () => void;
}) {
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!password || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const ok = await login(password);
      if (ok) {
        onAuthenticated();
      } else {
        setError("Incorrect password. Please try again.");
        setPassword("");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not sign in.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-[var(--cc-bg)] px-4">
      <form
        onSubmit={onSubmit}
        className="w-full max-w-sm rounded-2xl border border-[var(--cc-border)] bg-[var(--cc-surface)] p-6 shadow-[var(--cc-shadow-sm)]"
      >
        <div className="mb-4 flex items-center gap-3">
          <span className="flex h-10 w-10 items-center justify-center rounded-full bg-[var(--brand-blue)]">
            <Lock className="h-5 w-5 text-white" />
          </span>
          <div>
            <h1 className="text-base font-semibold text-[var(--cc-text)]">
              Consulate Assistant
            </h1>
            <p className="text-xs text-[var(--cc-text-soft)]">
              Enter the access password to continue.
            </p>
          </div>
        </div>

        <label className="block">
          <span className="mb-1 block text-[11px] font-semibold uppercase tracking-wide text-[var(--cc-text-soft)]">
            Password
          </span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoFocus
            maxLength={500}
            autoComplete="current-password"
            className="w-full rounded-xl border border-[var(--cc-border)] bg-[var(--cc-surface-2)] px-3 py-2 text-base sm:text-[14px] text-[var(--cc-text)] outline-none transition-shadow placeholder:text-[var(--cc-text-faint)] focus:border-[var(--cc-accent)] focus:ring-2 focus:ring-[var(--cc-ring)]"
          />
        </label>

        {error && (
          <p
            role="alert"
            className="mt-2 rounded-lg bg-red-500/10 px-2.5 py-1.5 text-xs text-red-600"
          >
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={!password || submitting}
          className="mt-4 flex w-full items-center justify-center gap-2 rounded-full bg-[var(--cc-accent)] px-4 py-2.5 text-sm font-semibold text-[var(--cc-accent-ink)] transition-all duration-150 hover:bg-[var(--cc-accent-hover)] active:scale-[0.99] disabled:opacity-40"
        >
          {submitting && <Loader2 className="h-4 w-4 animate-spin" />}
          {submitting ? "Signing in..." : "Sign in"}
        </button>
      </form>
    </div>
  );
}
