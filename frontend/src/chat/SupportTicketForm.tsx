import { useState, type FormEvent, type ReactNode } from "react";
import { Loader2, Send } from "lucide-react";
import { submitSupportTicket } from "./backendAdapter";

const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;

const inputClass =
  "w-full rounded-xl border border-[var(--cc-border)] bg-[var(--cc-surface-2)] px-3 py-2 text-[13.5px] text-[var(--cc-text)] outline-none transition-shadow placeholder:text-[var(--cc-text-faint)] focus:border-[var(--cc-accent)] focus:ring-2 focus:ring-[var(--cc-ring)]";

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[11px] font-semibold uppercase tracking-wide text-[var(--cc-text-soft)]">
        {label}
      </span>
      {children}
    </label>
  );
}

export function SupportTicketForm({
  defaultMessage,
  onCancel,
  onSubmitted,
}: {
  defaultMessage: string;
  onCancel: () => void;
  onSubmitted: () => void;
}) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [city, setCity] = useState("");
  const [state, setState] = useState("");
  const [message, setMessage] = useState(defaultMessage);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const emailValid = EMAIL_RE.test(email.trim());
  const canSubmit =
    name.trim() && emailValid && city.trim() && state.trim() && message.trim();

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!canSubmit || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      await submitSupportTicket({
        name: name.trim(),
        email: email.trim(),
        city: city.trim(),
        state: state.trim(),
        message: message.trim(),
      });
      onSubmitted();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not submit your query right now. Please try again.",
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="space-y-3 rounded-2xl border border-[var(--cc-border)] bg-[var(--cc-elevated)] p-4 shadow-[var(--cc-shadow-sm)]"
    >
      <div>
        <p className="text-[13.5px] font-semibold text-[var(--cc-text)]">
          Raise this with the consulate
        </p>
        <p className="mt-0.5 text-xs leading-relaxed text-[var(--cc-text-soft)]">
          Share a few details and someone from the consulate will get back to
          you by email.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-2.5">
        <Field label="Name">
          <input
            className={inputClass}
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Full name"
            maxLength={200}
            required
          />
        </Field>
        <Field label="Email">
          <input
            className={inputClass}
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com"
            maxLength={320}
            required
          />
        </Field>
        <Field label="City">
          <input
            className={inputClass}
            value={city}
            onChange={(e) => setCity(e.target.value)}
            placeholder="City"
            maxLength={100}
            required
          />
        </Field>
        <Field label="State">
          <input
            className={inputClass}
            value={state}
            onChange={(e) => setState(e.target.value)}
            placeholder="State"
            maxLength={100}
            required
          />
        </Field>
      </div>

      <Field label="Your question">
        <textarea
          className={`${inputClass} consulate-scroll min-h-20 resize-none rounded-xl`}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          rows={3}
          maxLength={5000}
          required
        />
      </Field>

      {error && (
        <p className="rounded-lg bg-red-500/10 px-2.5 py-1.5 text-xs text-red-600">
          {error}
        </p>
      )}

      <div className="flex items-center gap-2 pt-1">
        <button
          type="button"
          onClick={onCancel}
          className="rounded-full px-3 py-2 text-xs font-medium text-[var(--cc-text-soft)] transition-colors hover:bg-[var(--cc-surface-2)]"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={!canSubmit || submitting}
          className="flex flex-1 items-center justify-center gap-1.5 rounded-full bg-[var(--cc-accent)] px-4 py-2 text-xs font-semibold text-[var(--cc-accent-ink)] transition-all duration-150 hover:bg-[var(--cc-accent-hover)] active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40"
        >
          {submitting ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Send className="h-3.5 w-3.5" />
          )}
          {submitting ? "Submitting..." : "Submit query"}
        </button>
      </div>
    </form>
  );
}
