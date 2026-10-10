import { useRef, useState, type FormEvent, type ReactNode } from "react";
import {
  Camera,
  Compass,
  Loader2,
  MessageSquareHeart,
  Paperclip,
  Quote,
  Send,
  X,
} from "lucide-react";
import {
  submitCitizenCorner,
  type CitizenSubmissionType,
} from "./backendAdapter";

const SUBMISSION_TYPES: {
  value: CitizenSubmissionType;
  label: string;
  icon: typeof Quote;
}[] = [
  { value: "testimonial", label: "Testimonial", icon: Quote },
  { value: "experience", label: "Experience", icon: Compass },
  { value: "feedback", label: "Feedback", icon: MessageSquareHeart },
  { value: "photo", label: "Photo", icon: Camera },
];

const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;

// Mirrors app/api/main.py's _MAX_PHOTO_BYTES / _ALLOWED_PHOTO_CONTENT_TYPES --
// placeholder limits, not yet informed by "D8" (the actual consent/
// acceptance rules doc). Checked here too so a citizen finds out
// immediately rather than after a round trip to the server.
const MAX_PHOTO_BYTES = 8 * 1024 * 1024;
const ALLOWED_PHOTO_TYPES = ["image/jpeg", "image/png", "image/webp", "image/gif"];

const inputClass =
  "w-full rounded-xl border border-[var(--cc-border)] bg-[var(--cc-surface-2)] px-3 py-2 text-base sm:text-[13.5px] text-[var(--cc-text)] outline-none transition-shadow placeholder:text-[var(--cc-text-faint)] focus:border-[var(--cc-accent)] focus:ring-2 focus:ring-[var(--cc-ring)]";

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

export function CitizenCornerForm({
  onCancel,
  onSubmitted,
}: {
  onCancel: () => void;
  onSubmitted: (referenceNumber: string) => void;
}) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [submissionType, setSubmissionType] =
    useState<CitizenSubmissionType | null>(null);
  const [anonymous, setAnonymous] = useState(false);
  const [photo, setPhoto] = useState<File | null>(null);
  const [photoError, setPhotoError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const emailValid = EMAIL_RE.test(email.trim());
  const canSubmit =
    name.trim() &&
    emailValid &&
    title.trim() &&
    content.trim() &&
    submissionType &&
    !photoError;

  function handlePhotoChange(file: File | undefined) {
    if (!file) {
      setPhoto(null);
      setPhotoError(null);
      return;
    }
    if (!ALLOWED_PHOTO_TYPES.includes(file.type)) {
      setPhoto(null);
      setPhotoError("Photo must be a JPEG, PNG, WEBP, or GIF image.");
      return;
    }
    if (file.size > MAX_PHOTO_BYTES) {
      setPhoto(null);
      setPhotoError("Photo must be under 8MB.");
      return;
    }
    setPhotoError(null);
    setPhoto(file);
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!canSubmit || submitting || !submissionType) return;
    setSubmitting(true);
    setError(null);
    try {
      const { ticketId } = await submitCitizenCorner({
        name: name.trim(),
        email: email.trim(),
        title: title.trim(),
        content: content.trim(),
        submissionType,
        anonymous,
        photo: photo ?? undefined,
      });
      onSubmitted(ticketId);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not submit your feedback right now. Please try again.",
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
          Share with the consulate
        </p>
        <p className="mt-0.5 text-xs leading-relaxed text-[var(--cc-text-soft)]">
          A testimonial, feedback, experience, or photo.
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
      </div>

      <div>
        <span className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wide text-[var(--cc-text-soft)]">
          What is this?
        </span>
        <div className="flex flex-wrap gap-1.5">
          {SUBMISSION_TYPES.map(({ value, label, icon: Icon }) => {
            const selected = submissionType === value;
            return (
              <button
                key={value}
                type="button"
                onClick={() => setSubmissionType(value)}
                aria-pressed={selected}
                className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-semibold transition-all duration-150 ${
                  selected
                    ? "bg-[var(--cc-accent)] text-[var(--cc-accent-ink)] shadow-[var(--cc-shadow-sm)]"
                    : "border border-[var(--cc-border)] text-[var(--cc-text-soft)] hover:border-[var(--cc-accent)] hover:bg-[var(--cc-accent-soft)] hover:text-[var(--cc-accent-text)]"
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                {label}
              </button>
            );
          })}
        </div>
      </div>

      <Field label="Title">
        <input
          className={inputClass}
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="A short title"
          maxLength={200}
          required
        />
      </Field>

      <Field label="Details">
        <textarea
          className={`${inputClass} consulate-scroll min-h-20 resize-none rounded-xl`}
          value={content}
          onChange={(e) => setContent(e.target.value)}
          rows={3}
          maxLength={5000}
          required
        />
      </Field>

      <div className="flex items-center justify-between gap-2">
        <label className="flex items-center gap-2 text-xs text-[var(--cc-text-soft)]">
          <input
            type="checkbox"
            checked={anonymous}
            onChange={(e) => setAnonymous(e.target.checked)}
            className="h-3.5 w-3.5 rounded border-[var(--cc-border-strong)] text-[var(--cc-accent)] focus:ring-[var(--cc-accent)]"
          />
          Publish anonymously if approved
        </label>

        <input
          ref={fileInputRef}
          type="file"
          accept={ALLOWED_PHOTO_TYPES.join(",")}
          className="hidden"
          onChange={(e) => handlePhotoChange(e.target.files?.[0])}
        />
        {!photo && (
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="flex shrink-0 items-center gap-1 rounded-full border border-[var(--cc-border)] px-2.5 py-1 text-xs font-medium text-[var(--cc-text-soft)] transition-colors hover:bg-[var(--cc-surface-2)]"
          >
            <Paperclip className="h-3.5 w-3.5" />
            Attach photo
          </button>
        )}
      </div>

      {photo && (
        <div className="flex items-center gap-2 rounded-lg border border-[var(--cc-border)] bg-[var(--cc-surface-2)] px-2.5 py-1.5 text-xs text-[var(--cc-text-soft)]">
          <span className="truncate">{photo.name}</span>
          <button
            type="button"
            onClick={() => handlePhotoChange(undefined)}
            aria-label="Remove photo"
            className="ml-auto shrink-0 text-[var(--cc-text-faint)] transition-colors hover:text-[var(--cc-text)]"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}
      {photoError && (
        <p className="rounded-lg bg-red-500/10 px-2.5 py-1.5 text-xs text-red-600">
          {photoError}
        </p>
      )}

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
          {submitting ? "Submitting..." : "Submit"}
        </button>
      </div>
    </form>
  );
}
