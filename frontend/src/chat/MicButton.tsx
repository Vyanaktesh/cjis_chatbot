import { useEffect, useRef, useState } from "react";
import { Loader2, Mic, Square } from "lucide-react";
import { unstable_useComposerInput } from "@assistant-ui/react";
import { transcribeAudio } from "./backendAdapter";

/** Mic button for the composer: records with the browser's MediaRecorder,
 * sends the audio to the self-hosted /transcribe endpoint, and writes the
 * returned text into the composer input for the user to review and edit
 * before sending (it never auto-sends).
 *
 * Text is written via unstable_useComposerInput().setText -- the supported
 * headless bridge in this @assistant-ui/react version (0.15.x does NOT export
 * a top-level useComposerRuntime hook). It mirrors ComposerPrimitive.Input, so
 * the existing typed text and the transcript stay in one value.
 *
 * NOTE: browsers only expose getUserMedia in a "secure context" -- https:// or
 * http://localhost. Over plain http on a LAN IP (e.g. a phone hitting
 * http://192.168.x.x), navigator.mediaDevices is undefined and the mic cannot
 * work; we surface that clearly instead of failing silently. */
export function MicButton() {
  const composer = unstable_useComposerInput();
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  // Release the mic if the component unmounts mid-recording.
  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // Auto-dismiss an error message after a few seconds so it doesn't linger.
  useEffect(() => {
    if (!error) return;
    const t = setTimeout(() => setError(null), 7000);
    return () => clearTimeout(t);
  }, [error]);

  async function start() {
    setError(null);
    if (typeof MediaRecorder === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      // The usual cause on a phone demo: the site is open over plain http on a
      // LAN IP, which is not a secure context, so the mic API is unavailable.
      setError(
        typeof window !== "undefined" && window.isSecureContext === false
          ? "Voice needs a secure connection. Open the site over https:// (or on the same computer via localhost)."
          : "Voice recording isn't supported in this browser.",
      );
      return;
    }
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      const name = err instanceof DOMException ? err.name : "";
      if (name === "NotAllowedError" || name === "SecurityError") {
        setError("Microphone access was blocked. Allow it in your browser, then try again.");
      } else if (name === "NotFoundError" || name === "OverconstrainedError") {
        setError("No microphone was found on this device.");
      } else {
        setError("Couldn't start recording. Please try again.");
      }
      return;
    }
    streamRef.current = stream;
    chunksRef.current = [];
    const recorder = new MediaRecorder(stream);
    recorderRef.current = recorder;
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    recorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      const blob = new Blob(chunksRef.current, {
        type: recorder.mimeType || "audio/webm",
      });
      if (blob.size > 0) void sendForTranscription(blob);
    };
    recorder.start();
    setRecording(true);
  }

  function stop() {
    setRecording(false);
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== "inactive") recorder.stop();
  }

  async function sendForTranscription(blob: Blob) {
    setBusy(true);
    setError(null);
    try {
      const text = (await transcribeAudio(blob)).trim();
      if (text) {
        const existing = composer.value;
        const joiner = existing && !existing.endsWith(" ") ? " " : "";
        composer.setText(existing + joiner + text);
      } else {
        setError("Didn't catch that. Please try again.");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Transcription failed.");
    } finally {
      setBusy(false);
    }
  }

  const title = recording ? "Stop recording" : "Record a voice message";

  return (
    <div className="relative shrink-0">
      {error && (
        <div
          role="alert"
          className="absolute bottom-12 right-0 z-10 w-56 rounded-lg border border-[var(--cc-border)] bg-[var(--cc-surface)] px-3 py-2 text-xs leading-snug text-red-600 shadow-[var(--cc-shadow)]"
        >
          {error}
        </div>
      )}
      <button
        type="button"
        onClick={recording ? stop : start}
        disabled={busy}
        aria-label={title}
        aria-pressed={recording}
        title={title}
        className={`flex h-10 w-10 items-center justify-center rounded-full border transition-all duration-150 active:scale-95 disabled:opacity-40 ${
          recording
            ? "animate-pulse border-red-500 bg-red-500/10 text-red-600"
            : "border-[var(--cc-border)] bg-[var(--cc-surface-2)] text-[var(--cc-text-soft)] hover:border-[var(--cc-accent)] hover:text-[var(--cc-accent-text)]"
        }`}
      >
        {busy ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : recording ? (
          <Square className="h-4 w-4" />
        ) : (
          <Mic className="h-4 w-4" />
        )}
      </button>
    </div>
  );
}
