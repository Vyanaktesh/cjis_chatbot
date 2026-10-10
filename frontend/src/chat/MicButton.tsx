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
 * the existing typed text and the transcript stay in one value. */
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

  async function start() {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setError("Recording isn't supported in this browser.");
      return;
    }
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setError("Microphone access was blocked.");
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

  const title = error ?? (recording ? "Stop recording" : "Record a voice message");

  return (
    <button
      type="button"
      onClick={recording ? stop : start}
      disabled={busy}
      aria-label={recording ? "Stop recording" : "Record a voice message"}
      aria-pressed={recording}
      title={title}
      className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-full border transition-all duration-150 active:scale-95 disabled:opacity-40 ${
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
  );
}
