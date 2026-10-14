import { useEffect, useRef, useState } from "react";
import { ArrowLeft, Keyboard, Loader2, MessageSquareText, Mic, Square, X } from "lucide-react";
import { useAui, useAuiState } from "@assistant-ui/react";
import { markNextQuestionAsVoice, transcribeAudio, type ChatResponse } from "./backendAdapter";
import { cleanForSpeech, speak, stopSpeaking } from "./speak";

/**
 * Full-panel voice conversation, in the style of Gemini Live / ChatGPT voice:
 * the visitor talks, the question goes straight to the backend (nothing is
 * typed into the chat box and there is no Send button), and the answer is read
 * aloud. The same question and the full written answer, with its sources and
 * the usual follow-up prompts, are in the chat underneath for when the
 * visitor goes back (← / "Type instead" / "End").
 *
 * Recording stops on its own after a short silence once the visitor has
 * spoken, or when they tap stop. It never restarts by itself after an answer
 * (so the assistant's own voice is never recorded); a tap asks the next one.
 */

type Phase = "starting" | "listening" | "transcribing" | "thinking" | "speaking" | "idle" | "error";

// Silence detection on the microphone's RMS level (0..1). The thresholds sit
// above the room's own background noise, learned while the mic is open, so a
// noisy room doesn't count as talking and a quiet one doesn't need shouting.
const MIN_SPEECH_LEVEL = 0.03; // talking is at least this loud...
const SPEECH_OVER_NOISE = 3; // ...and this many times the background noise
const SILENCE_OVER_NOISE = 1.8; // quieter than this x background = silence
const MIN_SPEECH_MS = 250; // a cough or a click is not a question
const SILENCE_MS = 1600; // quiet this long after talking -> stop and send
const NO_SPEECH_MS = 9000; // nothing said at all -> stop
const MAX_MS = 45000; // hard cap on one question

const STATUS: Record<Phase, string> = {
  starting: "Starting the microphone…",
  listening: "Listening…",
  transcribing: "Got it…",
  thinking: "Looking that up…",
  speaking: "Speaking",
  idle: "Tap the mic to ask a question",
  error: "",
};

export function VoiceMode({
  lastResult,
  onExit,
}: {
  lastResult: ChatResponse | null;
  onExit: () => void;
}) {
  const aui = useAui();
  // Id of the latest reply if it ended in an error, else null.
  const failedReplyId = useAuiState((s) => {
    const m = s.thread.messages.at(-1);
    return m?.role === "assistant" && m.status?.type === "incomplete" && m.status.reason === "error" ? m.id : null;
  });

  const [phase, setPhase] = useState<Phase>("starting");
  const [error, setError] = useState<string | null>(null);
  const [heard, setHeard] = useState(""); // what the visitor said
  const [reply, setReply] = useState(""); // what is being / was read aloud
  const [autoStop, setAutoStop] = useState(true); // false: no working level meter

  const orbRef = useRef<HTMLDivElement>(null);
  const mainButtonRef = useRef<HTMLButtonElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const meterRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const discardRef = useRef(false); // stop without sending (back / end)
  const pendingRef = useRef(false); // a question was sent, waiting for its answer
  const seenResultRef = useRef<ChatResponse | null>(lastResult);
  const mountedRef = useRef(true);

  function releaseMic() {
    if (meterRef.current) clearInterval(meterRef.current);
    meterRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    void audioCtxRef.current?.close().catch(() => undefined);
    audioCtxRef.current = null;
    orbRef.current?.style.setProperty("--level", "0");
  }

  async function startListening() {
    setError(null);
    stopSpeaking();
    if (typeof MediaRecorder === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      setPhase("error");
      setError(
        window.isSecureContext === false
          ? "Voice needs a secure connection (https://)."
          : "Voice recording isn't supported in this browser.",
      );
      return;
    }
    setPhase("starting");
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      });
    } catch (err) {
      const name = err instanceof DOMException ? err.name : "";
      setPhase("error");
      setError(
        name === "NotAllowedError" || name === "SecurityError"
          ? "Microphone access was blocked. Allow it in your browser, then tap the mic."
          : name === "NotFoundError"
            ? "No microphone was found on this device."
            : "Couldn't start the microphone. Please try again.",
      );
      return;
    }
    if (!mountedRef.current) {
      stream.getTracks().forEach((t) => t.stop());
      return;
    }
    streamRef.current = stream;
    chunksRef.current = [];
    discardRef.current = false;

    const recorder = new MediaRecorder(stream);
    recorderRef.current = recorder;
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };

    // Level meter + silence detection. If the browser gives us no working
    // meter (no AudioContext, or one that stays suspended, as Safari can
    // outside a tap), auto-stop is off and the visitor taps stop: a recording
    // is never thrown away for a "silence" we couldn't actually measure.
    let spoke = false;
    let loudSince = 0;
    let quietSince = 0;
    let noise = 0.01; // background level, learned while nobody is talking
    let started = performance.now();
    let meter: (() => void) | null = null;
    try {
      const ctx = new AudioContext();
      audioCtxRef.current = ctx;
      if (ctx.state !== "running") await ctx.resume().catch(() => undefined);
      if (!mountedRef.current || streamRef.current !== stream) return;
      if (ctx.state === "running") {
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 1024;
        ctx.createMediaStreamSource(stream).connect(analyser);
        const buf = new Float32Array(analyser.fftSize);
        const tick = () => {
          analyser.getFloatTimeDomainData(buf);
          let sum = 0;
          for (const v of buf) sum += v * v;
          const rms = Math.sqrt(sum / buf.length);
          orbRef.current?.style.setProperty("--level", Math.min(1, rms * 9).toFixed(3));
          const now = performance.now();
          if (rms > Math.max(MIN_SPEECH_LEVEL, noise * SPEECH_OVER_NOISE)) {
            loudSince = loudSince || now;
            if (now - loudSince >= MIN_SPEECH_MS) spoke = true;
            quietSince = 0;
          } else {
            loudSince = 0;
            if (!spoke) noise = noise * 0.95 + rms * 0.05; // learn the room
            else if (rms < Math.max(noise * SILENCE_OVER_NOISE, MIN_SPEECH_LEVEL * 0.6)) {
              quietSince = quietSince || now;
            }
          }
          const silentLongEnough = spoke && quietSince > 0 && now - quietSince > SILENCE_MS;
          const nothingSaid = !spoke && !loudSince && now - started > NO_SPEECH_MS;
          if (silentLongEnough || nothingSaid || now - started > MAX_MS) {
            if (meterRef.current) clearInterval(meterRef.current);
            meterRef.current = null;
            if (nothingSaid) discardRef.current = true;
            stopListening();
          }
        };
        meter = tick;
      }
    } catch {
      // No AudioContext: still works, the visitor just taps stop.
    }
    if (!mountedRef.current || streamRef.current !== stream) return;

    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
      releaseMic();
      if (!mountedRef.current) return;
      if (discardRef.current || blob.size === 0) {
        if (!spoke && discardRef.current) {
          setPhase("idle");
          setError("I didn't hear anything. Tap the mic and ask your question.");
        }
        return;
      }
      void sendAudio(blob);
    };
    recorder.start();
    started = performance.now();
    // A timer, not requestAnimationFrame: animation frames stop while the tab
    // is in the background, which would leave the recording running forever.
    if (meter) meterRef.current = setInterval(meter, 50);
    setAutoStop(Boolean(meter));
    setPhase("listening");
  }

  function stopListening() {
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== "inactive") recorder.stop();
    else releaseMic();
  }

  async function sendAudio(blob: Blob) {
    setPhase("transcribing");
    let text = "";
    try {
      text = (await transcribeAudio(blob)).trim();
    } catch (err) {
      if (!mountedRef.current) return;
      setPhase("error");
      setError(err instanceof Error ? err.message : "Couldn't understand that. Please try again.");
      return;
    }
    if (!mountedRef.current) return;
    if (!text) {
      setPhase("idle");
      setError("I didn't catch that. Tap the mic and try again.");
      return;
    }
    setHeard(text);
    setReply("");
    setPhase("thinking");
    // Straight to the backend: no text in the chat box, no Send button. The
    // question still appears in the chat thread, with the full written answer.
    markNextQuestionAsVoice();
    pendingRef.current = true;
    aui.thread.append({ content: [{ type: "text", text }] });
  }

  // The answer arrived: read it aloud.
  useEffect(() => {
    if (!lastResult || lastResult === seenResultRef.current) return;
    seenResultRef.current = lastResult;
    if (!pendingRef.current) return;
    pendingRef.current = false;
    const spoken = lastResult.spoken_text || cleanForSpeech(lastResult.answer || "");
    setReply(spoken);
    setPhase("speaking");
    speak(spoken, {
      onEnd: () => {
        if (mountedRef.current) setPhase((p) => (p === "speaking" ? "idle" : p));
      },
    });
  }, [lastResult]);

  // The request failed (network error / timeout / server error): the chat
  // thread shows the error on the reply; here, say so and let the visitor try
  // again. Watched through the reply's own status, because a fast failure can
  // start and stop the run before a render ever sees it running.
  useEffect(() => {
    if (!failedReplyId || !pendingRef.current) return;
    pendingRef.current = false;
    setPhase("error");
    setError("Couldn't reach the assistant. Check your connection and tap the mic to try again.");
  }, [failedReplyId]);

  // Start listening straight away (entering voice mode was the tap), and clean
  // everything up when leaving.
  useEffect(() => {
    mountedRef.current = true;
    void startListening();
    return () => {
      mountedRef.current = false;
      discardRef.current = true;
      const recorder = recorderRef.current;
      if (recorder && recorder.state !== "inactive") recorder.stop();
      releaseMic();
      stopSpeaking();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function exit() {
    discardRef.current = true;
    stopListening();
    stopSpeaking();
    onExit();
  }

  // Keyboard: focus the main button, and Escape goes back to the chat.
  useEffect(() => {
    mainButtonRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") exit();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function mainAction() {
    if (phase === "listening") stopListening();
    // While it's speaking, a tap interrupts and listens for the next question.
    else if (phase === "speaking" || phase === "idle" || phase === "error") void startListening();
  }

  const busy = phase === "starting" || phase === "transcribing" || phase === "thinking";
  const mainLabel =
    phase === "listening" ? "Stop and send" : phase === "speaking" ? "Interrupt and ask another question" : "Ask a question";

  return (
    <div
      className="voice-mode absolute inset-0 z-30 flex flex-col text-white"
      role="dialog"
      aria-label="Voice conversation"
    >
      {/* Top bar */}
      <div className="flex items-center px-3 pt-3">
        <button
          type="button"
          onClick={exit}
          aria-label="Back to chat"
          className="flex items-center gap-1.5 rounded-full px-3 py-2 text-[13px] font-medium text-white/85 transition-colors hover:bg-white/10 hover:text-white"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to chat
        </button>
      </div>

      {/* Orb + status */}
      <div className="flex flex-1 flex-col items-center justify-center px-6">
        <div className="relative flex h-44 w-44 items-center justify-center">
          <div className={`voice-halo ${phase === "thinking" || phase === "transcribing" ? "is-thinking" : ""}`} />
          {phase === "listening" && (
            <>
              <span className="voice-ring" />
              <span className="voice-ring" style={{ animationDelay: "1s" }} />
            </>
          )}
          <div
            ref={orbRef}
            className={`voice-orb ${phase === "speaking" ? "is-speaking" : ""} ${
              phase === "thinking" || phase === "transcribing" ? "is-thinking" : ""
            } ${phase === "idle" || phase === "error" ? "is-idle" : ""}`}
          />
        </div>

        <p aria-live="polite" className="mt-3 min-h-[1.75rem] text-center text-lg font-semibold tracking-tight">
          {phase === "error" ? "Something went wrong" : STATUS[phase]}
        </p>
        {phase === "listening" && !autoStop && (
          <p className="mt-1 text-center text-[13px] text-white/70">Tap the red button when you've finished.</p>
        )}
        {error && <p className="mt-1 max-w-[18rem] text-center text-[13px] leading-snug text-white/75">{error}</p>}

        {/* Captions */}
        <div className="mt-3 w-full max-w-[20rem] space-y-2 text-center">
          {heard && (
            <p className="text-[13px] leading-snug text-white/60">
              <span className="font-semibold text-white/80">You: </span>
              {heard}
            </p>
          )}
          {reply && (phase === "speaking" || phase === "idle") && (
            <p className="consulate-scroll max-h-24 overflow-y-auto text-[14px] leading-relaxed text-white/90">{reply}</p>
          )}
        </div>
        {reply && phase === "idle" && (
          <button
            type="button"
            onClick={exit}
            className="mt-3 flex items-center gap-1.5 rounded-full border border-white/25 px-3.5 py-1.5 text-[12.5px] font-medium text-white/85 transition-colors hover:bg-white/10"
          >
            <MessageSquareText className="h-3.5 w-3.5" />
            See the full answer and sources in chat
          </button>
        )}
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-8 px-6 pt-2 pb-[calc(1.5rem+env(safe-area-inset-bottom))]">
        <button
          type="button"
          onClick={exit}
          aria-label="Type instead"
          title="Type instead"
          className="flex h-12 w-12 items-center justify-center rounded-full bg-white/10 text-white transition-colors hover:bg-white/20"
        >
          <Keyboard className="h-5 w-5" />
        </button>
        <button
          type="button"
          ref={mainButtonRef}
          onClick={mainAction}
          disabled={busy}
          aria-label={mainLabel}
          title={mainLabel}
          className={`flex h-[72px] w-[72px] items-center justify-center rounded-full shadow-lg transition-all duration-150 active:scale-95 disabled:cursor-default ${
            phase === "listening" ? "bg-red-500 text-white hover:bg-red-600" : "bg-white text-[var(--brand-blue-dark)] hover:bg-white/90"
          }`}
        >
          {busy ? (
            <Loader2 className="h-7 w-7 animate-spin" />
          ) : phase === "listening" ? (
            <Square className="h-6 w-6 fill-current" />
          ) : (
            <Mic className="h-7 w-7" />
          )}
        </button>
        <button
          type="button"
          onClick={exit}
          aria-label="End voice"
          title="End voice"
          className="flex h-12 w-12 items-center justify-center rounded-full bg-white/10 text-white transition-colors hover:bg-red-500/80"
        >
          <X className="h-5 w-5" />
        </button>
      </div>
    </div>
  );
}
