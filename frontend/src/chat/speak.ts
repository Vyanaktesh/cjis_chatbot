/**
 * Text-to-speech behind a small interface, so a server TTS can replace the
 * browser engine later without touching callers. For now it wraps the browser's
 * built-in speechSynthesis (free, on-device, no server cost).
 *
 * iOS note: Safari only allows speech that was unlocked by a user gesture.
 * warmUpSpeech() is called from the mic tap (a gesture) so later speak() calls
 * in the same session are allowed.
 */

export type SpeakOptions = { rate?: number; onEnd?: () => void };

export function speechSupported(): boolean {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

/** Unlock speech on iOS. Call from inside a user-gesture handler (e.g. the mic
 * tap). Speaks a silent utterance (which also kicks off async voice loading)
 * so later speak() calls are permitted and have voices ready. */
export function warmUpSpeech(): void {
  if (!speechSupported()) return;
  try {
    window.speechSynthesis.getVoices(); // trigger async voice loading
    const u = new SpeechSynthesisUtterance(" ");
    u.volume = 0;
    window.speechSynthesis.speak(u);
    window.speechSynthesis.cancel();
  } catch {
    // best-effort; ignore
  }
}

function pickEnglishVoice(): SpeechSynthesisVoice | null {
  try {
    const voices = window.speechSynthesis.getVoices();
    if (!voices.length) return null;
    return voices.find((v) => /^en\b/i.test(v.lang)) ?? voices[0];
  } catch {
    return null;
  }
}

export function speak(text: string, opts: SpeakOptions = {}): void {
  if (!speechSupported() || !text.trim()) {
    opts.onEnd?.();
    return;
  }

  // onEnd exactly once: after the last sentence, on an error/cancel, or from
  // the watchdog below.
  let ended = false;
  let watchdog: ReturnType<typeof setTimeout> | undefined;
  const finish = () => {
    if (ended) return;
    ended = true;
    clearTimeout(watchdog);
    opts.onEnd?.();
  };

  const utter = () => {
    try {
      window.speechSynthesis.cancel(); // never overlap utterances
      // One utterance per sentence: Chrome silently stops a single utterance
      // after ~15 seconds and never fires onend.
      const parts = splitSentences(text);
      const voice = pickEnglishVoice();
      parts.forEach((part, i) => {
        const u = new SpeechSynthesisUtterance(part);
        if (voice) u.voice = voice;
        u.rate = opts.rate ?? 1;
        u.lang = "en-US";
        if (i === parts.length - 1) u.onend = finish;
        u.onerror = finish;
        window.speechSynthesis.speak(u);
      });
      // If the engine never reports the end (it happens), don't leave the
      // caller stuck in "speaking": allow ~2.5 words a second plus a margin.
      const words = text.split(/\s+/).length;
      watchdog = setTimeout(finish, (words / 2.5) * 1000 * (1 / (opts.rate ?? 1)) + 8000);
    } catch {
      finish();
    }
  };

  // Voices load asynchronously; on a fresh page getVoices() can be empty, and
  // speaking then produces no sound. Wait for them (once), with a short timeout
  // fallback so we never hang if the event never fires.
  if (window.speechSynthesis.getVoices().length > 0) {
    utter();
    return;
  }
  let fired = false;
  const go = () => {
    if (fired) return;
    fired = true;
    window.speechSynthesis.removeEventListener("voiceschanged", go);
    utter();
  };
  window.speechSynthesis.addEventListener("voiceschanged", go);
  setTimeout(go, 500);
}

/** Split text into sentences for speaking; very long sentences are further
 * split at commas so no single utterance runs much past ~10 seconds. */
export function splitSentences(text: string): string[] {
  // Split only at sentence punctuation followed by a space, so "$41.00" and
  // "U.S." stay whole.
  const sentences = text.split(/(?<=[.!?])\s+/);
  const out: string[] = [];
  for (const s of sentences.map((x) => x.trim()).filter(Boolean)) {
    if (s.split(/\s+/).length <= 30) {
      out.push(s);
      continue;
    }
    let cur = "";
    for (const piece of s.split(/(?<=,)\s+/)) {
      if (cur && (cur + " " + piece).split(/\s+/).length > 30) {
        out.push(cur);
        cur = piece;
      } else cur = cur ? `${cur} ${piece}` : piece;
    }
    if (cur) out.push(cur);
  }
  return out.length ? out : [text];
}

export function stopSpeaking(): void {
  if (!speechSupported()) return;
  try {
    window.speechSynthesis.cancel();
  } catch {
    // ignore
  }
}

/** Clean arbitrary answer text for reading aloud: drop [n] citation markers,
 * URLs and basic markdown, collapse whitespace. Mirrors the backend's
 * strip_for_speech so the "Read full answer" control sounds right too. */
export function cleanForSpeech(text: string): string {
  return text
    .replace(/\s*\[\d+\]/g, "")
    .replace(/\bhttps?:\/\/\S+/gi, "")
    .replace(/\(\s*\)/g, "")
    .replace(/[*_`#]+/g, "")
    .replace(/\s*\n+\s*/g, ". ")
    .replace(/\s{2,}/g, " ")
    .trim();
}
