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

  const utter = () => {
    try {
      window.speechSynthesis.cancel(); // never overlap utterances
      const u = new SpeechSynthesisUtterance(text);
      const voice = pickEnglishVoice();
      if (voice) u.voice = voice;
      u.rate = opts.rate ?? 1;
      u.lang = "en-US";
      u.onend = () => opts.onEnd?.();
      u.onerror = () => opts.onEnd?.();
      window.speechSynthesis.speak(u);
    } catch {
      opts.onEnd?.();
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
