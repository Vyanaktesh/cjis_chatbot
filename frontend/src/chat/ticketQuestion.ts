// Replies that just accept the "Would you like more details on this?" offer
// ("yes", "tell me more", the button's own text). Mirrors
// is_more_details_request in app/generation/prompt.py.
const ACCEPT_WORD =
  "(?:yes|yeah|yep|yup|ya|sure|ok(?:ay)?|please|go ahead|tell me|show me|give me|more|full|details?|info(?:rmation)?|explain|thanks?|thank you)";
const ACCEPTS_MORE_RE = new RegExp(`^\\s*${ACCEPT_WORD}(?:[\\s,]+${ACCEPT_WORD})*[\\s!.,?]*$`, "i");

/** The question to pre-fill in "Raise a query": the newest user message that
 * is an actual question, skipping "Yes, more details please" and the like --
 * the consulate needs the citizen's question, not the click that followed it. */
export function lastRealQuestion(userTexts: string[]): string {
  for (let i = userTexts.length - 1; i >= 0; i--) {
    const text = userTexts[i].trim();
    if (text && !ACCEPTS_MORE_RE.test(text)) return text;
  }
  return userTexts.at(-1)?.trim() ?? "";
}
