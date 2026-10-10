/** Returns the URL only if it is a plain http(s) link, otherwise undefined.
 * Citation links come from the API; a `javascript:` or `data:` value must
 * never reach an `href`. */
export function safeUrl(value: string | undefined | null): string | undefined {
  if (!value) return undefined;
  try {
    const parsed = new URL(value);
    return parsed.protocol === "http:" || parsed.protocol === "https:"
      ? parsed.toString()
      : undefined;
  } catch {
    return undefined;
  }
}
