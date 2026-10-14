import { describe, expect, it } from "vitest";
import { cleanForSpeech, splitSentences } from "./speak";

describe("cleanForSpeech", () => {
  it("strips citation markers, URLs and markdown for reading aloud", () => {
    const out = cleanForSpeech(
      "Fees are **$100** [1][2]. See https://vfs.example/oci.pdf for the form.\n- step one",
    );
    expect(out).not.toContain("[1]");
    expect(out).not.toContain("[2]");
    expect(out).not.toContain("http");
    expect(out).not.toContain("**");
    expect(out).not.toContain("\n");
    expect(out).toContain("$100"); // the fee survives
  });

  it("keeps fees and deadlines intact", () => {
    const out = cleanForSpeech("The fee is $275 and the deadline is 30 days [1].");
    expect(out).toContain("$275");
    expect(out).toContain("30 days");
  });
});

describe("splitSentences", () => {
  it("splits at sentence ends but keeps amounts like $41.00 whole", () => {
    expect(splitSentences("The fee is $41.00 per executant. Pay by money order! Is that clear?")).toEqual([
      "The fee is $41.00 per executant.",
      "Pay by money order!",
      "Is that clear?",
    ]);
  });

  it("breaks a very long sentence at commas so no utterance runs too long", () => {
    const long = Array.from({ length: 8 }, (_, i) => `clause number ${i} has a few more words`).join(", ") + ".";
    const parts = splitSentences(long);
    expect(parts.length).toBeGreaterThan(1);
    for (const p of parts) expect(p.split(/\s+/).length).toBeLessThanOrEqual(30);
    expect(parts.join(" ")).toBe(long);
  });
});
