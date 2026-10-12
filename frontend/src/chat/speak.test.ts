import { describe, expect, it } from "vitest";
import { cleanForSpeech } from "./speak";

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
