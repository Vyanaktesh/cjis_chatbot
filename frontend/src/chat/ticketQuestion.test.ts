import { describe, expect, it } from "vitest";
import { lastRealQuestion } from "./ticketQuestion";

describe("lastRealQuestion", () => {
  it("skips the 'more details' click and returns the citizen's question", () => {
    expect(lastRealQuestion(["How do I renew my passport?", "Yes, more details please"])).toBe(
      "How do I renew my passport?",
    );
    expect(lastRealQuestion(["How do I renew my passport?", "yes", "tell me more"])).toBe(
      "How do I renew my passport?",
    );
  });

  it("returns the newest real question when there are several", () => {
    expect(lastRealQuestion(["OCI fees?", "yes", "And for a child?"])).toBe("And for a child?");
  });

  it("falls back to the last message, or empty", () => {
    expect(lastRealQuestion(["yes"])).toBe("yes");
    expect(lastRealQuestion([])).toBe("");
  });
});
