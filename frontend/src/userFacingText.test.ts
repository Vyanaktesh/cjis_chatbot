import { describe, expect, it } from "vitest";
import indexHtml from "../index.html?raw";

// Text users read must have no em dashes or en dashes (including the HTML
// entities). Comments are ignored; only code and visible text are checked.
const DASH = /—|–|&mdash;|&ndash;/;

const sources = import.meta.glob(["./**/*.{ts,tsx}", "!./**/*.test.ts"], {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

function withoutComments(code: string): string {
  return code
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/(^|[^:])\/\/.*$/gm, "$1");
}

describe("user-facing text", () => {
  it("scans the component files", () => {
    expect(Object.keys(sources).length).toBeGreaterThan(3);
  });

  it("has no em or en dashes in any component", () => {
    const offenders = Object.entries(sources).flatMap(([file, code]) =>
      withoutComments(code)
        .split("\n")
        .filter((line) => DASH.test(line))
        .map((line) => `${file}: ${line.trim()}`),
    );
    expect(offenders).toEqual([]);
  });

  it("has no em or en dash in the page title", () => {
    expect(indexHtml).not.toMatch(DASH);
  });
});
