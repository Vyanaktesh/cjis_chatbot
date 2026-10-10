import { describe, expect, it } from "vitest";
import { safeUrl } from "./safeUrl";

describe("safeUrl", () => {
  it("keeps http and https links", () => {
    expect(safeUrl("https://www.indiainatlanta.gov.in/oci")).toBe(
      "https://www.indiainatlanta.gov.in/oci",
    );
    expect(safeUrl("http://example.org/a")).toBe("http://example.org/a");
  });

  it("drops script and data schemes", () => {
    expect(safeUrl("javascript:alert(1)")).toBeUndefined();
    expect(safeUrl("JaVaScRiPt:alert(1)")).toBeUndefined();
    expect(safeUrl("data:text/html,<script>alert(1)</script>")).toBeUndefined();
  });

  it("drops empty or malformed values", () => {
    expect(safeUrl(undefined)).toBeUndefined();
    expect(safeUrl("")).toBeUndefined();
    expect(safeUrl("not a url")).toBeUndefined();
  });
});
