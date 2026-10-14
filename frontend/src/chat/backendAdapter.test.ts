import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  createBackendAdapter,
  markNextQuestionAsVoice,
  type ChatResponse,
} from "./backendAdapter";

type RunArgs = Parameters<ReturnType<typeof createBackendAdapter>["run"]>[0];

function runArgs(abortSignal: AbortSignal = new AbortController().signal): RunArgs {
  return {
    messages: [{ role: "user", content: [{ type: "text", text: "How do I renew my passport?" }] }],
    abortSignal,
  } as unknown as RunArgs;
}

function jsonResponse(body: Partial<ChatResponse>, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const ok = (over: Partial<ChatResponse> = {}): Partial<ChatResponse> => ({
  session_id: "s1",
  query: "q",
  answer: "Renew online [1].",
  grounded: true,
  citations: [],
  retrieved_count: 1,
  ...over,
});

async function failure(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (err) {
    return err;
  }
  throw new Error("expected the adapter to reject");
}

describe("backendAdapter.run", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  const fetchMock = () => globalThis.fetch as unknown as ReturnType<typeof vi.fn>;
  const run = (signal?: AbortSignal) => createBackendAdapter().run(runArgs(signal)) as Promise<unknown>;

  it("returns the answer text and a source part per citation", async () => {
    fetchMock().mockResolvedValue(
      jsonResponse(
        ok({
          citations: [
            { index: 1, chunk_id: "c1", source_url: "https://example.gov/a", service_category: "oci", canonical: true },
          ],
        }),
      ),
    );
    const onResult = vi.fn();
    const result = (await createBackendAdapter(onResult).run(runArgs())) as { content: { type: string; text?: string; title?: string }[] };

    expect(result.content[0]).toEqual({ type: "text", text: "Renew online [1]." });
    expect(result.content[1]).toMatchObject({ type: "source", title: "[1] OCI source" });
    expect(onResult).toHaveBeenCalledOnce();
  });

  it("puts the source's readable title in the chip title when the backend sends one", async () => {
    fetchMock().mockResolvedValue(
      jsonResponse(
        ok({
          citations: [
            {
              index: 1, chunk_id: "c1", source_url: "https://indiainatlanta.gov.in/eoial_pages/MjY1",
              service_category: "oci", canonical: true, source_title: "FAQ's on OCI related matters",
            },
          ],
        }),
      ),
    );
    const result = (await createBackendAdapter().run(runArgs())) as unknown as { content: { type: string; title?: string }[] };
    expect(result.content[1]).toMatchObject({ type: "source", title: "[1] OCI FAQ's on OCI related matters" });
  });

  it("explains an over-long message without calling the server", async () => {
    const args = {
      ...runArgs(),
      messages: [{ role: "user", content: [{ type: "text", text: "x".repeat(2500) }] }],
    } as unknown as RunArgs;
    await expect(createBackendAdapter().run(args) as Promise<unknown>).rejects.toThrow(
      /too long \(2,500 characters\)/,
    );
    expect(fetchMock()).not.toHaveBeenCalled();
  });

  it("gives a 422 a 'shorten it' message, not 'couldn't be reached'", async () => {
    fetchMock().mockResolvedValue(jsonResponse({}, 422));
    await expect(run()).rejects.toThrow(/couldn't be processed/);
  });

  it("tells the user to slow down on a 429 (and shows no 'Error:' prefix)", async () => {
    fetchMock().mockResolvedValue(jsonResponse({}, 429));
    const err = await failure(run());
    expect(String(err)).toBe("You're sending messages too quickly. Please wait a minute and try again.");
  });

  it("reports the HTTP status for other server errors", async () => {
    fetchMock().mockResolvedValue(new Response("boom", { status: 500 }));
    expect(String(await failure(run()))).toContain("HTTP 500");
  });

  it("explains a network failure", async () => {
    fetchMock().mockRejectedValue(new TypeError("Failed to fetch"));
    expect(String(await failure(run()))).toContain("Couldn't reach the assistant");
  });

  it("explains an unreadable response", async () => {
    fetchMock().mockResolvedValue(new Response("<html>gateway error</html>", { status: 200 }));
    expect(String(await failure(run()))).toContain("couldn't be read");
  });

  it("shows a fallback instead of an empty bubble when the answer is blank", async () => {
    fetchMock().mockResolvedValue(jsonResponse(ok({ answer: "   " })));
    const result = (await run()) as { content: { text?: string }[] };
    expect(result.content[0].text).toContain("couldn't put an answer together");
  });

  it("gives up after 90 seconds instead of waiting forever", async () => {
    vi.useFakeTimers();
    fetchMock().mockImplementation(
      (_url: string, init: RequestInit) =>
        new Promise((_resolve, reject) => {
          init.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
        }),
    );
    const pending = failure(run());
    await vi.advanceTimersByTimeAsync(90_000);
    expect(String(await pending)).toContain("taking longer than expected");
  });

  const bodyOf = (call: number) =>
    JSON.parse((fetchMock().mock.calls[call][1] as RequestInit).body as string);

  it("asks for text mode by default", async () => {
    fetchMock().mockResolvedValue(jsonResponse(ok()));
    await run();
    expect(bodyOf(0).mode).toBe("text");
  });

  it("asks for voice mode once after the mic marks a question, then reverts", async () => {
    // Fresh Response per call -- a body can only be read once.
    fetchMock().mockImplementation(() => Promise.resolve(jsonResponse(ok({ spoken_text: "Renew online." }))));
    markNextQuestionAsVoice();
    await run();
    await run();
    expect(bodyOf(0).mode).toBe("voice");
    expect(bodyOf(1).mode).toBe("text"); // one-shot: typed questions stay text
  });

  it("lets a user cancel through without turning it into an error message", async () => {
    fetchMock().mockImplementation(
      (_url: string, init: RequestInit) =>
        new Promise((_resolve, reject) => {
          init.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
        }),
    );
    const userCancel = new AbortController();
    const pending = failure(run(userCancel.signal));
    userCancel.abort();
    const err = (await pending) as Error;
    expect(err.name).toBe("AbortError");
  });
});
