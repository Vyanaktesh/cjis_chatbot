"""
Optional generation backend: Google's Gemini API (Phase 7 follow-up).

Real trade-off, surfaced and accepted, not hidden: the original Phase 7
design was self-hosted-only (Qwen3 via llama.cpp) specifically so no
question or retrieved consular-document excerpt ever left this machine.
CPU-only Qwen3-4B latency (30s-150s/answer, see LOCAL_SETUP.md) proved too
slow for real use, so this module adds Gemini as a *selectable* backend
(`GENERATION_BACKEND=gemini` in .env) for whoever wants the speed and is
okay with that trade-off. Qwen remains the default
(`GENERATION_BACKEND=qwen`, or simply unset) and is unaffected by any of
this -- nothing here changes qwen_backend.py's behavior.

Requires a free Gemini API key (https://aistudio.google.com/apikey) set as
GEMINI_API_KEY in .env. Uses the `google-genai` package (the actively
maintained SDK; the older `google-generativeai` package is deprecated).
"""

import time
from typing import Any, Optional

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Free-tier Gemini API quotas are low (a handful of requests/minute) and get
# hit fast during real testing -- a 429 here is expected, routine traffic,
# not a broken system. A short exponential backoff retry resolves most of
# them transparently since per-minute quotas reset quickly.
_RATE_LIMIT_RETRY_DELAYS = (2, 5, 10)  # seconds


class GenerationUnavailable(Exception):
    """Raised when the Gemini backend cannot produce an answer after
    retries (rate-limited, or Google's API erroring/unreachable). Caught in
    app/generation/service.py so this degrades to a clear message instead
    of an unhandled 500 -- same "fail honestly, don't crash" principle as
    NO_CONTEXT_ANSWER elsewhere in this module's caller."""


class GeminiGenerator:
    """Same generate(messages) -> str interface as QwenGenerator (see
    qwen_backend.py), so app/generation/service.py can swap backends
    without caring which one it's holding."""

    _client = None  # class-level: one client per process is enough, no local model to load

    def __init__(self):
        if GeminiGenerator._client is None:
            from google import genai
            from google.genai import types as genai_types

            settings = get_settings()
            if not settings.gemini_api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set in .env, but GENERATION_BACKEND=gemini. "
                    "Get a free key from https://aistudio.google.com/apikey and add it "
                    "to your .env file (see .env.example)."
                )
            logger.info(f"initializing Gemini client (model={settings.gemini_model})")
            # The SDK's default is NO timeout: one stalled call to Google would
            # hold a worker thread indefinitely, and enough of them freeze the
            # whole API. The SDK takes milliseconds.
            GeminiGenerator._client = genai.Client(
                api_key=settings.gemini_api_key,
                http_options=genai_types.HttpOptions(timeout=int(settings.gemini_timeout_seconds * 1000)),
            )
        self._client = GeminiGenerator._client

    def generate(
        self,
        messages: list[dict],
        *,
        max_tokens: Optional[int] = None,
        temperature: float = 0.2,
    ) -> str:
        from google.genai import errors, types

        settings = get_settings()
        system_instruction, contents = _to_gemini_contents(messages)
        config_kwargs: dict[str, Any] = dict(
            system_instruction=system_instruction,
            max_output_tokens=max_tokens or settings.gemini_max_output_tokens,
            temperature=temperature,
        )
        # thinking_budget controls the model's internal "thinking" pass. On
        # Gemini 2.5 Flash, budget=0 disables it (it otherwise competes with the
        # visible answer for max_output_tokens). On Gemini 3.x models, budget=0
        # is an INVALID_ARGUMENT (400) and the lite models are already fast
        # without any thinking, so a NEGATIVE budget here means "omit
        # thinking_config entirely and let the model use its default" -- the
        # correct choice for the 3.x lite default. Set it to 0 only if you
        # switch back to a 2.5 model.
        if settings.gemini_thinking_budget >= 0:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=settings.gemini_thinking_budget
            )
        config = types.GenerateContentConfig(**config_kwargs)

        attempts = len(_RATE_LIMIT_RETRY_DELAYS) + 1
        last_error: Optional[Exception] = None
        # Per-call timeout x retries could otherwise add up to well over a
        # minute while the visitor waits; stop retrying once the budget is spent.
        deadline = time.monotonic() + settings.gemini_total_budget_seconds
        for attempt in range(attempts):
            try:
                response = self._client.models.generate_content(
                    model=settings.gemini_model, contents=contents, config=config
                )
                break
            except errors.APIError as exc:
                last_error = exc
                # Only retry on transient conditions: 429 (rate limit/quota,
                # by far the most common in free-tier testing) and 5xx
                # (Google's side erroring, also usually transient). Anything
                # else (400 bad request, 401/403 auth) won't be fixed by
                # waiting, so fail fast instead of wasting ~17s retrying.
                is_retryable = exc.code == 429 or (exc.code is not None and exc.code >= 500)
                if not is_retryable or attempt >= len(_RATE_LIMIT_RETRY_DELAYS):
                    logger.warning(f"Gemini API error (code={exc.code}, status={exc.status}): {exc.message}")
                    raise GenerationUnavailable(
                        f"Gemini API error (code={exc.code}): {exc.message or exc}"
                    ) from exc
                delay = _RATE_LIMIT_RETRY_DELAYS[attempt]
                if time.monotonic() + delay >= deadline:
                    logger.warning(f"Gemini API error (code={exc.code}); retry budget spent, giving up")
                    raise GenerationUnavailable(
                        f"Gemini API error (code={exc.code}) and the retry time budget is spent"
                    ) from exc
                logger.warning(
                    f"Gemini API error (code={exc.code}), retrying in {delay}s "
                    f"(attempt {attempt + 1}/{len(_RATE_LIMIT_RETRY_DELAYS)})..."
                )
                time.sleep(delay)
        else:
            # Loop exhausted without a `break` -- shouldn't normally happen
            # since the last iteration raises instead, but fail safely.
            raise GenerationUnavailable(f"Gemini API unavailable after retries: {last_error}")

        if not response.text:
            finish_reason = response.candidates[0].finish_reason if response.candidates else None
            logger.warning(
                f"Gemini returned no visible text (finish_reason={finish_reason}); "
                "if this is MAX_TOKENS, gemini_max_output_tokens may need to be raised further."
            )
        return response.text or ""


def _to_gemini_contents(messages: list[dict[str, Any]]) -> tuple[Optional[str], list[dict]]:
    """Gemini's API takes the system prompt as a separate `system_instruction`
    (not a message in the list) and uses role "model" for assistant turns
    instead of "assistant" -- convert from the OpenAI-style
    {"role": ..., "content": ...} list that app/generation/prompt.py builds
    (shared with the Qwen backend, so no changes needed there)."""
    system_instruction = None
    contents = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            # prompt.py only ever builds one system message, but concatenate
            # rather than silently drop instructions if that ever changes.
            system_instruction = (
                f"{system_instruction}\n\n{content}" if system_instruction else content
            )
            continue
        gemini_role = "model" if role == "assistant" else "user"
        if contents and contents[-1]["role"] == gemini_role:
            # Two turns in a row from the same side (e.g. an earlier question
            # whose answer failed, followed by the new one): the API expects
            # alternating turns, so fold them into one turn with two parts.
            contents[-1]["parts"].append({"text": content})
        else:
            contents.append({"role": gemini_role, "parts": [{"text": content}]})
    return system_instruction, contents
