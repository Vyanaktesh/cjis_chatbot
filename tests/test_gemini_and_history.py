import pytest
from google.genai import errors

from app.core.config import Settings
from app.generation import gemini_backend
from app.generation.gemini_backend import GenerationUnavailable, GeminiGenerator, _to_gemini_contents
from app.generation.prompt import _trim_history


@pytest.fixture()
def fresh_gemini(monkeypatch):
    settings = Settings(_env_file=None, gemini_api_key="test-key", gemini_timeout_seconds=7, gemini_total_budget_seconds=1)
    monkeypatch.setattr(gemini_backend, "get_settings", lambda: settings)
    monkeypatch.setattr(GeminiGenerator, "_client", None)
    return settings


def test_the_gemini_client_has_a_request_timeout(fresh_gemini):
    generator = GeminiGenerator()
    # The SDK's default is no timeout at all; the setting is seconds, the SDK wants milliseconds.
    assert generator._client._api_client._http_options.timeout == 7000


def test_rate_limit_retries_stop_once_the_time_budget_is_spent(fresh_gemini, monkeypatch):
    sleeps = []
    monkeypatch.setattr(gemini_backend.time, "sleep", sleeps.append)

    class AlwaysRateLimited:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                raise errors.APIError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}})

    generator = GeminiGenerator()
    generator._client = AlwaysRateLimited
    with pytest.raises(GenerationUnavailable):
        generator.generate([{"role": "system", "content": "s"}, {"role": "user", "content": "q"}])
    assert sleeps == []  # the first retry delay (2s) already exceeds the 1s budget, so no waiting


def test_back_to_back_turns_from_one_side_are_merged_for_gemini():
    _, contents = _to_gemini_contents(
        [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "first question whose answer failed"},
            {"role": "user", "content": "retry with sources"},
            {"role": "assistant", "content": "answer"},
        ]
    )
    assert [c["role"] for c in contents] == ["user", "model"]
    assert [p["text"] for p in contents[0]["parts"]] == ["first question whose answer failed", "retry with sources"]


def test_history_never_starts_with_an_assistant_turn():
    history = [
        {"role": "assistant", "content": "orphaned answer"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    assert [m["role"] for m in _trim_history(history)] == ["user", "assistant"]


def test_consecutive_turns_from_the_same_speaker_are_merged_in_history():
    history = [
        {"role": "user", "content": "first try"},
        {"role": "user", "content": "second try"},
        {"role": "assistant", "content": "answer"},
    ]
    trimmed = _trim_history(history)
    assert [m["role"] for m in trimmed] == ["user", "assistant"]
    assert trimmed[0]["content"] == "first try\nsecond try"
