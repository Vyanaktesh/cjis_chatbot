"""The /transcribe content-type check must accept real browser recordings,
whose MIME types carry codec parameters (Chrome: audio/webm;codecs=opus,
Safari: audio/mp4), while still rejecting non-audio uploads."""
import os

os.environ.setdefault("WARMUP_ON_STARTUP", "false")

from app.api.main import _audio_type_accepted  # noqa: E402


def test_browser_recordings_with_codec_params_are_accepted():
    assert _audio_type_accepted("audio/webm;codecs=opus")
    assert _audio_type_accepted("audio/ogg; codecs=vorbis")
    assert _audio_type_accepted("audio/webm")
    assert _audio_type_accepted("audio/mp4")
    assert _audio_type_accepted("AUDIO/WEBM")  # case-insensitive


def test_missing_type_is_allowed_ffmpeg_detects_it():
    assert _audio_type_accepted(None)
    assert _audio_type_accepted("")


def test_non_audio_uploads_are_rejected():
    assert not _audio_type_accepted("text/plain")
    assert not _audio_type_accepted("image/png")
    assert not _audio_type_accepted("application/json")
