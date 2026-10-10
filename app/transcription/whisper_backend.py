"""
Self-hosted speech-to-text via whisper.cpp (pywhispercpp bindings).

Nothing leaves this machine -- the same self-hosted principle as the Qwen
generation path. The ggml model is downloaded once (by pywhispercpp, into its
own cache) on first use and then loaded into memory and kept there.

Browsers record audio as webm/opus (MediaRecorder); whisper.cpp wants 16 kHz
mono PCM, so the API layer converts with ffmpeg before calling transcribe()
here -- this module only ever sees a ready 16 kHz mono WAV path.

The model is a process-wide singleton loaded lazily on first transcription,
following the same class-level-instance pattern as
app/generation/gemini_backend.py's client. whisper.cpp's context is not safe
to run two transcriptions through at once, so a lock serializes them -- fine
for a single-box demo; revisit if transcription volume ever grows.
"""

import threading

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class WhisperTranscriber:
    _model = None
    _model_size: str | None = None
    _load_lock = threading.Lock()
    _transcribe_lock = threading.Lock()

    def __init__(self) -> None:
        settings = get_settings()
        size = settings.whisper_model_size
        # Double-checked locking: load once, even if two requests race on the
        # very first call before the model exists.
        if WhisperTranscriber._model is None or WhisperTranscriber._model_size != size:
            with WhisperTranscriber._load_lock:
                if WhisperTranscriber._model is None or WhisperTranscriber._model_size != size:
                    from pywhispercpp.model import Model

                    logger.info(f"loading whisper.cpp model (size={size}); first run downloads it")
                    WhisperTranscriber._model = Model(size, print_realtime=False, print_progress=False)
                    WhisperTranscriber._model_size = size
                    logger.info("whisper.cpp model loaded")
        self._model = WhisperTranscriber._model

    def transcribe_wav(self, wav_path: str) -> str:
        """Transcribe a 16 kHz mono WAV file to plain text."""
        with WhisperTranscriber._transcribe_lock:
            segments = self._model.transcribe(wav_path)
        return " ".join(seg.text.strip() for seg in segments).strip()
