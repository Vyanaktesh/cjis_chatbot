"""
Self-hosted generation backend: Qwen3 via llama.cpp (llama-cpp-python),
running a quantized GGUF file entirely on CPU — no network call, no
third-party API, satisfying the project's self-hostable requirement.

Model size note: the project brief specified Qwen3-8B. This build
environment's ~2 CPU cores / ~8GB RAM made an 8B model (~5GB file,
llama.cpp resident memory close to file size) too tight to reliably
verify alongside Postgres + Qdrant already running, so Qwen3-4B
(~2.5GB) is what's actually downloaded and tested here — a deliberate,
documented substitution, not a silent downgrade. Swapping back to 8B (or
anything larger) on real deployment hardware is a one-line config change
(`GENERATION_MODEL_PATH`), nothing in this module is 4B-specific.
"""

import threading
from typing import Optional

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# The underlying llama_cpp.Llama instance is shared (class-level singleton,
# see QwenGenerator._llm below) across every request in this process. It is
# NOT safe to call create_chat_completion() on it from more than one thread
# at a time -- concurrent calls (e.g. two chat turns sent close together, or
# two browser tabs) can hang or return garbled/interleaved output. This lock
# serializes every generation call so concurrent requests queue instead of
# racing.
_generation_lock = threading.Lock()


class QwenGenerator:
    _llm = None  # class-level: loading a GGUF file is expensive, share across instances in one process

    def __init__(self):
        if QwenGenerator._llm is None:
            from llama_cpp import Llama

            settings = get_settings()
            logger.info(f"loading generation model from {settings.generation_model_path} (first call only)...")
            QwenGenerator._llm = Llama(
                model_path=settings.generation_model_path,
                n_ctx=settings.generation_n_ctx,
                n_threads=settings.generation_n_threads,
                verbose=False,
            )
        self._llm = QwenGenerator._llm

    def generate(
        self,
        messages: list[dict],
        *,
        max_tokens: Optional[int] = None,
        temperature: float = 0.2,
    ) -> str:
        settings = get_settings()
        # Serialize access to the shared Llama instance -- see _generation_lock
        # comment above. Without this, two concurrent requests calling
        # create_chat_completion() on the same self._llm can hang or produce
        # garbled/interleaved output.
        with _generation_lock:
            out = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens or settings.generation_max_tokens,
                temperature=temperature,
            )
        return out["choices"][0]["message"]["content"]
