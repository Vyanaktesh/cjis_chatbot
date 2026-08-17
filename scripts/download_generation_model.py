#!/usr/bin/env python3
"""
Phase 7: downloads the self-hosted generation model (a quantized GGUF)
from Hugging Face Hub into data/models/, so `app/generation/qwen_backend.py`
has something to load. One-time (or per-model-change) setup step, not run
on every request.

Model choice: Qwen3-4B-Q4_K_M (~2.5GB), NOT the project brief's originally
specified Qwen3-8B — this build sandbox's ~2 CPU cores / ~8GB RAM made an
8B GGUF (~5GB file, llama.cpp resident memory close to file size) too
tight to reliably verify alongside Postgres + Qdrant already running.
This is a deliberate, user-confirmed trade-off, not a silent downgrade —
see README's Phase 7 section. To use 8B (or any other Qwen3 GGUF) instead,
change REPO_ID/FILENAME here and GENERATION_MODEL_PATH in .env/config.py.

Usage:
    python scripts/download_generation_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import hf_hub_download  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "data" / "models"

REPO_ID = "Qwen/Qwen3-4B-GGUF"
FILENAME = "Qwen3-4B-Q4_K_M.gguf"


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID}/{FILENAME} to {MODELS_DIR} ...")
    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=str(MODELS_DIR))
    size_gb = Path(path).stat().st_size / (1024**3)
    print(f"Done: {path} ({size_gb:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
