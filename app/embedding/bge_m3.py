"""
BAAI/bge-m3 embedding wrapper — dense + sparse vectors, CPU-compatible.

The project brief requires this to run on modest/no-GPU hardware, so
`use_fp16=False` and `devices=["cpu"]` are hardcoded rather than
auto-detected — fp16 on CPU is unsupported/unreliable in PyTorch and this
must stay CPU-runnable regardless of what hardware happens to be
available at build time.

The model is pulled once from the Hugging Face Hub on first use and
cached locally (~2.2GB) — after that first download, no network calls are
made at inference time, which is what keeps this self-hostable per the
brief (no third-party SaaS API in the core pipeline).
"""

import sys
import types
from dataclasses import dataclass

from app.core.logging_config import get_logger

logger = get_logger(__name__)

MODEL_NAME = "BAAI/bge-m3"
DENSE_DIM = 1024


def _ensure_datasets_importable() -> None:
    """FlagEmbedding (this module's dependency) unconditionally does a bare
    `import datasets` deep in its own import chain
    (FlagEmbedding/abc/finetune/embedder/AbsDataset.py), purely to support
    fine-tuning-dataset loading this project never uses -- we only ever
    call BGEM3FlagModel.encode() for inference. That import transitively
    needs pyarrow's `dataset` C extension, which on some machines gets
    blocked by corporate endpoint-security/Application Control policies
    (seen live: "An Application Control policy has blocked this file" on
    pyarrow's `_dataset` DLL) even though nothing in this codebase touches
    real HF `datasets` functionality.

    If the real import fails, register a minimal stub under
    sys.modules["datasets"] so FlagEmbedding's `import datasets` line
    succeeds anyway. This is safe specifically because that whole chain
    only ever does `import datasets` (never `from datasets import <name>`)
    at *import* time -- real attributes are only touched inside
    fine-tuning methods that this project never calls. On a machine where
    the real `datasets` import works fine, this is a no-op and the real
    package is used as normal."""
    if "datasets" in sys.modules:
        return
    try:
        import datasets  # noqa: F401
    except Exception as exc:
        logger.warning(
            f"real 'datasets' package unavailable ({exc!r}); substituting a stub. "
            "This is expected/harmless for this project -- BGE-M3 embedding "
            "inference never uses datasets' actual functionality, only "
            "FlagEmbedding's own import chain unconditionally imports it."
        )
        sys.modules["datasets"] = types.ModuleType("datasets")


@dataclass
class EmbeddingResult:
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


class BgeM3Embedder:
    _model = None  # class-level: loading is expensive, share it across instances in one process

    def __init__(self, batch_size: int = 12):
        self.batch_size = batch_size
        if BgeM3Embedder._model is None:
            _ensure_datasets_importable()
            from FlagEmbedding import BGEM3FlagModel

            logger.info(f"loading {MODEL_NAME} (first call in this process only)...")
            BgeM3Embedder._model = BGEM3FlagModel(MODEL_NAME, use_fp16=False, devices=["cpu"])
        self._model = BgeM3Embedder._model

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        if not texts:
            return []
        out = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            batch_size=self.batch_size,
        )
        results = []
        for dense_vec, lexical_weights in zip(out["dense_vecs"], out["lexical_weights"]):
            results.append(
                EmbeddingResult(
                    dense=[float(x) for x in dense_vec],
                    sparse_indices=[int(tok_id) for tok_id in lexical_weights.keys()],
                    sparse_values=[float(w) for w in lexical_weights.values()],
                )
            )
        return results
