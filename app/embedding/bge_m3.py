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

from dataclasses import dataclass

from app.core.logging_config import get_logger

logger = get_logger(__name__)

MODEL_NAME = "BAAI/bge-m3"
DENSE_DIM = 1024


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
