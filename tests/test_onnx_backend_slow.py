"""Optional slow, real-model tests for the ONNX Runtime backends.

Skipped unless the model directories are given explicitly. They never use the
network:

    SEEKR_TEST_EMBED_MODEL_DIR=/path/to/bge-small-en-v1.5 \
    SEEKR_TEST_RERANK_MODEL_DIR=/path/to/ms-marco-MiniLM-L-6-v2 \
        pytest -m slow tests/test_onnx_backend_slow.py

Each directory must contain ``onnx/model.onnx`` and ``tokenizer.json``
(``seekr models pull`` places them in the Hugging Face cache snapshot).
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

try:  # pragma: no cover - pytest is an optional development dependency
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

if pytest is not None:  # pragma: no cover - depends on the test runner
    pytestmark = pytest.mark.slow

_EMBED_DIR = os.environ.get("SEEKR_TEST_EMBED_MODEL_DIR", "").strip()
_RERANK_DIR = os.environ.get("SEEKR_TEST_RERANK_MODEL_DIR", "").strip()
_HAS_ORT = (
    importlib.util.find_spec("onnxruntime") is not None
    and importlib.util.find_spec("tokenizers") is not None
)
_HAS_ST = importlib.util.find_spec("sentence_transformers") is not None


@unittest.skipUnless(_HAS_ORT, "onnxruntime and tokenizers are required")
@unittest.skipUnless(_EMBED_DIR, "set SEEKR_TEST_EMBED_MODEL_DIR to a local model directory")
class RealOnnxEmbedderTests(unittest.TestCase):
    def _embedder(self):
        from seekr.onnx_backend import OnnxEmbedder

        return OnnxEmbedder("BAAI/bge-small-en-v1.5", model_dir=Path(_EMBED_DIR).expanduser())

    def test_real_model_dim_and_norm(self) -> None:
        import numpy as np

        embedder = self._embedder()
        self.assertEqual(embedder.dim, 384)
        docs = embedder.embed_documents(["write-ahead logging", "bias variance tradeoff"])
        self.assertEqual(docs.shape, (2, 384))
        np.testing.assert_allclose(np.linalg.norm(docs, axis=1), 1.0, rtol=1e-5)
        query = embedder.embed_query("what is a write-ahead log")
        self.assertGreater(float(docs[0] @ query), float(docs[1] @ query))

    def test_real_model_long_text(self) -> None:
        import numpy as np

        text = " ".join(f"token{i % 97}" for i in range(2000))
        vec = self._embedder().embed_documents([text])
        self.assertEqual(vec.shape, (1, 384))
        self.assertTrue(np.isfinite(vec).all())

    @unittest.skipUnless(_HAS_ST, "sentence-transformers is required for the comparison")
    def test_matches_sentence_transformers(self) -> None:
        import numpy as np

        from seekr.embed import SentenceTransformerEmbedder

        texts = [
            "SQLite stores the whole database in a single file.",
            "Gradient descent minimises a loss function step by step.",
            "The mitochondria is the powerhouse of the cell.",
        ]
        onnx = self._embedder().embed_documents(texts)
        torch = SentenceTransformerEmbedder(_EMBED_DIR, device="cpu").embed_documents(texts)
        cosines = (onnx * torch).sum(axis=1)
        self.assertGreaterEqual(float(np.min(cosines)), 0.995)


@unittest.skipUnless(_HAS_ORT, "onnxruntime and tokenizers are required")
@unittest.skipUnless(_RERANK_DIR, "set SEEKR_TEST_RERANK_MODEL_DIR to a local model directory")
class RealOnnxRerankerTests(unittest.TestCase):
    def test_real_reranker_orders_relevant_first(self) -> None:
        from seekr.onnx_backend import OnnxReranker

        reranker = OnnxReranker(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", model_dir=Path(_RERANK_DIR).expanduser()
        )
        scores = reranker.score(
            "how does write-ahead logging work",
            [
                "Bananas are rich in potassium.",
                "Write-ahead logging records changes in a log before applying them.",
            ],
        )
        self.assertEqual(len(scores), 2)
        self.assertGreater(scores[1], scores[0])


if __name__ == "__main__":
    unittest.main()
