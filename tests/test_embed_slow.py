"""Optional slow, real-model tests for the sentence-transformers backend.

These tests are marked ``slow`` when pytest is available. They are skipped
unless numpy, sentence-transformers and an explicitly provisioned local model
directory are all present. They never use the network and never download.

Pre-provision the default ``BAAI/bge-small-en-v1.5`` model in a directory and
opt in explicitly:

    SEEKR_TEST_EMBED_MODEL_DIR=/path/to/bge-small-en-v1.5 \
        PYTHONPATH=src pytest -m slow tests/test_embed_slow.py

Once the opt-in is set, a missing or broken model fails the test. It is not
silently skipped.
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


_MODEL_DIR_ENV = "SEEKR_TEST_EMBED_MODEL_DIR"
_MODEL_DIR = os.environ.get(_MODEL_DIR_ENV)
_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_HAS_SENTENCE_TRANSFORMERS = importlib.util.find_spec("sentence_transformers") is not None
_OPTED_IN = bool(_MODEL_DIR and _MODEL_DIR.strip())


@unittest.skipUnless(
    _HAS_NUMPY and _HAS_SENTENCE_TRANSFORMERS,
    "numpy and sentence-transformers are required for the real-model test",
)
@unittest.skipUnless(
    _OPTED_IN,
    f"set {_MODEL_DIR_ENV} to a pre-provisioned local model directory",
)
class RealModelTests(unittest.TestCase):
    def test_bge_shape_and_prefix(self) -> None:
        import numpy as np

        from seekr.embed import SentenceTransformerEmbedder

        assert _MODEL_DIR is not None
        model_dir = Path(_MODEL_DIR).expanduser()
        self.assertTrue(
            model_dir.is_dir(),
            f"{_MODEL_DIR_ENV} must point to an existing model directory: {model_dir}",
        )

        embedder = SentenceTransformerEmbedder(str(model_dir))
        # Once the user has opted in, a bad local model must fail the test
        # rather than be converted into a skip.
        self.assertEqual(embedder.dim, 384)
        documents = embedder.embed_documents(["hello world", "another passage"])
        query = embedder.embed_query("hello world")
        same_as_document = embedder.embed_documents(["hello world"])[0]

        self.assertEqual(documents.shape, (2, 384))
        self.assertEqual(documents.dtype, np.float32)
        self.assertEqual(query.shape, (384,))
        self.assertEqual(query.dtype, np.float32)
        self.assertTrue(np.allclose(np.linalg.norm(documents, axis=1), 1.0, atol=1e-5))
        self.assertAlmostEqual(float(np.linalg.norm(query)), 1.0, places=5)
        self.assertFalse(np.array_equal(query, same_as_document))


if __name__ == "__main__":
    unittest.main()
