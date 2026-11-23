"""The default install must not need or import PyTorch."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from seekr.config import Config

ROOT = Path(__file__).resolve().parents[1]


class DefaultBackendTests(unittest.TestCase):
    def test_default_backend_is_onnx(self) -> None:
        self.assertEqual(Config().embed_backend, "onnx")

    def test_cli_and_default_backend_never_import_torch(self) -> None:
        code = (
            "import sys, seekr.cli\n"
            "from seekr.embed import build_embedder\n"
            "from seekr.rerank import build_reranker\n"
            "from seekr.config import Config\n"
            "build_embedder(Config()); build_reranker(Config())\n"
            "bad = sorted(m for m in ('torch', 'sentence_transformers', 'transformers') "
            "if m in sys.modules)\n"
            "assert not bad, bad\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        subprocess.run([sys.executable, "-c", code], check=True, env=env)


if __name__ == "__main__":
    unittest.main()
