"""Slow evaluation test against the packaged demo course with the default real model.

Uses the default ONNX Runtime backend, so it checks what a default install gets.
"""

from __future__ import annotations

import importlib.util
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import seekr
from seekr import offline, provision
from seekr.config import Config
from seekr.evaluation import load_queries
from seekr.indexer import Indexer
from seekr.onnx_backend import OnnxEmbedder
from seekr.search import Searcher
from seekr.store import Store

try:  # pragma: no cover - pytest is an optional development dependency
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

if pytest is not None:  # pragma: no cover - depends on the test runner
    pytestmark = pytest.mark.slow

_MODEL_DIR_ENV = "SEEKR_TEST_EMBED_MODEL_DIR"


def _has_fts5() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        options = {row[0] for row in conn.execute("PRAGMA compile_options")}
    finally:
        conn.close()
    return "ENABLE_FTS5" in options


class DemoEvalSlowTests(unittest.TestCase):
    def _load_embedder(self) -> OnnxEmbedder:
        cfg = Config()
        raw_dir = os.environ.get(_MODEL_DIR_ENV)
        if raw_dir is not None and raw_dir.strip():
            model_dir = Path(raw_dir).expanduser()
            self.assertTrue(
                model_dir.is_dir(),
                f"{_MODEL_DIR_ENV} must point to an existing model directory: {model_dir}",
            )
            return OnnxEmbedder(cfg.embed_model, model_dir=model_dir)

        if (
            importlib.util.find_spec("numpy") is None
            or importlib.util.find_spec("sqlite_vec") is None
            or importlib.util.find_spec("onnxruntime") is None
            or not _has_fts5()
        ):
            self.skipTest(
                "numpy, sqlite-vec, onnxruntime and FTS5 are required "
                "for the demo evaluation slow test"
            )

        assets = provision.assets_for(cfg)
        offline.configure_offline_environment()
        if provision.is_cached(assets[0]):
            return OnnxEmbedder(cfg.embed_model)

        self.skipTest(
            f"set {_MODEL_DIR_ENV} or run `seekr models pull` to cache {cfg.embed_model!r}"
        )

    def test_demo_course_hybrid_recall_at_3(self) -> None:
        embedder = self._load_embedder()
        root = seekr.demo_course_path().resolve()
        queries = load_queries(root / "queries.jsonl")
        self.assertGreater(len(queries), 0)

        with tempfile.TemporaryDirectory(prefix="seekr-demo-eval-") as tmpdir:
            db_path = Path(tmpdir) / "index.db"
            cfg = Config(roots=(root,), db_path=db_path, embed_model=embedder.model_name)
            store = Store(
                db_path,
                embedder.model_name,
                embedder.dim,
                vector_quant=cfg.vector_quant,
            )
            try:
                report = Indexer(store, embedder, cfg).sync([root])
                self.assertEqual(report.errors, [])
                searcher = Searcher(store, embedder)
                matched = 0
                missed: list[str] = []
                for query in queries:
                    hits = searcher.search(query.query, k=3, mode="hybrid", per_file=1)
                    top3 = {
                        Path(hit.path).resolve().relative_to(root).as_posix()
                        for hit in hits
                    }
                    if top3 & query.relevant:
                        matched += 1
                    else:
                        missed.append(f"{query.id}: got {sorted(top3)}")
            finally:
                store.close()

        recall_at_3 = matched / len(queries)
        print(f"demo_course hybrid recall@3 = {recall_at_3:.4f} ({matched}/{len(queries)})")
        self.assertGreaterEqual(
            recall_at_3,
            0.8,
            f"demo_course hybrid recall@3={recall_at_3:.4f} ({matched}/{len(queries)}) < 0.8; "
            f"missed={missed}",
        )


if __name__ == "__main__":
    unittest.main()
