"""Tests for cross-encoder reranking and its Searcher integration.

The default suite is offline and dependency-free: the real model is replaced
through the shared :mod:`seekr.offline` adapter.  The slow test uses an
explicitly provisioned local model directory and never downloads anything.
"""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest import mock

from seekr import offline
from seekr.config import Config
from seekr.rerank import (
    CrossEncoderReranker,
    FakeReranker,
    Reranker,
    build_reranker,
)
from seekr.search import Searcher
from seekr.store import ChunkRow

try:  # pragma: no cover - pytest is an optional development dependency
    import pytest
except ImportError:  # pragma: no cover
    pytest = None


_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_HAS_SENTENCE_TRANSFORMERS = importlib.util.find_spec("sentence_transformers") is not None
_RERANK_MODEL_DIR_ENV = "SEEKR_TEST_RERANK_MODEL_DIR"
_RERANK_MODEL_DIR = os.environ.get(_RERANK_MODEL_DIR_ENV)
_OPTED_IN = bool(_RERANK_MODEL_DIR and _RERANK_MODEL_DIR.strip())


def _slow(test_class: type[unittest.TestCase]) -> type[unittest.TestCase]:
    if pytest is None:
        return test_class
    return pytest.mark.slow(test_class)


def _chunk(chunk_id: int, path: str, text: str) -> ChunkRow:
    return ChunkRow(
        id=chunk_id,
        doc_id=chunk_id,
        path=path,
        kind="text",
        text=text,
        locator=f"L{chunk_id}",
        loc_start=float(chunk_id),
    )


class _RecordingStore:
    """A minimal duck-typed store with a fixed keyword ranking."""

    def __init__(self, chunks: list[ChunkRow], keyword_order: list[int]) -> None:
        self.chunks = {chunk.id: chunk for chunk in chunks}
        self.keyword_order = list(keyword_order)
        self.keyword_calls: list[tuple[str, int, str | None]] = []
        self.fetch_calls: list[tuple[int, ...]] = []

    def keyword_search(
        self,
        match_query: str,
        limit: int,
        path_prefix: str | None = None,
    ) -> list[tuple[int, float]]:
        self.keyword_calls.append((match_query, limit, path_prefix))
        ordered = self.keyword_order
        if path_prefix is not None:
            ordered = [
                chunk_id
                for chunk_id in ordered
                if self.chunks[chunk_id].path == path_prefix
                or self.chunks[chunk_id].path.startswith(path_prefix + "/")
            ]
        return [
            (chunk_id, float(rank))
            for rank, chunk_id in enumerate(ordered[:limit], start=1)
        ]

    def fetch_chunks(self, ids: list[int]) -> dict[int, ChunkRow]:
        self.fetch_calls.append(tuple(ids))
        return {
            chunk_id: self.chunks[chunk_id]
            for chunk_id in ids
            if chunk_id in self.chunks
        }


class _ExplodingEmbedder:
    """Keyword-only searches must never touch the embedder."""

    model_name = "exploding"
    dim = 2

    def embed_documents(self, texts: list[str]) -> Any:
        raise AssertionError("search must not embed documents")

    def embed_query(self, text: str) -> Any:
        raise AssertionError("keyword-only search must not embed the query")


class _FakeCrossEncoder:
    """A sentence-transformers CrossEncoder stand-in."""

    def __init__(self, outputs: list[Any] | None = None) -> None:
        self.outputs = outputs
        self.calls: list[list[list[str]]] = []

    def predict(self, pairs: list[list[str]]) -> Any:
        self.calls.append([list(pair) for pair in pairs])
        if self.outputs is not None:
            return self.outputs
        return [0.0 for _ in pairs]


class RerankerProtocolAndFakeTests(unittest.TestCase):
    def test_fake_reranker_satisfies_the_runtime_protocol(self) -> None:
        self.assertIsInstance(FakeReranker(), Reranker)

    def test_fake_scores_the_first_matching_substring_and_defaults_to_zero(self) -> None:
        reranker = FakeReranker({"beta": 9.0, "alpha": 3.0})

        scores = reranker.score(
            "query",
            ["alpha beta", "nothing here", "beta alpha"],
        )

        self.assertEqual(scores, [9.0, 0.0, 9.0])
        self.assertEqual(
            reranker.calls,
            [("query", ["alpha beta", "nothing here", "beta alpha"])],
        )

    def test_fake_returns_an_empty_list_for_no_passages(self) -> None:
        reranker = FakeReranker({"alpha": 1.0})

        self.assertEqual(reranker.score("query", []), [])


class CrossEncoderRerankerTests(unittest.TestCase):
    def test_constructor_and_build_reranker_are_lazy(self) -> None:
        with mock.patch.object(offline, "load_local_model") as load:
            reranker = CrossEncoderReranker("org/cross-encoder")
            built = build_reranker(Config(rerank_model="org/other-cross-encoder"))

        load.assert_not_called()
        self.assertIsInstance(reranker, CrossEncoderReranker)
        self.assertEqual(reranker.model_name, "org/cross-encoder")
        self.assertIsInstance(built, CrossEncoderReranker)
        self.assertEqual(built.model_name, "org/other-cross-encoder")

    def test_empty_passages_return_empty_without_loading_the_model(self) -> None:
        with mock.patch.object(offline, "load_local_model") as load:
            reranker = CrossEncoderReranker("org/cross-encoder")
            scores = reranker.score("query", [])

        self.assertEqual(scores, [])
        load.assert_not_called()

    def test_scores_use_query_passage_pairs_and_reuse_the_loaded_model(self) -> None:
        model = _FakeCrossEncoder(outputs=[0.25, 0.75])
        with mock.patch.object(
            offline, "load_local_model", return_value=model
        ) as load:
            reranker = CrossEncoderReranker("org/cross-encoder")
            scores = reranker.score("first query", ["alpha", "beta"])
            second = reranker.score("second query", ["gamma", "delta"])

        self.assertEqual(scores, [0.25, 0.75])
        self.assertEqual(second, [0.25, 0.75])
        load.assert_called_once_with("org/cross-encoder", kind="cross_encoder")
        self.assertEqual(
            model.calls,
            [
                [["first query", "alpha"], ["first query", "beta"]],
                [["second query", "gamma"], ["second query", "delta"]],
            ],
        )

    def test_concurrent_first_use_loads_one_model_and_reuses_it(self) -> None:
        load_count = 0
        count_lock = threading.Lock()
        model = _FakeCrossEncoder()

        def loader(*args: object, **kwargs: object) -> _FakeCrossEncoder:
            nonlocal load_count
            with count_lock:
                load_count += 1
            time.sleep(0.05)
            return model

        reranker = CrossEncoderReranker("org/cross-encoder")
        barrier = threading.Barrier(8)

        def score_one(index: int) -> list[float]:
            barrier.wait(timeout=5)
            return reranker.score("query", [f"passage {index}"])

        with (
            mock.patch.object(offline, "load_local_model", side_effect=loader) as load,
            ThreadPoolExecutor(max_workers=8) as pool,
        ):
            results = list(pool.map(score_one, range(8)))

        self.assertEqual(results, [[0.0]] * 8)
        self.assertEqual(load_count, 1)
        load.assert_called_once_with("org/cross-encoder", kind="cross_encoder")
        self.assertEqual(len(model.calls), 8)

    def test_concurrent_scores_serialize_model_predictions(self) -> None:
        class Model:
            def __init__(self) -> None:
                self.guard = threading.Lock()
                self.active = 0
                self.max_active = 0
                self.calls = 0

            def predict(self, pairs: list[list[str]]) -> list[float]:
                with self.guard:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.05)
                with self.guard:
                    self.active -= 1
                    self.calls += 1
                return [0.0 for _ in pairs]

        model = Model()
        reranker = CrossEncoderReranker("org/cross-encoder")

        def score_one(index: int) -> list[float]:
            return reranker.score("query", [f"passage {index}"])

        with (
            mock.patch.object(offline, "load_local_model", return_value=model),
            ThreadPoolExecutor(max_workers=8) as pool,
        ):
            results = list(pool.map(score_one, range(8)))

        self.assertEqual(results, [[0.0]] * 8)
        self.assertEqual(model.max_active, 1)
        self.assertEqual(model.calls, 8)

    def test_wrong_score_count_raises_value_error(self) -> None:
        model = _FakeCrossEncoder(outputs=[1.0])
        with mock.patch.object(offline, "load_local_model", return_value=model):
            reranker = CrossEncoderReranker("org/cross-encoder")
            with self.assertRaises(ValueError) as raised:
                reranker.score("query", ["alpha", "beta"])

        self.assertIn("2", str(raised.exception))
        self.assertIn("1", str(raised.exception))

    def test_non_finite_or_non_numeric_scores_raise_value_error(self) -> None:
        invalid_values = [float("nan"), float("inf"), float("-inf"), "not-a-number"]
        for value in invalid_values:
            with self.subTest(value=value):
                model = _FakeCrossEncoder(outputs=[value])
                with mock.patch.object(
                    offline, "load_local_model", return_value=model
                ):
                    reranker = CrossEncoderReranker("org/cross-encoder")
                    with self.assertRaises(ValueError):
                        reranker.score("query", ["alpha"])

    def test_missing_provider_error_surfaces_on_first_score(self) -> None:
        unavailable = offline.ModelUnavailableError("cross-encoder is not available")
        with mock.patch.object(
            offline, "load_local_model", side_effect=unavailable
        ) as load:
            reranker = CrossEncoderReranker("org/cross-encoder")
            load.assert_not_called()
            with self.assertRaises(offline.ModelUnavailableError) as raised:
                reranker.score("query", ["alpha"])

        self.assertIs(raised.exception, unavailable)
        load.assert_called_once_with("org/cross-encoder", kind="cross_encoder")

    def test_import_does_not_eagerly_import_heavy_modules(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (
            "import sys\n"
            "import seekr.rerank\n"
            "heavy = ('numpy', 'sentence_transformers', 'torch')\n"
            "assert not any(name in sys.modules for name in heavy), "
            "[name for name in heavy if name in sys.modules]\n"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root / "src")
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


class SearcherRerankIntegrationTests(unittest.TestCase):
    def test_rerank_reorders_candidates_and_replaces_scores(self) -> None:
        chunks = [
            _chunk(1, "/notes/a.md", "alpha"),
            _chunk(2, "/notes/b.md", "beta"),
            _chunk(3, "/notes/c.md", "gamma"),
        ]
        store = _RecordingStore(chunks, keyword_order=[1, 2, 3])
        reranker = FakeReranker({"beta": 9.0})

        hits = Searcher(store, _ExplodingEmbedder(), reranker).search(
            "alpha beta gamma",
            mode="keyword",
            rerank=True,
        )

        self.assertEqual([hit.chunk_id for hit in hits], [2, 1, 3])
        self.assertEqual([hit.score for hit in hits], [9.0, 0.0, 0.0])
        self.assertEqual(
            reranker.calls[0][1],
            ["alpha", "beta", "gamma"],
        )

    def test_top_thirty_is_reranked_before_per_file_and_truncation(self) -> None:
        chunks = [
            _chunk(1, "/notes/a.md", "alpha first"),
            _chunk(2, "/notes/a.md", "alpha second"),
            _chunk(3, "/notes/b.md", "beta only"),
        ]
        chunks.extend(
            _chunk(chunk_id, f"/notes/f{chunk_id:02d}.md", f"plain {chunk_id}")
            for chunk_id in range(4, 41)
        )
        store = _RecordingStore(chunks, keyword_order=list(range(1, 41)))
        reranker = FakeReranker(
            {"alpha": 5.0, "beta": 1.0, "plain 40": 100.0}
        )

        hits = Searcher(store, _ExplodingEmbedder(), reranker).search(
            "alpha beta plain",
            mode="keyword",
            k=2,
            per_file=1,
            rerank=True,
        )

        self.assertEqual([hit.chunk_id for hit in hits], [1, 3])
        self.assertEqual([hit.score for hit in hits], [5.0, 1.0])
        self.assertEqual(len(reranker.calls), 1)
        self.assertEqual(len(reranker.calls[0][1]), 30)
        self.assertNotIn("plain 40", reranker.calls[0][1])

    def test_rerank_without_a_reranker_raises_value_error(self) -> None:
        store = _RecordingStore(
            [_chunk(1, "/notes/a.md", "alpha")],
            keyword_order=[1],
        )

        with self.assertRaises(ValueError):
            Searcher(store, _ExplodingEmbedder()).search(
                "alpha",
                mode="keyword",
                rerank=True,
            )

    def test_no_candidates_does_not_call_the_reranker(self) -> None:
        store = _RecordingStore([], keyword_order=[])
        reranker = FakeReranker({"alpha": 1.0})

        hits = Searcher(store, _ExplodingEmbedder(), reranker).search(
            "alpha",
            mode="keyword",
            rerank=True,
        )

        self.assertEqual(hits, [])
        self.assertEqual(reranker.calls, [])


@_slow
@unittest.skipUnless(
    _HAS_NUMPY and _HAS_SENTENCE_TRANSFORMERS,
    "numpy and sentence-transformers are required for the real-model test",
)
@unittest.skipUnless(
    _OPTED_IN,
    f"set {_RERANK_MODEL_DIR_ENV} to a pre-provisioned local model directory",
)
class RealCrossEncoderTests(unittest.TestCase):
    def test_cross_encoder_prefers_the_relevant_passage(self) -> None:
        assert _RERANK_MODEL_DIR is not None
        model_dir = Path(_RERANK_MODEL_DIR).expanduser()
        self.assertTrue(
            model_dir.is_dir(),
            f"{_RERANK_MODEL_DIR_ENV} must point to an existing model directory: "
            f"{model_dir}",
        )

        reranker = CrossEncoderReranker(str(model_dir))
        query = "How do neural networks learn from their mistakes?"
        passages = [
            (
                "Neural networks learn from errors using backpropagation to compute "
                "gradients of the loss and update each weight."
            ),
            "The mitochondria is the powerhouse of the cell.",
            "Rust ownership prevents data races without a garbage collector.",
        ]

        scores = reranker.score(query, passages)

        self.assertEqual(len(scores), len(passages))
        self.assertTrue(all(math.isfinite(score) for score in scores))
        self.assertEqual(max(range(len(scores)), key=scores.__getitem__), 0)


if __name__ == "__main__":
    unittest.main()
