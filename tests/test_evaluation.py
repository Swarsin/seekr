"""Tests for file-level evaluation metrics and the offline eval harness.

Metric and JSONL tests use only the standard library.  The end-to-end harness
test is split in two: a mock-backed test that always runs, and a real
``Store``/``FakeEmbedder`` integration test that skips when numpy, sqlite-vec
or FTS5 is unavailable.  No test downloads data or loads a real model.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest import mock

from seekr import cli_eval, evaluation
from seekr.embed import FakeEmbedder
from seekr.models import Hit, text_sha
from seekr.rerank import FakeReranker


def _has_fts5() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        options = {row[0] for row in conn.execute("PRAGMA compile_options")}
    finally:
        conn.close()
    return "ENABLE_FTS5" in options


_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_HAS_SQLITE_VEC = importlib.util.find_spec("sqlite_vec") is not None
_HAS_FTS5 = _has_fts5()


class MetricTests(unittest.TestCase):
    def test_recall_at_k(self) -> None:
        self.assertEqual(evaluation.recall_at_k(["a", "b", "c"], {"c", "z"}, 2), 0.0)
        self.assertEqual(evaluation.recall_at_k(["a", "b", "c"], {"c", "z"}, 3), 0.5)

    def test_recall_removes_duplicate_files_before_cutting_at_k(self) -> None:
        self.assertEqual(
            evaluation.recall_at_k(["a", "a", "b"], {"a", "b"}, 2),
            1.0,
        )

    def test_mrr_at_k(self) -> None:
        self.assertEqual(evaluation.mrr_at_k(["a", "b", "c"], {"b"}, 10), 0.5)
        self.assertEqual(evaluation.mrr_at_k(["a", "b", "c"], {"z"}, 10), 0.0)

    def test_mrr_stops_at_k_and_removes_duplicate_rankings(self) -> None:
        self.assertEqual(evaluation.mrr_at_k(["a", "a", "b"], {"b"}, 1), 0.0)
        self.assertEqual(evaluation.mrr_at_k(["a", "a", "b"], {"b"}, 2), 0.5)
        self.assertEqual(evaluation.mrr_at_k(["a", "a", "b"], {"b"}, 3), 0.5)

    def test_ndcg_at_k(self) -> None:
        self.assertEqual(evaluation.ndcg_at_k(["a", "b"], {"a"}, 10), 1.0)
        self.assertAlmostEqual(
            evaluation.ndcg_at_k(["b", "a"], {"a"}, 10),
            0.6309,
            places=4,
        )

    def test_ndcg_binary_gain_uses_all_relevant_files_for_ideal_dcg(self) -> None:
        # DCG = 1/log2(2) + 1/log2(3); IDCG has the same two terms.
        self.assertEqual(
            evaluation.ndcg_at_k(["a", "b"], {"a", "b"}, 10),
            1.0,
        )
        self.assertAlmostEqual(
            evaluation.ndcg_at_k(["b"], {"a", "b"}, 10),
            0.6131,
            places=4,
        )

    def test_empty_relevance_and_zero_k_are_zero(self) -> None:
        self.assertEqual(evaluation.recall_at_k(["a"], set(), 10), 0.0)
        self.assertEqual(evaluation.mrr_at_k(["a"], set(), 10), 0.0)
        self.assertEqual(evaluation.ndcg_at_k(["a"], set(), 10), 0.0)
        self.assertEqual(evaluation.ndcg_at_k(["a"], {"a"}, 0), 0.0)


class LoadQueriesTests(unittest.TestCase):
    def test_load_queries_parses_jsonl_and_ignores_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.jsonl"
            path.write_text(
                "\n"
                + json.dumps(
                    {
                        "id": "q1",
                        "query": "first needle",
                        "relevant": ["a.txt", "notes/b.md"],
                    }
                )
                + "\n\n"
                + json.dumps(
                    {
                        "id": "q2",
                        "query": "second needle",
                        "relevant": ["c.txt"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            queries = evaluation.load_queries(path)

        self.assertEqual([query.id for query in queries], ["q1", "q2"])
        self.assertEqual(queries[0].query, "first needle")
        self.assertEqual(queries[0].relevant, frozenset({"a.txt", "notes/b.md"}))
        self.assertEqual(queries[1].relevant, frozenset({"c.txt"}))

    def test_load_queries_reports_the_malformed_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.jsonl"
            path.write_text(
                json.dumps({"id": "ok", "query": "x", "relevant": ["a.txt"]})
                + "\n{not json}\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as raised:
                evaluation.load_queries(path)

        self.assertIn("line 2", str(raised.exception))

    def test_load_queries_rejects_a_non_list_relevant_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.jsonl"
            path.write_text(
                json.dumps({"id": "q", "query": "x", "relevant": "a.txt"}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as raised:
                evaluation.load_queries(path)

        self.assertIn("relevant", str(raised.exception))


class RenderTableTests(unittest.TestCase):
    def test_render_table_has_header_and_one_row_per_mode(self) -> None:
        results = {
            "keyword": {
                "recall@5": 1.0,
                "recall@10": 1.0,
                "mrr@10": 1.0,
                "ndcg@10": 1.0,
                "p50_ms": 2.0,
            },
            "vector": {
                "recall@5": 0.5,
                "recall@10": 0.75,
                "mrr@10": 0.5,
                "ndcg@10": 0.5,
                "p50_ms": 4.0,
            },
        }

        table = evaluation.render_table(results)
        lines = [line for line in table.splitlines() if line.strip()]

        self.assertIn("| mode |", lines[0].lower())
        for metric in ("recall@5", "recall@10", "mrr@10", "ndcg@10", "p50_ms"):
            self.assertIn(metric, lines[0])
        mode_rows = [line for line in lines if "keyword" in line or "vector" in line]
        self.assertEqual(len(mode_rows), 2)


class CliEvalTests(unittest.TestCase):
    def _fake_typer(self, output: list[object]) -> Any:
        exit_class = type("Exit", (Exception,), {})

        def echo(*args: object, **kwargs: object) -> None:
            output.append((args, kwargs))

        return SimpleNamespace(
            Option=lambda *args, **kwargs: None,
            echo=echo,
            Exit=exit_class,
        )

    def test_register_adds_the_eval_command(self) -> None:
        output: list[object] = []
        fake_typer = self._fake_typer(output)

        class FakeApp:
            def __init__(self) -> None:
                self.commands: dict[str, Any] = {}

            def command(self, name: str) -> Any:
                def decorate(function: Any) -> Any:
                    self.commands[name] = function
                    return function

                return decorate

        app = FakeApp()
        with mock.patch.dict(sys.modules, {"typer": fake_typer}):
            cli_eval.register(app)

        self.assertEqual(set(app.commands), {"eval"})

    def test_command_runs_selected_modes_and_writes_the_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out_path = Path(directory) / "results.md"
            queries = [evaluation.EvalQuery("q", "needle", frozenset({"a.txt"}))]
            embedder = object()
            results = {
                "keyword": {
                    "recall@5": 1.0,
                    "recall@10": 1.0,
                    "mrr@10": 1.0,
                    "ndcg@10": 1.0,
                    "p50_ms": 1.0,
                }
            }
            output: list[object] = []
            command = cli_eval._build_eval_command(self._fake_typer(output))

            with (
                mock.patch.object(cli_eval, "load_queries", return_value=queries),
                mock.patch.object(cli_eval, "load_config", return_value=object()),
                mock.patch.object(cli_eval, "build_embedder", return_value=embedder),
                mock.patch.object(cli_eval, "run_eval", return_value=results) as run_eval,
            ):
                command(
                    root=Path("/tmp/root"),
                    queries=Path("/tmp/queries.jsonl"),
                    modes=["keyword,vector"],
                    k=3,
                    out=out_path,
                )

            run_eval.assert_called_once_with(
                Path("/tmp/root"),
                queries,
                ["keyword", "vector"],
                embedder,
                None,
                k=3,
            )
            self.assertEqual(
                out_path.read_text(encoding="utf-8"),
                evaluation.render_table(results),
            )

        self.assertGreaterEqual(len(output), 2)


def _hit(path: Path, chunk_id: int) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        path=str(path),
        locator="",
        loc_start=0.0,
        snippet="",
        score=1.0,
        kind="text",
    )


class _FakeStore:
    instances: ClassVar[list[_FakeStore]] = []
    test_root: Path | None = None

    def __init__(self, db_path: Path, embed_model: str, dim: int) -> None:
        self.db_path = Path(db_path)
        self.embed_model = embed_model
        self.dim = dim
        self.closed = False
        self.indexed = False
        self.__class__.instances.append(self)

    def close(self) -> None:
        self.closed = True


class _FakeIndexer:
    def __init__(self, store: _FakeStore, embedder: Any, cfg: Any) -> None:
        self.store = store
        self.embedder = embedder
        self.cfg = cfg

    def sync(self, roots: list[Path]) -> Any:
        self.store.indexed = True
        return SimpleNamespace(errors=[], added=3, updated=0, unchanged=0)


class _FakeSearcher:
    calls: ClassVar[list[dict[str, Any]]] = []
    test_root: Path | None = None

    def __init__(self, store: _FakeStore, embedder: Any, reranker: Any) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker

    def get_hits(self, query: str, **kwargs: Any) -> list[Hit]:
        self.__class__.calls.append({"query": query, **kwargs})
        # A measurable delay proves the harness times the whole search call,
        # including mock query embedding and reranking work.
        time.sleep(0.01)
        root = self.__class__.test_root
        assert root is not None
        if query == "alpha":
            names = ["notes/a.md", "notes/a.md", "notes/b.md"]
        else:
            names = ["notes/c.md"]
        return [_hit(root / name, index + 1) for index, name in enumerate(names)]


class _FailingIndexer(_FakeIndexer):
    def sync(self, roots: list[Path]) -> Any:
        return SimpleNamespace(
            errors=[(str(roots[0]), "injected indexing failure")],
            added=0,
            updated=0,
            unchanged=0,
        )


class HarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "root"
        (self.root / "notes").mkdir(parents=True)
        for name in ("a.md", "b.md", "c.md"):
            (self.root / "notes" / name).write_text(f"text for {name}", encoding="utf-8")
        self.queries = [
            evaluation.EvalQuery("q1", "alpha", frozenset({"notes/a.md"})),
            evaluation.EvalQuery("q2", "omega", frozenset({"notes/missing.md"})),
        ]
        _FakeStore.instances = []
        _FakeSearcher.calls = []
        _FakeSearcher.test_root = self.root

    def _patches(self, indexer: type = _FakeIndexer):
        return (
            mock.patch.object(evaluation, "Store", _FakeStore),
            mock.patch.object(evaluation, "Indexer", indexer),
            mock.patch.object(evaluation, "Searcher", _FakeSearcher),
        )

    def test_run_eval_returns_all_modes_and_metrics(self) -> None:
        modes = ["keyword", "vector", "hybrid", "hybrid+rerank"]
        store_patch, indexer_patch, searcher_patch = self._patches()

        with store_patch, indexer_patch, searcher_patch:
            results = evaluation.run_eval(
                self.root,
                self.queries,
                modes,
                FakeEmbedder(),
                FakeReranker(),
                k=3,
            )

        self.assertEqual(list(results), modes)
        for mode in modes:
            self.assertEqual(
                set(results[mode]),
                {"recall@5", "recall@10", "mrr@10", "ndcg@10", "p50_ms"},
            )
            self.assertGreaterEqual(results[mode]["p50_ms"], 5.0)
        self.assertAlmostEqual(results["keyword"]["recall@5"], 0.5)
        self.assertAlmostEqual(results["keyword"]["recall@10"], 0.5)
        self.assertAlmostEqual(results["keyword"]["mrr@10"], 0.5)
        self.assertAlmostEqual(results["keyword"]["ndcg@10"], 0.5)

        self.assertEqual(len(_FakeStore.instances), 1)
        store = _FakeStore.instances[0]
        self.assertTrue(store.indexed)
        self.assertTrue(store.closed)
        self.assertFalse(store.db_path.parent.exists())

        self.assertTrue(_FakeSearcher.calls)
        for call in _FakeSearcher.calls:
            self.assertEqual(call["per_file"], 1)
            self.assertGreaterEqual(call["k"], 10)
        self.assertTrue(
            any(
                call["mode"] == "hybrid" and call["rerank"] is True
                for call in _FakeSearcher.calls
            )
        )
        self.assertTrue(
            any(
                call["mode"] == "keyword" and call["rerank"] is False
                for call in _FakeSearcher.calls
            )
        )

    def test_indexing_errors_close_the_temp_store_and_remove_it(self) -> None:
        store_patch, indexer_patch, searcher_patch = self._patches(_FailingIndexer)

        with store_patch, indexer_patch, searcher_patch, self.assertRaises(
            RuntimeError
        ) as raised:
            evaluation.run_eval(
                self.root,
                self.queries,
                ["keyword"],
                FakeEmbedder(),
                None,
            )

        self.assertIn("indexing failed", str(raised.exception).lower())
        store = _FakeStore.instances[0]
        self.assertTrue(store.closed)
        self.assertFalse(store.db_path.parent.exists())
        self.assertEqual(_FakeSearcher.calls, [])

    def test_hybrid_rerank_requires_a_reranker(self) -> None:
        with self.assertRaises(ValueError) as raised:
            evaluation.run_eval(
                self.root,
                self.queries,
                ["hybrid+rerank"],
                FakeEmbedder(),
                None,
            )

        self.assertIn("reranker", str(raised.exception).lower())

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as raised:
            evaluation.run_eval(
                self.root,
                self.queries,
                ["keyword", "magic"],
                FakeEmbedder(),
                None,
            )

        self.assertIn("magic", str(raised.exception))


@contextlib.contextmanager
def _stdlib_vector_backend():
    """Let the production Store run on plain SQLite for a keyword-only test."""

    def create_table(conn: sqlite3.Connection, dim: int) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks_vec ("
            "rowid INTEGER PRIMARY KEY, embedding BLOB NOT NULL)"
        )

    def serialize(vector: object) -> bytes:
        return bytes(vector)  # type: ignore[arg-type]

    def deserialize(payload: bytes | memoryview) -> bytes:
        return bytes(payload)

    with (
        mock.patch("seekr.store._load_vector_extension", lambda conn: None),
        mock.patch("seekr.store._create_vector_table", create_table),
        mock.patch("seekr.store._serialize_vector", serialize),
        mock.patch("seekr.store._deserialize_vector", deserialize),
    ):
        yield


class _ByteEmbedder:
    """Standard-library embedder for a real SQLite/FTS keyword evaluation."""

    model_name = "fake-bytes"
    dim = 8

    def embed_documents(self, texts: list[str]) -> list[bytes]:
        return [text_sha(text).encode("ascii")[: self.dim] for text in texts]

    def embed_query(self, text: str) -> bytes:
        return text_sha(text).encode("ascii")[: self.dim]


@unittest.skipUnless(_HAS_FTS5, "a loadable FTS5 build is required")
class StdlibStoreIntegrationTests(unittest.TestCase):
    def test_run_eval_keyword_with_real_store_and_indexer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "corpus"
            root.mkdir()
            (root / "alpha.txt").write_text(
                "uniquekeyword shared context", encoding="utf-8"
            )
            (root / "beta.txt").write_text("other words entirely", encoding="utf-8")
            queries = [
                evaluation.EvalQuery("q1", "uniquekeyword", frozenset({"alpha.txt"}))
            ]

            with _stdlib_vector_backend():
                results = evaluation.run_eval(
                    root,
                    queries,
                    ["keyword"],
                    _ByteEmbedder(),
                    None,
                )

        self.assertEqual(results["keyword"]["recall@5"], 1.0)
        self.assertEqual(
            set(results["keyword"]),
            {"recall@5", "recall@10", "mrr@10", "ndcg@10", "p50_ms"},
        )


@unittest.skipUnless(
    _HAS_NUMPY and _HAS_SQLITE_VEC and _HAS_FTS5,
    "numpy, sqlite-vec and a loadable FTS5 build are required",
)
class RealStoreIntegrationTests(unittest.TestCase):
    def test_run_eval_with_fake_embedder_and_real_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            root = tmp / "corpus"
            root.mkdir()
            (root / "alpha.txt").write_text(
                "alphaword shared context about gardens", encoding="utf-8"
            )
            (root / "beta.txt").write_text(
                "betaword kitchen and bread", encoding="utf-8"
            )
            (root / "gamma.md").write_text(
                "gammaword notebook about code", encoding="utf-8"
            )
            (root / "delta.md").write_text(
                "deltaword vector search notes", encoding="utf-8"
            )
            queries = [
                evaluation.EvalQuery(
                    "q1", "alphaword", frozenset({"alpha.txt"})
                )
            ]

            results = evaluation.run_eval(
                root,
                queries,
                ["keyword", "vector", "hybrid", "hybrid+rerank"],
                FakeEmbedder(dim=32),
                FakeReranker({"alphaword": 5.0}),
            )

        self.assertEqual(
            set(results),
            {"keyword", "vector", "hybrid", "hybrid+rerank"},
        )
        self.assertEqual(results["keyword"]["recall@5"], 1.0)
        for mode in results:
            self.assertEqual(
                set(results[mode]),
                {"recall@5", "recall@10", "mrr@10", "ndcg@10", "p50_ms"},
            )


if __name__ == "__main__":
    unittest.main()
