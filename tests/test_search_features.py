"""Tests for search --json output and --explain retriever attribution."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import tempfile
import threading
import time
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from seekr import client, daemon
from seekr.cli import app
from seekr.config import Config
from seekr.embed import FakeEmbedder
from seekr.indexer import Indexer
from seekr.models import Hit, HitExplain
from seekr.search import Searcher
from seekr.store import ChunkRow, Store

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_HAS_SQLITE_VEC = importlib.util.find_spec("sqlite_vec") is not None
_HAS_TYPER = importlib.util.find_spec("typer") is not None


def _has_fts5() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        options = {row[0] for row in conn.execute("PRAGMA compile_options")}
    finally:
        conn.close()
    return "ENABLE_FTS5" in options


_NEEDS_REAL_STORE = _HAS_NUMPY and _HAS_SQLITE_VEC and _has_fts5()


class _MemorySearchStore:
    def __init__(self) -> None:
        self.chunks: dict[int, ChunkRow] = {
            1: ChunkRow(1, 1, "/notes/a.md", "markdown", "alpha keyword vector", "§A", 1.0),
            2: ChunkRow(2, 2, "/notes/b.md", "markdown", "beta only vector", "§B", 1.0),
            3: ChunkRow(3, 3, "/notes/c.md", "text", "alpha only keyword", "", 0.0),
        }

    def has_chunks(self) -> bool:
        return True

    def keyword_search(
        self, match_query: str, limit: int, *, path_prefix: str | None = None
    ) -> list[tuple[int, float]]:
        del match_query, path_prefix
        return [(3, -4.5), (1, -2.25)][:limit]

    def vector_search(
        self, vec: object, limit: int, *, path_prefix: str | None = None
    ) -> list[tuple[int, float]]:
        del vec, path_prefix
        return [(2, 0.15), (1, 0.35)][:limit]

    def fetch_chunks(self, ids: list[int]) -> dict[int, ChunkRow]:
        return {cid: self.chunks[cid] for cid in ids if cid in self.chunks}


class _FixedReranker:
    def score(self, query: str, passages: list[str]) -> list[float]:
        del query
        return [float(index) for index, _ in enumerate(passages)]


class SearchExplainTests(unittest.TestCase):
    def test_explain_false_leaves_hit_explain_none(self) -> None:
        searcher = Searcher(_MemorySearchStore(), FakeEmbedder(dim=8))  # type: ignore[arg-type]

        hits = searcher.get_hits("alpha", mode="hybrid", k=3)

        self.assertTrue(hits)
        self.assertTrue(all(hit.explain is None for hit in hits))

    def test_explain_records_ranks_and_scores_across_all_modes(self) -> None:
        reranker = _FixedReranker()
        searcher = Searcher(
            _MemorySearchStore(),  # type: ignore[arg-type]
            FakeEmbedder(dim=8),
            reranker=reranker,  # type: ignore[arg-type]
        )

        kw_hits = searcher.get_hits("alpha", mode="keyword", k=3, explain=True)
        self.assertEqual([hit.chunk_id for hit in kw_hits], [3, 1])
        self.assertEqual(
            kw_hits[0].explain,
            HitExplain(keyword_rank=1, keyword_bm25=-4.5),
        )

        vec_hits = searcher.get_hits("alpha", mode="vector", k=3, explain=True)
        self.assertEqual([hit.chunk_id for hit in vec_hits], [2, 1])
        self.assertEqual(
            vec_hits[0].explain,
            HitExplain(vector_rank=1, vector_distance=0.15),
        )

        hybrid_hits = searcher.get_hits("alpha", mode="hybrid", k=3, explain=True)
        by_id = {hit.chunk_id: hit for hit in hybrid_hits}
        explain_1 = by_id[1].explain
        self.assertIsNotNone(explain_1)
        assert explain_1 is not None
        self.assertEqual(explain_1.keyword_rank, 2)
        self.assertAlmostEqual(explain_1.keyword_bm25 or 0.0, -2.25)
        self.assertEqual(explain_1.vector_rank, 2)
        self.assertAlmostEqual(explain_1.vector_distance or 0.0, 0.35)
        self.assertAlmostEqual(explain_1.rrf_score or 0.0, 2.0 / 62.0)
        self.assertIsNone(explain_1.rerank_rank)

        rerank_hits = searcher.get_hits(
            "alpha", mode="hybrid", k=3, rerank=True, explain=True
        )
        self.assertEqual(len(rerank_hits), 3)
        top_explain = rerank_hits[0].explain
        self.assertIsNotNone(top_explain)
        assert top_explain is not None
        self.assertEqual(top_explain.rerank_rank, 1)
        self.assertEqual(top_explain.rerank_score, 2.0)


@unittest.skipUnless(_NEEDS_REAL_STORE, "numpy, sqlite-vec and FTS5 are required")
class DaemonExplainRoundtripTests(unittest.TestCase):
    @staticmethod
    @contextmanager
    def _running(searcher: Searcher, tmp: Path) -> Iterator[Path]:
        sock_path = tmp / "s.sock"
        stop = threading.Event()
        errors: list[BaseException] = []

        def target() -> None:
            try:
                daemon.serve(searcher, sock_path, stop)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        deadline = time.monotonic() + 5.0
        while not sock_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        try:
            yield sock_path
        finally:
            stop.set()
            thread.join(timeout=5.0)
            if errors:
                raise AssertionError(f"daemon raised: {errors}")

    def test_daemon_search_preserves_explain_payload(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="seekr-exp-") as tmp:
            base = Path(tmp)
            root = base / "docs"
            root.mkdir()
            (root / "a.md").write_text("neural backpropagation gradients", encoding="utf-8")
            cfg = Config(roots=(root,), db_path=base / "index.db", embed_model="fake")
            embedder = FakeEmbedder(dim=8)
            store = Store(cfg.db_path, "fake", 8, vector_quant="int8")
            try:
                Indexer(store, embedder, cfg).sync([root])
                searcher = Searcher(store, embedder)
                inprocess = searcher.search("neural", mode="hybrid", k=5, explain=True)
                with self._running(searcher, base) as sock_path:
                    daemon_hits = client.try_daemon_search(
                        {
                            "op": "search",
                            "query": "neural",
                            "k": 5,
                            "mode": "hybrid",
                            "per_file": 2,
                            "rerank": False,
                            "path_prefix": None,
                            "explain": True,
                        },
                        sock_path=sock_path,
                    )
            finally:
                store.close()

        self.assertEqual(daemon_hits, inprocess)
        assert daemon_hits is not None and daemon_hits[0].explain is not None
        self.assertEqual(daemon_hits[0].explain.keyword_rank, 1)
        self.assertEqual(daemon_hits[0].explain.vector_rank, 1)


@unittest.skipUnless(_HAS_TYPER, "typer is not installed")
class CliJsonAndExplainTests(unittest.TestCase):
    def test_search_json_empty_returns_valid_json_array(self) -> None:
        from typer.testing import CliRunner

        with mock.patch("seekr.cli_search.get_hits", return_value=[]):
            result = CliRunner().invoke(app, ["search", "missing", "--json"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.stdout), [])

    def test_search_json_and_explain_include_retriever_breakdown(self) -> None:
        from typer.testing import CliRunner

        hit = Hit(
            chunk_id=4,
            path="/notes/a.md",
            locator="§Intro",
            loc_start=1.0,
            snippet="neural weights",
            score=0.0325,
            kind="markdown",
            explain=HitExplain(
                keyword_rank=1,
                keyword_bm25=-3.5,
                vector_rank=2,
                vector_distance=0.41,
                rrf_score=0.0325,
            ),
        )
        with mock.patch("seekr.cli_search.get_hits", return_value=[hit]) as get_hits:
            json_res = CliRunner().invoke(app, ["search", "neural", "--json", "--explain"])
            text_res = CliRunner().invoke(app, ["search", "neural", "--explain"])

        self.assertEqual(json_res.exit_code, 0, json_res.output)
        self.assertEqual(text_res.exit_code, 0, text_res.output)
        self.assertEqual(get_hits.call_args_list[0].kwargs["explain"], True)

        parsed = json.loads(json_res.stdout)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["chunk_id"], 4)
        self.assertEqual(parsed[0]["explain"]["keyword_rank"], 1)
        self.assertEqual(parsed[0]["explain"]["vector_rank"], 2)

        self.assertIn("explain:", text_res.output)
        self.assertIn("keyword=#1 (bm25=-3.5000)", text_res.output)
        self.assertIn("vector=#2 (dist=0.4100)", text_res.output)
        self.assertIn("rrf=0.0325", text_res.output)
