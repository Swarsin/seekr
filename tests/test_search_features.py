"""Tests for search --json output and --explain retriever attribution."""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest import mock

from seekr import cli_search, client, daemon
from seekr.cli import app
from seekr.config import Config
from seekr.embed import FakeEmbedder
from seekr.filters import normalise_kind, parse_since, parse_week, resolve_module
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


class FilterParsingTests(unittest.TestCase):
    def test_normalise_kind_and_rejects_unknown(self) -> None:
        self.assertIsNone(normalise_kind(None))
        self.assertIsNone(normalise_kind("   "))
        self.assertEqual(normalise_kind(" Code "), "code")
        self.assertEqual(normalise_kind("PDF"), "pdf")
        with self.assertRaises(ValueError) as raised:
            normalise_kind("image")
        self.assertIn("expected one of", str(raised.exception))

    def test_parse_since_durations_and_iso_dates(self) -> None:
        now = 1_700_000_000.0
        self.assertIsNone(parse_since(None, now=now))
        self.assertIsNone(parse_since("  ", now=now))
        self.assertEqual(parse_since("30m", now=now), int((now - 1800) * 1_000_000_000))
        self.assertEqual(parse_since("12h", now=now), int((now - 12 * 3600) * 1_000_000_000))
        self.assertEqual(parse_since("7d", now=now), int((now - 7 * 86400) * 1_000_000_000))
        self.assertEqual(parse_since("2w", now=now), int((now - 14 * 86400) * 1_000_000_000))

        expected_iso = int(datetime.fromisoformat("2026-01-15").timestamp() * 1_000_000_000)
        self.assertEqual(parse_since("2026-01-15"), expected_iso)

        for bad in ("0d", "-3d", "yesterday", "2026-99-99"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_since(bad, now=now)


@unittest.skipUnless(_NEEDS_REAL_STORE, "numpy, sqlite-vec and FTS5 are required")
class RealStoreFilterTests(unittest.TestCase):
    def test_kind_and_since_filters_apply_before_topk_across_modes_and_daemon(self) -> None:
        now = time.time()
        recent_mtime = now - 600
        old_mtime = now - 30 * 86400

        with tempfile.TemporaryDirectory(dir="/tmp", prefix="seekr-flt-") as tmp:
            base = Path(tmp)
            root = base / "repo"
            root.mkdir()
            # Create > 50 (default candidate depth) markdown docs with higher term frequency
            # so a post-limit filter would miss the code file completely.
            for idx in range(55):
                md_file = root / f"note_{idx:02d}.md"
                md_file.write_text("zephyr signal " * 5, encoding="utf-8")
                os.utime(md_file, (recent_mtime, recent_mtime))

            code_recent = root / "solver.py"
            code_recent.write_text("def solve():\n    return 'zephyr'\n", encoding="utf-8")
            os.utime(code_recent, (recent_mtime, recent_mtime))

            code_old = root / "legacy.py"
            code_old.write_text("def legacy():\n    return 'zephyr'\n", encoding="utf-8")
            os.utime(code_old, (old_mtime, old_mtime))

            cfg = Config(roots=(root,), db_path=base / "index.db", embed_model="fake")
            embedder = FakeEmbedder(dim=8)
            store = Store(cfg.db_path, "fake", 8, vector_quant="int8")
            try:
                Indexer(store, embedder, cfg).sync([root])
                searcher = Searcher(store, embedder)
                recent_ns = int((now - 7 * 86400) * 1_000_000_000)

                for mode in ("keyword", "vector", "hybrid"):
                    with self.subTest(mode=mode):
                        code_hits = searcher.get_hits(
                            "zephyr",
                            k=5,
                            mode=mode,
                            kind="code",
                            since_mtime_ns=recent_ns,
                        )
                        self.assertEqual(
                            [hit.path for hit in code_hits],
                            [str(code_recent.resolve())],
                        )

                        all_code_hits = searcher.get_hits(
                            "zephyr",
                            k=5,
                            mode=mode,
                            kind="code",
                        )
                        self.assertEqual(
                            {hit.path for hit in all_code_hits},
                            {str(code_recent.resolve()), str(code_old.resolve())},
                        )

                with DaemonExplainRoundtripTests._running(searcher, base) as sock_path:
                    daemon_hits = client.try_daemon_search(
                        {
                            "op": "search",
                            "query": "zephyr",
                            "k": 5,
                            "mode": "hybrid",
                            "per_file": 2,
                            "rerank": False,
                            "path_prefix": None,
                            "kind": "code",
                            "since_mtime_ns": recent_ns,
                        },
                        sock_path=sock_path,
                    )
                assert daemon_hits is not None
                self.assertEqual([hit.path for hit in daemon_hits], [str(code_recent.resolve())])

                with (
                    mock.patch("seekr.cli_search._daemon_search", return_value=None),
                    mock.patch("seekr.cli_search._build_searcher", return_value=searcher),
                ):
                    cli_hits = cli_search.get_hits(
                        "zephyr",
                        k=5,
                        mode="keyword",
                        kind="code",
                        since="7d",
                        cfg=cfg,
                    )
                    # Note: searcher was already closed by get_hits finally block above.
                    self.assertEqual([hit.path for hit in cli_hits], [str(code_recent.resolve())])
            finally:
                store.close()


@unittest.skipUnless(_HAS_TYPER, "typer is not installed")
class CliFilterTests(unittest.TestCase):
    def test_search_forwards_kind_and_since_and_rejects_bad_values(self) -> None:
        from typer.testing import CliRunner

        with mock.patch("seekr.cli_search.get_hits", return_value=[]) as get_hits:
            ok = CliRunner().invoke(
                app, ["search", "zephyr", "--kind", "code", "--since", "7d", "--json"]
            )
        self.assertEqual(ok.exit_code, 0, ok.output)
        get_hits.assert_called_once_with(
            "zephyr",
            k=10,
            mode="hybrid",
            per_file=2,
            rerank=False,
            path_prefix=None,
            kind="code",
            since="7d",
        )

        bad_kind = CliRunner().invoke(app, ["search", "zephyr", "--kind", "audio"])
        self.assertEqual(bad_kind.exit_code, 1)
        self.assertIn("--kind", bad_kind.output)

        bad_since = CliRunner().invoke(app, ["search", "zephyr", "--since", "notadate"])
        self.assertEqual(bad_since.exit_code, 1)
        self.assertIn("--since", bad_since.output)


@unittest.skipUnless(_NEEDS_REAL_STORE, "numpy, sqlite-vec and FTS5 are required")
class RealStoreSimilarTests(unittest.TestCase):
    def test_similar_indexed_and_unindexed_files_with_filters(self) -> None:
        now = time.time()
        recent_mtime = now - 300
        old_mtime = now - 30 * 86400

        with tempfile.TemporaryDirectory(dir="/tmp", prefix="seekr-sim-") as tmp:
            base = Path(tmp)
            root = base / "docs"
            root.mkdir()
            a_md = root / "a.md"
            b_md = root / "b.md"
            c_py = root / "c.py"
            old_md = root / "old.md"
            a_md.write_text("neural network gradient descent optimizer", encoding="utf-8")
            b_md.write_text("neural network optimizer backpropagation", encoding="utf-8")
            c_py.write_text("def energy():\n    return 'quantum hamiltonian'\n", encoding="utf-8")
            old_md.write_text("neural network gradient legacy archive", encoding="utf-8")
            os.utime(a_md, (recent_mtime, recent_mtime))
            os.utime(b_md, (recent_mtime, recent_mtime))
            os.utime(c_py, (recent_mtime, recent_mtime))
            os.utime(old_md, (old_mtime, old_mtime))

            cfg = Config(roots=(root,), db_path=base / "index.db", embed_model="fake")
            embedder = FakeEmbedder(dim=8)
            store = Store(cfg.db_path, "fake", 8, vector_quant="int8")
            try:
                Indexer(store, embedder, cfg).sync([root])
            finally:
                store.close()

            with mock.patch("seekr.cli_search.build_embedder") as build_emb:
                indexed_hits = cli_search.get_similar_hits(a_md, k=5, since="7d", cfg=cfg)
            build_emb.assert_not_called()
            self.assertNotIn(str(a_md.resolve()), [hit.path for hit in indexed_hits])
            self.assertNotIn(str(old_md.resolve()), [hit.path for hit in indexed_hits])
            self.assertEqual(indexed_hits[0].path, str(b_md.resolve()))
            self.assertGreater(indexed_hits[0].score, 0.0)

            draft = base / "draft.md"
            draft.write_text("neural network gradient descent optimizer", encoding="utf-8")
            with mock.patch("seekr.cli_search.build_embedder", return_value=embedder):
                unindexed_hits = cli_search.get_similar_hits(draft, k=5, cfg=cfg)
                code_only = cli_search.get_similar_hits(draft, k=5, kind="code", cfg=cfg)
            self.assertIn(str(a_md.resolve()), [hit.path for hit in unindexed_hits[:2]])
            self.assertEqual([hit.path for hit in code_only], [str(c_py.resolve())])

            with self.assertRaises(cli_search.SearchError):
                cli_search.get_similar_hits(base / "missing.md", cfg=cfg)


class DaemonHelperFeatureTests(unittest.TestCase):
    def test_try_daemon_hits_forwards_explain_and_normalised_filters(self) -> None:
        hit = Hit(
            chunk_id=1,
            path="/notes/a.md",
            locator="L1",
            loc_start=1.0,
            snippet="alpha",
            score=1.0,
            kind="text",
        )
        cfg = Config(
            db_path=Path("/tmp/seekr-daemon-helper/index.db"),
            embed_model="test/embed",
            rerank_model="test/rerank",
        )

        with mock.patch(
            "seekr.cli_search.try_daemon_search", return_value=[hit]
        ) as try_daemon:
            hits = cli_search.try_daemon_hits(
                "alpha",
                k=4,
                mode="keyword",
                per_file=1,
                path_prefix="/notes",
                explain=True,
                kind="CODE",
                since="7d",
                cfg=cfg,
            )

        self.assertEqual(hits, [hit])
        params = try_daemon.call_args.args[0]
        self.assertEqual(params["op"], "search")
        self.assertEqual(params["query"], "alpha")
        self.assertEqual(params["k"], 4)
        self.assertEqual(params["mode"], "keyword")
        self.assertEqual(params["per_file"], 1)
        self.assertEqual(params["path_prefix"], "/notes")
        self.assertTrue(params["explain"])
        self.assertEqual(params["kind"], "code")
        self.assertIsInstance(params["since_mtime_ns"], int)

    def test_get_hits_use_daemon_false_never_calls_the_daemon(self) -> None:
        searcher = mock.Mock()
        searcher.get_hits.return_value = []
        cfg = Config(db_path=Path("/tmp/seekr-daemon-helper/index.db"))

        with (
            mock.patch(
                "seekr.cli_search.try_daemon_search",
                side_effect=AssertionError("use_daemon=False must not probe"),
            ) as try_daemon,
            mock.patch(
                "seekr.cli_search._build_searcher", return_value=searcher
            ),
            mock.patch("seekr.cli_search.lifecycle_lock"),
        ):
            cli_search.get_hits("alpha", cfg=cfg, use_daemon=False)

        try_daemon.assert_not_called()


@unittest.skipUnless(_HAS_TYPER, "typer is not installed")
class CliSimilarTests(unittest.TestCase):
    def test_similar_command_forwards_args_and_formats_json_and_text(self) -> None:
        from typer.testing import CliRunner

        hit = Hit(
            chunk_id=9,
            path="/docs/related.md",
            locator="§Methods",
            loc_start=12.0,
            snippet="similar section",
            score=0.875,
            kind="markdown",
        )
        with mock.patch("seekr.cli_search.get_similar_hits", return_value=[hit]) as sim_mock:
            text_res = CliRunner().invoke(
                app,
                ["similar", "/docs/source.md", "-k", "4", "--kind", "markdown", "--since", "7d"],
            )
            json_res = CliRunner().invoke(app, ["similar", "/docs/source.md", "--json"])

        self.assertEqual(text_res.exit_code, 0, text_res.output)
        self.assertEqual(json_res.exit_code, 0, json_res.output)
        self.assertEqual(sim_mock.call_args_list[0].kwargs["k"], 4)
        self.assertEqual(sim_mock.call_args_list[0].kwargs["kind"], "markdown")
        self.assertEqual(sim_mock.call_args_list[0].kwargs["since"], "7d")
        self.assertIn("/docs/related.md", text_res.output)
        self.assertIn("0.8750", text_res.output)

        parsed = json.loads(json_res.stdout)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["path"], "/docs/related.md")

        with mock.patch(
            "seekr.cli_search.get_similar_hits",
            side_effect=cli_search.SearchError("file not found"),
        ):
            err_res = CliRunner().invoke(app, ["similar", "/docs/missing.md"])
        self.assertEqual(err_res.exit_code, 1)
        self.assertIn("file not found", err_res.output)


class ModuleWeekFilterTests(unittest.TestCase):
    def test_parse_week(self) -> None:
        self.assertIsNone(parse_week(None))
        self.assertIsNone(parse_week("   "))
        self.assertEqual(parse_week("4"), (4, 4))
        self.assertEqual(parse_week(" 3 - 5 "), (3, 5))
        self.assertEqual(parse_week("1-99"), (1, 99))
        for bad in ("5-3", "0", "abc", "100", "3-", "0-4", "1-100"):
            with (
                self.subTest(bad=bad),
                self.assertRaisesRegex(ValueError, f"invalid --week value {bad!r}"),
            ):
                parse_week(bad)
        with self.assertRaises(TypeError):
            parse_week(4)  # type: ignore[arg-type]

    def test_resolve_module(self) -> None:
        known = ["machine-learning", "maths", "data-structures"]
        self.assertIsNone(resolve_module(None, known))
        self.assertIsNone(resolve_module("   ", known))
        self.assertIsNone(resolve_module(None, []))
        self.assertEqual(resolve_module("Machine Learning", known), "machine-learning")
        self.assertEqual(resolve_module("data", known), "data-structures")
        self.assertEqual(resolve_module("ml", ["ml", "ml-labs"]), "ml")
        self.assertEqual(resolve_module("módulo ü", ["Módulo Ü", "4CCS1ML"]), "Módulo Ü")
        self.assertEqual(resolve_module("4ccs1ml", ["Módulo Ü", "4CCS1ML"]), "4CCS1ML")
        with self.assertRaisesRegex(
            ValueError, "ambiguous; matches: machine-learning, machine_learning"
        ):
            resolve_module("Machine Learning", ["machine_learning", "machine-learning"])
        with self.assertRaisesRegex(ValueError, "ambiguous; matches: machine-learning, maths"):
            resolve_module("ma", known)
        with self.assertRaisesRegex(
            ValueError,
            "unknown module 'xyz'; known modules: data-structures, machine-learning, maths",
        ):
            resolve_module("xyz", known)
        with self.assertRaisesRegex(ValueError, "unknown module '---'"):
            resolve_module("---", known)
        with self.assertRaisesRegex(
            ValueError, r"no modules are labelled yet; run `seekr index` first"
        ):
            resolve_module("ml", [])
        with self.assertRaises(TypeError):
            resolve_module(123, known)  # type: ignore[arg-type]

    @unittest.skipUnless(_NEEDS_REAL_STORE, "numpy, sqlite-vec and FTS5 are required")
    def test_searcher_filters_and_hit_labels(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="seekr-mw-") as tmp:
            base = Path(tmp)
            root = base / "course"
            (root / "ml" / "week1").mkdir(parents=True)
            (root / "ml" / "week3").mkdir(parents=True)
            (root / "ds" / "week1").mkdir(parents=True)
            (root / "machine-learning" / "week2").mkdir(parents=True)

            ml_w1 = root / "ml" / "week1" / "intro.md"
            ml_w3 = root / "ml" / "week3" / "descent.md"
            ds_w1 = root / "ds" / "week1" / "trees.md"
            ml_long_w2 = root / "machine-learning" / "week2" / "notes.md"
            ml_w1.write_text("gradient basics in week one", encoding="utf-8")
            ml_w3.write_text("gradient descent optimisation in week three", encoding="utf-8")
            ds_w1.write_text("gradient of tree cost in data structures", encoding="utf-8")
            ml_long_w2.write_text("gradient flow in machine learning week two", encoding="utf-8")

            cfg = Config(roots=(root,), db_path=base / "index.db", embed_model="fake")
            embedder = FakeEmbedder(dim=8)
            store = Store(cfg.db_path, "fake", 8, vector_quant="int8")
            try:
                Indexer(store, embedder, cfg).sync([root])
                searcher = Searcher(store, embedder)

                for mode in ("keyword", "vector", "hybrid"):
                    with self.subTest(mode=mode):
                        hits = searcher.get_hits(
                            "gradient",
                            k=10,
                            mode=mode,  # type: ignore[arg-type]
                            module="ml",
                            week_range=(2, 3),
                        )
                        self.assertEqual([(h.module, h.week) for h in hits], [("ml", 3)])

                search_hits = searcher.search(
                    "gradient", k=10, mode="hybrid", module="ML", week_range=(2, 3)
                )
                self.assertEqual([(h.module, h.week) for h in search_hits], [("ml", 3)])

                display_hits = searcher.get_hits(
                    "gradient", k=10, mode="hybrid", module="Machine Learning"
                )
                self.assertEqual(
                    [(h.module, h.week) for h in display_hits],
                    [("machine-learning", 2)],
                )

                week1_hits = searcher.get_hits(
                    "gradient", k=10, mode="hybrid", week_range=(1, 1)
                )
                self.assertEqual(
                    {(h.module, h.week) for h in week1_hits},
                    {("ml", 1), ("ds", 1)},
                )

                query_vecs = store.get_document_vectors(str(ds_w1.resolve()))
                sim_hits = searcher.similar(
                    query_vecs, k=10, module="ml", week_range=(2, 3)
                )
                self.assertEqual([(h.module, h.week) for h in sim_hits], [("ml", 3)])
                sim_prefix_hits = searcher.similar(query_vecs, k=10, module="mach")
                self.assertEqual(
                    [(h.module, h.week) for h in sim_prefix_hits],
                    [("machine-learning", 2)],
                )

                with self.assertRaisesRegex(ValueError, "ambiguous; matches: machine-learning, ml"):
                    searcher.get_hits("gradient", module="m")

                with self.assertRaisesRegex(ValueError, "unknown module 'bio'"):
                    searcher.get_hits("gradient", module="bio")
                with self.assertRaisesRegex(ValueError, "unknown module 'bio'"):
                    searcher.similar(query_vecs, module="bio")
                for bad_range in ((3, 2), (0, 2), (1, 100), (True, 2), "12"):
                    with self.subTest(bad_range=bad_range):
                        with self.assertRaises(ValueError):
                            searcher.get_hits(
                                "gradient", week_range=bad_range  # type: ignore[arg-type]
                            )
                        with self.assertRaises(ValueError):
                            searcher.similar(
                                query_vecs, week_range=bad_range  # type: ignore[arg-type]
                            )
            finally:
                store.close()

            empty_store = Store(base / "empty.db", "fake", 8, vector_quant="int8")
            try:
                empty_searcher = Searcher(empty_store, embedder)
                with self.assertRaisesRegex(ValueError, "no modules are labelled yet"):
                    empty_searcher.get_hits("gradient", module="ml")
                with self.assertRaisesRegex(ValueError, "no modules are labelled yet"):
                    empty_searcher.similar(
                        [embedder.embed_query("gradient")], module="ml"
                    )
            finally:
                empty_store.close()


@unittest.skipUnless(_HAS_TYPER, "typer is not installed")
class CliModuleWeekTests(unittest.TestCase):
    def test_search_module_week_options_reach_get_hits(self) -> None:
        from typer.testing import CliRunner

        with mock.patch("seekr.cli_search.get_hits", return_value=[]) as get_hits:
            result = CliRunner().invoke(
                app, ["search", "x", "--module", "ml", "--week", "2-3"]
            )

        self.assertEqual(result.exit_code, 0, result.output)
        get_hits.assert_called_once_with(
            "x",
            k=10,
            mode="hybrid",
            per_file=2,
            rerank=False,
            path_prefix=None,
            module="ml",
            week="2-3",
        )

    def test_text_output_has_label(self) -> None:
        from typer.testing import CliRunner

        hits = [
            Hit(
                chunk_id=1,
                path="/course/machine-learning/week4/trees.md",
                locator="§Trees",
                loc_start=1.0,
                snippet="decision tree notes",
                score=0.875,
                kind="markdown",
                module="machine-learning",
                week=4,
            ),
            Hit(
                chunk_id=2,
                path="/course/4CCS1ML/syllabus.md",
                locator="L1",
                loc_start=1.0,
                snippet="syllabus",
                score=0.5,
                kind="markdown",
                module="4CCS1ML",
                week=None,
            ),
            Hit(
                chunk_id=3,
                path="/course/week2/slides.md",
                locator="L1",
                loc_start=1.0,
                snippet="slides",
                score=0.4,
                kind="markdown",
                module=None,
                week=2,
            ),
            Hit(
                chunk_id=4,
                path="/course/readme.md",
                locator="L1",
                loc_start=1.0,
                snippet="readme",
                score=0.3,
                kind="markdown",
            ),
        ]
        with mock.patch("seekr.cli_search.get_hits", return_value=hits):
            result = CliRunner().invoke(app, ["search", "tree"])

        self.assertEqual(result.exit_code, 0, result.output)
        lines = result.output.splitlines()
        self.assertIn(
            "machine learning · Week 4 · /course/machine-learning/week4/trees.md  (§Trees)  0.8750",
            lines,
        )
        self.assertIn("4CCS1ML · /course/4CCS1ML/syllabus.md  (L1)  0.5000", lines)
        self.assertIn("Week 2 · /course/week2/slides.md  (L1)  0.4000", lines)
        self.assertIn("/course/readme.md  (L1)  0.3000", lines)

    def test_bad_week_exits_1_with_message(self) -> None:
        from typer.testing import CliRunner

        result = CliRunner().invoke(app, ["search", "x", "--week", "5-3"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("invalid --week value", result.output)

    def test_similar_and_pick_forward_module_and_week(self) -> None:
        from typer.testing import CliRunner

        with mock.patch("seekr.cli_search.get_similar_hits", return_value=[]) as sim_mock:
            sim_res = CliRunner().invoke(
                app,
                ["similar", "/course/a.md", "--module", "ml", "--week", "2-3"],
            )
        self.assertEqual(sim_res.exit_code, 0, sim_res.output)
        self.assertEqual(sim_mock.call_args.kwargs["module"], "ml")
        self.assertEqual(sim_mock.call_args.kwargs["week"], "2-3")

        bad_sim = CliRunner().invoke(app, ["similar", "/course/a.md", "--week", "abc"])
        self.assertEqual(bad_sim.exit_code, 1)
        self.assertIn("invalid --week value", bad_sim.output)

        with mock.patch("seekr.cli_open.get_hits", return_value=[]) as pick_mock:
            pick_res = CliRunner().invoke(
                app,
                ["pick", "x", "--module", "ml", "--week", "2-3"],
            )
        self.assertEqual(pick_res.exit_code, 0, pick_res.output)
        pick_mock.assert_called_once_with(
            "x",
            k=10,
            mode="hybrid",
            per_file=2,
            rerank=False,
            path_prefix=None,
            module="ml",
            week="2-3",
        )

        bad_pick = CliRunner().invoke(app, ["pick", "x", "--week", "0"])
        self.assertEqual(bad_pick.exit_code, 1)
        self.assertIn("invalid --week value", bad_pick.output)
