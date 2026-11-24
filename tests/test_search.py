"""Tests for hybrid retrieval and the standalone ``seekr search`` entry point.

The default suite deliberately runs without numpy, sqlite-vec or Typer: the
retrieval logic is exercised with small recording stores that return the exact
rankings configured by each test, and hostile FTS strings are checked against
the stdlib SQLite FTS5 implementation.  Tests that need the real production
:class:`~seekr.store.Store` skip unless numpy, sqlite-vec and FTS5 are present.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from _platform import requires_legacy_daemon, use_posix_stored_paths
from seekr import cli_search
from seekr.config import Config
from seekr.embed import FakeEmbedder
from seekr.models import ChunkSpec, Hit, text_sha
from seekr.search import Searcher, fts_query, rrf
from seekr.store import (
    ChunkRow,
    ModelMismatchError,
    SchemaVersionError,
    Store,
    StoredChunk,
)

try:  # pragma: no cover - pytest is an optional development dependency
    import pytest
except ImportError:  # pragma: no cover
    pytest = None


def _has_fts5() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        options = [row[0] for row in conn.execute("PRAGMA compile_options")]
    finally:
        conn.close()
    return "ENABLE_FTS5" in options


def _slow(test_class: type[unittest.TestCase]) -> type[unittest.TestCase]:
    if pytest is None:
        return test_class
    return pytest.mark.slow(test_class)


_HAS_FTS5 = _has_fts5()
_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_HAS_SQLITE_VEC = importlib.util.find_spec("sqlite_vec") is not None
_HAS_TYPER = importlib.util.find_spec("typer") is not None
_HAS_SENTENCE_TRANSFORMERS = importlib.util.find_spec("sentence_transformers") is not None
_MODEL_DIR_ENV = "SEEKR_TEST_EMBED_MODEL_DIR"
_MODEL_DIR = os.environ.get(_MODEL_DIR_ENV)
_OPTED_IN = bool(_MODEL_DIR and _MODEL_DIR.strip())
_NEEDS_REAL_STORE = unittest.skipUnless(
    _HAS_NUMPY and _HAS_SQLITE_VEC and _HAS_FTS5,
    "numpy, sqlite-vec and a loadable FTS5 build are required",
)


def _chunk(
    chunk_id: int,
    path: str,
    text: str,
    *,
    locator: str = "",
    loc_start: float = 0.0,
    kind: str = "text",
    doc_id: int | None = None,
) -> ChunkRow:
    return ChunkRow(
        id=chunk_id,
        doc_id=chunk_id if doc_id is None else doc_id,
        path=path,
        kind=kind,
        text=text,
        locator=locator,
        loc_start=loc_start,
    )


def _prefix_matches(path: str, prefix: str | None) -> bool:
    if prefix is None:
        return True
    return path == prefix or path.startswith(prefix + "/")


class _BytesVector:
    """Minimal stand-in for a float32 vector when only keyword paths are used."""

    def __init__(self, payload: bytes = b"\x00" * 8) -> None:
        self.payload = payload

    def tobytes(self) -> bytes:
        return self.payload


@contextlib.contextmanager
def _fake_vector_backend():
    """Give the Store a plain vec table so stdlib FTS tests need no sqlite-vec."""

    def create_table(conn: sqlite3.Connection, dim: int) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks_vec ("
            "rowid INTEGER PRIMARY KEY, embedding BLOB NOT NULL)"
        )

    def serialize(vector: _BytesVector) -> bytes:
        return vector.tobytes()

    with (
        mock.patch("seekr.store._load_vector_extension", lambda conn: None),
        mock.patch("seekr.store._create_vector_table", create_table),
        mock.patch("seekr.store._serialize_vector", serialize),
    ):
        yield


class _RecordingStore:
    """A tiny duck-typed store that returns the rankings a test configures."""

    def __init__(
        self,
        chunks: list[ChunkRow],
        *,
        keyword_order: list[int] | None = None,
        vector_order: list[int] | None = None,
        keyword_scores: dict[int, float] | None = None,
        vector_scores: dict[int, float] | None = None,
    ) -> None:
        self.chunks = {chunk.id: chunk for chunk in chunks}
        self.keyword_order = list(keyword_order or [])
        self.vector_order = list(vector_order or [])
        self.keyword_scores = dict(keyword_scores or {})
        self.vector_scores = dict(vector_scores or {})
        self.depth = 0
        self.max_depth = 0
        self.keyword_calls: list[tuple[str, int, int, str | None]] = []
        self.vector_calls: list[tuple[object, int, int, str | None]] = []
        self.fetch_calls: list[tuple[tuple[int, ...], int]] = []

    @contextlib.contextmanager
    def transaction(self):
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        try:
            yield
        finally:
            self.depth -= 1

    def stats(self) -> dict[str, int]:
        return {
            "documents": len({chunk.path for chunk in self.chunks.values()}),
            "chunks": len(self.chunks),
        }

    def _scoped(self, ids: list[int], path_prefix: str | None) -> list[int]:
        if path_prefix is None:
            return ids
        return [
            chunk_id
            for chunk_id in ids
            if chunk_id in self.chunks
            and _prefix_matches(self.chunks[chunk_id].path, path_prefix)
        ]

    def keyword_search(
        self,
        match_query: str,
        limit: int,
        *,
        path_prefix: str | None = None,
    ) -> list[tuple[int, float]]:
        self.keyword_calls.append((match_query, int(limit), self.depth, path_prefix))
        ids = self._scoped(self.keyword_order, path_prefix)[: int(limit)]
        return [
            (chunk_id, self.keyword_scores.get(chunk_id, -float(rank)))
            for rank, chunk_id in enumerate(ids, start=1)
        ]

    def vector_search(
        self,
        vec: object,
        limit: int,
        *,
        path_prefix: str | None = None,
    ) -> list[tuple[int, float]]:
        self.vector_calls.append((vec, int(limit), self.depth, path_prefix))
        ids = self._scoped(self.vector_order, path_prefix)[: int(limit)]
        return [
            (chunk_id, self.vector_scores.get(chunk_id, float(rank)))
            for rank, chunk_id in enumerate(ids, start=1)
        ]

    def fetch_chunks(self, ids: list[int]) -> dict[int, ChunkRow]:
        self.fetch_calls.append((tuple(ids), self.depth))
        return {
            chunk_id: self.chunks[chunk_id]
            for chunk_id in sorted({int(chunk_id) for chunk_id in ids})
            if chunk_id in self.chunks
        }


class _RecordingEmbedder:
    model_name = "recording"
    dim = 2

    def __init__(self, store: _RecordingStore | None = None) -> None:
        self.store = store
        self.queries: list[str] = []
        self.depth_at_embed: int | None = None

    def embed_documents(self, texts: list[str]) -> Any:
        raise AssertionError("search must not embed documents")

    def embed_query(self, text: str) -> Any:
        self.queries.append(text)
        if self.store is not None:
            self.depth_at_embed = self.store.depth
        return [1.0, 0.0]


class _ExplodingEmbedder:
    model_name = "exploding"
    dim = 2

    def embed_documents(self, texts: list[str]) -> Any:
        raise AssertionError("search must not embed documents")

    def embed_query(self, text: str) -> Any:
        raise AssertionError("keyword-only search must not embed the query")


class _RecordingReranker:
    def __init__(
        self,
        scores_by_substring: dict[str, float] | None = None,
        store: _RecordingStore | None = None,
    ) -> None:
        self.scores_by_substring = dict(scores_by_substring or {})
        self.store = store
        self.calls: list[tuple[str, list[str]]] = []
        self.depth_at_score: int | None = None

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        if self.store is not None:
            self.depth_at_score = self.store.depth
        scores: list[float] = []
        for passage in passages:
            matched = 0.0
            for needle, value in self.scores_by_substring.items():
                if needle in passage:
                    matched = float(value)
                    break
            scores.append(matched)
        return scores


class QueryBuildingTests(unittest.TestCase):
    def test_fts_query_wraps_plain_words(self) -> None:
        self.assertEqual(fts_query('say "hello'), '"say" OR "hello"')
        self.assertEqual(fts_query("a AND -b"), '"a" OR "AND" OR "b"')
        self.assertEqual(fts_query("col:val (x*)"), '"col" OR "val" OR "x"')
        self.assertEqual(fts_query("it's"), '"it" OR "s"')
        self.assertEqual(fts_query("4CCS1ML café"), '"4CCS1ML" OR "café"')

    def test_fts_query_empty_returns_empty_string(self) -> None:
        for raw in ("", "   ", "!!!", "-*():", '"', "\n\t"):
            with self.subTest(raw=raw):
                self.assertEqual(fts_query(raw), "")

    @unittest.skipUnless(_HAS_FTS5, "SQLite was not built with FTS5")
    def test_fts_query_output_is_safe_for_real_sqlite_fts(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE VIRTUAL TABLE docs USING fts5(text)")
            conn.execute("INSERT INTO docs(text) VALUES ('say hello col val x it s')")
            for raw in ('say "hello', "a AND -b", "col:val (x*)", "it's", "-*():"):
                with self.subTest(raw=raw):
                    match = fts_query(raw)
                    if match:
                        rows = conn.execute(
                            "SELECT rowid FROM docs WHERE docs MATCH ?", (match,)
                        ).fetchall()
                    else:
                        rows = []
                    self.assertIsInstance(rows, list)
        finally:
            conn.close()

    def test_rrf_orders_by_combined_rank(self) -> None:
        ranked = rrf([[1, 2, 3], [3, 1, 4]])

        self.assertEqual([chunk_id for chunk_id, _ in ranked], [1, 3, 2, 4])
        scores = dict(ranked)
        self.assertAlmostEqual(scores[1], 1.0 / 61 + 1.0 / 62)
        self.assertAlmostEqual(scores[3], 1.0 / 63 + 1.0 / 61)
        self.assertAlmostEqual(scores[2], 1.0 / 62)
        self.assertAlmostEqual(scores[4], 1.0 / 63)

    def test_rrf_ties_are_deterministic(self) -> None:
        self.assertEqual([chunk_id for chunk_id, _ in rrf([[1], [2]])], [1, 2])
        self.assertEqual(rrf([]), [])
        self.assertEqual(len(rrf([[1, 1]])), 1)


class SearcherBehaviourTests(unittest.TestCase):
    def test_empty_whitespace_query_returns_empty_without_retrieval(self) -> None:
        store = _RecordingStore([_chunk(1, "/notes/a.md", "alpha")], keyword_order=[1])
        embedder = _RecordingEmbedder(store)

        hits = Searcher(store, embedder).search("   \n\t")

        self.assertEqual(hits, [])
        self.assertEqual(store.keyword_calls, [])
        self.assertEqual(store.vector_calls, [])
        self.assertEqual(store.fetch_calls, [])
        self.assertEqual(embedder.queries, [])

    def test_empty_index_returns_empty_without_embedding(self) -> None:
        store = _RecordingStore([])
        embedder = _RecordingEmbedder(store)
        searcher = Searcher(store, embedder)

        for mode in ("keyword", "vector", "hybrid"):
            with self.subTest(mode=mode):
                self.assertEqual(searcher.search("alpha", mode=mode), [])

        self.assertEqual(store.keyword_calls, [])
        self.assertEqual(store.vector_calls, [])
        self.assertEqual(embedder.queries, [])

    def test_keyword_mode_finds_exact_token_and_keeps_raw_score(self) -> None:
        chunk = _chunk(
            7,
            "/notes/codes.md",
            "The error E1234 appears here",
            locator="L3-4",
            loc_start=3.0,
            kind="markdown",
        )
        store = _RecordingStore(
            [chunk], keyword_order=[7], keyword_scores={7: -2.5}
        )

        hits = Searcher(store, _ExplodingEmbedder()).search("E1234", mode="keyword")

        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertEqual(hit.chunk_id, 7)
        self.assertEqual(hit.path, "/notes/codes.md")
        self.assertEqual(hit.locator, "L3-4")
        self.assertEqual(hit.loc_start, 3.0)
        self.assertEqual(hit.kind, "markdown")
        self.assertEqual(hit.score, -2.5)
        self.assertIn("E1234", hit.snippet)
        self.assertEqual(store.keyword_calls[0][0], '"E1234"')
        self.assertGreaterEqual(store.keyword_calls[0][1], 50)
        self.assertEqual(store.keyword_calls[0][2], 1)
        self.assertIsNone(store.keyword_calls[0][3])

    def test_vector_mode_embeds_query_and_keeps_raw_distance(self) -> None:
        chunk = _chunk(7, "/notes/a.md", "alpha beta")
        store = _RecordingStore(
            [chunk], vector_order=[7], vector_scores={7: 0.25}
        )
        embedder = _RecordingEmbedder(store)

        hits = Searcher(store, embedder).search("alpha", mode="vector", k=1)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].score, 0.25)
        self.assertEqual(embedder.queries, ["alpha"])
        self.assertEqual(store.vector_calls[0][0], [1.0, 0.0])
        self.assertEqual(store.vector_calls[0][2], 1)
        self.assertIsNone(store.vector_calls[0][3])

    def test_hybrid_mode_fuses_with_rrf(self) -> None:
        chunks = [
            _chunk(1, "/notes/a.md", "alpha"),
            _chunk(2, "/notes/b.md", "alpha beta"),
            _chunk(3, "/notes/c.md", "beta"),
        ]
        store = _RecordingStore(
            chunks,
            keyword_order=[1, 2],
            vector_order=[2, 3],
            keyword_scores={1: -1.0, 2: -2.0},
            vector_scores={2: 0.1, 3: 0.2},
        )

        hits = Searcher(store, _RecordingEmbedder(store)).search(
            "alpha beta", mode="hybrid", k=3
        )

        self.assertEqual([hit.chunk_id for hit in hits], [2, 1, 3])
        scores = {hit.chunk_id: hit.score for hit in hits}
        self.assertAlmostEqual(scores[2], 1.0 / 61 + 1.0 / 62)
        self.assertAlmostEqual(scores[1], 1.0 / 61)
        self.assertAlmostEqual(scores[3], 1.0 / 62)

    def test_per_file_cap_keeps_at_most_n_per_path(self) -> None:
        chunks = [
            _chunk(1, "/notes/a.md", "common one"),
            _chunk(2, "/notes/a.md", "common two"),
            _chunk(3, "/notes/a.md", "common three"),
            _chunk(4, "/notes/b.md", "common four"),
            _chunk(5, "/notes/b.md", "common five"),
            _chunk(6, "/other/c.md", "common six"),
        ]
        store = _RecordingStore(chunks, keyword_order=[1, 2, 3, 4, 5, 6])

        hits = Searcher(store, _ExplodingEmbedder()).search(
            "common", mode="keyword", k=10, per_file=2
        )

        counts: dict[str, int] = {}
        for hit in hits:
            counts[hit.path] = counts.get(hit.path, 0) + 1
        self.assertLessEqual(max(counts.values()), 2)
        self.assertEqual(counts["/notes/a.md"], 2)
        self.assertEqual(counts["/notes/b.md"], 2)
        self.assertEqual(counts["/other/c.md"], 1)

    def test_path_prefix_is_component_aware(self) -> None:
        chunks = [
            _chunk(1, "/notes/a.md", "needle"),
            _chunk(2, "/notes2/b.md", "needle"),
            _chunk(3, "/notes/sub/c.md", "needle"),
        ]
        store = _RecordingStore(chunks, keyword_order=[1, 2, 3])
        searcher = Searcher(store, _ExplodingEmbedder())

        for prefix in ("/notes", "/notes/"):
            with self.subTest(prefix=prefix):
                hits = searcher.search(
                    "needle", mode="keyword", k=10, path_prefix=prefix
                )
                self.assertEqual([hit.chunk_id for hit in hits], [1, 3])
                self.assertEqual(store.keyword_calls[-1][3], "/notes")

        self.assertEqual(
            [
                hit.chunk_id
                for hit in searcher.search("needle", mode="keyword", path_prefix="/")
            ],
            [1, 2, 3],
        )
        self.assertIsNone(store.keyword_calls[-1][3])
        self.assertEqual(
            [
                hit.chunk_id
                for hit in searcher.search("needle", mode="keyword", path_prefix="")
            ],
            [1, 2, 3],
        )
        self.assertIsNone(store.keyword_calls[-1][3])
        self.assertEqual(
            [
                hit.chunk_id
                for hit in searcher.search(
                    "needle", mode="keyword", path_prefix="/notes/a.md"
                )
            ],
            [1],
        )
        self.assertEqual(store.keyword_calls[-1][3], "/notes/a.md")

    def test_path_prefix_is_passed_to_keyword_retriever_before_its_limit(self) -> None:
        chunks = [
            _chunk(chunk_id, f"/other/f{chunk_id:02d}.md", "needle")
            for chunk_id in range(1, 61)
        ]
        chunks.append(_chunk(61, "/notes/target.md", "needle"))
        store = _RecordingStore(chunks, keyword_order=list(range(1, 62)))

        hits = Searcher(store, _ExplodingEmbedder()).search(
            "needle", mode="keyword", k=1, path_prefix="/notes"
        )

        self.assertEqual([hit.chunk_id for hit in hits], [61])
        self.assertEqual(len(store.keyword_calls), 1)
        self.assertEqual(store.keyword_calls[0][1], 50)
        self.assertEqual(store.keyword_calls[0][3], "/notes")
        self.assertEqual(len(store.fetch_calls), 1)

    def test_path_prefix_is_passed_to_vector_retriever_before_its_limit(self) -> None:
        chunks = [
            _chunk(chunk_id, f"/other/f{chunk_id:02d}.md", "noise")
            for chunk_id in range(1, 61)
        ]
        chunks.append(_chunk(61, "/notes/target.md", "target"))
        store = _RecordingStore(chunks, vector_order=list(range(1, 62)))
        embedder = _RecordingEmbedder(store)

        hits = Searcher(store, embedder).search(
            "anything", mode="vector", k=1, path_prefix="/notes"
        )

        self.assertEqual([hit.chunk_id for hit in hits], [61])
        self.assertEqual(len(store.vector_calls), 1)
        self.assertEqual(store.vector_calls[0][1], 50)
        self.assertEqual(store.vector_calls[0][3], "/notes")
        self.assertEqual(embedder.queries, ["anything"])

    def test_path_prefix_is_passed_to_both_hybrid_retrievers(self) -> None:
        chunks = [
            _chunk(1, "/notes/a.md", "alpha beta"),
            _chunk(2, "/notes2/b.md", "alpha beta"),
        ]
        store = _RecordingStore(
            chunks, keyword_order=[1, 2], vector_order=[2, 1]
        )
        searcher = Searcher(store, _RecordingEmbedder(store))

        searcher.search("alpha beta", mode="hybrid", path_prefix="/notes/")

        self.assertEqual(len(store.keyword_calls), 1)
        self.assertEqual(len(store.vector_calls), 1)
        self.assertEqual(store.keyword_calls[0][3], "/notes")
        self.assertEqual(store.vector_calls[0][3], "/notes")

    def test_snippet_contains_query_term_and_fits_window(self) -> None:
        text = "filler " * 100 + "E1234 is the code"
        store = _RecordingStore(
            [_chunk(1, "/notes/a.md", text)], keyword_order=[1]
        )

        hits = Searcher(store, _ExplodingEmbedder()).search("E1234", mode="keyword")

        self.assertEqual(len(hits), 1)
        self.assertLessEqual(len(hits[0].snippet), 242)
        self.assertIn("E1234", hits[0].snippet)
        self.assertEqual(
            [hits[0].snippet[a:b] for a, b in hits[0].highlights], ["E1234"]
        )

    def test_snippet_prefers_window_with_most_query_terms(self) -> None:
        text = "alpha " + "filler " * 20 + "beta " + "tail " * 200
        store = _RecordingStore(
            [_chunk(1, "/notes/a.md", text)], keyword_order=[1]
        )

        hits = Searcher(store, _ExplodingEmbedder()).search(
            "alpha beta", mode="keyword"
        )

        self.assertLessEqual(len(hits[0].snippet), 242)
        self.assertIn("alpha", hits[0].snippet)
        self.assertIn("beta", hits[0].snippet)
        self.assertEqual(
            [hits[0].snippet[a:b] for a, b in hits[0].highlights], ["alpha", "beta"]
        )

    def test_snippet_falls_back_to_first_window(self) -> None:
        text = "z" * 300
        store = _RecordingStore(
            [_chunk(1, "/notes/a.md", text)], keyword_order=[1]
        )

        hits = Searcher(store, _ExplodingEmbedder()).search(
            "absent", mode="keyword"
        )

        self.assertEqual(hits[0].snippet, text[:240] + "…")
        self.assertEqual(hits[0].highlights, ())

    def test_invalid_mode_raises_value_error(self) -> None:
        store = _RecordingStore([_chunk(1, "/notes/a.md", "alpha")])

        with self.assertRaises(ValueError):
            Searcher(store, _ExplodingEmbedder()).search("alpha", mode="bogus")

    def test_rerank_true_without_reranker_raises_value_error(self) -> None:
        store = _RecordingStore(
            [_chunk(1, "/notes/a.md", "alpha")], keyword_order=[1]
        )

        with self.assertRaises(ValueError):
            Searcher(store, _ExplodingEmbedder()).search(
                "alpha", mode="keyword", rerank=True
            )

    def test_rerank_contract_precedes_zero_k_short_circuit(self) -> None:
        store = _RecordingStore(
            [_chunk(1, "/notes/a.md", "alpha")], keyword_order=[1]
        )
        searcher = Searcher(store, _ExplodingEmbedder())

        with self.assertRaises(ValueError):
            searcher.get_hits("alpha", k=0, mode="keyword", rerank=True)
        self.assertEqual(searcher.get_hits("alpha", k=0, mode="keyword"), [])

    def test_rerank_reorders_and_replaces_scores(self) -> None:
        chunks = [
            _chunk(1, "/notes/a.md", "alpha"),
            _chunk(2, "/notes/b.md", "beta"),
        ]
        store = _RecordingStore(chunks, keyword_order=[1, 2])
        reranker = _RecordingReranker({"beta": 9.0})

        hits = Searcher(store, _ExplodingEmbedder(), reranker).search(
            "alpha beta", mode="keyword", rerank=True
        )

        self.assertEqual([hit.chunk_id for hit in hits], [2, 1])
        self.assertEqual(hits[0].score, 9.0)
        self.assertEqual(
            reranker.calls[0][1], ["alpha", "beta"]
        )

    def test_rerank_runs_after_the_snapshot_is_released(self) -> None:
        store = _RecordingStore(
            [_chunk(1, "/notes/a.md", "alpha beta")], keyword_order=[1]
        )
        reranker = _RecordingReranker({"alpha": 1.0}, store)

        Searcher(store, _ExplodingEmbedder(), reranker).search(
            "alpha", mode="keyword", rerank=True
        )

        self.assertEqual(reranker.depth_at_score, 0)
        self.assertEqual(store.depth, 0)

    def test_rerank_runs_before_the_per_file_cap(self) -> None:
        chunks = [
            _chunk(1, "/notes/a.md", "alpha"),
            _chunk(2, "/notes/a.md", "beta"),
            _chunk(3, "/notes/a.md", "gamma"),
        ]
        store = _RecordingStore(chunks, keyword_order=[1, 2, 3])
        reranker = _RecordingReranker({"gamma": 9.0})

        hits = Searcher(store, _ExplodingEmbedder(), reranker).search(
            "alpha beta gamma", mode="keyword", k=2, per_file=1, rerank=True
        )

        self.assertEqual([hit.chunk_id for hit in hits], [3])

    def test_rerank_sees_at_most_thirty_fused_candidates(self) -> None:
        chunks = [
            _chunk(chunk_id, f"/notes/f{chunk_id:02d}.md", f"text {chunk_id}")
            for chunk_id in range(1, 41)
        ]
        store = _RecordingStore(chunks, keyword_order=list(range(1, 41)))
        reranker = _RecordingReranker()

        Searcher(store, _ExplodingEmbedder(), reranker).search(
            "text", mode="keyword", k=40, rerank=True
        )

        self.assertEqual(len(reranker.calls[0][1]), 30)

    def test_snapshot_covers_retrieval_and_fetch_and_embedding_is_before_lock(
        self,
    ) -> None:
        chunks = [
            _chunk(1, "/notes/a.md", "alpha"),
            _chunk(2, "/notes/b.md", "beta"),
        ]
        store = _RecordingStore(
            chunks, keyword_order=[1, 2], vector_order=[2, 1]
        )
        embedder = _RecordingEmbedder(store)

        Searcher(store, embedder).search("alpha beta", mode="hybrid", k=2)

        self.assertEqual(embedder.depth_at_embed, 0)
        self.assertEqual(store.max_depth, 1)
        self.assertEqual(store.depth, 0)
        self.assertTrue(store.keyword_calls)
        self.assertTrue(store.vector_calls)
        self.assertTrue(all(call[2] == 1 for call in store.keyword_calls))
        self.assertTrue(all(call[2] == 1 for call in store.vector_calls))
        self.assertTrue(store.fetch_calls)
        self.assertTrue(all(call[1] == 1 for call in store.fetch_calls))


class CliEntryPointTests(unittest.TestCase):
    def test_whitespace_query_short_circuits_before_any_backend_or_daemon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(db_path=Path(tmp) / "missing.db")
            with (
                mock.patch("seekr.cli_search.build_embedder") as build_embedder,
                mock.patch("seekr.cli_search.Store") as store,
                mock.patch("seekr.cli_search.try_daemon_search") as try_daemon,
                mock.patch("seekr.cli_search.lifecycle_lock") as lock,
            ):
                for query in ("", "   ", "\t\n"):
                    with self.subTest(query=query):
                        self.assertEqual(
                            cli_search.get_hits(query, mode="hybrid", cfg=cfg), []
                        )

        build_embedder.assert_not_called()
        store.assert_not_called()
        try_daemon.assert_not_called()
        lock.assert_not_called()

    def test_empty_missing_index_vector_modes_skip_model_and_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(db_path=Path(tmp) / "missing.db")
            with (
                mock.patch("seekr.cli_search.build_embedder") as build_embedder,
                mock.patch("seekr.cli_search.Store") as store,
                mock.patch("seekr.cli_search.try_daemon_search", return_value=None),
            ):
                for mode in ("vector", "hybrid"):
                    with self.subTest(mode=mode):
                        self.assertEqual(
                            cli_search.get_hits("alpha", mode=mode, cfg=cfg), []
                        )

        build_embedder.assert_not_called()
        store.assert_not_called()

    def test_empty_index_with_metadata_skips_model_and_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "index.db"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                conn.execute("CREATE TABLE documents(id INTEGER PRIMARY KEY)")
                conn.execute("CREATE TABLE chunks(id INTEGER PRIMARY KEY)")
                conn.executemany(
                    "INSERT INTO meta VALUES (?, ?)",
                    (("embed_model", "test/model"), ("dim", "4")),
                )
                conn.commit()
            finally:
                conn.close()

            cfg = Config(db_path=db_path, embed_model="test/model")
            with (
                mock.patch("seekr.cli_search.build_embedder") as build_embedder,
                mock.patch("seekr.cli_search.Store") as store,
                mock.patch("seekr.cli_search.try_daemon_search", return_value=None),
            ):
                for mode in ("vector", "hybrid"):
                    with self.subTest(mode=mode):
                        self.assertEqual(
                            cli_search.get_hits("alpha", mode=mode, cfg=cfg), []
                        )

        build_embedder.assert_not_called()
        store.assert_not_called()

    def test_keyword_with_missing_index_returns_empty_without_building_a_model(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "missing.db"
            cfg = Config(db_path=db_path)
            with mock.patch("seekr.cli_search.build_embedder") as build_embedder:
                hits = cli_search.get_hits("anything", mode="keyword", cfg=cfg)

        self.assertEqual(hits, [])
        build_embedder.assert_not_called()

    def test_model_mismatch_is_reported_without_building_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "index.db"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                conn.executemany(
                    "INSERT INTO meta VALUES (?, ?)",
                    (("embed_model", "old/model"), ("dim", "8")),
                )
                conn.commit()
            finally:
                conn.close()

            with mock.patch(
                "seekr.cli_search.build_embedder"
            ) as build_embedder, self.assertRaises(ModelMismatchError) as raised:
                cli_search.get_hits(
                    "anything", mode="keyword", cfg=Config(db_path=db_path)
                )

        build_embedder.assert_not_called()
        self.assertIn("seekr reindex", str(raised.exception))

    def test_database_with_data_but_no_embedding_metadata_needs_reindex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "index.db"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("CREATE TABLE documents(id INTEGER PRIMARY KEY)")
                conn.execute("CREATE TABLE chunks(id INTEGER PRIMARY KEY)")
                conn.execute("INSERT INTO documents(id) VALUES (1)")
                conn.commit()
            finally:
                conn.close()

            with self.assertRaises(ModelMismatchError) as raised:
                cli_search.get_hits(
                    "anything", mode="keyword", cfg=Config(db_path=db_path)
                )

        self.assertIn("seekr reindex", str(raised.exception))

    def test_future_schema_version_is_rejected_without_building_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "index.db"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                conn.execute("CREATE TABLE documents(id INTEGER PRIMARY KEY)")
                conn.execute("CREATE TABLE chunks(id INTEGER PRIMARY KEY)")
                conn.executemany(
                    "INSERT INTO meta VALUES (?, ?)",
                    (
                        ("embed_model", Config().embed_model),
                        ("dim", "384"),
                        ("schema_version", "99"),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            with (
                mock.patch("seekr.cli_search.build_embedder") as build_embedder,
                self.assertRaises(SchemaVersionError) as raised,
            ):
                cli_search.get_hits(
                    "anything", mode="hybrid", cfg=Config(db_path=db_path)
                )

        build_embedder.assert_not_called()
        self.assertIn("schema version 99", str(raised.exception))

    def test_injected_searcher_is_used_without_owning_or_closing_it(self) -> None:
        expected = [
            Hit(
                chunk_id=1,
                path="/notes/a.md",
                locator="",
                loc_start=0.0,
                snippet="alpha",
                score=1.0,
                kind="text",
            )
        ]
        store = mock.Mock()
        searcher = mock.Mock()
        searcher.store = store
        searcher.get_hits.return_value = expected

        with mock.patch("seekr.cli_search.lifecycle_lock") as lifecycle:
            hits = cli_search.get_hits(
                "alpha",
                k=2,
                mode="keyword",
                per_file=1,
                rerank=False,
                path_prefix="/notes",
                searcher=searcher,
            )

        self.assertEqual(hits, expected)
        searcher.get_hits.assert_called_once_with(
            "alpha", k=2, mode="keyword", per_file=1, rerank=False, path_prefix="/notes"
        )
        store.close.assert_not_called()
        lifecycle.assert_not_called()

    def test_owned_searcher_is_closed_inside_the_shared_lifecycle_lock(self) -> None:
        events: list[tuple[str, object]] = []

        @contextlib.contextmanager
        def recording_lock(path, *, shared: bool = True):
            events.append(("lock", (Path(path), shared)))
            try:
                yield object()
            finally:
                events.append(("unlock", (Path(path), shared)))

        store = mock.Mock()
        store.close.side_effect = lambda: events.append(("close", None))
        searcher = mock.Mock()
        searcher.store = store
        searcher.get_hits.side_effect = lambda *args, **kwargs: (
            events.append(("search", None)) or []
        )

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "index.db"
            cfg = Config(db_path=db_path)
            with (
                mock.patch("seekr.cli_search.lifecycle_lock", recording_lock),
                mock.patch(
                    "seekr.cli_search._build_searcher", return_value=searcher
                ) as build_searcher,
            ):
                hits = cli_search.get_hits("alpha", mode="keyword", cfg=cfg)

        self.assertEqual(hits, [])
        build_searcher.assert_called_once_with(cfg, "keyword", False)
        store.close.assert_called_once_with()
        self.assertEqual(
            events,
            [
                ("lock", (db_path, True)),
                ("search", None),
                ("close", None),
                ("unlock", (db_path, True)),
            ],
        )

    def test_invalid_mode_and_limits_are_rejected_early(self) -> None:
        with self.assertRaises(cli_search.SearchError):
            cli_search.get_hits("alpha", mode="bogus")
        with self.assertRaises(cli_search.SearchError):
            cli_search.get_hits("alpha", k=0, searcher=mock.Mock())
        with self.assertRaises(cli_search.SearchError):
            cli_search.get_hits("alpha", per_file=0, searcher=mock.Mock())


@unittest.skipUnless(_HAS_TYPER, "typer is not installed")
class CliCommandTests(unittest.TestCase):
    def _app(self) -> Any:
        # Use the integrated root app: a single-command Typer app is flattened
        # by Click, so invoking it as ``search QUERY`` would not match the
        # installed command name.
        from seekr.cli import app

        return app

    def test_register_adds_a_search_command(self) -> None:
        app = self._app()

        self.assertIn(
            "search", [command.name for command in app.registered_commands]
        )

    def test_search_whitespace_query_prints_no_results_without_backends(self) -> None:
        from typer.testing import CliRunner

        app = self._app()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(db_path=Path(tmp) / "missing.db")
            with (
                mock.patch("seekr.cli_search.load_config", return_value=cfg),
                mock.patch("seekr.cli_search.build_embedder") as build_embedder,
                mock.patch("seekr.cli_search.Store") as store,
                mock.patch("seekr.cli_search.try_daemon_search") as try_daemon,
            ):
                result = CliRunner().invoke(app, ["search", "   "])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("No results.", result.output)
        build_embedder.assert_not_called()
        store.assert_not_called()
        try_daemon.assert_not_called()

    def test_search_empty_index_hybrid_prints_no_results_without_backends(self) -> None:
        from typer.testing import CliRunner

        app = self._app()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(db_path=Path(tmp) / "missing.db")
            with (
                mock.patch("seekr.cli_search.load_config", return_value=cfg),
                mock.patch("seekr.cli_search.build_embedder") as build_embedder,
                mock.patch("seekr.cli_search.Store") as store,
                mock.patch("seekr.cli_search.try_daemon_search", return_value=None),
            ):
                result = CliRunner().invoke(
                    app, ["search", "alpha", "--mode", "hybrid"]
                )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("No results.", result.output)
        build_embedder.assert_not_called()
        store.assert_not_called()

    def test_search_forwards_options_and_prints_the_documented_blocks(self) -> None:
        from typer.testing import CliRunner

        hit = Hit(
            chunk_id=1,
            path="/notes/a.md",
            locator="L1-2",
            loc_start=1.0,
            snippet="E1234 is here",
            score=1.25,
            kind="text",
        )
        app = self._app()
        with mock.patch(
            "seekr.cli_search.get_hits", return_value=[hit]
        ) as get_hits:
            result = CliRunner().invoke(
                app,
                [
                    "search",
                    "E1234",
                    "-k",
                    "3",
                    "--mode",
                    "keyword",
                    "--per-file",
                    "1",
                    "--path",
                    "/notes",
                    "--rerank",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        get_hits.assert_called_once_with(
            "E1234",
            k=3,
            mode="keyword",
            per_file=1,
            rerank=True,
            path_prefix="/notes",
        )
        self.assertIn("/notes/a.md", result.output)
        self.assertIn("(L1-2)", result.output)
        self.assertIn("1.2500", result.output)
        self.assertIn("E1234 is here", result.output)

    def test_search_prints_no_results_for_an_empty_answer(self) -> None:
        from typer.testing import CliRunner

        app = self._app()
        with mock.patch("seekr.cli_search.get_hits", return_value=[]):
            result = CliRunner().invoke(app, ["search", "nothing"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("No results.", result.output)

    def test_search_reports_clean_errors(self) -> None:
        from typer.testing import CliRunner

        app = self._app()
        with mock.patch(
            "seekr.cli_search.get_hits",
            side_effect=cli_search.SearchError("index is unavailable"),
        ):
            result = CliRunner().invoke(app, ["search", "alpha"])

        self.assertEqual(result.exit_code, 1)
        output = result.output + (getattr(result, "stderr", "") or "")
        self.assertIn("index is unavailable", output)


@unittest.skipUnless(_HAS_FTS5, "SQLite was not built with FTS5")
class StdlibStoreSearchTests(unittest.TestCase):
    """Real SQLite FTS retrieval, with only the vec table stubbed for keyword."""

    def setUp(self) -> None:
        use_posix_stored_paths(self)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "index.db"
        self._fake_backend = _fake_vector_backend()
        self._fake_backend.__enter__()
        self.addCleanup(self._fake_backend.__exit__, None, None, None)
        self.store = Store(self.db_path, "test/model", 8)
        self.addCleanup(self.store.close)

    def _add(self, path: str, texts: list[str]) -> None:
        doc_id = self.store.upsert_document(path, 1, 1, "a" * 64, "text")
        self.store.replace_chunks(
            doc_id,
            [
                StoredChunk(
                    spec=ChunkSpec(text=text, locator="", loc_start=0.0),
                    text_sha=text_sha(text),
                    vector=_BytesVector(),
                )
                for text in texts
            ],
        )

    def test_keyword_mode_uses_real_fts(self) -> None:
        self._add("/notes/codes.md", ["alpha common", "E1234 exact"])
        self._add("/notes/unrelated.md", ["other words"])

        hits = Searcher(self.store, _ExplodingEmbedder()).search(
            "E1234", mode="keyword", k=10
        )

        self.assertEqual([hit.path for hit in hits], ["/notes/codes.md"])
        self.assertIn("E1234", hits[0].snippet)

    def test_per_file_cap_real_fts(self) -> None:
        self._add("/notes/a.md", [f"common word {i}" for i in range(5)])
        self._add("/notes/b.md", ["common word other"])

        hits = Searcher(self.store, _ExplodingEmbedder()).search(
            "common", mode="keyword", k=10, per_file=2
        )

        paths = [hit.path for hit in hits]
        self.assertLessEqual(paths.count("/notes/a.md"), 2)
        self.assertLessEqual(paths.count("/notes/b.md"), 2)

    def test_path_prefix_filters_before_the_retriever_limit(self) -> None:
        for i in range(60):
            self._add(f"/other/f{i:02d}.md", ["needle"])
        self._add("/notes/target.md", ["needle " + "filler " * 200])

        hits = Searcher(self.store, _ExplodingEmbedder()).search(
            "needle", mode="keyword", k=1, path_prefix="/notes"
        )

        self.assertEqual([hit.path for hit in hits], ["/notes/target.md"])

    def test_path_prefix_is_component_aware_real_fts(self) -> None:
        self._add("/notes/a.md", ["needle notes"])
        self._add("/notes2/b.md", ["needle notes2"])

        hits = Searcher(self.store, _ExplodingEmbedder()).search(
            "needle", mode="keyword", k=10, path_prefix="/notes"
        )

        self.assertEqual([hit.path for hit in hits], ["/notes/a.md"])

    def test_standalone_keyword_search_opens_store_without_loading_a_model(self) -> None:
        self._add("/notes/codes.md", ["E1234 exact"])
        cfg = Config(db_path=self.db_path, embed_model="test/model")

        with mock.patch(
            "seekr.cli_search.build_embedder", return_value=_ExplodingEmbedder()
        ) as build_embedder:
            hits = cli_search.get_hits("E1234", mode="keyword", cfg=cfg)

        build_embedder.assert_called_once_with(cfg)
        self.assertEqual([hit.path for hit in hits], ["/notes/codes.md"])
        self.assertIn("E1234", hits[0].snippet)


@_NEEDS_REAL_STORE
class RealStoreSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        use_posix_stored_paths(self)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "index.db"
        self.stores: list[Store] = []

    def tearDown(self) -> None:
        for store in reversed(self.stores):
            store.close()

    def open_store(self, dim: int = 2) -> Store:
        store = Store(self.db_path, "test/model", dim)
        self.stores.append(store)
        return store

    @staticmethod
    def _vec(values: list[float]) -> Any:
        import numpy as np

        vector = np.asarray(values, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm

    def _add(self, store: Store, path: str, chunks: list[tuple[str, Any]]) -> None:
        doc_id = store.upsert_document(path, 1, 1, "a" * 64, "text")
        stored = [
            StoredChunk(
                spec=ChunkSpec(text=text, locator="", loc_start=0.0),
                text_sha=text_sha(text),
                vector=vector,
            )
            for text, vector in chunks
        ]
        store.replace_chunks(doc_id, stored)

    def test_keyword_mode_finds_exact_token(self) -> None:
        store = self.open_store()
        self._add(
            store,
            "/notes/codes.md",
            [("alpha common", self._vec([1.0, 0.0])), ("E1234 exact", self._vec([0.0, 1.0]))],
        )
        self._add(store, "/notes/unrelated.md", [("other words", self._vec([1.0, 1.0]))])

        hits = Searcher(store, _ExplodingEmbedder()).search(
            "E1234", mode="keyword", k=10
        )

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].path, "/notes/codes.md")
        self.assertIn("E1234", hits[0].snippet)

    def test_empty_index_returns_empty_without_embedding(self) -> None:
        store = self.open_store()
        searcher = Searcher(store, _ExplodingEmbedder())

        for mode in ("keyword", "vector", "hybrid"):
            with self.subTest(mode=mode):
                self.assertEqual(searcher.search("alpha", mode=mode), [])

    def test_per_file_cap_real_store(self) -> None:
        store = self.open_store()
        self._add(
            store,
            "/notes/a.md",
            [(f"common word {i}", self._vec([1.0, 0.0])) for i in range(5)],
        )
        self._add(store, "/notes/b.md", [("common word other", self._vec([0.0, 1.0]))])

        hits = Searcher(store, _ExplodingEmbedder()).search(
            "common", mode="keyword", k=10, per_file=2
        )

        paths = [hit.path for hit in hits]
        self.assertLessEqual(paths.count("/notes/a.md"), 2)
        self.assertLessEqual(paths.count("/notes/b.md"), 2)

    def test_path_prefix_filters_component_aware_real_store(self) -> None:
        store = self.open_store()
        self._add(store, "/notes/a.md", [("needle notes", self._vec([1.0, 0.0]))])
        self._add(store, "/notes2/b.md", [("needle notes2", self._vec([0.0, 1.0]))])

        hits = Searcher(store, _ExplodingEmbedder()).search(
            "needle", mode="keyword", k=10, path_prefix="/notes"
        )

        self.assertEqual([hit.path for hit in hits], ["/notes/a.md"])

    def test_keyword_path_filter_is_applied_before_the_limit(self) -> None:
        store = self.open_store()
        for i in range(60):
            self._add(
                store,
                f"/other/f{i:02d}.md",
                [("needle", self._vec([1.0, 0.0]))],
            )
        self._add(
            store,
            "/notes/target.md",
            [("needle " + "filler " * 200, self._vec([0.0, 1.0]))],
        )
        calls: list[tuple[int, str | None]] = []
        original = store.keyword_search

        def recording(
            match_query: str,
            limit: int,
            *,
            path_prefix: str | None = None,
        ) -> list[tuple[int, float]]:
            calls.append((int(limit), path_prefix))
            return original(match_query, limit, path_prefix=path_prefix)

        store.keyword_search = recording  # type: ignore[method-assign]

        hits = Searcher(store, _ExplodingEmbedder()).search(
            "needle", mode="keyword", k=1, path_prefix="/notes"
        )

        self.assertEqual([hit.path for hit in hits], ["/notes/target.md"])
        self.assertEqual(calls, [(50, "/notes")])

    def test_vector_path_filter_is_applied_before_the_limit(self) -> None:
        store = self.open_store()
        query_vector = self._vec([1.0, 0.0])
        for i in range(60):
            self._add(
                store,
                f"/other/f{i:02d}.md",
                [(f"outside {i}", query_vector)],
            )
        self._add(store, "/notes/target.md", [("inside", self._vec([0.0, 1.0]))])
        calls: list[tuple[int, str | None]] = []
        original = store.vector_search

        def recording(
            vec: Any,
            limit: int,
            *,
            path_prefix: str | None = None,
        ) -> list[tuple[int, float]]:
            calls.append((int(limit), path_prefix))
            return original(vec, limit, path_prefix=path_prefix)

        store.vector_search = recording  # type: ignore[method-assign]

        hits = Searcher(store, _FixedEmbedder(query_vector)).search(
            "anything", mode="vector", k=1, path_prefix="/notes"
        )

        self.assertEqual([hit.path for hit in hits], ["/notes/target.md"])
        self.assertEqual(calls, [(50, "/notes")])


class _FixedEmbedder:
    model_name = "fixed"

    def __init__(self, vector: Any) -> None:
        self.vector = vector
        self.dim = int(vector.shape[0])

    def embed_documents(self, texts: list[str]) -> Any:
        raise AssertionError("search must not embed documents")

    def embed_query(self, text: str) -> Any:
        return self.vector


class SearchHookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config(db_path=Path("/tmp/seekr-search-hook/index.db"), embed_model="fake")

    def test_use_daemon_false_skips_daemon_and_forwards_embedder(self) -> None:
        hits = [Hit(1, "/notes/a.md", "L1", 1.0, "alpha", 1.0, "text")]
        searcher = mock.Mock()
        searcher.get_hits.return_value = hits
        embedder = FakeEmbedder(dim=8)

        with (
            mock.patch(
                "seekr.cli_search.try_daemon_search",
                side_effect=AssertionError("daemon must be skipped"),
            ) as try_daemon,
            mock.patch(
                "seekr.cli_search._build_searcher", return_value=searcher
            ) as build_searcher,
            mock.patch("seekr.cli_search.lifecycle_lock"),
        ):
            result = cli_search.get_hits(
                "alpha",
                k=3,
                mode="hybrid",
                cfg=self.cfg,
                embedder=embedder,
                use_daemon=False,
            )

        self.assertEqual(result, hits)
        try_daemon.assert_not_called()
        build_searcher.assert_called_once_with(
            self.cfg, "hybrid", False, embedder=embedder
        )
        searcher.get_hits.assert_called_once()
        searcher.store.close.assert_called_once_with()

    def test_existing_build_searcher_call_shape_is_unchanged(self) -> None:
        searcher = mock.Mock()
        searcher.get_hits.return_value = []
        with (
            mock.patch(
                "seekr.cli_search.try_daemon_search", return_value=None
            ),
            mock.patch(
                "seekr.cli_search._build_searcher", return_value=searcher
            ) as build_searcher,
            mock.patch("seekr.cli_search.lifecycle_lock"),
        ):
            cli_search.get_hits("alpha", mode="keyword", cfg=self.cfg)

        build_searcher.assert_called_once_with(self.cfg, "keyword", False)

    def test_searcher_wins_over_a_mismatched_injected_embedder(self) -> None:
        expected = [Hit(1, "/notes/a.md", "L1", 1.0, "alpha", 1.0, "text")]
        searcher = mock.Mock()
        searcher.get_hits.return_value = expected
        embedder = FakeEmbedder(dim=8)
        embedder.model_name = "different/model"

        with (
            mock.patch("seekr.cli_search._daemon_search") as daemon,
            mock.patch("seekr.cli_search._build_searcher") as build_searcher,
        ):
            hits = cli_search.get_hits(
                "alpha", cfg=self.cfg, searcher=searcher, embedder=embedder
            )

        self.assertEqual(hits, expected)
        daemon.assert_not_called()
        build_searcher.assert_not_called()

    @requires_legacy_daemon
    def test_try_daemon_hits_returns_list_none_and_propagates_errors(self) -> None:
        expected = [Hit(1, "/notes/a.md", "L1", 1.0, "alpha", 1.0, "text")]
        with mock.patch(
            "seekr.cli_search.try_daemon_search", return_value=expected
        ) as try_daemon:
            self.assertEqual(
                cli_search.try_daemon_hits("alpha", cfg=self.cfg), expected
            )
        self.assertEqual(try_daemon.call_args.args[0]["query"], "alpha")

        with mock.patch(
            "seekr.cli_search.try_daemon_search",
            side_effect=cli_search.DaemonMismatchError("config mismatch"),
        ):
            self.assertIsNone(cli_search.try_daemon_hits("alpha", cfg=self.cfg))

        with (
            mock.patch(
                "seekr.cli_search.try_daemon_search",
                side_effect=cli_search.DaemonError("daemon timed out"),
            ),
            self.assertRaises(cli_search.DaemonError),
        ):
            cli_search.try_daemon_hits("alpha", cfg=self.cfg)

    def test_try_daemon_hits_empty_query_short_circuits(self) -> None:
        with mock.patch("seekr.cli_search.try_daemon_search") as try_daemon:
            self.assertEqual(cli_search.try_daemon_hits("   ", cfg=self.cfg), [])
        try_daemon.assert_not_called()

        with self.assertRaises(cli_search.SearchError):
            cli_search.try_daemon_hits("alpha", mode="nonsense", cfg=self.cfg)  # type: ignore[arg-type]

    def test_try_daemon_hits_shares_normalised_filters(self) -> None:
        with mock.patch("seekr.cli_search._daemon_search", return_value=[]) as daemon:
            cli_search.try_daemon_hits(
                "alpha", kind="CODE", since="1d", explain=True, cfg=self.cfg
            )

        kwargs = daemon.call_args.kwargs
        self.assertEqual(kwargs["kind"], "code")
        self.assertIsInstance(kwargs["since_mtime_ns"], int)
        self.assertTrue(kwargs["explain"])

    def test_injected_embedder_model_mismatch_is_a_search_error(self) -> None:
        embedder = FakeEmbedder(dim=8)
        embedder.model_name = "other/model"
        with (
            mock.patch(
                "seekr.cli_search._validated_store_info",
                return_value=(Path("/tmp/seekr-search-hook/index.db"), 8, "float32"),
            ),
            mock.patch("seekr.cli_search.Store") as store,
            self.assertRaises(cli_search.SearchError),
        ):
            cli_search._build_searcher(self.cfg, "keyword", False, embedder=embedder)
        store.assert_not_called()

    def test_keyword_build_searcher_does_not_touch_dim(self) -> None:
        class NoDimEmbedder:
            model_name = "fake"

            @property
            def dim(self) -> int:
                raise AssertionError("keyword mode must not read dim")

            def embed_documents(self, texts: list[str]) -> Any:
                raise AssertionError("keyword mode must not embed documents")

            def embed_query(self, text: str) -> Any:
                raise AssertionError("keyword mode must not embed the query")

        fake_store = object()
        with (
            mock.patch(
                "seekr.cli_search._validated_store_info",
                return_value=(Path("/tmp/seekr-search-hook/index.db"), 8, "float32"),
            ),
            mock.patch("seekr.cli_search.Store", return_value=fake_store),
            mock.patch("seekr.cli_search.Searcher") as searcher_cls,
        ):
            cli_search._build_searcher(
                self.cfg, "keyword", False, embedder=NoDimEmbedder()  # type: ignore[arg-type]
            )

        searcher_cls.assert_called_once_with(fake_store, mock.ANY, None)


@_slow
@unittest.skipUnless(
    _HAS_NUMPY and _HAS_SQLITE_VEC and _HAS_FTS5 and _HAS_SENTENCE_TRANSFORMERS,
    "numpy, sqlite-vec, FTS5 and sentence-transformers are required",
)
@unittest.skipUnless(
    _OPTED_IN,
    f"set {_MODEL_DIR_ENV} to a pre-provisioned local model directory",
)
class RealModelSearchTests(unittest.TestCase):
    def test_semantic_match_with_bge(self) -> None:
        from seekr.embed import SentenceTransformerEmbedder

        assert _MODEL_DIR is not None
        model_dir = Path(_MODEL_DIR).expanduser()
        self.assertTrue(model_dir.is_dir(), f"model directory not found: {model_dir}")

        embedder = SentenceTransformerEmbedder(str(model_dir))
        documents = [
            "Backpropagation computes gradients of the loss with respect to each weight.",
            "The mitochondria is the powerhouse of the cell.",
            "Rust ownership prevents data races without a garbage collector.",
        ]
        vectors = embedder.embed_documents(documents)

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "index.db", str(model_dir), embedder.dim)
            self.addCleanup(store.close)
            for index, (text, vector) in enumerate(zip(documents, vectors)):
                doc_id = store.upsert_document(
                    f"/corpus/doc{index}.md", 1, 1, "a" * 64, "text"
                )
                store.replace_chunks(
                    doc_id,
                    [
                        StoredChunk(
                            spec=ChunkSpec(text=text, locator="", loc_start=0.0),
                            text_sha=text_sha(text),
                            vector=vector,
                        )
                    ],
                )

            hits = Searcher(store, embedder).search(
                "how do neural networks learn from their mistakes",
                k=3,
                mode="vector",
            )

        self.assertTrue(
            any("Backpropagation" in hit.snippet for hit in hits),
            [hit.snippet for hit in hits],
        )


if __name__ == "__main__":
    unittest.main()
