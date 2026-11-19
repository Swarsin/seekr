"""Tests for the SQLite store.

Most tests run against a lightweight stand-in for the ``chunks_vec`` table so
that the FTS5, transaction and metadata behaviour can be exercised on hosts
where the optional sqlite-vec extension is not installed.  The tests in
``RealSqliteVecTests`` always use the production sqlite-vec backend and are
skipped only when that package is unavailable.
"""

from __future__ import annotations

import importlib.util
import itertools
import sqlite3
import sys
import tempfile
import threading
import types
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest import mock

from seekr.config import Config
from seekr.models import ChunkSpec, text_sha
from seekr.store import (
    ChunkRow,
    DocRow,
    ModelMismatchError,
    Store,
    StoredChunk,
    _load_vector_extension,
)


def _has_fts5() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        options = [row[0] for row in conn.execute("PRAGMA compile_options")]
    finally:
        conn.close()
    return "ENABLE_FTS5" in options


_HAS_FTS5 = _has_fts5()
_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_HAS_SQLITE_VEC = importlib.util.find_spec("sqlite_vec") is not None


def _unit_vector(seed: int, dim: int = 8) -> object:
    import numpy as np

    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def _stored_chunk(
    text: str,
    seed: int,
    *,
    dim: int = 8,
    locator: str = "",
    loc_start: float = 0.0,
) -> StoredChunk:
    return StoredChunk(
        spec=ChunkSpec(text=text, locator=locator, loc_start=loc_start),
        text_sha=text_sha(text),
        vector=_unit_vector(seed, dim),
    )


def _stored_chunk_with_vector(
    text: str,
    vector: object,
    *,
    locator: str = "",
    loc_start: float = 0.0,
) -> StoredChunk:
    return StoredChunk(
        spec=ChunkSpec(text=text, locator=locator, loc_start=loc_start),
        text_sha=text_sha(text),
        vector=vector,
    )


def _add_document(
    store: Store,
    path: str,
    *chunks: StoredChunk,
    kind: str = "markdown",
    mtime_ns: int = 1,
    size: int = 10,
    sha256: str = "a" * 64,
) -> int:
    doc_id = store.upsert_document(path, mtime_ns, size, sha256, kind)
    if chunks:
        store.replace_chunks(doc_id, list(chunks))
    return doc_id


def _raw_vector_ids(db_path: Path) -> set[int]:
    conn = sqlite3.connect(db_path)
    try:
        return {int(row[0]) for row in conn.execute("SELECT rowid FROM chunks_vec")}
    finally:
        conn.close()


@contextmanager
def _fake_vector_backend():
    """Provide a plain table where the real vec0 virtual table would live.

    This lets the stdlib SQLite part of the store (FTS5 triggers, transactions,
    metadata) be tested without the optional extension.  It deliberately does
    not implement MATCH/KNN queries, so vector-search tests still require the
    real backend.
    """

    def create_table(conn: sqlite3.Connection, dim: int) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks_vec ("
            "rowid INTEGER PRIMARY KEY, embedding BLOB NOT NULL)"
        )

    def serialize(vector: object) -> bytes:
        if isinstance(vector, (bytes, bytearray)):
            return bytes(vector)
        return vector.tobytes()

    with (
        mock.patch("seekr.store._load_vector_extension", lambda conn: None),
        mock.patch("seekr.store._create_vector_table", create_table),
        mock.patch("seekr.store._serialize_vector", serialize),
    ):
        yield


class _FakeBackendTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "index.db"
        self._fake_backend = _fake_vector_backend()
        self._fake_backend.__enter__()
        self.addCleanup(self._fake_backend.__exit__, None, None, None)
        self._stores: list[Store] = []

    def tearDown(self) -> None:
        for store in reversed(self._stores):
            store.close()

    def open_store(self, model: str = "test/model", dim: int = 8) -> Store:
        store = Store(self.db_path, model, dim)
        self._stores.append(store)
        return store


@unittest.skipUnless(_HAS_NUMPY, "numpy is required for the vector store tests")
class StoreBasicsTests(_FakeBackendTestCase):
    def test_roundtrip_and_stats(self) -> None:
        store = self.open_store()
        doc_id = store.upsert_document("/notes/a.md", 123, 456, "f" * 64, "markdown")
        store.replace_chunks(
            doc_id,
            [
                _stored_chunk("alpha beta", 1, locator="L1-2", loc_start=1.0),
                _stored_chunk("gamma delta", 2, locator="L3-4", loc_start=3.0),
            ],
        )

        self.assertEqual(store.stats(), {"documents": 1, "chunks": 2})
        self.assertEqual(
            store.get_document("/notes/a.md"),
            DocRow(
                id=doc_id,
                path="/notes/a.md",
                mtime_ns=123,
                size=456,
                sha256="f" * 64,
                kind="markdown",
            ),
        )

        hits = store.keyword_search('"alpha"', 10)
        self.assertEqual(len(hits), 1)
        chunk_id, score = hits[0]
        self.assertIsInstance(score, float)
        chunk = store.fetch_chunks([chunk_id])[chunk_id]
        self.assertEqual(
            chunk,
            ChunkRow(
                id=chunk_id,
                doc_id=doc_id,
                path="/notes/a.md",
                kind="markdown",
                text="alpha beta",
                locator="L1-2",
                loc_start=1.0,
            ),
        )
        self.assertIsInstance(chunk, ChunkRow)

    def test_keyword_search_finds_token_and_ranks_best_first(self) -> None:
        store = self.open_store()
        doc_id = _add_document(
            store,
            "/notes/codes.md",
            _stored_chunk("unrelated words", 1),
            _stored_chunk("E1234 appears here", 2),
            _stored_chunk("E1234 appears again and again", 3),
        )

        hits = store.keyword_search("E1234", 10)

        self.assertEqual(len(hits), 2)
        self.assertEqual([score for _, score in hits], sorted(score for _, score in hits))
        self.assertTrue(all(0 <= chunk_id for chunk_id, _ in hits))
        self.assertEqual(len(store.fetch_chunks([chunk_id for chunk_id, _ in hits])), 2)
        self.assertEqual(store.get_document("/notes/codes.md").id, doc_id)
        self.assertEqual(store.keyword_search("", 10), [])
        self.assertEqual(store.keyword_search("   ", 10), [])

    def test_keyword_search_malformed_syntax_never_raises(self) -> None:
        store = self.open_store()
        _add_document(store, "/notes/a.md", _stored_chunk("E1234 text", 1))

        for query in ('say "hello', "a AND -b", "col:val (x*)", "it's"):
            with self.subTest(query=query):
                store.keyword_search(query, 10)

    def test_fetch_chunks_is_deterministic_and_ignores_unknown_ids(self) -> None:
        store = self.open_store()
        doc_id = _add_document(
            store,
            "/notes/a.md",
            _stored_chunk("first token", 1),
            _stored_chunk("second token", 2),
        )
        first_id = store.keyword_search('"first"', 10)[0][0]
        second_id = store.keyword_search('"second"', 10)[0][0]

        chunks = store.fetch_chunks([second_id, 999_999, first_id, second_id])

        self.assertEqual(list(chunks), [first_id, second_id])
        self.assertTrue(all(chunk.doc_id == doc_id for chunk in chunks.values()))
        self.assertEqual(store.fetch_chunks([]), {})

    def test_replace_chunks_removes_old_fts_and_vectors(self) -> None:
        store = self.open_store()
        doc_id = _add_document(
            store,
            "/notes/a.md",
            _stored_chunk("obsoleteword only", 1),
        )
        old_sha = text_sha("obsoleteword only")
        old_id = store.keyword_search("obsoleteword", 10)[0][0]
        self.assertIsNotNone(store.find_vector_by_text_sha(old_sha))

        store.replace_chunks(doc_id, [_stored_chunk("freshword only", 2)])

        self.assertEqual(store.stats(), {"documents": 1, "chunks": 1})
        self.assertEqual(store.keyword_search("obsoleteword", 10), [])
        self.assertEqual(len(store.keyword_search("freshword", 10)), 1)
        new_id = store.keyword_search("freshword", 10)[0][0]
        self.assertEqual(_raw_vector_ids(self.db_path), {new_id})
        self.assertNotEqual(new_id, old_id)
        self.assertIsNone(store.find_vector_by_text_sha(old_sha))
        self.assertIsNotNone(store.find_vector_by_text_sha(text_sha("freshword only")))

    def test_replace_chunks_is_atomic_on_serialization_error(self) -> None:
        store = self.open_store()
        doc_id = _add_document(
            store,
            "/notes/a.md",
            _stored_chunk("keepword", 1),
        )
        old_sha = text_sha("keepword")

        with (
            mock.patch(
                "seekr.store._serialize_vector",
                side_effect=RuntimeError("vector serialization failed"),
            ),
            self.assertRaises(RuntimeError),
        ):
            store.replace_chunks(doc_id, [_stored_chunk("changedword", 2)])

        self.assertEqual(store.stats(), {"documents": 1, "chunks": 1})
        self.assertEqual(len(store.keyword_search("keepword", 10)), 1)
        self.assertEqual(store.keyword_search("changedword", 10), [])
        self.assertIsNotNone(store.find_vector_by_text_sha(old_sha))
        self.assertIsNone(store.find_vector_by_text_sha(text_sha("changedword")))

    def test_delete_document_cascades_chunks_fts_and_vectors(self) -> None:
        store = self.open_store()
        _add_document(
            store,
            "/notes/a.md",
            _stored_chunk("deletemeone", 1),
            _stored_chunk("deletemetwo", 2),
        )
        chunk_ids = {
            chunk_id for chunk_id, _ in store.keyword_search("deletemeone OR deletemetwo", 10)
        }
        self.assertEqual(_raw_vector_ids(self.db_path), chunk_ids)

        store.delete_document("/notes/a.md")

        self.assertEqual(_raw_vector_ids(self.db_path), set())
        self.assertEqual(store.stats(), {"documents": 0, "chunks": 0})
        self.assertIsNone(store.get_document("/notes/a.md"))
        self.assertEqual(store.keyword_search("deletemeone", 10), [])
        self.assertEqual(store.keyword_search("deletemetwo", 10), [])
        self.assertIsNone(store.find_vector_by_text_sha(text_sha("deletemeone")))
        self.assertIsNone(store.find_vector_by_text_sha(text_sha("deletemetwo")))
        store.delete_document("/notes/a.md")

    def test_find_vector_by_text_sha_returns_float32_copy(self) -> None:
        import numpy as np

        store = self.open_store()
        first = _stored_chunk("same text", 5)
        second = _stored_chunk("same text", 6)
        _add_document(store, "/notes/a.md", first, second)

        found = store.find_vector_by_text_sha(text_sha("same text"))

        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found.shape, (8,))
        self.assertEqual(found.dtype, np.float32)
        self.assertTrue(np.allclose(found, first.vector))
        found[0] = 99.0
        again = store.find_vector_by_text_sha(text_sha("same text"))
        assert again is not None
        self.assertFalse(np.allclose(again, found))
        self.assertIsNone(store.find_vector_by_text_sha("not-a-real-sha"))

    def test_upsert_document_is_idempotent_on_path(self) -> None:
        store = self.open_store()
        first_id = store.upsert_document("/notes/a.md", 1, 10, "a" * 64, "markdown")
        second_id = store.upsert_document("/notes/a.md", 2, 20, "b" * 64, "text")

        self.assertEqual(first_id, second_id)
        self.assertEqual(store.stats(), {"documents": 1, "chunks": 0})
        self.assertEqual(
            store.get_document("/notes/a.md"),
            DocRow(
                id=first_id,
                path="/notes/a.md",
                mtime_ns=2,
                size=20,
                sha256="b" * 64,
                kind="text",
            ),
        )

    def test_stats_has_exactly_the_documented_keys(self) -> None:
        store = self.open_store()

        self.assertEqual(store.stats(), {"documents": 0, "chunks": 0})
        self.assertEqual(set(store.stats()), {"documents", "chunks"})


@unittest.skipUnless(_HAS_NUMPY, "numpy is required for the vector store tests")
class StoreMetadataTests(_FakeBackendTestCase):
    def test_model_mismatch_raises_and_preserves_data(self) -> None:
        store = self.open_store()
        _add_document(store, "/notes/a.md", _stored_chunk("keepword", 1))
        store.close()

        for model, dim in (("other/model", 8), ("test/model", 16)):
            with self.subTest(model=model, dim=dim):
                with self.assertRaises(ModelMismatchError) as raised:
                    Store(self.db_path, model, dim)
                self.assertIn("seekr reindex", str(raised.exception))

        reopened = self.open_store()
        self.assertEqual(reopened.stats(), {"documents": 1, "chunks": 1})
        self.assertEqual(len(reopened.keyword_search("keepword", 10)), 1)

    def test_changed_config_embedding_model_is_rejected(self) -> None:
        cfg = Config()
        store = self.open_store(model=cfg.embed_model)
        store.close()
        changed = replace(cfg, embed_model="BAAI/another-model")

        with self.assertRaises(ModelMismatchError) as raised:
            Store(self.db_path, changed.embed_model, 8)

        self.assertIn("seekr reindex", str(raised.exception))

    def test_database_with_data_but_no_metadata_requires_reindex(self) -> None:
        store = self.open_store()
        _add_document(store, "/notes/a.md")
        store.close()

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM meta")
            conn.commit()
        finally:
            conn.close()

        with self.assertRaises(ModelMismatchError) as raised:
            Store(self.db_path, "test/model", 8)
        self.assertIn("seekr reindex", str(raised.exception))

    def test_last_error_count_persists_and_does_not_change_stats(self) -> None:
        store = self.open_store()
        self.assertEqual(store.get_last_error_count(), 0)

        store.set_last_error_count(3)

        self.assertEqual(store.get_last_error_count(), 3)
        self.assertEqual(store.stats(), {"documents": 0, "chunks": 0})
        store.close()

        reopened = self.open_store()
        self.assertEqual(reopened.get_last_error_count(), 3)
        reopened.set_last_error_count(0)
        self.assertEqual(reopened.get_last_error_count(), 0)


@unittest.skipUnless(_HAS_NUMPY, "numpy is required for the vector store tests")
class StorePathTests(_FakeBackendTestCase):
    def test_list_paths_is_sorted_and_component_aware(self) -> None:
        store = self.open_store()
        paths = [
            "/notes/sub/b.md",
            "/notes/a.md",
            "/notes2/c.md",
            "/other.md",
        ]
        for path in paths:
            _add_document(store, path)

        self.assertEqual(
            store.list_paths(),
            sorted(paths),
        )
        self.assertEqual(
            store.list_paths("/notes"),
            ["/notes/a.md", "/notes/sub/b.md"],
        )
        self.assertEqual(
            store.list_paths("/notes/"),
            ["/notes/a.md", "/notes/sub/b.md"],
        )
        self.assertEqual(store.list_paths("/notes/sub"), ["/notes/sub/b.md"])
        self.assertEqual(store.list_paths("/notes/a.md"), ["/notes/a.md"])
        self.assertEqual(store.list_paths("/notes2"), ["/notes2/c.md"])
        self.assertEqual(store.list_paths("/missing"), [])
        self.assertEqual(store.list_paths(""), sorted(paths))
        self.assertEqual(store.list_paths("/"), sorted(paths))


@unittest.skipUnless(_HAS_NUMPY, "numpy is required for the vector store tests")
class StoreMoveTests(_FakeBackendTestCase):
    def test_move_document_preserves_ids_chunks_vectors_and_fts(self) -> None:
        import numpy as np

        store = self.open_store()
        src = "/notes/old.md"
        dst = "/notes/new.md"
        doc_id = store.upsert_document(src, 10, 20, "a" * 64, "markdown")
        first = _stored_chunk("renameword", 11, locator="L1-2", loc_start=1.0)
        second = _stored_chunk("otherword", 12, locator="L3-4", loc_start=3.0)
        store.replace_chunks(doc_id, [first, second])
        old_ids = {
            chunk_id
            for chunk_id, _ in store.keyword_search('"renameword" OR "otherword"', 10)
        }

        moved_id = store.move_document(src, dst, mtime_ns=99, size=88)

        self.assertEqual(moved_id, doc_id)
        self.assertIsNone(store.get_document(src))
        moved = store.get_document(dst)
        self.assertEqual(
            moved,
            DocRow(
                id=doc_id,
                path=dst,
                mtime_ns=99,
                size=88,
                sha256="a" * 64,
                kind="markdown",
            ),
        )
        self.assertEqual(store.stats(), {"documents": 1, "chunks": 2})
        rename_hits = store.keyword_search('"renameword"', 10)
        self.assertEqual(len(rename_hits), 1)
        self.assertIn(rename_hits[0][0], old_ids)
        for chunk_id, _ in store.keyword_search('"renameword" OR "otherword"', 10):
            self.assertIn(chunk_id, old_ids)
            self.assertEqual(store.fetch_chunks([chunk_id])[chunk_id].path, dst)
        vector = store.find_vector_by_text_sha(text_sha("renameword"))
        assert vector is not None
        self.assertTrue(np.allclose(vector, first.vector))

    def test_move_same_path_updates_metadata(self) -> None:
        store = self.open_store()
        path = "/notes/a.md"
        doc_id = store.upsert_document(path, 1, 2, "a" * 64, "text")

        moved_id = store.move_document(path, path, mtime_ns=5, size=6)

        self.assertEqual(moved_id, doc_id)
        self.assertEqual(
            store.get_document(path),
            DocRow(id=doc_id, path=path, mtime_ns=5, size=6, sha256="a" * 64, kind="text"),
        )

    def test_move_unknown_source_raises_key_error(self) -> None:
        store = self.open_store()

        with self.assertRaises(KeyError):
            store.move_document("/missing.md", "/new.md", mtime_ns=1, size=1)

        self.assertEqual(store.stats(), {"documents": 0, "chunks": 0})

    def test_move_to_existing_path_raises_without_changes(self) -> None:
        store = self.open_store()
        src_id = _add_document(store, "/notes/src.md")
        dst_id = _add_document(store, "/notes/dst.md")

        with self.assertRaises(ValueError):
            store.move_document("/notes/src.md", "/notes/dst.md", mtime_ns=9, size=9)

        self.assertEqual(store.get_document("/notes/src.md").id, src_id)
        self.assertEqual(store.get_document("/notes/dst.md").id, dst_id)
        self.assertEqual(store.stats(), {"documents": 2, "chunks": 0})


@unittest.skipUnless(_HAS_NUMPY, "numpy is required for the vector store tests")
class StoreTransactionTests(_FakeBackendTestCase):
    def test_write_transaction_rolls_back_documents_chunks_and_vectors(self) -> None:
        store = self.open_store()
        doc_id = _add_document(store, "/notes/a.md", _stored_chunk("keepword", 1))
        old_sha = text_sha("keepword")

        with self.assertRaises(RuntimeError), store.transaction(write=True):
            store.replace_chunks(doc_id, [_stored_chunk("changedword", 2)])
            _add_document(store, "/notes/b.md", _stored_chunk("newword", 3))
            raise RuntimeError("rollback")

        self.assertEqual(store.stats(), {"documents": 1, "chunks": 1})
        self.assertIsNone(store.get_document("/notes/b.md"))
        self.assertEqual(len(store.keyword_search("keepword", 10)), 1)
        self.assertEqual(store.keyword_search("changedword", 10), [])
        self.assertEqual(store.keyword_search("newword", 10), [])
        self.assertIsNotNone(store.find_vector_by_text_sha(old_sha))
        self.assertIsNone(store.find_vector_by_text_sha(text_sha("changedword")))

    def test_nested_transaction_inner_rollback_keeps_outer_writes(self) -> None:
        store = self.open_store()

        with store.transaction(write=True):
            _add_document(store, "/notes/outer.md", _stored_chunk("outerword", 1))
            with self.assertRaises(RuntimeError), store.transaction(write=True):
                _add_document(store, "/notes/inner.md", _stored_chunk("innerword", 2))
                raise RuntimeError("inner")
            self.assertIsNone(store.get_document("/notes/inner.md"))
            self.assertEqual(store.keyword_search("innerword", 10), [])
            self.assertIsNotNone(store.get_document("/notes/outer.md"))

        self.assertEqual(store.stats(), {"documents": 1, "chunks": 1})
        self.assertEqual(len(store.keyword_search("outerword", 10)), 1)
        self.assertIsNone(store.get_document("/notes/inner.md"))

    def test_outer_rollback_discards_committed_savepoints(self) -> None:
        store = self.open_store()

        with self.assertRaises(RuntimeError), store.transaction(write=True):
            _add_document(store, "/notes/a.md")
            with store.transaction(write=True):
                _add_document(store, "/notes/b.md")
            raise RuntimeError("outer")

        self.assertEqual(store.stats(), {"documents": 0, "chunks": 0})
        self.assertIsNone(store.get_document("/notes/a.md"))
        self.assertIsNone(store.get_document("/notes/b.md"))

    def test_read_transaction_blocks_writers_until_it_exits(self) -> None:
        store = self.open_store()
        _add_document(store, "/notes/a.md", _stored_chunk("existingword", 1))
        started = threading.Event()
        done = threading.Event()

        def writer() -> None:
            started.set()
            _add_document(store, "/notes/b.md", _stored_chunk("newword", 2))
            done.set()

        thread = threading.Thread(target=writer, daemon=True)
        with store.transaction():
            thread.start()
            self.assertTrue(started.wait(2.0))
            self.assertFalse(done.wait(0.1))
            self.assertIsNone(store.get_document("/notes/b.md"))
            self.assertEqual(store.stats(), {"documents": 1, "chunks": 1})

        thread.join(2.0)
        self.assertFalse(thread.is_alive())
        self.assertTrue(done.is_set())
        self.assertIsNotNone(store.get_document("/notes/b.md"))


@unittest.skipUnless(_HAS_NUMPY, "numpy is required for the vector store tests")
class StoreThreadingTests(_FakeBackendTestCase):
    def test_shared_connection_serves_concurrent_readers(self) -> None:
        store = self.open_store()
        _add_document(store, "/notes/a.md", _stored_chunk("parallelword", 1))
        errors: list[str] = []
        barrier = threading.Barrier(4)

        def reader() -> None:
            try:
                barrier.wait(2.0)
                for _ in range(20):
                    hits = store.keyword_search('"parallelword"', 10)
                    if not hits:
                        errors.append("keyword search returned no hits")
                        return
                    chunk = store.fetch_chunks([hits[0][0]])[hits[0][0]]
                    if chunk.path != "/notes/a.md":
                        errors.append(f"unexpected path: {chunk.path}")
                        return
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(repr(exc))

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5.0)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])


class StdlibStoreTests(_FakeBackendTestCase):
    """SQLite/FTS5/metadata/rollback tests that need neither numpy nor sqlite-vec.

    ``chunks_vec`` is a plain table here, so these tests make no claim about the
    real vec0 backend; they only cover the stdlib-SQLite portions of the store.
    """

    @staticmethod
    def _bytes_chunk(
        text: str,
        *,
        fill: int = 0,
        locator: str = "",
        loc_start: float = 0.0,
    ) -> StoredChunk:
        return StoredChunk(
            spec=ChunkSpec(text=text, locator=locator, loc_start=loc_start),
            text_sha=text_sha(text),
            vector=bytes([fill % 256]) * 32,
        )

    def test_fts_replace_delete_and_explicit_fake_vector_rows(self) -> None:
        store = self.open_store()
        doc_id = store.upsert_document("/notes/a.md", 1, 2, "a" * 64, "markdown")
        store.replace_chunks(doc_id, [self._bytes_chunk("firsttoken", fill=1)])
        first_id = store.keyword_search("firsttoken", 10)[0][0]
        self.assertEqual(_raw_vector_ids(self.db_path), {first_id})

        store.replace_chunks(doc_id, [self._bytes_chunk("secondtoken", fill=2)])
        second_id = store.keyword_search("secondtoken", 10)[0][0]
        self.assertEqual(store.keyword_search("firsttoken", 10), [])
        self.assertEqual(_raw_vector_ids(self.db_path), {second_id})
        self.assertNotEqual(first_id, second_id)

        store.delete_document("/notes/a.md")
        self.assertEqual(store.keyword_search("secondtoken", 10), [])
        self.assertEqual(_raw_vector_ids(self.db_path), set())
        self.assertEqual(store.stats(), {"documents": 0, "chunks": 0})

    def test_fts_update_trigger_stays_in_sync(self) -> None:
        store = self.open_store()
        doc_id = store.upsert_document("/notes/a.md", 1, 2, "a" * 64, "text")
        store.replace_chunks(doc_id, [self._bytes_chunk("beforetoken", fill=1)])
        chunk_id = store.keyword_search("beforetoken", 10)[0][0]

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "UPDATE chunks SET text = ? WHERE id = ?",
                ("aftertoken", chunk_id),
            )
            conn.commit()
        finally:
            conn.close()

        self.assertEqual(store.keyword_search("beforetoken", 10), [])
        self.assertEqual(len(store.keyword_search("aftertoken", 10)), 1)
        self.assertEqual(store.fetch_chunks([chunk_id])[chunk_id].text, "aftertoken")

    def test_keyword_path_prefix_filters_before_limit(self) -> None:
        store = self.open_store()
        _add_document(
            store,
            "/notes/allowed/a.md",
            self._bytes_chunk("sharedtoken alpha", fill=1),
        )
        _add_document(
            store,
            "/notes/blocked/b.md",
            self._bytes_chunk("sharedtoken " * 10 + "beta", fill=2),
        )

        unfiltered = store.keyword_search('"sharedtoken"', 1)
        self.assertEqual(len(unfiltered), 1)
        self.assertEqual(
            store.fetch_chunks([unfiltered[0][0]])[unfiltered[0][0]].path,
            "/notes/blocked/b.md",
        )

        filtered = store.keyword_search(
            '"sharedtoken"', 1, path_prefix="/notes/allowed"
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(
            store.fetch_chunks([filtered[0][0]])[filtered[0][0]].path,
            "/notes/allowed/a.md",
        )
        self.assertEqual(
            store.keyword_search('"sharedtoken"', 10, path_prefix="/notes/none"),
            [],
        )

    def test_metadata_mismatch_and_error_count_without_numpy(self) -> None:
        store = self.open_store()
        store.set_last_error_count(2)
        store.close()

        with self.assertRaises(ModelMismatchError) as raised:
            Store(self.db_path, "other/model", 8)
        self.assertIn("seekr reindex", str(raised.exception))

        reopened = self.open_store()
        self.assertEqual(reopened.get_last_error_count(), 2)
        self.assertEqual(reopened.stats(), {"documents": 0, "chunks": 0})

    def test_list_paths_component_aware_without_numpy(self) -> None:
        store = self.open_store()
        for path in ("/notes/a.md", "/notes/sub/b.md", "/notes2/c.md"):
            store.upsert_document(path, 1, 1, "a" * 64, "text")

        self.assertEqual(
            store.list_paths("/notes"),
            ["/notes/a.md", "/notes/sub/b.md"],
        )
        self.assertEqual(store.list_paths("/notes/a.md"), ["/notes/a.md"])

    def test_write_transaction_rolls_back_fts_meta_and_fake_vectors(self) -> None:
        store = self.open_store()
        doc_id = _add_document(store, "/notes/a.md", self._bytes_chunk("keepword", fill=1))
        old_id = store.keyword_search("keepword", 10)[0][0]

        with self.assertRaises(RuntimeError), store.transaction(write=True):
            store.replace_chunks(doc_id, [self._bytes_chunk("changedword", fill=2)])
            _add_document(store, "/notes/b.md")
            store.set_last_error_count(9)
            raise RuntimeError("rollback")

        self.assertEqual(store.stats(), {"documents": 1, "chunks": 1})
        self.assertEqual(len(store.keyword_search("keepword", 10)), 1)
        self.assertEqual(store.keyword_search("changedword", 10), [])
        self.assertIsNone(store.get_document("/notes/b.md"))
        self.assertEqual(store.get_last_error_count(), 0)
        self.assertEqual(_raw_vector_ids(self.db_path), {old_id})


@unittest.skipUnless(
    _HAS_SQLITE_VEC and _HAS_NUMPY and _HAS_FTS5,
    "the real sqlite-vec backend and numpy are required",
)
class RealSqliteVecTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "index.db"
        self.store = Store(self.db_path, "test/model", 8)
        self.addCleanup(self.store.close)

    def test_vector_search_nearest_first_and_serialised(self) -> None:
        import numpy as np

        first = _stored_chunk("alpha", 1)
        second = _stored_chunk("beta", 2)
        _add_document(self.store, "/notes/a.md", first, second)
        first_id = self.store.keyword_search('"alpha"', 10)[0][0]
        second_id = self.store.keyword_search('"beta"', 10)[0][0]

        hits = self.store.vector_search(first.vector, 2)

        self.assertEqual([chunk_id for chunk_id, _ in hits], [first_id, second_id])
        self.assertAlmostEqual(hits[0][1], 0.0, places=5)
        self.assertEqual([distance for _, distance in hits], sorted(d for _, d in hits))
        found = self.store.find_vector_by_text_sha(text_sha("alpha"))
        assert found is not None
        self.assertEqual(found.shape, (8,))
        self.assertEqual(found.dtype, np.float32)
        self.assertTrue(np.allclose(found, first.vector))

    def test_vector_search_rejects_wrong_dimension(self) -> None:
        import numpy as np

        with self.assertRaises(ValueError):
            self.store.vector_search(np.zeros(3, dtype=np.float32), 5)

    def test_vector_search_path_prefix_filters_before_top_k(self) -> None:
        import numpy as np

        query = np.zeros(8, dtype=np.float32)
        query[0] = 1.0
        blocked_vector = query.copy()
        allowed_vector = np.zeros(8, dtype=np.float32)
        allowed_vector[1] = 1.0
        _add_document(
            self.store,
            "/notes/blocked/b.md",
            _stored_chunk_with_vector("blockedone", blocked_vector),
            _stored_chunk_with_vector("blockedtwo", blocked_vector),
        )
        _add_document(
            self.store,
            "/notes/allowed/a.md",
            _stored_chunk_with_vector("allowedone", allowed_vector),
        )
        allowed_id = self.store.keyword_search('"allowedone"', 10)[0][0]

        global_top = self.store.vector_search(query, 1)
        self.assertEqual(len(global_top), 1)
        self.assertEqual(
            self.store.fetch_chunks([global_top[0][0]])[global_top[0][0]].path,
            "/notes/blocked/b.md",
        )

        filtered = self.store.vector_search(query, 1, path_prefix="/notes/allowed")
        self.assertEqual([chunk_id for chunk_id, _ in filtered], [allowed_id])
        self.assertEqual(
            self.store.vector_search(query, 1, path_prefix="/notes/allowed2"),
            [],
        )

    def test_vector_search_k_above_vec0_cap_uses_scalar_distance(self) -> None:
        import numpy as np

        zero = np.zeros(8, dtype=np.float32)
        zero[0] = 1.0
        one = np.zeros(8, dtype=np.float32)
        one[1] = 1.0
        two = np.zeros(8, dtype=np.float32)
        two[2] = 1.0
        _add_document(
            self.store,
            "/notes/a.md",
            _stored_chunk_with_vector("largekzero", zero),
            _stored_chunk_with_vector("largekone", one),
            _stored_chunk_with_vector("largektwo", two),
        )
        zero_id = self.store.keyword_search('"largekzero"', 10)[0][0]

        hits = self.store.vector_search(zero, 5000)

        self.assertEqual(len(hits), 3)
        self.assertEqual(hits[0][0], zero_id)
        self.assertTrue(all(left[1] <= right[1] for left, right in itertools.pairwise(hits)))

    def test_replace_chunks_removes_old_vector_rows(self) -> None:
        old_chunk = _stored_chunk("oldvectorword", 7)
        replacement = _stored_chunk("newvectorword", 8)
        doc_id = _add_document(self.store, "/notes/a.md", old_chunk)
        old_id = self.store.keyword_search('"oldvectorword"', 10)[0][0]

        self.store.replace_chunks(doc_id, [replacement])

        self.assertEqual(self.store.keyword_search("oldvectorword", 10), [])
        self.assertIsNone(self.store.find_vector_by_text_sha(text_sha("oldvectorword")))
        remaining = self.store.vector_search(old_chunk.vector, 10)
        self.assertNotIn(old_id, [chunk_id for chunk_id, _ in remaining])
        self.assertIsNotNone(self.store.find_vector_by_text_sha(text_sha("newvectorword")))

    def test_delete_document_removes_vector_rows(self) -> None:
        chunk = _stored_chunk("gonevectorword", 9)
        _add_document(self.store, "/notes/a.md", chunk)

        self.store.delete_document("/notes/a.md")

        self.assertEqual(self.store.stats(), {"documents": 0, "chunks": 0})
        self.assertEqual(self.store.vector_search(chunk.vector, 10), [])
        self.assertIsNone(self.store.find_vector_by_text_sha(text_sha("gonevectorword")))


class VectorExtensionLoaderTests(unittest.TestCase):
    def test_loader_enables_and_disables_extension_loading(self) -> None:
        events: list[object] = []

        class FakeConnection:
            def enable_load_extension(self, enabled: bool) -> None:
                events.append(("enable", enabled))

        fake_module = types.SimpleNamespace(load=lambda conn: events.append("load"))
        with mock.patch.dict(sys.modules, {"sqlite_vec": fake_module}):
            _load_vector_extension(FakeConnection())

        self.assertEqual(events, [("enable", True), "load", ("enable", False)])

    def test_loader_disables_extension_loading_when_load_fails(self) -> None:
        events: list[object] = []

        class FakeConnection:
            def enable_load_extension(self, enabled: bool) -> None:
                events.append(("enable", enabled))

        def fail(conn: object) -> None:
            events.append("load")
            raise RuntimeError("extension failed")

        fake_module = types.SimpleNamespace(load=fail)
        with (
            mock.patch.dict(sys.modules, {"sqlite_vec": fake_module}),
            self.assertRaises(RuntimeError),
        ):
            _load_vector_extension(FakeConnection())

        self.assertEqual(events, [("enable", True), "load", ("enable", False)])


class MissingVectorBackendTests(unittest.TestCase):
    @unittest.skipIf(_HAS_SQLITE_VEC, "sqlite-vec is installed")
    def test_construction_raises_instead_of_silently_degrading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ImportError):
            Store(Path(tmp) / "index.db", "test/model", 8)


if __name__ == "__main__":
    unittest.main()
