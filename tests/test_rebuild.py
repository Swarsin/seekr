"""Isolated tests for lifecycle locking, safe reindex rebuilds and the CLI."""

from __future__ import annotations

import importlib.util
import os
import sqlite3
import stat
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from seekr.config import Config
from seekr.indexer import SyncReport
from seekr.lifecycle import (
    LifecycleError,
    LifecycleLock,
    LifecycleLockError,
    RebuildError,
    lifecycle_lock,
    rebuild_index,
)
from seekr.models import text_sha

_HAS_TYPER = importlib.util.find_spec("typer") is not None


def _has_fts5() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        options = {row[0] for row in conn.execute("PRAGMA compile_options")}
    finally:
        conn.close()
    return "ENABLE_FTS5" in options


_HAS_FTS5 = _has_fts5()


def _make_database(path: Path, documents: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                mtime_ns INTEGER NOT NULL DEFAULT 0,
                size INTEGER NOT NULL DEFAULT 0,
                sha256 TEXT NOT NULL DEFAULT '',
                kind TEXT NOT NULL DEFAULT 'text'
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
                text TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        for number in range(documents):
            conn.execute(
                "INSERT INTO documents(path) VALUES (?)", (f"/old/document-{number}.txt",)
            )
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('marker', 'old')")
        conn.commit()
    finally:
        conn.close()


def _document_count(path: Path) -> int | None:
    if not path.exists():
        return None
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return int(row[0]) if row else None
    finally:
        conn.close()


class _FakeEmbedder:
    model_name = "fake"
    dim = 8

    def embed_documents(self, texts):
        return [f"vec:{text}" for text in texts]

    def embed_query(self, text):
        return f"vec:{text}"


class _ByteEmbedder:
    model_name = "fake"
    dim = 8

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[bytes]:
        batch = list(texts)
        self.calls.append(batch)
        return [text_sha(text).encode("ascii")[: self.dim] for text in batch]

    def embed_query(self, text: str) -> bytes:
        return text_sha(text).encode("ascii")[: self.dim]


@contextmanager
def _stdlib_vector_backend():
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


class _FakeStore:
    def __init__(
        self, path: Path, embed_model: str, dim: int, *, documents: int = 1
    ) -> None:
        self.path = Path(path)
        self.embed_model = embed_model
        self.dim = dim
        self.documents = documents
        self.closed = False
        _make_database(self.path, documents)

    def stats(self) -> dict[str, int]:
        return {"documents": self.documents, "chunks": self.documents}

    def close(self) -> None:
        self.closed = True


class _FakeIndexer:
    def __init__(self, report: SyncReport) -> None:
        self.report = report
        self.sync_calls: list[list[Path]] = []

    def sync(self, roots: list[Path]) -> SyncReport:
        self.sync_calls.append(list(roots))
        return self.report


class _FakeStatusStore:
    def __init__(self, stats: dict[str, int], errors: int) -> None:
        self._stats = stats
        self._errors = errors
        self.closed = False

    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def get_last_error_count(self) -> int:
        return self._errors

    def set_last_error_count(self, n: int) -> None:
        self._errors = int(n)

    def close(self) -> None:
        self.closed = True


class LifecycleLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.db = self.tmp / "index.db"

    def test_shared_locks_coexist_and_block_exclusive(self) -> None:
        with lifecycle_lock(self.db, shared=True), lifecycle_lock(self.db, shared=True):
            contender = LifecycleLock(self.db, shared=False, blocking=False)
            with self.assertRaises(LifecycleLockError):
                contender.acquire()

        exclusive = LifecycleLock(self.db, shared=False, blocking=False)
        exclusive.acquire()
        try:
            with self.assertRaises(
                LifecycleLockError
            ), lifecycle_lock(self.db, shared=True, blocking=False):
                pass
        finally:
            exclusive.release()

    def test_lock_file_is_private_and_a_sibling(self) -> None:
        with lifecycle_lock(self.db, shared=True):
            pass

        lock_path = self.db.with_name(self.db.name + ".lock")
        self.assertTrue(lock_path.is_file())
        if os.name == "posix":  # Windows has no POSIX permission bits
            mode = stat.S_IMODE(lock_path.stat().st_mode)
            self.assertEqual(mode & 0o077, 0)


class RebuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "root"
        self.root.mkdir()
        self.db = self.tmp / "index.db"
        self.cfg = Config(roots=(self.root,), db_path=self.db)
        self.embedder = _FakeEmbedder()

    def _rebuild(
        self,
        report: SyncReport,
        *,
        new_documents: int = 1,
        lock_timeout: float = 0.0,
        force: bool = False,
        allow_empty: bool = False,
        indexer_factory=None,
    ) -> SyncReport:
        store_factory = lambda path, model, dim: _FakeStore(
            path, model, dim, documents=new_documents
        )
        factory = indexer_factory or (lambda store, embedder, cfg: _FakeIndexer(report))
        return rebuild_index(
            self.cfg,
            embedder=self.embedder,
            store_factory=store_factory,
            indexer_factory=factory,
            lock_timeout=lock_timeout,
            force=force,
            allow_empty=allow_empty,
        )

    def test_success_swaps_in_the_fresh_database(self) -> None:
        _make_database(self.db, documents=2)
        report = SyncReport(added=1)

        result = self._rebuild(report)

        self.assertIs(result, report)
        self.assertEqual(_document_count(self.db), 1)
        self.assertEqual(list(self.tmp.glob(".*.rebuild-*")), [])

    def test_no_roots_refuses_without_touching_live_database(self) -> None:
        _make_database(self.db, documents=2)
        cfg = Config(roots=(), db_path=self.db)

        with self.assertRaises(RebuildError):
            rebuild_index(cfg, embedder=self.embedder)

        self.assertEqual(_document_count(self.db), 2)
        self.assertEqual(list(self.tmp.glob(".*.rebuild-*")), [])

    def test_per_file_errors_abort_without_touching_live_database(self) -> None:
        _make_database(self.db, documents=2)
        report = SyncReport(added=1, errors=[("bad.txt", "boom")])

        with self.assertRaises(RebuildError):
            self._rebuild(report)

        self.assertEqual(_document_count(self.db), 2)
        self.assertEqual(list(self.tmp.glob(".*.rebuild-*")), [])

    def test_rebuild_exception_leaves_live_database_intact(self) -> None:
        _make_database(self.db, documents=2)

        def exploding_indexer(store, embedder, cfg):
            class Exploding:
                def sync(self, roots):
                    raise RuntimeError("sync exploded")

            return Exploding()

        with self.assertRaises(RebuildError):
            self._rebuild(SyncReport(), indexer_factory=exploding_indexer)

        self.assertEqual(_document_count(self.db), 2)
        self.assertEqual(list(self.tmp.glob(".*.rebuild-*")), [])

    def test_empty_rebuild_does_not_replace_nonempty_live_database(self) -> None:
        _make_database(self.db, documents=2)

        with self.assertRaises(RebuildError):
            self._rebuild(SyncReport(unchanged=0), new_documents=0)

        self.assertEqual(_document_count(self.db), 2)
        self.assertEqual(list(self.tmp.glob(".*.rebuild-*")), [])

    def test_empty_rebuild_can_be_allowed(self) -> None:
        _make_database(self.db, documents=2)

        report = self._rebuild(
            SyncReport(skipped=1), new_documents=0, allow_empty=True
        )

        self.assertEqual(report.skipped, 1)
        self.assertIsNotNone(_document_count(self.db))
        self.assertEqual(_document_count(self.db), 0)

    def test_live_connection_blocks_swap_and_preserves_database(self) -> None:
        _make_database(self.db, documents=2)
        report = SyncReport(added=1)

        with lifecycle_lock(self.db, shared=True), self.assertRaises(RebuildError):
            self._rebuild(report, lock_timeout=0.0)

        self.assertEqual(_document_count(self.db), 2)
        self.assertEqual(list(self.tmp.glob(".*.rebuild-*")), [])

    def test_exclusive_rebuild_lock_prevents_concurrent_rebuild(self) -> None:
        _make_database(self.db, documents=2)

        with lifecycle_lock(self.db, shared=False), self.assertRaises(RebuildError):
            self._rebuild(SyncReport(added=1))

        self.assertEqual(_document_count(self.db), 2)
        self.assertEqual(list(self.tmp.glob(".*.rebuild-*")), [])

    def test_new_database_replaces_missing_live_database(self) -> None:
        result = self._rebuild(SyncReport(added=1))

        self.assertEqual(result.added, 1)
        self.assertEqual(_document_count(self.db), 1)

    def test_busy_wal_checkpoint_refuses_swap_and_keeps_live_database(self) -> None:
        _make_database(self.db, documents=2)
        writer = sqlite3.connect(str(self.db))
        self.addCleanup(writer.close)
        writer.execute("PRAGMA journal_mode=WAL").fetchone()
        reader = sqlite3.connect(str(self.db), timeout=0.1)
        self.addCleanup(reader.close)
        reader.execute("BEGIN")
        reader.execute("SELECT COUNT(*) FROM documents").fetchone()
        writer.execute(
            "INSERT INTO documents(path) VALUES ('/old/committed-after-snapshot.txt')"
        )
        writer.commit()
        wal = Path(str(self.db) + "-wal")
        self.assertTrue(wal.exists())

        with self.assertRaises(RebuildError) as raised:
            self._rebuild(SyncReport(added=1))

        self.assertIsInstance(raised.exception, LifecycleError)
        self.assertIn("busy", str(raised.exception).lower())
        self.assertEqual(_document_count(self.db), 3)
        self.assertTrue(wal.exists())

        reader.rollback()

    def test_failed_atomic_replace_keeps_live_database(self) -> None:
        _make_database(self.db, documents=2)

        with mock.patch(
            "seekr.lifecycle.os.replace", side_effect=OSError("replace failed")
        ), self.assertRaises(RebuildError):
            self._rebuild(SyncReport(added=1))

        self.assertEqual(_document_count(self.db), 2)
        self.assertEqual(list(self.tmp.glob(".*.rebuild-*")), [])

    def test_replace_retries_then_explains(self) -> None:
        _make_database(self.db, documents=2)
        sleeps: list[float] = []

        with mock.patch(
            "seekr.lifecycle.os.replace", side_effect=PermissionError(13, "in use")
        ) as replace, mock.patch(
            "seekr.lifecycle._sleep", side_effect=sleeps.append
        ), self.assertRaises(RebuildError) as raised:
            self._rebuild(SyncReport(added=1))

        self.assertEqual(
            str(raised.exception), "index is in use; close other seekr windows and retry"
        )
        self.assertEqual(replace.call_count, 11)
        self.assertEqual(sleeps, [0.1] * 10)
        self.assertEqual(_document_count(self.db), 2)
        self.assertEqual(list(self.tmp.glob(".*.rebuild-*")), [])

    def test_replace_succeeds_after_transient_permission_errors(self) -> None:
        _make_database(self.db, documents=2)
        real_replace = os.replace
        attempts: list[int] = []

        def flaky(src: str, dst: str) -> None:
            attempts.append(1)
            if len(attempts) < 3:
                raise PermissionError(13, "sharing violation")
            real_replace(src, dst)

        with mock.patch("seekr.lifecycle.os.replace", side_effect=flaky), mock.patch(
            "seekr.lifecycle._sleep"
        ) as sleep:
            self._rebuild(SyncReport(added=1))

        self.assertEqual(len(attempts), 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(_document_count(self.db), 1)

    def test_other_replace_errors_are_not_retried(self) -> None:
        _make_database(self.db, documents=2)
        with mock.patch(
            "seekr.lifecycle.os.replace", side_effect=OSError("disk gone")
        ) as replace, mock.patch("seekr.lifecycle._sleep") as sleep, self.assertRaises(
            RebuildError
        ) as raised:
            self._rebuild(SyncReport(added=1))
        self.assertEqual(replace.call_count, 1)
        sleep.assert_not_called()
        self.assertIn("cannot atomically replace live index", str(raised.exception))

    def test_success_can_replace_a_corrupt_live_database(self) -> None:
        self.db.write_bytes(b"not a sqlite database")

        result = self._rebuild(SyncReport(added=1))

        self.assertEqual(result.added, 1)
        self.assertEqual(_document_count(self.db), 1)

    def test_failure_keeps_a_corrupt_live_database_byte_for_byte(self) -> None:
        original = b"not a sqlite database"
        self.db.write_bytes(original)
        report = SyncReport(added=1, errors=[("bad.txt", "boom")])

        with self.assertRaises(RebuildError):
            self._rebuild(report)

        self.assertEqual(self.db.read_bytes(), original)


@unittest.skipUnless(_HAS_FTS5, "FTS5 is required for the SQLite Store rebuild tests")
class StdlibStoreRebuildTests(unittest.TestCase):
    """Exercise the production Store swap path without numpy/sqlite-vec."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "root"
        self.root.mkdir()
        self.db = self.tmp / "index.db"
        self.cfg = Config(roots=(self.root,), db_path=self.db)
        self._backend = _stdlib_vector_backend()
        self._backend.__enter__()
        self.addCleanup(self._backend.__exit__, None, None, None)

    def _old_store(self):
        from seekr.store import Store

        return Store(self.db, self.cfg.embed_model, 8)

    def test_stdlib_rebuild_swaps_in_a_fresh_searchable_store(self) -> None:
        from seekr.store import Store

        old_store = self._old_store()
        try:
            old_store.upsert_document("/old/only.txt", 1, 1, "a" * 64, "text")
        finally:
            old_store.close()
        (self.root / "one.txt").write_text(_paragraph("alpha"), encoding="utf-8")
        (self.root / "two.txt").write_text(_paragraph("bravo"), encoding="utf-8")

        report = rebuild_index(self.cfg, embedder=_ByteEmbedder())

        self.assertEqual(report.added, 2)
        live = Store(self.db, self.cfg.embed_model, 8)
        try:
            self.assertEqual(live.stats()["documents"], 2)
            self.assertIsNone(live.get_document("/old/only.txt"))
            self.assertEqual(len(live.keyword_search("alpha", 10)), 1)
        finally:
            live.close()
        self.assertEqual(list(self.tmp.glob(".*.rebuild-*")), [])

    def test_stdlib_rebuild_error_keeps_the_old_searchable_store(self) -> None:
        from seekr.store import Store

        old_store = self._old_store()
        try:
            old_store.upsert_document("/old/keep.txt", 1, 1, "a" * 64, "text")
            doc_id = old_store.get_document("/old/keep.txt").id  # type: ignore[union-attr]
            from seekr.models import ChunkSpec
            from seekr.store import StoredChunk

            old_store.replace_chunks(
                doc_id,
                [
                    StoredChunk(
                        spec=ChunkSpec("keepword old text", "", 0.0),
                        text_sha=text_sha("keepword old text"),
                        vector=b"12345678",
                    )
                ],
            )
        finally:
            old_store.close()
        (self.root / "good.txt").write_text(_paragraph("alpha"), encoding="utf-8")
        (self.root / "bad.txt").write_bytes(b"\x00\x01binary")

        with self.assertRaises(RebuildError):
            rebuild_index(self.cfg, embedder=_ByteEmbedder())

        live = Store(self.db, self.cfg.embed_model, 8)
        try:
            self.assertEqual(live.stats(), {"documents": 1, "chunks": 1})
            self.assertIsNotNone(live.get_document("/old/keep.txt"))
            self.assertEqual(len(live.keyword_search("keepword", 10)), 1)
        finally:
            live.close()
        self.assertEqual(list(self.tmp.glob(".*.rebuild-*")), [])


def _paragraph(prefix: str) -> str:
    return " ".join([prefix, *(f"word{number}" for number in range(9))])


@unittest.skipUnless(_HAS_TYPER, "typer is required for CLI tests")
class CliRebuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "root"
        self.root.mkdir()
        self.cfg = Config(roots=(self.root,), db_path=self.tmp / "index.db")

    @staticmethod
    def _runner():
        from typer.testing import CliRunner

        return CliRunner()

    def _app(self):
        import typer

        from seekr.cli_index import register

        app = typer.Typer()
        register(app)
        return app

    def test_register_exposes_index_reindex_and_status(self) -> None:
        app = self._app()
        runner = self._runner()

        for command in ("index", "reindex", "status"):
            result = runner.invoke(app, [command, "--help"])
            self.assertEqual(result.exit_code, 0, result.output)

    def test_reindex_prints_report(self) -> None:
        report = SyncReport(added=2, unchanged=1)
        with (
            mock.patch("seekr.cli_index._load_config", return_value=self.cfg),
            mock.patch(
                "seekr.cli_index.rebuild_index", return_value=report
            ) as rebuild,
        ):
            result = self._runner().invoke(self._app(), ["reindex"])

        self.assertEqual(result.exit_code, 0, result.output)
        rebuild.assert_called_once_with(self.cfg, force=False, allow_empty=False)
        self.assertIn("added: 2", result.output)
        self.assertIn("unchanged: 1", result.output)

    def test_reindex_failure_is_nonzero_and_does_not_print_success(self) -> None:
        with (
            mock.patch("seekr.cli_index._load_config", return_value=self.cfg),
            mock.patch(
                "seekr.cli_index.rebuild_index",
                side_effect=RebuildError("refusing to rebuild: no roots"),
            ),
        ):
            result = self._runner().invoke(self._app(), ["reindex"])

        self.assertEqual(result.exit_code, 1)
        self.assertIn("refusing to rebuild", result.output)

    def test_index_without_roots_exits_nonzero(self) -> None:
        cfg = Config(roots=(), db_path=self.cfg.db_path)
        with mock.patch("seekr.cli_index._load_config", return_value=cfg):
            result = self._runner().invoke(self._app(), ["index"])

        self.assertEqual(result.exit_code, 1)
        self.assertIn("no index roots", result.output)

    def test_status_prints_counts_and_last_error_count(self) -> None:
        store = _FakeStatusStore({"documents": 4, "chunks": 9}, errors=2)
        with (
            mock.patch("seekr.cli_index._load_config", return_value=self.cfg),
            mock.patch("seekr.cli_index._stored_dim", return_value=8),
            mock.patch("seekr.cli_index._open_store", return_value=store),
        ):
            result = self._runner().invoke(self._app(), ["status"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("documents: 4", result.output)
        self.assertIn("chunks: 9", result.output)
        self.assertIn("last error count: 2", result.output)
        self.assertTrue(store.closed)

    def test_status_on_missing_database_prints_zero(self) -> None:
        with mock.patch("seekr.cli_index._load_config", return_value=self.cfg):
            result = self._runner().invoke(self._app(), ["status"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("documents: 0", result.output)
        self.assertIn("chunks: 0", result.output)
        self.assertIn("last error count: 0", result.output)

    def test_index_uses_config_roots_and_prints_report(self) -> None:
        report = SyncReport(added=1, skipped=2)
        store = _FakeStatusStore({"documents": 1, "chunks": 1}, errors=0)
        fake_indexer = _FakeIndexer(report)

        with (
            mock.patch("seekr.cli_index._load_config", return_value=self.cfg),
            mock.patch("seekr.cli_index._build_embedder", return_value=_FakeEmbedder()),
            mock.patch("seekr.cli_index._open_store", return_value=store),
            mock.patch("seekr.cli_index.Indexer", return_value=fake_indexer),
        ):
            result = self._runner().invoke(self._app(), ["index"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(fake_indexer.sync_calls, [[self.root]])
        self.assertIn("added: 1", result.output)
        self.assertIn("skipped: 2", result.output)
        self.assertTrue(store.closed)


if __name__ == "__main__":
    unittest.main()
