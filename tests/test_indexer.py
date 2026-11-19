"""Tests for incremental indexing, failure isolation and move handling.

The indexer is written against the ``Store`` protocol, so most tests use a
small deterministic in-memory store and embedder made only from the standard
library.  A separate ``RealStoreIndexerTests`` class exercises the production
SQLite store and is skipped when numpy/sqlite-vec are unavailable.
"""

from __future__ import annotations

import copy
import errno
import importlib.util
import os
import sqlite3
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

from seekr.config import Config
from seekr.extract import ExtractionError
from seekr.indexer import Indexer
from seekr.models import text_sha
from seekr.store import DocRow

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_HAS_SQLITE_VEC = importlib.util.find_spec("sqlite_vec") is not None


def _has_fts5() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        options = {row[0] for row in conn.execute("PRAGMA compile_options")}
    finally:
        conn.close()
    return "ENABLE_FTS5" in options


_HAS_FTS5 = _has_fts5()


@dataclass
class _FakeChunk:
    id: int
    doc_id: int
    text: str
    text_sha: str
    locator: str
    loc_start: float
    vector: object


class FakeStore:
    """In-memory Store implementation using only the standard library."""

    def __init__(self) -> None:
        self.documents: dict[str, DocRow] = {}
        self.chunks: dict[int, _FakeChunk] = {}
        self._next_doc_id = 1
        self._next_chunk_id = 1
        self.move_calls: list[tuple[str, str, int, int]] = []
        self.delete_calls: list[str] = []
        self.last_error_count = 0
        self.fail_replace = False
        self.fail_upsert = False
        self._snapshots: list[tuple[object, ...]] = []

    # -- test helpers ------------------------------------------------------
    def _snapshot(self) -> tuple[object, ...]:
        return (
            copy.deepcopy(self.documents),
            copy.deepcopy(self.chunks),
            self._next_doc_id,
            self._next_chunk_id,
        )

    def _restore(self, snapshot: tuple[object, ...]) -> None:
        documents, chunks, next_doc_id, next_chunk_id = snapshot
        self.documents = documents  # type: ignore[assignment]
        self.chunks = chunks  # type: ignore[assignment]
        self._next_doc_id = next_doc_id  # type: ignore[assignment]
        self._next_chunk_id = next_chunk_id  # type: ignore[assignment]

    @contextmanager
    def transaction(self, *, write: bool = False):
        snapshot = self._snapshot() if write else None
        if write:
            self._snapshots.append(snapshot)  # type: ignore[arg-type]
        try:
            yield
        except BaseException:
            if write:
                self._snapshots.pop()
                self._restore(snapshot)  # type: ignore[arg-type]
            raise
        else:
            if write:
                self._snapshots.pop()

    # -- Store protocol ----------------------------------------------------
    def upsert_document(
        self, path: str, mtime_ns: int, size: int, sha256: str, kind: str
    ) -> int:
        if self.fail_upsert:
            raise RuntimeError("injected document write failure")
        existing = self.documents.get(path)
        if existing is None:
            doc_id = self._next_doc_id
            self._next_doc_id += 1
        else:
            doc_id = existing.id
        self.documents[path] = DocRow(
            id=doc_id,
            path=path,
            mtime_ns=int(mtime_ns),
            size=int(size),
            sha256=sha256,
            kind=kind,
        )
        return doc_id

    def get_document(self, path: str) -> DocRow | None:
        return self.documents.get(str(path))

    def replace_chunks(self, doc_id: int, chunks: list[object]) -> None:
        for chunk_id, chunk in list(self.chunks.items()):
            if chunk.doc_id == doc_id:
                del self.chunks[chunk_id]
        for item in chunks:
            chunk_id = self._next_chunk_id
            self._next_chunk_id += 1
            self.chunks[chunk_id] = _FakeChunk(
                id=chunk_id,
                doc_id=doc_id,
                text=item.spec.text,
                text_sha=item.text_sha,
                locator=item.spec.locator,
                loc_start=float(item.spec.loc_start),
                vector=item.vector,
            )
        if self.fail_replace:
            raise RuntimeError("injected chunk write failure")

    def delete_document(self, path: str) -> None:
        self.delete_calls.append(str(path))
        doc = self.documents.pop(str(path), None)
        if doc is None:
            return
        for chunk_id, chunk in list(self.chunks.items()):
            if chunk.doc_id == doc.id:
                del self.chunks[chunk_id]

    def move_document(self, src: str, dst: str, *, mtime_ns: int, size: int) -> int:
        self.move_calls.append((str(src), str(dst), int(mtime_ns), int(size)))
        doc = self.documents.get(str(src))
        if doc is None:
            raise KeyError(f"document not found: {src}")
        if src != dst and str(dst) in self.documents:
            raise ValueError(f"destination is already indexed: {dst}")
        del self.documents[str(src)]
        self.documents[str(dst)] = DocRow(
            id=doc.id,
            path=str(dst),
            mtime_ns=int(mtime_ns),
            size=int(size),
            sha256=doc.sha256,
            kind=doc.kind,
        )
        return doc.id

    def list_paths(self, prefix: str | None = None) -> list[str]:
        paths = sorted(self.documents)
        if prefix is None:
            return paths
        base = str(prefix).rstrip("/")
        if not base:
            return paths
        return [path for path in paths if path == base or path.startswith(base + "/")]

    def find_vector_by_text_sha(self, sha: str) -> object | None:
        for chunk in self.chunks.values():
            if chunk.text_sha == sha:
                return chunk.vector
        return None

    def stats(self) -> dict[str, int]:
        return {"documents": len(self.documents), "chunks": len(self.chunks)}

    def set_last_error_count(self, n: int) -> None:
        self.last_error_count = int(n)

    def get_last_error_count(self) -> int:
        return self.last_error_count


class RecordingEmbedder:
    """Deterministic embedder that records every batch."""

    model_name = "fake"
    dim = 8

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail = False

    def embed_documents(self, texts: list[str]) -> list[str]:
        batch = list(texts)
        self.calls.append(batch)
        if self.fail:
            raise RuntimeError("injected embedding failure")
        return [f"vec:{text_sha(text)}" for text in batch]

    def embed_query(self, text: str) -> str:
        return f"vec:{text_sha(text)}"


class ByteEmbedder(RecordingEmbedder):
    """An embedder whose vectors can be stored by the stdlib Store harness."""

    def embed_documents(self, texts: list[str]) -> list[bytes]:
        batch = list(texts)
        self.calls.append(batch)
        if self.fail:
            raise RuntimeError("injected embedding failure")
        return [text_sha(text).encode("ascii")[: self.dim] for text in batch]


@contextmanager
def _stdlib_vector_backend():
    """Let the production Store run on plain SQLite when sqlite-vec is absent."""

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


def _paragraph(prefix: str) -> str:
    return " ".join([prefix, *(f"word{number}" for number in range(9))])


def _short_paragraph(prefix: str) -> str:
    return f"{prefix} short unique words"


class _IndexerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "root"
        self.root.mkdir()
        self.store = FakeStore()
        self.embedder = RecordingEmbedder()
        self.cfg = self._config()
        self.indexer = Indexer(self.store, self.embedder, self.cfg)

    def _config(self, **overrides: object) -> Config:
        values: dict[str, object] = {
            "roots": (self.root,),
            "db_path": self.tmp / "index.db",
            "chunk_target_words": 10,
            "chunk_max_words": 10,
            "chunk_overlap_words": 0,
        }
        values.update(overrides)
        return Config(**values)  # type: ignore[arg-type]

    def write(self, relative: str, text: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def stored(self, path: Path) -> DocRow | None:
        return self.store.get_document(str(path.resolve()))


class SyncTests(_IndexerTestCase):
    def test_sync_adds_files(self) -> None:
        self.write("a.txt", "alpha beta gamma")
        self.write("b.txt", "delta epsilon")
        self.write("c.txt", "zeta eta theta")

        report = self.indexer.sync([self.root])

        self.assertEqual((report.added, report.updated, report.unchanged), (3, 0, 0))
        self.assertEqual(report.errors, [])
        self.assertEqual(self.store.stats(), {"documents": 3, "chunks": 3})
        self.assertEqual(len(self.embedder.calls), 3)

    def test_resync_unchanged_makes_no_embed_calls(self) -> None:
        self.write("a.txt", "alpha beta gamma")
        self.write("b.txt", "delta epsilon")
        self.write("c.txt", "zeta eta theta")
        self.indexer.sync([self.root])
        calls = len(self.embedder.calls)

        report = self.indexer.sync([self.root])

        self.assertEqual((report.added, report.updated, report.unchanged), (0, 0, 3))
        self.assertEqual(len(self.embedder.calls), calls)

    def test_touch_only_is_unchanged_and_not_reembedded(self) -> None:
        path = self.write("a.txt", "alpha beta gamma")
        self.indexer.sync([self.root])
        calls = len(self.embedder.calls)
        new_mtime = path.stat().st_mtime_ns + 1_000_000_000
        os.utime(path, ns=(new_mtime, new_mtime))

        report = self.indexer.sync([self.root])

        self.assertEqual((report.added, report.updated, report.unchanged), (0, 0, 1))
        self.assertEqual(len(self.embedder.calls), calls)
        doc = self.stored(path)
        assert doc is not None
        self.assertEqual(doc.mtime_ns, new_mtime)

    def test_edit_one_paragraph_embeds_only_changed_chunks(self) -> None:
        first = _short_paragraph("alpha")
        second = _short_paragraph("bravo")
        third = _short_paragraph("charlie")
        path = self.write(
            "notes.md",
            "\n".join(
                [
                    f"# Section A\n{first}",
                    f"# Section B\n{second}",
                    f"# Section C\n{third}",
                ]
            ),
        )
        self.indexer.sync([self.root])
        self.assertEqual(len(self.embedder.calls), 1)
        self.assertEqual(len(self.embedder.calls[0]), 3)

        replacement = _short_paragraph("delta")
        path.write_text(
            "\n".join(
                [
                    f"# Section A\n{first}",
                    f"# Section B\n{replacement}",
                    f"# Section C\n{third}",
                ]
            ),
            encoding="utf-8",
        )
        report = self.indexer.sync([self.root])

        self.assertEqual((report.added, report.updated, report.unchanged), (0, 1, 0))
        self.assertEqual(len(self.embedder.calls), 2)
        self.assertEqual(len(self.embedder.calls[1]), 1)
        self.assertIn("delta", self.embedder.calls[1][0])

    def test_rename_costs_zero_embeddings(self) -> None:
        first = _paragraph("alpha")
        second = _paragraph("bravo")
        path = self.write("old.txt", f"{first}\n{second}")
        self.indexer.sync([self.root])
        calls = list(self.embedder.calls)
        chunk_count = self.store.stats()["chunks"]

        renamed = self.root / "new.txt"
        path.rename(renamed)
        report = self.indexer.sync([self.root])

        self.assertEqual((report.added, report.removed), (1, 1))
        self.assertEqual(self.embedder.calls, calls)
        self.assertIsNone(self.stored(path))
        self.assertIsNotNone(self.stored(renamed))
        self.assertEqual(self.store.stats(), {"documents": 1, "chunks": chunk_count})

    def test_deleted_file_is_removed(self) -> None:
        path = self.write("a.txt", "alpha beta gamma")
        self.indexer.sync([self.root])

        path.unlink()
        report = self.indexer.sync([self.root])

        self.assertEqual((report.added, report.removed), (0, 1))
        self.assertEqual(self.store.stats(), {"documents": 0, "chunks": 0})

    def test_ignore_globs_and_oversize_skipped(self) -> None:
        self.write("node_modules/dependency.md", "ignored dependency")
        self.write("large.txt", "x" * 100)
        cfg = self._config(max_file_bytes=32)
        indexer = Indexer(self.store, self.embedder, cfg)

        report = indexer.sync([self.root])

        self.assertEqual(report.skipped, 2)
        self.assertEqual(report.added, 0)
        self.assertEqual(self.store.stats(), {"documents": 0, "chunks": 0})
        self.assertEqual(self.embedder.calls, [])

    def test_corrupt_file_recorded_and_sync_continues(self) -> None:
        self.write("bad.pdf", "%PDF-1.4 garbage")
        self.write("good.txt", "alpha beta gamma")
        real_extract = __import__("seekr.indexer", fromlist=["extract"]).extract

        def fake_extract(path: Path, kind: str):
            if path.suffix == ".pdf":
                raise ExtractionError("corrupt pdf")
            return real_extract(path, kind)

        with mock.patch("seekr.indexer.extract", side_effect=fake_extract):
            report = self.indexer.sync([self.root])

        self.assertEqual(report.added, 1)
        self.assertEqual(len(report.errors), 1)
        self.assertIn("corrupt pdf", report.errors[0][1])
        self.assertEqual(self.store.stats(), {"documents": 1, "chunks": 1})

    def test_file_vanishes_between_scan_and_read_is_removed(self) -> None:
        path = self.write("vanish.txt", "alpha beta gamma")
        self.indexer.sync([self.root])
        path.write_text("changed content so extraction is attempted", encoding="utf-8")

        def vanishing_extract(source: Path, kind: str):
            source.unlink()
            raise FileNotFoundError(source)

        with mock.patch("seekr.indexer.extract", side_effect=vanishing_extract):
            report = self.indexer.sync([self.root])

        self.assertEqual(report.errors, [])
        self.assertEqual(report.removed, 1)
        self.assertIsNone(self.stored(path))

    def test_symlink_loop_does_not_hang_or_create_documents(self) -> None:
        self.write("real.txt", "alpha beta gamma")
        os.symlink(self.root, self.root / "self")
        os.symlink(self.root / "loop_b", self.root / "loop_a")
        os.symlink(self.root / "loop_a", self.root / "loop_b")

        report = self.indexer.sync([self.root])

        self.assertEqual(report.added, 1)
        self.assertEqual(report.errors, [])
        self.assertEqual(self.store.stats(), {"documents": 1, "chunks": 1})

    def test_symlinked_file_outside_root_is_not_indexed(self) -> None:
        outside = self.tmp / "outside"
        outside.mkdir()
        secret = outside / "secret.md"
        secret.write_text("outside secret", encoding="utf-8")
        os.symlink(secret, self.root / "link.md")

        report = self.indexer.sync([self.root])

        self.assertEqual(report.added, 0)
        self.assertEqual(report.skipped, 1)
        self.assertTrue(any("outside" in message for _, message in report.warnings))
        self.assertEqual(self.store.stats(), {"documents": 0, "chunks": 0})

    def test_component_containment_prevents_pruning_sibling_root(self) -> None:
        sibling = self.tmp / "root2"
        sibling.mkdir()
        sibling_doc = str(sibling / "missing.md")
        self.store.upsert_document(sibling_doc, 1, 1, "a" * 64, "markdown")

        report = self.indexer.sync([self.root])

        self.assertEqual(report.removed, 0)
        self.assertIsNotNone(self.store.get_document(sibling_doc))

    def test_missing_root_is_not_pruned(self) -> None:
        doc_path = str(self.root / "gone.md")
        self.store.upsert_document(doc_path, 1, 1, "a" * 64, "markdown")
        self.root.rmdir()

        report = self.indexer.sync([self.root])

        self.assertEqual(len(report.errors), 1)
        self.assertEqual(report.removed, 0)
        self.assertIsNotNone(self.store.get_document(doc_path))

    def test_walk_error_disables_pruning_for_that_root(self) -> None:
        doc_path = str(self.root / "never-seen.md")
        self.store.upsert_document(doc_path, 1, 1, "a" * 64, "markdown")

        def failing_walk(root, topdown=True, onerror=None, followlinks=False):
            if onerror is not None:
                onerror(OSError(errno.EACCES, "permission denied", str(root)))
            return iter(())

        with mock.patch("seekr.indexer.os.walk", side_effect=failing_walk):
            report = self.indexer.sync([self.root])

        self.assertEqual(len(report.errors), 1)
        self.assertEqual(report.removed, 0)
        self.assertIsNotNone(self.store.get_document(doc_path))

    def test_empty_roots_do_not_prune_anything(self) -> None:
        doc_path = str(self.root / "gone.md")
        self.store.upsert_document(doc_path, 1, 1, "a" * 64, "markdown")
        self.root.rmdir()

        report = self.indexer.sync([])

        self.assertEqual(report.removed, 0)
        self.assertEqual(report.errors, [])
        self.assertIsNotNone(self.store.get_document(doc_path))

    def test_sync_report_includes_extraction_warnings(self) -> None:
        (self.root / "latin.txt").write_bytes(b"caf\xe9 notes")

        report = self.indexer.sync([self.root])

        self.assertEqual(report.added, 1)
        self.assertEqual(report.errors, [])
        self.assertTrue(
            any("latin-1" in message for _, message in report.warnings)
        )

    def test_get_report_returns_latest_report_and_error_count(self) -> None:
        self.assertIsNone(self.indexer.get_report())
        self.write("bad.txt", "")
        (self.root / "bad.txt").write_bytes(b"\x00\x01binary")

        report = self.indexer.sync([self.root])

        self.assertIs(self.indexer.get_report(), report)
        self.assertEqual(len(report.errors), 1)
        self.assertEqual(self.store.get_last_error_count(), 1)


class PreservationAndRetryTests(_IndexerTestCase):
    def test_extraction_failure_keeps_last_good_metadata_and_chunks(self) -> None:
        path = self.write("notes.txt", "alpha beta gamma")
        self.indexer.sync([self.root])
        old_doc = self.stored(path)
        assert old_doc is not None
        old_chunks = copy.deepcopy(self.store.chunks)

        path.write_bytes(b"\x00\x01binary")
        report = self.indexer.sync([self.root])

        self.assertEqual(len(report.errors), 1)
        new_doc = self.stored(path)
        assert new_doc is not None
        self.assertEqual(new_doc.mtime_ns, old_doc.mtime_ns)
        self.assertEqual(new_doc.sha256, old_doc.sha256)
        self.assertEqual(self.store.chunks, old_chunks)

    def test_embedding_failure_keeps_last_good_metadata_and_chunks(self) -> None:
        path = self.write("notes.txt", "alpha beta gamma")
        self.indexer.sync([self.root])
        old_doc = self.stored(path)
        old_chunks = copy.deepcopy(self.store.chunks)
        path.write_text("delta epsilon zeta", encoding="utf-8")
        self.embedder.fail = True

        report = self.indexer.sync([self.root])

        self.assertEqual(len(report.errors), 1)
        new_doc = self.stored(path)
        assert new_doc is not None and old_doc is not None
        self.assertEqual(new_doc.mtime_ns, old_doc.mtime_ns)
        self.assertEqual(new_doc.sha256, old_doc.sha256)
        self.assertEqual(self.store.chunks, old_chunks)

    def test_atomic_write_failure_rolls_back_metadata_and_chunks(self) -> None:
        path = self.write("notes.txt", "alpha beta gamma")
        self.indexer.sync([self.root])
        old_doc = self.stored(path)
        old_chunks = copy.deepcopy(self.store.chunks)
        path.write_text("delta epsilon zeta", encoding="utf-8")
        self.store.fail_replace = True

        report = self.indexer.sync([self.root])

        self.assertEqual(len(report.errors), 1)
        new_doc = self.stored(path)
        assert new_doc is not None and old_doc is not None
        self.assertEqual(new_doc.mtime_ns, old_doc.mtime_ns)
        self.assertEqual(new_doc.sha256, old_doc.sha256)
        self.assertEqual(self.store.chunks, old_chunks)

    def test_failed_new_document_write_leaves_no_partial_document(self) -> None:
        self.write("notes.txt", "alpha beta gamma")
        self.store.fail_replace = True

        report = self.indexer.sync([self.root])

        self.assertEqual(len(report.errors), 1)
        self.assertEqual(self.store.stats(), {"documents": 0, "chunks": 0})

    def test_transient_change_is_retried_then_indexed(self) -> None:
        self.write("churn.txt", "initial text")
        real_extract = __import__("seekr.indexer", fromlist=["extract"]).extract
        calls = 0

        def churning_extract(source: Path, kind: str):
            nonlocal calls
            calls += 1
            if calls == 1:
                source.write_text("settled text after one write", encoding="utf-8")
                stamp = 2_000_000_000
                os.utime(source, ns=(stamp, stamp))
            return real_extract(source, kind)

        with mock.patch("seekr.indexer.extract", side_effect=churning_extract):
            report = self.indexer.sync([self.root])

        self.assertEqual(calls, 2)
        self.assertEqual(report.added, 1)
        self.assertEqual(report.errors, [])
        self.assertEqual(len(self.embedder.calls), 1)

    def test_changing_file_is_retried_a_bounded_number_of_times(self) -> None:
        self.write("churn.txt", "initial text")
        real_extract = __import__("seekr.indexer", fromlist=["extract"]).extract
        calls = 0

        def churning_extract(source: Path, kind: str):
            nonlocal calls
            calls += 1
            source.write_text(f"attempt {calls} text {'x' * calls}", encoding="utf-8")
            stamp = 1_000_000_000 + calls * 1_000_000
            os.utime(source, ns=(stamp, stamp))
            return real_extract(source, kind)

        with mock.patch("seekr.indexer.extract", side_effect=churning_extract):
            report = self.indexer.sync([self.root])

        self.assertEqual(calls, 3)
        self.assertEqual(len(report.errors), 1)
        self.assertEqual(self.embedder.calls, [])
        self.assertEqual(self.store.stats(), {"documents": 0, "chunks": 0})


class MovePathTests(_IndexerTestCase):
    def test_unchanged_rename_preserves_chunks_without_embedding(self) -> None:
        src = self.write("old.txt", _paragraph("alpha"))
        self.indexer.sync([self.root])
        calls = list(self.embedder.calls)
        old_doc = self.stored(src)
        assert old_doc is not None
        old_chunk_ids = set(self.store.chunks)
        dst = self.root / "new.txt"
        src.rename(dst)

        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "unchanged")
        self.assertEqual(self.embedder.calls, calls)
        self.assertEqual(len(self.store.move_calls), 1)
        self.assertIsNone(self.stored(src))
        new_doc = self.stored(dst)
        assert new_doc is not None
        self.assertEqual(new_doc.id, old_doc.id)
        self.assertEqual(set(self.store.chunks), old_chunk_ids)

    def test_changed_content_reuses_source_until_destination_is_prepared(self) -> None:
        src = self.write("old.txt", _paragraph("alpha"))
        self.indexer.sync([self.root])
        calls = len(self.embedder.calls)
        dst = self.root / "new.txt"
        src.rename(dst)
        dst.write_text(_paragraph("bravo"), encoding="utf-8")

        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "added")
        self.assertEqual(len(self.embedder.calls), calls + 1)
        self.assertIsNone(self.stored(src))
        self.assertIsNotNone(self.stored(dst))

    def test_destination_already_prepared_costs_zero_embeddings(self) -> None:
        src = self.write("old.txt", _paragraph("alpha"))
        self.indexer.sync([self.root])
        dst = self.root / "new.txt"
        src.rename(dst)
        self.indexer.index_path(dst)
        calls = len(self.embedder.calls)

        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "removed")
        self.assertEqual(len(self.embedder.calls), calls)
        self.assertEqual(self.store.move_calls, [])
        self.assertIsNone(self.stored(src))
        self.assertIsNotNone(self.stored(dst))

    def test_failed_destination_preparation_keeps_source_document(self) -> None:
        src = self.write("old.txt", _paragraph("alpha"))
        self.indexer.sync([self.root])
        calls = list(self.embedder.calls)
        dst = self.root / "new.txt"
        src.rename(dst)
        dst.write_text(_paragraph("bravo"), encoding="utf-8")

        def failing_extract(path: Path, kind: str):
            raise ExtractionError("cannot read destination")

        with mock.patch("seekr.indexer.extract", side_effect=failing_extract):
            result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "skipped")
        self.assertEqual(self.embedder.calls, calls)
        self.assertIsNotNone(self.stored(src))
        self.assertIsNone(self.stored(dst))

    def test_missing_destination_removes_source_document(self) -> None:
        src = self.write("old.txt", _paragraph("alpha"))
        self.indexer.sync([self.root])
        dst = self.root / "new.txt"
        src.unlink()

        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "removed")
        self.assertIsNone(self.stored(src))
        self.assertIsNone(self.stored(dst))
        self.assertEqual(self.store.stats(), {"documents": 0, "chunks": 0})

    def test_chained_move_with_vanished_destination_removes_source(self) -> None:
        src = self.write("a.txt", _paragraph("alpha"))
        self.indexer.sync([self.root])
        second = self.root / "b.txt"
        third = self.root / "c.txt"
        src.rename(second)
        second.rename(third)

        result = self.indexer.move_path(src, second)

        self.assertEqual(result, "removed")
        self.assertIsNone(self.stored(src))
        self.assertIsNone(self.stored(second))
        self.assertIsNone(self.stored(third))
        self.assertEqual(self.store.stats(), {"documents": 0, "chunks": 0})

    def test_changed_extension_reindexes_with_new_kind(self) -> None:
        src = self.write("old.txt", "alpha beta gamma")
        self.indexer.sync([self.root])
        dst = self.root / "new.md"
        src.rename(dst)

        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "added")
        self.assertEqual(self.store.move_calls, [])
        self.assertIsNone(self.stored(src))
        doc = self.stored(dst)
        assert doc is not None
        self.assertEqual(doc.kind, "markdown")

    def test_overwrite_with_stale_destination_row_is_reindexed(self) -> None:
        src = self.write("source.txt", "alpha alpha alpha")
        dst = self.write("dest.txt", "bravo bravo bravo")
        self.indexer.sync([self.root])
        src_doc = self.stored(src)
        dst_doc = self.stored(dst)
        assert src_doc is not None and dst_doc is not None
        old_mtime = dst_doc.mtime_ns
        old_size = dst_doc.size
        calls = len(self.embedder.calls)

        src.rename(dst)
        os.utime(dst, ns=(old_mtime, old_mtime))
        self.assertEqual(dst.stat().st_size, old_size)

        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "updated")
        self.assertEqual(len(self.embedder.calls), calls)
        new_doc = self.stored(dst)
        assert new_doc is not None
        self.assertEqual(new_doc.sha256, src_doc.sha256)
        self.assertIsNone(self.stored(src))
        texts = {
            chunk.text
            for chunk in self.store.chunks.values()
            if chunk.doc_id == new_doc.id
        }
        self.assertEqual(texts, {"alpha alpha alpha"})

    def test_move_to_ignored_destination_removes_stale_source(self) -> None:
        cfg = self._config(ignore_globs=("ignored.txt",))
        self.indexer = Indexer(self.store, self.embedder, cfg)
        src = self.write("old.txt", _paragraph("alpha"))
        self.indexer.sync([self.root])
        calls = list(self.embedder.calls)
        dst = self.root / "ignored.txt"
        src.rename(dst)

        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "removed")
        self.assertEqual(self.embedder.calls, calls)
        self.assertIsNone(self.stored(src))
        self.assertIsNone(self.stored(dst))
        self.assertEqual(self.store.stats(), {"documents": 0, "chunks": 0})

    def test_move_to_unsupported_destination_removes_stale_source(self) -> None:
        src = self.write("old.txt", _paragraph("alpha"))
        self.indexer.sync([self.root])
        calls = list(self.embedder.calls)
        dst = self.root / "new.unknown"
        src.rename(dst)

        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "removed")
        self.assertEqual(self.embedder.calls, calls)
        self.assertIsNone(self.stored(src))
        self.assertEqual(self.store.stats(), {"documents": 0, "chunks": 0})

    def test_move_to_oversized_destination_removes_stale_source(self) -> None:
        src = self.write("old.txt", "alpha")
        self.indexer.sync([self.root])
        calls = list(self.embedder.calls)
        self.indexer = Indexer(
            self.store, self.embedder, self._config(max_file_bytes=16)
        )
        dst = self.root / "new.txt"
        src.rename(dst)
        dst.write_text("alpha plus far too much extra content", encoding="utf-8")

        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "removed")
        self.assertEqual(self.embedder.calls, calls)
        self.assertIsNone(self.stored(src))
        self.assertEqual(self.store.stats(), {"documents": 0, "chunks": 0})

    def test_chained_moves_reconcile_each_source_in_order(self) -> None:
        first = self.write("a.txt", _paragraph("alpha"))
        self.indexer.sync([self.root])
        calls = list(self.embedder.calls)
        second = self.root / "b.txt"
        third = self.root / "c.txt"

        first.rename(second)
        self.assertEqual(self.indexer.move_path(first, second), "unchanged")
        second.rename(third)
        self.assertEqual(self.indexer.move_path(second, third), "unchanged")

        self.assertEqual(self.embedder.calls, calls)
        self.assertIsNone(self.stored(first))
        self.assertIsNone(self.stored(second))
        self.assertIsNotNone(self.stored(third))

    def test_code_language_change_is_not_a_plain_move(self) -> None:
        src = self.write("old.py", "def alpha():\n    return 1\n")
        self.indexer.sync([self.root])
        dst = self.root / "new.js"
        src.rename(dst)

        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "added")
        self.assertEqual(self.store.move_calls, [])
        self.assertIsNone(self.stored(src))
        doc = self.stored(dst)
        assert doc is not None
        self.assertEqual(doc.kind, "code")


@unittest.skipUnless(_HAS_FTS5, "FTS5 is required for the SQLite Store harness")
class StdlibStoreIndexerTests(unittest.TestCase):
    """Exercise the production Store SQLite code without numpy/sqlite-vec."""

    def setUp(self) -> None:
        from seekr.store import Store

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "root"
        self.root.mkdir()
        self.cfg = Config(
            roots=(self.root,),
            db_path=self.tmp / "index.db",
            chunk_target_words=10,
            chunk_max_words=10,
            chunk_overlap_words=0,
        )
        self._backend = _stdlib_vector_backend()
        self._backend.__enter__()
        self.addCleanup(self._backend.__exit__, None, None, None)
        self.store = Store(self.cfg.db_path, self.cfg.embed_model, 8)
        self.addCleanup(self.store.close)
        self.embedder = ByteEmbedder()
        self.indexer = Indexer(self.store, self.embedder, self.cfg)

    def write(self, relative: str, text: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_add_resync_rename_delete_and_failure_with_real_store(self) -> None:
        first = self.write("first.txt", _paragraph("alpha"))
        second = self.write("second.txt", _paragraph("bravo"))
        report = self.indexer.sync([self.root])
        self.assertEqual(report.added, 2)
        calls = len(self.embedder.calls)

        report = self.indexer.sync([self.root])
        self.assertEqual(report.unchanged, 2)
        self.assertEqual(len(self.embedder.calls), calls)

        renamed = self.root / "renamed.txt"
        first.rename(renamed)
        report = self.indexer.sync([self.root])
        self.assertEqual((report.added, report.removed), (1, 1))
        self.assertEqual(len(self.embedder.calls), calls)

        second.write_bytes(b"\x00\x01binary")
        report = self.indexer.sync([self.root])
        self.assertEqual(len(report.errors), 1)
        old = self.store.get_document(str(second.resolve()))
        assert old is not None
        self.assertEqual(len(self.store.keyword_search("bravo", 10)), 1)

        second.unlink()
        report = self.indexer.sync([self.root])
        self.assertEqual(report.removed, 1)
        self.assertIsNone(self.store.get_document(str(second.resolve())))

    def test_real_store_move_path_preserves_chunk_ids_without_embedding(self) -> None:
        src = self.write("old.txt", _paragraph("alpha"))
        self.indexer.sync([self.root])
        calls = len(self.embedder.calls)
        old_id = self.store.keyword_search('"alpha"', 10)[0][0]

        dst = self.root / "new.txt"
        src.rename(dst)
        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "unchanged")
        self.assertEqual(len(self.embedder.calls), calls)
        hits = self.store.keyword_search('"alpha"', 10)
        self.assertEqual(hits[0][0], old_id)
        chunk = self.store.fetch_chunks([old_id])[old_id]
        self.assertEqual(chunk.path, str(dst.resolve()))


@unittest.skipUnless(
    _HAS_NUMPY and _HAS_SQLITE_VEC and _HAS_FTS5,
    "the production sqlite-vec/numpy stack is required",
)
class RealStoreIndexerTests(unittest.TestCase):
    def setUp(self) -> None:
        from seekr.embed import FakeEmbedder
        from seekr.store import Store

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "root"
        self.root.mkdir()
        self.cfg = Config(
            roots=(self.root,),
            db_path=self.tmp / "index.db",
            chunk_target_words=10,
            chunk_max_words=10,
            chunk_overlap_words=0,
        )
        self.store = Store(self.cfg.db_path, self.cfg.embed_model, 8)
        self.addCleanup(self.store.close)
        self.embedder = FakeEmbedder(dim=8)
        self.indexer = Indexer(self.store, self.embedder, self.cfg)

    def write(self, relative: str, text: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_full_sync_add_resync_rename_and_delete(self) -> None:
        first = self.write("first.txt", _paragraph("alpha"))
        second = self.write("second.txt", _paragraph("bravo"))
        report = self.indexer.sync([self.root])
        self.assertEqual(report.added, 2)
        calls = len(self.embedder.calls)

        report = self.indexer.sync([self.root])
        self.assertEqual(report.unchanged, 2)
        self.assertEqual(len(self.embedder.calls), calls)

        renamed = self.root / "renamed.txt"
        first.rename(renamed)
        report = self.indexer.sync([self.root])
        self.assertEqual((report.added, report.removed), (1, 1))
        self.assertEqual(len(self.embedder.calls), calls)

        second.unlink()
        report = self.indexer.sync([self.root])
        self.assertEqual(report.removed, 1)
        self.assertEqual(self.store.stats(), {"documents": 1, "chunks": 1})

    def test_real_move_path_preserves_chunk_ids_and_vectors(self) -> None:
        src = self.write("old.txt", _paragraph("alpha"))
        self.indexer.sync([self.root])
        calls = len(self.embedder.calls)
        old_chunk_id = self.store.keyword_search('"alpha"', 10)[0][0]

        dst = self.root / "new.txt"
        src.rename(dst)
        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "unchanged")
        self.assertEqual(len(self.embedder.calls), calls)
        hits = self.store.keyword_search('"alpha"', 10)
        self.assertEqual(hits[0][0], old_chunk_id)
        self.assertEqual(self.store.fetch_chunks([old_chunk_id])[old_chunk_id].path, str(dst))

    def test_real_failure_preserves_last_good_document(self) -> None:
        path = self.write("notes.txt", "keepword alpha beta")
        self.indexer.sync([self.root])
        old_doc = self.store.get_document(str(path))
        assert old_doc is not None

        path.write_bytes(b"\x00\x01binary")
        report = self.indexer.sync([self.root])

        self.assertEqual(len(report.errors), 1)
        current = self.store.get_document(str(path))
        assert current is not None
        self.assertEqual(current.sha256, old_doc.sha256)
        self.assertEqual(len(self.store.keyword_search("keepword", 10)), 1)


if __name__ == "__main__":
    unittest.main()
