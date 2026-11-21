"""Tests for the Unix-socket query daemon and its CLI fast path."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import socket
import sqlite3
import stat
import tempfile
import threading
import time
import unittest
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest import mock

from _cli_output import strip_ansi
from seekr import cli_daemon, cli_search, client, daemon
from seekr.config import Config
from seekr.embed import FakeEmbedder
from seekr.indexer import Indexer
from seekr.models import Hit
from seekr.search import Searcher
from seekr.store import Store

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


_HAS_FTS5 = _has_fts5()
_NEEDS_REAL_STORE = _HAS_NUMPY and _HAS_SQLITE_VEC and _HAS_FTS5


def _wait_for(predicate, timeout: float = 5.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def _hit(path: str = "/notes/a.md", score: float = 1.0) -> Hit:
    return Hit(
        chunk_id=1,
        path=path,
        locator="L1-L2",
        loc_start=1.0,
        snippet="alpha snippet",
        score=score,
        kind="text",
    )


class _StoreStub:
    def __init__(self) -> None:
        self.stats_calls = 0

    def stats(self) -> dict[str, int]:
        self.stats_calls += 1
        return {"documents": 2, "chunks": 5}


class _FakeSearcher:
    def __init__(
        self,
        hits: list[Hit] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.store = _StoreStub()
        self.daemon_metadata = dict(metadata or {})
        self.hits = list(hits if hits is not None else [_hit()])
        self.error = error
        self.reranker = None
        self.calls: list[dict[str, Any]] = []
        self.reranker_seen: list[bool] = []

    def search(
        self,
        query: str,
        k: int = 10,
        mode: str = "hybrid",
        per_file: int = 2,
        rerank: bool = False,
        path_prefix: str | None = None,
    ) -> list[Hit]:
        self.calls.append(
            {
                "query": query,
                "k": k,
                "mode": mode,
                "per_file": per_file,
                "rerank": rerank,
                "path_prefix": path_prefix,
            }
        )
        self.reranker_seen.append(self.reranker is not None)
        if self.error is not None:
            raise self.error
        return list(self.hits)


class _FakeReranker:
    def __init__(self) -> None:
        self.calls = 0

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.calls += 1
        return [float(len(passages) - index) for index in range(len(passages))]


@contextlib.contextmanager
def _running_server(
    searcher: Any,
    tmp_path: Path,
    *,
    reranker_factory: Any = None,
) -> Iterator[tuple[Path, threading.Event, threading.Thread, list[Exception]]]:
    sock_path = tmp_path / "seekr.sock"
    stop_event = threading.Event()
    errors: list[Exception] = []

    def run() -> None:
        try:
            daemon.serve(
                searcher,
                sock_path,
                stop_event,
                reranker_factory=reranker_factory,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced by the context manager
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    def ready() -> bool:
        if not sock_path.exists():
            return False
        try:
            with _connection(sock_path) as sock:
                _send(sock, {"op": "ping"})
                return _read_json(sock) == {"ok": True}
        except (OSError, ValueError):
            return False

    if not _wait_for(ready):
        stop_event.set()
        thread.join(timeout=5.0)
        raise AssertionError("daemon did not create its socket")
    try:
        yield sock_path, stop_event, thread, errors
    finally:
        stop_event.set()
        thread.join(timeout=5.0)
        if thread.is_alive():
            raise AssertionError("daemon did not stop")
        if errors:
            raise errors[0]


@contextlib.contextmanager
def _connection(sock_path: Path, timeout: float = 2.0):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        sock.connect(str(sock_path))
        yield sock
    finally:
        sock.close()


def _send(sock: socket.socket, payload: dict[str, Any] | bytes) -> None:
    if isinstance(payload, bytes):
        sock.sendall(payload)
    else:
        sock.sendall(json.dumps(payload).encode("utf-8") + b"\n")


def _read_line(sock: socket.socket) -> bytes:
    data = b""
    while not data.endswith(b"\n"):
        chunk = sock.recv(1)
        if not chunk:
            break
        data += chunk
    return data


def _read_json(sock: socket.socket) -> dict[str, Any]:
    return json.loads(_read_line(sock))


def _request(sock_path: Path, payload: dict[str, Any], timeout: float = 2.0) -> dict[str, Any]:
    with _connection(sock_path, timeout) as sock:
        _send(sock, payload)
        return _read_json(sock)


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_search_roundtrip_matches_inprocess_results(self) -> None:
        expected = [_hit("/notes/a.md", 0.75), _hit("/notes/b.md", 0.25)]
        searcher = _FakeSearcher(expected)
        params = {
            "op": "search",
            "query": "alpha",
            "k": 2,
            "mode": "hybrid",
            "per_file": 1,
            "rerank": False,
            "path_prefix": "/notes",
        }

        with _running_server(searcher, self.tmp) as (sock_path, *_):
            daemon_hits = client.try_daemon_search(params, sock_path=sock_path)

        self.assertEqual(daemon_hits, expected)
        self.assertEqual(
            searcher.calls,
            [
                {
                    "query": "alpha",
                    "k": 2,
                    "mode": "hybrid",
                    "per_file": 1,
                    "rerank": False,
                    "path_prefix": "/notes",
                }
            ],
        )

    def test_ping(self) -> None:
        with _running_server(_FakeSearcher(), self.tmp) as (sock_path, *_):
            response = _request(sock_path, {"op": "ping"})

        self.assertEqual(response, {"ok": True})

    def test_pipelined_newline_delimited_requests_get_two_responses(self) -> None:
        with (
            _running_server(_FakeSearcher(), self.tmp) as (sock_path, *_),
            _connection(sock_path) as sock,
        ):
            sock.sendall(b'{"op":"ping"}\n{"op":"ping"}\n')
            self.assertEqual(_read_json(sock), {"ok": True})
            self.assertEqual(_read_json(sock), {"ok": True})

    def test_status_returns_stats(self) -> None:
        searcher = _FakeSearcher()
        with _running_server(searcher, self.tmp) as (sock_path, *_):
            response = _request(sock_path, {"op": "status"})

        self.assertTrue(response["ok"])
        self.assertEqual(response["stats"], {"documents": 2, "chunks": 5})
        self.assertEqual(searcher.store.stats_calls, 1)

    def test_malformed_json_and_unknown_op_keep_connection_usable(self) -> None:
        with (
            _running_server(_FakeSearcher(), self.tmp) as (sock_path, *_),
            _connection(sock_path) as sock,
        ):
            _send(sock, b"this is not json\n")
            broken = _read_json(sock)
            self.assertFalse(broken["ok"])
            self.assertIn("error", broken)

            _send(sock, {"op": "bogus"})
            unknown = _read_json(sock)
            self.assertFalse(unknown["ok"])
            self.assertIn("bogus", unknown["error"])

            _send(sock, {"op": "ping"})
            self.assertEqual(_read_json(sock), {"ok": True})

    def test_blank_line_returns_error_and_connection_stays_usable(self) -> None:
        with (
            _running_server(_FakeSearcher(), self.tmp) as (sock_path, *_),
            _connection(sock_path) as sock,
        ):
            _send(sock, b"\n")
            self.assertFalse(_read_json(sock)["ok"])
            _send(sock, {"op": "ping"})
            self.assertEqual(_read_json(sock), {"ok": True})

    def test_config_mismatch_is_reported_and_connection_survives(self) -> None:
        searcher = _FakeSearcher(
            metadata={
                "db_path": str((self.tmp / "daemon.db").resolve()),
                "embed_model": "daemon/model",
            }
        )
        with (
            _running_server(searcher, self.tmp) as (sock_path, *_),
            _connection(sock_path) as sock,
        ):
            _send(
                sock,
                {
                    "op": "search",
                    "query": "alpha",
                    "k": 1,
                    "mode": "keyword",
                    "per_file": 1,
                    "rerank": False,
                    "path_prefix": None,
                    "expect": {
                        "db_path": str((self.tmp / "local.db").resolve()),
                        "embed_model": "local/model",
                    },
                },
            )
            response = _read_json(sock)
            self.assertFalse(response["ok"])
            self.assertTrue(response["error"].startswith("config mismatch"))

            _send(sock, {"op": "ping"})
            self.assertEqual(_read_json(sock), {"ok": True})

    def test_matching_expectation_allows_search(self) -> None:
        db_path = (self.tmp / "index.db").resolve()
        searcher = _FakeSearcher(
            metadata={"db_path": str(db_path), "embed_model": "test/model"}
        )
        params = {
            "op": "search",
            "query": "alpha",
            "k": 1,
            "mode": "keyword",
            "per_file": 1,
            "rerank": False,
            "path_prefix": None,
        }
        with _running_server(searcher, self.tmp) as (sock_path, *_):
            hits = client.try_daemon_search(
                params,
                sock_path=sock_path,
                expected={"db_path": str(db_path), "embed_model": "test/model"},
            )

        self.assertEqual(hits, searcher.hits)

    def test_client_raises_mismatch_for_expected_config(self) -> None:
        searcher = _FakeSearcher(
            metadata={
                "db_path": str((self.tmp / "daemon.db").resolve()),
                "embed_model": "daemon/model",
            }
        )
        params = {
            "op": "search",
            "query": "alpha",
            "k": 1,
            "mode": "keyword",
            "per_file": 1,
            "rerank": False,
            "path_prefix": None,
        }
        with (
            _running_server(searcher, self.tmp) as (sock_path, *_),
            self.assertRaises(client.DaemonMismatchError),
        ):
            client.try_daemon_search(
                params,
                sock_path=sock_path,
                expected={
                    "db_path": str((self.tmp / "local.db").resolve()),
                    "embed_model": "local/model",
                },
            )

    def test_reranker_is_built_lazily_only_once(self) -> None:
        searcher = _FakeSearcher()
        built: list[_FakeReranker] = []

        def factory() -> _FakeReranker:
            reranker = _FakeReranker()
            built.append(reranker)
            return reranker

        params = {
            "op": "search",
            "query": "alpha",
            "k": 1,
            "mode": "hybrid",
            "per_file": 1,
            "rerank": True,
            "path_prefix": None,
        }
        with _running_server(
            searcher, self.tmp, reranker_factory=factory
        ) as (sock_path, *_):
            first = client.try_daemon_search(params, sock_path=sock_path)
            second = client.try_daemon_search(params, sock_path=sock_path)

        self.assertEqual(len(built), 1)
        self.assertIs(searcher.reranker, built[0])
        self.assertTrue(first is not None and second is not None)
        self.assertEqual(searcher.reranker_seen, [True, True])

    def test_rerank_without_reranker_returns_server_error(self) -> None:
        searcher = _FakeSearcher()
        params = {
            "op": "search",
            "query": "alpha",
            "k": 1,
            "mode": "hybrid",
            "per_file": 1,
            "rerank": True,
            "path_prefix": None,
        }
        with (
            _running_server(searcher, self.tmp) as (sock_path, *_),
            self.assertRaises(client.DaemonError) as raised,
        ):
            client.try_daemon_search(params, sock_path=sock_path)

        self.assertIn("rerank", str(raised.exception).lower())

    def test_search_failure_is_reported_without_killing_the_server(self) -> None:
        searcher = _FakeSearcher(error=RuntimeError("boom"))
        params = {
            "op": "search",
            "query": "alpha",
            "k": 1,
            "mode": "keyword",
            "per_file": 1,
            "rerank": False,
            "path_prefix": None,
        }
        with _running_server(searcher, self.tmp) as (sock_path, *_):
            try:
                client.try_daemon_search(params, sock_path=sock_path)
            except client.DaemonError as exc:
                message = str(exc)
            else:
                self.fail("search failure did not raise DaemonError")
            self.assertIn("boom", message)
            self.assertEqual(_request(sock_path, {"op": "ping"}), {"ok": True})

    def test_oversized_request_is_rejected_and_server_stays_up(self) -> None:
        with (
            _running_server(_FakeSearcher(), self.tmp) as (sock_path, *_),
            _connection(sock_path) as sock,
        ):
            try:
                _send(sock, b"x" * (daemon.MAX_REQUEST_BYTES + 1))
            except OSError:
                pass
            self.assertEqual(_request(sock_path, {"op": "ping"}), {"ok": True})


class PeerIdentityTests(unittest.TestCase):
    def test_same_uid_socketpair_credentials_are_accepted(self) -> None:
        left, right = socket.socketpair()
        self.addCleanup(left.close)
        self.addCleanup(right.close)

        credentials = client._peer_credentials(left)

        self.assertEqual(credentials[1:], (os.getuid(), os.getgid()))
        client._verify_peer(left)

    def test_mismatched_peer_credentials_reject_before_any_query_bytes(self) -> None:
        left, right = socket.socketpair()
        self.addCleanup(left.close)
        self.addCleanup(right.close)
        with (
            mock.patch("seekr.client._connect", return_value=left),
            mock.patch(
                "seekr.client._peer_credentials",
                return_value=(123, os.getuid() + 1, os.getgid()),
            ),
            self.assertRaises(client.DaemonError) as raised,
        ):
            client.try_daemon_search(
                {"op": "search", "query": "secret"}, sock_path=Path("/unused")
            )

        self.assertIn("credential", str(raised.exception).lower())
        self.assertEqual(right.recv(1), b"")


class SocketLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_socket_permissions_are_0600(self) -> None:
        with _running_server(_FakeSearcher(), self.tmp) as (sock_path, *_):
            mode = stat.S_IMODE(os.lstat(sock_path).st_mode)

        self.assertEqual(mode, 0o600)

    def test_socket_is_removed_on_shutdown(self) -> None:
        with _running_server(_FakeSearcher(), self.tmp) as (sock_path, stop, thread, _):
            pass

        self.assertFalse(sock_path.exists())
        self.assertTrue(stop.is_set())
        self.assertFalse(thread.is_alive())

    def test_regular_file_at_socket_path_is_never_deleted(self) -> None:
        sock_path = self.tmp / "seekr.sock"
        sock_path.write_text("do not delete", encoding="utf-8")

        with self.assertRaises(daemon.DaemonError):
            daemon.serve(_FakeSearcher(), sock_path, threading.Event())

        self.assertEqual(sock_path.read_text(encoding="utf-8"), "do not delete")

    def test_active_socket_is_not_replaced(self) -> None:
        with _running_server(_FakeSearcher(), self.tmp) as (sock_path, *_):
            self.assertRaises(
                daemon.DaemonError,
                daemon.serve,
                _FakeSearcher(),
                sock_path,
                threading.Event(),
            )
            self.assertEqual(_request(sock_path, {"op": "ping"}), {"ok": True})

    def test_stale_socket_is_replaced(self) -> None:
        sock_path = self.tmp / "seekr.sock"
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            stale.bind(str(sock_path))
            stale.listen(1)
        finally:
            stale.close()
        self.assertTrue(sock_path.exists())

        with _running_server(_FakeSearcher(), self.tmp) as (replacement, *_):
            self.assertEqual(_request(replacement, {"op": "ping"}), {"ok": True})

    def test_client_returns_none_for_missing_or_stale_socket(self) -> None:
        self.assertIsNone(
            client.try_daemon_search({}, sock_path=self.tmp / "missing.sock")
        )

        regular = self.tmp / "stale.sock"
        regular.write_text("stale", encoding="utf-8")
        self.assertIsNone(client.try_daemon_search({}, sock_path=regular))
        self.assertTrue(regular.exists())

    def test_stop_terminates_with_an_idle_connection(self) -> None:
        with _running_server(_FakeSearcher(), self.tmp) as (sock_path, stop, thread, _):
            idle = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            idle.connect(str(sock_path))
            try:
                stop.set()
                thread.join(timeout=5.0)
            finally:
                idle.close()

        self.assertFalse(thread.is_alive())
        self.assertFalse(sock_path.exists())

    def test_client_does_not_fall_back_on_timeout(self) -> None:
        sock_path = self.tmp / "slow.sock"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(sock_path))
        listener.listen(1)
        release = threading.Event()

        def hold() -> None:
            conn, _ = listener.accept()
            release.wait(5.0)
            conn.close()

        thread = threading.Thread(target=hold, daemon=True)
        thread.start()
        try:
            with self.assertRaises(client.DaemonError):
                client.try_daemon_search({}, sock_path=sock_path, timeout=0.1)
        finally:
            release.set()
            thread.join(timeout=5.0)
            listener.close()

    def test_socket_path_honours_xdg_runtime_dir(self) -> None:
        with mock.patch.dict(
            os.environ, {"XDG_RUNTIME_DIR": str(self.tmp)}, clear=False
        ):
            self.assertEqual(daemon.socket_path(), self.tmp / "seekr.sock")

    def test_socket_path_falls_back_to_tmp(self) -> None:
        with mock.patch.dict(os.environ, clear=True):
            self.assertEqual(
                daemon.socket_path(), Path("/tmp") / f"seekr-{os.getuid()}.sock"
            )


class CliFastPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.cfg = Config(
            db_path=self.tmp / "index.db",
            embed_model="test/embed",
            rerank_model="test/rerank",
        )

    def test_daemon_hits_bypass_all_local_model_and_database_setup(self) -> None:
        expected = [_hit()]
        params = {
            "op": "search",
            "query": "alpha",
            "k": 3,
            "mode": "hybrid",
            "per_file": 1,
            "rerank": True,
            "path_prefix": "/notes",
        }
        with (
            mock.patch(
                "seekr.cli_search.try_daemon_search", return_value=expected
            ) as try_daemon,
            mock.patch("seekr.cli_search.build_embedder") as build_embedder,
            mock.patch("seekr.cli_search.Store") as store,
            mock.patch("seekr.cli_search.lifecycle_lock") as lock,
        ):
            hits = cli_search.get_hits(
                "alpha",
                k=3,
                mode="hybrid",
                per_file=1,
                rerank=True,
                path_prefix="/notes",
                cfg=self.cfg,
            )

        self.assertEqual(hits, expected)
        build_embedder.assert_not_called()
        store.assert_not_called()
        lock.assert_not_called()
        self.assertEqual(try_daemon.call_args.args[0], params)
        expected_config = try_daemon.call_args.kwargs["expected"]
        self.assertEqual(
            expected_config["db_path"], str(self.cfg.db_path.resolve())
        )
        self.assertEqual(expected_config["embed_model"], "test/embed")
        self.assertEqual(expected_config["rerank_model"], "test/rerank")

    def test_cli_fast_path_roundtrips_with_a_running_daemon(self) -> None:
        searcher = _FakeSearcher(
            metadata={
                "db_path": str(self.cfg.db_path.resolve()),
                "embed_model": self.cfg.embed_model,
            }
        )
        with (
            _running_server(searcher, self.tmp) as (sock_path, *_),
            mock.patch("seekr.client.socket_path", return_value=sock_path),
            mock.patch("seekr.cli_search.build_embedder") as build_embedder,
            mock.patch("seekr.cli_search.lifecycle_lock") as lock,
        ):
            hits = cli_search.get_hits("alpha", mode="keyword", cfg=self.cfg)

        self.assertEqual(hits, searcher.hits)
        build_embedder.assert_not_called()
        lock.assert_not_called()
        self.assertEqual(searcher.calls[0]["query"], "alpha")

    def test_mismatched_daemon_falls_back_to_local_search(self) -> None:
        expected = [_hit("/notes/local.md")]
        searcher = mock.Mock()
        searcher.store = mock.Mock()
        searcher.get_hits.return_value = expected
        with (
            mock.patch(
                "seekr.cli_search.try_daemon_search",
                side_effect=client.DaemonMismatchError("config mismatch"),
            ),
            mock.patch("seekr.cli_search.lifecycle_lock"),
            mock.patch(
                "seekr.cli_search._build_searcher", return_value=searcher
            ) as build_searcher,
        ):
            hits = cli_search.get_hits("alpha", mode="keyword", cfg=self.cfg)

        self.assertEqual(hits, expected)
        build_searcher.assert_called_once_with(self.cfg, "keyword", False)
        searcher.store.close.assert_called_once_with()

    def test_daemon_server_error_propagates_without_local_fallback(self) -> None:
        with (
            mock.patch(
                "seekr.cli_search.try_daemon_search",
                side_effect=client.DaemonError("daemon timed out"),
            ),
            mock.patch("seekr.cli_search._build_searcher") as build_searcher,
            self.assertRaises(client.DaemonError) as raised,
        ):
            cli_search.get_hits("alpha", cfg=self.cfg)

        self.assertIn("timed out", str(raised.exception))
        build_searcher.assert_not_called()

    def test_empty_daemon_result_is_not_retried_locally(self) -> None:
        with (
            mock.patch("seekr.cli_search.try_daemon_search", return_value=[]),
            mock.patch("seekr.cli_search._build_searcher") as build_searcher,
        ):
            hits = cli_search.get_hits("alpha", cfg=self.cfg)

        self.assertEqual(hits, [])
        build_searcher.assert_not_called()

    def test_try_daemon_hits_roundtrips_without_opening_local_backends(self) -> None:
        searcher = _FakeSearcher(
            metadata={
                "db_path": str(self.cfg.db_path.resolve()),
                "embed_model": self.cfg.embed_model,
            }
        )

        def daemon_call(params, expected=None):
            return client.try_daemon_search(
                params, sock_path=sock_path, expected=expected
            )

        with (
            _running_server(searcher, self.tmp) as (sock_path, *_),
            mock.patch(
                "seekr.cli_search.try_daemon_search", side_effect=daemon_call
            ),
            mock.patch("seekr.cli_search.build_embedder") as build_embedder,
            mock.patch("seekr.cli_search.Store") as store,
            mock.patch("seekr.cli_search.lifecycle_lock") as lock,
        ):
            hits = cli_search.try_daemon_hits("alpha", mode="keyword", cfg=self.cfg)

        self.assertEqual(hits, searcher.hits)
        build_embedder.assert_not_called()
        store.assert_not_called()
        lock.assert_not_called()
        self.assertEqual(searcher.calls[0]["query"], "alpha")

    def test_try_daemon_hits_returns_none_without_a_listener(self) -> None:
        missing = self.tmp / "missing.sock"

        def daemon_call(params, expected=None):
            return client.try_daemon_search(
                params, sock_path=missing, expected=expected
            )

        with (
            mock.patch("seekr.cli_search.try_daemon_search", side_effect=daemon_call),
            mock.patch("seekr.cli_search._build_searcher") as build_searcher,
        ):
            hits = cli_search.try_daemon_hits("alpha", cfg=self.cfg)

        self.assertIsNone(hits)
        build_searcher.assert_not_called()


class DaemonRunnerTests(unittest.TestCase):
    def test_run_daemon_holds_the_shared_lock_for_store_and_watcher(self) -> None:
        events: list[object] = []
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(roots=(Path(tmp),), db_path=Path(tmp) / "index.db")
            embedder = mock.Mock()
            embedder.dim = 8
            store = mock.Mock()
            store.close.side_effect = lambda: events.append("store.close")
            searcher = mock.Mock()
            indexer = mock.Mock()
            indexer.sync.side_effect = lambda roots: events.append("sync")
            watcher = mock.Mock()
            watcher.start.side_effect = lambda **kwargs: events.append(
                ("watcher.start", kwargs)
            )
            watcher.resume.side_effect = lambda: events.append("watcher.resume")
            watcher.stop.side_effect = lambda: events.append("watcher.stop")
            stop_event = threading.Event()

            @contextlib.contextmanager
            def lock(path, *, shared=True):
                events.append(("lock", Path(path), shared))
                try:
                    yield object()
                finally:
                    events.append("unlock")

            def serve(searcher_arg, sock_path, stop_event_arg, **kwargs):
                events.append("serve")

            with (
                mock.patch("seekr.cli_daemon.Store", return_value=store),
                mock.patch("seekr.cli_daemon.Searcher", return_value=searcher),
                mock.patch("seekr.cli_daemon.Indexer", return_value=indexer),
                mock.patch("seekr.cli_daemon.Watcher", return_value=watcher),
                mock.patch("seekr.cli_daemon.serve", side_effect=serve),
                mock.patch("seekr.cli_daemon.lifecycle_lock", side_effect=lock),
                mock.patch(
                    "seekr.cli_daemon.socket_path",
                    return_value=Path(tmp) / "seekr.sock",
                ),
            ):
                cli_daemon.run_daemon(
                    cfg=cfg,
                    embedder=embedder,
                    stop_event=stop_event,
                    handle_signals=False,
                )

        self.assertEqual(events[0], ("lock", cfg.db_path, True))
        self.assertIn(("watcher.start", {"paused": True}), events)
        self.assertIn("sync", events)
        self.assertLess(
            events.index(("watcher.start", {"paused": True})),
            events.index("sync"),
        )
        self.assertLess(events.index("sync"), events.index("watcher.resume"))
        self.assertLess(events.index("watcher.resume"), events.index("serve"))
        self.assertLess(events.index("serve"), events.index("watcher.stop"))
        self.assertLess(events.index("watcher.stop"), events.index("store.close"))
        self.assertEqual(events[-1], "unlock")
        indexer.sync.assert_called_once_with(list(cfg.roots))
        self.assertEqual(searcher.daemon_metadata["embed_model"], cfg.embed_model)
        self.assertEqual(searcher.daemon_metadata["dim"], 8)

    def test_run_daemon_closes_store_and_releases_lock_when_startup_fails(self) -> None:
        events: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(roots=(Path(tmp),), db_path=Path(tmp) / "index.db")
            embedder = mock.Mock()
            embedder.dim = 8
            store = mock.Mock()
            store.close.side_effect = lambda: events.append("close")
            watcher = mock.Mock()
            watcher.start.side_effect = RuntimeError("no observer")

            @contextlib.contextmanager
            def lock(path, *, shared=True):
                events.append("lock")
                try:
                    yield object()
                finally:
                    events.append("unlock")

            with (
                mock.patch("seekr.cli_daemon.Store", return_value=store),
                mock.patch("seekr.cli_daemon.Searcher"),
                mock.patch("seekr.cli_daemon.Indexer"),
                mock.patch("seekr.cli_daemon.Watcher", return_value=watcher),
                mock.patch("seekr.cli_daemon.lifecycle_lock", side_effect=lock),
                self.assertRaises(RuntimeError),
            ):
                cli_daemon.run_daemon(
                    cfg=cfg,
                    embedder=embedder,
                    handle_signals=False,
                )

        self.assertEqual(events, ["lock", "close", "unlock"])

    def test_run_daemon_with_existing_db_defers_embedder_load_until_serve(self) -> None:
        import sqlite3

        events: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "index.db"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                conn.executemany(
                    "INSERT INTO meta(key, value) VALUES (?, ?)",
                    (("embed_model", "BAAI/bge-small-en-v1.5"), ("dim", "12")),
                )
                conn.commit()
            finally:
                conn.close()

            cfg = Config(roots=(Path(tmp),), db_path=db_path)
            dim_ready = threading.Event()

            class LazyEmbedder:
                model_name = cfg.embed_model

                @property
                def dim(self) -> int:
                    events.append("embedder.dim")
                    dim_ready.set()
                    return 12

            store = mock.Mock()
            searcher = mock.Mock()
            indexer = mock.Mock()
            indexer.sync.side_effect = lambda roots: events.append("sync")
            watcher = mock.Mock()
            watcher.resume.side_effect = lambda: events.append("watcher.resume")
            stop_event = threading.Event()

            @contextlib.contextmanager
            def lock(path, *, shared=True):
                yield object()

            def serve(searcher_arg, sock_path, stop_event_arg, **kwargs):
                events.append("serve")
                dim_ready.wait(timeout=2.0)

            def open_store(*args, **kwargs):
                events.append("store")
                return store

            with (
                mock.patch("seekr.cli_daemon.build_embedder", return_value=LazyEmbedder()),
                mock.patch("seekr.cli_daemon.Store", side_effect=open_store) as store_cls,
                mock.patch("seekr.cli_daemon.Searcher", return_value=searcher),
                mock.patch("seekr.cli_daemon.Indexer", return_value=indexer),
                mock.patch("seekr.cli_daemon.Watcher", return_value=watcher),
                mock.patch("seekr.cli_daemon.serve", side_effect=serve),
                mock.patch("seekr.cli_daemon.lifecycle_lock", side_effect=lock),
                mock.patch(
                    "seekr.cli_daemon.socket_path",
                    return_value=Path(tmp) / "seekr.sock",
                ),
            ):
                cli_daemon.run_daemon(
                    cfg=cfg,
                    stop_event=stop_event,
                    handle_signals=False,
                )

        self.assertIn("serve", events)
        self.assertIn("embedder.dim", events)
        self.assertLess(events.index("store"), events.index("embedder.dim"))
        self.assertLess(events.index("sync"), events.index("embedder.dim"))
        self.assertLess(events.index("watcher.resume"), events.index("embedder.dim"))
        store_cls.assert_called_once_with(
            db_path, cfg.embed_model, 12, vector_quant=cfg.vector_quant
        )
        self.assertEqual(searcher.daemon_metadata["dim"], 12)
        self.assertEqual(searcher.daemon_metadata["vector_quant"], cfg.vector_quant)


class ServiceUnitTests(unittest.TestCase):
    def test_service_unit_runs_seekr_daemon_as_a_user_unit(self) -> None:
        service = (
            Path(__file__).resolve().parents[1] / "contrib" / "seekr.service"
        ).read_text(encoding="utf-8")

        self.assertIn("ExecStart=", service)
        self.assertIn("seekr daemon", service)
        self.assertIn("WantedBy=default.target", service)


@unittest.skipUnless(_HAS_TYPER, "typer is not installed")
class DaemonCommandRegistrationTests(unittest.TestCase):
    def test_register_adds_a_daemon_command(self) -> None:
        import typer
        from typer.testing import CliRunner

        app = typer.Typer()
        cli_daemon.register(app)
        result = CliRunner().invoke(app, ["daemon", "--help"])
        output = strip_ansi(result.output)

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--socket", output)
        self.assertIn("--debounce", output)


@unittest.skipUnless(
    _NEEDS_REAL_STORE, "numpy, sqlite-vec and FTS5 are required"
)
class RealStoreDaemonTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "notes"
        self.root.mkdir()
        (self.root / "neural.md").write_text(
            "zebra backpropagation gradients", encoding="utf-8"
        )
        self.cfg = Config(
            roots=(self.root,),
            db_path=self.tmp / "index.db",
            embed_model="fake",
        )
        self.embedder = FakeEmbedder(dim=8)
        self.store = Store(self.cfg.db_path, "fake", 8)
        self.addCleanup(self.store.close)
        self.indexer = Indexer(self.store, self.embedder, self.cfg)
        self.indexer.sync([self.root])
        self.searcher = Searcher(self.store, self.embedder)
        self.sock_path = self.tmp / "seekr.sock"

    def test_search_roundtrip_matches_inprocess_results(self) -> None:
        params = {
            "op": "search",
            "query": "zebra",
            "k": 5,
            "mode": "hybrid",
            "per_file": 2,
            "rerank": False,
            "path_prefix": None,
        }
        inprocess = self.searcher.search(
            "zebra", k=5, mode="hybrid", per_file=2
        )

        stop_event = threading.Event()
        errors: list[BaseException] = []

        def run() -> None:
            try:
                daemon.serve(self.searcher, self.sock_path, stop_event)
            except Exception as exc:  # noqa: BLE001 - asserted below
                errors.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            self.assertTrue(_wait_for(lambda: self.sock_path.exists()))
            daemon_hits = client.try_daemon_search(params, sock_path=self.sock_path)
        finally:
            stop_event.set()
            thread.join(timeout=5.0)

        self.assertEqual(errors, [])
        self.assertEqual(daemon_hits, inprocess)
        self.assertTrue(daemon_hits)
        self.assertEqual(asdict(daemon_hits[0]), asdict(inprocess[0]))

    def test_daemon_search_uses_current_database_metadata(self) -> None:
        params = {
            "op": "search",
            "query": "zebra",
            "k": 5,
            "mode": "keyword",
            "per_file": 2,
            "rerank": False,
            "path_prefix": None,
            "expect": {
                "db_path": str(Path(self.cfg.db_path).resolve()),
                "embed_model": "fake",
            },
        }
        stop_event = threading.Event()
        thread = threading.Thread(
            target=daemon.serve,
            args=(self.searcher, self.sock_path, stop_event),
            daemon=True,
        )
        thread.start()
        try:
            self.assertTrue(_wait_for(lambda: self.sock_path.exists()))
            hits = client.try_daemon_search(params, sock_path=self.sock_path)
        finally:
            stop_event.set()
            thread.join(timeout=5.0)

        self.assertTrue(hits is not None)
        self.assertEqual(hits[0].path, str((self.root / "neural.md").resolve()))


class ConfigFastPathUnitTests(unittest.TestCase):
    def test_load_config_is_the_only_configuration_work(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch("seekr.cli_search.load_config") as load,
            mock.patch("seekr.cli_search.try_daemon_search", return_value=[]),
        ):
            cfg = Config(db_path=Path(tmp) / "index.db")
            load.return_value = cfg
            cli_search.get_hits("alpha")

        load.assert_called_once_with()


class SocketPathLimitTests(unittest.TestCase):
    def _long_path(self, tmp: str | Path) -> Path:
        deep = Path(tmp) / ("d" * 120)
        deep.mkdir()
        return deep / "seekr.sock"

    def test_problem_reported_for_long_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            msg = daemon.socket_path_problem(self._long_path(tmp))
        self.assertIsNotNone(msg)
        assert msg is not None
        self.assertIn("socket path too long", msg)

    def test_short_path_has_no_problem(self) -> None:
        self.assertIsNone(daemon.socket_path_problem(Path("/tmp/seekr.sock")))

    def test_exact_length_boundary_matches_af_unix_limit(self) -> None:
        at_limit = Path("/" + "a" * (daemon.AF_UNIX_PATH_MAX - 1))
        over_limit = Path("/" + "a" * daemon.AF_UNIX_PATH_MAX)
        self.assertEqual(len(os.fsencode(str(at_limit))), daemon.AF_UNIX_PATH_MAX)
        self.assertIsNone(daemon.socket_path_problem(at_limit))
        self.assertIsNone(daemon.socket_path_problem(str(at_limit)))
        problem = daemon.socket_path_problem(over_limit)
        self.assertEqual(
            problem,
            f"socket path too long ({daemon.AF_UNIX_PATH_MAX + 1} bytes, "
            f"limit {daemon.AF_UNIX_PATH_MAX})",
        )
        # Byte length, not character count, determines AF_UNIX sun_path overflow.
        multibyte = Path("/" + "é" * 54)
        encoded_len = len(os.fsencode(str(multibyte)))
        if encoded_len > daemon.AF_UNIX_PATH_MAX:
            self.assertEqual(
                daemon.socket_path_problem(multibyte),
                f"socket path too long ({encoded_len} bytes, limit {daemon.AF_UNIX_PATH_MAX})",
            )

    def test_socket_is_active_false_for_long_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(daemon._socket_is_active(self._long_path(tmp)))

    def test_try_daemon_search_returns_none_for_long_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                client.try_daemon_search({"query": "x"}, sock_path=self._long_path(tmp))
            )

    def test_serve_refuses_long_path(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaises(daemon.DaemonError) as raised,
        ):
            daemon._bind_listener(self._long_path(tmp))
        message = str(raised.exception)
        self.assertIn("socket path too long", message)
        self.assertIn("XDG_RUNTIME_DIR", message)

    def test_search_falls_back_to_local_when_xdg_runtime_dir_is_long(self) -> None:
        expected = [_hit("/notes/local.md")]
        searcher = mock.Mock()
        searcher.store = mock.Mock()
        searcher.get_hits.return_value = expected
        with tempfile.TemporaryDirectory() as tmp:
            long_runtime_dir = Path(tmp) / ("r" * 120)
            long_runtime_dir.mkdir()
            cfg = Config(db_path=Path(tmp) / "index.db", embed_model="fake")
            with (
                mock.patch.dict(
                    os.environ, {"XDG_RUNTIME_DIR": str(long_runtime_dir)}, clear=False
                ),
                mock.patch("seekr.cli_search.lifecycle_lock"),
                mock.patch(
                    "seekr.cli_search._build_searcher", return_value=searcher
                ) as build_searcher,
            ):
                hits = cli_search.get_hits("alpha", mode="keyword", cfg=cfg)
        self.assertEqual(hits, expected)
        build_searcher.assert_called_once_with(cfg, "keyword", False)
        searcher.store.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
