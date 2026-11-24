"""Tests for the unified loopback service.

Everything runs offline: the embedder is :class:`FakeEmbedder`, observers are
in-memory fakes, and config, data and state live in temporary directories.
Only loopback HTTP is used.  Every test stops its service.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import ClassVar

import pytest

from seekr import __version__, paths
from seekr.embed import FakeEmbedder
from seekr.service import SeekrService, ServiceAlreadyRunning, ServiceStopping
from seekr.service_discovery import read_service_info


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


class FakeObserver:
    created: ClassVar[list[FakeObserver]] = []

    def __init__(self) -> None:
        self.schedules: list[tuple[object, str, bool]] = []
        self.started = False
        self.stopped = False
        FakeObserver.created.append(self)

    def schedule(self, handler: object, path: str, recursive: bool) -> None:
        self.schedules.append((handler, path, recursive))

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def join(self, timeout: float | None = None) -> None:
        return None


class _Event:
    def __init__(self, event_type: str, src_path: str) -> None:
        self.event_type = event_type
        self.src_path = src_path
        self.dest_path = ""
        self.is_directory = False


def _request(port: int, method: str, path: str, *, token: str, body: object = None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, method=method
    )
    req.add_header("X-Seekr-Token", token)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as exc:
        with exc:
            return exc.code, json.loads(exc.read() or b"null")


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"):
        target = tmp_path / name.lower()
        target.mkdir(mode=0o700)
        monkeypatch.setenv(name, str(target))
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text("alpha notes about xylophones\n", encoding="utf-8")
    config = tmp_path / "config.toml"
    config.write_text(
        'embed_model = "fake"\n'
        f"db_path = {json.dumps(str(tmp_path / 'index.db'))}\n"
        'vector_quant = "int8"\n'
        f"roots = [{json.dumps(str(root))}]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SEEKR_CONFIG", str(config))
    FakeObserver.created = []
    return tmp_path


@pytest.fixture(autouse=True)
def no_leaked_threads():
    before = set(threading.enumerate())
    yield

    def leaked() -> list[threading.Thread]:
        return [
            thread
            for thread in threading.enumerate()
            if thread not in before
            and thread.is_alive()
            and thread.name != "seekr-embed-prewarm"
        ]

    assert _wait_for(lambda: not leaked(), timeout=2.0), leaked()


def _service(request: pytest.FixtureRequest, **kwargs) -> SeekrService:
    kwargs.setdefault("embedder_factory", lambda cfg: FakeEmbedder())
    kwargs.setdefault("observer_factory", FakeObserver)
    kwargs.setdefault("debounce_s", 0.05)
    kwargs.setdefault("port", 0)
    svc = SeekrService(**kwargs)
    request.addfinalizer(svc.stop)
    return svc


def _wait_idle(svc: SeekrService) -> None:
    assert _wait_for(lambda: svc.backend.status().activity["lane"] == "idle")


def _documents(svc: SeekrService) -> int:
    return int(svc.backend.status().index["documents"] or 0)


def test_start_writes_discovery_and_answers_status(env, request) -> None:
    svc = _service(request)
    port = svc.start()

    info = read_service_info(paths.service_file())
    assert info is not None
    assert info.port == port
    assert info.pid == os.getpid()
    assert info.version == __version__
    status, body = _request(port, "GET", "/api/status", token=info.token)
    assert status == 200
    assert body["daemon"]["present"] is False
    assert body["service"] == {"running": True, "pid": os.getpid(), "version": __version__}

    svc.stop()
    assert read_service_info(paths.service_file()) is None
    svc.stop()  # idempotent


def test_initial_sync_indexes_configured_roots(env, request) -> None:
    svc = _service(request)
    svc.start()

    assert _wait_for(lambda: _documents(svc) == 1)


def test_handshake_line(env, request, capsys) -> None:
    svc = _service(request, handshake_stdout=True)
    port = svc.start()

    first = capsys.readouterr().out.splitlines()[0]
    payload = json.loads(first)
    info = read_service_info(paths.service_file())
    assert info is not None
    assert list(payload) == ["event", "port", "pid", "version", "token"]
    assert payload == {
        "event": "ready",
        "port": port,
        "pid": os.getpid(),
        "version": __version__,
        "token": info.token,
    }


def test_handshake_reports_existing_service(env, request, capsys) -> None:
    first = _service(request)
    port = first.start()
    info = read_service_info(paths.service_file())
    assert info is not None

    second = _service(request, handshake_stdout=True)
    with pytest.raises(ServiceAlreadyRunning) as caught:
        second.start()

    assert caught.value.port == port
    payload = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert payload == {
        "event": "existing",
        "port": port,
        "pid": info.pid,
        "version": info.version,
        "token": info.token,
    }


def test_second_instance_refused(env, request) -> None:
    first = _service(request)
    port = first.start()
    before = read_service_info(paths.service_file())

    with pytest.raises(ServiceAlreadyRunning) as caught:
        _service(request).start()

    assert caught.value.port == port
    assert read_service_info(paths.service_file()) == before
    assert before is not None and before.pid == os.getpid()


def test_stale_discovery_file_is_replaced(env, request) -> None:
    from seekr.service_discovery import ServiceInfo, write_service_info

    write_service_info(
        ServiceInfo(pid=os.getpid(), port=1, token="old", version="0", started_at="x"),
        paths.service_file(),
    )
    svc = _service(request)
    port = svc.start()

    info = read_service_info(paths.service_file())
    assert info is not None and info.port == port and info.token != "old"


def test_shutdown_endpoint_stops_service(env, request) -> None:
    svc = _service(request)
    port = svc.start()
    info = read_service_info(paths.service_file())
    assert info is not None

    status, _ = _request(port, "POST", "/api/shutdown", token=info.token, body={})

    assert status == 202
    assert _wait_for(lambda: read_service_info(paths.service_file()) is None)
    svc.serve_forever()  # returns at once once stopped


def test_stdin_eof_stops_service(env, request) -> None:
    read_fd, write_fd = os.pipe()
    stdin = os.fdopen(read_fd, "r", encoding="utf-8")
    request.addfinalizer(stdin.close)
    svc = _service(request, watch_stdin=True, stdin=stdin)
    svc.start()
    assert read_service_info(paths.service_file()) is not None

    os.close(write_fd)

    assert _wait_for(lambda: read_service_info(paths.service_file()) is None)


def test_added_root_is_watched_without_restart(env, request) -> None:
    svc = _service(request)
    port = svc.start()
    info = read_service_info(paths.service_file())
    assert info is not None
    _wait_idle(svc)
    extra = env / "extra"
    extra.mkdir()
    _, roots = _request(port, "GET", "/api/roots", token=info.token)

    status, _ = _request(
        port,
        "POST",
        "/api/roots",
        token=info.token,
        body={"path": str(extra), "revision": roots["revision"]},
    )

    assert status == 200
    watched = {entry[1] for entry in FakeObserver.created[-1].schedules}
    assert str(extra.resolve()) in watched
    assert FakeObserver.created[0].stopped
    note = extra / "b.md"
    note.write_text("beta notes about marimbas\n", encoding="utf-8")
    handler = FakeObserver.created[-1].schedules[0][0]
    handler.dispatch(_Event("created", str(note)))
    assert _wait_for(lambda: _documents(svc) == 2)


def test_watcher_indexing_runs_in_engine_lane(env, request) -> None:
    svc = _service(request)
    svc.start()
    _wait_idle(svc)
    assert _documents(svc) == 1
    release = threading.Event()
    holding = threading.Event()

    def hold() -> None:
        holding.set()
        release.wait(5)

    blocker = threading.Thread(
        target=svc.backend.run_in_lane, args=("index", hold), daemon=True
    )
    blocker.start()
    request.addfinalizer(lambda: (release.set(), blocker.join(5)))
    assert holding.wait(5)

    note = env / "docs" / "c.md"
    note.write_text("gamma notes about drums\n", encoding="utf-8")
    handler = FakeObserver.created[-1].schedules[0][0]
    handler.dispatch(_Event("created", str(note)))
    time.sleep(0.3)
    assert _documents(svc) == 1
    # Searches are never refused because the write lane is held.
    from seekr.web_contract import SearchRequest

    for mode in ("hybrid", "vector", "keyword"):
        assert svc.backend.search(SearchRequest("xylophones", mode=mode)).hits

    release.set()
    assert _wait_for(lambda: _documents(svc) == 2)


def test_model_dir_reaches_onnx_backend(env, tmp_path) -> None:
    from seekr.config import Config
    from seekr.embed import build_embedder

    embedder = build_embedder(
        Config(embed_backend="onnx"), model_dir=tmp_path / "bundled"
    )
    assert embedder._model_dir == tmp_path / "bundled"


def test_cli_status_and_stop(env, request) -> None:
    from typer.testing import CliRunner

    from seekr.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["service", "status"])
    assert result.exit_code == 0
    assert "not running" in result.output

    svc = _service(request)
    port = svc.start()
    result = runner.invoke(app, ["service", "status"])
    assert result.exit_code == 0
    assert f"port: {port}" in result.output
    assert f"pid: {os.getpid()}" in result.output
    assert f"version: {__version__}" in result.output
    assert "uptime:" in result.output

    result = runner.invoke(app, ["service", "stop"])
    assert result.exit_code == 0, result.output
    assert read_service_info(paths.service_file()) is None


def test_cli_stop_when_not_running(env) -> None:
    from typer.testing import CliRunner

    from seekr.cli import app

    result = CliRunner().invoke(app, ["service", "stop"])
    assert result.exit_code == 0
    assert "not running" in result.output


def test_cli_start_with_existing_service_and_handshake_exits_zero(
    env, request, monkeypatch
) -> None:
    from typer.testing import CliRunner

    from seekr import cli_service
    from seekr.cli import app

    first = _service(request)
    port = first.start()
    monkeypatch.setattr(
        cli_service,
        "_create_service",
        lambda **kwargs: _service(request, **kwargs),
    )

    runner = CliRunner()
    result = runner.invoke(app, ["service", "start", "--handshake-stdout"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output.splitlines()[-1])["port"] == port

    result = runner.invoke(app, ["service", "start"])
    assert result.exit_code == 1
    assert f"already running on port {port}" in result.output


def test_stdin_stream_default_is_not_read_without_watch_stdin(env, request) -> None:
    stdin = io.StringIO("")
    svc = _service(request, stdin=stdin)
    svc.start()
    time.sleep(0.1)
    assert read_service_info(paths.service_file()) is not None


def test_search_succeeds_while_watcher_action_holds_lane(env, request) -> None:
    from seekr.web_contract import SearchRequest

    svc = _service(request)
    svc.start()
    _wait_idle(svc)
    release = threading.Event()
    holding = threading.Event()
    original = svc._watch_indexer

    def slow_indexer():
        holding.set()
        release.wait(5)
        return original()

    svc._watch_indexer = slow_indexer  # type: ignore[method-assign]
    request.addfinalizer(release.set)
    note = env / "docs" / "d.md"
    note.write_text("delta notes\n", encoding="utf-8")
    FakeObserver.created[-1].schedules[0][0].dispatch(_Event("created", str(note)))
    assert holding.wait(5)
    assert svc.backend.status().activity["lane"] == "index"

    result = svc.backend.search(SearchRequest("xylophones", mode="hybrid"))

    assert result.hits
    release.set()


def test_second_start_in_another_process_is_refused(env, request) -> None:
    first = _service(request)
    port = first.start()
    before = read_service_info(paths.service_file())
    code = (
        # Windows ignores XDG_STATE_HOME: share the parent's state folder.
        "import os, pathlib\n"
        "from seekr import paths\n"
        "paths.state_dir = lambda *, platform=None: "
        "pathlib.Path(os.environ['XDG_STATE_HOME']) / 'seekr'\n"
        "from seekr.embed import FakeEmbedder\n"
        "from seekr.service import SeekrService, ServiceAlreadyRunning\n"
        "svc = SeekrService(port=0, handshake_stdout=True,"
        " embedder_factory=lambda cfg: FakeEmbedder())\n"
        "try:\n"
        "    svc.start()\n"
        "    print('started')\n"
        "except ServiceAlreadyRunning as exc:\n"
        "    print('refused', exc.port)\n"
        "finally:\n"
        "    svc.stop()\n"
    )
    child_env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    child_env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [src, child_env.get("PYTHONPATH")])
    )

    done = subprocess.run(
        [sys.executable, "-c", code],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert json.loads(lines[0])["event"] == "existing"
    assert lines[-1] == f"refused {port}"
    assert read_service_info(paths.service_file()) == before


def test_lock_holder_that_is_stopping_is_not_attached_to(env, request, capsys) -> None:
    first = _service(request)
    first.start()
    before = read_service_info(paths.service_file())
    first.backend.begin_shutdown()  # /api/status now refuses

    with pytest.raises(ServiceStopping):
        _service(request, handshake_stdout=True).start()

    assert capsys.readouterr().out == ""
    assert read_service_info(paths.service_file()) == before


def test_restart_after_stop_reacquires_lifetime_lock(env, request) -> None:
    first = _service(request)
    first.start()
    first.stop()

    second = _service(request)
    port = second.start()

    info = read_service_info(paths.service_file())
    assert info is not None and info.port == port


def test_watch_store_is_closed_inside_engine_lane(env, request) -> None:
    svc = _service(request)
    svc.start()
    _wait_idle(svc)
    lanes: list[object] = []
    original = svc._close_watch_store

    def record() -> None:
        lanes.append(svc.backend._activity()["lane"])
        original()

    svc._close_watch_store = record  # type: ignore[method-assign]
    svc.stop()

    assert lanes == ["index"]


def test_cli_status_ignores_stale_file_with_live_pid(env) -> None:
    from typer.testing import CliRunner

    from seekr.cli import app
    from seekr.service_discovery import ServiceInfo, write_service_info

    write_service_info(
        ServiceInfo(pid=os.getpid(), port=1, token="old", version="0", started_at="x"),
        paths.service_file(),
    )

    result = CliRunner().invoke(app, ["service", "status"])

    assert result.exit_code == 0
    assert "not running" in result.output


def test_lifetime_lock_is_released_when_teardown_fails(env, request) -> None:
    first = _service(request)
    first.start()
    assert first._server is not None

    real_close = first._server.server_close
    request.addfinalizer(real_close)

    def broken_close() -> None:
        raise OSError("close failed")

    first._server.server_close = broken_close  # type: ignore[method-assign]
    with pytest.raises(OSError):
        first.stop()

    second = _service(request)
    assert second.start() > 0


# -- pausing for a rebuild ----------------------------------------------------


def _open_watch_store(svc: SeekrService) -> None:
    svc.backend.run_in_lane("index", svc._watch_indexer)
    assert svc._watch_store is not None and svc._watch_lock is not None


def _exclusive_lock_available(db_path: Path) -> bool:
    from seekr.lifecycle import LifecycleLock, LifecycleLockError

    lock = LifecycleLock(db_path, shared=False, blocking=False)
    try:
        lock.acquire()
    except LifecycleLockError:
        return False
    lock.release()
    return True


def test_paused_closes_watch_store_releases_locks_and_reopens(env, request) -> None:
    svc = _service(request)
    svc.start()
    _wait_idle(svc)
    _open_watch_store(svc)
    db_path = env / "index.db"
    assert not _exclusive_lock_available(db_path)

    with svc.paused():
        assert svc._watch_store is None and svc._watch_lock is None
        assert _exclusive_lock_available(db_path)
        assert svc._watcher._paused is True
        # The engine lane stays held, so no index job can start meanwhile.
        assert svc.backend._activity()["lane"] == "index"

    assert svc._watch_store is not None
    assert svc._watcher._paused is False
    assert svc.backend._activity()["lane"] == "idle"


def test_paused_reopens_and_resumes_when_the_body_fails(env, request) -> None:
    svc = _service(request)
    svc.start()
    _wait_idle(svc)
    _open_watch_store(svc)

    with pytest.raises(RuntimeError, match="boom"), svc.paused():
        raise RuntimeError("boom")

    assert svc._watch_store is not None
    assert svc._watcher._paused is False
    assert svc.backend._activity()["lane"] == "idle"


def test_watcher_events_during_pause_apply_after_it(env, request) -> None:
    svc = _service(request)
    svc.start()
    _wait_idle(svc)
    note = env / "docs" / "c.md"
    with svc.paused():
        note.write_text("gamma notes about drums\n", encoding="utf-8")
        handler = FakeObserver.created[-1].schedules[0][0]
        handler.dispatch(_Event("created", str(note)))
        time.sleep(0.3)
        assert svc._watch_store is None
    assert _wait_for(lambda: _documents(svc) == 2)


def _reindex_via_api(svc: SeekrService, port: int, body: object = None) -> dict:
    info = read_service_info(paths.service_file())
    assert info is not None
    status, started = _request(port, "POST", "/api/reindex", token=info.token, body=body or {})
    assert status == 202, started
    job_id = started["job_id"]
    job: dict = {}

    def finished() -> bool:
        nonlocal job
        _status, job = _request(port, "GET", f"/api/jobs/{job_id}", token=info.token)
        return job["state"] != "running"

    assert _wait_for(finished, timeout=10)
    return job


def test_reindex_route_rebuilds_through_the_pause(env, request, monkeypatch) -> None:
    from seekr import lifecycle

    svc = _service(request)
    port = svc.start()
    _wait_idle(svc)
    _open_watch_store(svc)
    db_path = env / "index.db"
    real_rebuild = lifecycle.rebuild_index
    during: dict[str, object] = {}

    def checked_rebuild(cfg, **kwargs):
        # Observed while the rebuild runs, before any swap.
        during.update(
            watch_store=svc._watch_store,
            watch_lock=svc._watch_lock,
            exclusive_free=_exclusive_lock_available(db_path),
            watcher_paused=svc._watcher._paused,
            lane=svc.backend._activity()["lane"],
        )
        if svc._watch_store is not None or not during["exclusive_free"]:
            # Fail at once instead of waiting out the swap's upgrade timeout.
            raise RuntimeError("rebuild started while the service held the index")
        return real_rebuild(cfg, **kwargs)

    monkeypatch.setattr(lifecycle, "rebuild_index", checked_rebuild)
    (env / "docs" / "b.md").write_text("beta notes about drums\n", encoding="utf-8")

    started = time.monotonic()
    job = _reindex_via_api(svc, port)

    assert time.monotonic() - started < 5.0
    assert during == {
        "watch_store": None,
        "watch_lock": None,
        "exclusive_free": True,
        "watcher_paused": True,
        "lane": "index",
    }
    assert job["operation"] == "index"
    assert job["state"] == "succeeded", job
    assert job["report"]["added"] == 2
    assert _documents(svc) == 2
    assert svc._watch_store is not None  # reopened on the new database
    assert svc._watcher._paused is False
    _wait_idle(svc)


def test_reindex_failure_still_resumes_watcher(env, request) -> None:
    svc = _service(request)
    port = svc.start()
    _wait_idle(svc)
    _open_watch_store(svc)
    (env / "docs" / "a.md").unlink()

    job = _reindex_via_api(svc, port)

    assert job["state"] == "failed"
    assert "empty rebuild" in job["error"]
    assert _documents(svc) == 1  # live index left untouched
    assert svc._watch_store is not None
    assert svc._watcher._paused is False


def test_reindex_refuses_while_another_job_runs(env, request) -> None:
    svc = _service(request)
    port = svc.start()
    _wait_idle(svc)
    info = read_service_info(paths.service_file())
    assert info is not None
    release = threading.Event()
    holding = threading.Event()

    def hold() -> None:
        holding.set()
        release.wait(5)

    blocker = threading.Thread(
        target=svc.backend.run_in_lane, args=("index", hold), daemon=True
    )
    blocker.start()
    request.addfinalizer(lambda: (release.set(), blocker.join(5)))
    assert holding.wait(5)

    status, body = _request(port, "POST", "/api/reindex", token=info.token, body={})
    assert status == 409
    assert body["error"]["code"] == "busy"


@pytest.mark.skipif(sys.platform != "win32", reason="windows")
def test_reindex_with_open_reader_succeeds_via_service_pause(env, request) -> None:
    # On Windows an open SQLite handle makes os.replace fail; the watcher's
    # long-lived Store is exactly such a reader until the pause closes it.
    svc = _service(request)
    port = svc.start()
    _wait_idle(svc)
    _open_watch_store(svc)
    from seekr.web_contract import SearchRequest

    assert svc.backend.search(SearchRequest("xylophones", mode="keyword")).hits

    job = _reindex_via_api(svc, port)

    assert job["state"] == "succeeded", job
    assert _documents(svc) == 1
    assert svc.backend.search(SearchRequest("xylophones", mode="keyword")).hits
