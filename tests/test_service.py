"""Tests for the unified loopback service.

Everything runs offline: the embedder is :class:`FakeEmbedder`, observers are
in-memory fakes, and config, data and state live in temporary directories.
Only loopback HTTP is used.  Every test stops its service.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import ClassVar

import pytest

from seekr import __version__, paths
from seekr.embed import FakeEmbedder
from seekr.service import SeekrService, ServiceAlreadyRunning
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
