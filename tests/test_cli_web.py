"""Tests for the ``seekr web`` launcher without requiring the web service."""

from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from _cli_output import strip_ansi
from seekr import cli_web

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
_HAS_TYPER = importlib.util.find_spec("typer") is not None

pytestmark = pytest.mark.skipif(not _HAS_TYPER, reason="typer is not installed")


class CleanupBackend:
    """Minimal backend used only to observe lifecycle cleanup."""

    def __init__(self) -> None:
        self.shutdown_calls = 0
        self.close_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class RecordingRun:
    def __init__(self, *, url: str = "http://127.0.0.1:54321", code: int = 0) -> None:
        self.url = url
        self.code = code
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        backend: Any,
        *,
        port: int,
        open_browser: bool,
        stop_event: threading.Event,
        on_listen: Any,
        on_browser_error: Any,
    ) -> int:
        self.calls.append(
            {
                "backend": backend,
                "port": port,
                "open_browser": open_browser,
                "stop_event": stop_event,
                "on_listen": on_listen,
                "on_browser_error": on_browser_error,
            }
        )
        on_listen(self.url)
        return self.code


def _runner() -> Any:
    from typer.testing import CliRunner

    return CliRunner()


def _invoke(args: list[str]) -> Any:
    from seekr.cli import app

    return _runner().invoke(app, args)


def test_register_adds_web_with_frozen_options() -> None:
    import typer

    fresh = typer.Typer()
    cli_web.register(fresh)

    result = _runner().invoke(fresh, ["web", "--help"])
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "--port" in output
    assert "--open-browser" in output
    assert "8000" in output
    assert "ephemeral" in output


def test_help_does_not_create_backend_or_run_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_web,
        "_create_backend",
        lambda: pytest.fail("help must not initialize the backend"),
    )
    monkeypatch.setattr(
        cli_web,
        "_run_server",
        lambda *args, **kwargs: pytest.fail("help must not start a server"),
    )

    result = _invoke(["web", "--help"])
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "--port" in output
    assert "--open-browser" in output


def test_port_zero_and_open_browser_pass_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = CleanupBackend()
    recording = RecordingRun()
    monkeypatch.setattr(cli_web, "_create_backend", lambda: backend)
    monkeypatch.setattr(cli_web, "_run_server", recording)

    result = _invoke(["web", "--foreground", "--port", "0", "--open-browser"])

    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:54321" in result.output
    assert "Ctrl+C" in result.output
    assert len(recording.calls) == 1
    call = recording.calls[0]
    assert call["backend"] is backend
    assert call["port"] == 0
    assert call["open_browser"] is True
    assert isinstance(call["stop_event"], threading.Event)
    assert not call["stop_event"].is_set()


def test_default_port_and_browser_are_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = RecordingRun()
    monkeypatch.setattr(cli_web, "_create_backend", CleanupBackend)
    monkeypatch.setattr(cli_web, "_run_server", recording)

    result = _invoke(["web", "--foreground"])

    assert result.exit_code == 0, result.output
    assert recording.calls[0]["port"] == 8000
    assert recording.calls[0]["open_browser"] is False


def test_invalid_port_rejected_before_backend_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_web,
        "_create_backend",
        lambda: pytest.fail("invalid port must not create a backend"),
    )
    for value in ("-1", "65536"):
        result = _invoke(["web", "--port", value])
        assert result.exit_code == 2, result.output
        assert "between 0 and 65535" in result.output


def test_occupied_port_reports_clear_error_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    backend = CleanupBackend()
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        monkeypatch.setattr(cli_web, "_create_backend", lambda: backend)

        result = _invoke(["web", "--foreground", "--port", str(port)])
    finally:
        listener.close()

    assert result.exit_code == 1, result.output
    assert "already in use" in result.output
    assert backend.shutdown_calls == 1
    assert backend.close_calls == 1


def test_backend_factory_failure_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> Any:
        raise RuntimeError("fixture factory failure")

    monkeypatch.setattr(cli_web, "_create_backend", fail)

    result = _invoke(["web", "--foreground"])

    assert result.exit_code == 1
    assert "could not initialize the web service" in result.output
    assert "fixture factory failure" in result.output


def test_run_server_unexpected_failure_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_web, "_create_backend", CleanupBackend)

    def fail(*args: Any, **kwargs: Any) -> int:
        raise RuntimeError("fixture server failure")

    monkeypatch.setattr(cli_web, "_run_server", fail)

    result = _invoke(["web", "--foreground"])

    assert result.exit_code == 1
    assert "stopped unexpectedly" in result.output


def test_keyboard_interrupt_exits_zero_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_web, "_create_backend", CleanupBackend)

    def interrupt(*args: Any, **kwargs: Any) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_web, "_run_server", interrupt)

    result = _invoke(["web", "--foreground"])

    assert result.exit_code == 0, result.output


def test_nonzero_server_exit_is_propagated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_web, "_create_backend", CleanupBackend)
    monkeypatch.setattr(cli_web, "_run_server", RecordingRun(code=1))

    result = _invoke(["web", "--foreground"])

    assert result.exit_code == 1


def test_browser_failure_message_does_not_fail_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_web, "_create_backend", CleanupBackend)

    def run_with_browser_warning(
        backend: Any,
        *,
        port: int,
        open_browser: bool,
        stop_event: threading.Event,
        on_listen: Any,
        on_browser_error: Any,
    ) -> int:
        on_listen("http://127.0.0.1:54321")
        on_browser_error("could not open a browser (fixture); open the URL manually")
        return 0

    monkeypatch.setattr(cli_web, "_run_server", run_with_browser_warning)

    result = _invoke(["web", "--foreground", "--open-browser"])

    assert result.exit_code == 0, result.output
    assert "could not open a browser" in result.output


def test_importing_cli_is_lightweight() -> None:
    script = (
        "import sys\n"
        "import seekr.cli\n"
        "import seekr.cli_web\n"
        "heavy = ('http.server', 'seekr.web_server', 'seekr.web_service', "
        "'numpy', 'sentence_transformers', 'sqlite_vec', 'torch')\n"
        "print(' '.join(name for name in heavy if name in sys.modules))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_web_wires_service_shutdown_request_to_stop_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callbacks: list[object] = []

    class SignalBackend(CleanupBackend):
        def set_shutdown_callback(self, callback: object) -> None:
            callbacks.append(callback)

    recording = RecordingRun()
    monkeypatch.setattr(cli_web, "_create_backend", SignalBackend)
    monkeypatch.setattr(cli_web, "_run_server", recording)

    result = _invoke(["web", "--foreground"])

    assert result.exit_code == 0, result.output
    stop_event = recording.calls[0]["stop_event"]
    assert len(callbacks) == 1
    assert not stop_event.is_set()
    callbacks[0]()
    assert stop_event.is_set()


# ---- background service path -------------------------------------------------


def _info(port: int = 8000) -> Any:
    from seekr.service_discovery import ServiceInfo

    return ServiceInfo(
        pid=1, port=port, token="t", version="1", started_at="2026-01-01T00:00:00+00:00"
    )


class FakeProc:
    def __init__(self, exit_code: int | None = None) -> None:
        self.exit_code = exit_code

    def poll(self) -> int | None:
        return self.exit_code


def _no_foreground(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_web, "_create_backend", lambda: pytest.fail("must not run in the foreground")
    )


def test_background_reuses_running_service(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_foreground(monkeypatch)
    monkeypatch.setattr(cli_web, "_find_service", lambda: _info(9123))
    monkeypatch.setattr(
        cli_web, "_spawn_service", lambda *a, **k: pytest.fail("must not spawn")
    )

    result = _invoke(["web"])

    assert result.exit_code == 0, result.output
    assert "seekr is running at http://127.0.0.1:9123/" in result.output


def test_background_notes_port_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_web, "_find_service", lambda: _info(9123))

    result = _invoke(["web", "--port", "8500"])

    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:9123/" in result.output
    assert "port 9123" in result.output


def test_background_spawns_then_prints_url_and_opens_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_foreground(monkeypatch)
    answers = iter([None, None, _info(8000)])
    spawned: list[int] = []
    opened: list[str] = []
    monkeypatch.setattr(cli_web, "_find_service", lambda: next(answers))
    monkeypatch.setattr(cli_web, "_POLL_SECONDS", 0.0)
    monkeypatch.setattr(
        cli_web, "_spawn_service", lambda port: spawned.append(port) or FakeProc()
    )
    monkeypatch.setattr(cli_web, "_open_url", lambda url: opened.append(url) or True)

    result = _invoke(["web", "--open-browser"])

    assert result.exit_code == 0, result.output
    assert spawned == [8000]
    assert opened == ["http://127.0.0.1:8000/"]
    assert "seekr is running at http://127.0.0.1:8000/" in result.output


def test_background_failure_exits_nonzero_with_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_web, "_find_service", lambda: None)
    monkeypatch.setattr(cli_web, "_spawn_service", lambda port: FakeProc(exit_code=1))
    monkeypatch.setattr(cli_web, "_POLL_SECONDS", 0.0)

    result = _invoke(["web"])

    assert result.exit_code == 1
    assert "seekr service start" in result.output


def test_background_timeout_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_web, "_find_service", lambda: None)
    monkeypatch.setattr(cli_web, "_spawn_service", lambda port: FakeProc())
    monkeypatch.setattr(cli_web, "_POLL_SECONDS", 0.0)
    monkeypatch.setattr(cli_web, "_START_WAIT_SECONDS", 0.0)

    result = _invoke(["web"])

    assert result.exit_code == 1
    assert "seekr service start" in result.output


def test_spawn_argv_and_posix_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr(
        cli_web.subprocess,
        "Popen",
        lambda argv, **kw: calls.append((argv, kw)) or FakeProc(),
    )
    monkeypatch.setattr(cli_web.sys, "platform", "linux")

    cli_web._spawn_service(8123)

    argv, kw = calls[0]
    assert argv == [sys.executable, "-m", "seekr", "service", "start", "--port", "8123"]
    assert kw["start_new_session"] is True
    assert "creationflags" not in kw
    assert kw["stdin"] is subprocess.DEVNULL
    assert kw["stdout"] is subprocess.DEVNULL
    assert kw["stderr"] is subprocess.DEVNULL
    assert kw["close_fds"] is True


def test_spawn_flags_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli_web.subprocess, "Popen", lambda argv, **kw: calls.append(kw) or FakeProc()
    )
    monkeypatch.setattr(cli_web.sys, "platform", "win32")

    cli_web._spawn_service(8000)

    kw = calls[0]
    assert "start_new_session" not in kw
    assert kw["creationflags"] == 0x00000008 | 0x00000200


def test_foreground_flag_keeps_old_behaviour(monkeypatch: pytest.MonkeyPatch) -> None:
    recording = RecordingRun()
    monkeypatch.setattr(cli_web, "_create_backend", CleanupBackend)
    monkeypatch.setattr(cli_web, "_run_server", recording)
    monkeypatch.setattr(
        cli_web, "_find_service", lambda: pytest.fail("foreground skips discovery")
    )

    result = _invoke(["web", "--foreground"])

    assert result.exit_code == 0, result.output
    assert len(recording.calls) == 1


def test_background_invalid_port_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_web, "_find_service", lambda: pytest.fail("validate the port first")
    )

    result = _invoke(["web", "--port", "70000"])

    assert result.exit_code == 2
