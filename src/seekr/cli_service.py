"""The ``seekr service`` command group: start, stop and status.

Importing this module loads nothing heavy; the service, HTTP server, Store and
models are imported only when a subcommand runs.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import typer

__all__ = ["register"]

_STOP_WAIT_SECONDS = 10.0
_LEGACY_DAEMON_RUNNING = (
    "a seekr daemon from an older version is still running; stop it first "
    "(seekr daemon is replaced by seekr service)"
)


def _legacy_daemon_running() -> bool:
    """Return whether a pre-0.3 ``seekr daemon`` answers on its Unix socket.

    The service never creates that socket, so the deprecated ``seekr daemon``
    alias (which runs the service) cannot trip over its own check.
    """
    from .daemon import _socket_is_active, legacy_daemon_supported, socket_path

    if not legacy_daemon_supported():
        return False
    try:
        return _socket_is_active(socket_path())
    except OSError:
        return False


def _create_service(**kwargs: Any) -> Any:
    """Build the service lazily (monkeypatchable in tests)."""
    from .service import SeekrService

    return SeekrService(**kwargs)


def _format_uptime(started_at: str) -> str:
    try:
        started = datetime.fromisoformat(started_at)
    except ValueError:
        return "unknown"
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    seconds = max(0, int((datetime.now(UTC) - started).total_seconds()))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s"


def _build_start(typer_module: Any) -> Any:
    def start(
        port: int = typer_module.Option(
            0, "--port", metavar="PORT", help="Loopback port (0 picks a free one)."
        ),
        model_dir: Path | None = typer_module.Option(
            None, "--model-dir", metavar="DIR", help="Use a bundled ONNX model directory."
        ),
        handshake_stdout: bool = typer_module.Option(
            False, "--handshake-stdout", help="Print a JSON ready line on stdout."
        ),
        watch_stdin: bool = typer_module.Option(
            False, "--watch-stdin", help="Stop when stdin reaches end of file."
        ),
    ) -> None:
        """Run the seekr service (HTTP API, web UI, watcher) until stopped."""
        from .service import ServiceAlreadyRunning

        if not 0 <= port <= 65535:
            typer_module.echo("seekr service: --port must be between 0 and 65535", err=True)
            raise typer_module.Exit(code=2)
        if _legacy_daemon_running():
            # It would keep writing the index with old code beside the service.
            typer_module.echo(f"seekr service: {_LEGACY_DAEMON_RUNNING}", err=True)
            raise typer_module.Exit(code=1)
        service = _create_service(
            port=port,
            model_dir=model_dir,
            handshake_stdout=handshake_stdout,
            watch_stdin=watch_stdin,
        )
        try:
            bound = service.start()
        except ServiceAlreadyRunning as exc:
            if handshake_stdout:
                return
            typer_module.echo(f"seekr service: {exc}", err=True)
            raise typer_module.Exit(code=1) from None
        except (OSError, RuntimeError, ValueError) as exc:
            typer_module.echo(f"seekr service: could not start: {exc}", err=True)
            raise typer_module.Exit(code=1) from None
        if not handshake_stdout:
            typer_module.echo(f"Seekr service: http://127.0.0.1:{bound}")
            typer_module.echo("Press Ctrl+C to stop.")
        service.serve_forever()

    return start


def _build_stop(typer_module: Any) -> Any:
    def stop() -> None:
        """Ask the running seekr service to stop and wait for it."""
        import http.client

        from . import paths
        from .service import service_request
        from .service_discovery import read_service_info

        path = paths.service_file()
        info = read_service_info(path)
        if info is None:
            typer_module.echo("seekr service: not running")
            return
        try:
            status, _ = service_request(
                info.port, "POST", "/api/shutdown", token=info.token, body={}
            )
        except (OSError, http.client.HTTPException) as exc:
            typer_module.echo(f"seekr service: could not reach the service: {exc}", err=True)
            raise typer_module.Exit(code=1) from None
        if status != 202:
            typer_module.echo(f"seekr service: shutdown was refused (HTTP {status})", err=True)
            raise typer_module.Exit(code=1)
        deadline = time.monotonic() + _STOP_WAIT_SECONDS
        while read_service_info(path) is not None:
            if time.monotonic() >= deadline:
                typer_module.echo("seekr service: still stopping after 10 s", err=True)
                raise typer_module.Exit(code=1)
            time.sleep(0.05)
        typer_module.echo("seekr service: stopped")

    return stop


def _build_status(typer_module: Any) -> Any:
    def status() -> None:
        """Show the running seekr service's pid, port, version and uptime."""
        from .service import find_running_service

        # Probing the API (not just the pid) rejects stale files whose pid was
        # reused and services that are already stopping.
        info = find_running_service(timeout=1.0)
        if info is None:
            typer_module.echo("seekr service: not running")
            return
        typer_module.echo(f"pid: {info.pid}")
        typer_module.echo(f"port: {info.port}")
        typer_module.echo(f"version: {info.version}")
        typer_module.echo(f"uptime: {_format_uptime(info.started_at)}")

    return status


def register(app: typer.Typer) -> None:
    """Register the ``service`` command group on the main Typer app."""
    import typer

    group = typer.Typer(help="Run and control the background seekr service.")
    group.command("start")(_build_start(typer))
    group.command("stop")(_build_stop(typer))
    group.command("status")(_build_status(typer))
    app.add_typer(group, name="service")
