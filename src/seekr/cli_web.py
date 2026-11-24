"""The ``seekr web`` command.

Runtime imports of the HTTP server and application service stay inside the
command callback so ``import seekr.cli`` (and ``seekr web --help``) remains
lightweight: no ``http.server``, model, Store, database, or worker thread is
loaded until the command actually runs.
"""

from __future__ import annotations

import errno
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import typer

__all__ = ["register"]


def _create_backend() -> Any:
    """Build the application service lazily (monkeypatchable in tests)."""
    from .web_service import WebService

    return WebService()


def _run_server(
    backend: Any,
    *,
    port: int,
    open_browser: bool,
    stop_event: threading.Event,
    on_listen: Callable[[str], None],
    on_browser_error: Callable[[str], None],
) -> int:
    """Run the HTTP server lazily (monkeypatchable in tests)."""
    from .web_server import run_web

    return run_web(
        backend,
        port=port,
        open_browser=open_browser,
        stop_event=stop_event,
        on_listen=on_listen,
        on_browser_error=on_browser_error,
    )


_START_WAIT_SECONDS = 10.0
_POLL_SECONDS = 0.1


def _find_service() -> Any:
    """Return the live service's discovery record, or None (patchable)."""
    from .service import find_running_service

    return find_running_service()


def _spawn_kwargs(platform: str | None = None) -> dict[str, Any]:
    """Popen options that detach the service from this terminal."""
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if (platform if platform is not None else sys.platform) == "win32":
        kwargs["creationflags"] = getattr(
            subprocess, "DETACHED_PROCESS", 0x00000008
        ) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
    else:
        kwargs["start_new_session"] = True
    return kwargs


def _spawn_service(port: int, platform: str | None = None) -> Any:
    """Start ``seekr service start`` detached from this process."""
    argv = [sys.executable, "-m", "seekr", "service", "start", "--port", str(port)]
    return subprocess.Popen(argv, **_spawn_kwargs(platform))


def _wait_for_service(proc: Any, timeout: float = _START_WAIT_SECONDS) -> Any:
    """Poll until the spawned service answers, it exits, or *timeout* passes."""
    deadline = time.monotonic() + timeout
    while True:
        info = _find_service()
        if info is not None:
            return info
        exited = getattr(proc, "poll", lambda: None)() is not None
        if exited or time.monotonic() >= deadline:
            return _find_service()
        time.sleep(_POLL_SECONDS)


def _open_url(url: str) -> bool:
    import webbrowser

    return bool(webbrowser.open(url))


def _run_background(typer_module: Any, port: int, open_browser: bool) -> None:
    info = _find_service()
    if info is None:
        try:
            proc = _spawn_service(port)
        except OSError as exc:
            typer_module.echo(
                f"seekr web: could not start the background service: {exc}",
                err=True,
            )
            raise typer_module.Exit(code=1) from None
        info = _wait_for_service(proc)
        if info is None:
            typer_module.echo(
                "seekr web: the background service did not start within "
                f"{int(_START_WAIT_SECONDS)} s; run `seekr service start` "
                "to see the error.",
                err=True,
            )
            raise typer_module.Exit(code=1)
    elif port not in (0, info.port):
        typer_module.echo(
            f"seekr web: the service is already running on port {info.port}; "
            f"ignoring --port {port}.",
            err=True,
        )
    url = f"http://127.0.0.1:{info.port}/"
    typer_module.echo(f"seekr is running at {url}")
    if open_browser and not _open_url(url):
        typer_module.echo("seekr web: could not open a browser", err=True)


def _build_web_command(typer_module: Any) -> Any:
    def web(
        port: int = typer_module.Option(
            8000,
            "--port",
            metavar="PORT",
            help="Local loopback port to serve on (use 0 for an ephemeral port).",
        ),
        open_browser: bool = typer_module.Option(
            False,
            "--open-browser",
            help="Open the local web interface in the default browser.",
        ),
        foreground: bool = typer_module.Option(
            False,
            "--foreground",
            help="Serve in this terminal instead of the background service.",
        ),
    ) -> None:
        """Open the local browser interface (starts the background service)."""
        import typer

        if not 0 <= port <= 65535:
            typer.echo(
                "seekr web: --port must be between 0 and 65535",
                err=True,
            )
            raise typer.Exit(code=2)

        if not foreground:
            _run_background(typer, port, open_browser)
            return

        stop_event = threading.Event()

        def on_listen(url: str) -> None:
            typer.echo(f"Seekr web interface: {url}")
            typer.echo("Press Ctrl+C to stop.")

        def on_browser_error(message: str) -> None:
            typer.echo(f"seekr web: {message}", err=True)

        try:
            backend = _create_backend()
        except Exception as exc:  # noqa: BLE001 - report a safe startup error
            typer.echo(
                f"seekr web: could not initialize the web service: {exc}",
                err=True,
            )
            raise typer.Exit(code=1) from None

        set_callback = getattr(backend, "set_shutdown_callback", None)
        if callable(set_callback):
            set_callback(stop_event.set)

        try:
            exit_code = _run_server(
                backend,
                port=port,
                open_browser=open_browser,
                stop_event=stop_event,
                on_listen=on_listen,
                on_browser_error=on_browser_error,
            )
        except KeyboardInterrupt:
            exit_code = 0
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                typer.echo(
                    f"seekr web: port {port} is already in use; "
                    "choose another --port (use --port 0 for an ephemeral port).",
                    err=True,
                )
            else:
                typer.echo(
                    f"seekr web: could not start the local server: {exc}",
                    err=True,
                )
            raise typer.Exit(code=1) from None
        except Exception as exc:  # noqa: BLE001 - report a safe startup error
            typer.echo(
                f"seekr web: the local server stopped unexpectedly: {exc}",
                err=True,
            )
            raise typer.Exit(code=1) from None

        if exit_code:
            raise typer.Exit(code=exit_code)

    return web


def register(app: typer.Typer) -> None:
    """Register the ``web`` command on the main Typer application."""
    import typer

    app.command("web")(_build_web_command(typer))
