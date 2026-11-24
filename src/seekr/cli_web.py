"""The ``seekr web`` command.

Runtime imports of the HTTP server and application service stay inside the
command callback so ``import seekr.cli`` (and ``seekr web --help``) remains
lightweight: no ``http.server``, model, Store, database, or worker thread is
loaded until the command actually runs.
"""

from __future__ import annotations

import errno
import threading
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
    ) -> None:
        """Serve the local browser interface over loopback."""
        import typer

        if not 0 <= port <= 65535:
            typer.echo(
                "seekr web: --port must be between 0 and 65535",
                err=True,
            )
            raise typer.Exit(code=2)

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
