"""The ``seekr daemon`` command and its testable run loop.

The command module only exposes :func:`register`; importing it does not load a
model, database, watchdog observer or the optional reranker.  The daemon holds
one shared lifecycle lock and one Store for its whole lifetime, runs the
initial sync while the watcher buffers events, and then serves queries while a
single watcher thread keeps the index current.
"""

from __future__ import annotations

import logging
import signal
import sqlite3
import threading
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import Config, load_config
from .daemon import serve, socket_path
from .embed import Embedder, build_embedder
from .indexer import Indexer
from .lifecycle import lifecycle_lock
from .logging_config import log_event
from .search import Searcher
from .store import Store
from .watcher import Watcher

_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    import typer

__all__ = ["register", "run_daemon"]


def _noop() -> None:
    return None


def _install_signal_handlers(stop_event: threading.Event) -> Callable[[], None]:
    if threading.current_thread() is not threading.main_thread():
        return _noop

    previous: dict[int, Any] = {}

    def request_stop(signum: int, frame: Any) -> None:
        stop_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)
        except (OSError, ValueError):
            continue

    def restore() -> None:
        for signum, handler in previous.items():
            try:
                signal.signal(signum, handler)
            except (OSError, ValueError):
                continue

    return restore


def _reranker_factory(cfg: Config) -> Callable[[], object]:
    def build() -> object:
        from .rerank import build_reranker

        return build_reranker(cfg)

    return build


def _stored_meta(cfg: Config) -> dict[str, str] | None:
    db_path = Path(cfg.db_path)
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path), timeout=1.0)
    except sqlite3.Error:
        return None
    try:
        rows = conn.execute("SELECT key, value FROM meta").fetchall()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    return {str(key): str(value) for key, value in rows}


def run_daemon(
    *,
    cfg: Config | None = None,
    sock_path: Path | None = None,
    debounce_s: float = 1.0,
    stop_event: threading.Event | None = None,
    embedder: Embedder | None = None,
    handle_signals: bool = True,
) -> None:
    """Run the watcher and query daemon until *stop_event* is set."""
    selected_cfg = cfg if cfg is not None else load_config()
    selected_socket = Path(sock_path) if sock_path is not None else socket_path()
    event = stop_event if stop_event is not None else threading.Event()
    restore_signals = (
        _install_signal_handlers(event) if handle_signals else _noop
    )

    watcher: Watcher | None = None
    store: Store | None = None
    try:
        # The lock covers the whole Store lifetime, including the watcher and
        # every request thread, so `seekr reindex` cannot swap the database
        # underneath a running daemon.
        with lifecycle_lock(selected_cfg.db_path, shared=True):
            active_embedder = (
                embedder if embedder is not None else build_embedder(selected_cfg)
            )
            dim: int | None = None
            if embedder is None:
                stored_meta = _stored_meta(selected_cfg)
                if stored_meta is not None:
                    try:
                        candidate_dim = int(stored_meta.get("dim", "0"))
                    except (TypeError, ValueError):
                        candidate_dim = 0
                    if candidate_dim > 0:
                        dim = candidate_dim
            if dim is None:
                dim = int(active_embedder.dim)

            store = Store(
                selected_cfg.db_path,
                selected_cfg.embed_model,
                dim,
                vector_quant=selected_cfg.vector_quant,
            )
            try:
                searcher = Searcher(store, active_embedder, None)
                searcher.daemon_metadata = {  # type: ignore[attr-defined]
                    "db_path": str(
                        Path(selected_cfg.db_path).expanduser().resolve(strict=False)
                    ),
                    "embed_model": selected_cfg.embed_model,
                    "dim": dim,
                    "vector_quant": selected_cfg.vector_quant,
                    "rerank_model": selected_cfg.rerank_model,
                }
                indexer = Indexer(store, active_embedder, selected_cfg)

                # Start the observer first but hold the writer until the initial
                # scan has finished, so events during startup are buffered and
                # there is never more than one writer at a time.
                log_event(
                    _LOGGER,
                    logging.INFO,
                    "daemon.start",
                    db_path=selected_cfg.db_path,
                    socket=selected_socket,
                    debounce_s=debounce_s,
                    quant=selected_cfg.vector_quant,
                )
                watcher = Watcher(
                    indexer,
                    list(selected_cfg.roots),
                    debounce_s,
                )
                watcher.start(paused=True)
                sync_report = indexer.sync(list(selected_cfg.roots))
                if sync_report is not None:
                    log_event(
                        _LOGGER,
                        logging.INFO,
                        "daemon.sync_done",
                        added=sync_report.added,
                        updated=sync_report.updated,
                        unchanged=sync_report.unchanged,
                        removed=sync_report.removed,
                        errors=len(sync_report.errors),
                    )
                else:
                    log_event(_LOGGER, logging.INFO, "daemon.sync_done")
                watcher.resume()

                if embedder is None:

                    def _prewarm() -> None:
                        if event.is_set():
                            return
                        with suppress(ImportError, OSError, RuntimeError, ValueError):
                            _ = active_embedder.dim

                    threading.Thread(
                        target=_prewarm,
                        name="seekr-embed-prewarm",
                        daemon=True,
                    ).start()

                serve(
                    searcher,
                    selected_socket,
                    event,
                    reranker_factory=_reranker_factory(selected_cfg),
                )
            finally:
                try:
                    if watcher is not None:
                        watcher.stop()
                finally:
                    store.close()
                    store = None
    finally:
        log_event(_LOGGER, logging.INFO, "daemon.stop", socket=selected_socket)
        restore_signals()


def _build_daemon_command(typer_module: Any) -> Any:
    def daemon(
        socket_file: str | None = typer_module.Option(
            None, "--socket", help="Unix socket path to serve on."
        ),
        debounce_s: float = typer_module.Option(
            1.0, "--debounce", min=0.0, help="Per-path debounce in seconds."
        ),
    ) -> None:
        """Watch configured roots and serve warm searches until interrupted."""
        target = Path(socket_file).expanduser() if socket_file else None
        try:
            run_daemon(sock_path=target, debounce_s=debounce_s)
        except KeyboardInterrupt:
            return
        except Exception as exc:
            typer_module.echo(f"seekr daemon: {exc}", err=True)
            raise typer_module.Exit(code=1) from exc

    daemon.__doc__ = "Watch configured roots and serve warm searches until interrupted."
    return daemon


def register(app: typer.Typer) -> None:
    """Register the ``daemon`` command on the main Typer application."""
    import typer

    app.command("daemon")(_build_daemon_command(typer))
