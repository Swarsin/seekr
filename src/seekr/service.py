"""The unified seekr service: one loopback HTTP process for every client.

:class:`SeekrService` composes the web application service (engine lane, jobs,
HTTP API and web UI) with the file watcher and a warm embedder.  Clients find
it through the user-only discovery file written by
:mod:`seekr.service_discovery`.

Every index write runs in the web service's engine lane: web index jobs take
it directly and watcher actions wait for it, so there is never more than one
writer.  The watcher's long-lived Store is opened lazily inside the lane by
:meth:`SeekrService._watch_indexer` and closed by
:meth:`SeekrService._close_watch_store` (on stop, or when roots change).

Importing this module is cheap: the HTTP server, Store, Indexer and models are
loaded only by :meth:`SeekrService.start`.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import signal
import sqlite3
import sys
import threading
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from . import __version__, paths
from .config import Config, ConfigError, default_config_path, load_config
from .logging_config import log_event
from .service_client import service_answers, service_request
from .service_discovery import (
    ServiceInfo,
    read_service_info,
    remove_service_info,
    write_service_info,
)

if TYPE_CHECKING:
    from .embed import Embedder
    from .web_server import SeekrHTTPServer
    from .web_service import WebService

__all__ = [
    "SeekrService",
    "ServiceAlreadyRunning",
    "ServiceStopping",
    "find_running_service",
    "service_request",
    "start_prewarm",
    "stored_dim",
]

_LOGGER = logging.getLogger(__name__)
# Generous: /api/status may wait on a busy index for its own bounded timeout.
_LIVENESS_TIMEOUT = 6.0


class ServiceAlreadyRunning(RuntimeError):
    """Another live service already answers on ``port``."""

    def __init__(self, port: int, info: ServiceInfo | None = None) -> None:
        super().__init__(f"seekr service is already running on port {port}")
        self.port = port
        self.info = info


class ServiceStopping(RuntimeError):
    """Another service holds the lifetime lock but is stopping or unresponsive."""


# -- shared helpers (also used by ``seekr daemon``) --------------------------


def stored_dim(cfg: Config) -> int | None:
    """Return the vector dimension recorded in the index, without a model."""
    db_path = Path(cfg.db_path)
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path), timeout=1.0)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'dim'").fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    try:
        dim = int(row[0]) if row is not None else 0
    except (TypeError, ValueError):
        return None
    return dim if dim > 0 else None


def start_prewarm(embedder: Embedder, stop_event: threading.Event) -> threading.Thread:
    """Load the model on a daemon thread so the first query is warm."""

    def prewarm() -> None:
        if stop_event.is_set():
            return
        with suppress(ImportError, OSError, RuntimeError, ValueError):
            _ = embedder.dim

    thread = threading.Thread(target=prewarm, name="seekr-embed-prewarm", daemon=True)
    thread.start()
    return thread


def _answers(info: ServiceInfo) -> bool:
    return service_answers(info, timeout=_LIVENESS_TIMEOUT)


def find_running_service(path: Path | None = None) -> ServiceInfo | None:
    """Return the discovery record of a live service that answers, or None.

    A service that has begun shutting down refuses ``/api/status`` and so
    counts as not running.
    """
    info = read_service_info(path or paths.service_file())
    if info is None or not _answers(info):
        return None
    return info


class _WatchIndexer:
    """Indexer stand-in for the watcher; resolved inside the engine lane."""

    def __init__(self, service: SeekrService) -> None:
        self._service = service

    def __getattr__(self, name: str) -> Any:
        return getattr(self._service._watch_indexer(), name)


class SeekrService:
    """The loopback service process: HTTP API, web UI, watcher and warm model."""

    def __init__(
        self,
        *,
        config_path: Path | None = None,
        port: int = 0,
        model_dir: Path | None = None,
        handshake_stdout: bool = False,
        watch_stdin: bool = False,
        embedder_factory: Callable[[Config], Embedder] | None = None,
        observer_factory: Callable[[], Any] | None = None,
        debounce_s: float = 1.0,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        """Configure the service; nothing is bound or loaded until :meth:`start`.

        ``embedder_factory`` (called with the Config) and ``observer_factory``
        (used for native and polling observers alike) exist for tests.
        """
        self._config_path = Path(config_path) if config_path is not None else None
        self._port = int(port)
        self._model_dir = Path(model_dir) if model_dir is not None else None
        self._handshake_stdout = handshake_stdout
        self._watch_stdin = watch_stdin
        self._embedder_factory = embedder_factory
        self._observer_factory = observer_factory
        self._debounce_s = debounce_s
        self._stdin = stdin
        self._stdout = stdout

        self._lock = threading.Lock()
        self._started = False
        self._stopping = False
        self._stopped = threading.Event()
        self._stop_requested = threading.Event()
        self._stop_thread: threading.Thread | None = None

        self._service_file: Path | None = None
        self._lifetime_lock: Any | None = None
        self._web: WebService | None = None
        self._server: SeekrHTTPServer | None = None
        self._serve_thread: threading.Thread | None = None
        self._watcher: Any | None = None

        self._cfg_lock = threading.Lock()
        self._cfg: Config | None = None
        self._embedders: dict[str, Embedder] = {}
        # Watcher Store state; touched only while holding the engine lane.
        self._watch_store: Any | None = None
        self._watch_lock: Any | None = None
        self._watch_index: Any | None = None
        self._watch_stale = False

    # -- public lifecycle --------------------------------------------------

    @property
    def backend(self) -> WebService:
        if self._web is None:
            raise RuntimeError("the service has not been started")
        return self._web

    def start(self) -> int:
        """Bind, publish discovery, start watching and warming; return the port.

        Raises :class:`ServiceAlreadyRunning` if another live service answers
        (after printing an ``existing`` handshake line when enabled).
        """
        with self._lock:
            if self._started:
                raise RuntimeError("the service was already started")
            self._started = True

        from .locks import FileLock

        service_file = paths.service_file()
        service_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Held exclusively for the whole service lifetime and released only
        # after the discovery file is removed: whoever holds it owns
        # service.json.  The discovery file and HTTP probe only describe the
        # holder for the error message and the ``existing`` handshake.
        lifetime_lock = FileLock(service_file.with_name("service.lock"))
        if not lifetime_lock.acquire(shared=False, blocking=False):
            existing = find_running_service(service_file)
            if existing is None:
                raise ServiceStopping(
                    "another seekr service is still stopping or not responding; "
                    "try again shortly"
                )
            if self._handshake_stdout:
                self._emit("existing", existing)
            raise ServiceAlreadyRunning(existing.port, existing)
        self._lifetime_lock = lifetime_lock
        try:
            info = self._bind_and_publish(service_file)
        except BaseException:
            self.stop()
            raise

        try:
            self._start_background(info)
        except BaseException:
            self.stop()
            raise
        if self._handshake_stdout:
            self._emit("ready", info)
        return info.port

    def serve_forever(self) -> None:
        """Block until the service stops (shutdown request, EOF, signal)."""
        restore = _install_stop_signals(self._stop_requested)
        try:
            while not self._stopped.is_set():
                if self._stop_requested.wait(0.2):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            restore()
            self.stop()

    def stop(self) -> None:
        """Stop everything; idempotent, and later callers wait for the first."""
        with self._lock:
            first = not self._stopping
            self._stopping = True
        if not first:
            self._stopped.wait()
            thread = self._stop_thread
            if thread is not None and thread is not threading.current_thread():
                thread.join()
            return
        try:
            self._teardown()
        finally:
            self._stopped.set()

    # -- startup -----------------------------------------------------------

    def _bind_and_publish(self, service_file: Path) -> ServiceInfo:
        from .web_server import create_server
        from .web_service import WebService

        self._cfg = self._read_config()
        self._web = WebService(
            self._config_path,
            embedder_factory=self._embedder_for,
            daemon_probe=lambda _path: False,
            on_roots_changed=self._roots_changed,
        )
        token = secrets.token_urlsafe(32)
        self._server = create_server(self._web, port=self._port, token=token)
        info = ServiceInfo(
            pid=os.getpid(),
            port=self._server.bound_port,
            token=token,
            version=__version__,
            started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        write_service_info(info, service_file)
        self._service_file = service_file
        self._web.set_shutdown_callback(self._request_stop)
        self._serve_thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.1},
            name="seekr-service-http",
            daemon=True,
        )
        self._serve_thread.start()
        log_event(_LOGGER, logging.INFO, "service.start", port=info.port, pid=info.pid)
        return info

    def _start_background(self, info: ServiceInfo) -> None:
        from .watcher import Watcher

        cfg = self._cfg
        roots = list(cfg.roots) if cfg is not None else []
        factory = self._observer_factory
        self._watcher = Watcher(
            _WatchIndexer(self),
            roots,
            self._debounce_s,
            observer_factory=factory,
            polling_observer_factory=(lambda _interval: factory()) if factory else None,
            poll_interval=cfg.poll_interval_seconds if cfg is not None else 15.0,
            run_exclusive=self._run_exclusive,
        )
        self._watcher.start()
        if cfg is not None:
            start_prewarm(self._embedder_for(cfg), self._stop_requested)
        if roots:
            from .web_contract import ApiError

            try:
                self.backend.start_index()
            except ApiError as exc:
                log_event(
                    _LOGGER, logging.WARNING, "service.initial_sync_skipped", code=exc.code
                )
        if self._watch_stdin:
            threading.Thread(
                target=self._wait_for_stdin_eof,
                name="seekr-service-stdin",
                daemon=True,
            ).start()

    def _emit(self, event: str, info: ServiceInfo) -> None:
        line = json.dumps(
            {
                "event": event,
                "port": info.port,
                "pid": info.pid,
                "version": info.version,
                "token": info.token,
            }
        )
        stream = self._stdout if self._stdout is not None else sys.stdout
        stream.write(line + "\n")
        stream.flush()

    def _wait_for_stdin_eof(self) -> None:
        stream = self._stdin if self._stdin is not None else sys.stdin
        with suppress(OSError, ValueError):
            while stream.readline():
                pass
        self._request_stop()

    # -- config and models -------------------------------------------------

    def _read_config(self) -> Config | None:
        try:
            return load_config(self._config_path or default_config_path())
        except ConfigError as exc:
            log_event(_LOGGER, logging.WARNING, "service.config_invalid", error=str(exc))
            return None

    def _embedder_for(self, cfg: Config) -> Embedder:
        """Return the one warm embedder per model, shared by every lane user."""
        with self._cfg_lock:
            embedder = self._embedders.get(cfg.embed_model)
            if embedder is None:
                if self._embedder_factory is not None:
                    embedder = self._embedder_factory(cfg)
                else:
                    from .embed import build_embedder

                    embedder = build_embedder(cfg, model_dir=self._model_dir)
                self._embedders[cfg.embed_model] = embedder
            return embedder

    def _roots_changed(self) -> None:
        cfg = self._read_config()
        with self._cfg_lock:
            self._cfg = cfg
            # The open watcher Indexer holds the old roots; reopen on next use.
            self._watch_stale = True
        watcher = self._watcher
        if watcher is not None:
            watcher.set_roots(list(cfg.roots) if cfg is not None else [])

    # -- watcher Store (engine lane only) ----------------------------------

    def _run_exclusive(self, action: Callable[[], None]) -> None:
        self.backend.run_in_lane("index", action)

    def _watch_indexer(self) -> Any:
        if self._watch_stale:
            self._close_watch_store()
        if self._watch_index is not None:
            return self._watch_index
        with self._cfg_lock:
            cfg = self._cfg
            self._watch_stale = False
        if cfg is None:
            raise RuntimeError("the configuration is invalid; fix config.toml")
        from .indexer import Indexer
        from .lifecycle import LifecycleLock
        from .store import Store

        embedder = self._embedder_for(cfg)
        # A shared lifecycle lock for the Store's lifetime keeps `seekr reindex`
        # from swapping the database underneath the watcher.
        lock = LifecycleLock(cfg.db_path, shared=True, blocking=False).acquire()
        try:
            dim = stored_dim(cfg) or int(embedder.dim)
            store = Store(
                cfg.db_path,
                cfg.embed_model,
                dim,
                vector_quant=cfg.vector_quant,
                embed_backend=cfg.embed_backend,
            )
        except BaseException:
            lock.release()
            raise
        self._watch_lock = lock
        self._watch_store = store
        self._watch_index = Indexer(store, embedder, cfg)
        return self._watch_index

    def _close_watch_store(self) -> None:
        store, lock = self._watch_store, self._watch_lock
        self._watch_index = self._watch_store = self._watch_lock = None
        if store is not None:
            with suppress(Exception):
                store.close()
        if lock is not None:
            with suppress(Exception):
                lock.release()

    # -- shutdown ----------------------------------------------------------

    def _request_stop(self) -> None:
        """Stop from a fresh thread (safe from HTTP handlers and stdin)."""
        self._stop_requested.set()
        with self._lock:
            if self._stopping or self._stop_thread is not None:
                return
            self._stop_thread = threading.Thread(
                target=self.stop, name="seekr-service-stop", daemon=False
            )
            self._stop_thread.start()

    def _teardown(self) -> None:
        self._stop_requested.set()
        try:
            self._stop_components()
            if self._service_file is not None:
                remove_service_info(self._service_file, os.getpid())
        finally:
            # Never leak ownership: release even if a step above raised.
            if self._lifetime_lock is not None:
                self._lifetime_lock.release()
        log_event(_LOGGER, logging.INFO, "service.stop")

    def _stop_components(self) -> None:
        web, watcher, server = self._web, self._watcher, self._server
        if web is not None:
            web.begin_shutdown()  # wakes watcher actions waiting for the lane
        if watcher is not None:
            watcher.stop()
        if web is not None:
            web.shutdown()  # waits for the active job
            # The watcher Store belongs to the engine lane, even at shutdown.
            web.run_in_lane("index", self._close_watch_store, during_shutdown=True)
        if server is not None:
            # BaseServer.shutdown() blocks forever unless serve_forever runs.
            if self._serve_thread is not None:
                server.shutdown()
                self._serve_thread.join()
            server.server_close()

def _install_stop_signals(stop: threading.Event) -> Callable[[], None]:
    if threading.current_thread() is not threading.main_thread():
        return lambda: None
    previous: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[signum] = signal.signal(signum, lambda *_args: stop.set())
        except (OSError, ValueError):
            continue

    def restore() -> None:
        for signum, handler in previous.items():
            with suppress(OSError, ValueError):
                signal.signal(signum, handler)

    return restore
