"""Application service backing the local web interface.

All Store, embedder, indexer, daemon and config work lives here so the HTTP
transport can stay a thin router over :class:`seekr.web_contract.WebBackend`.
Heavy modules are imported inside the operations that need them; importing or
constructing :class:`WebService` never loads a model or opens a database.
"""

from __future__ import annotations

import errno
import logging
import math
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import pathnorm, platform_info
from .config import (
    DEFAULT_EMBED_MODEL,
    Config,
    ConfigError,
    default_config_path,
    load_config,
)
from .course import display_module
from .locations import format_location
from .web_config import add_root as config_add_root
from .web_config import inspect_config, read_roots
from .web_config import remove_root as config_remove_root
from .web_contract import (
    LIMITS,
    NO_SERVICE,
    ApiError,
    CourseModuleDTO,
    CourseView,
    HitDTO,
    JobView,
    PreviewView,
    RootsView,
    SearchRequest,
    SearchResult,
    SimilarRequest,
    StatusView,
)

if TYPE_CHECKING:
    from .embed import Embedder

__all__ = ["WebService"]

_LOGGER = logging.getLogger(__name__)
_LOCK_WAIT_SECONDS = 2.0
_READ_BUSY_TIMEOUT_SECONDS = 2.0
# How long a reindex waits for in-flight searches before the exclusive swap.
_REINDEX_SWAP_WAIT_SECONDS = 10.0
_STATUS_BUSY_TIMEOUT_SECONDS = 5.0
_DIAGNOSTIC_LIMIT = LIMITS["errors_retained"]
_JOBS_RETAINED = LIMITS["jobs_retained"]
_DAEMON_PROBE_TIMEOUT = 0.2
_DOWNLOAD_OUTPUT_LINES = 50
_DEFAULT_PREVIEW_ZOOM = 1.5
_MAX_PREVIEW_PNG_WIDTH = 2000.0
_PDF_PAGE_LOCATOR_RE = re.compile(r"p\.(\d+)(?:-\d+)?\Z")
_OFFLINE_ENV_KEYS = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
)


def _explain_payload(value: object) -> dict[str, object] | None:
    """Map a ``HitExplain`` to a JSON-safe dict (non-finite numbers become ``None``)."""
    if value is None:
        return None
    from dataclasses import fields, is_dataclass

    if not is_dataclass(value):
        return None
    payload: dict[str, object] = {}
    for item in fields(value):
        raw = getattr(value, item.name)
        if isinstance(raw, float) and not math.isfinite(raw):
            raw = None
        payload[item.name] = raw
    return payload


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_message(exc: BaseException) -> str:
    text = str(exc).strip()
    return text or exc.__class__.__name__


def _is_module_filter_error(message: str) -> bool:
    return (
        message.startswith("unknown module ")
        or "is ambiguous; matches:" in message
        or "no modules are labelled yet" in message
    )


def _is_week_filter_error(message: str) -> bool:
    return "invalid --week value" in message or "week_range must" in message


def _is_busy_error(exc: BaseException) -> bool:
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    message = str(exc).lower()
    return "locked" in message or "busy" in message


def _socket_is_live(path: Path) -> bool:
    from .daemon import legacy_daemon_supported, socket_path_problem

    if not legacy_daemon_supported():
        return False
    if socket_path_problem(path) is not None:
        return False
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(_DAEMON_PROBE_TIMEOUT)
    try:
        probe.connect(str(path))
    except (FileNotFoundError, ConnectionRefusedError):
        return False
    except OSError as exc:
        return exc.errno not in (errno.ENOENT, errno.ECONNREFUSED, errno.ENOTSOCK)
    else:
        return True
    finally:
        probe.close()


_SQLITE_INT_MIN = -(2**63)
_SQLITE_INT_MAX = 2**63 - 1
_REQUIRED_INDEX_TABLES = frozenset(
    {"documents", "chunks", "chunks_fts", "chunks_vec", "meta"}
)


def _db_size_bytes(db_path: Path) -> int | None:
    total = 0
    for candidate in (
        db_path,
        db_path.with_name(db_path.name + "-wal"),
        db_path.with_name(db_path.name + "-shm"),
    ):
        try:
            total += candidate.stat().st_size
        except FileNotFoundError:
            continue
        except OSError:
            return None
    return total


class _SqliteBusyError(RuntimeError):
    """A bounded read-only inspection or open timed out on SQLite locking."""


@dataclass(frozen=True)
class _IndexSnapshot:
    state: str
    db_path: Path
    documents: int | None = None
    chunks: int | None = None
    size_bytes: int | None = None
    last_error_count: int | None = None
    model: str | None = None
    dim: int | None = None
    vector_quant: str | None = None
    schema_version: int | None = None
    message: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "db_path": str(self.db_path) if self.db_path.parts else None,
            "documents": self.documents,
            "chunks": self.chunks,
            "size_bytes": self.size_bytes,
            "last_error_count": self.last_error_count,
            "model": self.model,
            "dim": self.dim,
            "vector_quant": self.vector_quant,
            "schema_version": self.schema_version,
            "message": self.message,
        }


@dataclass
class _JobRecord:
    job_id: str
    operation: str
    state: str
    started_at: str
    started_monotonic: float
    finished_at: str | None = None
    finished_monotonic: float | None = None
    report: dict[str, object] | None = None
    error: str | None = None


class _KeywordEmbedder:
    """A never-used embedder placeholder for model-free keyword Searcher use."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @property
    def dim(self) -> int:
        raise RuntimeError("keyword search must not load or inspect the embedding model")

    def embed_documents(self, texts: list[str]) -> Any:
        raise RuntimeError("keyword search must not embed documents")

    def embed_query(self, text: str) -> Any:
        raise RuntimeError("keyword search must not embed the query")


class WebService:
    """Concrete backend for the frozen :class:`WebBackend` protocol.

    The constructor accepts optional injection hooks for tests and embedding
    callers.  Only ``config_path`` is needed in production.  Background jobs
    run on non-daemon threads and are joined by :meth:`shutdown`.
    """

    def __init__(
        self,
        config_path: Path | None = None,
        *,
        embedder_factory: Callable[[Config], Embedder] | None = None,
        indexer_factory: Callable[..., object] | None = None,
        store_factory: Callable[..., object] | None = None,
        searcher_factory: Callable[..., object] | None = None,
        local_search: Callable[..., list[object]] | None = None,
        daemon_search: Callable[..., list[object] | None] | None = None,
        daemon_socket_path: Path | Callable[[], Path] | None = None,
        daemon_probe: Callable[[Path], bool] | None = None,
        lock_wait_seconds: float = _LOCK_WAIT_SECONDS,
        read_busy_timeout: float = _READ_BUSY_TIMEOUT_SECONDS,
        on_shutdown: Callable[[], None] | None = None,
        on_roots_changed: Callable[[], None] | None = None,
        reindex_runner: Callable[[Callable[[], Any]], Any] | None = None,
        service_identity: Mapping[str, object] | None = None,
    ) -> None:
        """Configure the backend; heavy modules load only when used.

        ``service_identity`` (``{"pid", "version"}``) is set by the background
        service so status reports ``service.running``; a plain ``seekr web
        --foreground`` backend leaves it unset.

        ``reindex_runner`` wraps the rebuild of a reindex job (the job thread
        already holds the engine lane); the service uses it to pause the
        watcher and close its Store around the swap.
        """
        configured = default_config_path() if config_path is None else Path(config_path)
        configured = configured.expanduser()
        if not configured.is_absolute():
            configured = Path.cwd() / configured
        self._config_path = Path(os.path.abspath(os.fspath(configured)))

        self._embedder_factory = embedder_factory
        self._indexer_factory = indexer_factory
        self._store_factory = store_factory
        self._searcher_factory = searcher_factory
        self._local_search = local_search
        self._daemon_search = daemon_search
        self._daemon_socket_path = daemon_socket_path
        self._daemon_probe = daemon_probe
        self._lock_wait_seconds = max(0.0, float(lock_wait_seconds))
        self._read_busy_timeout = max(0.0, float(read_busy_timeout))

        self._on_shutdown = on_shutdown
        self._on_roots_changed = on_roots_changed
        self._reindex_runner = reindex_runner
        self._service_status: dict[str, object] = (
            dict(NO_SERVICE)
            if service_identity is None
            else {
                "running": True,
                "pid": service_identity.get("pid"),
                "version": service_identity.get("version"),
            }
        )
        self._shutdown_requested = False
        self._state_lock = threading.RLock()
        # Signalled whenever the lane is released or shutdown begins.
        self._lane_free = threading.Condition(self._state_lock)
        self._config_write_lock = threading.Lock()
        # Serialises model construction; searches never wait on the lane.
        self._model_load_lock = threading.Lock()
        # One warm cross-encoder per (model, backend), built at most once.
        self._reranker_lock = threading.Lock()
        self._rerankers: dict[tuple[str, str], Any] = {}
        self._lane_kind: str | None = None
        self._lane_job_id: str | None = None
        self._root_writers = 0
        self._shutting_down = False
        self._shutdown_complete = False
        self._active_job_thread: threading.Thread | None = None
        self._jobs: dict[str, _JobRecord] = {}

        self._model_name: str | None = None
        self._model_status = "unchecked"
        self._model_dim: int | None = None
        self._model_message: str | None = None
        self._embedder: Embedder | None = None
        self._embedder_model: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def begin_shutdown(self) -> None:
        """Stop admitting new work without waiting for the active job."""
        with self._state_lock:
            self._shutting_down = True
            self._lane_free.notify_all()

    def shutdown(self) -> None:
        """Stop accepting work and join the active background job."""
        with self._state_lock:
            if self._shutdown_complete:
                return
        self.begin_shutdown()
        with self._state_lock:
            thread = self._active_job_thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join()
        with self._state_lock:
            self._shutdown_complete = True

    close = shutdown

    def set_shutdown_callback(self, callback: Callable[[], None] | None) -> None:
        """Set the hook :meth:`request_shutdown` invokes (replaces any earlier one)."""
        self._on_shutdown = callback

    def request_shutdown(self) -> None:
        """Ask the hosting process to stop, without stopping it on this thread.

        The callback must only signal (for example set an event); the host owns
        the actual listener and backend teardown.  Repeated calls signal once.
        """
        with self._state_lock:
            if self._shutdown_requested:
                return
            self._shutdown_requested = True
            callback = self._on_shutdown
        if callback is None:
            return
        try:
            callback()
        except Exception:  # noqa: BLE001 - a signalling hook must not fail the request
            _LOGGER.warning("event=web_shutdown_callback_failed")

    def _reject_if_shutting_down(self) -> None:
        with self._state_lock:
            shutting_down = self._shutting_down
        if shutting_down:
            raise ApiError(409, "busy", "the web service is shutting down")

    # -- config/model state ------------------------------------------------

    def _load_config(self) -> Config:
        try:
            cfg = load_config(self._config_path)
        except ConfigError as exc:
            raise ApiError(
                409,
                "config_invalid",
                f"invalid config in {self._config_path}: {_safe_message(exc)}",
            ) from exc
        self._activate_config_model(cfg)
        return cfg

    def _activate_config_model(self, cfg: Config) -> None:
        with self._state_lock:
            if self._model_name == cfg.embed_model:
                return
            self._model_name = cfg.embed_model
            self._model_status = "unchecked"
            self._model_dim = None
            self._model_message = None
            self._embedder = None
            self._embedder_model = None

    def _model_snapshot(self, model_name: str) -> dict[str, object]:
        with self._state_lock:
            active = self._model_name == model_name
            state = self._model_status if active else "unchecked"
            dim = self._model_dim if active and state == "ready" else None
            message = self._model_message if active and state == "unavailable" else None
        return {"state": state, "model": model_name, "dim": dim, "message": message}

    def _set_model_loading(self, model_name: str) -> None:
        with self._state_lock:
            if self._model_name != model_name:
                return
            self._model_status = "loading"
            self._model_message = None

    def _set_model_unavailable(self, model_name: str, message: str) -> None:
        with self._state_lock:
            if self._model_name != model_name:
                return
            self._model_status = "unavailable"
            self._model_dim = None
            self._model_message = message
            self._embedder = None
            self._embedder_model = None

    def _cache_model_success(
        self, model_name: str, embedder: Embedder, dim: int
    ) -> None:
        with self._state_lock:
            if self._model_name != model_name:
                return
            self._model_status = "ready"
            self._model_dim = dim
            self._model_message = None
            self._embedder = embedder
            self._embedder_model = model_name

    def _cached_embedder(self, model_name: str) -> Embedder | None:
        with self._state_lock:
            if (
                self._model_name == model_name
                and self._model_status == "ready"
                and self._embedder_model == model_name
            ):
                return self._embedder
        return None

    def _model_state_for(self, model_name: str) -> str:
        with self._state_lock:
            if self._model_name != model_name:
                return "unchecked"
            return self._model_status

    def _build_embedder(self, cfg: Config) -> Embedder:
        if self._embedder_factory is not None:
            return self._embedder_factory(cfg)
        from .embed import build_embedder

        return build_embedder(cfg)

    def _load_model_for_lane(self, cfg: Config) -> tuple[Embedder, int]:
        with self._model_load_lock:
            return self._load_model_locked(cfg)

    def _load_model_locked(self, cfg: Config) -> tuple[Embedder, int]:
        cached = self._cached_embedder(cfg.embed_model)
        if cached is not None:
            dim = int(cached.dim)
            if dim <= 0:
                self._set_model_unavailable(
                    cfg.embed_model, "embedding model reported a non-positive dimension"
                )
                raise ApiError(503, "model_unavailable", "embedding model is unavailable")
            return cached, dim

        self._set_model_loading(cfg.embed_model)
        try:
            embedder = self._build_embedder(cfg)
            dim = int(embedder.dim)
            if dim <= 0:
                raise ValueError("embedding model reported a non-positive dimension")
        except Exception as exc:
            message = _safe_message(exc)
            self._set_model_unavailable(cfg.embed_model, message)
            raise ApiError(
                503,
                "model_unavailable",
                f"embedding model {cfg.embed_model!r} is unavailable: {message}",
            ) from exc
        self._cache_model_success(cfg.embed_model, embedder, dim)
        return embedder, dim

    # -- lane --------------------------------------------------------------

    def _begin_lane(self, kind: str, job_id: str | None = None) -> None:
        with self._state_lock:
            if self._shutting_down or self._lane_kind is not None or self._root_writers:
                raise ApiError(409, "busy", "another engine operation is already running")
            self._lane_kind = kind
            self._lane_job_id = job_id

    def _end_lane(self) -> None:
        with self._state_lock:
            self._lane_kind = None
            self._lane_job_id = None
            self._lane_free.notify_all()

    def run_in_lane(
        self,
        kind: str,
        fn: Callable[[], Any],
        *,
        wait: bool = True,
        during_shutdown: bool = False,
    ) -> Any:
        """Run ``fn`` holding the engine lane, waiting for it when ``wait``.

        Raises ``409 busy`` once shutdown has begun (also for waiters), unless
        ``during_shutdown`` is set for final cleanup by the host.
        """
        with self.lane(kind, wait=wait, during_shutdown=during_shutdown):
            return fn()

    @contextmanager
    def lane(
        self, kind: str, *, wait: bool = True, during_shutdown: bool = False
    ) -> Iterator[None]:
        """Hold the engine lane for the ``with`` body (see :meth:`run_in_lane`)."""
        with self._state_lock:
            while (
                wait
                and (during_shutdown or not self._shutting_down)
                and (self._lane_kind is not None or self._root_writers)
            ):
                self._lane_free.wait()
            if during_shutdown and self._lane_kind is None and not self._root_writers:
                self._lane_kind = kind
            else:
                self._begin_lane(kind)
        try:
            yield
        finally:
            self._end_lane()

    def _activity(self) -> dict[str, object]:
        with self._state_lock:
            return {"lane": self._lane_kind or "idle", "job_id": self._lane_job_id}

    # -- daemon ------------------------------------------------------------

    def _daemon_socket(self) -> Path | None:
        from .daemon import legacy_daemon_supported

        if not legacy_daemon_supported():
            return None
        configured = self._daemon_socket_path
        if configured is None:
            from .daemon import socket_path

            return socket_path()
        if callable(configured):
            return Path(configured())
        return Path(configured)

    def _daemon_status(self) -> tuple[bool, Path | None]:
        try:
            path = self._daemon_socket()
        except Exception:  # noqa: BLE001 - status must stay usable
            return False, None
        if path is None:
            return False, None
        probe = self._daemon_probe or _socket_is_live
        try:
            return bool(probe(path)), path
        except Exception:  # noqa: BLE001 - liveness detection is best-effort
            return False, path

    def daemon_present(self) -> bool:
        """Return only socket liveness; never claim indexing activity."""
        return self._daemon_status()[0]

    # -- jobs --------------------------------------------------------------

    def _new_job(self, operation: str) -> _JobRecord:
        with self._state_lock:
            while len(self._jobs) >= _JOBS_RETAINED:
                victim = next(
                    (job_id for job_id, record in self._jobs.items() if record.state != "running"),
                    None,
                )
                if victim is None:
                    raise ApiError(409, "busy", "too many running jobs")
                del self._jobs[victim]
            record = _JobRecord(
                job_id=uuid.uuid4().hex,
                operation=operation,
                state="running",
                started_at=_now_iso(),
                started_monotonic=time.monotonic(),
            )
            self._jobs[record.job_id] = record
            return record

    def _set_job_finished_locked(
        self,
        job_id: str,
        state: str,
        *,
        report: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        record = self._jobs.get(job_id)
        if record is None:
            return
        record.state = state
        record.finished_at = _now_iso()
        record.finished_monotonic = time.monotonic()
        record.report = report
        record.error = error

    def _finish_job_and_release_lane(
        self,
        job_id: str,
        state: str,
        *,
        report: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        """Publish a terminal job and clear the engine lane atomically."""
        with self._state_lock:
            self._set_job_finished_locked(
                job_id, state, report=report, error=error
            )
            if self._active_job_thread is threading.current_thread():
                self._active_job_thread = None
            self._lane_kind = None
            self._lane_job_id = None
            self._lane_free.notify_all()

    def _job_view(self, record: _JobRecord, now: float) -> JobView:
        finished = record.finished_monotonic
        elapsed = (finished if finished is not None else now) - record.started_monotonic
        return JobView(
            job_id=record.job_id,
            operation=record.operation,
            state=record.state,
            started_at=record.started_at,
            finished_at=record.finished_at,
            elapsed_ms=max(0, int(elapsed * 1000.0)),
            report=record.report,
            error=record.error,
        )

    def job(self, job_id: str) -> JobView:
        with self._state_lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise ApiError(404, "job_not_found", "job was not found or has expired")
            return self._job_view(record, time.monotonic())

    def _start_job(
        self,
        operation: str,
        target: Callable[[str], None],
    ) -> str:
        self._begin_lane(operation)
        record: _JobRecord | None = None
        try:
            record = self._new_job(operation)
            thread = threading.Thread(
                target=target,
                args=(record.job_id,),
                name=f"seekr-web-{operation}",
                daemon=False,
            )
            # Publishing the thread and starting it under the state lock closes
            # the shutdown race: shutdown either sees a live thread to join or
            # this request observes _shutting_down and starts nothing.
            with self._state_lock:
                if self._shutting_down:
                    raise ApiError(
                        409, "busy", "the web service is shutting down"
                    )
                self._lane_job_id = record.job_id
                self._active_job_thread = thread
                try:
                    thread.start()
                except BaseException:
                    self._active_job_thread = None
                    raise
            return record.job_id
        except ApiError as exc:
            if record is not None:
                self._finish_job_and_release_lane(
                    record.job_id, "failed", error=exc.message
                )
            else:
                self._end_lane()
            raise
        except Exception as exc:
            if record is not None:
                self._finish_job_and_release_lane(
                    record.job_id,
                    "failed",
                    error=f"job could not start: {_safe_message(exc)}",
                )
            else:
                self._end_lane()
            raise ApiError(500, "internal_error", "job could not start") from exc

    # -- status ------------------------------------------------------------

    def status(self) -> StatusView:
        # A stopping service must not look live to a new service instance.
        self._reject_if_shutting_down()
        inspection = inspect_config(self._config_path)
        cfg: Config | None = None
        if inspection.state == "ok":
            try:
                cfg = load_config(self._config_path)
                self._activate_config_model(cfg)
            except ConfigError as exc:
                inspection = type(inspection)(
                    state="invalid",
                    path=inspection.path,
                    revision=inspection.revision,
                    roots=inspection.roots,
                    message=f"invalid config: {_safe_message(exc)}",
                )

        if cfg is None:
            index = _IndexSnapshot(
                state="error",
                db_path=Path(""),
                message="configuration is invalid; the index cannot be inspected",
            )
        else:
            try:
                index = self._inspect_index(cfg, _STATUS_BUSY_TIMEOUT_SECONDS)
            except _SqliteBusyError as exc:
                index = _IndexSnapshot(
                    state="error",
                    db_path=Path(cfg.db_path),
                    message=f"index inspection is busy: {_safe_message(exc)}",
                )
            except Exception as exc:  # noqa: BLE001 - status must stay usable
                index = _IndexSnapshot(
                    state="error",
                    db_path=Path(cfg.db_path),
                    message=f"index inspection failed: {_safe_message(exc)}",
                )

        model_name = (
            cfg.embed_model if cfg is not None else (self._model_name or DEFAULT_EMBED_MODEL)
        )
        present, socket_path = self._daemon_status()
        config_state = inspection.state
        return StatusView(
            index=index.as_dict(),
            model=self._model_snapshot(model_name),
            daemon={
                "present": present,
                "socket": None if socket_path is None else str(socket_path),
            },
            config={
                "state": config_state,
                "path": str(inspection.path),
                "revision": inspection.revision,
                "roots": inspection.roots,
                "message": inspection.message,
            },
            activity=self._activity(),
            service=dict(self._service_status),
        )

    # -- index inspection --------------------------------------------------

    def _inspect_index(self, cfg: Config, busy_timeout: float) -> _IndexSnapshot:
        from .store import (
            SCHEMA_VERSION,
            VECTOR_QUANTS,
            SchemaVersionError,
            validate_schema_version,
        )

        db_path = Path(cfg.db_path).expanduser()
        if not db_path.exists():
            return _IndexSnapshot(
                state="missing",
                db_path=db_path,
                documents=0,
                chunks=0,
                size_bytes=0,
                last_error_count=0,
            )

        try:
            connection = sqlite3.connect(
                f"{db_path.resolve().as_uri()}?mode=ro",
                uri=True,
                timeout=max(0.0, float(busy_timeout)),
            )
        except sqlite3.OperationalError as exc:
            if _is_busy_error(exc):
                raise _SqliteBusyError(_safe_message(exc)) from exc
            return _IndexSnapshot(
                state="error",
                db_path=db_path,
                message=f"cannot open index: {_safe_message(exc)}",
            )
        except sqlite3.Error as exc:
            return _IndexSnapshot(
                state="error",
                db_path=db_path,
                message=f"cannot open index: {_safe_message(exc)}",
            )

        try:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            has_data = False
            if "documents" in tables or "chunks" in tables:
                checks: list[str] = []
                if "documents" in tables:
                    checks.append("EXISTS(SELECT 1 FROM documents)")
                if "chunks" in tables:
                    checks.append("EXISTS(SELECT 1 FROM chunks)")
                row = connection.execute("SELECT " + " OR ".join(checks)).fetchone()
                has_data = bool(row[0]) if row else False

            documents = (
                int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
                if "documents" in tables
                else 0
            )
            chunks = (
                int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
                if "chunks" in tables
                else 0
            )
            meta: dict[str, str] | None = None
            if "meta" in tables:
                meta = {
                    str(key): str(value)
                    for key, value in connection.execute("SELECT key, value FROM meta").fetchall()
                }
            size_bytes = _db_size_bytes(db_path)
        except sqlite3.OperationalError as exc:
            if _is_busy_error(exc):
                raise _SqliteBusyError(_safe_message(exc)) from exc
            return _IndexSnapshot(
                state="error",
                db_path=db_path,
                size_bytes=_db_size_bytes(db_path),
                message=f"cannot inspect index: {_safe_message(exc)}",
            )
        except sqlite3.Error as exc:
            return _IndexSnapshot(
                state="error",
                db_path=db_path,
                size_bytes=_db_size_bytes(db_path),
                message=f"cannot inspect index: {_safe_message(exc)}",
            )
        finally:
            connection.close()

        present_required = _REQUIRED_INDEX_TABLES & tables
        missing_tables = sorted(_REQUIRED_INDEX_TABLES - tables)
        if missing_tables and present_required:
            # Data without model metadata is the contract's incompatible case,
            # even when other structural tables are also missing. Every other
            # partial required schema is damaged/corrupt, not an empty index.
            if has_data and "meta" not in tables:
                return _IndexSnapshot(
                    state="incompatible",
                    db_path=db_path,
                    documents=documents,
                    chunks=chunks,
                    size_bytes=size_bytes,
                    message="index has data but no embedding metadata",
                )
            return _IndexSnapshot(
                state="error",
                db_path=db_path,
                documents=documents,
                chunks=chunks,
                size_bytes=size_bytes,
                message=(
                    "index is missing required table(s): "
                    + ", ".join(missing_tables)
                    + "; run `seekr reindex`"
                ),
            )

        if meta is None:
            if has_data:
                return _IndexSnapshot(
                    state="incompatible",
                    db_path=db_path,
                    documents=documents,
                    chunks=chunks,
                    size_bytes=size_bytes,
                    message="index has data but no embedding metadata",
                )
            return _IndexSnapshot(
                state="empty",
                db_path=db_path,
                documents=documents,
                chunks=chunks,
                size_bytes=size_bytes,
            )

        try:
            schema_version = validate_schema_version(meta)
        except SchemaVersionError as exc:
            return _IndexSnapshot(
                state="incompatible",
                db_path=db_path,
                documents=documents,
                chunks=chunks,
                size_bytes=size_bytes,
                message=_safe_message(exc),
            )
        if schema_version < SCHEMA_VERSION:
            return _IndexSnapshot(
                state="incompatible",
                db_path=db_path,
                documents=documents,
                chunks=chunks,
                size_bytes=size_bytes,
                schema_version=schema_version,
                message=(
                    f"index uses schema version {schema_version}, older than supported "
                    f"version {SCHEMA_VERSION}; run `seekr index` once to upgrade it"
                ),
            )

        try:
            last_error_count = int(meta.get("last_error_count", "0"))
        except (TypeError, ValueError):
            last_error_count = None

        stored_model = meta.get("embed_model")
        stored_dim = meta.get("dim")
        if stored_model is None and stored_dim is None:
            if has_data:
                return _IndexSnapshot(
                    state="incompatible",
                    db_path=db_path,
                    documents=documents,
                    chunks=chunks,
                    size_bytes=size_bytes,
                    last_error_count=last_error_count,
                    schema_version=schema_version,
                    message="index has data but no recorded embedding model",
                )
            return _IndexSnapshot(
                state="empty",
                db_path=db_path,
                documents=documents,
                chunks=chunks,
                size_bytes=size_bytes,
                last_error_count=last_error_count,
                schema_version=schema_version,
            )
        if stored_model is None or stored_dim is None:
            return _IndexSnapshot(
                state="incompatible",
                db_path=db_path,
                documents=documents,
                chunks=chunks,
                size_bytes=size_bytes,
                last_error_count=last_error_count,
                schema_version=schema_version,
                message="index has incomplete embedding metadata",
            )
        try:
            stored_dim_value = int(stored_dim)
        except (TypeError, ValueError):
            stored_dim_value = 0
        stored_quant = meta.get("vector_quant", "float32")
        if (
            not stored_model.strip()
            or stored_dim_value <= 0
            or stored_model.strip() != cfg.embed_model
            or stored_quant not in VECTOR_QUANTS
        ):
            return _IndexSnapshot(
                state="incompatible",
                db_path=db_path,
                documents=documents,
                chunks=chunks,
                size_bytes=size_bytes,
                last_error_count=last_error_count,
                model=stored_model,
                dim=stored_dim_value or None,
                vector_quant=stored_quant,
                schema_version=schema_version,
                message=(
                    "index embedding model or quantization is incompatible with the "
                    f"configured model {cfg.embed_model!r}"
                ),
            )

        state = "ready" if has_data else "empty"
        return _IndexSnapshot(
            state=state,
            db_path=db_path,
            documents=documents,
            chunks=chunks,
            size_bytes=size_bytes,
            last_error_count=last_error_count,
            model=stored_model,
            dim=stored_dim_value,
            vector_quant=stored_quant,
            schema_version=schema_version,
        )

    def _preflight_search(self, cfg: Config) -> _IndexSnapshot:
        try:
            snapshot = self._inspect_index(cfg, self._read_busy_timeout)
        except _SqliteBusyError as exc:
            raise ApiError(409, "busy", f"index is busy: {_safe_message(exc)}") from exc
        if snapshot.state == "incompatible":
            raise ApiError(
                409,
                "incompatible_index",
                snapshot.message or "the configured model or index schema is incompatible",
            )
        if snapshot.state == "error":
            raise ApiError(
                503,
                "index_unavailable",
                snapshot.message or "the index database is unreadable",
            )
        return snapshot

    # -- roots -------------------------------------------------------------

    def roots(self) -> RootsView:
        try:
            snapshot = read_roots(self._config_path)
        except ApiError:
            raise
        except Exception as exc:
            raise ApiError(
                500, "internal_error", f"cannot read configured roots: {_safe_message(exc)}"
            ) from exc
        return RootsView(
            roots=snapshot.roots,
            config_path=str(snapshot.path),
            revision=snapshot.revision,
        )

    def add_root(self, path: str, revision: str) -> RootsView:
        self._reject_if_shutting_down()
        with self._state_lock:
            if self._lane_kind is not None:
                raise ApiError(409, "busy", "another engine operation is already running")
            self._root_writers += 1
        try:
            with self._config_write_lock:
                snapshot = config_add_root(self._config_path, path, revision)
        except ApiError:
            raise
        except Exception as exc:
            raise ApiError(
                500, "internal_error", f"cannot add the root: {_safe_message(exc)}"
            ) from exc
        finally:
            with self._state_lock:
                self._root_writers = max(0, self._root_writers - 1)
                self._lane_free.notify_all()

        self._notify_roots_changed()
        return RootsView(
            roots=snapshot.roots,
            config_path=str(snapshot.path),
            revision=snapshot.revision,
            daemon_restart_required=False,
            warning=None,
        )

    def remove_root(self, path: str, revision: str) -> RootsView:
        self._reject_if_shutting_down()
        self._begin_lane("index")
        try:
            if self.daemon_present():
                raise ApiError(
                    409,
                    "daemon_active",
                    "a seekr daemon is running and maintains the index; stop the running "
                    "`seekr daemon` process before removing a folder",
                )
            cfg = self._load_config()
            db_path = Path(cfg.db_path).expanduser()
            index_snapshot: _IndexSnapshot | None = None
            if db_path.exists():
                try:
                    index_snapshot = self._inspect_index(cfg, self._read_busy_timeout)
                except _SqliteBusyError as exc:
                    raise ApiError(
                        409, "busy", f"index is busy: {_safe_message(exc)}"
                    ) from exc

            lock: Any = None
            if index_snapshot is not None and index_snapshot.state == "ready":
                lock = self._acquire_lifecycle_lock(
                    cfg.db_path, "index is in use; could not remove folder"
                )
            try:
                try:
                    with self._config_write_lock:
                        snapshot = config_remove_root(self._config_path, path, revision)
                except ApiError:
                    raise
                except Exception as exc:
                    raise ApiError(
                        500, "internal_error", f"cannot remove the root: {_safe_message(exc)}"
                    ) from exc

                prefixes = self._removed_root_prefixes(path, cfg.roots)
                if (
                    index_snapshot is not None
                    and index_snapshot.state == "ready"
                    and index_snapshot.dim is not None
                    and prefixes
                ):
                    try:
                        self._with_opened_store(
                            cfg,
                            index_snapshot,
                            read_only=False,
                            fn=lambda store: [
                                store.delete_documents_under(prefix)
                                for prefix in prefixes
                            ],
                        )
                    except Exception as exc:
                        raise self._map_index_read_error(exc) from exc
            finally:
                if lock is not None:
                    with suppress(Exception):
                        lock.release()
            # Still holding the lane: a watcher action waiting for it must see
            # the new roots, or it could re-index a file under the removed
            # folder.  The callback only swaps observers and never takes the
            # lane, so this cannot deadlock.
            self._notify_roots_changed()
        finally:
            self._end_lane()

        return RootsView(
            roots=snapshot.roots,
            config_path=str(snapshot.path),
            revision=snapshot.revision,
        )

    def _notify_roots_changed(self) -> None:
        callback = self._on_roots_changed
        if callback is None:
            return
        try:
            callback()
        except Exception as exc:  # noqa: BLE001 - the config change already succeeded
            _LOGGER.warning(
                "event=web_roots_changed_callback_failed error=%r", _safe_message(exc)
            )

    @staticmethod
    def _removed_root_prefixes(path: str, configured_roots: Sequence[Path]) -> list[str]:
        prefixes: list[str] = []
        try:
            canonical = str(Path(path).expanduser().resolve(strict=False))
        except (OSError, RuntimeError, ValueError):
            canonical = ""
        try:
            expanded = str(Path(path).expanduser())
        except (OSError, RuntimeError, ValueError):
            expanded = ""
        for candidate in (canonical, expanded):
            if candidate and candidate != "/" and candidate not in prefixes:
                prefixes.append(candidate)
        if canonical:
            for existing in configured_roots:
                try:
                    existing_resolved = str(existing.expanduser().resolve(strict=False))
                except (OSError, RuntimeError, ValueError):
                    continue
                if pathnorm.same_path(existing_resolved, canonical):
                    exp = str(existing.expanduser())
                    if exp and exp != "/" and exp not in prefixes:
                        prefixes.append(exp)
        return prefixes

    # -- course, preview, open ---------------------------------------------

    def _acquire_lifecycle_lock(self, db_path: Path, message_prefix: str) -> Any:
        from .lifecycle import LifecycleLock, LifecycleLockError

        deadline = time.monotonic() + self._lock_wait_seconds
        while True:
            candidate = LifecycleLock(db_path, shared=True, blocking=False)
            try:
                candidate.acquire()
            except LifecycleLockError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ApiError(
                        409,
                        "busy",
                        f"{message_prefix}: {_safe_message(exc)}",
                    ) from exc
                time.sleep(min(0.05, remaining))
            else:
                return candidate

    def _with_opened_store(
        self,
        cfg: Config,
        snapshot: _IndexSnapshot,
        *,
        read_only: bool,
        fn: Callable[[Any], Any],
    ) -> Any:
        if snapshot.dim is None or snapshot.vector_quant is None:
            raise ApiError(503, "index_unavailable", "index metadata could not be read")
        from .store import Store

        store_factory = self._store_factory or Store
        kwargs: dict[str, Any] = {"vector_quant": snapshot.vector_quant}
        if read_only:
            kwargs["read_only"] = True
        store: object | None = None
        try:
            try:
                store = store_factory(
                    snapshot.db_path,
                    cfg.embed_model,
                    snapshot.dim,
                    **kwargs,
                )
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
                if _is_busy_error(exc):
                    raise ApiError(409, "busy", f"index is busy: {_safe_message(exc)}") from exc
                raise
            return fn(store)
        finally:
            if store is not None:
                with suppress(Exception):
                    store.close()  # type: ignore[attr-defined]

    def _with_read_only_store(
        self,
        cfg: Config,
        snapshot: _IndexSnapshot,
        fn: Callable[[Any], Any],
    ) -> Any:
        lock = self._acquire_lifecycle_lock(
            cfg.db_path,
            "index is busy; could not take a lifecycle lock",
        )
        try:
            return self._with_opened_store(cfg, snapshot, read_only=True, fn=fn)
        finally:
            with suppress(Exception):
                lock.release()

    @staticmethod
    def _map_index_read_error(exc: BaseException) -> ApiError:
        from .lifecycle import LifecycleError
        from .store import ModelMismatchError, SchemaVersionError

        if isinstance(exc, ApiError):
            return exc
        if isinstance(exc, (ModelMismatchError, SchemaVersionError)):
            return ApiError(409, "incompatible_index", _safe_message(exc))
        if isinstance(exc, LifecycleError) or _is_busy_error(exc):
            return ApiError(409, "busy", f"index is busy: {_safe_message(exc)}")
        if isinstance(exc, (sqlite3.Error, OSError)):
            return ApiError(
                503, "index_unavailable", f"cannot read the index: {_safe_message(exc)}"
            )
        return ApiError(500, "internal_error", f"index operation failed: {_safe_message(exc)}")

    def course(self) -> CourseView:
        self._reject_if_shutting_down()
        cfg = self._load_config()
        snapshot = self._preflight_search(cfg)
        if snapshot.state in ("missing", "empty"):
            return CourseView(modules=())
        try:
            rows = self._with_read_only_store(
                cfg, snapshot, lambda store: store.course_summary()
            )
        except Exception as exc:
            raise self._map_index_read_error(exc) from exc

        by_module: dict[str, tuple[set[int], int]] = {}
        for module, week, file_count in rows:
            if module is None:
                continue
            weeks, total_files = by_module.setdefault(module, (set(), 0))
            if week is not None:
                weeks.add(int(week))
            by_module[module] = (weeks, total_files + int(file_count))

        modules = tuple(
            CourseModuleDTO(
                name=name,
                display=display_module(name) or name,
                weeks=tuple(sorted(weeks)),
                files=total_files,
            )
            for name, (weeks, total_files) in sorted(
                by_module.items(),
                key=lambda item: (item[0].casefold(), item[0]),
            )
        )
        return CourseView(modules=modules)

    def _fetch_chunk(self, chunk_id: int) -> Any:
        self._reject_if_shutting_down()
        if type(chunk_id) is not int or chunk_id <= 0:
            raise ApiError(
                400,
                "invalid_request",
                "chunk must be a positive integer",
                {"chunk": "chunk must be a positive integer"},
            )
        if chunk_id > _SQLITE_INT_MAX:
            raise ApiError(404, "not_found", f"chunk {chunk_id} not found")
        cfg = self._load_config()
        snapshot = self._preflight_search(cfg)
        if snapshot.state in ("missing", "empty"):
            raise ApiError(404, "not_found", f"chunk {chunk_id} not found")
        try:
            chunks = self._with_read_only_store(
                cfg, snapshot, lambda store: store.fetch_chunks([chunk_id])
            )
        except Exception as exc:
            raise self._map_index_read_error(exc) from exc
        row = chunks.get(chunk_id)
        if row is None:
            raise ApiError(404, "not_found", f"chunk {chunk_id} not found")
        return row

    def preview(self, chunk_id: int) -> PreviewView:
        row = self._fetch_chunk(chunk_id)
        is_wsl = platform_info.is_wsl()
        mounts = platform_info.read_mounts() if is_wsl else ()
        win = platform_info.windows_path(row.path, mounts) if is_wsl else None
        location = format_location(row.path, row.locator, windows_path=win)
        page_image = False
        if row.kind == "pdf":
            try:
                page_image = Path(row.path).is_file()
            except OSError:
                page_image = False
        return PreviewView(
            chunk=int(row.id),
            kind=str(row.kind),
            path=str(row.path),
            locator=str(row.locator),
            location=location,
            text=str(row.text),
            page_image=page_image,
        )

    @staticmethod
    def _parse_pdf_page_index(locator: str) -> int:
        text = locator.strip()
        if not text:
            return 0
        match = _PDF_PAGE_LOCATOR_RE.fullmatch(text)
        if match is None:
            raise ApiError(404, "not_found", "PDF chunk has no valid page locator")
        page_num = int(match.group(1))
        if page_num < 1:
            raise ApiError(404, "not_found", "PDF page number out of range")
        return page_num - 1

    def preview_page_png(self, chunk_id: int) -> bytes:
        row = self._fetch_chunk(chunk_id)
        if row.kind != "pdf":
            raise ApiError(
                404, "not_found", "page preview is only available for PDF chunks"
            )
        page_index = self._parse_pdf_page_index(str(row.locator))
        pdf_path = Path(str(row.path))
        try:
            exists = pdf_path.is_file()
        except OSError:
            exists = False
        if not exists:
            raise ApiError(404, "not_found", "PDF file is not available on disk")

        try:
            import pymupdf
        except ImportError:
            try:
                import fitz as pymupdf  # type: ignore[no-redef]
            except ImportError as exc:
                raise ApiError(
                    500, "internal_error", "PyMuPDF is required to render PDF pages"
                ) from exc

        try:
            doc = pymupdf.open(pdf_path)
        except FileNotFoundError as exc:
            raise ApiError(
                404, "not_found", "PDF file is not available on disk"
            ) from exc
        except Exception as exc:
            raise ApiError(404, "not_found", "PDF file cannot be opened") from exc

        try:
            if getattr(doc, "needs_pass", False):
                raise ApiError(404, "not_found", "encrypted PDF cannot be previewed")
            page_count = len(doc)
            if page_index < 0 or page_index >= page_count:
                raise ApiError(404, "not_found", "PDF page number out of range")
            page = doc[page_index]
            width = float(page.rect.width)
            zoom = _DEFAULT_PREVIEW_ZOOM
            if width > 0 and width * zoom > _MAX_PREVIEW_PNG_WIDTH:
                zoom = min(_DEFAULT_PREVIEW_ZOOM, _MAX_PREVIEW_PNG_WIDTH / width)
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
            if pixmap.width > int(_MAX_PREVIEW_PNG_WIDTH) and width > 0:
                adjusted = (_MAX_PREVIEW_PNG_WIDTH - 0.5) / width
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(adjusted, adjusted))
            return bytes(pixmap.tobytes("png"))
        except ApiError:
            raise
        except Exception as exc:
            raise ApiError(404, "not_found", "PDF page could not be rendered") from exc
        finally:
            doc.close()

    def open_chunk(self, chunk_id: int) -> None:
        from .opener import Argv, OpenFailed, open_target, run_open

        row = self._fetch_chunk(chunk_id)
        is_wsl = platform_info.is_wsl()
        mounts = platform_info.read_mounts() if is_wsl else ()

        def to_windows(path_value: str) -> str | None:
            return platform_info.windows_path(
                path_value, mounts
            ) or platform_info.wslpath_windows(path_value)

        try:
            action = open_target(
                str(row.path),
                str(row.locator),
                str(row.kind),
                env=os.environ,
                which=shutil.which,
                wsl=is_wsl,
                to_windows=to_windows,
            )
        except ValueError as exc:
            raise ApiError(500, "internal_error", _safe_message(exc)) from exc

        name = action.argv[0] if isinstance(action, Argv) else "startfile"
        try:
            run_open(action, wait=False)
        except FileNotFoundError as exc:
            raise ApiError(
                500,
                "internal_error",
                f"cannot run {name!r}: executable not found",
            ) from exc
        except OpenFailed as exc:
            raise ApiError(500, "internal_error", str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise ApiError(
                500,
                "internal_error",
                f"cannot run {name!r}: {_safe_message(exc)}",
            ) from exc

    # -- search ------------------------------------------------------------

    @staticmethod
    def _hit_dto(
        hit: object,
        *,
        is_wsl: bool = False,
        mounts: Sequence[platform_info.Mount] = (),
        explain: bool = False,
    ) -> HitDTO:
        raw_module = getattr(hit, "module", None)
        raw_week = getattr(hit, "week", None)
        raw_highlights = getattr(hit, "highlights", ()) or ()
        path = str(hit.path)  # type: ignore[attr-defined]
        locator = str(hit.locator)  # type: ignore[attr-defined]
        win = platform_info.windows_path(path, mounts) if is_wsl else None
        return HitDTO(
            chunk_id=int(hit.chunk_id),  # type: ignore[attr-defined]
            path=path,
            locator=locator,
            loc_start=float(hit.loc_start),  # type: ignore[attr-defined]
            snippet=str(hit.snippet),  # type: ignore[attr-defined]
            score=float(hit.score),  # type: ignore[attr-defined]
            kind=str(hit.kind),  # type: ignore[attr-defined]
            module=str(raw_module) if raw_module is not None else None,
            week=int(raw_week) if raw_week is not None else None,
            highlights=tuple((int(a), int(b)) for a, b in raw_highlights),
            location=format_location(path, locator, windows_path=win),
            posix_location=format_location(path, locator, windows_path=None),
            explain=_explain_payload(getattr(hit, "explain", None)) if explain else None,
        )

    def _search_result(
        self,
        hits: list[object],
        started: float,
        *,
        spread_hits: list[object] | None = None,
        explain: bool = False,
    ) -> SearchResult:
        took_ms = max(0.0, (time.monotonic() - started) * 1000.0)
        spread_payload: tuple[dict[str, object], ...] | None = None
        if spread_hits is not None:
            from .spread import compute_spread, spread_to_dict

            groups = compute_spread(spread_hits)  # type: ignore[arg-type]
            spread_payload = tuple(spread_to_dict(group) for group in groups)
        is_wsl = platform_info.is_wsl()
        mounts = platform_info.read_mounts() if is_wsl else ()
        return SearchResult(
            hits=tuple(
                self._hit_dto(hit, is_wsl=is_wsl, mounts=mounts, explain=explain)
                for hit in hits
            ),
            took_ms=took_ms,
            spread=spread_payload,
        )

    @staticmethod
    def _spread_request(request: SearchRequest) -> SearchRequest:
        from .spread import SPREAD_CANDIDATES

        return replace(
            request, k=SPREAD_CANDIDATES, per_file=2, rerank=False, explain=False
        )

    @staticmethod
    def _optional_filter_kwargs(request: SearchRequest) -> dict[str, object]:
        extra: dict[str, object] = {}
        if request.module is not None:
            extra["module"] = request.module
        if request.week is not None:
            extra["week"] = request.week
        return extra

    def _call_daemon_search(self, cfg: Config, request: SearchRequest) -> list[object] | None:
        extra = self._optional_filter_kwargs(request)
        if self._daemon_search is not None:
            return self._daemon_search(
                request.query,
                k=request.k,
                mode=request.mode,
                per_file=request.per_file,
                rerank=request.rerank,
                path_prefix=request.path,
                explain=request.explain,
                kind=request.kind,
                since=request.since,
                cfg=cfg,
                **extra,
            )
        from .cli_search import try_daemon_hits

        return try_daemon_hits(
            request.query,
            k=request.k,
            mode=request.mode,
            per_file=request.per_file,
            rerank=request.rerank,
            path_prefix=request.path,
            explain=request.explain,
            kind=request.kind,
            since=request.since,
            cfg=cfg,
            **extra,
        )

    def _reranker_for(self, cfg: Config) -> Any:
        """Return the warm reranker for ``cfg``, building it once under a lock."""
        key = (cfg.rerank_model, cfg.embed_backend)
        with self._reranker_lock:
            reranker = self._rerankers.get(key)
            if reranker is None:
                from .cli_search import _build_reranker

                reranker = _build_reranker(cfg)
                self._rerankers[key] = reranker
            return reranker

    def _call_local_search(
        self, cfg: Config, request: SearchRequest, embedder: Embedder
    ) -> list[object]:
        kwargs: dict[str, Any] = {
            "k": request.k,
            "mode": request.mode,
            "per_file": request.per_file,
            "rerank": request.rerank,
            "explain": request.explain,
            "path_prefix": request.path,
            "kind": request.kind,
            "since": request.since,
            "cfg": cfg,
            "embedder": embedder,
            "use_daemon": False,
            **self._optional_filter_kwargs(request),
        }
        if request.rerank and self._local_search is None:
            kwargs["reranker"] = self._reranker_for(cfg)
        if self._local_search is not None:
            return self._local_search(request.query, **kwargs)
        from .cli_search import get_hits

        return list(get_hits(request.query, **kwargs))

    def _map_search_error(self, exc: BaseException) -> ApiError:
        from .cli_search import SearchError
        from .client import DaemonError, DaemonMismatchError
        from .lifecycle import LifecycleError
        from .store import ModelMismatchError, SchemaVersionError

        if isinstance(exc, ApiError):
            return exc
        if isinstance(exc, DaemonMismatchError):
            return ApiError(409, "busy", "daemon configuration changed; retry the search")
        if isinstance(exc, DaemonError):
            message = _safe_message(exc)
            if _is_module_filter_error(message):
                return ApiError(400, "invalid_request", message, {"module": message})
            if _is_week_filter_error(message):
                return ApiError(400, "invalid_request", message, {"week": message})
            return ApiError(503, "daemon_failure", f"daemon search failed: {message}")
        if isinstance(exc, (ModelMismatchError, SchemaVersionError)):
            return ApiError(409, "incompatible_index", _safe_message(exc))
        if isinstance(exc, LifecycleError):
            return ApiError(409, "busy", f"index is in use: {_safe_message(exc)}")
        if _is_busy_error(exc):
            return ApiError(409, "busy", f"index is busy: {_safe_message(exc)}")
        if isinstance(exc, ValueError):
            message = _safe_message(exc)
            if _is_module_filter_error(message):
                return ApiError(400, "invalid_request", message, {"module": message})
            if _is_week_filter_error(message):
                return ApiError(400, "invalid_request", message, {"week": message})
        if isinstance(exc, (SearchError, sqlite3.Error, OSError)):
            return ApiError(503, "index_unavailable", f"cannot read the index: {_safe_message(exc)}")
        return ApiError(500, "internal_error", "search failed unexpectedly")

    def _keyword_search(
        self, cfg: Config, request: SearchRequest, snapshot: _IndexSnapshot
    ) -> tuple[list[object], list[object] | None]:
        from .cli_search import get_hits
        from .search import Searcher

        searcher_factory = self._searcher_factory or Searcher

        def run_with_store(store: object) -> tuple[list[object], list[object] | None]:
            keyword_embedder = self._cached_embedder(cfg.embed_model) or _KeywordEmbedder(
                cfg.embed_model
            )
            searcher = searcher_factory(store, keyword_embedder)

            def run_query(req: SearchRequest) -> list[object]:
                return list(
                    get_hits(
                        req.query,
                        k=req.k,
                        mode="keyword",
                        per_file=req.per_file,
                        rerank=req.rerank,
                        explain=req.explain,
                        path_prefix=req.path,
                        kind=req.kind,
                        since=req.since,
                        cfg=cfg,
                        searcher=searcher,
                        use_daemon=False,
                        **self._optional_filter_kwargs(req),
                    )
                )

            hits = run_query(request)
            spread_hits: list[object] | None = None
            if request.spread:
                spread_hits = run_query(self._spread_request(request)) if hits else []
            return hits, spread_hits

        return self._with_read_only_store(cfg, snapshot, run_with_store)

    def _reject_if_downloading(self) -> None:
        with self._state_lock:
            if self._lane_kind == "download":
                raise ApiError(409, "busy", "a model download is running")

    def _search_embedder(self, cfg: Config) -> Embedder:
        """Return the embedder for a search without waiting on index writes.

        Searches read the index concurrently with index jobs and watcher
        writes (SQLite WAL readers).  Only the first model load is exclusive:
        it shows as the ``search`` lane when the lane is free, and otherwise
        waits only for the model-load lock.
        """
        self._reject_if_downloading()
        took_lane = False
        if self._cached_embedder(cfg.embed_model) is None:
            with self._state_lock:
                if self._lane_kind is None and not self._root_writers:
                    self._lane_kind = "search"
                    took_lane = True
        try:
            return self._load_model_for_lane(cfg)[0]
        finally:
            if took_lane:
                self._end_lane()

    def _hybrid_local_search(
        self, cfg: Config, request: SearchRequest
    ) -> tuple[list[object], list[object] | None]:
        embedder = self._search_embedder(cfg)
        hits = self._call_local_search(cfg, request, embedder)
        spread_hits: list[object] | None = None
        if request.spread:
            spread_hits = (
                self._call_local_search(cfg, self._spread_request(request), embedder)
                if hits
                else []
            )
        return hits, spread_hits

    @staticmethod
    def _validate_search_filters(request: SearchRequest) -> None:
        """Validate kind/since/module/week before any daemon or local search path runs."""
        from .filters import normalise_kind, parse_since, parse_week

        try:
            normalise_kind(request.kind)
        except (ValueError, TypeError, OverflowError, OSError) as exc:
            message = _safe_message(exc)
            raise ApiError(
                400,
                "invalid_request",
                f"invalid kind filter: {message}",
                {"kind": message},
            ) from exc

        try:
            since_mtime_ns = parse_since(request.since)
        except (ValueError, TypeError, OverflowError, OSError) as exc:
            message = _safe_message(exc)
            raise ApiError(
                400,
                "invalid_request",
                f"invalid since filter: {message}",
                {"since": message},
            ) from exc
        if since_mtime_ns is not None and not (
            _SQLITE_INT_MIN <= since_mtime_ns <= _SQLITE_INT_MAX
        ):
            message = "since filter is outside the supported timestamp range"
            raise ApiError(
                400,
                "invalid_request",
                message,
                {"since": message},
            )

        if request.module is not None and not isinstance(request.module, str):
            message = "module filter must be a string"
            raise ApiError(
                400,
                "invalid_request",
                f"invalid module filter: {message}",
                {"module": message},
            )

        try:
            parse_week(request.week)
        except (ValueError, TypeError) as exc:
            message = _safe_message(exc)
            raise ApiError(
                400,
                "invalid_request",
                f"invalid week filter: {message}",
                {"week": message},
            ) from exc

    def search(self, request: SearchRequest) -> SearchResult:
        self._reject_if_shutting_down()
        started = time.monotonic()
        if request.mode not in ("keyword", "vector", "hybrid"):
            raise ApiError(400, "invalid_request", "mode must be keyword, vector or hybrid")
        if type(request.spread) is not bool:
            raise ApiError(
                400,
                "invalid_request",
                "spread must be a boolean",
                {"spread": "spread must be a boolean"},
            )
        for flag in ("rerank", "explain"):
            if type(getattr(request, flag)) is not bool:
                raise ApiError(
                    400,
                    "invalid_request",
                    f"{flag} must be a boolean",
                    {flag: f"{flag} must be a boolean"},
                )
        if request.rerank and request.mode == "keyword":
            raise ApiError(
                400,
                "invalid_request",
                "rerank is not available for keyword mode",
                {"rerank": "rerank is not available for keyword mode"},
            )
        cfg = self._load_config()

        empty_spread: list[object] | None = [] if request.spread else None
        if not request.query.strip():
            return self._search_result([], started, spread_hits=empty_spread)

        self._validate_search_filters(request)

        present, _socket_path = self._daemon_status()
        if present:
            from .client import DaemonError, DaemonMismatchError

            try:
                daemon_hits = self._call_daemon_search(cfg, request)
            except DaemonMismatchError:
                daemon_hits = None
            except Exception as exc:
                raise self._map_search_error(exc) from exc
            if daemon_hits is not None:
                daemon_spread: list[object] | None = None
                if request.spread:
                    if daemon_hits:
                        try:
                            spread_result = self._call_daemon_search(
                                cfg, self._spread_request(request)
                            )
                            if spread_result is None:
                                raise DaemonError(
                                    "daemon became unavailable during spread search"
                                )
                            daemon_spread = list(spread_result)
                        except Exception as exc:
                            raise self._map_search_error(exc) from exc
                    else:
                        daemon_spread = []
                return self._search_result(
                    list(daemon_hits),
                    started,
                    spread_hits=daemon_spread,
                    explain=request.explain,
                )

        snapshot = self._preflight_search(cfg)
        if snapshot.state in ("missing", "empty"):
            return self._search_result([], started, spread_hits=empty_spread)

        if request.mode == "keyword":
            try:
                hits, spread_hits = self._keyword_search(cfg, request, snapshot)
            except Exception as exc:
                raise self._map_search_error(exc) from exc
            return self._search_result(
                hits, started, spread_hits=spread_hits, explain=request.explain
            )

        if self._model_state_for(cfg.embed_model) == "unavailable":
            raise ApiError(
                503,
                "model_unavailable",
                f"embedding model {cfg.embed_model!r} is unavailable; run an explicit model check",
            )
        try:
            hits, spread_hits = self._hybrid_local_search(cfg, request)
        except Exception as exc:
            raise self._map_search_error(exc) from exc
        return self._search_result(
            hits, started, spread_hits=spread_hits, explain=request.explain
        )

    def similar(self, request: SimilarRequest) -> SearchResult:
        self._reject_if_shutting_down()
        started = time.monotonic()
        if type(request.path) is not str or not pathnorm.is_absolute(request.path):
            raise ApiError(
                400, "invalid_request", "path must be an absolute path",
                {"path": "path must be an absolute path"},
            )
        for name, low, high in (
            ("k", LIMITS["k_min"], LIMITS["k_max"]),
            ("per_file", LIMITS["per_file_min"], LIMITS["per_file_max"]),
        ):
            value = getattr(request, name)
            if type(value) is not int or not low <= value <= high:
                message = f"{name} must be an integer from {low} to {high}"
                raise ApiError(400, "invalid_request", message, {name: message})
        self._validate_search_filters(request)  # type: ignore[arg-type]
        cfg = self._load_config()

        from .cli_search import get_similar_hits

        self._reject_if_downloading()
        try:
            hits = list(
                get_similar_hits(
                    request.path,
                    k=request.k,
                    per_file=request.per_file,
                    path_prefix=request.path_prefix,
                    kind=request.kind,
                    since=request.since,
                    module=request.module,
                    week=request.week,
                    cfg=cfg,
                    use_service=False,
                )
            )
        except Exception as exc:
            raise self._map_search_error(exc) from exc
        return self._search_result(hits, started)

    # -- jobs: model check -------------------------------------------------

    def _run_model_check_job(self, job_id: str, cfg: Config) -> None:
        state = "failed"
        report: dict[str, object] | None = None
        error: str | None = "model check stopped unexpectedly"
        try:
            self._set_model_loading(cfg.embed_model)
            embedder = self._build_embedder(cfg)
            dim = int(embedder.dim)
            if dim <= 0:
                raise ValueError("embedding model reported a non-positive dimension")
        except Exception as exc:  # noqa: BLE001 - the model backend is injected
            error = _safe_message(exc)
            self._set_model_unavailable(cfg.embed_model, error)
        else:
            self._cache_model_success(cfg.embed_model, embedder, dim)
            state = "succeeded"
            report = {"model": cfg.embed_model, "dim": dim}
            error = None
        finally:
            self._finish_job_and_release_lane(
                job_id, state, report=report, error=error
            )

    def start_model_check(self) -> str:
        self._reject_if_shutting_down()
        cfg = self._load_config()
        return self._start_job(
            "model_check", lambda job_id: self._run_model_check_job(job_id, cfg)
        )

    # -- jobs: index -------------------------------------------------------

    @staticmethod
    def _serialise_report(report: object) -> dict[str, object]:
        errors = list(getattr(report, "errors", []))
        warnings = list(getattr(report, "warnings", []))

        def diagnostics(items: list[tuple[str, str]]) -> list[dict[str, str]]:
            return [
                {"path": str(path), "message": str(message)}
                for path, message in items[:_DIAGNOSTIC_LIMIT]
            ]

        return {
            "added": int(getattr(report, "added", 0)),
            "updated": int(getattr(report, "updated", 0)),
            "unchanged": int(getattr(report, "unchanged", 0)),
            "removed": int(getattr(report, "removed", 0)),
            "skipped": int(getattr(report, "skipped", 0)),
            "errors": diagnostics(errors),
            "warnings": diagnostics(warnings),
            "error_total": len(errors),
            "warning_total": len(warnings),
            "errors_truncated": len(errors) > _DIAGNOSTIC_LIMIT,
            "warnings_truncated": len(warnings) > _DIAGNOSTIC_LIMIT,
        }

    def _run_index_job(
        self, job_id: str, cfg: Config, targets: Sequence[Path] | None = None
    ) -> None:
        store: object | None = None
        lock: object | None = None
        state = "failed"
        report_payload: dict[str, object] | None = None
        error: str | None = "index job stopped unexpectedly"
        try:
            embedder, dim = self._load_model_for_lane(cfg)
            from .indexer import Indexer
            from .store import Store

            lock = self._acquire_lifecycle_lock(
                cfg.db_path, "index is in use; could not start indexing"
            )

            store_factory = self._store_factory or Store
            indexer_factory = self._indexer_factory or Indexer
            store = store_factory(
                cfg.db_path,
                cfg.embed_model,
                dim,
                vector_quant=cfg.vector_quant,
            )
            indexer = indexer_factory(store, embedder, cfg)
            report = indexer.sync(list(cfg.roots if targets is None else targets))
            report_payload = self._serialise_report(report)
            state = "partial" if getattr(report, "errors", []) else "succeeded"
            error = None
        except ApiError as exc:
            error = exc.message
        except Exception as exc:  # noqa: BLE001 - report a safe job failure
            error = _safe_message(exc)
        finally:
            try:
                if store is not None:
                    with suppress(Exception):
                        store.close()
                if lock is not None:
                    with suppress(Exception):
                        lock.release()
            finally:
                # Keep the lane busy until Store/lock cleanup is finished, then
                # publish the terminal job and release the lane atomically.
                self._finish_job_and_release_lane(
                    job_id, state, report=report_payload, error=error
                )

    def start_index(self) -> str:
        self._reject_if_shutting_down()
        cfg = self._load_config()
        if not cfg.roots:
            raise ApiError(400, "no_roots", "no roots are configured; add a folder first")
        return self._start_index_job(cfg, None)

    def start_index_paths(self, paths: list[str]) -> str:
        """Index specific files or directories (not necessarily configured roots)."""
        self._reject_if_shutting_down()
        if not isinstance(paths, list) or not paths:
            raise ApiError(
                400, "invalid_request", "paths must be a non-empty list",
                {"paths": "paths must be a non-empty list"},
            )
        targets: list[Path] = []
        for raw in paths:
            if type(raw) is not str or not raw.strip():
                message = "each path must be a non-empty string"
                raise ApiError(400, "invalid_request", message, {"paths": message})
            if not pathnorm.is_absolute(raw):
                message = "each path must be an absolute path"
                raise ApiError(400, "invalid_request", message, {"paths": message})
            target = Path(raw)
            if not target.exists():
                message = f"path does not exist: {raw}"
                raise ApiError(400, "invalid_request", message, {"paths": message})
            targets.append(target)
        return self._start_index_job(self._load_config(), targets)

    def _start_index_job(self, cfg: Config, targets: list[Path] | None) -> str:
        if self.daemon_present():
            raise ApiError(
                409,
                "daemon_active",
                "a seekr daemon is running and maintains the index; stop the running "
                "`seekr daemon` process before starting a web index scan",
            )
        # Missing/empty indexes remain allowed; incompatible/unreadable ones
        # have already been refused by the shared preflight.
        snapshot = self._preflight_search(cfg)
        if (
            snapshot.vector_quant is not None
            and snapshot.vector_quant != cfg.vector_quant
        ):
            raise ApiError(
                409,
                "incompatible_index",
                "index stores vector quantization "
                f"{snapshot.vector_quant!r}, but the configuration uses "
                f"{cfg.vector_quant!r}; run `seekr reindex` to rebuild it",
            )
        if self._model_state_for(cfg.embed_model) == "unavailable":
            raise ApiError(
                503,
                "model_unavailable",
                f"embedding model {cfg.embed_model!r} is unavailable; run an explicit model check",
            )
        return self._start_job(
            "index", lambda job_id: self._run_index_job(job_id, cfg, targets)
        )

    # -- jobs: reindex -----------------------------------------------------

    def start_reindex(self, force: bool = False, allow_empty: bool = False) -> str:
        """Rebuild the index in a fresh database and swap it in, as a job.

        The job is an ``index`` operation holding the engine lane.  The swap
        itself runs inside ``reindex_runner`` (the service's pause), so no
        Store of this process is open on the live database during it.
        """
        self._reject_if_shutting_down()
        cfg = self._load_config()
        if not cfg.roots:
            raise ApiError(400, "no_roots", "no roots are configured; add a folder first")
        if self.daemon_present():
            raise ApiError(
                409,
                "daemon_active",
                "a seekr daemon is running and maintains the index; stop the running "
                "`seekr daemon` process before reindexing",
            )
        if self._model_state_for(cfg.embed_model) == "unavailable":
            raise ApiError(
                503,
                "model_unavailable",
                f"embedding model {cfg.embed_model!r} is unavailable; run an explicit model check",
            )
        return self._start_job(
            "index",
            lambda job_id: self._run_reindex_job(
                job_id, cfg, force=bool(force), allow_empty=bool(allow_empty)
            ),
        )

    def _run_reindex_job(
        self, job_id: str, cfg: Config, *, force: bool, allow_empty: bool
    ) -> None:
        state = "failed"
        report_payload: dict[str, object] | None = None
        error: str | None = "reindex job stopped unexpectedly"
        try:
            embedder, _dim = self._load_model_for_lane(cfg)
            from .lifecycle import rebuild_index

            def rebuild() -> object:
                return rebuild_index(
                    cfg,
                    embedder=embedder,
                    lock_timeout=_REINDEX_SWAP_WAIT_SECONDS,
                    force=force,
                    allow_empty=allow_empty,
                )

            runner = self._reindex_runner
            report = runner(rebuild) if runner is not None else rebuild()
            report_payload = self._serialise_report(report)
            state = "partial" if getattr(report, "errors", []) else "succeeded"
            error = None
        except ApiError as exc:
            error = exc.message
        except Exception as exc:  # noqa: BLE001 - report a safe job failure
            error = _safe_message(exc)
        finally:
            self._finish_job_and_release_lane(
                job_id, state, report=report_payload, error=error
            )

    # -- jobs: download ----------------------------------------------------

    def _run_download_job(self, job_id: str, rerank: bool) -> None:
        state = "failed"
        report: dict[str, object] | None = None
        error: str | None = "model download stopped unexpectedly"
        try:
            cmd = [
                sys.executable,
                "-m",
                "seekr",
                "models",
                "pull",
                "--yes",
                "--grammars",
            ]
            if rerank:
                cmd.append("--rerank")
            env = dict(os.environ)
            for key in _OFFLINE_ENV_KEYS:
                env.pop(key, None)
            env["SEEKR_CONFIG"] = str(self._config_path)
            result = subprocess.run(
                cmd,
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
            )
            stdout_text = getattr(result, "stdout", None) or ""
            stderr_text = getattr(result, "stderr", None) or ""
            combined = "\n".join(part for part in (stdout_text, stderr_text) if part)
            tail_lines = combined.splitlines()[-_DOWNLOAD_OUTPUT_LINES:]
            returncode = int(getattr(result, "returncode", 1))
            report = {
                "rerank": rerank,
                "returncode": returncode,
                "lines": tail_lines,
            }
            if returncode == 0:
                state = "succeeded"
                error = None
                with self._state_lock:
                    if self._model_status == "unavailable":
                        self._model_status = "unchecked"
                        self._model_message = None
            else:
                last_line = next(
                    (line.strip() for line in reversed(tail_lines) if line.strip()),
                    None,
                )
                error = last_line or f"model download failed with exit code {returncode}"
        except Exception as exc:  # noqa: BLE001 - report a safe job failure
            error = _safe_message(exc)
        finally:
            self._finish_job_and_release_lane(
                job_id, state, report=report, error=error
            )

    def start_download(self, rerank: bool = False) -> str:
        if type(rerank) is not bool:
            raise ApiError(
                400,
                "invalid_request",
                "rerank must be a boolean",
                {"rerank": "rerank must be a boolean"},
            )
        self._reject_if_shutting_down()
        self._load_config()
        return self._start_job(
            "download",
            lambda job_id: self._run_download_job(job_id, rerank),
        )
