"""Incremental filesystem watching for the seekr service and daemon.

Events are debounced per path and executed by one worker thread, so indexing,
deletes, directory scans and moves are serialized through a single writer.
Moves between indexed paths are routed through :meth:`Indexer.move_path` rather
than a delete followed by an insert, which preserves vectors and embedding
work.

Watchdog is imported only when watching starts, so importing this module (and
the CLI) stays cheap and works on hosts without the optional dependency.
"""

from __future__ import annotations

import math
import os
import stat
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from . import pathnorm, platform_info
from .platform_info import Mount

__all__ = ["Watcher", "WatcherError"]

_IDLE_WAIT = 0.25
_EVENT_TYPES = frozenset({"created", "modified", "moved", "deleted"})


class WatcherError(RuntimeError):
    """The watcher could not be created, started or stopped safely."""


@dataclass(frozen=True)
class _Pending:
    key: tuple[str, ...]
    due: float
    seq: int
    kind: str
    src: Path
    dst: Path | None = None


def _decode_path(value: object) -> str:
    if isinstance(value, bytes):
        return os.fsdecode(value)
    return os.fspath(value)  # type: ignore[arg-type]


class _EventDispatch:
    """Duck-typed watchdog event handler; no watchdog import is required."""

    def __init__(self, watcher: Watcher) -> None:
        self._watcher = watcher

    def dispatch(self, event: object) -> None:
        try:
            event_type = getattr(event, "event_type", None)
            if not isinstance(event_type, str) or event_type not in _EVENT_TYPES:
                return
            raw_src = getattr(event, "src_path", None)
            if raw_src is None:
                return
            raw_dst = getattr(event, "dest_path", None)
            destination = _decode_path(raw_dst) if raw_dst else None
            self._watcher._handle_event(
                event_type,
                _decode_path(raw_src),
                destination,
                is_directory=bool(getattr(event, "is_directory", False)),
            )
        except Exception as exc:  # noqa: BLE001 - a bad event must not kill the observer
            self._watcher._record_error("cannot handle filesystem event", exc)


class Watcher:
    """Debounced watchdog observer with one serialized writer thread."""

    def __init__(
        self,
        indexer: Any,
        roots: list[Path],
        debounce_s: float = 1.0,
        *,
        observer_factory: Callable[[], Any] | None = None,
        poll_interval: float = 15.0,
        mounts: Sequence[Mount] | None = None,
        polling_observer_factory: Callable[[float], Any] | None = None,
        run_exclusive: Callable[[Callable[[], None]], None] | None = None,
    ) -> None:
        try:
            debounce = float(debounce_s)
        except (TypeError, ValueError) as exc:
            raise WatcherError("debounce_s must be a non-negative number") from exc
        if not math.isfinite(debounce) or debounce < 0:
            raise WatcherError("debounce_s must be a non-negative finite number")
        if isinstance(poll_interval, bool) or not isinstance(
            poll_interval, (int, float)
        ):
            raise WatcherError("poll_interval must be a positive finite number")
        poll_interval_s = float(poll_interval)
        if not math.isfinite(poll_interval_s) or poll_interval_s <= 0:
            raise WatcherError("poll_interval must be a positive finite number")

        self._indexer = indexer
        self._run_exclusive = run_exclusive
        self._mounts = None if mounts is None else tuple(mounts)
        self._configure_roots(roots)
        self._debounce_s = debounce
        self._poll_interval = poll_interval_s
        self._observer_factory = observer_factory
        self._polling_observer_factory = polling_observer_factory

        self._condition = threading.Condition()
        self._pending: dict[tuple[str, ...], _Pending] = {}
        self._sequence = 0
        self._paused = False
        self._stopping = False
        self._started = False

        self._worker: threading.Thread | None = None
        self._observer: Any | None = None
        self._polling_observer: Any | None = None
        self._handler: _EventDispatch | None = None
        # Serialises observer replacement (set_roots) against start/stop.
        self._observer_lock = threading.Lock()

        self._error_lock = threading.Lock()
        self._errors: list[tuple[str, str]] = []

    # -- lifecycle ---------------------------------------------------------

    def start(self, *, paused: bool = False) -> None:
        """Start the observer and writer.

        ``paused=True`` starts the observer while buffering events for a caller
        that is running an initial ``Indexer.sync`` first.  Call :meth:`resume`
        when that sync has finished.
        """
        with self._condition:
            if self._started:
                raise WatcherError("watcher is already started")
            self._paused = bool(paused)
            self._stopping = False
            self._started = True

        worker = threading.Thread(
            target=self._run,
            name="seekr-watcher",
            daemon=True,
        )
        self._worker = worker
        worker.start()

        handler = _EventDispatch(self)
        with self._observer_lock:
            try:
                observer, polling_observer = self._launch_observers(handler)
            except Exception as exc:
                self._abort_start(worker)
                if isinstance(exc, WatcherError):
                    raise
                raise WatcherError(f"cannot start file watcher: {exc}") from exc

            self._observer = observer
            self._polling_observer = polling_observer
            self._handler = handler

    def set_roots(self, roots: Sequence[Path]) -> None:
        """Replace the watched roots, rescheduling a running watcher at once.

        Pending events that fall outside the new roots are dropped.  Both
        observers are rebuilt because watchdog observers cannot be restarted
        and the native/polled split may change with the new roots.
        """
        with self._observer_lock:
            with self._condition:
                self._configure_roots(roots)
                for key, pending in list(self._pending.items()):
                    if self._inside_roots(pending.src) is None and (
                        pending.dst is None or self._inside_roots(pending.dst) is None
                    ):
                        del self._pending[key]
                running = self._started and not self._stopping
            handler = self._handler
            if not running or handler is None:
                return
            old = (self._observer, self._polling_observer)
            self._observer = None
            self._polling_observer = None
            try:
                self._observer, self._polling_observer = self._launch_observers(
                    handler
                )
            finally:
                for observer in old:
                    self._stop_observer(observer, timeout=5.0)

    def pause(self) -> None:
        """Stop applying events, buffering them until :meth:`resume`.

        An action already handed to ``run_exclusive`` still runs when it gets
        the engine lane; callers that hold the lane keep it out meanwhile.
        """
        with self._condition:
            self._paused = True
            self._condition.notify_all()

    def resume(self) -> None:
        """Apply events again after :meth:`pause` or ``start(paused=True)``."""
        with self._condition:
            self._paused = False
            self._condition.notify_all()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the observer and writer; safe to call more than once."""
        with self._condition:
            self._stopping = True
            self._pending.clear()
            self._condition.notify_all()

        with self._observer_lock:
            observer = self._observer
            polling_observer = self._polling_observer
            self._observer = None
            self._polling_observer = None
            self._handler = None
        self._stop_observer(observer, timeout=timeout)
        self._stop_observer(polling_observer, timeout=timeout)

        worker = self._worker
        self._worker = None
        if (
            worker is not None
            and worker is not threading.current_thread()
        ):
            worker.join(timeout=timeout)
            # Never leave the single writer alive when a caller may close the
            # Store immediately after stop().  At worst this waits for the one
            # indexing operation already in flight.
            while worker.is_alive():
                worker.join()

        with self._condition:
            self._started = False

    @property
    def polled_roots(self) -> tuple[Path, ...]:
        """Resolved watch roots scheduled on the polling observer."""
        return self._polled_roots

    @property
    def errors(self) -> list[tuple[str, str]]:
        with self._error_lock:
            return list(self._errors)

    # -- observer setup ----------------------------------------------------

    def _configure_roots(self, roots: Sequence[Path]) -> None:
        self._roots = tuple(Path(root).expanduser() for root in roots)
        self._resolved_roots = tuple(self._resolve(root) for root in self._roots)
        active_mounts = (
            tuple(platform_info.read_mounts()) if self._mounts is None else self._mounts
        )
        native_roots, polled_roots = platform_info.split_roots(
            self._roots, active_mounts
        )
        self._native_roots = tuple(native_roots)
        self._polled_roots = tuple(polled_roots)

    def _launch_observers(self, handler: _EventDispatch) -> tuple[Any, Any]:
        """Build, schedule and start the native and polling observers."""
        observer: Any | None = None
        polling_observer: Any | None = None
        started_ids: set[int] = set()
        try:
            if self._native_roots:
                observer = self._build_observer()
                self._schedule_watches(observer, handler, self._native_roots)
            if self._polled_roots:
                polling_observer = self._build_polling_observer()
                self._schedule_watches(polling_observer, handler, self._polled_roots)
            for candidate in (observer, polling_observer):
                if candidate is not None:
                    candidate.start()
                    started_ids.add(id(candidate))
        except Exception:
            for candidate in (observer, polling_observer):
                if candidate is not None:
                    self._stop_observer(
                        candidate,
                        timeout=5.0,
                        join=id(candidate) in started_ids,
                        context="cannot stop filesystem observer during startup",
                    )
            raise
        return observer, polling_observer

    def _build_observer(self) -> Any:
        if self._observer_factory is not None:
            return self._observer_factory()
        try:
            from watchdog.observers import Observer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise WatcherError(
                "watchdog is required for `seekr daemon`; install the watchdog package"
            ) from exc
        return Observer()

    def _build_polling_observer(self) -> Any:
        if self._polling_observer_factory is not None:
            return self._polling_observer_factory(self._poll_interval)
        try:
            from watchdog.observers.polling import PollingObserver
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise WatcherError(
                "watchdog is required for `seekr daemon`; install the watchdog package"
            ) from exc
        return PollingObserver(timeout=self._poll_interval)

    def _schedule_watches(
        self,
        observer: Any,
        handler: _EventDispatch,
        roots: Sequence[Path],
    ) -> None:
        watches: dict[str, bool] = {}
        for resolved in roots:
            try:
                root_stat = resolved.stat()
            except OSError as exc:
                self._record_error(
                    f"cannot watch missing or unreadable root {resolved}", exc
                )
                continue
            if stat.S_ISDIR(root_stat.st_mode):
                watch_path = resolved
                recursive = True
            elif stat.S_ISREG(root_stat.st_mode):
                watch_path = resolved.parent
                recursive = False
            else:
                self._record_error(f"cannot watch non-file root {resolved}", None)
                continue
            key = str(watch_path)
            watches[key] = watches.get(key, False) or recursive

        for watch_path, recursive in sorted(watches.items()):
            observer.schedule(handler, watch_path, recursive=recursive)

    def _stop_observer(
        self,
        observer: Any | None,
        *,
        timeout: float,
        join: bool = True,
        context: str = "cannot stop filesystem observer",
    ) -> None:
        if observer is None:
            return
        try:
            observer.stop()
        except Exception as exc:  # noqa: BLE001 - third-party observer
            self._record_error(context, exc)
        if not join:
            return
        try:
            observer.join(timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - third-party observer
            self._record_error("cannot join filesystem observer", exc)

    def _abort_start(self, worker: threading.Thread) -> None:
        with self._condition:
            self._stopping = True
            self._pending.clear()
            self._condition.notify_all()
        worker.join(timeout=5.0)
        with self._condition:
            self._started = False
        self._handler = None
        self._observer = None
        self._polling_observer = None

    # -- event intake ------------------------------------------------------

    def _handle_event(
        self,
        event_type: str,
        src_path: str | os.PathLike[str],
        dest_path: str | os.PathLike[str] | None = None,
        *,
        is_directory: bool = False,
    ) -> None:
        if not self._started:
            return

        source = self._inside_roots(src_path)
        if event_type == "moved":
            destination = (
                self._inside_roots(dest_path) if dest_path is not None else None
            )
            if source is None and destination is None:
                return
            if is_directory:
                if source is not None and destination is not None:
                    self._schedule("move_dir", source, destination)
                elif destination is not None:
                    self._schedule("scan_dir", destination)
                else:
                    assert source is not None
                    self._schedule("remove_dir", source)
            else:
                if source is not None and destination is not None:
                    self._schedule("move", source, destination)
                elif destination is not None:
                    self._schedule("index", destination)
                else:
                    assert source is not None
                    self._schedule("remove", source)
            return

        if source is None:
            return
        if event_type == "created":
            self._schedule("scan_dir" if is_directory else "index", source)
        elif event_type == "modified":
            # Directory mtimes change for every child event; only child paths
            # need work.  A modified file is re-indexed.
            if not is_directory:
                self._schedule("index", source)
        elif event_type == "deleted":
            self._schedule("remove_dir" if is_directory else "remove", source)

    def _schedule(self, kind: str, src: Path, dst: Path | None = None) -> None:
        with self._condition:
            if self._stopping:
                return
            moved_from = src
            if dst is not None and kind in ("move", "move_dir"):
                src = self._merge_pending_move(kind, src, dst)
            if kind == "move":
                # The move reconciles the old path.  A pending index of it
                # (macOS FSEvents reports a just-written file as modified
                # before its rename) would find it gone and drop its vectors
                # first, so the move would re-embed the destination.
                for old_path in {moved_from, src}:
                    self._pending.pop(("index", str(old_path)), None)
                    self._pending.pop(("remove", str(old_path)), None)
            # Move entries are keyed by (source, destination): keying only by
            # the destination lets a later move with the same target discard an
            # earlier move before its source is reconciled.
            if dst is None:
                key = (kind, str(src))
            else:
                key = (kind, str(src), str(dst))
            self._sequence += 1
            self._pending[key] = _Pending(
                key=key,
                due=time.monotonic() + self._debounce_s,
                seq=self._sequence,
                kind=kind,
                src=src,
                dst=dst,
            )
            self._condition.notify_all()

    def _merge_pending_move(self, kind: str, src: Path, dst: Path) -> Path:
        """Collapse a debounced A→B, B→C chain into one A→C move.

        The caller holds ``self._condition``.  A replacement for the same
        source path is also folded, so at most one move for a source family is
        pending.  Moves with different sources and the same destination remain
        distinct.
        """
        while True:
            for key, pending in list(self._pending.items()):
                if pending.kind != kind:
                    continue
                if pending.src == src or pending.dst == src:
                    if pending.dst == src:
                        src = pending.src
                    del self._pending[key]
                    break
            else:
                return src

    def _inside_roots(self, raw_path: object) -> Path | None:
        try:
            # Canonical form, as the indexer records it (NFC on macOS).
            resolved = Path(
                pathnorm.canonical_path(Path(_decode_path(raw_path)).expanduser())
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            return None
        # Case- and Unicode-form-insensitive on macOS and Windows; exact,
        # component-wise containment on Linux.
        target = str(resolved)
        for root in self._resolved_roots:
            if pathnorm.has_prefix(target, str(root)):
                return resolved
        return None

    # -- worker ------------------------------------------------------------

    def _run(self) -> None:
        while True:
            due: list[_Pending] = []
            with self._condition:
                while not due:
                    if self._stopping:
                        return
                    if self._paused:
                        self._condition.wait(_IDLE_WAIT)
                        continue
                    if not self._pending:
                        self._condition.wait(_IDLE_WAIT)
                        continue
                    now = time.monotonic()
                    due = sorted(
                        (
                            pending
                            for pending in self._pending.values()
                            if pending.due <= now
                        ),
                        key=lambda pending: (pending.due, pending.seq),
                    )
                    if due:
                        for pending in due:
                            self._pending.pop(pending.key, None)
                        break
                    next_due = min(
                        pending.due for pending in self._pending.values()
                    )
                    self._condition.wait(
                        max(0.001, min(next_due - now, _IDLE_WAIT))
                    )

            for pending in due:
                with self._condition:
                    if self._stopping:
                        return
                try:
                    if self._run_exclusive is None:
                        self._execute(pending)
                    else:
                        self._run_exclusive(partial(self._execute, pending))
                except Exception as exc:  # noqa: BLE001 - isolate every event
                    self._record_error(
                        f"cannot apply {pending.kind} for {pending.src}", exc
                    )

    def _execute(self, pending: _Pending) -> None:
        if pending.kind == "index":
            self._indexer.index_path(pending.src)
        elif pending.kind == "remove":
            self._indexer.remove_path(pending.src)
        elif pending.kind == "move":
            assert pending.dst is not None
            self._indexer.move_path(pending.src, pending.dst)
        elif pending.kind == "scan_dir":
            self._indexer.sync([pending.src])
        elif pending.kind == "remove_dir":
            self._remove_directory(pending.src)
        elif pending.kind == "move_dir":
            assert pending.dst is not None
            self._move_directory(pending.src, pending.dst)

    def _remove_directory(self, directory: Path) -> None:
        store = getattr(self._indexer, "store", None)
        list_paths = getattr(store, "list_paths", None)
        if not callable(list_paths):
            self._record_error(
                f"cannot remove indexed files below {directory}: store has no list_paths",
                None,
            )
            return
        try:
            indexed = list(list_paths(str(directory)))
        except Exception as exc:  # noqa: BLE001 - store is an injected backend
            self._record_error(f"cannot list indexed files below {directory}", exc)
            return
        for path in indexed:
            self._indexer.remove_path(Path(path))

    def _move_directory(self, source: Path, destination: Path) -> None:
        for file_path in self._walk_files(destination):
            try:
                relative = file_path.relative_to(destination)
            except ValueError:
                continue
            self._indexer.move_path(source / relative, file_path)
        # Anything left at the old path (deleted files, ignored files) must go.
        self._remove_directory(source)

    def _walk_files(self, directory: Path) -> list[Path]:
        files: list[Path] = []

        def onerror(exc: OSError) -> None:
            self._record_error(f"cannot read directory {directory}", exc)

        for dirpath, dirnames, filenames in os.walk(
            directory,
            topdown=True,
            onerror=onerror,
            followlinks=False,
        ):
            dirnames[:] = [
                name
                for name in dirnames
                if not os.path.islink(os.path.join(dirpath, name))
            ]
            for name in filenames:
                candidate = Path(dirpath) / name
                resolved = self._inside_roots(candidate)
                if resolved is not None:
                    files.append(resolved)
        files.sort(key=str)
        return files

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _resolve(path: Path) -> Path:
        return platform_info.resolve_root(path)

    def _record_error(self, message: str, exc: BaseException | None) -> None:
        detail = message
        if exc is not None:
            text = str(exc).strip()
            detail = f"{message}: {text or exc.__class__.__name__}"
        with self._error_lock:
            self._errors.append((detail, exc.__class__.__name__ if exc else ""))
            del self._errors[:-100]
