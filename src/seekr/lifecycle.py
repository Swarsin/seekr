"""Database lifecycle locking and crash-safe reindex rebuilds.

Every owner of a :class:`seekr.store.Store` connection that wants to
cooperate with ``seekr reindex`` should hold a shared lifecycle lock for the
whole lifetime of the store::

    from seekr.lifecycle import lifecycle_lock

    with lifecycle_lock(cfg.db_path, shared=True):
        store = Store(cfg.db_path, cfg.embed_model, embedder.dim)
        try:
            ... use the store ...
        finally:
            store.close()

A rebuild creates a fresh sibling database while holding a shared lock and
then upgrades to an exclusive lock only for the final atomic swap.  Existing
daemons or standalone commands that hold a shared lock therefore prevent the
swap instead of having their database replaced underneath them.
"""

from __future__ import annotations

import errno
import os
import sqlite3
import stat
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Self

from .config import Config
from .embed import Embedder, build_embedder
from .indexer import Indexer, SyncReport
from .store import Store

try:  # pragma: no cover - the target platform is Linux, but import stays safe.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

__all__ = [
    "LifecycleError",
    "LifecycleLock",
    "LifecycleLockError",
    "RebuildError",
    "lifecycle_lock",
    "lock_path_for",
    "rebuild_index",
]


class LifecycleError(RuntimeError):
    """Base class for lifecycle locking and rebuild failures."""


class LifecycleLockError(LifecycleError):
    """A lifecycle lock could not be acquired or upgraded."""


class RebuildError(LifecycleError):
    """A rebuild was refused or failed without replacing the live database."""


def lock_path_for(db_path: Path) -> Path:
    """Return the advisory lock file used for *db_path*."""
    path = Path(db_path)
    return path.with_name(path.name + ".lock")


class LifecycleLock:
    """A Linux ``flock`` advisory lock held on a sibling lock file."""

    def __init__(
        self, db_path: Path, shared: bool = True, blocking: bool = True
    ) -> None:
        self.db_path = Path(db_path)
        self.lock_path = lock_path_for(self.db_path)
        self.shared = bool(shared)
        self.blocking = bool(blocking)
        self._fd: int | None = None

    @property
    def acquired(self) -> bool:
        return self._fd is not None

    def acquire(self) -> LifecycleLock:
        """Acquire the lock, raising :class:`LifecycleLockError` when busy."""
        if self._fd is not None:
            return self
        if fcntl is None:  # pragma: no cover
            raise LifecycleLockError(
                "lifecycle locks require Linux advisory file locking (fcntl)"
            )

        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self.lock_path), os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as exc:
            raise LifecycleLockError(
                f"cannot open lifecycle lock {self.lock_path}: {exc}"
            ) from exc

        operation = fcntl.LOCK_SH if self.shared else fcntl.LOCK_EX
        if not self.blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(fd, operation)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise LifecycleLockError(self._busy_message()) from exc
            raise LifecycleLockError(
                f"cannot lock {self.lock_path}: {exc}"
            ) from exc

        self._fd = fd
        return self

    def upgrade(self, timeout: float = 0.0) -> None:
        """Atomically upgrade an acquired shared lock to an exclusive lock."""
        if self._fd is None:
            raise LifecycleLockError("lifecycle lock is not acquired")
        if not self.shared:
            return
        if fcntl is None:  # pragma: no cover
            raise LifecycleLockError(
                "lifecycle locks require Linux advisory file locking (fcntl)"
            )

        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise LifecycleLockError(
                        f"cannot upgrade lifecycle lock {self.lock_path}: {exc}"
                    ) from exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LifecycleLockError(
                        self._busy_message(mode="exclusive")
                    ) from exc
                time.sleep(min(0.05, remaining))
            else:
                self.shared = False
                return

    def release(self) -> None:
        """Release the lock; safe to call more than once."""
        fd = self._fd
        self._fd = None
        if fd is None:
            return
        if fcntl is not None:  # pragma: no branch
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        try:
            os.close(fd)
        except OSError:
            pass

    def _busy_message(self, mode: str | None = None) -> str:
        mode = mode or ("shared" if self.shared else "exclusive")
        return (
            f"index database is in use; could not take a {mode} lifecycle lock on "
            f"{self.lock_path}. Stop the seekr daemon or wait for other seekr "
            "processes, then retry."
        )

    def __enter__(self) -> Self:
        return self.acquire()

    def __exit__(self, *exc_info: object) -> bool:
        self.release()
        return False


@contextmanager
def lifecycle_lock(
    db_path: Path, shared: bool = True, blocking: bool = True
) -> Iterator[LifecycleLock]:
    """Hold a lifecycle lock for the duration of the ``with`` block."""
    lock = LifecycleLock(db_path, shared=shared, blocking=blocking)
    with lock:
        yield lock


def _error_text(exc: BaseException) -> str:
    message = str(exc).strip()
    return message or exc.__class__.__name__


def _validate_rebuild_roots(cfg: Config) -> None:
    if not cfg.roots:
        raise RebuildError(
            "refusing to rebuild: no roots configured; live index left untouched"
        )
    for raw_root in cfg.roots:
        root = Path(raw_root).expanduser()
        try:
            root_stat = root.stat()
        except OSError as exc:
            raise RebuildError(
                f"refusing to rebuild: root {root} is unavailable ({_error_text(exc)}); "
                "live index left untouched"
            ) from exc
        if stat.S_ISDIR(root_stat.st_mode):
            try:
                with os.scandir(root) as entries:
                    next(entries, None)
            except OSError as exc:
                raise RebuildError(
                    f"refusing to rebuild: root {root} is unreadable "
                    f"({_error_text(exc)}); live index left untouched"
                ) from exc
        elif not stat.S_ISREG(root_stat.st_mode):
            raise RebuildError(
                f"refusing to rebuild: root {root} is not a regular file or directory; "
                "live index left untouched"
            )


def _temporary_db_path(db_path: Path) -> Path:
    token = uuid.uuid4().hex
    return db_path.parent / f".{db_path.name}.rebuild-{os.getpid()}-{token}.db"


def _existing_document_count(db_path: Path) -> int | None:
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path), timeout=1.0)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    return int(row[0]) if row else 0


def _remove_sidecars(path: Path, *, strict: bool) -> None:
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(str(path) + suffix)
        try:
            sidecar.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            if strict:
                raise RebuildError(
                    f"cannot remove stale SQLite sidecar {sidecar}: {_error_text(exc)}"
                ) from exc


def _checkpoint_database(path: Path, *, allow_invalid: bool = False) -> None:
    if not path.exists():
        raise RebuildError(f"rebuild did not create a database at {path}")
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(path), timeout=5.0)
        checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint is None:
            raise RebuildError(
                f"cannot checkpoint database {path}: WAL checkpoint returned no result"
            )
        if checkpoint[0]:
            # A busy TRUNCATE checkpoint means another connection still holds a
            # read transaction over WAL frames.  Removing the sidecars or
            # replacing the main file now could discard those frames.
            raise RebuildError(
                f"cannot checkpoint database {path}: WAL checkpoint is busy "
                "(another SQLite connection still holds a read transaction); "
                "refusing to remove sidecars or swap the database. Close other "
                "seekr connections and retry."
            )
    except sqlite3.OperationalError as exc:
        # Locked, unreadable or I/O-failing databases are not safe to replace.
        raise RebuildError(f"cannot checkpoint database {path}: {exc}") from exc
    except sqlite3.DatabaseError as exc:
        if not allow_invalid:
            raise RebuildError(f"cannot checkpoint database {path}: {exc}") from exc
        # An explicitly corrupt/not-a-database live file has no checkpointable
        # data; a reindex is a valid recovery path for it.
        _remove_sidecars(path, strict=False)
        return
    finally:
        if conn is not None:
            conn.close()
    _remove_sidecars(path, strict=True)


def _fsync_file(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _replace_database(temp_path: Path, live_path: Path) -> None:
    _checkpoint_database(temp_path)
    if live_path.exists():
        # The exclusive lifecycle lock means no cooperating Store is open.  A
        # successful checkpoint folds any WAL frames into the old main file
        # before its sidecars are removed, so a failed os.replace cannot lose
        # data; a busy checkpoint refuses the swap before any sidecar removal.
        _checkpoint_database(live_path, allow_invalid=True)
    _fsync_file(temp_path)
    try:
        os.replace(str(temp_path), str(live_path))
    except OSError as exc:
        raise RebuildError(
            f"cannot atomically replace live index {live_path}: {_error_text(exc)}"
        ) from exc
    _remove_sidecars(live_path, strict=False)
    _fsync_directory(live_path.parent)


def _remove_database_files(path: Path) -> None:
    for suffix in ("", "-wal", "-shm", "-journal"):
        try:
            Path(str(path) + suffix).unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def rebuild_index(
    cfg: Config,
    *,
    embedder: Embedder | None = None,
    store_factory: Callable[..., Store] | None = None,
    indexer_factory: Callable[..., Indexer] | None = None,
    lock_timeout: float = 0.0,
    force: bool = False,
    allow_empty: bool = False,
) -> SyncReport:
    """Build a fresh sibling database and atomically replace the live index.

    The live database is only replaced after the fresh database has been built,
    checked, checkpointed and the lifecycle lock has been upgraded to exclusive.
    Any error before the atomic rename leaves the live database untouched.

    ``force`` only overrides the per-file indexing-error guard.  Replacing a
    non-empty live index with an empty rebuild requires ``allow_empty``.
    """
    _validate_rebuild_roots(cfg)

    store_factory = store_factory or Store
    indexer_factory = indexer_factory or Indexer
    live_path = Path(cfg.db_path).expanduser()
    try:
        live_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RebuildError(
            f"cannot create index directory {live_path.parent}: {_error_text(exc)}"
        ) from exc
    temp_path = _temporary_db_path(live_path)
    store = None
    lock = LifecycleLock(live_path, shared=True, blocking=False)

    try:
        try:
            lock.acquire()
        except LifecycleLockError as exc:
            raise RebuildError(
                f"cannot start reindex: {exc}; live index left untouched"
            ) from exc
        old_count = _existing_document_count(live_path)

        if embedder is None:
            embedder = build_embedder(cfg)
        try:
            dim = int(embedder.dim)
        except Exception as exc:
            raise RebuildError(
                f"cannot determine embedding dimension: {_error_text(exc)}"
            ) from exc

        try:
            store = store_factory(temp_path, cfg.embed_model, dim)
            indexer = indexer_factory(store, embedder, cfg)
            report = indexer.sync(list(cfg.roots))
            stats = store.stats()
        except RebuildError:
            raise
        except Exception as exc:
            raise RebuildError(
                f"cannot build fresh index: {_error_text(exc)}; live index left untouched"
            ) from exc

        if report.errors and not force:
            first = report.errors[0]
            raise RebuildError(
                "refusing to replace the live index after "
                f"{len(report.errors)} indexing error(s) "
                f"(first: {first[0]}: {first[1]}); live index left untouched"
            )
        if int(stats.get("documents", 0)) == 0 and old_count and not allow_empty:
            raise RebuildError(
                "refusing to replace a non-empty live index with an empty rebuild; "
                "live index left untouched"
            )

        try:
            store.close()
        except Exception as exc:
            raise RebuildError(
                f"cannot close fresh index: {_error_text(exc)}; live index left untouched"
            ) from exc
        finally:
            store = None

        _checkpoint_database(temp_path)

        try:
            lock.upgrade(timeout=lock_timeout)
        except LifecycleLockError as exc:
            raise RebuildError(
                f"cannot swap in the rebuilt index: {exc}; live index left untouched"
            ) from exc

        # Re-check after the exclusive upgrade: a daemon may have added
        # documents during the build, and an empty rebuild must not erase them.
        current_count = _existing_document_count(live_path)
        if (
            int(stats.get("documents", 0)) == 0
            and not allow_empty
            and live_path.exists()
            and current_count != 0
        ):
            raise RebuildError(
                "refusing to replace a non-empty live index with an empty rebuild; "
                "live index left untouched"
            )

        _replace_database(temp_path, live_path)
        return report
    except RebuildError:
        raise
    except Exception as exc:
        raise RebuildError(
            f"reindex failed: {_error_text(exc)}; live index left untouched"
        ) from exc
    finally:
        if store is not None:
            # Best-effort cleanup: never mask the rebuild outcome with a close error.
            try:
                store.close()
            except Exception:  # noqa: BLE001, S110
                pass
        lock.release()
        _remove_database_files(temp_path)
