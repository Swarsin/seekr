"""Tests for the portable shared/exclusive file lock."""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Self

import pytest

from seekr.locks import FileLock, LockTimeout

CTX = multiprocessing.get_context("spawn")
WAIT = 30


def _hold(path: str, shared: bool, ready, release) -> None:
    lock = FileLock(Path(path))
    ok = lock.acquire(shared=shared, blocking=False)
    ready.set() if ok else None
    if ok:
        release.wait(WAIT)
        lock.release()


class _Child:
    def __init__(self, path: Path, shared: bool) -> None:
        self.ready = CTX.Event()
        self.release = CTX.Event()
        self.proc = CTX.Process(
            target=_hold, args=(str(path), shared, self.ready, self.release)
        )

    def __enter__(self) -> Self:
        self.proc.start()
        assert self.ready.wait(WAIT), "child never acquired the lock"
        return self

    def stop(self) -> None:
        self.release.set()
        self.proc.join(WAIT)

    def __exit__(self, *exc: object) -> None:
        self.stop()


@pytest.fixture
def lock_path(tmp_path: Path) -> Path:
    return tmp_path / "x.lock"


def test_shared_locks_coexist(lock_path: Path) -> None:
    with _Child(lock_path, shared=True):
        lock = FileLock(lock_path)
        assert lock.acquire(shared=True, blocking=False) is True
        lock.release()


def test_shared_blocks_exclusive(lock_path: Path) -> None:
    with _Child(lock_path, shared=True):
        lock = FileLock(lock_path)
        assert lock.acquire(shared=False, blocking=False) is False


def test_exclusive_blocks_shared(lock_path: Path) -> None:
    with _Child(lock_path, shared=False):
        lock = FileLock(lock_path)
        assert lock.acquire(shared=True, blocking=False) is False


def test_exclusive_available_after_release(lock_path: Path) -> None:
    with _Child(lock_path, shared=True) as child:
        child.stop()
        lock = FileLock(lock_path)
        assert lock.acquire(shared=False, blocking=False) is True
        lock.release()


def test_upgrade_waits_for_other_readers(lock_path: Path) -> None:
    lock = FileLock(lock_path)
    assert lock.acquire(shared=True, blocking=False)
    try:
        with _Child(lock_path, shared=True) as child:
            assert lock.upgrade(blocking=False) is False
            child.stop()
            assert lock.upgrade(blocking=False) is True
    finally:
        lock.release()


def test_release_is_idempotent(lock_path: Path) -> None:
    lock = FileLock(lock_path)
    lock.release()
    assert lock.acquire(shared=True)
    lock.release()
    lock.release()
    assert lock.acquire(shared=False, blocking=False)
    lock.release()


def test_context_manager_releases(lock_path: Path) -> None:
    with FileLock(lock_path) as lock:
        assert lock is not None
    other = FileLock(lock_path)
    assert other.acquire(shared=False, blocking=False)
    other.release()


def test_lock_timeout_is_runtime_error() -> None:
    assert issubclass(LockTimeout, RuntimeError)
