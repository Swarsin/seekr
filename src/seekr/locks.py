"""Portable shared/exclusive advisory file locks.

POSIX uses ``fcntl.flock``.  Windows uses ``LockFileEx``/``UnlockFileEx`` on
byte 0 of the lock file.  Both give shared (many readers) and exclusive
(single writer) modes, and both are released automatically if the process dies.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import Self

__all__ = ["FileLock", "LockTimeout"]

_IS_WINDOWS = os.name == "nt"

if _IS_WINDOWS:  # pragma: no cover - exercised only on Windows
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _LOCKFILE_FAIL_IMMEDIATELY = 0x1
    _LOCKFILE_EXCLUSIVE_LOCK = 0x2
    _ERROR_IO_PENDING = 997
    _ERROR_LOCK_VIOLATION = 33

    class _Offset(ctypes.Structure):
        _fields_ = [("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD)]

    class _OffsetUnion(ctypes.Union):
        _fields_ = [("s", _Offset), ("Pointer", ctypes.c_void_p)]

    class _Overlapped(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [
            ("Internal", ctypes.c_void_p),
            ("InternalHigh", ctypes.c_void_p),
            ("u", _OffsetUnion),
            ("hEvent", wintypes.HANDLE),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.LockFileEx.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    ]
    _kernel32.LockFileEx.restype = wintypes.BOOL
    _kernel32.UnlockFileEx.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    ]
    _kernel32.UnlockFileEx.restype = wintypes.BOOL

    def _try_lock(fd: int, *, shared: bool, blocking: bool) -> bool:
        flags = 0 if shared else _LOCKFILE_EXCLUSIVE_LOCK
        if not blocking:
            flags |= _LOCKFILE_FAIL_IMMEDIATELY
        handle = msvcrt.get_osfhandle(fd)
        overlapped = _Overlapped()  # offset 0
        if _kernel32.LockFileEx(handle, flags, 0, 1, 0, ctypes.byref(overlapped)):
            return True
        err = ctypes.get_last_error()
        if not blocking and err in (_ERROR_LOCK_VIOLATION, _ERROR_IO_PENDING):
            return False
        raise OSError(err, ctypes.FormatError(err))

    def _unlock(fd: int) -> None:
        handle = msvcrt.get_osfhandle(fd)
        overlapped = _Overlapped()
        if not _kernel32.UnlockFileEx(handle, 0, 1, 0, ctypes.byref(overlapped)):
            err = ctypes.get_last_error()
            raise OSError(err, ctypes.FormatError(err))

else:
    import fcntl

    def _try_lock(fd: int, *, shared: bool, blocking: bool) -> bool:
        op = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        if not blocking:
            op |= fcntl.LOCK_NB
        try:
            fcntl.flock(fd, op)
        except OSError as exc:
            if not blocking and exc.errno in (errno.EACCES, errno.EAGAIN):
                return False
            raise
        return True

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


class LockTimeout(RuntimeError):
    """A lock could not be taken within the allowed time."""


class FileLock:
    """A shared or exclusive advisory lock on *path* (created if missing)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None
        self._shared = True

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self, *, shared: bool, blocking: bool = True) -> bool:
        """Take the lock; return False if busy and ``blocking`` is False.

        Raises :class:`OSError` for failures other than the lock being busy.
        """
        if self._fd is not None:
            raise RuntimeError("lock is already held; release it first")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            ok = _try_lock(fd, shared=shared, blocking=blocking)
        except BaseException:
            os.close(fd)
            raise
        if not ok:
            os.close(fd)
            return False
        self._fd = fd
        self._shared = shared
        return True

    def upgrade(self, *, blocking: bool = True) -> bool:
        """Convert a held shared lock to exclusive; False if busy (non-blocking).

        On POSIX this is an in-place ``flock`` conversion, so a failed
        non-blocking attempt leaves the shared lock held.  Windows cannot
        convert, so the shared lock is released and an exclusive one taken;
        this is not atomic, and on failure the shared lock is re-acquired
        (another writer may have slipped in between, in which case the lock is
        left released and :class:`LockTimeout` is raised).
        """
        if self._fd is None:
            raise RuntimeError("lock is not held")
        if not self._shared:
            return True
        fd = self._fd
        if not _IS_WINDOWS:
            if _try_lock(fd, shared=False, blocking=blocking):
                self._shared = False
                return True
            return False
        _unlock(fd)  # pragma: no cover - Windows only
        if _try_lock(fd, shared=False, blocking=blocking):  # pragma: no cover
            self._shared = False
            return True
        if _try_lock(fd, shared=True, blocking=False):  # pragma: no cover
            return False
        self.release()  # pragma: no cover
        raise LockTimeout(  # pragma: no cover
            f"lost the shared lock on {self.path} while upgrading"
        )

    def release(self) -> None:
        """Release the lock; safe to call more than once."""
        fd = self._fd
        self._fd = None
        if fd is None:
            return
        try:
            _unlock(fd)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass

    def __enter__(self) -> Self:
        self.acquire(shared=False)
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.release()
        return False
