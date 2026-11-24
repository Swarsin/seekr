"""User-only service discovery file.

The running service records its pid, port, auth token and version in a small
JSON file so local clients can find it.  The token authorises API calls, so the
file is created 0600 (POSIX) inside a directory that others cannot write to; on
Windows it lives under the per-user ``%LOCALAPPDATA%`` and inherits its ACL.
"""

from __future__ import annotations

import json
import os
import stat
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = [
    "ServiceInfo",
    "pid_alive",
    "read_service_info",
    "remove_service_info",
    "write_service_info",
]

_IS_WINDOWS = os.name == "nt"

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5
_PID_LIMIT = 2**31


@dataclass(frozen=True)
class ServiceInfo:
    pid: int
    port: int
    token: str
    version: str
    started_at: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> ServiceInfo:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid service file: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("service file must contain a JSON object")  # noqa: TRY004
        values: dict[str, object] = {}
        for name, kind in (
            ("pid", int),
            ("port", int),
            ("token", str),
            ("version", str),
            ("started_at", str),
        ):
            value = data.get(name)
            if not isinstance(value, kind) or isinstance(value, bool):
                raise ValueError(f"service file field {name!r} must be {kind.__name__}")  # noqa: TRY004
            values[name] = value
        if not 0 < values["pid"] < _PID_LIMIT:  # type: ignore[operator]
            raise ValueError("service file field 'pid' is out of range")
        if not 0 < values["port"] < 65536:  # type: ignore[operator]
            raise ValueError("service file field 'port' is out of range")
        return cls(**values)  # type: ignore[arg-type]


def pid_alive(pid: int) -> bool:
    if not 0 < pid < _PID_LIMIT:
        return False
    if _IS_WINDOWS:  # pragma: no cover - exercised only on Windows
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, OverflowError):
        return False
    return True


def _pid_alive_windows(pid: int) -> bool:  # pragma: no cover - Windows only
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def write_service_info(info: ServiceInfo, path: Path) -> None:
    parent = path.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix" and stat.S_IMODE(parent.stat().st_mode) & 0o022:
        raise PermissionError(
            f"{parent} is writable by group or others; refusing to store the service token there"
        )
    tmp = parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(info.to_json())
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_raw(path: Path) -> ServiceInfo | None:
    try:
        text = path.read_text(encoding="utf-8")
        return ServiceInfo.from_json(text)
    except (OSError, ValueError):
        return None


def read_service_info(path: Path) -> ServiceInfo | None:
    info = _read_raw(path)
    if info is None or not pid_alive(info.pid):
        return None
    return info


def remove_service_info(path: Path, pid: int) -> None:
    info = _read_raw(path)
    if info is None or info.pid != pid:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass
