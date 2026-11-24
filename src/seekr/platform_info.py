"""Platform and mount-table helpers for detecting slow filesystems."""

from __future__ import annotations

import ntpath
import os
import posixpath
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import pathnorm

__all__ = [
    "SLOW_FSTYPES",
    "Mount",
    "is_slow_path",
    "is_wsl",
    "mount_for",
    "parse_macos_mount",
    "parse_mounts",
    "read_mounts",
    "windows_path",
    "wslpath_windows",
]

_OSRELEASE_PATH = Path("/proc/sys/kernel/osrelease")
_MOUNTS_PATH = Path("/proc/mounts")
_WINDOWS_DRIVE_RE = re.compile(r"([A-Za-z]):\\?")

# Linux (``/proc/mounts``) and macOS (``mount``) names of network filesystems.
SLOW_FSTYPES = frozenset(
    {
        "9p",
        "drvfs",
        "cifs",
        "smb3",
        "nfs",
        "nfs4",
        "fuse.sshfs",
        "smbfs",
        "afpfs",
        "webdav",
    }
)
# GetDriveTypeW result for a network drive.
_DRIVE_REMOTE = 4
# ``//me@nas/share on /Volumes/share (smbfs, nodev, nosuid)``
_MACOS_MOUNT_RE = re.compile(r"^(?P<source>.+?) on (?P<point>/.*?) \((?P<opts>[^()]*)\)$")

_MOUNT_ESCAPES = {
    r"\040": " ",
    r"\011": "\t",
    r"\012": "\n",
    r"\134": "\\",
}
_MOUNT_ESCAPE_RE = re.compile(r"\\(?:040|011|012|134)")


@dataclass(frozen=True)
class Mount:
    """A mounted filesystem entry parsed from ``/proc/mounts``."""

    source: str
    mount_point: str
    fstype: str


def is_wsl(osrelease: str | None = None) -> bool:
    """Return ``True`` when running under Windows Subsystem for Linux."""
    if osrelease is None:
        try:
            text = _OSRELEASE_PATH.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
    else:
        text = osrelease
    return "microsoft" in text.lower()


def _unescape_mount_field(value: str) -> str:
    return _MOUNT_ESCAPE_RE.sub(lambda match: _MOUNT_ESCAPES[match.group(0)], value)


def parse_mounts(text: str) -> list[Mount]:
    """Parse ``/proc/mounts`` lines, decoding octal field escapes."""
    mounts: list[Mount] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        mounts.append(
            Mount(
                source=_unescape_mount_field(parts[0]),
                mount_point=_unescape_mount_field(parts[1]),
                fstype=_unescape_mount_field(parts[2]),
            )
        )
    return mounts


def parse_macos_mount(text: str) -> list[Mount]:
    """Parse macOS ``mount`` output (``SRC on /point (fstype, opts...)``)."""
    mounts: list[Mount] = []
    for raw_line in text.splitlines():
        match = _MACOS_MOUNT_RE.match(raw_line.strip())
        if match is None:
            continue
        fstype = match.group("opts").split(",", 1)[0].strip()
        if not fstype:
            continue
        mounts.append(
            Mount(
                source=match.group("source"),
                mount_point=match.group("point"),
                fstype=fstype,
            )
        )
    return mounts


def read_mounts(
    *,
    platform: str = sys.platform,
    run: Callable[..., Any] = subprocess.run,
) -> list[Mount]:
    """Return the mount table for *platform*, or ``[]`` when unavailable.

    Linux reads ``/proc/mounts`` and macOS parses ``mount`` output.  Windows
    needs no table: :func:`is_slow_path` asks for the drive type instead.
    """
    if platform == "win32":
        return []
    if platform == "darwin":
        try:
            result = run(
                ["mount"], capture_output=True, text=True, timeout=2, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            return []
        if getattr(result, "returncode", 1) != 0:
            return []
        return parse_macos_mount(getattr(result, "stdout", "") or "")
    try:
        text = _MOUNTS_PATH.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return parse_mounts(text)


def _normalize_posix_path(value: Path | str) -> str:
    raw = os.fspath(value)
    if not raw:
        return ""
    normalized = posixpath.normpath(raw)
    if normalized.startswith("//"):
        normalized = "/" + normalized.lstrip("/")
    return normalized


def mount_for(path: Path, mounts: Sequence[Mount]) -> Mount | None:
    """Return the mount entry with the longest matching prefix for *path*."""
    target = _normalize_posix_path(path)
    if not target.startswith("/"):
        return None

    best: Mount | None = None
    best_len = -1
    for mount in mounts:
        mount_point = _normalize_posix_path(mount.mount_point)
        if not mount_point.startswith("/"):
            continue
        if mount_point == "/":
            matched = True
        else:
            matched = target == mount_point or target.startswith(mount_point + "/")
        if matched and len(mount_point) >= best_len:
            best = mount
            best_len = len(mount_point)
    return best


def _windows_drive_type(root: str) -> int:
    """Return ``GetDriveTypeW(root)``; only called on Windows."""
    import ctypes

    windll = getattr(ctypes, "windll")  # noqa: B009 - absent off Windows
    return int(windll.kernel32.GetDriveTypeW(root))


def _is_slow_windows_path(path: Path | str) -> bool:
    raw = os.fspath(path).replace("/", "\\")
    drive, _rest = ntpath.splitdrive(raw)
    if drive.startswith("\\\\"):
        return True  # UNC share (\\server\share)
    if len(drive) != 2 or drive[1] != ":":
        return False
    try:
        return _windows_drive_type(drive + "\\") == _DRIVE_REMOTE
    except (OSError, AttributeError, ValueError):
        return False


def is_slow_path(
    path: Path,
    mounts: Sequence[Mount] | None = None,
    *,
    platform: str = sys.platform,
) -> bool:
    """Return ``True`` if *path* resides on a filesystem requiring polling.

    Windows: a UNC path or a ``DRIVE_REMOTE`` drive letter.  Linux and macOS:
    the longest matching mount (read with :func:`read_mounts` when *mounts*
    is omitted) has a network filesystem type.
    """
    if platform == "win32":
        return _is_slow_windows_path(path)
    if mounts is None:
        mounts = read_mounts(platform=platform)
    mount = mount_for(path, mounts)
    return mount is not None and mount.fstype in SLOW_FSTYPES


def resolve_root(path: Path | str) -> Path:
    """Expand and resolve a watch root without requiring it to exist.

    The result is in canonical form (NFC on macOS, as the indexer stores it).
    """
    expanded = Path(path).expanduser()
    try:
        resolved = expanded.resolve(strict=False)
    except (OSError, RuntimeError):
        resolved = expanded
    text = str(resolved)
    normalized = pathnorm.normalize_unicode(text)
    return resolved if normalized == text else Path(normalized)


def is_polled_root(
    root: Path | str, mounts: Sequence[Mount], *, platform: str = sys.platform
) -> bool:
    """Return ``True`` if the watcher polls *root* instead of native events."""
    expanded = Path(root).expanduser()
    return is_slow_path(resolve_root(expanded), mounts, platform=platform) or (
        is_slow_path(expanded, mounts, platform=platform)
    )


def split_roots(
    roots: Sequence[Path | str],
    mounts: Sequence[Mount],
    *,
    platform: str = sys.platform,
) -> tuple[list[Path], list[Path]]:
    """Return resolved ``(native, polled)`` watch roots, in configured order."""
    native: list[Path] = []
    polled: list[Path] = []
    for root in roots:
        target = polled if is_polled_root(root, mounts, platform=platform) else native
        target.append(resolve_root(root))
    return native, polled


def windows_path(path: str, mounts: Sequence[Mount]) -> str | None:
    """Convert a POSIX *path* on a Windows drive mount to ``X:\\...`` form."""
    mount = mount_for(Path(path), mounts)
    if mount is None:
        return None
    match = _WINDOWS_DRIVE_RE.fullmatch(mount.source)
    if match is None:
        return None
    drive = match.group(1)
    target = _normalize_posix_path(Path(path))
    mount_point = _normalize_posix_path(mount.mount_point)
    if target == mount_point:
        return f"{drive}:\\"
    rel = target.lstrip("/") if mount_point == "/" else target[len(mount_point) + 1 :]
    win_rel = rel.replace("/", "\\")
    return f"{drive}:\\{win_rel}"


def wslpath_windows(
    path: str,
    run: Callable[..., Any] = subprocess.run,
) -> str | None:
    """Convert *path* via ``wslpath -w``, returning ``None`` on any failure."""
    try:
        result = run(
            ["wslpath", "-w", path],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if getattr(result, "returncode", 1) != 0:
        return None
    converted = (getattr(result, "stdout", "") or "").strip()
    return converted or None
