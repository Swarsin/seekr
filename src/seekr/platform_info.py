"""Platform and mount-table helpers for detecting slow filesystems."""

from __future__ import annotations

import os
import posixpath
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "SLOW_FSTYPES",
    "Mount",
    "is_slow_path",
    "is_wsl",
    "mount_for",
    "parse_mounts",
    "read_mounts",
]

_OSRELEASE_PATH = Path("/proc/sys/kernel/osrelease")
_MOUNTS_PATH = Path("/proc/mounts")

SLOW_FSTYPES = frozenset(
    {"9p", "drvfs", "cifs", "smb3", "nfs", "nfs4", "fuse.sshfs"}
)

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


def read_mounts() -> list[Mount]:
    """Read and parse ``/proc/mounts``, returning ``[]`` when unreadable."""
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


def is_slow_path(path: Path, mounts: Sequence[Mount]) -> bool:
    """Return ``True`` if *path* resides on a filesystem requiring polling."""
    mount = mount_for(path, mounts)
    return mount is not None and mount.fstype in SLOW_FSTYPES
