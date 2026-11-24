"""Platform-aware path normalisation: canonical form, case folding, long paths.

Every function takes an injected ``platform`` (default :data:`sys.platform`)
so the Windows and macOS rules are testable on any OS.

* Windows and macOS filesystems are case-insensitive by default, so path
  prefix comparisons fold case there.
* macOS may hand back file names in decomposed Unicode (NFD); canonical
  paths are NFC there so the same file always gets the same stored path.
* Windows paths longer than ``MAX_PATH`` need the ``\\\\?\\`` prefix to open.

On Linux :func:`canonical_path` is exactly ``Path.resolve(strict=False)``, so
existing indexes keep identical stored paths.
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

__all__ = [
    "canonical_path",
    "case_insensitive",
    "has_prefix",
    "normalize_unicode",
    "os_path",
    "path_separator",
]

_CASE_INSENSITIVE_PLATFORMS = frozenset({"win32", "darwin"})
# MAX_PATH is 260 including the terminating NUL.
_WINDOWS_MAX_PATH = 259
_LONG_PREFIX = "\\\\?\\"
_UNC_LONG_PREFIX = "\\\\?\\UNC\\"


def case_insensitive(platform: str = sys.platform) -> bool:
    """Return ``True`` where the default filesystem ignores letter case."""
    return platform in _CASE_INSENSITIVE_PLATFORMS


def path_separator(platform: str = sys.platform) -> str:
    """Return the separator used in stored paths on *platform*."""
    return "\\" if platform == "win32" else "/"


def normalize_unicode(text: str, *, platform: str = sys.platform) -> str:
    """Return *text* NFC-normalised on macOS, unchanged elsewhere."""
    if platform == "darwin":
        return unicodedata.normalize("NFC", text)
    return text


def canonical_path(path: Path | str, *, platform: str = sys.platform) -> str:
    """Return the resolved path as stored in the index (NFC on macOS).

    Raises ``OSError``/``RuntimeError`` like :meth:`Path.resolve`.
    """
    resolved = str(Path(path).resolve(strict=False))
    return normalize_unicode(resolved, platform=platform)


def _comparable(value: str, platform: str) -> str:
    if platform == "win32":
        value = value.replace("/", "\\")
    value = normalize_unicode(value, platform=platform)
    if case_insensitive(platform):
        value = value.casefold()
    return value


def has_prefix(path: str, prefix: str, *, platform: str = sys.platform) -> bool:
    """Return whether *path* is *prefix* or below it, on a separator boundary.

    Case is folded on Windows and macOS; Windows also accepts ``/`` in either
    argument.  On Linux ``\\`` is an ordinary file-name character.
    """
    sep = path_separator(platform)
    target = _comparable(str(path), platform)
    base = _comparable(str(prefix), platform).rstrip(sep)
    if target == base:
        return True
    return target.startswith(base + sep)


def os_path(path: str, *, platform: str = sys.platform) -> str:
    """Return *path* in a form the OS can open, adding ``\\\\?\\`` when needed.

    Only Windows paths longer than ``MAX_PATH`` change: local paths gain
    ``\\\\?\\`` and UNC paths become ``\\\\?\\UNC\\server\\share\\...``.
    Elsewhere (and for already-prefixed paths) it returns *path* unchanged.
    """
    value = str(path)
    if platform != "win32" or len(value) <= _WINDOWS_MAX_PATH:
        return value
    if value.startswith((_LONG_PREFIX, "\\\\.\\")):
        return value
    if value.startswith("\\\\"):
        return _UNC_LONG_PREFIX + value[2:]
    return _LONG_PREFIX + value
