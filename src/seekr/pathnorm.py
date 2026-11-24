"""Platform-aware path normalisation: canonical form, case folding, long paths.

Every function takes an injected ``platform``.  When it is omitted the module
attribute :data:`_PLATFORM` (:data:`sys.platform`) applies, read at call time,
so tests can switch every caller to the Windows or macOS rules at once.

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
    "same_path",
]

# The platform whose path rules apply by default; patched by tests.
_PLATFORM = sys.platform

_CASE_INSENSITIVE_PLATFORMS = frozenset({"win32", "darwin"})
# MAX_PATH is 260 including the terminating NUL.
_WINDOWS_MAX_PATH = 259
_LONG_PREFIX = "\\\\?\\"
_UNC_LONG_PREFIX = "\\\\?\\UNC\\"


def _platform(platform: str | None) -> str:
    return _PLATFORM if platform is None else platform


def case_insensitive(platform: str | None = None) -> bool:
    """Return ``True`` where the default filesystem ignores letter case."""
    return _platform(platform) in _CASE_INSENSITIVE_PLATFORMS


def path_separator(platform: str | None = None) -> str:
    """Return the separator used in stored paths on *platform*."""
    return "\\" if _platform(platform) == "win32" else "/"


def normalize_unicode(text: str, *, platform: str | None = None) -> str:
    """Return *text* NFC-normalised on macOS, unchanged elsewhere."""
    if _platform(platform) == "darwin":
        return unicodedata.normalize("NFC", text)
    return text


def canonical_path(path: Path | str, *, platform: str | None = None) -> str:
    """Return the resolved path as stored in the index (NFC on macOS).

    Raises ``OSError``/``RuntimeError`` like :meth:`Path.resolve`.
    """
    resolved = str(Path(path).resolve(strict=False))
    return normalize_unicode(resolved, platform=platform)


def _comparable(value: str, platform: str | None) -> str:
    platform = _platform(platform)
    if platform == "win32":
        value = value.replace("/", "\\")
    value = normalize_unicode(value, platform=platform)
    if case_insensitive(platform):
        value = value.casefold()
    return value


def same_path(a: Path | str, b: Path | str, *, platform: str | None = None) -> bool:
    """Return whether two (already resolved) paths name the same location.

    Exact string equality on Linux; case- and Unicode-form-insensitive on
    macOS; case-insensitive and ``/``-tolerant on Windows.
    """
    return _comparable(str(a), platform) == _comparable(str(b), platform)


def has_prefix(path: str, prefix: str, *, platform: str | None = None) -> bool:
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


def os_path(path: str, *, platform: str | None = None) -> str:
    """Return *path* in a form the OS can open, adding ``\\\\?\\`` when needed.

    Only Windows paths longer than ``MAX_PATH`` change: local paths gain
    ``\\\\?\\`` and UNC paths become ``\\\\?\\UNC\\server\\share\\...``.
    Elsewhere (and for already-prefixed paths) it returns *path* unchanged.
    """
    value = str(path)
    if _platform(platform) != "win32" or len(value) <= _WINDOWS_MAX_PATH:
        return value
    if value.startswith((_LONG_PREFIX, "\\\\.\\")):
        return value
    if value.startswith("\\\\"):
        return _UNC_LONG_PREFIX + value[2:]
    return _LONG_PREFIX + value
