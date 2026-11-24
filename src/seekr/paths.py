"""Per-platform locations for seekr's config, data, state and cache."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from platformdirs import PlatformDirs


def _dirs(platform: str) -> PlatformDirs:
    """Return the platformdirs implementation for ``platform`` (not the host)."""
    if platform == "win32":
        from platformdirs.windows import Windows as impl
    elif platform == "darwin":
        from platformdirs.macos import MacOS as impl
    else:
        from platformdirs.unix import Unix as impl
    return impl("seekr", appauthor=False, roaming=False)


def _xdg(env_var: str, fallback: Path) -> Path:
    value = os.environ.get(env_var)
    if value:
        return Path(value).expanduser() / "seekr"
    return Path.home() / fallback / "seekr"


def config_dir(*, platform: str = sys.platform) -> Path:
    """Return the directory holding config.toml."""
    if platform.startswith("linux"):
        return _xdg("XDG_CONFIG_HOME", Path(".config"))
    return Path(_dirs(platform).user_config_dir)


def data_dir(*, platform: str = sys.platform) -> Path:
    """Return the directory holding the index database."""
    if platform.startswith("linux"):
        return _xdg("XDG_DATA_HOME", Path(".local") / "share")
    return Path(_dirs(platform).user_data_dir)


def state_dir(*, platform: str = sys.platform) -> Path:
    """Return the directory for runtime state such as the service file."""
    return Path(_dirs(platform).user_state_dir)


def cache_dir(*, platform: str = sys.platform) -> Path:
    """Return the cache directory."""
    return Path(_dirs(platform).user_cache_dir)


def service_file(*, platform: str = sys.platform) -> Path:
    """Return the path of the background service's state file."""
    return state_dir(platform=platform) / "service.json"
