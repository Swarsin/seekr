"""Shared test setup."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from seekr import paths


@pytest.fixture(autouse=True)
def _isolated_service_state(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a developer's running ``seekr service`` out of the test suite.

    CLI and MCP searches route through a discovered service.  Each test gets
    its own empty state directory (``paths.service_file`` resolves through
    ``paths.state_dir``), so no service is found unless the test starts one.
    This works on every platform, unlike ``XDG_STATE_HOME`` alone, which
    platformdirs ignores on macOS and Windows.  The patched ``state_dir``
    follows ``XDG_STATE_HOME`` so a test that sets it (and the Linux
    subprocesses it spawns) still agree on one folder.  ``test_paths`` checks
    the real defaults.  Off Linux, config and data directories follow
    ``XDG_CONFIG_HOME`` and ``XDG_DATA_HOME`` the same way.
    """
    if request.module.__name__.rsplit(".", 1)[-1] == "test_paths":
        return
    state = Path(tmp_path_factory.mktemp("state"))
    state.chmod(0o700)
    monkeypatch.setenv("XDG_STATE_HOME", str(state))

    def state_dir(*, platform: str | None = None) -> Path:
        del platform
        return Path(os.environ["XDG_STATE_HOME"]) / "seekr"

    monkeypatch.setattr(paths, "state_dir", state_dir)

    if not sys.platform.startswith("linux"):
        # platformdirs ignores XDG_CONFIG_HOME and XDG_DATA_HOME on Windows
        # (and only sometimes honours them on macOS), so tests that isolate
        # config and the index through them would read and write the real
        # user folders.  Give config and data the Linux rules here, with a
        # per-test fallback instead of the real home.
        fallback = Path(tmp_path_factory.mktemp("home"))
        for name, var in (("config_dir", "XDG_CONFIG_HOME"), ("data_dir", "XDG_DATA_HOME")):
            monkeypatch.setattr(paths, name, _xdg_dir(var, fallback / name))


def _xdg_dir(var: str, fallback: Path) -> Callable[..., Path]:
    def resolve(*, platform: str | None = None) -> Path:
        del platform
        value = os.environ.get(var)
        return (Path(value).expanduser() if value else fallback) / "seekr"

    return resolve
