"""Shared test setup."""

from __future__ import annotations

import os
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
    the real defaults.
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
