"""Shared test setup."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_service_state(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a developer's running ``seekr service`` out of the test suite.

    CLI and MCP searches route through a discovered service; pointing the state
    directory at an empty temporary folder means none is found unless a test
    starts one itself.  ``test_paths`` checks the real platform defaults.
    """
    if request.module.__name__.rsplit(".", 1)[-1] == "test_paths":
        return
    state = Path(tmp_path_factory.getbasetemp()) / "isolated-state"
    state.mkdir(mode=0o700, exist_ok=True)
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
