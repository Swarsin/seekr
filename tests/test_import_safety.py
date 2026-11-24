"""The CLI must import and search on platforms without Unix sockets."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_SCRIPT = textwrap.dedent(
    """
    import os
    import socket

    del socket.AF_UNIX
    del os.getuid

    import pytest  # noqa: F401  (only to prove the environment is intact)

    from seekr import cli, cli_index, cli_search, client, daemon
    from seekr.web_service import _socket_is_live  # noqa: F401

    assert daemon.legacy_daemon_supported() is False

    try:
        cli.app(["--help"])
    except SystemExit as exc:
        assert exc.code in (0, None), exc.code

    class FakeSearcher:
        def get_hits(self, query, **kwargs):
            return ["hit:" + query]

    assert cli_search.get_hits("alpha", mode="keyword", searcher=FakeSearcher()) == [
        "hit:alpha"
    ]

    def boom(*args, **kwargs):
        raise AssertionError("legacy daemon must not be contacted")

    cli_search.try_daemon_search = boom
    cli_search._service_hits = lambda *a, **k: None
    cli_search._build_searcher = lambda *a, **k: None

    from seekr.config import load_config

    cfg = load_config()
    assert cli_search.get_hits("alpha", mode="keyword", cfg=cfg) == []
    assert cli_search.try_daemon_hits("alpha", cfg=cfg) is None
    assert client.try_daemon_status() is None
    assert cli_index._daemon_status() == (False, None)
    assert cli_index._polled_watch_info(True, None, cfg) == (
        [],
        cfg.poll_interval_seconds,
    )

    from pathlib import Path

    from seekr import doctor, web_service

    assert web_service._socket_is_live(Path("x")) is False
    service = web_service.WebService(Path(os.environ["XDG_CONFIG_HOME"]) / "c.toml")
    assert service._daemon_socket() is None
    assert service._daemon_status() == (False, None)
    assert service.daemon_present() is False
    assert service.status().to_dict()["daemon"] == {"present": False, "socket": None}
    state = doctor._default_daemon_state()
    assert state.startswith("unavailable"), state

    from seekr import cli_daemon

    try:
        cli_daemon.run_daemon(cfg=cfg)
    except daemon.DaemonError:
        pass
    else:
        raise AssertionError("run_daemon must refuse")
    try:
        daemon.serve(object(), Path("x.sock"), None)
    except daemon.DaemonError:
        pass
    else:
        raise AssertionError("serve must refuse")

    import contextlib
    import io

    for argv in (["status"], ["status", "--json"]):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                cli.app(argv, standalone_mode=False)
            except SystemExit as exc:
                assert exc.code in (0, None), (argv, exc.code)
        text = out.getvalue()
        assert text.strip(), argv
        assert "unavailable" not in text, text
        assert "seekr.sock" not in text, text
        if "--json" in argv:
            import json

            payload = json.loads(text)
            assert payload["daemon"] == {"running": False, "socket": None}, payload
    print("ok")
    """
)


def test_cli_imports_and_help_without_af_unix(tmp_path: Path) -> None:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("XDG_", "SEEKR_"))
    }
    for name in ("config", "data", "state", "cache", "runtime"):
        (tmp_path / name).mkdir()
        env[f"XDG_{name.upper()}_HOME"] = str(tmp_path / name)
    env["XDG_RUNTIME_DIR"] = str(tmp_path / "runtime")
    env["HOME"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("ok")
