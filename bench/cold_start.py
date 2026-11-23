#!/usr/bin/env python3
"""Measure cold one-shot search time and daemon time-to-ready.

Each one-shot run is a fresh ``seekr search`` subprocess with no daemon
available, so it pays for interpreter start-up, imports, model loading and
query embedding. The daemon measurement starts ``seekr daemon`` and polls its
socket until it answers ``ping``, then until a search answers (warm model).

The configuration file must point at an existing index whose models are
cached locally. A private ``XDG_RUNTIME_DIR`` keeps any running daemon out of
the measurement.

    python bench/cold_start.py --config ~/.config/seekr/config.toml --runs 5
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

DEFAULT_QUERY = "bias variance tradeoff"


def _seekr() -> list[str]:
    return [sys.executable, "-m", "seekr"]


def _one_shot(env: dict[str, str], query: str) -> float:
    started = time.perf_counter()
    subprocess.run(
        [*_seekr(), "search", query, "-k", "3"],
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return time.perf_counter() - started


def _daemon_ready(env: dict[str, str], query: str, timeout: float = 120.0) -> tuple[float, float]:
    from seekr.client import _send_request

    sock = Path(env["XDG_RUNTIME_DIR"]) / "seekr.sock"
    started = time.perf_counter()
    proc = subprocess.Popen(
        [*_seekr(), "daemon"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        ping_s = None
        while time.perf_counter() - started < timeout:
            if proc.poll() is not None:
                raise RuntimeError(f"seekr daemon exited with code {proc.returncode}")
            try:
                if ping_s is None and _send_request({"op": "ping"}, sock, 1.0):
                    ping_s = time.perf_counter() - started
                if ping_s is not None:
                    reply = _send_request(
                        {"op": "search", "query": query, "k": 3}, sock, 60.0
                    )
                    if reply and isinstance(reply.get("hits"), list):
                        search_s = time.perf_counter() - started
                        return ping_s, search_s
            except (OSError, RuntimeError, ValueError):
                time.sleep(0.02)  # socket not ready yet
                continue
            time.sleep(0.02)
        raise RuntimeError("daemon did not become ready in time")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="seekr-cold-") as runtime:
        os.chmod(runtime, 0o700)
        env = dict(os.environ)
        env["SEEKR_CONFIG"] = str(args.config.expanduser().resolve())
        env["XDG_RUNTIME_DIR"] = runtime

        _one_shot(env, args.query)  # warm the OS file cache once, untimed
        one_shot = [_one_shot(env, args.query) for _ in range(args.runs)]
        daemon = [_daemon_ready(env, args.query) for _ in range(args.runs)]

    print(f"runs: {args.runs}")
    print(f"one_shot_search_median_s: {statistics.median(one_shot):.3f}")
    print(f"one_shot_search_all_s: {', '.join(f'{v:.3f}' for v in one_shot)}")
    print(f"daemon_ping_ready_median_s: {statistics.median(d[0] for d in daemon):.3f}")
    print(f"daemon_first_search_median_s: {statistics.median(d[1] for d in daemon):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
