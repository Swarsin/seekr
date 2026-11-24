"""Tests for the user-only service discovery file."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from seekr.service_discovery import (
    ServiceInfo,
    pid_alive,
    read_service_info,
    remove_service_info,
    write_service_info,
)

INFO = ServiceInfo(
    pid=os.getpid(),
    port=8765,
    token="tok-abc",
    version="1.2.3",
    started_at="2026-01-01T00:00:00+00:00",
)

posix_only = pytest.mark.skipif(os.name != "posix", reason="posix perms")


def dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


@pytest.fixture
def p(tmp_path: Path) -> Path:
    return tmp_path / "state" / "service.json"


def test_json_roundtrip() -> None:
    assert ServiceInfo.from_json(INFO.to_json()) == INFO


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        '{"pid": 1}',
        '{"pid": "1", "port": 2, "token": "t", "version": "v", "started_at": "s"}',
        '{"pid": true, "port": 2, "token": "t", "version": "v", "started_at": "s"}',
    ],
)
def test_from_json_bad_shape(text: str) -> None:
    with pytest.raises(ValueError):
        ServiceInfo.from_json(text)


def test_roundtrip_and_permissions(p: Path) -> None:
    write_service_info(INFO, p)
    assert read_service_info(p) == INFO
    if os.name == "posix":
        assert stat.S_IMODE(p.stat().st_mode) == 0o600
        assert stat.S_IMODE(p.parent.stat().st_mode) == 0o700
    assert [x.name for x in p.parent.iterdir()] == ["service.json"]


def test_overwrite(p: Path) -> None:
    write_service_info(INFO, p)
    new = replace(INFO, port=9999)
    write_service_info(new, p)
    assert read_service_info(p) == new


def test_missing_is_none(p: Path) -> None:
    assert read_service_info(p) is None


def test_stale_pid_is_ignored(p: Path) -> None:
    write_service_info(replace(INFO, pid=dead_pid()), p)
    assert read_service_info(p) is None


def test_garbage_file_is_ignored(p: Path) -> None:
    p.parent.mkdir(parents=True)
    p.write_text("{{{ garbage")
    assert read_service_info(p) is None


def test_unreadable_is_ignored(p: Path) -> None:
    p.mkdir(parents=True)  # a directory where the file should be -> OSError
    assert read_service_info(p) is None


def test_remove_only_own_pid(p: Path) -> None:
    write_service_info(INFO, p)
    remove_service_info(p, INFO.pid + 1)
    assert p.exists()
    remove_service_info(p, INFO.pid)
    assert not p.exists()
    remove_service_info(p, INFO.pid)  # missing is fine


def test_remove_works_for_dead_pid(p: Path) -> None:
    pid = dead_pid()
    write_service_info(replace(INFO, pid=pid), p)
    remove_service_info(p, pid)
    assert not p.exists()


def test_pid_alive() -> None:
    assert pid_alive(os.getpid())
    assert not pid_alive(0)
    assert not pid_alive(-5)
    assert not pid_alive(dead_pid())


@posix_only
def test_refuses_group_writable_parent(p: Path) -> None:
    p.parent.mkdir(parents=True)
    p.parent.chmod(0o770)
    with pytest.raises(PermissionError):
        write_service_info(INFO, p)
    assert not p.exists()


@posix_only
def test_failure_cleans_temp_file(p: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: object, **k: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        write_service_info(INFO, p)
    assert list(p.parent.iterdir()) == []
