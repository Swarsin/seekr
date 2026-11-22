"""First-run setup helpers for ``seekr init``."""

from __future__ import annotations

import os
import shlex
import stat as stat_module
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import course, extract, indexer, platform_info, web_config
from .config import Config

__all__ = [
    "ScanSummary",
    "example_commands",
    "render_config",
    "replace_roots",
    "scan_roots",
    "suggest_windows_documents",
    "write_config",
]


@dataclass(frozen=True)
class ScanSummary:
    """Counts by extracted kind and detected course modules across roots."""

    files_by_kind: dict[str, int]
    modules: dict[str, tuple[int, ...]]
    total: int


def _resolve_loose(path: Path) -> Path | None:
    try:
        return Path(path).expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        return None


def _resolve_within_root(path: Path, root: Path) -> Path | None:
    resolved = _resolve_loose(path)
    if resolved is None or not resolved.is_relative_to(root):
        return None
    try:
        resolved_stat = resolved.stat()
    except OSError:
        return None
    if not stat_module.S_ISREG(resolved_stat.st_mode):
        return None
    return resolved


def _iter_root_files(root: Path, globs: Sequence[str], seen: set[str]) -> list[Path]:
    """Discover regular files under *root*, matching ``Indexer`` walk semantics."""
    try:
        root_stat = root.stat()
    except OSError:
        return []

    discovered: list[Path] = []
    if stat_module.S_ISREG(root_stat.st_mode):
        if indexer.is_ignored(root, root, globs):
            return []
        key = str(root)
        if key not in seen:
            seen.add(key)
            discovered.append(root)
        return discovered

    if not stat_module.S_ISDIR(root_stat.st_mode):
        return []

    walked: list[Path] = []
    try:
        for dirpath, dirnames, filenames in os.walk(
            root, topdown=True, onerror=lambda _exc: None, followlinks=False
        ):
            dirnames[:] = [
                name
                for name in dirnames
                if not os.path.islink(os.path.join(dirpath, name))
            ]
            for name in filenames:
                walked.append(Path(dirpath) / name)
    except OSError:
        return []

    walked.sort(key=str)
    for file_path in walked:
        if indexer.is_ignored(file_path, root, globs):
            continue
        try:
            lstat_res = os.lstat(file_path)
        except OSError:
            resolved = _resolve_within_root(file_path, root)
            if resolved is None:
                continue
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            if indexer.is_ignored(resolved, None, globs):
                continue
            discovered.append(resolved)
            continue

        if stat_module.S_ISLNK(lstat_res.st_mode):
            resolved = _resolve_within_root(file_path, root)
            if resolved is None:
                continue
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            if indexer.is_ignored(resolved, None, globs):
                continue
            discovered.append(resolved)
            continue

        if not stat_module.S_ISREG(lstat_res.st_mode):
            seen.add(str(file_path))
            continue

        key = str(file_path)
        if key in seen:
            continue
        seen.add(key)
        discovered.append(file_path)

    return discovered


def scan_roots(roots: Sequence[Path], cfg: Config) -> ScanSummary:
    """Walk *roots* and summarize supported file counts, kinds, and course modules."""
    resolved_roots: list[Path] = []
    for raw_root in roots:
        resolved = _resolve_loose(Path(raw_root))
        if resolved is not None:
            resolved_roots.append(resolved)

    week_re = course.compile_week_pattern(cfg.week_pattern)
    seen: set[str] = set()
    files_by_kind: dict[str, int] = {}
    weeks_by_module: dict[str, set[int]] = {}

    for root in resolved_roots:
        for file_path in _iter_root_files(root, cfg.ignore_globs, seen):
            kind = extract.kind_for(file_path)
            if kind is None:
                continue
            files_by_kind[kind] = files_by_kind.get(kind, 0) + 1
            if cfg.course_structure:
                lbl = course.label_for(file_path, resolved_roots, week_re)
                if lbl.module is not None:
                    weeks = weeks_by_module.setdefault(lbl.module, set())
                    if lbl.week is not None:
                        weeks.add(lbl.week)

    sorted_kinds = {kind: files_by_kind[kind] for kind in sorted(files_by_kind)}
    sorted_modules = {
        module: tuple(sorted(weeks_by_module[module]))
        for module in sorted(weeks_by_module)
    }
    return ScanSummary(
        files_by_kind=sorted_kinds,
        modules=sorted_modules,
        total=sum(sorted_kinds.values()),
    )


def render_config(roots: Sequence[Path], *, db_path: Path | None = None) -> str:
    """Render minimal TOML configuration text using valid TOML basic strings."""
    encoded_roots = ", ".join(web_config.toml_string(os.fspath(root)) for root in roots)
    lines = [f"roots = [{encoded_roots}]"]
    if db_path is not None:
        lines.append(f"db_path = {web_config.toml_string(os.fspath(db_path))}")
    return "\n".join(lines) + "\n"


def write_config(path: Path, text: str, *, force: bool) -> None:
    """Atomically write *text* to *path*, raising ``FileExistsError`` unless *force*."""
    target = Path(path).expanduser()
    if (target.exists() or target.is_symlink()) and not force:
        raise FileExistsError(f"config already exists at {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    temp_path: Path | None = None
    try:
        descriptor, temp_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temp_path = Path(temp_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = None
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

        if (target.exists() or target.is_symlink()) and not force:
            raise FileExistsError(f"config already exists at {target}")

        os.replace(temp_path, target)
        temp_path = None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass


def replace_roots(path: Path, roots: Sequence[Path]) -> None:
    """Replace only ``roots`` in an existing config, preserving comments and other keys."""
    snapshot = web_config.read_roots(path)
    web_config.set_roots(path, roots, revision=snapshot.revision)


def suggest_windows_documents(run: Callable[..., Any] = subprocess.run) -> Path | None:
    """Return the Windows Documents folder path under WSL if it exists."""
    if not platform_info.is_wsl():
        return None

    try:
        win_proc = run(
            ["cmd.exe", "/c", "echo", "%USERPROFILE%"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if getattr(win_proc, "returncode", 1) != 0:
        return None
    raw_profile = (getattr(win_proc, "stdout", "") or "").strip()
    if not raw_profile or raw_profile == "%USERPROFILE%":
        return None

    try:
        wsl_proc = run(
            ["wslpath", "-u", raw_profile],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if getattr(wsl_proc, "returncode", 1) != 0:
        return None
    linux_profile = (getattr(wsl_proc, "stdout", "") or "").strip()
    if not linux_profile:
        return None

    docs_dir = Path(linux_profile) / "Documents"
    try:
        if docs_dir.is_dir():
            return docs_dir
    except OSError:
        return None
    return None


def example_commands(summary: ScanSummary) -> list[str]:
    """Return three copy-pasteable example CLI commands tailored to *summary*."""
    first = (
        'seekr search "bias variance"'
        if "machine-learning" in summary.modules
        else 'seekr search "introduction"'
    )

    if summary.modules:
        with_weeks = [
            mod for mod, weeks in sorted(summary.modules.items()) if weeks
        ]
        if "machine-learning" in with_weeks:
            chosen_mod = "machine-learning"
        elif with_weeks:
            chosen_mod = with_weeks[0]
        else:
            chosen_mod = next(iter(sorted(summary.modules)))
        weeks = summary.modules[chosen_mod]
        if weeks:
            second = (
                f'seekr search "lecture notes" --module {shlex.quote(chosen_mod)} '
                f"--week {weeks[0]}"
            )
        else:
            second = f'seekr search "lecture notes" --module {shlex.quote(chosen_mod)}'
    elif summary.files_by_kind:
        top_kind = max(
            sorted(summary.files_by_kind),
            key=lambda kind: summary.files_by_kind[kind],
        )
        second = f'seekr search "notes" --kind {shlex.quote(top_kind)}'
    else:
        second = 'seekr search "notes" --per-file 1'

    return [first, second, "seekr status"]
