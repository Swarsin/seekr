"""The ``seekr init`` command for one-command setup and demo mode."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import typer

import seekr

from . import cli_index, config, setup_flow
from .config import Config, ConfigError
from .indexer import Indexer, SyncReport
from .lifecycle import LifecycleError, lifecycle_lock
from .web_contract import ApiError

__all__ = ["init", "register"]


def _fatal(message: str) -> NoReturn:
    typer.echo(f"seekr init: {message}", err=True)
    raise typer.Exit(code=1)


def _validate_directory(raw_path: Path | str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.exists():
        _fatal(f"folder does not exist: {candidate}")
    if not candidate.is_dir():
        _fatal(f"path is not a directory: {candidate}")
    return candidate.resolve()


def _resolve_user_roots(raw_paths: list[Path] | None, *, yes: bool) -> list[Path]:
    if raw_paths:
        selected = raw_paths
    elif yes:
        _fatal("pass the folder to index, e.g. seekr init ~/Documents/ai-course")
    else:
        default_dir = str(
            setup_flow.suggest_windows_documents() or (Path.home() / "Documents")
        )
        entered = typer.prompt("Folder with your notes", default=default_dir)
        stripped = entered.strip()
        if not stripped:
            _fatal("folder path must not be empty")
        selected = [Path(stripped)]

    resolved: list[Path] = []
    seen: set[Path] = set()
    for item in selected:
        validated = _validate_directory(item)
        if validated not in seen:
            seen.add(validated)
            resolved.append(validated)
    return resolved


def _format_weeks(weeks: tuple[int, ...]) -> str:
    if len(weeks) == 1:
        return f"week {weeks[0]}"
    if weeks == tuple(range(weeks[0], weeks[-1] + 1)):
        return f"weeks {weeks[0]}-{weeks[-1]}"
    return f"weeks {', '.join(str(week) for week in weeks)}"


def _print_scan_summary(summary: setup_flow.ScanSummary) -> None:
    if summary.files_by_kind:
        breakdown = ", ".join(
            f"{count} {kind}" for kind, count in sorted(summary.files_by_kind.items())
        )
        typer.echo(f"Found {summary.total} files: {breakdown}")
    else:
        typer.echo(f"Found {summary.total} files.")

    if summary.modules:
        parts = []
        for mod, weeks in sorted(summary.modules.items()):
            if weeks:
                parts.append(f"{mod} ({_format_weeks(weeks)})")
            else:
                parts.append(mod)
        typer.echo(f"Modules: {', '.join(parts)}")


def _pull_models(*, yes: bool, config_path: Path | None = None) -> None:
    cmd = [sys.executable, "-m", "seekr", "models", "pull", "--grammars"]
    if yes:
        cmd.append("--yes")

    had_prev = "SEEKR_CONFIG" in os.environ
    prev_val = os.environ.get("SEEKR_CONFIG")
    try:
        if config_path is not None:
            os.environ["SEEKR_CONFIG"] = str(config_path)
        proc = subprocess.run(cmd, check=False)
    finally:
        if config_path is not None:
            if had_prev and prev_val is not None:
                os.environ["SEEKR_CONFIG"] = prev_val
            else:
                os.environ.pop("SEEKR_CONFIG", None)

    if proc.returncode != 0:
        _fatal("model download failed or was aborted")


def _print_index_report(report: SyncReport) -> None:
    indexed_count = report.added + report.updated
    parts = [
        f"{report.added} added",
        f"{report.updated} updated",
        f"{report.unchanged} unchanged",
    ]
    if report.skipped:
        parts.append(f"{report.skipped} skipped")
    if report.errors:
        parts.append(f"{len(report.errors)} errors")
    typer.echo(f"Indexed {indexed_count} file(s) ({', '.join(parts)})")
    for path, message in report.errors:
        typer.echo(f"error: {path}: {message}", err=True)
    for path, message in report.warnings:
        typer.echo(f"warning: {path}: {message}", err=True)


def _index_roots(cfg: Config, roots: list[Path], total_files: int) -> None:
    try:
        embedder = cli_index._build_embedder(cfg)
        with lifecycle_lock(cfg.db_path, shared=True):
            store = cli_index._open_store(cfg, embedder.dim)
            try:
                with typer.progressbar(length=total_files, label="Indexing") as bar:
                    report = Indexer(store, embedder, cfg).sync(
                        roots,
                        on_file=lambda _path: bar.update(1),
                    )
                store.set_last_error_count(len(report.errors))
            finally:
                store.close()
    except typer.Exit:
        raise
    except (LifecycleError, Exception) as exc:  # noqa: BLE001 - CLI boundary
        _fatal(str(exc))
    _print_index_report(report)


def _print_examples(commands: list[str], *, env_prefix: str = "") -> None:
    typer.echo("Try these commands:")
    for cmd in commands:
        typer.echo(f"  {env_prefix}{cmd}")


def _copy_demo_course(demo_dest: Path) -> None:
    src_course = seekr.demo_course_path()
    if not src_course.is_dir():
        _fatal(f"packaged demo course not found at {src_course}")

    try:
        if demo_dest.is_symlink():
            demo_dest.unlink()
        elif demo_dest.exists():
            if demo_dest.is_dir():
                shutil.rmtree(demo_dest)
            else:
                demo_dest.unlink()

        demo_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_course, demo_dest)
    except OSError as exc:
        _fatal(f"cannot copy demo course to {demo_dest}: {exc}")


def _run_demo(*, yes: bool, no_download: bool) -> None:
    data_dir = config.default_data_dir()
    demo_dest = data_dir / "demo-course"
    _copy_demo_course(demo_dest)

    cfg_path = config.default_config_path()
    has_existing = cfg_path.exists() or cfg_path.is_symlink()

    if not has_existing:
        summary = setup_flow.scan_roots([demo_dest], Config())
        _print_scan_summary(summary)
        try:
            setup_flow.write_config(
                cfg_path,
                setup_flow.render_config([demo_dest]),
                force=False,
            )
            typer.echo(f"Config written to {cfg_path}")
            if not no_download:
                _pull_models(yes=yes)
            active_cfg = config.load_config(cfg_path)
        except (ConfigError, FileExistsError, OSError) as exc:
            _fatal(str(exc))
        _index_roots(active_cfg, [demo_dest], summary.total)
        _print_examples(setup_flow.example_commands(summary))
        return

    demo_toml = cfg_path.parent / "demo.toml"
    demo_db = data_dir / "demo.db"
    summary = setup_flow.scan_roots(
        [demo_dest], Config(roots=(demo_dest,), db_path=demo_db)
    )
    _print_scan_summary(summary)
    try:
        setup_flow.write_config(
            demo_toml,
            setup_flow.render_config([demo_dest], db_path=demo_db),
            force=True,
        )
        typer.echo(f"Demo config written to {demo_toml}")
        if not no_download:
            _pull_models(yes=yes, config_path=demo_toml)
        active_cfg = config.load_config(demo_toml)
    except (ConfigError, FileExistsError, OSError) as exc:
        _fatal(str(exc))
    _index_roots(active_cfg, [demo_dest], summary.total)
    env_prefix = f"SEEKR_CONFIG={shlex.quote(str(demo_toml))} "
    _print_examples(setup_flow.example_commands(summary), env_prefix=env_prefix)


def init(
    paths: list[Path] = typer.Argument(
        None,
        metavar="PATH",
        help="Folder(s) with your notes or code to index.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Accept prompts without asking.",
    ),
    no_download: bool = typer.Option(
        False,
        "--no-download",
        help="Skip downloading models and code grammars.",
    ),
    demo: bool = typer.Option(
        False,
        "--demo",
        help="Copy and index the packaged demo course.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Replace roots in an existing config file.",
    ),
) -> None:
    """Set up seekr, write config.toml, download models, and build the first index."""
    raw_paths = list(paths) if isinstance(paths, (list, tuple)) and paths else None
    yes_flag = yes if isinstance(yes, bool) else False
    no_download_flag = no_download if isinstance(no_download, bool) else False
    demo_flag = demo if isinstance(demo, bool) else False
    force_flag = force if isinstance(force, bool) else False

    if demo_flag:
        if raw_paths:
            _fatal("cannot combine PATH arguments with --demo")
        _run_demo(yes=yes_flag, no_download=no_download_flag)
        return

    resolved_roots = _resolve_user_roots(raw_paths, yes=yes_flag)

    cfg_path = config.default_config_path()
    if cfg_path.is_file() and not force_flag:
        _fatal(f"config already exists at {cfg_path}; use --force to replace it")
    if (cfg_path.exists() or cfg_path.is_symlink()) and not cfg_path.is_file():
        _fatal(f"config path exists and is not a regular file: {cfg_path}")

    try:
        base_cfg = (
            config.load_config(cfg_path)
            if (cfg_path.is_file() and force_flag)
            else Config()
        )
    except ConfigError as exc:
        _fatal(str(exc))

    summary = setup_flow.scan_roots(resolved_roots, base_cfg)
    _print_scan_summary(summary)

    try:
        if cfg_path.is_file():
            setup_flow.replace_roots(cfg_path, resolved_roots)
        else:
            setup_flow.write_config(
                cfg_path,
                setup_flow.render_config(resolved_roots),
                force=False,
            )
    except (ApiError, ConfigError, FileExistsError, OSError) as exc:
        message = exc.message if isinstance(exc, ApiError) else str(exc)
        _fatal(message)

    typer.echo(f"Config written to {cfg_path}")

    if not no_download_flag:
        _pull_models(yes=yes_flag)

    try:
        active_cfg = config.load_config(cfg_path)
    except ConfigError as exc:
        _fatal(str(exc))

    _index_roots(active_cfg, list(active_cfg.roots), summary.total)
    _print_examples(setup_flow.example_commands(summary))


def register(app: typer.Typer) -> None:
    """Register the ``init`` command on *app*."""
    app.command("init")(init)
