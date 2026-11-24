"""Environment and readiness diagnostics used by the ``seekr doctor`` command."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import platform_info
from .config import Config, ConfigError, default_config_path, load_config
from .platform_info import Mount, is_polled_root
from .provision import ModelAsset, assets_for, is_cached, missing_grammars

VectorLoader = Callable[[sqlite3.Connection], None]

_HINT_NO_LOAD_EXTENSION = (
    "this Python was built without loadable SQLite extension support; "
    "use a CPython build with extension support (for example a uv-managed CPython)"
)
_HINT_NO_FTS5 = (
    "this SQLite build lacks FTS5; use a Python build whose SQLite includes FTS5"
)
_HINT_NO_SQLITE_VEC = (
    "install the sqlite-vec package in the environment that runs seekr, then re-run doctor"
)


@dataclass(frozen=True)
class DoctorCheck:
    """The result of one environment or readiness check."""

    name: str
    ok: bool
    detail: str
    hint: str = ""
    severity: Literal["error", "info"] = "error"


@dataclass(frozen=True)
class DoctorReport:
    """The complete set of doctor checks."""

    checks: tuple[DoctorCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks if check.severity == "error")

    def render(self) -> str:
        lines: list[str] = []
        for check in self.checks:
            if check.ok:
                status = "ok"
            elif check.severity == "info":
                status = "info"
            else:
                status = "FAIL"
            lines.append(f"[{status}] {check.name}: {check.detail}")
            if not check.ok and check.hint:
                lines.append(f"       hint: {check.hint}")
        if self.ok:
            lines.append("doctor: all checks passed")
        else:
            lines.append("doctor: one or more checks failed")
        return "\n".join(lines)


def _default_daemon_state() -> str:
    from .daemon import _socket_is_active, socket_path, socket_path_problem

    sock = socket_path()
    problem = socket_path_problem(sock)
    if problem is not None:
        return f"unavailable ({problem})"
    try:
        active = _socket_is_active(sock)
    except OSError:
        return f"stopped ({sock})"
    return f"running ({sock})" if active else f"stopped ({sock})"


@dataclass
class DoctorProbes:
    """Offline probes injected into :func:`run_doctor`."""

    model_cached: Callable[[ModelAsset], bool] = is_cached
    grammars_missing: Callable[[], list[str]] = missing_grammars
    mounts: Callable[[], list[Mount]] = platform_info.read_mounts
    is_wsl: Callable[[], bool] = platform_info.is_wsl
    daemon_state: Callable[[], str] = _default_daemon_state


def _memory_connection() -> sqlite3.Connection:
    return sqlite3.connect(":memory:")


def _default_vector_loader(conn: sqlite3.Connection) -> None:
    import sqlite_vec

    sqlite_vec.load(conn)


def _check_fts5(conn: sqlite3.Connection) -> DoctorCheck:
    try:
        options = [str(row[0]) for row in conn.execute("PRAGMA compile_options")]
    except sqlite3.Error as exc:
        return DoctorCheck("FTS5", False, f"could not read compile options: {exc}", _HINT_NO_FTS5)
    if "ENABLE_FTS5" in options:
        return DoctorCheck("FTS5", True, "SQLite was compiled with FTS5")
    return DoctorCheck("FTS5", False, "SQLite was not compiled with ENABLE_FTS5", _HINT_NO_FTS5)


def _check_loadable_extensions(conn: sqlite3.Connection) -> DoctorCheck:
    try:
        conn.enable_load_extension(True)
    except (AttributeError, sqlite3.NotSupportedError) as exc:
        return DoctorCheck(
            "loadable extensions",
            False,
            f"extension loading is unsupported: {exc}",
            _HINT_NO_LOAD_EXTENSION,
        )
    except sqlite3.Error as exc:
        return DoctorCheck(
            "loadable extensions",
            False,
            f"could not enable extension loading: {exc}",
            _HINT_NO_LOAD_EXTENSION,
        )
    return DoctorCheck(
        "loadable extensions",
        True,
        "Python supports loading SQLite extensions",
    )


def _check_sqlite_vec(conn: sqlite3.Connection, loader: VectorLoader) -> DoctorCheck:
    row: tuple[object, ...] | None = None
    try:
        loader(conn)
        row = conn.execute("SELECT vec_version()").fetchone()
    except (ImportError, OSError, RuntimeError, sqlite3.Error) as exc:
        # Doctor reports expected extension-loading failures instead of crashing.
        return DoctorCheck(
            "sqlite-vec",
            False,
            f"failed to load or probe: {exc}",
            _HINT_NO_SQLITE_VEC,
        )

    version = str(row[0]) if row and row[0] is not None else "unknown"
    return DoctorCheck("sqlite-vec", True, f"loaded and probed (vec_version {version})")


def _sqlite_checks(
    connect: Callable[[], sqlite3.Connection],
    vector_loader: VectorLoader | None,
) -> list[DoctorCheck]:
    try:
        conn = connect()
    except sqlite3.Error as exc:
        return [
            DoctorCheck(
                "SQLite",
                False,
                f"could not open an in-memory database: {exc}",
                "check this Python's sqlite3 module and SQLite build",
            )
        ]

    checks: list[DoctorCheck] = []
    try:
        checks.append(_check_fts5(conn))
        if not hasattr(sqlite3.Connection, "enable_load_extension"):
            checks.append(
                DoctorCheck(
                    "loadable extensions",
                    False,
                    "sqlite3.Connection has no enable_load_extension()",
                    _HINT_NO_LOAD_EXTENSION,
                )
            )
            checks.append(
                DoctorCheck(
                    "sqlite-vec",
                    False,
                    "not checked because extension loading is unavailable",
                    _HINT_NO_LOAD_EXTENSION,
                )
            )
        else:
            extension_check = _check_loadable_extensions(conn)
            checks.append(extension_check)
            if extension_check.ok:
                try:
                    checks.append(
                        _check_sqlite_vec(conn, vector_loader or _default_vector_loader)
                    )
                finally:
                    try:
                        conn.enable_load_extension(False)
                    except (AttributeError, sqlite3.Error):
                        pass
            else:
                checks.append(
                    DoctorCheck(
                        "sqlite-vec",
                        False,
                        "not checked because extension loading is unavailable",
                        _HINT_NO_LOAD_EXTENSION,
                    )
                )
    finally:
        conn.close()
    return checks


def _check_config(cfg: Config | None) -> tuple[DoctorCheck, Config | None]:
    if cfg is not None:
        return (
            DoctorCheck(
                "config",
                True,
                f"loaded ({len(cfg.roots)} root(s))",
                severity="error",
            ),
            cfg,
        )

    config_path = default_config_path()
    try:
        loaded = load_config()
    except ConfigError as exc:
        return (
            DoctorCheck(
                "config",
                False,
                str(exc),
                f"fix or remove {config_path}",
                severity="error",
            ),
            None,
        )
    detail = (
        f"loaded {config_path}"
        if config_path.is_file()
        else f"using defaults ({config_path} not present)"
    )
    return DoctorCheck("config", True, detail, severity="error"), loaded


def _check_roots(cfg: Config | None) -> DoctorCheck:
    if cfg is None:
        return DoctorCheck(
            "roots",
            False,
            "not checked because config could not be loaded",
            "fix configuration errors and re-run `seekr doctor`",
            severity="error",
        )
    if not cfg.roots:
        return DoctorCheck(
            "roots",
            True,
            "none configured (add `roots = [...]` to config.toml or pass paths to `seekr index`)",
            severity="error",
        )
    problems: list[str] = []
    for raw_root in cfg.roots:
        root = Path(raw_root).expanduser()
        if not root.exists():
            problems.append(f"{root} (missing)")
            continue
        if not root.is_dir():
            problems.append(f"{root} (not a directory)")
            continue
        try:
            with os.scandir(root):
                pass
        except OSError as exc:
            problems.append(f"{root} (not readable: {exc})")
    if problems:
        return DoctorCheck(
            "roots",
            False,
            "; ".join(problems),
            "update `roots` in config.toml or check directory permissions",
            severity="error",
        )
    unit = "directory" if len(cfg.roots) == 1 else "directories"
    return DoctorCheck("roots", True, f"{len(cfg.roots)} readable {unit}", severity="error")


def _check_models(cfg: Config | None, probes: DoctorProbes) -> list[DoctorCheck]:
    if cfg is None:
        return [
            DoctorCheck(
                "embedding model",
                False,
                "not checked because config could not be loaded",
                "seekr models pull",
                severity="error",
            ),
            DoctorCheck(
                "reranker model",
                False,
                "not checked because config could not be loaded",
                "seekr models pull --rerank",
                severity="info",
            ),
        ]

    checks: list[DoctorCheck] = []
    for asset in assets_for(cfg, rerank=True):
        cached = probes.model_cached(asset)
        if asset.purpose == "embedding":
            checks.append(
                DoctorCheck(
                    "embedding model",
                    cached,
                    f"{asset.repo_id} ({'cached' if cached else 'not cached'}, "
                    f"{asset.backend} backend)",
                    "" if cached else "seekr models pull",
                    severity="error",
                )
            )
        else:
            checks.append(
                DoctorCheck(
                    "reranker model",
                    cached,
                    f"{asset.repo_id} "
                    f"({'cached' if cached else 'not cached; optional'}, {asset.backend} backend)",
                    "" if cached else "seekr models pull --rerank",
                    severity="info",
                )
            )
    return checks


def _check_grammars(probes: DoctorProbes) -> DoctorCheck:
    missing = probes.grammars_missing()
    if not missing:
        return DoctorCheck(
            "code grammars",
            True,
            "all supported AST grammars installed",
            severity="info",
        )
    return DoctorCheck(
        "code grammars",
        False,
        f"missing: {', '.join(missing)}",
        "seekr models pull --grammars",
        severity="info",
    )


def _check_wsl_roots(cfg: Config | None, probes: DoctorProbes) -> DoctorCheck | None:
    if not probes.is_wsl():
        return None
    if cfg is None or not cfg.roots:
        return DoctorCheck(
            "windows drive roots",
            True,
            "none configured",
            severity="info",
        )

    mounts = probes.mounts()
    polled: list[str] = []
    for raw_root in cfg.roots:
        if is_polled_root(raw_root, mounts):
            polled.append(str(Path(raw_root).expanduser()))

    if not polled:
        return DoctorCheck(
            "windows drive roots",
            True,
            "none (all configured roots use native Linux filesystems)",
            severity="info",
        )
    return DoctorCheck(
        "windows drive roots",
        False,
        f"polled every {cfg.poll_interval_seconds}s: {', '.join(polled)}",
        severity="info",
    )


def _check_daemon(probes: DoctorProbes) -> DoctorCheck:
    state = probes.daemon_state()
    running = state.strip().lower().startswith("running")
    return DoctorCheck(
        "daemon",
        running,
        state,
        "" if running else "run `seekr daemon` for warm search and live indexing",
        severity="info",
    )


def run_doctor(
    *,
    connect: Callable[[], sqlite3.Connection] = _memory_connection,
    vector_loader: VectorLoader | None = None,
    cfg: Config | None = None,
    probes: DoctorProbes | None = None,
    env_only: bool = False,
) -> DoctorReport:
    """Run environment and readiness checks and return a report instead of raising."""
    checks: list[DoctorCheck] = _sqlite_checks(connect, vector_loader)
    if env_only:
        return DoctorReport(tuple(checks))

    active_probes = probes if probes is not None else DoctorProbes()
    config_check, loaded_cfg = _check_config(cfg)
    checks.append(config_check)
    checks.append(_check_roots(loaded_cfg))
    checks.extend(_check_models(loaded_cfg, active_probes))
    checks.append(_check_grammars(active_probes))
    wsl_check = _check_wsl_roots(loaded_cfg, active_probes)
    if wsl_check is not None:
        checks.append(wsl_check)
    checks.append(_check_daemon(active_probes))
    return DoctorReport(tuple(checks))


def main() -> int:
    """Print the doctor report and return a process exit code."""
    report = run_doctor()
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
