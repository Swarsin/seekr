"""Environment diagnostics used by the ``seekr doctor`` command."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

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
    """The result of one environment check."""

    name: str
    ok: bool
    detail: str
    hint: str = ""


@dataclass(frozen=True)
class DoctorReport:
    """The complete set of doctor checks."""

    checks: tuple[DoctorCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def render(self) -> str:
        lines: list[str] = []
        for check in self.checks:
            status = "ok" if check.ok else "FAIL"
            lines.append(f"[{status}] {check.name}: {check.detail}")
            if not check.ok and check.hint:
                lines.append(f"       hint: {check.hint}")
        if self.ok:
            lines.append("doctor: all checks passed")
        else:
            lines.append("doctor: one or more checks failed")
        return "\n".join(lines)


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


def run_doctor(
    *,
    connect: Callable[[], sqlite3.Connection] = _memory_connection,
    vector_loader: VectorLoader | None = None,
) -> DoctorReport:
    """Run every environment check and return a report instead of raising."""
    try:
        conn = connect()
    except sqlite3.Error as exc:
        check = DoctorCheck(
            "SQLite",
            False,
            f"could not open an in-memory database: {exc}",
            "check this Python's sqlite3 module and SQLite build",
        )
        return DoctorReport((check,))

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

    return DoctorReport(tuple(checks))


def main() -> int:
    """Print the doctor report and return a process exit code."""
    report = run_doctor()
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
