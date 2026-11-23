"""Tests for the CLI and doctor diagnostics."""

from __future__ import annotations

import importlib.util
import sqlite3
import unittest
from typing import Any
from unittest import mock

from seekr import __version__
from seekr.doctor import DoctorCheck, DoctorReport, run_doctor


def _has_fts5() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        options = [row[0] for row in conn.execute("PRAGMA compile_options")]
    finally:
        conn.close()
    return "ENABLE_FTS5" in options


_HAS_FTS5 = _has_fts5()
_HAS_LOAD_EXTENSION = hasattr(sqlite3.Connection, "enable_load_extension")
_HAS_TYPER = importlib.util.find_spec("typer") is not None
_HAS_SQLITE_VEC = importlib.util.find_spec("sqlite_vec") is not None


class _FakeRows:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def fetchone(self) -> tuple[object, ...] | None:
        return self._rows[0] if self._rows else None


class _RecordingConnection:
    """Duck-typed connection used to observe the extension-loading lifecycle."""

    def __init__(self) -> None:
        self.events: list[object] = []
        self.closed = False

    def execute(self, sql: str) -> _FakeRows:
        if sql.startswith("PRAGMA compile_options"):
            return _FakeRows([("ENABLE_FTS5",)])
        if sql.startswith("SELECT vec_version"):
            self.events.append("probe")
            return _FakeRows([("fake-vec",)])
        raise AssertionError(f"unexpected SQL: {sql}")

    def enable_load_extension(self, enabled: bool) -> None:
        self.events.append(("enable", enabled))

    def close(self) -> None:
        self.closed = True


class _UnsupportedExtensionConnection:
    """Duck-typed connection whose SQLite build rejects extension loading."""

    def __init__(self) -> None:
        self.disable_calls = 0
        self.closed = False

    def execute(self, sql: str) -> _FakeRows:
        if sql.startswith("PRAGMA compile_options"):
            return _FakeRows([("ENABLE_FTS5",)])
        raise AssertionError(f"unexpected SQL: {sql}")

    def enable_load_extension(self, enabled: bool) -> None:
        if enabled:
            raise sqlite3.NotSupportedError("loadable extensions are not supported")
        self.disable_calls += 1

    def close(self) -> None:
        self.closed = True


class VersionTests(unittest.TestCase):
    def test_version_constant(self) -> None:
        self.assertEqual(__version__, "0.2.0")


class DoctorReportTests(unittest.TestCase):
    def test_missing_sqlite_vec_is_reported_with_a_hint(self) -> None:
        def missing_loader(conn: sqlite3.Connection) -> None:
            raise ImportError("No module named 'sqlite_vec'")

        report = run_doctor(vector_loader=missing_loader)
        output = report.render()

        self.assertFalse(report.ok)
        self.assertIn("FTS5", output)
        self.assertIn("sqlite-vec", output)
        self.assertIn("FAIL", output)
        self.assertIn("hint:", output)

    @unittest.skipUnless(
        _HAS_FTS5 and _HAS_LOAD_EXTENSION,
        "host SQLite cannot run the stubbed doctor probe",
    )
    def test_report_passes_with_stub_vector_extension(self) -> None:
        def stub_loader(conn: sqlite3.Connection) -> None:
            conn.create_function("vec_version", 0, lambda: "test-vec")

        report = run_doctor(vector_loader=stub_loader, env_only=True)
        output = report.render()

        self.assertTrue(report.ok, output)
        self.assertIn("FTS5", output)
        self.assertIn("sqlite-vec", output)
        self.assertIn("test-vec", output)
        self.assertIn("all checks passed", output)

    @unittest.skipUnless(_HAS_LOAD_EXTENSION, "host SQLite cannot load extensions")
    def test_enable_load_extension_not_supported_is_diagnosed_as_unsupported(self) -> None:
        conn = _UnsupportedExtensionConnection()

        def unexpected_loader(target: sqlite3.Connection) -> None:
            raise AssertionError("vector loader must not run when enabling fails")

        report = run_doctor(connect=lambda: conn, vector_loader=unexpected_loader)
        checks = {check.name: check for check in report.checks}

        self.assertFalse(report.ok)
        self.assertIn("loadable extensions", checks)
        self.assertFalse(checks["loadable extensions"].ok)
        self.assertIn("unsupported", checks["loadable extensions"].detail.lower())
        self.assertIn(
            "without loadable SQLite extension support",
            checks["loadable extensions"].hint,
        )
        self.assertIn("sqlite-vec", checks)
        self.assertFalse(checks["sqlite-vec"].ok)
        self.assertIn("not checked", checks["sqlite-vec"].detail)
        self.assertNotIn("install the sqlite-vec package", checks["sqlite-vec"].hint)
        self.assertNotIn("failed to load or probe", report.render())
        self.assertEqual(conn.disable_calls, 0)
        self.assertTrue(conn.closed)

    @unittest.skipUnless(_HAS_LOAD_EXTENSION, "host SQLite cannot load extensions")
    def test_extension_loading_is_disabled_after_the_probe(self) -> None:
        conn = _RecordingConnection()

        def stub_loader(target: Any) -> None:
            target.events.append("load")

        report = run_doctor(connect=lambda: conn, vector_loader=stub_loader, env_only=True)

        self.assertTrue(report.ok, report.render())
        self.assertEqual(
            conn.events,
            [("enable", True), "load", "probe", ("enable", False)],
        )
        self.assertTrue(conn.closed)


@unittest.skipUnless(_HAS_TYPER, "typer is not installed")
class CliTests(unittest.TestCase):
    @staticmethod
    def _runner() -> Any:
        from typer.testing import CliRunner

        return CliRunner()

    def test_cli_is_a_real_typer_app(self) -> None:
        import typer

        from seekr.cli import app

        self.assertIsInstance(app, typer.Typer)

    def test_cli_version(self) -> None:
        from seekr.cli import app

        result = self._runner().invoke(app, ["--version"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("seekr 0.2.0", result.output)

    @unittest.skipUnless(
        _HAS_SQLITE_VEC and _HAS_FTS5 and _HAS_LOAD_EXTENSION,
        "sqlite_vec and a loadable FTS5 SQLite are required",
    )
    def test_doctor_passes_here(self) -> None:
        from seekr.cli import app

        result = self._runner().invoke(app, ["doctor", "--env-only"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("FTS5", result.output)
        self.assertIn("sqlite-vec", result.output)

    def test_doctor_failure_exits_nonzero(self) -> None:
        from seekr.cli import app

        report = DoctorReport(
            checks=(DoctorCheck("FTS5", False, "SQLite was not compiled with FTS5", "hint"),)
        )
        with mock.patch("seekr.doctor.run_doctor", return_value=report):
            result = self._runner().invoke(app, ["doctor"])

        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("FTS5", result.output)


if __name__ == "__main__":
    unittest.main()
