"""Tests for the expanded ``seekr doctor`` diagnostics."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from seekr.config import Config, ConfigError
from seekr.doctor import DoctorCheck, DoctorProbes, DoctorReport, run_doctor
from seekr.platform_info import Mount

_HAS_TYPER = importlib.util.find_spec("typer") is not None


class _FakeRows:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def fetchone(self) -> tuple[object, ...] | None:
        return self._rows[0] if self._rows else None


class _HealthySqliteConnection:
    def execute(self, sql: str) -> _FakeRows:
        if sql.startswith("PRAGMA compile_options"):
            return _FakeRows([("ENABLE_FTS5",)])
        if sql.startswith("SELECT vec_version"):
            return _FakeRows([("v0.1.0",)])
        raise AssertionError(f"unexpected SQL: {sql}")

    def enable_load_extension(self, enabled: bool) -> None:
        pass

    def close(self) -> None:
        pass


def _healthy_sqlite_kwargs() -> dict[str, object]:
    return {
        "connect": _HealthySqliteConnection,
        "vector_loader": lambda conn: None,
    }


class DoctorExpandedTests(unittest.TestCase):
    def test_doctor_info_checks_do_not_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Config(roots=(root,))
            probes = DoctorProbes(
                model_cached=lambda asset: asset.purpose == "embedding",
                grammars_missing=lambda: ["python", "rust"],
                mounts=list,
                is_wsl=lambda: False,
                daemon_state=lambda: "stopped (/tmp/seekr.sock)",
            )
            report = run_doctor(cfg=cfg, probes=probes, **_healthy_sqlite_kwargs())  # type: ignore[arg-type]

        self.assertTrue(report.ok, report.render())
        checks = {check.name: check for check in report.checks}
        self.assertEqual(checks["config"].severity, "error")
        self.assertTrue(checks["config"].ok)
        self.assertEqual(checks["roots"].severity, "error")
        self.assertTrue(checks["roots"].ok)
        self.assertEqual(checks["embedding model"].severity, "error")
        self.assertTrue(checks["embedding model"].ok)
        self.assertEqual(checks["reranker model"].severity, "info")
        self.assertFalse(checks["reranker model"].ok)
        self.assertEqual(checks["reranker model"].hint, "seekr models pull --rerank")
        self.assertEqual(checks["code grammars"].severity, "info")
        self.assertFalse(checks["code grammars"].ok)
        self.assertEqual(checks["code grammars"].hint, "seekr models pull --grammars")
        self.assertNotIn("windows drive roots", checks)
        self.assertEqual(checks["daemon"].severity, "info")
        self.assertFalse(checks["daemon"].ok)

        rendered = report.render()
        self.assertIn("[info] reranker model:", rendered)
        self.assertIn("[info] code grammars:", rendered)
        self.assertIn("[info] daemon:", rendered)
        self.assertNotIn("[FAIL]", rendered)
        self.assertIn("doctor: all checks passed", rendered)

        # Direct DoctorReport check with info failure vs error failure.
        info_only = DoctorReport(
            checks=(DoctorCheck("code grammars", False, "missing", "hint", severity="info"),)
        )
        self.assertTrue(info_only.ok)
        self.assertIn("[info] code grammars:", info_only.render())

    def test_doctor_missing_model_fails_with_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(roots=(Path(tmp),))
            probes = DoctorProbes(
                model_cached=lambda asset: False,
                grammars_missing=list,
                mounts=list,
                is_wsl=lambda: False,
                daemon_state=lambda: "running (/tmp/seekr.sock)",
            )
            report = run_doctor(cfg=cfg, probes=probes, **_healthy_sqlite_kwargs())  # type: ignore[arg-type]

        self.assertFalse(report.ok)
        checks = {check.name: check for check in report.checks}
        self.assertFalse(checks["embedding model"].ok)
        self.assertEqual(checks["embedding model"].severity, "error")
        self.assertEqual(checks["embedding model"].hint, "seekr models pull")
        self.assertTrue(checks["daemon"].ok)

        rendered = report.render()
        self.assertIn("[FAIL] embedding model:", rendered)
        self.assertIn("hint: seekr models pull", rendered)
        self.assertIn("doctor: one or more checks failed", rendered)

    def test_doctor_env_only(self) -> None:
        def boom(*_args: object, **_kwargs: object):
            raise AssertionError("probes must not be called when env_only=True")

        probes = DoctorProbes(
            model_cached=boom,
            grammars_missing=boom,
            mounts=boom,
            is_wsl=boom,
            daemon_state=boom,
        )
        report = run_doctor(
            env_only=True,
            probes=probes,
            **_healthy_sqlite_kwargs(),  # type: ignore[arg-type]
        )

        self.assertTrue(report.ok, report.render())
        self.assertEqual(
            [check.name for check in report.checks],
            ["FTS5", "loadable extensions", "sqlite-vec"],
        )

        if _HAS_TYPER:
            from typer.testing import CliRunner

            from seekr.cli import app

            captured: list[bool] = []

            def fake_run_doctor(*, env_only: bool = False, **_kwargs: object) -> DoctorReport:
                captured.append(env_only)
                return DoctorReport((DoctorCheck("FTS5", True, "ok"),))

            with mock.patch("seekr.doctor.run_doctor", side_effect=fake_run_doctor):
                res = CliRunner().invoke(app, ["doctor", "--env-only"])
            self.assertEqual(res.exit_code, 0, res.output)
            self.assertEqual(captured, [True])

    def test_doctor_wsl_polled_roots_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            polled_dir = Path(tmp) / "mnt" / "c" / "course"
            polled_dir.mkdir(parents=True)
            cfg = Config(roots=(polled_dir,), poll_interval_seconds=15)
            mounts = [
                Mount(source="/dev/sda1", mount_point="/", fstype="ext4"),
                Mount(source="C:\\", mount_point=str(Path(tmp) / "mnt" / "c"), fstype="drvfs"),
            ]
            probes = DoctorProbes(
                model_cached=lambda _asset: True,
                grammars_missing=list,
                mounts=lambda: mounts,
                is_wsl=lambda: True,
                daemon_state=lambda: "running",
            )
            report = run_doctor(cfg=cfg, probes=probes, **_healthy_sqlite_kwargs())  # type: ignore[arg-type]

        self.assertTrue(report.ok, report.render())
        checks = {check.name: check for check in report.checks}
        self.assertIn("windows drive roots", checks)
        wsl_check = checks["windows drive roots"]
        self.assertEqual(wsl_check.severity, "info")
        self.assertFalse(wsl_check.ok)
        self.assertIn("polled every 15s:", wsl_check.detail)
        self.assertIn(str(polled_dir), wsl_check.detail)
        self.assertIn("[info] windows drive roots: polled every 15s:", report.render())

        # On WSL with no slow mounts, windows drive roots passes as ok.
        probes_fast = DoctorProbes(
            model_cached=lambda _asset: True,
            grammars_missing=list,
            mounts=lambda: [Mount(source="/dev/sda1", mount_point="/", fstype="ext4")],
            is_wsl=lambda: True,
            daemon_state=lambda: "running",
        )
        fast_report = run_doctor(cfg=cfg, probes=probes_fast, **_healthy_sqlite_kwargs())  # type: ignore[arg-type]
        fast_checks = {check.name: check for check in fast_report.checks}
        self.assertTrue(fast_checks["windows drive roots"].ok)

    def test_doctor_roots_validation(self) -> None:
        probes = DoctorProbes(
            model_cached=lambda _asset: True,
            grammars_missing=list,
            mounts=list,
            is_wsl=lambda: False,
            daemon_state=lambda: "running",
        )
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            valid_dir = base / "valid"
            valid_dir.mkdir()
            missing_dir = base / "missing"
            regular_file = base / "file.txt"
            regular_file.write_text("hi", encoding="utf-8")

            ok_report = run_doctor(
                cfg=Config(roots=(valid_dir,)),
                probes=probes,
                **_healthy_sqlite_kwargs(),  # type: ignore[arg-type]
            )
            self.assertTrue(ok_report.ok, ok_report.render())

            missing_report = run_doctor(
                cfg=Config(roots=(missing_dir,)),
                probes=probes,
                **_healthy_sqlite_kwargs(),  # type: ignore[arg-type]
            )
            self.assertFalse(missing_report.ok)
            missing_checks = {c.name: c for c in missing_report.checks}
            self.assertFalse(missing_checks["roots"].ok)
            self.assertEqual(missing_checks["roots"].severity, "error")

            file_report = run_doctor(
                cfg=Config(roots=(regular_file,)),
                probes=probes,
                **_healthy_sqlite_kwargs(),  # type: ignore[arg-type]
            )
            self.assertFalse(file_report.ok)
            file_checks = {c.name: c for c in file_report.checks}
            self.assertFalse(file_checks["roots"].ok)

            if os.geteuid() != 0:
                unreadable = base / "unreadable"
                unreadable.mkdir()
                unreadable.chmod(0o000)
                try:
                    unreadable_report = run_doctor(
                        cfg=Config(roots=(unreadable,)),
                        probes=probes,
                        **_healthy_sqlite_kwargs(),  # type: ignore[arg-type]
                    )
                finally:
                    unreadable.chmod(0o755)
                self.assertFalse(unreadable_report.ok)
                unreadable_checks = {c.name: c for c in unreadable_report.checks}
                self.assertFalse(unreadable_checks["roots"].ok)

        # Invalid config fails gracefully in config and roots checks.
        with mock.patch("seekr.doctor.load_config", side_effect=ConfigError("bad toml")):
            bad_cfg_report = run_doctor(
                probes=probes,
                **_healthy_sqlite_kwargs(),  # type: ignore[arg-type]
            )
        self.assertFalse(bad_cfg_report.ok)
        bad_checks = {c.name: c for c in bad_cfg_report.checks}
        self.assertFalse(bad_checks["config"].ok)
        self.assertEqual(bad_checks["config"].severity, "error")
        self.assertFalse(bad_checks["roots"].ok)


    def test_doctor_detail_names_backend(self) -> None:
        seen: list[object] = []

        def cached(asset: object) -> bool:
            seen.append(asset)
            return True

        for backend in ("onnx", "torch"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                seen.clear()
                probes = DoctorProbes(
                    model_cached=cached,
                    grammars_missing=list,
                    mounts=list,
                    is_wsl=lambda: False,
                    daemon_state=lambda: "stopped (/tmp/seekr.sock)",
                )
                cfg = Config(roots=(Path(tmp),), embed_backend=backend)
                report = run_doctor(cfg=cfg, probes=probes, **_healthy_sqlite_kwargs())  # type: ignore[arg-type]
                checks = {check.name: check for check in report.checks}
                self.assertIn(f"{backend} backend", checks["embedding model"].detail)
                self.assertIn(f"{backend} backend", checks["reranker model"].detail)
                self.assertEqual({a.backend for a in seen}, {backend})  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
