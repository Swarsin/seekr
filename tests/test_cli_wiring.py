"""Regression tests for wiring the command modules into the root CLI."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from _cli_output import strip_ansi

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CLI_PATH = SRC / "seekr" / "cli.py"
_HAS_TYPER = importlib.util.find_spec("typer") is not None

COMMAND_MODULES = (
    "cli_daemon",
    "cli_eval",
    "cli_index",
    "cli_init",
    "cli_models",
    "cli_open",
    "cli_search",
    "cli_service",
    "cli_web",
)
MODULE_COMMANDS = (
    "daemon",
    "eval",
    "index",
    "init",
    "reindex",
    "status",
    "pick",
    "open",
    "search",
    "web",
)


class CliRegistrationSourceTests(unittest.TestCase):
    """Cheap contracts that also run when Typer is unavailable."""

    def test_command_modules_are_registered_before_main(self) -> None:
        source = CLI_PATH.read_text(encoding="utf-8")
        main_at = source.index('if __name__ == "__main__":')

        for module in COMMAND_MODULES:
            registration = f"{module}.register(app)"
            with self.subTest(registration=registration):
                self.assertIn(registration, source)
                self.assertLess(
                    source.index(registration),
                    main_at,
                    f"{registration} must run before the __main__ entry point",
                )

    def test_module_registrations_are_one_ordered_block(self) -> None:
        source = CLI_PATH.read_text(encoding="utf-8")
        registrations = [
            line.strip()
            for line in source.splitlines()
            if line.strip().endswith(".register(app)")
        ]

        self.assertEqual(
            registrations,
            [f"{module}.register(app)" for module in COMMAND_MODULES],
        )

    def test_version_and_doctor_remain_registered(self) -> None:
        source = CLI_PATH.read_text(encoding="utf-8")

        self.assertIn("--version", source)
        self.assertIn("def doctor(", source)


@unittest.skipUnless(_HAS_TYPER, "typer is not installed")
class CliRegistrationTests(unittest.TestCase):
    @staticmethod
    def _runner() -> Any:
        from typer.testing import CliRunner

        return CliRunner()

    def _invoke(self, args: list[str]) -> Any:
        from seekr.cli import app

        return self._runner().invoke(app, args)

    def test_root_help_lists_all_commands(self) -> None:
        result = self._invoke(["--help"])
        output = strip_ansi(result.output)

        self.assertEqual(result.exit_code, 0, result.output)
        for command in ("doctor", "models", *MODULE_COMMANDS):
            self.assertIn(command, output)

    def test_all_module_command_names_are_on_the_app(self) -> None:
        from seekr.cli import app

        names = {info.name for info in app.registered_commands if info.name}
        groups = {info.name for info in app.registered_groups if info.name}

        self.assertTrue(
            set(MODULE_COMMANDS) <= names,
            f"missing commands: {sorted(set(MODULE_COMMANDS) - names)}",
        )
        self.assertIn("models", groups)

    def test_models_subcommands_have_help(self) -> None:
        models_help = self._invoke(["models", "--help"])
        self.assertEqual(models_help.exit_code, 0, models_help.output)
        models_output = strip_ansi(models_help.output)
        self.assertIn("pull", models_output)
        self.assertIn("status", models_output)

        pull_help = self._invoke(["models", "pull", "--help"])
        self.assertEqual(pull_help.exit_code, 0, pull_help.output)
        pull_output = strip_ansi(pull_help.output)
        for option in ("--rerank", "--grammars", "--all", "--yes"):
            self.assertIn(option, pull_output)

        status_help = self._invoke(["models", "status", "--help"])
        self.assertEqual(status_help.exit_code, 0, status_help.output)

    def test_version_still_works(self) -> None:
        from seekr import __version__

        result = self._invoke(["--version"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn(f"seekr {__version__}", result.output)

    def test_each_registered_command_has_help(self) -> None:
        expected_options = {
            "daemon": ("--socket", "--debounce"),
            "eval": ("--root", "--queries", "--modes", "--k", "--out"),
            "index": ("PATHS",),
            "init": ("PATH", "--yes", "--no-download", "--demo", "--force"),
            "open": ("TARGET",),
            "pick": ("QUERY", "--per-file", "--rerank", "--path"),
            "reindex": ("--force", "--allow-empty"),
            "status": ("Print",),
            "search": ("QUERY", "--mode", "--per-file", "--rerank", "--path"),
            "web": ("--port", "--open-browser"),
        }
        for command, options in expected_options.items():
            with self.subTest(command=command):
                result = self._invoke([command, "--help"])
                output = strip_ansi(result.output)
                self.assertEqual(result.exit_code, 0, result.output)
                for option in options:
                    self.assertIn(option, output)

    def test_importing_cli_does_not_load_model_backends(self) -> None:
        script = (
            "import sys\n"
            "import seekr.cli\n"
            "heavy = ('http.server', 'seekr.web_server', 'numpy', "
            "'sentence_transformers', 'sqlite_vec', 'torch')\n"
            "print(' '.join(name for name in heavy if name in sys.modules))\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(SRC), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)

        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(_HAS_TYPER, "typer is not installed")
class DaemonAliasTests(unittest.TestCase):
    def test_daemon_warns_ignores_options_and_runs_service(self) -> None:
        from typer.testing import CliRunner

        from seekr.cli import app

        calls: list[dict[str, object]] = []

        class FakeService:
            def __init__(self, **kwargs: object) -> None:
                calls.append(kwargs)

            def start(self) -> int:
                return 4321

            def serve_forever(self) -> None:
                return None

        with mock.patch("seekr.cli_service._create_service", FakeService):
            result = CliRunner().invoke(
                app, ["daemon", "--socket", "/tmp/x.sock", "--debounce", "2"]
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn(
            'warning: "seekr daemon" is deprecated and will be removed in 0.4.0; '
            'use "seekr service start"',
            result.output,
        )
        self.assertIn("--socket and --debounce are ignored", result.output)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["port"], 0)
