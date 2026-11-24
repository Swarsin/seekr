"""Tests for pick-line formatting, parsing, opening and the rofi popup."""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from seekr import cli_open
from seekr import opener as opener_module
from seekr.models import Hit
from seekr.opener import (
    OPEN_SUCCESS_CODES,
    Argv,
    OpenFailed,
    StartFile,
    format_pick_line,
    open_command,
    open_target,
    parse_pick_line,
    run_open,
)
from seekr.platform_info import Mount

ROOT = Path(__file__).resolve().parents[1]
POPUP_PATH = ROOT / "contrib" / "seekr-popup.sh"


def _hit(
    path: str,
    locator: str = "",
    snippet: str = "a snippet",
    kind: str = "text",
) -> Hit:
    return Hit(
        chunk_id=1,
        path=path,
        locator=locator,
        loc_start=1.0,
        snippet=snippet,
        score=1.0,
        kind=kind,
    )


def _run_popup(rofi_body: str, seekr_body: str) -> subprocess.CompletedProcess[str]:
    """Run the popup with fake rofi/seekr commands first on PATH."""
    with tempfile.TemporaryDirectory() as directory:
        bin_dir = Path(directory)
        for name, body in (("rofi", rofi_body), ("seekr", seekr_body)):
            script = bin_dir / name
            script.write_text(body, encoding="utf-8")
            script.chmod(0o755)
        env = dict(os.environ)
        env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
        return subprocess.run(
            [shutil.which("bash") or "bash", str(POPUP_PATH)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )


class PickLineTests(unittest.TestCase):
    def test_raw_windows_drive_path_is_taken_as_typed(self) -> None:
        win = {"platform": "win32"}
        self.assertEqual(
            parse_pick_line(r"C:\Users\me\a.py:L10-12", **win), (r"C:\Users\me\a.py", "L10-12")
        )
        self.assertEqual(
            parse_pick_line("C:/x/talk.vtt:00:01:02\tsnippet", **win), ("C:/x/talk.vtt", "00:01:02")
        )
        self.assertEqual(parse_pick_line(r"D:\notes\a.md", **win), (r"D:\notes\a.md", ""))
        # Escaped pick lines still round-trip on Windows.
        hit = _hit(r"C:\Users\me\a.md", "§Methods: Results")
        self.assertEqual(
            parse_pick_line(format_pick_line(hit), **win), (r"C:\Users\me\a.md", "§Methods: Results")
        )
        # Elsewhere a backslash is an escape and the first colon ends the path.
        self.assertEqual(parse_pick_line(r"C:\\x:L1", platform="linux"), ("C", r"\x:L1"))

    def test_pick_line_roundtrip_with_colons_in_path(self) -> None:
        path = "/a/b:c/d.md"
        locator = "§Intro"

        line = format_pick_line(_hit(path, locator))

        self.assertEqual(parse_pick_line(line), (path, locator))

    def test_pick_line_collapses_newlines(self) -> None:
        line = format_pick_line(_hit("/notes.md", "§Intro", "first\nsecond\r\nthird"))

        self.assertNotIn("\n", line)
        self.assertNotIn("\r", line)
        self.assertTrue(line.endswith("\tfirst second third"))

    def test_pick_line_shows_common_colon_locators_verbatim(self) -> None:
        transcript = format_pick_line(_hit("/video.vtt", "00:01:02", "hello"))
        heading = format_pick_line(_hit("/notes.md", "§Methods: Results", "body"))

        self.assertEqual(transcript.split("\t", 1)[0], "/video.vtt:00:01:02")
        self.assertEqual(heading.split("\t", 1)[0], "/notes.md:§Methods: Results")

    def test_pick_line_roundtrips_escaped_path_and_locator_data(self) -> None:
        cases = [
            ("/a/b:c/d.md", "§Intro"),
            ("/tmp/a\tb\nc\r.md", "00:01:02"),
            ("/tmp/100%/über:😀.md", "§Methods: Results > Part 1"),
            ("/tmp/back\\slash.md", "a\\b:c"),
            ("/tmp/control\x01\x1b\u2028.md", "loc\x00:ator\x7f"),
            ("-leading: colon.md", "L1-2"),
        ]

        for path, locator in cases:
            with self.subTest(path=path, locator=locator):
                line = format_pick_line(_hit(path, locator, "a\nb\tc"))
                self.assertEqual(line.count("\t"), 1)
                self.assertEqual(len(line.splitlines()), 1)
                self.assertEqual(parse_pick_line(line), (path, locator))

        visible = format_pick_line(
            _hit("/tmp/100%/über:😀.md", "§Methods: Results > Part 1", "body")
        )
        self.assertIn("100%", visible)
        self.assertIn("über", visible)
        self.assertIn("😀", visible)
        self.assertIn("§Methods: Results > Part 1", visible)

    def test_pick_line_empty_locator_keeps_the_separator(self) -> None:
        line = format_pick_line(_hit("/a/b:c.txt", "", "snippet"))

        self.assertEqual(line, "/a/b\\:c.txt:\tsnippet")
        self.assertEqual(parse_pick_line(line), ("/a/b:c.txt", ""))

    def test_parse_raw_path_locator_is_compatible_for_unambiguous_inputs(self) -> None:
        self.assertEqual(
            parse_pick_line("/tmp/video.vtt:00:01:02"),
            ("/tmp/video.vtt", "00:01:02"),
        )
        self.assertEqual(
            parse_pick_line("/tmp/notes.md:§Methods: Results"),
            ("/tmp/notes.md", "§Methods: Results"),
        )
        self.assertEqual(parse_pick_line("/tmp/plain.txt"), ("/tmp/plain.txt", ""))

    def test_snippet_is_sanitised_and_bounded(self) -> None:
        snippet = ("word\n\t\x1b" * 500) + "tail"
        line = format_pick_line(_hit("/notes.md", "§Intro", snippet))
        rendered = line.split("\t", 1)[1]

        self.assertLessEqual(len(rendered), 240)
        self.assertNotIn("\x1b", rendered)
        self.assertTrue(rendered.startswith("word word"))
        self.assertTrue(rendered.endswith("…"))


class OpenCommandTests(unittest.TestCase):
    def test_open_command_pdf_zathura_page(self) -> None:
        calls: list[str] = []

        def which(name: str) -> str | None:
            calls.append(name)
            return "/usr/bin/zathura"

        argv = open_command("/papers/paper.pdf", "p.12", "pdf", {}, which)

        self.assertEqual(argv, ["zathura", "-P", "12", "/papers/paper.pdf"])
        self.assertEqual(calls, ["zathura"])

    def test_open_command_pdf_fallback_xdg_open(self) -> None:
        argv = open_command("/papers/paper.pdf", "p.12", "pdf", {}, lambda _name: None)

        self.assertEqual(argv, ["xdg-open", "/papers/paper.pdf"])

    def test_open_command_code_line_uses_editor_env(self) -> None:
        self.assertEqual(
            open_command("file.py", "L10-20", "code", {"EDITOR": "nano"}, lambda _: None),
            ["nano", "+10", "file.py"],
        )
        self.assertEqual(
            open_command("file.py", "L10-20", "code", {}, lambda _: None),
            ["nvim", "+10", "file.py"],
        )

    def test_open_command_editor_env_is_split_without_a_shell(self) -> None:
        argv = open_command("file.py", "L3-4", "code", {"EDITOR": "code --wait"}, lambda _: None)
        malformed = open_command(
            "file.py", "L3-4", "code", {"EDITOR": "'unterminated"}, lambda _: None
        )

        self.assertEqual(argv, ["code", "--wait", "+3", "file.py"])
        self.assertEqual(malformed, ["nvim", "+3", "file.py"])

    def test_open_command_text_line_uses_editor(self) -> None:
        argv = open_command("notes.txt", "L7-9", "text", {}, lambda _: None)

        self.assertEqual(argv, ["nvim", "+7", "notes.txt"])

    def test_open_command_transcript_uses_xdg_open(self) -> None:
        calls: list[str] = []

        def which(name: str) -> str | None:
            calls.append(name)
            return "/usr/bin/zathura"

        argv = open_command("lecture.vtt", "00:01:02", "transcript", {}, which)

        self.assertEqual(argv, ["xdg-open", "lecture.vtt"])
        self.assertEqual(calls, [])

    def test_open_command_unknown_or_unlocated_uses_xdg_open(self) -> None:
        self.assertEqual(
            open_command("archive.bin", "", "binary", {}, lambda _: None),
            ["xdg-open", "archive.bin"],
        )
        self.assertEqual(
            open_command("paper.pdf", "not-a-page", "pdf", {}, lambda _: None),
            ["xdg-open", "paper.pdf"],
        )
        self.assertEqual(
            open_command("notes.md", "", "markdown", {}, lambda _: None),
            ["xdg-open", "notes.md"],
        )

    def test_open_command_prefixes_leading_hyphen_paths(self) -> None:
        self.assertEqual(
            open_command("-notes.md", "L1-2", "markdown", {}, lambda _: None),
            ["nvim", "+1", "./-notes.md"],
        )
        self.assertEqual(
            open_command("-paper.pdf", "p.2", "pdf", {}, lambda _: "zathura"),
            ["zathura", "-P", "2", "./-paper.pdf"],
        )
        self.assertEqual(
            open_command("-file.bin", "", "binary", {}, lambda _: None),
            ["xdg-open", "./-file.bin"],
        )

    def test_open_command_neutralises_leading_plus_editor_commands(self) -> None:
        for path in ("+!cmd", "+foo", "+"):
            with self.subTest(path=path):
                argv = open_command(path, "L3-4", "markdown", {"EDITOR": "nano"}, lambda _: None)
                self.assertEqual(argv, ["nano", "+3", f"./{path}"])

    def test_wsl_open_prefers_wslview(self) -> None:
        self.assertEqual(
            open_command(
                "/mnt/c/Users/me/a.pdf",
                "p.14",
                "pdf",
                {"EDITOR": "nano"},
                lambda name: "/usr/bin/wslview" if name == "wslview" else "/usr/bin/zathura",
                wsl=True,
                to_windows=lambda _: r"C:\Users\me\a.pdf",
            ),
            ["wslview", "/mnt/c/Users/me/a.pdf"],
        )
        self.assertEqual(
            open_command(
                "/mnt/c/Users/me/code.py",
                "L10-20",
                "code",
                {"EDITOR": "nano"},
                lambda name: "/usr/bin/wslview" if name == "wslview" else None,
                wsl=True,
                to_windows=lambda _: r"C:\Users\me\code.py",
            ),
            ["wslview", "/mnt/c/Users/me/code.py"],
        )
        self.assertEqual(
            open_command(
                "-report.pdf",
                "p.2",
                "pdf",
                {},
                lambda name: "/usr/bin/wslview" if name == "wslview" else None,
                wsl=True,
                to_windows=lambda _: None,
            ),
            ["wslview", "./-report.pdf"],
        )

    def test_wsl_open_falls_back_to_explorer(self) -> None:
        self.assertEqual(
            open_command(
                "/mnt/c/Users/me/a.pdf",
                "p.14",
                "pdf",
                {"EDITOR": "nano"},
                lambda name: "/usr/bin/zathura" if name == "zathura" else None,
                wsl=True,
                to_windows=lambda p: r"C:\Users\me\a.pdf" if p == "/mnt/c/Users/me/a.pdf" else None,
            ),
            ["explorer.exe", r"C:\Users\me\a.pdf"],
        )
        self.assertEqual(
            open_command(
                "/mnt/c/Users/me/code.py",
                "L10-20",
                "code",
                {"EDITOR": "nano"},
                lambda _: None,
                wsl=True,
                to_windows=lambda _: r"C:\Users\me\code.py",
            ),
            ["explorer.exe", r"C:\Users\me\code.py"],
        )

    def test_wsl_open_unconvertible_raises(self) -> None:
        for converter in (lambda _: None, lambda _: "", None):
            with (
                self.subTest(converter=converter),
                self.assertRaisesRegex(
                    ValueError,
                    r"cannot convert /home/me/a\.pdf to a Windows path; install wslu for wslview",
                ),
            ):
                open_command(
                    "/home/me/a.pdf",
                    "p.1",
                    "pdf",
                    {},
                    lambda _: None,
                    wsl=True,
                    to_windows=converter,
                )


class MarkdownHeadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "notes.md"
        self.path.write_text(
            "# Intro\n"
            "\n"
            "Some intro text.\n"
            "\n"
            "# Methods: Results\n"
            "\n"
            "More text.\n"
            "\n"
            "## Nested: Detail\n"
            "\n"
            "Detail text.\n",
            encoding="utf-8",
        )

    def test_open_command_markdown_heading_uses_exact_editor_line(self) -> None:
        self.assertEqual(
            open_command(str(self.path), "§Intro", "markdown", {}, lambda _: None),
            ["nvim", "+1", str(self.path)],
        )
        self.assertEqual(
            open_command(str(self.path), "§Methods: Results", "markdown", {}, lambda _: None),
            ["nvim", "+5", str(self.path)],
        )
        self.assertEqual(
            open_command(
                str(self.path),
                "§Methods: Results > Nested: Detail",
                "markdown",
                {},
                lambda _: None,
            ),
            ["nvim", "+9", str(self.path)],
        )

    def test_missing_heading_or_file_falls_back_to_xdg_open(self) -> None:
        self.assertEqual(
            open_command(str(self.path), "§Missing", "markdown", {}, lambda _: None),
            ["xdg-open", str(self.path)],
        )
        missing = Path(self.dir.name) / "missing.md"
        self.assertEqual(
            open_command(str(missing), "§Intro", "markdown", {}, lambda _: None),
            ["xdg-open", str(missing)],
        )


class _FakeExit(Exception):
    def __init__(self, code: int = 0) -> None:
        super().__init__(code)
        self.exit_code = code


class _FakeTyper(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("typer")
        self.output: list[str] = []
        self.Argument = lambda default, **_kwargs: default
        self.Option = lambda default, *_args, **_kwargs: default
        self.Exit = _FakeExit
        self.echo = self._echo

    def _echo(self, *args: Any, **_kwargs: Any) -> None:
        self.output.append(" ".join(str(arg) for arg in args))


class _FakeApp:
    def __init__(self) -> None:
        self.commands: dict[str, Any] = {}

    def command(self, name: str) -> Any:
        def decorator(function: Any) -> Any:
            self.commands[name] = function
            return function

        return decorator


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        # The CLI cases below check the Linux and WSL opener rules; Windows
        # and macOS hosts would otherwise pick their native opener.
        linux_rules = functools.partial(opener_module.open_target, platform="linux")
        patcher = mock.patch.object(cli_open, "open_target", linux_rules)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _registered(self) -> tuple[_FakeApp, _FakeTyper]:
        fake_typer = _FakeTyper()
        app = _FakeApp()
        with mock.patch.dict(sys.modules, {"typer": fake_typer}):
            cli_open.register(app)
        return app, fake_typer

    def test_register_adds_pick_and_open(self) -> None:
        app, _ = self._registered()

        self.assertEqual(set(app.commands), {"pick", "open"})

    def test_pick_uses_shared_get_hits_and_prints_pick_lines(self) -> None:
        app, fake_typer = self._registered()
        hits = [
            _hit("/notes/a.md", "§A", "first\nsnippet", "markdown"),
            _hit("/video.vtt", "00:01:02", "second", "transcript"),
        ]

        with mock.patch.object(cli_open, "get_hits", return_value=hits) as get_hits:
            app.commands["pick"](query="needle")

        get_hits.assert_called_once_with(
            "needle",
            k=10,
            mode="hybrid",
            per_file=2,
            rerank=False,
            path_prefix=None,
        )
        self.assertEqual(
            fake_typer.output,
            [format_pick_line(hit) for hit in hits],
        )

    def test_open_command_runs_expected_argv_without_a_shell(self) -> None:
        app, _ = self._registered()
        py_file = Path(tempfile.gettempdir()) / "seekr-open-test.py"
        completed = types.SimpleNamespace(returncode=0)

        with (
            mock.patch("seekr.platform_info.is_wsl", return_value=False),
            mock.patch.dict(os.environ, {"EDITOR": "nano"}),
            mock.patch.object(opener_module.subprocess, "run", return_value=completed) as run,
            mock.patch.object(opener_module.subprocess, "Popen") as popen,
        ):
            app.commands["open"](target=f"{py_file}:L10-12")

        # The CLI waits for the editor: blocking run, never a detached Popen.
        run.assert_called_once_with(["nano", "+10", str(py_file)], check=False)
        popen.assert_not_called()
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_open_never_runs_a_script_hit_on_windows_or_macos(self) -> None:
        app, _ = self._registered()
        completed = types.SimpleNamespace(returncode=0)
        expected = {"win32": ["notepad.exe"], "darwin": ["open", "-t"]}
        for platform, prefix in expected.items():
            for name in ("hit.py", "hit.js"):
                with self.subTest(platform=platform, name=name):
                    script = str(Path(tempfile.gettempdir()) / name)
                    rules = functools.partial(opener_module.open_target, platform=platform)
                    env = {k: v for k, v in os.environ.items() if k not in ("EDITOR", "SEEKR_OPENER")}
                    with (
                        mock.patch.object(cli_open, "open_target", rules),
                        mock.patch("seekr.platform_info.is_wsl", return_value=False),
                        mock.patch.dict(os.environ, env, clear=True),
                        mock.patch.object(os, "startfile", create=True) as startfile,
                        mock.patch.object(opener_module.subprocess, "run", return_value=completed) as run,
                    ):
                        app.commands["open"](target=f"{script}:L3")
                    startfile.assert_not_called()
                    run.assert_called_once_with([*prefix, script], check=False)

    def test_open_accepts_a_pick_line_and_never_fakes_transcript_media(self) -> None:
        app, _ = self._registered()
        vtt_file = Path(tempfile.gettempdir()) / "seekr-open-test.vtt"
        line = format_pick_line(_hit(str(vtt_file), "00:01:02", "hello", "transcript"))
        completed = types.SimpleNamespace(returncode=0)

        with (
            mock.patch("seekr.platform_info.is_wsl", return_value=False),
            mock.patch.object(opener_module.subprocess, "run", return_value=completed) as run,
        ):
            app.commands["open"](target=line)

        run.assert_called_once_with(["xdg-open", str(vtt_file)], check=False)

    def test_open_resolves_markdown_heading_to_editor_line(self) -> None:
        app, _ = self._registered()
        with tempfile.TemporaryDirectory() as directory:
            notes = Path(directory) / "notes.md"
            notes.write_text("# Intro\n\nbody\n", encoding="utf-8")
            line = format_pick_line(_hit(str(notes), "§Intro", "body", "markdown"))
            completed = types.SimpleNamespace(returncode=0)

            with (
                mock.patch("seekr.platform_info.is_wsl", return_value=False),
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(opener_module.subprocess, "run", return_value=completed) as run,
            ):
                app.commands["open"](target=line)

        run.assert_called_once_with(["nvim", "+1", str(notes)], check=False)

    def test_open_wsl_uses_wslview_when_available(self) -> None:
        app, _ = self._registered()
        completed = types.SimpleNamespace(returncode=0)

        with (
            mock.patch("seekr.platform_info.is_wsl", return_value=True),
            mock.patch.object(
                cli_open.shutil,
                "which",
                side_effect=lambda name: "/usr/bin/wslview" if name == "wslview" else None,
            ),
            mock.patch.object(opener_module.subprocess, "run", return_value=completed) as run,
        ):
            app.commands["open"](target="/mnt/c/Users/me/a.pdf:p.14")

        run.assert_called_once_with(["wslview", "/mnt/c/Users/me/a.pdf"], check=False)

    def test_explorer_exit_code_one_is_success(self) -> None:
        self.assertEqual(OPEN_SUCCESS_CODES, {"explorer.exe": frozenset({0, 1})})
        app, _ = self._registered()
        mounts = [Mount(source="C:\\", mount_point="/mnt/c", fstype="9p")]

        with (
            mock.patch("seekr.platform_info.is_wsl", return_value=True),
            mock.patch("seekr.platform_info.read_mounts", return_value=mounts),
            mock.patch.object(cli_open.shutil, "which", return_value=None),
            mock.patch.object(
                opener_module.subprocess,
                "run",
                return_value=types.SimpleNamespace(returncode=1),
            ) as run,
        ):
            app.commands["open"](target="/mnt/c/Users/me/a.pdf:p.14")

        run.assert_called_once_with(["explorer.exe", r"C:\Users\me\a.pdf"], check=False)

        with (
            mock.patch("seekr.platform_info.is_wsl", return_value=True),
            mock.patch("seekr.platform_info.read_mounts", return_value=mounts),
            mock.patch.object(cli_open.shutil, "which", return_value=None),
            mock.patch.object(
                opener_module.subprocess,
                "run",
                return_value=types.SimpleNamespace(returncode=2),
            ),
            self.assertRaises(_FakeExit) as ctx,
        ):
            app.commands["open"](target="/mnt/c/Users/me/a.pdf:p.14")
        self.assertEqual(ctx.exception.exit_code, 2)

    def test_open_wsl_falls_back_to_wslpath_and_reports_unconvertible_error(self) -> None:
        app, fake_typer = self._registered()
        ext4_mounts = [Mount(source="/dev/sdc", mount_point="/", fstype="ext4")]

        with (
            mock.patch("seekr.platform_info.is_wsl", return_value=True),
            mock.patch("seekr.platform_info.read_mounts", return_value=ext4_mounts),
            mock.patch(
                "seekr.platform_info.wslpath_windows",
                return_value=r"\\wsl.localhost\Ubuntu\home\me\a.pdf",
            ),
            mock.patch.object(cli_open.shutil, "which", return_value=None),
            mock.patch.object(
                opener_module.subprocess,
                "run",
                return_value=types.SimpleNamespace(returncode=1),
            ) as run,
        ):
            app.commands["open"](target="/home/me/a.pdf:p.14")

        run.assert_called_once_with(
            ["explorer.exe", r"\\wsl.localhost\Ubuntu\home\me\a.pdf"], check=False
        )

        with (
            mock.patch("seekr.platform_info.is_wsl", return_value=True),
            mock.patch("seekr.platform_info.read_mounts", return_value=ext4_mounts),
            mock.patch("seekr.platform_info.wslpath_windows", return_value=None),
            mock.patch.object(cli_open.shutil, "which", return_value=None),
            mock.patch.object(opener_module.subprocess, "run") as run,
            self.assertRaises(_FakeExit) as ctx,
        ):
            app.commands["open"](target="/home/me/a.pdf:p.14")

        self.assertEqual(ctx.exception.exit_code, 1)
        run.assert_not_called()
        self.assertTrue(
            any("cannot convert /home/me/a.pdf to a Windows path" in line for line in fake_typer.output)
        )


class OpenTargetTests(unittest.TestCase):
    def test_windows_uses_startfile_and_ignores_locator(self) -> None:
        action = open_target(
            "C:\\a.pdf", "p.3", "pdf", env={}, which=lambda _n: "/x", platform="win32"
        )
        self.assertEqual(action, StartFile("C:\\a.pdf"))

    def test_windows_documents_use_the_default_application(self) -> None:
        for path, kind in (("C:\\a.docx", "word"), ("C:\\a.pptx", "slides")):
            action = open_target(path, "", kind, env={}, which=lambda _n: None, platform="win32")
            self.assertEqual(action, StartFile(path))

    def test_windows_scripts_open_in_notepad_never_the_default_verb(self) -> None:
        for path in ("C:\\x\\a.py", "C:\\x\\a.js"):
            action = open_target(
                path, "L3", "code", env={}, which=lambda _n: None, platform="win32"
            )
            self.assertEqual(action, Argv(["notepad.exe", path]))

    def test_windows_text_like_kinds_open_in_notepad(self) -> None:
        for kind in ("text", "markdown", "notebook", "transcript", ""):
            action = open_target("C:\\a.x", "", kind, env={}, which=lambda _n: None, platform="win32")
            self.assertEqual(action, Argv(["notepad.exe", "C:\\a.x"]))

    def test_macos_documents_use_open(self) -> None:
        action = open_target("/a.pdf", "p.2", "pdf", env={}, which=lambda _n: None, platform="darwin")
        self.assertEqual(action, Argv(["open", "/a.pdf"]))

    def test_macos_scripts_open_in_the_text_editor(self) -> None:
        for path in ("/x/a.py", "/x/a.js"):
            action = open_target(path, "L3", "code", env={}, which=lambda _n: None, platform="darwin")
            self.assertEqual(action, Argv(["open", "-t", path]))

    def test_macos_text_editor_path_cannot_become_an_option(self) -> None:
        action = open_target("-a.py", "", "code", env={}, which=lambda _n: None, platform="darwin")
        self.assertEqual(action, Argv(["open", "-t", "./-a.py"]))

    def test_editor_is_honoured_on_windows_and_macos(self) -> None:
        for platform, path in (("win32", "C:\\a.py"), ("darwin", "/a.js")):
            with self.subTest(platform=platform):
                env = {"EDITOR": "nano"}
                located = open_target(path, "L7", "code", env=env, which=lambda _n: None, platform=platform)
                self.assertEqual(located, Argv(["nano", "+7", path]))
                unlocated = open_target(path, "", "code", env=env, which=lambda _n: None, platform=platform)
                self.assertEqual(unlocated, Argv(["nano", path]))

    def test_windows_editor_keeps_backslashes_and_quoted_spaces(self) -> None:
        env = {"EDITOR": '"C:\\Program Files\\Notepad++\\notepad++.exe" -multiInst'}
        action = open_target("C:\\a.py", "", "code", env=env, which=lambda _n: None, platform="win32")
        self.assertEqual(
            action,
            Argv(["C:\\Program Files\\Notepad++\\notepad++.exe", "-multiInst", "C:\\a.py"]),
        )

    def test_editor_does_not_replace_the_document_application(self) -> None:
        env = {"EDITOR": "nano"}
        self.assertEqual(
            open_target("C:\\a.pdf", "p.1", "pdf", env=env, which=lambda _n: None, platform="win32"),
            StartFile("C:\\a.pdf"),
        )
        self.assertEqual(
            open_target("/a.pdf", "p.1", "pdf", env=env, which=lambda _n: None, platform="darwin"),
            Argv(["open", "/a.pdf"]),
        )

    def test_seekr_opener_wins_on_every_platform_and_kind(self) -> None:
        env = {"SEEKR_OPENER": "my-opener --flag", "EDITOR": "nano"}
        for platform in ("win32", "darwin", "linux"):
            for kind in ("pdf", "code"):
                with self.subTest(platform=platform, kind=kind):
                    action = open_target("/a.x", "L2", kind, env=env, which=lambda _n: "/x", platform=platform)
                    self.assertEqual(action, Argv(["my-opener", "--flag", "/a.x"]))

    def test_seekr_opener_wins_on_wsl(self) -> None:
        action = open_target(
            "/mnt/c/a.py", "L1", "code", env={"SEEKR_OPENER": "code"}, which=lambda _n: "/x",
            platform="linux", wsl=True, to_windows=lambda _p: "C:\\a.py",
        )
        self.assertEqual(action, Argv(["code", "/mnt/c/a.py"]))

    def test_blank_seekr_opener_is_ignored(self) -> None:
        action = open_target("/a.py", "", "code", env={"SEEKR_OPENER": "  "}, which=lambda _n: None, platform="darwin")
        self.assertEqual(action, Argv(["open", "-t", "/a.py"]))

    def test_linux_matches_open_command(self) -> None:
        env = {"EDITOR": "nano"}
        which = lambda _n: None
        action = open_target("/a.py", "L10", "code", env=env, which=which, platform="linux")
        self.assertEqual(action, Argv(open_command("/a.py", "L10", "code", env, which)))

    def test_wsl_matches_open_command(self) -> None:
        which = lambda _n: None
        to_win = lambda p: "C:\\a.pdf"
        action = open_target(
            "/mnt/c/a.pdf", "p.1", "pdf", env={}, which=which, platform="linux",
            wsl=True, to_windows=to_win,
        )
        self.assertEqual(action, Argv(["explorer.exe", "C:\\a.pdf"]))


class RunOpenTests(unittest.TestCase):
    def test_startfile_is_called(self) -> None:
        with mock.patch.object(os, "startfile", create=True) as start:
            run_open(StartFile("C:\\a.pdf"))
        start.assert_called_once_with("C:\\a.pdf")

    def test_argv_runs_without_shell(self) -> None:
        with mock.patch.object(
            opener_module.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)
        ) as run:
            run_open(Argv(["open", "/a"]))
        run.assert_called_once_with(["open", "/a"], check=False)

    def test_nowait_launches_detached_and_ignores_exit_code(self) -> None:
        with (
            mock.patch.object(opener_module.subprocess, "Popen") as popen,
            mock.patch.object(opener_module.subprocess, "run") as run,
        ):
            run_open(Argv(["xdg-open", "x"]), wait=False)
        run.assert_not_called()
        self.assertEqual(popen.call_args.args, (["xdg-open", "x"],))
        kwargs = popen.call_args.kwargs
        self.assertNotIn("shell", kwargs)
        for stream in ("stdin", "stdout", "stderr"):
            self.assertEqual(kwargs[stream], opener_module.subprocess.DEVNULL)
        self.assertTrue(kwargs.get("start_new_session") or kwargs.get("creationflags"))

    def test_nowait_launch_failure_propagates(self) -> None:
        with mock.patch.object(
            opener_module.subprocess, "Popen", side_effect=FileNotFoundError("x")
        ), self.assertRaises(FileNotFoundError):
            run_open(Argv(["nope", "x"]), wait=False)

    def test_explorer_one_ok_and_other_codes_raise(self) -> None:
        with mock.patch.object(
            opener_module.subprocess, "run", return_value=types.SimpleNamespace(returncode=1)
        ):
            run_open(Argv(["explorer.exe", "x"]))
            with self.assertRaises(OpenFailed) as ctx:
                run_open(Argv(["xdg-open", "x"]))
        self.assertEqual(ctx.exception.returncode, 1)
        self.assertEqual(ctx.exception.program, "xdg-open")


class PopupScriptTests(unittest.TestCase):
    def test_popup_script_has_two_prompts_and_cancellation_guards(self) -> None:
        source = POPUP_PATH.read_text(encoding="utf-8")

        self.assertGreaterEqual(source.count("rofi -dmenu"), 2)
        self.assertIn("seekr pick", source)
        self.assertIn("seekr open", source)
        self.assertNotIn("eval ", source)
        self.assertGreaterEqual(source.count("exit 0"), 2)

    @unittest.skipUnless(shutil.which("bash"), "bash is required for the popup tests")
    def test_popup_reports_pick_failure_instead_of_cancelling(self) -> None:
        rofi = "#!/usr/bin/env bash\nprintf 'query\\n'\nexit 0\n"
        seekr = (
            "#!/usr/bin/env bash\n"
            'if [[ "${1:-}" == "pick" ]]; then\n'
            "    printf 'pick exploded\\n' >&2\n"
            "    exit 7\n"
            "fi\n"
            "exit 0\n"
        )

        result = _run_popup(rofi, seekr)

        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("pick failed", result.stderr.lower())
        self.assertIn("pick exploded", result.stderr)

    @unittest.skipUnless(shutil.which("bash"), "bash is required for the popup tests")
    def test_popup_rofi_cancellation_is_a_quiet_zero_exit(self) -> None:
        rofi = "#!/usr/bin/env bash\nexit 1\n"
        seekr = "#!/usr/bin/env bash\nexit 0\n"

        result = _run_popup(rofi, seekr)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")

    @unittest.skipUnless(shutil.which("bash"), "bash is required for the syntax check")
    def test_popup_script_has_valid_shell_syntax(self) -> None:
        result = subprocess.run(
            [shutil.which("bash") or "bash", "-n", str(POPUP_PATH)],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
