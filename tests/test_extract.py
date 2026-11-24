"""Tests for file type detection and text extraction."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from seekr.extract import ExtractionError, extract, kind_for, language_for

try:  # PyMuPDF is a runtime dependency, but keep these tests runnable without it.
    import pymupdf as _fitz
except ImportError:  # pragma: no cover - exercised only in incomplete environments
    try:
        import fitz as _fitz
    except ImportError:
        _fitz = None


class _TemporaryDirectoryTestCase(unittest.TestCase):
    """Give each test an isolated directory that is removed afterwards."""

    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.directory = Path(self._temporary_directory.name)

    def write_text(self, name: str, text: str, encoding: str = "utf-8") -> Path:
        path = self.directory / name
        path.write_text(text, encoding=encoding, newline="")  # exact bytes, no CRLF
        return path


class KindForTests(unittest.TestCase):
    def test_kind_for_extensions(self) -> None:
        cases = {
            "notes.md": "markdown",
            "notes.markdown": "markdown",
            "notes.txt": "text",
            "notes.rst": "text",
            "main.py": "code",
            "main.c": "code",
            "main.h": "code",
            "main.cpp": "code",
            "main.hpp": "code",
            "Main.java": "code",
            "app.js": "code",
            "app.ts": "code",
            "app.tsx": "code",
            "lib.rs": "code",
            "main.go": "code",
            "paper.pdf": "pdf",
            "lecture.vtt": "transcript",
            "lecture.srt": "transcript",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(kind_for(Path(name)), expected)

    def test_kind_for_unknown_and_uppercase_extensions(self) -> None:
        self.assertIsNone(kind_for(Path("photo.png")))
        self.assertIsNone(kind_for(Path("README")))
        self.assertIsNone(kind_for(Path("archive.tar.gz")))
        self.assertEqual(kind_for(Path("REPORT.PDF")), "pdf")
        self.assertEqual(kind_for(Path("NOTES.MD")), "markdown")

    def test_language_for_code_extensions(self) -> None:
        cases = {
            "main.py": "python",
            "main.c": "c",
            "main.h": "c",
            "main.cpp": "cpp",
            "main.hpp": "cpp",
            "Main.java": "java",
            "app.js": "javascript",
            "app.ts": "typescript",
            "app.tsx": "tsx",
            "lib.rs": "rust",
            "main.go": "go",
            "MAIN.PY": "python",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(language_for(Path(name)), expected)

    def test_language_for_non_code_files_is_none(self) -> None:
        for name in ("notes.md", "notes.txt", "paper.pdf", "lecture.vtt", "photo.png"):
            with self.subTest(name=name):
                self.assertIsNone(language_for(Path(name)))


class TextExtractionTests(_TemporaryDirectoryTestCase):
    def test_text_utf8_is_exact(self) -> None:
        payload = "Hello, w\u00f6rld!\n\nSecond line.\n"
        path = self.write_text("notes.txt", payload)

        result = extract(path, "text")

        self.assertEqual(len(result.units), 1)
        unit = result.units[0]
        self.assertEqual(unit.text, payload)
        self.assertEqual(unit.locator, "")
        self.assertEqual(unit.loc_start, 0.0)
        self.assertEqual(result.warnings, [])

    def test_text_latin1_fallback_warns(self) -> None:
        path = self.directory / "latin1.txt"
        path.write_bytes(b"caf\xe9\n")

        result = extract(path, "text")

        self.assertEqual(len(result.units), 1)
        self.assertEqual(result.units[0].text, "caf\u00e9\n")
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("latin-1", result.warnings[0].lower())

    def test_binary_file_with_txt_extension_raises(self) -> None:
        path = self.directory / "binary.txt"
        path.write_bytes(b"\x00\x01\x02not really text")

        with self.assertRaises(ExtractionError) as raised:
            extract(path, "text")

        self.assertIn("binary", str(raised.exception).lower())

    def test_markdown_and_code_use_one_default_unit(self) -> None:
        cases = [
            ("markdown", "notes.md", "# Title\n\nBody\n"),
            ("code", "main.py", "def answer():\n    return 42\n"),
        ]
        for kind, name, payload in cases:
            with self.subTest(kind=kind):
                path = self.write_text(name, payload)
                result = extract(path, kind)
                self.assertEqual(len(result.units), 1)
                self.assertEqual(result.units[0].text, payload)
                self.assertEqual(result.units[0].locator, "")
                self.assertEqual(result.units[0].loc_start, 0.0)
                self.assertEqual(result.warnings, [])

    def test_missing_file_keeps_file_not_found_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            extract(self.directory / "missing.txt", "text")

    def test_directory_raises_extraction_error(self) -> None:
        with self.assertRaises(ExtractionError):
            extract(self.directory, "text")

    def test_unknown_kind_raises_extraction_error(self) -> None:
        path = self.write_text("data.bin", "content")

        with self.assertRaises(ExtractionError):
            extract(path, "spreadsheet")


class TranscriptExtractionTests(_TemporaryDirectoryTestCase):
    def test_vtt_cues(self) -> None:
        path = self.write_text(
            "lecture.vtt",
            "WEBVTT\n\n00:01:02.500 --> 00:01:05.000\n<c>hello</c> world\n",
        )

        result = extract(path, "transcript")

        self.assertEqual(len(result.units), 1)
        unit = result.units[0]
        self.assertEqual(unit.locator, "00:01:02")
        self.assertEqual(unit.loc_start, 62.5)
        self.assertEqual(unit.text, "hello world")
        self.assertEqual(result.warnings, [])

    def test_srt_cues(self) -> None:
        path = self.write_text(
            "lecture.srt",
            "1\n"
            "00:00:01,000 --> 00:00:02,500\n"
            "Hello <i>world</i>\n"
            "\n"
            "2\n"
            "00:00:02,750 --> 00:00:03,900\n"
            "Second cue\n",
        )

        result = extract(path, "transcript")

        self.assertEqual(len(result.units), 2)
        self.assertEqual(result.units[0].locator, "00:00:01")
        self.assertEqual(result.units[0].loc_start, 1.0)
        self.assertEqual(result.units[0].text, "Hello world")
        self.assertEqual(result.units[1].locator, "00:00:02")
        self.assertEqual(result.units[1].loc_start, 2.75)
        self.assertEqual(result.units[1].text, "Second cue")
        self.assertEqual(result.warnings, [])

    def test_vtt_identifiers_settings_and_entities(self) -> None:
        path = self.write_text(
            "captions.vtt",
            "WEBVTT\n"
            "Kind: captions\n"
            "Language: en\n"
            "\n"
            "intro\n"
            "00:00:00.000 --> 00:00:02.000 align:start position:0%\n"
            "<v Alice>Hello</v> &amp; welcome\n"
            "\n"
            "00:00:03.250 --> 00:00:04.000\n"
            "Second line\n"
            "continues here\n",
        )

        result = extract(path, "transcript")

        self.assertEqual(len(result.units), 2)
        self.assertEqual(result.units[0].locator, "00:00:00")
        self.assertEqual(result.units[0].text, "Hello & welcome")
        self.assertEqual(result.units[1].locator, "00:00:03")
        self.assertEqual(result.units[1].loc_start, 3.25)
        self.assertEqual(result.units[1].text, "Second line continues here")
        self.assertEqual(result.warnings, [])

    def test_malformed_timestamp_warns_and_continues(self) -> None:
        path = self.write_text(
            "broken.vtt",
            "WEBVTT\n"
            "\n"
            "00:00:bad --> 00:00:02.000\n"
            "broken cue\n"
            "\n"
            "00:00:03.000 --> 00:00:04.000\n"
            "valid cue\n",
        )

        result = extract(path, "transcript")

        self.assertEqual(len(result.units), 1)
        self.assertEqual(result.units[0].text, "valid cue")
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("malformed", result.warnings[0].lower())
        self.assertIn("line 3", result.warnings[0])

    def test_empty_cue_warns(self) -> None:
        path = self.write_text(
            "empty-cue.vtt",
            "WEBVTT\n"
            "\n"
            "00:00:01.000 --> 00:00:02.000\n"
            "<c></c>\n"
            "\n"
            "00:00:03.000 --> 00:00:04.000\n"
            "kept\n",
        )

        result = extract(path, "transcript")

        self.assertEqual(len(result.units), 1)
        self.assertEqual(result.units[0].text, "kept")
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("no text", result.warnings[0].lower())


class PdfBackendTests(_TemporaryDirectoryTestCase):
    def test_missing_pymupdf_reports_extraction_error(self) -> None:
        path = self.directory / "paper.pdf"
        path.write_bytes(b"%PDF-1.4")

        with (
            mock.patch.dict(sys.modules, {"pymupdf": None, "fitz": None}),
            self.assertRaises(ExtractionError) as raised,
        ):
            extract(path, "pdf")

        self.assertIn("pymupdf", str(raised.exception).lower())


@unittest.skipUnless(_fitz is not None, "PyMuPDF is not installed")
class PdfExtractionTests(_TemporaryDirectoryTestCase):
    def _words(self, prefix: str, count: int) -> str:
        return " ".join(f"{prefix}{index}" for index in range(count))

    def _make_pdf(self, name: str, pages: list[str | None]) -> Path:
        assert _fitz is not None
        path = self.directory / name
        document = _fitz.open()
        try:
            for page_text in pages:
                page = document.new_page()
                if page_text is not None:
                    page.insert_textbox(
                        _fitz.Rect(72, 72, 540, 720),
                        page_text,
                        fontsize=11,
                    )
            document.save(path)
        finally:
            document.close()
        return path

    def test_pdf_pages(self) -> None:
        path = self._make_pdf(
            "three-pages.pdf",
            [self._words("alpha", 25), None, self._words("gamma", 25)],
        )

        result = extract(path, "pdf")

        self.assertEqual([unit.locator for unit in result.units], ["p.1", "p.3"])
        self.assertEqual([unit.loc_start for unit in result.units], [1.0, 3.0])
        self.assertIn("alpha0", result.units[0].text)
        self.assertIn("gamma0", result.units[1].text)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("page 2", result.warnings[0])

    def test_pdf_sparse_page_threshold_is_twenty_words(self) -> None:
        path = self._make_pdf(
            "threshold.pdf",
            [self._words("sparse", 19), self._words("enough", 20)],
        )

        result = extract(path, "pdf")

        self.assertEqual([unit.locator for unit in result.units], ["p.2"])
        self.assertIn("enough0", result.units[0].text)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("page 1", result.warnings[0])

    def test_textless_pdf_warns_instead_of_raising(self) -> None:
        path = self._make_pdf("blank.pdf", [None, None])

        result = extract(path, "pdf")

        self.assertEqual(result.units, [])
        self.assertTrue(any("page 1" in warning for warning in result.warnings))
        self.assertTrue(any("page 2" in warning for warning in result.warnings))
        self.assertTrue(any("no extractable text" in warning for warning in result.warnings))

    def test_corrupt_pdf_raises(self) -> None:
        path = self.directory / "corrupt.pdf"
        path.write_bytes(b"%PDF-1.4 garbage")

        with self.assertRaises(ExtractionError) as raised:
            extract(path, "pdf")

        self.assertIn("pdf", str(raised.exception).lower())
        # A chained PyMuPDF error would pin the half-open document, keeping
        # the file locked on Windows for as long as the error is held.
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)

    def test_encrypted_pdf_raises(self) -> None:
        assert _fitz is not None
        path = self.directory / "encrypted.pdf"
        document = _fitz.open()
        try:
            page = document.new_page()
            page.insert_text((72, 72), self._words("secret", 25))
            document.save(
                path,
                encryption=_fitz.PDF_ENCRYPT_AES_256,
                owner_pw="owner-password",
                user_pw="user-password",
            )
        finally:
            document.close()

        with self.assertRaises(ExtractionError) as raised:
            extract(path, "pdf")

        self.assertIn("encrypt", str(raised.exception).lower())

    def test_missing_pdf_keeps_file_not_found_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            extract(self.directory / "missing.pdf", "pdf")


class LongPathOpenTests(_TemporaryDirectoryTestCase):
    """Files are opened through ``pathnorm.os_path`` (the Windows long-path form)."""

    def _redirect(self, real: Path):
        requested: list[str] = []

        def os_path(path: str, **_kwargs: object) -> str:
            requested.append(path)
            return str(real)

        return requested, mock.patch("seekr.pathnorm.os_path", side_effect=os_path)

    def test_text_and_transcript_open_via_os_path(self) -> None:
        real = self.write_text("real.txt", "from the long-path form")
        logical = self.directory / "logical.txt"
        requested, patch = self._redirect(real)
        with patch:
            result = extract(logical, "text")
        self.assertEqual(result.units[0].text, "from the long-path form")
        self.assertEqual(requested, [str(logical)])

    def test_notebook_opens_via_os_path(self) -> None:
        from seekr.extract_documents import extract_notebook

        notebook = (
            '{"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": '
            '[{"cell_type": "markdown", "metadata": {}, "source": ["hello cells"]}]}'
        )
        real = self.write_text("real.ipynb", notebook)
        requested, patch = self._redirect(real)
        with patch:
            result = extract_notebook(self.directory / "logical.ipynb")
        self.assertIn("hello cells", result.units[0].text)
        self.assertEqual(requested, [str(self.directory / "logical.ipynb")])

    def test_os_path_is_identity_off_windows(self) -> None:
        from seekr import pathnorm

        if sys.platform != "win32":
            long_path = str(self.directory / ("d" * 200) / ("e" * 200))
            self.assertEqual(pathnorm.os_path(long_path), long_path)


class LazyImportTests(unittest.TestCase):
    def test_importing_extract_does_not_import_pymupdf(self) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(source_root)
        script = (
            "import sys\n"
            "import seekr.extract\n"
            "print('pymupdf' in sys.modules, 'fitz' in sys.modules)\n"
        )

        completed = subprocess.run(
            [sys.executable, "-B", "-c", script],
            capture_output=True,
            check=False,
            env=env,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "False False")


if __name__ == "__main__":
    unittest.main()
