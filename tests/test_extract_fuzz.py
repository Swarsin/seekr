"""Fuzz and adversarial edge-case tests for text, PDF and transcript extraction."""

from __future__ import annotations

import importlib.util
import tempfile
import time
import unittest
import warnings
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from seekr.chunking import chunk
from seekr.config import Config
from seekr.extract import ExtractionError, extract

_HAS_PYMUPDF = (
    importlib.util.find_spec("pymupdf") is not None
    or importlib.util.find_spec("fitz") is not None
)

_HYPOTHESIS_SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


class MalformedPdfFuzzTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.cfg = Config()

    @unittest.skipUnless(_HAS_PYMUPDF, "PyMuPDF is required")
    def test_malformed_and_edge_pdfs_never_crash(self) -> None:
        cases: dict[str, bytes] = {
            "empty.pdf": b"",
            "header_only.pdf": b"%PDF-1.7\n",
            "truncated_trailer.pdf": b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\nstartxref\n999999\n%%EOF",
            "corrupt_xref.pdf": b"%PDF-1.5\nxref\n0 100\n0000000000 65535 f \ntrailer\n<< /Size 100 >>\n",
            "nul_filled.pdf": b"\x00" * 2048,
            "random_header.pdf": b"%PDF-1.4\n" + bytes(range(256)) * 4,
        }
        for name, payload in cases.items():
            with self.subTest(name=name):
                path = self.dir / name
                path.write_bytes(payload)
                try:
                    result = extract(path, "pdf")
                except ExtractionError:
                    continue
                chunks = chunk("pdf", result.units, self.cfg)
                self.assertIsInstance(chunks, list)

    @unittest.skipUnless(_HAS_PYMUPDF, "PyMuPDF is required")
    def test_directory_named_pdf_raises_extraction_error(self) -> None:
        bad_dir = self.dir / "bad.pdf"
        bad_dir.mkdir()

        with self.assertRaises(ExtractionError):
            extract(bad_dir, "pdf")


@unittest.skipUnless(_HAS_PYMUPDF, "PyMuPDF is required")
@_HYPOTHESIS_SETTINGS
@given(payload=st.binary(min_size=0, max_size=4096))
def test_fuzz_random_bytes_as_pdf_either_raises_extraction_error_or_chunks(payload: bytes) -> None:
    cfg = Config()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fuzz.pdf"
        path.write_bytes(payload)
        try:
            result = extract(path, "pdf")
        except ExtractionError:
            return
        chunks = chunk("pdf", result.units, cfg)
        assert all(item.text.strip() for item in chunks)


class OddEncodingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.cfg = Config()

    def test_utf8_bom_markdown_strips_bom_for_heading_detection(self) -> None:
        path = self.dir / "bom.md"
        path.write_bytes("# Intro Title\n\nSome searchable content.\n".encode("utf-8-sig"))

        result = extract(path, "markdown")
        chunks = chunk("markdown", result.units, self.cfg)

        self.assertEqual(result.warnings, [])
        self.assertFalse(result.units[0].text.startswith("\ufeff"))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].locator, "§Intro Title")

    def test_utf16_le_and_be_boms_decode_with_warning(self) -> None:
        text = "Alpha beta gamma line.\nSecond line here.\n"
        for name, enc in (("le.txt", "utf-16-le"), ("be.txt", "utf-16-be")):
            bom = b"\xff\xfe" if enc == "utf-16-le" else b"\xfe\xff"
            with self.subTest(enc=enc):
                path = self.dir / name
                path.write_bytes(bom + text.encode(enc))
                result = extract(path, "text")
                self.assertEqual(result.units[0].text, text)
                self.assertEqual(len(result.warnings), 1)
                self.assertIn("utf-16", result.warnings[0].lower())
                chunks = chunk("text", result.units, self.cfg)
                self.assertEqual(len(chunks), 1)

    def test_latin1_shift_jis_and_invalid_utf8_never_crash(self) -> None:
        cases = {
            "latin1.txt": b"na\xefve r\xe9sum\xe9\n",
            "shiftjis.txt": "日本語テスト\n".encode("shift_jis"),
            "invalid_utf8.md": b"# Header\n\xff\xfe\xfd\x80\x81 body text\n",
            "truncated_utf16_bom.txt": b"\xff\xfeA\x00B",
        }
        for name, payload in cases.items():
            kind = "markdown" if name.endswith(".md") else "text"
            with self.subTest(name=name):
                path = self.dir / name
                path.write_bytes(payload)
                try:
                    result = extract(path, kind)
                except ExtractionError:
                    continue
                chunks = chunk(kind, result.units, self.cfg)
                self.assertTrue(all(item.text.strip() for item in chunks))

    def test_nul_bytes_without_utf16_bom_raise_extraction_error(self) -> None:
        path = self.dir / "blob.txt"
        path.write_bytes(b"header\x00binarypayload")

        with self.assertRaises(ExtractionError):
            extract(path, "text")


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    kind=st.sampled_from(["text", "markdown", "code", "transcript"]),
    payload=st.binary(min_size=0, max_size=4096),
)
def test_fuzz_random_bytes_as_text_kinds(kind: str, payload: bytes) -> None:
    cfg = Config()
    suffix = {
        "text": ".txt",
        "markdown": ".md",
        "code": ".py",
        "transcript": ".vtt",
    }[kind]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"fuzz{suffix}"
        path.write_bytes(payload)
        try:
            result = extract(path, kind)
        except ExtractionError:
            return
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            chunks = chunk(kind, result.units, cfg)
        assert all(item.text.strip() for item in chunks)


class HugeLineAndEmptyFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.cfg = Config()

    def test_huge_single_lines_extract_and_chunk_quickly(self) -> None:
        cases = [
            ("huge_token.txt", "text", "x" * 500_000),
            ("huge_words.txt", "text", " ".join(f"w{i}" for i in range(20_000))),
            ("huge_heading.md", "markdown", "# " + "h" * 100_000 + "\n" + " ".join(["body"] * 5000)),
            (
                "huge_cue.vtt",
                "transcript",
                "WEBVTT\n\n00:00:01.000 --> 00:00:10.000\n" + " ".join(["word"] * 10_000) + "\n",
            ),
        ]
        for name, kind, content in cases:
            with self.subTest(name=name):
                path = self.dir / name
                path.write_text(content, encoding="utf-8")
                started = time.monotonic()
                result = extract(path, kind)
                chunks = chunk(kind, result.units, self.cfg)
                elapsed = time.monotonic() - started
                self.assertLess(elapsed, 2.0, f"{name} took {elapsed:.2f}s")
                self.assertGreater(len(chunks), 0)
                self.assertTrue(all(item.text.strip() for item in chunks))

    def test_empty_and_whitespace_files_produce_no_chunks(self) -> None:
        cases = [
            ("empty.txt", "text", ""),
            ("ws.txt", "text", "   \n\t\n "),
            ("empty.md", "markdown", ""),
            ("ws.md", "markdown", "\n\n   \n"),
            ("empty.py", "code", ""),
            ("empty.vtt", "transcript", ""),
            ("ws.srt", "transcript", "\n   \n"),
        ]
        for name, kind, content in cases:
            with self.subTest(name=name):
                path = self.dir / name
                path.write_text(content, encoding="utf-8")
                result = extract(path, kind)
                chunks = chunk(kind, result.units, self.cfg, language="python" if kind == "code" else None)
                self.assertEqual(chunks, [])


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    lines=st.lists(
        st.one_of(
            st.sampled_from(
                [
                    "WEBVTT",
                    "NOTE comment block",
                    "STYLE",
                    "REGION",
                    "00:00:01.000 --> 00:00:03.500",
                    "01:02:03,400 --> 01:02:09,000 align:start",
                    "99:99:99 --> 00:00:01",
                    "not-a-timestamp --> invalid",
                    "<v Alice>Hello &amp; welcome</v>",
                    "",
                ]
            ),
            st.text(min_size=0, max_size=80),
        ),
        min_size=0,
        max_size=30,
    )
)
def test_fuzz_transcript_lines_extract_and_chunk(lines: list[str]) -> None:
    cfg = Config()
    payload = "\n".join(lines).replace("\x00", "")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fuzz.vtt"
        path.write_text(payload, encoding="utf-8", errors="replace")
        result = extract(path, "transcript")
        chunks = chunk("transcript", result.units, cfg)
        assert all(item.text.strip() for item in chunks)


def _sample_docx_bytes() -> bytes:
    import io

    import docx

    doc = docx.Document()
    doc.add_heading("Topic", level=1)
    doc.add_paragraph("Sample body text for truncation fuzzing.")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _sample_pptx_bytes() -> bytes:
    import io

    import pptx

    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Deck Title"
    slide.placeholders[1].text = "Slide body text for truncation fuzzing."
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


_VALID_DOCX_BYTES = _sample_docx_bytes()
_VALID_PPTX_BYTES = _sample_pptx_bytes()


@_HYPOTHESIS_SETTINGS
@given(
    kind=st.sampled_from(["word", "slides", "notebook"]),
    payload=st.binary(min_size=0, max_size=4096),
)
def test_fuzz_random_bytes_as_document_kinds(kind: str, payload: bytes) -> None:
    cfg = Config()
    suffix = {"word": ".docx", "slides": ".pptx", "notebook": ".ipynb"}[kind]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"fuzz{suffix}"
        path.write_bytes(payload)
        try:
            result = extract(path, kind)
        except ExtractionError:
            return
        chunks = chunk(kind, result.units, cfg)
        assert all(item.text.strip() for item in chunks)


@_HYPOTHESIS_SETTINGS
@given(
    kind=st.sampled_from(["word", "slides"]),
    data=st.data(),
)
def test_fuzz_truncated_office_archives(kind: str, data: st.DataObject) -> None:
    cfg = Config()
    sample = _VALID_DOCX_BYTES if kind == "word" else _VALID_PPTX_BYTES
    suffix = ".docx" if kind == "word" else ".pptx"
    cutoff = data.draw(st.integers(min_value=0, max_value=len(sample) - 1))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"truncated{suffix}"
        path.write_bytes(sample[:cutoff])
        try:
            result = extract(path, kind)
        except ExtractionError:
            return
        chunks = chunk(kind, result.units, cfg)
        assert all(item.text.strip() for item in chunks)


class CorruptOfficeAndNotebookTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.cfg = Config()

    def test_corrupt_xml_inside_office_archives_raises_extraction_error(self) -> None:
        import io
        import zipfile

        for kind, sample, target_entry, ext in (
            ("word", _VALID_DOCX_BYTES, "word/document.xml", ".docx"),
            ("slides", _VALID_PPTX_BYTES, "ppt/presentation.xml", ".pptx"),
        ):
            with self.subTest(kind=kind):
                buf = io.BytesIO()
                with (
                    zipfile.ZipFile(io.BytesIO(sample), "r") as zin,
                    zipfile.ZipFile(buf, "w") as zout,
                ):
                    for item in zin.infolist():
                        content = (
                            b"<broken-xml"
                            if item.filename == target_entry
                            else zin.read(item.filename)
                        )
                        zout.writestr(item, content)
                path = self.dir / f"corrupt{ext}"
                path.write_bytes(buf.getvalue())
                with self.assertRaises(ExtractionError):
                    extract(path, kind)

    def test_docx_missing_required_openxml_attribute_raises_extraction_error(self) -> None:
        import io
        import zipfile

        for target_entry, old_attr in (
            ("word/document.xml", b'w:val="Heading1"'),
            ("word/styles.xml", b'w:val="Normal"'),
        ):
            with self.subTest(target_entry=target_entry):
                buf = io.BytesIO()
                with (
                    zipfile.ZipFile(io.BytesIO(_VALID_DOCX_BYTES), "r") as zin,
                    zipfile.ZipFile(buf, "w") as zout,
                ):
                    for item in zin.infolist():
                        data = zin.read(item.filename)
                        if item.filename == target_entry:
                            self.assertIn(old_attr, data)
                            data = data.replace(old_attr, b"")
                        zout.writestr(item, data)
                path = self.dir / f"schema_{Path(target_entry).stem}.docx"
                path.write_bytes(buf.getvalue())
                with self.assertRaises(ExtractionError):
                    extract(path, "word")

    def test_truncated_notebook_and_bad_cell_shapes_raise_extraction_error(self) -> None:
        cases = {
            "truncated.ipynb": b'{"nbformat": 4, "cells": [{"cell_type": "code"',
            "bad_source.ipynb": b'{"nbformat": 4, "cells": [{"cell_type": "code", "source": 42}]}',
            "non_dict_cell.ipynb": b'{"nbformat": 4, "cells": ["not-a-cell"]}',
            "invalid_utf8.ipynb": b'{"nbformat": 4, "cells": []}\xff\xfe',
        }
        for name, payload in cases.items():
            with self.subTest(name=name):
                path = self.dir / name
                path.write_bytes(payload)
                with self.assertRaises(ExtractionError):
                    extract(path, "notebook")

