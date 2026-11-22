"""Tests for Word (.docx), PowerPoint (.pptx) and Jupyter notebook (.ipynb) extraction."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import docx
import pptx
from pptx.util import Inches

from seekr.chunking import chunk
from seekr.config import Config
from seekr.extract import ExtractionError, extract, kind_for
from seekr.extract_documents import extract_docx, extract_notebook, extract_pptx


class _TempDirTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.cfg = Config()


class KindDetectionTests(unittest.TestCase):
    def test_kind_for_new_suffixes(self) -> None:
        cases = {
            "lecture.docx": "word",
            "LECTURE.DOCX": "word",
            "deck.pptx": "slides",
            "DECK.PPTX": "slides",
            "lab.ipynb": "notebook",
            "LAB.IPYNB": "notebook",
            "legacy.doc": None,
            "legacy.ppt": None,
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(kind_for(Path(filename)), expected)


class DocxExtractionTests(_TempDirTestCase):
    def test_docx_headings_become_sections(self) -> None:
        doc = docx.Document()
        doc.add_heading("Regularisation", level=1)
        doc.add_paragraph("Penalise large weights to curb variance.")
        doc.add_heading("L2 penalty", level=2)
        doc.add_paragraph("Ridge shrinks coefficients smoothly toward zero.")
        path = self.dir / "notes.docx"
        doc.save(path)

        specs = chunk("word", extract(path, "word").units, self.cfg)

        self.assertEqual(
            [s.locator for s in specs],
            ["§Regularisation", "§Regularisation > L2 penalty"],
        )
        self.assertIn("Penalise large weights", specs[0].text)
        self.assertIn("Ridge shrinks coefficients", specs[1].text)

    def test_docx_title_style_and_deep_headings(self) -> None:
        doc = docx.Document()
        doc.add_heading("Course Overview", level=0)
        doc.add_paragraph("Welcome paragraph.")
        doc.add_heading("Deep Topic", level=7)
        doc.add_paragraph("Clamped heading content.")
        path = self.dir / "title.docx"
        doc.save(path)

        result = extract_docx(path)
        self.assertEqual(len(result.units), 1)
        lines = result.units[0].text.splitlines()
        self.assertEqual(lines[0], "# Course Overview")
        self.assertEqual(lines[2], "###### Deep Topic")

        specs = chunk("word", result.units, self.cfg)
        self.assertEqual(
            [s.locator for s in specs],
            ["§Course Overview", "§Course Overview > Deep Topic"],
        )

    def test_docx_without_headings_uses_paragraph_locator(self) -> None:
        doc = docx.Document()
        doc.add_paragraph("First plain paragraph without headings.")
        doc.add_paragraph("Second plain paragraph without headings.")
        path = self.dir / "plain.docx"
        doc.save(path)

        specs = chunk("word", extract(path, "word").units, self.cfg)

        self.assertEqual(len(specs), 1)
        self.assertTrue(specs[0].locator.startswith("¶"))
        self.assertEqual(specs[0].locator, "¶1")

    def test_docx_table_rows_are_text(self) -> None:
        doc = docx.Document()
        doc.add_paragraph("Before table.")
        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "a"
        table.cell(0, 1).text = "b"
        table.cell(1, 0).text = "c"
        table.cell(1, 1).text = "d"
        doc.add_paragraph("After table.")
        path = self.dir / "table.docx"
        doc.save(path)

        result = extract(path, "word")

        self.assertEqual(len(result.units), 1)
        text = result.units[0].text
        self.assertIn("a | b", text)
        self.assertIn("c | d", text)
        lines = text.splitlines()
        self.assertEqual(lines, ["Before table.", "a | b", "c | d", "After table."])

    def test_docx_skips_headers_and_footers(self) -> None:
        doc = docx.Document()
        section = doc.sections[0]
        section.header.paragraphs[0].text = "CONFIDENTIAL HEADER"
        section.footer.paragraphs[0].text = "PAGE FOOTER"
        doc.add_paragraph("Body content only.")
        path = self.dir / "headers.docx"
        doc.save(path)

        result = extract(path, "word")
        self.assertEqual(len(result.units), 1)
        self.assertNotIn("CONFIDENTIAL HEADER", result.units[0].text)
        self.assertNotIn("PAGE FOOTER", result.units[0].text)
        self.assertIn("Body content only.", result.units[0].text)

    def test_docx_empty_document_returns_no_units(self) -> None:
        doc = docx.Document()
        path = self.dir / "empty.docx"
        doc.save(path)

        result = extract(path, "word")
        self.assertEqual(result.units, [])
        self.assertEqual(result.warnings, [])
        self.assertEqual(chunk("word", result.units, self.cfg), [])

    def test_docx_missing_and_error_cases(self) -> None:
        with self.assertRaises(FileNotFoundError):
            extract(self.dir / "missing.docx", "word")

        subdir = self.dir / "dir.docx"
        subdir.mkdir()
        with self.assertRaises(ExtractionError):
            extract(subdir, "word")

        ole = self.dir / "encrypted.docx"
        ole.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 512)
        with self.assertRaises(ExtractionError):
            extract(ole, "word")

        with (
            mock.patch.dict(sys.modules, {"docx": None}),
            self.assertRaises(ExtractionError) as raised,
        ):
            extract(ole, "word")
        self.assertIn("python-docx", str(raised.exception).lower())


class PptxExtractionTests(_TempDirTestCase):
    def test_pptx_one_unit_per_slide_with_notes(self) -> None:
        prs = pptx.Presentation()
        slide1 = prs.slides.add_slide(prs.slide_layouts[0])
        slide1.shapes.title.text = "Linear Regression"
        slide1.placeholders[1].text = "Minimise squared residuals."

        slide2 = prs.slides.add_slide(prs.slide_layouts[1])
        slide2.shapes.title.text = "Regularisation"
        slide2.placeholders[1].text = "Add L2 weight decay."
        table = slide2.shapes.add_table(2, 2, Inches(1), Inches(3), Inches(4), Inches(1.5)).table
        table.cell(0, 0).text = "Norm"
        table.cell(0, 1).text = "Effect"
        table.cell(1, 0).text = "L1"
        table.cell(1, 1).text = "Sparsity"
        slide2.notes_slide.notes_text_frame.text = "remember the bias term"

        path = self.dir / "deck.pptx"
        prs.save(path)

        result = extract(path, "slides")
        units = result.units

        self.assertEqual(result.warnings, [])
        self.assertEqual([u.locator for u in units], ["slide 1", "slide 2"])
        self.assertEqual([u.loc_start for u in units], [1.0, 2.0])
        self.assertIn("Linear Regression", units[0].text)
        self.assertIn("Minimise squared residuals.", units[0].text)
        self.assertIn("Norm | Effect", units[1].text)
        self.assertIn("L1 | Sparsity", units[1].text)
        self.assertIn("Speaker notes: remember the bias term", units[1].text)
        self.assertEqual(units[1].text.count("Regularisation"), 1)

    def test_pptx_empty_slide_warns(self) -> None:
        prs = pptx.Presentation()
        slide1 = prs.slides.add_slide(prs.slide_layouts[0])
        slide1.shapes.title.text = "First Slide"
        prs.slides.add_slide(prs.slide_layouts[6])  # blank slide
        slide3 = prs.slides.add_slide(prs.slide_layouts[1])
        slide3.shapes.title.text = "Third Slide"
        path = self.dir / "sparse.pptx"
        prs.save(path)

        result = extract_pptx(path)

        self.assertEqual([u.locator for u in result.units], ["slide 1", "slide 3"])
        self.assertEqual([u.loc_start for u in result.units], [1.0, 3.0])
        self.assertEqual(result.warnings, ["slide 2 has no text"])

    def test_pptx_missing_and_error_cases(self) -> None:
        with self.assertRaises(FileNotFoundError):
            extract(self.dir / "missing.pptx", "slides")

        subdir = self.dir / "dir.pptx"
        subdir.mkdir()
        with self.assertRaises(ExtractionError):
            extract(subdir, "slides")

        ole = self.dir / "encrypted.pptx"
        ole.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 512)
        with self.assertRaises(ExtractionError):
            extract(ole, "slides")

        with (
            mock.patch.dict(sys.modules, {"pptx": None}),
            self.assertRaises(ExtractionError) as raised,
        ):
            extract(ole, "slides")
        self.assertIn("python-pptx", str(raised.exception).lower())


class NotebookExtractionTests(_TempDirTestCase):
    def _write_nb(self, name: str, payload: object) -> Path:
        path = self.dir / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_notebook_cells(self) -> None:
        nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# Lab 1\n", "Explain gradient descent."],
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": 1,
                    "source": "def step(w, grad):\n    return w - 0.1 * grad",
                    "outputs": [
                        {
                            "output_type": "stream",
                            "name": "stdout",
                            "text": ["EXCLUDED_OUTPUT_TOKEN"],
                        }
                    ],
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": "   \n  ",
                },
            ],
        }
        path = self._write_nb("lab.ipynb", nb)

        result = extract(path, "notebook")
        units = result.units

        self.assertEqual(result.warnings, [])
        self.assertEqual([u.locator for u in units], ["cell 1", "cell 2"])
        self.assertEqual([u.loc_start for u in units], [1.0, 2.0])
        self.assertIn("Explain gradient descent.", units[0].text)
        self.assertIn("def step(w, grad):", units[1].text)
        self.assertNotIn("EXCLUDED_OUTPUT_TOKEN", units[1].text)

    def test_notebook_raw_cell_and_sparse_numbering(self) -> None:
        nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "cells": [
                {"cell_type": "markdown", "source": ""},
                {"cell_type": "raw", "source": ["raw appendix note"]},
            ],
        }
        path = self._write_nb("raw.ipynb", nb)

        units = extract_notebook(path).units

        self.assertEqual([u.locator for u in units], ["cell 2"])
        self.assertEqual([u.loc_start for u in units], [2.0])
        self.assertEqual(units[0].text, "raw appendix note")

    def test_notebook_bad_source_type_is_extraction_error(self) -> None:
        bad_payloads: list[object] = [
            {"nbformat": 4, "cells": [{"cell_type": "code", "source": 123}]},
            {"nbformat": 4, "cells": [{"cell_type": "markdown", "source": ["ok", 42]}]},
            {"nbformat": 4, "cells": [{"cell_type": "code", "source": None}]},
            {"nbformat": 4, "cells": [{"cell_type": "code"}]},
            {"nbformat": 4, "cells": "not-a-list"},
            {"nbformat": 4},
            {"nbformat": 3, "cells": []},
            ["not", "a", "dict"],
        ]
        for idx, payload in enumerate(bad_payloads):
            with self.subTest(idx=idx):
                path = self._write_nb(f"bad_{idx}.ipynb", payload)
                with self.assertRaises(ExtractionError):
                    extract(path, "notebook")

    def test_notebook_missing_and_invalid_json(self) -> None:
        with self.assertRaises(FileNotFoundError):
            extract(self.dir / "missing.ipynb", "notebook")

        bad_json = self.dir / "corrupt.ipynb"
        bad_json.write_text("{not valid json", encoding="utf-8")
        with self.assertRaises(ExtractionError):
            extract(bad_json, "notebook")


class LazyOfficeImportTests(unittest.TestCase):
    def test_importing_extract_modules_does_not_import_docx_or_pptx(self) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(source_root)
        script = (
            "import sys\n"
            "import seekr.extract\n"
            "import seekr.extract_documents\n"
            "print('docx' in sys.modules, 'pptx' in sys.modules)\n"
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
