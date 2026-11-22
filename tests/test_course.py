"""Tests for path-based course module and week labeling."""

from __future__ import annotations

import unittest
from pathlib import Path

from seekr.course import (
    DEFAULT_WEEK_PATTERN,
    CourseLabel,
    compile_week_pattern,
    display_module,
    label_for,
    labels_signature,
)

WEEK = compile_week_pattern(DEFAULT_WEEK_PATTERN)
ROOT = Path("/c/ai-course")


class LabelForTests(unittest.TestCase):
    def check(
        self,
        rel: str,
        module: str | None,
        week: int | None,
        roots: tuple[Path, ...] = (ROOT,),
    ) -> None:
        self.assertEqual(label_for(ROOT / rel, roots, WEEK), CourseLabel(module, week))

    def test_standard_layout(self) -> None:
        self.check("machine-learning/week4/lecture4.pdf", "machine-learning", 4)

    def test_week_variants(self) -> None:
        for folder, n in [
            ("week1", 1),
            ("Week 03", 3),
            ("wk-2", 2),
            ("Week 4 - Trees", 4),
            ("W10", 10),
        ]:
            self.check(f"ml/{folder}/a.md", "ml", n)

    def test_not_weeks(self) -> None:
        for folder in ("weekly-notes", "week123", "web", "w"):
            self.check(f"ml/{folder}/a.md", "ml", None)

    def test_nearest_ancestor_wins(self) -> None:
        self.check("ml/week2/extra/week5/a.md", "ml", 5)

    def test_nested_below_week(self) -> None:
        self.check("ml/week2/labs/a.ipynb", "ml", 2)

    def test_root_is_a_module(self) -> None:
        self.assertEqual(
            label_for(Path("/c/ml/week3/a.md"), (Path("/c/ml"),), WEEK),
            CourseLabel("ml", 3),
        )

    def test_file_directly_in_root(self) -> None:
        self.check("syllabus.pdf", None, None)

    def test_file_named_like_week_directly_in_root(self) -> None:
        self.check("week1.pdf", None, None)
        self.assertEqual(
            label_for(Path("/c/ml/week1.pdf"), (Path("/c/ml"),), WEEK),
            CourseLabel(None, None),
        )

    def test_module_without_weeks(self) -> None:
        self.check("logic/notes.md", "logic", None)

    def test_outside_roots(self) -> None:
        self.assertEqual(label_for(Path("/elsewhere/a.md"), (ROOT,), WEEK), CourseLabel())

    def test_empty_roots_and_root_path_itself(self) -> None:
        self.assertEqual(label_for(ROOT / "ml/week1/a.md", (), WEEK), CourseLabel())
        self.assertEqual(label_for(ROOT, (ROOT,), WEEK), CourseLabel())

    def test_longest_root_wins(self) -> None:
        roots = (ROOT, ROOT / "ml")
        self.assertEqual(label_for(ROOT / "ml/week1/a.md", roots, WEEK), CourseLabel("ml", 1))

    def test_unicode_and_codes(self) -> None:
        self.check("Módulo Ü/week1/a.md", "Módulo Ü", 1)
        self.check("4CCS1ML/week2/a.md", "4CCS1ML", 2)


class DisplayTests(unittest.TestCase):
    def test_display(self) -> None:
        self.assertEqual(display_module("machine-learning"), "machine learning")
        self.assertEqual(display_module("4CCS1ML"), "4CCS1ML")
        self.assertEqual(display_module("data__structures"), "data structures")
        self.assertEqual(display_module("Módulo_Ü"), "Módulo Ü")


class PatternTests(unittest.TestCase):
    def test_rejects_bad_patterns(self) -> None:
        for bad in ("(", r"week\d+", "(a)(b)"):
            with self.assertRaises(ValueError):
                compile_week_pattern(bad)

    def test_case_insensitive_custom_pattern(self) -> None:
        semana_re = compile_week_pattern(r"^semana(\d+)$")
        self.assertEqual(
            label_for(ROOT / "ml/SEMANA07/a.md", (ROOT,), semana_re),
            CourseLabel("ml", 7),
        )


class SignatureTests(unittest.TestCase):
    def test_changes_with_inputs(self) -> None:
        a = labels_signature([ROOT], DEFAULT_WEEK_PATTERN, True)
        self.assertNotEqual(a, labels_signature([ROOT], DEFAULT_WEEK_PATTERN, False))
        self.assertNotEqual(a, labels_signature([ROOT, Path("/x")], DEFAULT_WEEK_PATTERN, True))
        self.assertNotEqual(a, labels_signature([ROOT], r"^semana(\d+)$", True))
        self.assertEqual(a, labels_signature([ROOT], DEFAULT_WEEK_PATTERN, True))
        self.assertEqual(
            labels_signature([ROOT, Path("/x")], DEFAULT_WEEK_PATTERN, True),
            labels_signature([Path("/x"), ROOT], DEFAULT_WEEK_PATTERN, True),
        )


if __name__ == "__main__":
    unittest.main()
