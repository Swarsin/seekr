"""Tests for cross-course 'Appears in' spread grouping and formatting."""

from __future__ import annotations

import unittest

from seekr.models import Hit
from seekr.spread import (
    SPREAD_CANDIDATES,
    SpreadGroup,
    compute_spread,
    format_spread,
    spread_to_dict,
)


def _hit(chunk_id: int, module: str | None, week: int | None) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        path=f"/course/{module or 'root'}/{week or 'other'}/{chunk_id}.md",
        locator="L1",
        loc_start=1.0,
        snippet="topic match",
        score=1.0 / max(1, chunk_id),
        kind="markdown",
        module=module,
        week=week,
    )


class SpreadTests(unittest.TestCase):
    def test_spread_candidates_constant(self) -> None:
        self.assertEqual(SPREAD_CANDIDATES, 30)

    def test_groups_ordered_with_first_marked(self) -> None:
        hits = [
            _hit(1, "ml", 4),
            _hit(2, "ml", 2),
            _hit(3, "ds", 1),
            _hit(4, None, None),
            _hit(5, "ml", None),
        ]
        groups = compute_spread(hits)
        self.assertEqual(
            [(g.module, g.week, g.first) for g in groups],
            [
                ("ds", 1, True),
                ("ml", 2, True),
                ("ml", 4, False),
                ("ml", None, False),
                (None, None, False),
            ],
        )
        self.assertEqual(groups[0].chunk_ids, (3,))
        self.assertEqual(groups[1].chunk_ids, (2,))
        self.assertEqual(groups[2].chunk_ids, (1,))
        self.assertEqual(groups[3].chunk_ids, (5,))
        self.assertEqual(groups[4].chunk_ids, (4,))

    def test_deduplicates_chunk_ids_and_orders_modules_case_insensitively(self) -> None:
        hits = [
            _hit(10, "Beta", 3),
            _hit(11, "alpha", 5),
            _hit(12, "alpha", 5),
            _hit(11, "alpha", 5),
            _hit(13, None, 2),
            _hit(14, None, 4),
        ]
        groups = compute_spread(hits)
        self.assertEqual(
            groups,
            [
                SpreadGroup(module="alpha", week=5, first=True, chunk_ids=(11, 12)),
                SpreadGroup(module="Beta", week=3, first=True, chunk_ids=(10,)),
                SpreadGroup(module=None, week=2, first=False, chunk_ids=(13,)),
                SpreadGroup(module=None, week=4, first=False, chunk_ids=(14,)),
            ],
        )

    def test_empty_hits_returns_empty(self) -> None:
        self.assertEqual(compute_spread([]), [])
        self.assertEqual(format_spread([]), "")

    def test_spread_to_dict_shape(self) -> None:
        group = SpreadGroup(module="ml", week=2, first=True, chunk_ids=(2, 9))
        self.assertEqual(
            spread_to_dict(group),
            {"module": "ml", "week": 2, "first": True, "chunk_ids": [2, 9]},
        )

    def test_format(self) -> None:
        groups = compute_spread(
            [
                _hit(1, "machine-learning", 4),
                _hit(2, "machine-learning", 2),
                _hit(3, "machine-learning", None),
                _hit(4, "4CCS1ML", 1),
                _hit(5, None, None),
            ]
        )
        formatted = format_spread(groups)
        expected = (
            "Appears in: 4CCS1ML: Week 1 (first)\n"
            "Appears in: machine learning: Week 2 (first) · Week 4 · other\n"
            "Appears in: other: other"
        )
        self.assertEqual(formatted, expected)
        self.assertIn("machine learning: Week 2 (first) · Week 4", formatted)


if __name__ == "__main__":
    unittest.main()
