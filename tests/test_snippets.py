"""Tests for word-aligned snippets and Porter-stemmed query highlights."""

from __future__ import annotations

import unittest

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from seekr.snippets import Snippet, make_snippet, stem_terms

_HYPOTHESIS_SETTINGS = settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

_WORD_ST = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=12)
_SEP_ST = st.sampled_from([" ", " ", " ", ". ", "! ", "? "])

LONG_TEXT = (
    "Linear regression fits a hyperplane to minimise squared residuals. "
    "Regularisation shrinks large weights toward zero. "
    "Cross validation estimates generalisation error across folds. "
    "Gradient descent updates parameters iteratively along the negative gradient. "
    "The bias–variance tradeoff explains why overly flexible models overfit noisy samples "
    "while overly rigid models underfit true structure. "
    "Feature scaling often accelerates convergence."
)


def _appears_at_word_boundaries(haystack: str, needle: str) -> bool:
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            return False
        left_ok = idx == 0 or haystack[idx - 1].isspace()
        end_idx = idx + len(needle)
        right_ok = end_idx == len(haystack) or haystack[end_idx].isspace()
        if left_ok and right_ok:
            return True
        start = idx + 1


class SnippetTests(unittest.TestCase):
    @_HYPOTHESIS_SETTINGS
    @given(
        words=st.lists(_WORD_ST, min_size=0, max_size=90),
        seps=st.lists(_SEP_ST, min_size=0, max_size=90),
        query_words=st.lists(_WORD_ST, min_size=0, max_size=5),
    )
    def test_never_cuts_words(
        self,
        words: list[str],
        seps: list[str],
        query_words: list[str],
    ) -> None:
        if not words:
            text = ""
        else:
            parts = [words[0]]
            for idx, word in enumerate(words[1:]):
                sep = seps[idx % len(seps)] if seps else " "
                parts.append(sep + word)
            text = "".join(parts)
        query = " ".join(query_words)

        snippet = make_snippet(text, query)
        self.assertIsInstance(snippet, Snippet)
        self.assertLessEqual(len(snippet.text), 242)
        body = snippet.text.strip("…")
        self.assertTrue(body == "" or _appears_at_word_boundaries(text, body))

        prev_end = 0
        for start, end in snippet.highlights:
            self.assertGreaterEqual(start, prev_end)
            self.assertLess(start, end)
            self.assertLessEqual(end, len(snippet.text))
            self.assertTrue(stem_terms(snippet.text[start:end]) <= stem_terms(query))
            prev_end = end

    def test_stemmed_highlights(self) -> None:
        snippet = make_snippet("The model learned slowly. Learning rates matter.", "learning")
        marked = [snippet.text[a:b] for a, b in snippet.highlights]
        self.assertEqual(marked, ["learned", "Learning"])

    def test_prefers_sentence_with_most_terms(self) -> None:
        snippet = make_snippet(LONG_TEXT, "bias variance")
        self.assertTrue(snippet.text.lstrip("…").startswith("The bias–variance tradeoff"))
        marked = [snippet.text[a:b].lower() for a, b in snippet.highlights]
        self.assertEqual(marked, ["bias", "variance"])

    def test_no_overlap_uses_first_sentence(self) -> None:
        snippet = make_snippet("First sentence here. Second one.", "zebra")
        self.assertTrue(snippet.text.startswith("First sentence here."))
        self.assertEqual(snippet.highlights, ())

    def test_strips_markdown_noise(self) -> None:
        heading = make_snippet("## Heading\n- item text", "item")
        self.assertFalse(heading.text.startswith("#"))
        self.assertEqual(heading.text, "Heading item text")
        self.assertEqual([heading.text[a:b] for a, b in heading.highlights], ["item"])

        quoted = make_snippet("> ## Quoted heading\n* bullet point", "bullet")
        self.assertEqual(quoted.text, "Quoted heading bullet point")
        self.assertEqual([quoted.text[a:b] for a, b in quoted.highlights], ["bullet"])

    def test_empty_query_for_similar(self) -> None:
        snippet = make_snippet("First sentence here. Second sentence later.", "")
        self.assertEqual(snippet.highlights, ())
        self.assertTrue(snippet.text.startswith("First sentence here."))

    def test_shifts_window_when_match_is_far_into_sentence(self) -> None:
        sentence = " ".join(f"word{i}" for i in range(60)) + " targetterm " + " ".join(
            f"after{i}" for i in range(20)
        )
        snippet = make_snippet(sentence, "targetterm")
        self.assertTrue(snippet.text.startswith("…"))
        self.assertLessEqual(len(snippet.text), 242)
        self.assertIn("targetterm", snippet.text)
        marked = [snippet.text[a:b] for a, b in snippet.highlights]
        self.assertEqual(marked, ["targetterm"])
        body = snippet.text.strip("…")
        self.assertTrue(_appears_at_word_boundaries(sentence, body))

    def test_single_long_token_without_whitespace_never_empty(self) -> None:
        text = "z" * 300
        snippet = make_snippet(text, "absent")
        self.assertEqual(snippet.text, ("z" * 240) + "…")
        self.assertEqual(snippet.highlights, ())

    def test_stem_terms_and_blank_text(self) -> None:
        self.assertEqual(stem_terms("Learning LEARNED"), {"learn"})
        self.assertEqual(make_snippet("   \n\t  ", "learning"), Snippet("", ()))


if __name__ == "__main__":
    unittest.main()
