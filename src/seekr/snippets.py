"""Word-aligned snippet selection and Porter-stemmed query highlights."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from functools import lru_cache

import snowballstemmer

__all__ = ["Snippet", "make_snippet", "stem_terms"]

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_LINE_NOISE_RE = re.compile(r"^[ \t]*(?:#{1,6}[ \t]+|>[ \t]+|[-*+][ \t]+)+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_STEMMER = snowballstemmer.stemmer("porter")
_STEMMER_LOCK = threading.Lock()


@dataclass(frozen=True)
class Snippet:
    """A snippet bounded to complete words with highlighted token spans."""

    text: str
    highlights: tuple[tuple[int, int], ...] = ()


@lru_cache(maxsize=8192)
def _stem_token(folded: str) -> str:
    with _STEMMER_LOCK:
        return _STEMMER.stemWord(folded)


def stem_terms(text: str) -> set[str]:
    """Return Porter-stemmed, casefolded ``\\w+`` terms from ``text``."""
    if not isinstance(text, str) or not text:
        return set()
    return {_stem_token(match.group(0).casefold()) for match in _TOKEN_RE.finditer(text)}


def _clean_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    cleaned_lines = [_LINE_NOISE_RE.sub("", line) for line in text.splitlines()]
    return " ".join(" ".join(cleaned_lines).split())


def _sentence_spans(cleaned: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for sep in _SENTENCE_SPLIT_RE.finditer(cleaned):
        spans.append((pos, sep.start()))
        pos = sep.end()
    spans.append((pos, len(cleaned)))
    return spans


def _pick_target_sentence(cleaned: str, query_stems: set[str]) -> tuple[int, int]:
    spans = _sentence_spans(cleaned)
    if not query_stems:
        return spans[0]
    best_span = spans[0]
    best_overlap = 0
    for start, end in spans:
        overlap = len(query_stems & stem_terms(cleaned[start:end]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_span = (start, end)
    return best_span


def _choose_window_start(
    cleaned: str,
    target_start: int,
    target_end: int,
    query_stems: set[str],
    max_chars: int,
) -> int:
    if not query_stems:
        return target_start
    first_match: tuple[int, int] | None = None
    for match in _TOKEN_RE.finditer(cleaned, target_start, target_end):
        if _stem_token(match.group(0).casefold()) in query_stems:
            first_match = (match.start(), match.end())
            break
    if first_match is None:
        return target_start
    match_start, match_end = first_match
    threshold = target_start + max(0, max_chars - 40)
    if match_end <= threshold:
        return target_start
    lookback = min(60, max(0, max_chars // 2))
    anchor = max(target_start, match_start - lookback)
    if anchor == target_start or cleaned[anchor - 1].isspace():
        return anchor
    pos = anchor
    while pos < match_start and not cleaned[pos].isspace():
        pos += 1
    while pos < match_start and cleaned[pos].isspace():
        pos += 1
    return pos


def make_snippet(text: str, query: str, *, max_chars: int = 240) -> Snippet:
    """Extract a word-aligned snippet around the best-matching sentence."""
    cleaned = _clean_text(text)
    if not cleaned or max_chars <= 0:
        return Snippet("", ())

    query_stems = stem_terms(query)
    target_start, target_end = _pick_target_sentence(cleaned, query_stems)
    start = _choose_window_start(cleaned, target_start, target_end, query_stems, max_chars)

    end = min(len(cleaned), start + max_chars)
    was_cut = end < len(cleaned)
    if was_cut and not cleaned[end].isspace():
        last_ws = cleaned.rfind(" ", start, end)
        if last_ws != -1:
            end = last_ws

    body = cleaned[start:end].rstrip()
    if not body:
        end = min(len(cleaned), start + max_chars)
        was_cut = end < len(cleaned)
        body = cleaned[start:end].rstrip()

    prefix = "…" if start > 0 else ""
    suffix = "…" if was_cut else ""
    final_text = f"{prefix}{body}{suffix}"

    if not query_stems:
        return Snippet(final_text, ())
    highlights = tuple(
        (match.start(), match.end())
        for match in _TOKEN_RE.finditer(final_text)
        if _stem_token(match.group(0).casefold()) in query_stems
    )
    return Snippet(final_text, highlights)
