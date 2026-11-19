"""Property-based tests for all seekr chunkers."""

from __future__ import annotations

import re
import warnings

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from seekr.chunking import chunk
from seekr.config import Config
from seekr.models import ExtractedUnit

_HYPOTHESIS_SETTINGS = settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

_LOCATOR_RE = re.compile(r"^L(\d+)-(\d+)$")
_WHITESPACE_ONLY = st.text(alphabet=st.sampled_from([" ", "\t", "\n", "\r"]), min_size=0, max_size=30)
_WORD_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-/"


@st.composite
def _word_token(draw: st.DrawFn) -> str:
    return draw(st.text(alphabet=_WORD_ALPHABET, min_size=1, max_size=10))


@st.composite
def _chunk_cfg(draw: st.DrawFn) -> Config:
    target = draw(st.integers(min_value=1, max_value=40))
    max_words = draw(st.integers(min_value=target, max_value=max(target, 60)))
    overlap = draw(st.integers(min_value=0, max_value=target - 1))
    return Config(
        chunk_target_words=target,
        chunk_max_words=max_words,
        chunk_overlap_words=overlap,
    )


@_HYPOTHESIS_SETTINGS
@given(
    kind=st.sampled_from(["text", "pdf"]),
    cfg=_chunk_cfg(),
    unit_words=st.lists(st.lists(_word_token(), min_size=0, max_size=120), min_size=1, max_size=4),
)
def test_prose_and_pdf_word_coverage_size_and_overlap(
    kind: str,
    cfg: Config,
    unit_words: list[list[str]],
) -> None:
    units: list[ExtractedUnit] = []
    expected_by_loc: dict[float, tuple[str, list[str]]] = {}
    for index, words in enumerate(unit_words, start=1):
        locator = f"p.{index}" if kind == "pdf" else f"unit.{index}"
        loc_start = float(index)
        units.append(ExtractedUnit(text=" ".join(words), locator=locator, loc_start=loc_start))
        if words:
            expected_by_loc[loc_start] = (locator, words)

    chunks = chunk(kind, units, cfg)

    effective_size = max(1, min(cfg.chunk_target_words, cfg.chunk_max_words))
    effective_overlap = min(max(cfg.chunk_overlap_words, 0), max(effective_size - 1, 0))

    grouped: dict[float, list[list[str]]] = {}
    for item in chunks:
        assert item.text.strip() != ""
        words = item.text.split()
        assert 1 <= len(words) <= cfg.chunk_max_words
        assert len(words) <= effective_size
        assert item.loc_start in expected_by_loc
        assert item.locator == expected_by_loc[item.loc_start][0]
        grouped.setdefault(item.loc_start, []).append(words)

    assert list(grouped.keys()) == list(expected_by_loc.keys())
    for loc_start, (_locator, expected_words) in expected_by_loc.items():
        unit_chunks = grouped[loc_start]
        reconstructed = list(unit_chunks[0])
        for following in unit_chunks[1:]:
            assert len(following) > effective_overlap
            assert unit_chunks[0] is not following
            reconstructed.extend(following[effective_overlap:])
        assert reconstructed == expected_words


@_HYPOTHESIS_SETTINGS
@given(
    cfg=_chunk_cfg(),
    blocks=st.lists(
        st.one_of(
            st.tuples(
                st.just("atx"),
                st.integers(min_value=1, max_value=6),
                st.lists(_word_token(), min_size=1, max_size=5),
            ),
            st.tuples(
                st.just("setext"),
                st.sampled_from(["===", "---"]),
                st.lists(_word_token(), min_size=1, max_size=5),
            ),
            st.tuples(
                st.just("fence"),
                st.sampled_from(["```", "~~~"]),
                st.lists(st.lists(_word_token(), min_size=1, max_size=6), min_size=1, max_size=4),
            ),
            st.tuples(
                st.just("prose"),
                st.lists(_word_token(), min_size=1, max_size=50),
                st.just(None),
            ),
            st.tuples(st.just("blank"), st.just(None), st.just(None)),
        ),
        min_size=1,
        max_size=10,
    ),
)
def test_markdown_word_coverage_and_size_limits(cfg: Config, blocks: list[tuple[object, ...]]) -> None:
    lines: list[str] = []
    tagged_tokens: list[str] = []
    counter = 0

    def tag_words(words: list[str]) -> str:
        nonlocal counter
        tagged = []
        for word in words:
            token = f"w{counter}_{word}"
            counter += 1
            tagged_tokens.append(token)
            tagged.append(token)
        return " ".join(tagged)

    for block in blocks:
        kind = block[0]
        if kind == "atx":
            level = int(block[1])  # type: ignore[arg-type]
            title_words = block[2]
            assert isinstance(title_words, list)
            lines.append(f"{'#' * level} {tag_words(title_words)}")
        elif kind == "setext":
            underline = str(block[1])
            title_words = block[2]
            assert isinstance(title_words, list)
            lines.append(tag_words(title_words))
            lines.append(underline)
        elif kind == "fence":
            fence = str(block[1])
            inner_lines = block[2]
            assert isinstance(inner_lines, list)
            lines.append(fence)
            for inner_words in inner_lines:
                lines.append(f"# {tag_words(inner_words)}")
            lines.append(fence)
        elif kind == "prose":
            prose_words = block[1]
            assert isinstance(prose_words, list)
            lines.append(tag_words(prose_words))
        else:
            lines.append("")

    doc = "\n".join(lines)
    chunks = chunk("markdown", [ExtractedUnit(text=doc, locator="", loc_start=0.0)], cfg)

    if not tagged_tokens and not doc.strip():
        assert chunks == []
        return

    assert len(chunks) >= 1
    first_seen: dict[str, tuple[int, int]] = {}
    for chunk_idx, item in enumerate(chunks):
        assert item.text.strip() != ""
        words = item.text.split()
        assert 1 <= len(words) <= cfg.chunk_max_words
        assert item.loc_start >= 1.0
        for word_idx, word in enumerate(words):
            if word not in first_seen:
                first_seen[word] = (chunk_idx, word_idx)

    previous_pos = (-1, -1)
    for token in tagged_tokens:
        assert token in first_seen
        pos = first_seen[token]
        assert pos > previous_pos
        previous_pos = pos


@_HYPOTHESIS_SETTINGS
@given(
    cfg=_chunk_cfg(),
    cue_word_lists=st.lists(
        st.one_of(
            st.just([]),
            st.lists(_word_token(), min_size=1, max_size=65),
        ),
        min_size=1,
        max_size=12,
    ),
)
def test_transcript_word_preservation_and_cue_size_contract(
    cfg: Config,
    cue_word_lists: list[list[str]],
) -> None:
    units: list[ExtractedUnit] = []
    non_empty_cues: list[tuple[str, float, list[str]]] = []
    for index, words in enumerate(cue_word_lists):
        seconds = index * 5
        minutes, secs = divmod(seconds, 60)
        locator = f"00:{minutes:02d}:{secs:02d}"
        loc_start = float(seconds)
        text = " ".join(words) if words else "   "
        units.append(ExtractedUnit(text=text, locator=locator, loc_start=loc_start))
        if words:
            non_empty_cues.append((locator, loc_start, words))

    chunks = chunk("transcript", units, cfg)

    expected_words = [word for _loc, _start, words in non_empty_cues for word in words]
    if not expected_words:
        assert chunks == []
        return

    output_words = [word for item in chunks for word in item.text.split()]
    assert output_words == expected_words

    # Match each output chunk against the contiguous slice of non-empty cues it owns.
    cue_index = 0
    for item in chunks:
        assert item.text.strip() != ""
        words = item.text.split()
        assert len(words) >= 1
        assert cue_index < len(non_empty_cues)
        first_loc, first_start, _ = non_empty_cues[cue_index]
        assert item.locator == first_loc
        assert item.loc_start == first_start

        consumed_words: list[str] = []
        cues_in_chunk = 0
        while cue_index < len(non_empty_cues) and len(consumed_words) < len(words):
            consumed_words.extend(non_empty_cues[cue_index][2])
            cue_index += 1
            cues_in_chunk += 1

        assert consumed_words == words
        if len(words) > cfg.chunk_target_words:
            # Oversized cue contract: atomic cues larger than chunk_target_words stay intact
            # and are never merged with adjacent cues.
            assert cues_in_chunk == 1
        else:
            assert len(words) <= cfg.chunk_max_words

    assert cue_index == len(non_empty_cues)


@_HYPOTHESIS_SETTINGS
@given(
    cfg=_chunk_cfg(),
    raw_lines=st.lists(
        st.one_of(
            st.just(""),
            st.lists(_word_token(), min_size=1, max_size=6).map(" ".join),
        ),
        min_size=1,
        max_size=180,
    ),
)
def test_code_line_fallback_coverage_and_window_limits(
    cfg: Config,
    raw_lines: list[str],
) -> None:
    text = "\n".join(raw_lines)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        chunks = chunk("code", [ExtractedUnit(text=text, locator="", loc_start=0.0)], cfg, language=None)

    split_lines = text.splitlines()
    non_empty_lines = [line for line in split_lines if line.strip()]
    if not non_empty_lines:
        assert chunks == []
        return

    assert len(chunks) >= 1
    covered_lines: list[str] = []
    for item in chunks:
        assert item.text.strip() != ""
        chunk_lines = item.text.splitlines()
        assert 1 <= len(chunk_lines) <= 60
        assert item.loc_start >= 1.0
        match = _LOCATOR_RE.match(item.locator)
        assert match is not None
        start_line = int(match.group(1))
        end_line = int(match.group(2))
        assert 1 <= start_line <= end_line <= len(split_lines)
        assert end_line - start_line + 1 <= 60
        assert float(start_line) == item.loc_start
        covered_lines.extend(line for line in chunk_lines if line.strip())

    for line in non_empty_lines:
        assert line in covered_lines


@_HYPOTHESIS_SETTINGS
@given(
    kind=st.sampled_from(["text", "pdf", "markdown", "code", "transcript"]),
    cfg=_chunk_cfg(),
    whitespace_units=st.lists(_WHITESPACE_ONLY, min_size=0, max_size=5),
)
def test_whitespace_only_inputs_return_no_chunks(
    kind: str,
    cfg: Config,
    whitespace_units: list[str],
) -> None:
    units = [ExtractedUnit(text=text, locator="loc", loc_start=0.0) for text in whitespace_units]
    assert chunk(kind, units, cfg) == []
