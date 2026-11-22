"""Tests for chunking prose, markdown, code and transcripts."""

from __future__ import annotations

import itertools
import sys
import types
import unittest
import warnings
from unittest import mock

from seekr import chunking
from seekr.chunking import (
    ChunkingWarning,
    _ast_chunks,
    _AstFallback,
    ast_support,
    chunk,
)
from seekr.config import Config
from seekr.models import ChunkSpec, ExtractedUnit

_PYTHON_AST_READY, _PYTHON_AST_REASON = ast_support("python")


def _cfg(**overrides: object) -> Config:
    values: dict[str, object] = {
        "chunk_target_words": 300,
        "chunk_max_words": 450,
        "chunk_overlap_words": 50,
    }
    values.update(overrides)
    return Config(**values)  # type: ignore[arg-type]


def _words(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{index}" for index in range(count)]


def _unit(text: str, locator: str = "", loc_start: float = 0.0) -> ExtractedUnit:
    return ExtractedUnit(text=text, locator=locator, loc_start=loc_start)


class ProseChunkTests(unittest.TestCase):
    def test_prose_respects_max_and_overlap(self) -> None:
        words = _words("w", 2000)

        chunks = chunk("text", [_unit(" ".join(words))], Config())

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(isinstance(item, ChunkSpec) for item in chunks))
        self.assertTrue(all(len(item.text.split()) <= 450 for item in chunks))
        self.assertEqual(len(chunks[0].text.split()), 300)
        for left, right in itertools.pairwise(chunks):
            self.assertEqual(left.text.split()[-50:], right.text.split()[:50])
        self.assertEqual(
            {word for item in chunks for word in item.text.split()},
            set(words),
        )

    def test_short_text_is_one_chunk(self) -> None:
        chunks = chunk("text", [_unit("alpha beta gamma")], Config())

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "alpha beta gamma")
        self.assertEqual(chunks[0].locator, "")
        self.assertEqual(chunks[0].loc_start, 0.0)

    def test_zero_overlap_does_not_duplicate_boundary_words(self) -> None:
        words = _words("w", 30)
        cfg = _cfg(chunk_target_words=10, chunk_max_words=10, chunk_overlap_words=0)

        chunks = chunk("text", [_unit(" ".join(words))], cfg)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(
            [len(item.text.split()) for item in chunks],
            [10, 10, 10],
        )
        self.assertEqual(
            [item.text.split()[0] for item in chunks],
            ["w0", "w10", "w20"],
        )

    def test_pdf_windows_keep_the_page_locator(self) -> None:
        unit = _unit(" ".join(_words("page", 700)), locator="p.4", loc_start=4.0)

        chunks = chunk("pdf", [unit], Config())

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(item.locator == "p.4" for item in chunks))
        self.assertTrue(all(item.loc_start == 4.0 for item in chunks))

    def test_multiple_units_keep_their_own_locators(self) -> None:
        units = [
            _unit("first page words", locator="p.1", loc_start=1.0),
            _unit("second page words", locator="p.2", loc_start=2.0),
        ]

        chunks = chunk("pdf", units, Config())

        self.assertEqual([item.locator for item in chunks], ["p.1", "p.2"])

    def test_empty_and_whitespace_units_return_empty_list(self) -> None:
        self.assertEqual(chunk("text", [], Config()), [])
        self.assertEqual(chunk("text", [_unit(" \n\t ")], Config()), [])

    def test_unknown_kind_with_content_raises(self) -> None:
        with self.assertRaises(ValueError):
            chunk("spreadsheet", [_unit("hello")], Config())


class MarkdownChunkTests(unittest.TestCase):
    def test_markdown_headings_become_locators(self) -> None:
        text = "# A\n\nintro words\n\n## B\n\nnested words\n"

        chunks = chunk("markdown", [_unit(text)], Config())

        self.assertEqual([item.locator for item in chunks], ["\u00a7A", "\u00a7A > B"])
        self.assertEqual([item.loc_start for item in chunks], [1.0, 5.0])
        self.assertIn("intro words", chunks[0].text)
        self.assertIn("nested words", chunks[1].text)

    def test_nested_heading_stack_resets_on_a_higher_level(self) -> None:
        text = "# A\n### C\n## B\n# D\n"

        chunks = chunk("markdown", [_unit(text)], Config())

        self.assertEqual(
            [item.locator for item in chunks],
            ["\u00a7A", "\u00a7A > C", "\u00a7A > B", "\u00a7D"],
        )

    def test_preamble_text_is_preserved(self) -> None:
        text = "preamble words\n\n# Heading\n\nbody words\n"

        chunks = chunk("markdown", [_unit(text)], Config())

        self.assertEqual([item.locator for item in chunks], ["", "\u00a7Heading"])
        self.assertEqual(chunks[0].loc_start, 1.0)
        self.assertIn("preamble", chunks[0].text)

    def test_headings_inside_fenced_code_are_not_split(self) -> None:
        text = "# Real\n\n```python\n# not a heading\ndef f():\n    pass\n```\n"

        chunks = chunk("markdown", [_unit(text)], Config())

        self.assertEqual([item.locator for item in chunks], ["\u00a7Real"])
        self.assertIn("# not a heading", chunks[0].text)

    def test_long_markdown_section_is_windowed_with_source_lines(self) -> None:
        body = " ".join(_words("section", 900))
        text = f"# Long\n\n{body}\n"

        chunks = chunk("markdown", [_unit(text)], Config())

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(item.locator == "\u00a7Long" for item in chunks))
        self.assertTrue(all(len(item.text.split()) <= 450 for item in chunks))
        self.assertEqual(chunks[0].loc_start, 1.0)
        self.assertGreater(chunks[1].loc_start, chunks[0].loc_start)

    def test_setext_heading_is_supported(self) -> None:
        text = "Setext title\n============\n\nbody\n"

        chunks = chunk("markdown", [_unit(text)], Config())

        self.assertEqual([item.locator for item in chunks], ["\u00a7Setext title"])
        self.assertIn("body", chunks[0].text)


@unittest.skipUnless(_PYTHON_AST_READY, _PYTHON_AST_REASON)
class RealTreeSitterCodeChunkTests(unittest.TestCase):
    """Integration tests against the real tree-sitter-language-pack parser."""

    def test_python_functions_and_class_become_ast_chunks(self) -> None:
        source = (
            "def alpha():\n"
            "    return 1\n"
            "\n"
            "def beta():\n"
            "    return 2\n"
            "\n"
            "class Gamma:\n"
            "    def method(self):\n"
            "        return 3\n"
        )

        chunks = chunk("code", [_unit(source)], Config(), language="python")

        self.assertEqual([item.locator for item in chunks], ["L1-2", "L4-5", "L7-9"])
        self.assertEqual([item.loc_start for item in chunks], [1.0, 4.0, 7.0])
        self.assertIn("def alpha", chunks[0].text)
        self.assertIn("class Gamma", chunks[2].text)
        self.assertIn("def method", chunks[2].text)

    def test_module_level_code_is_preserved(self) -> None:
        source = (
            "import os\n"
            "LIMIT = 3\n"
            "\n"
            "def alpha():\n"
            "    return LIMIT\n"
        )

        chunks = chunk("code", [_unit(source)], Config(), language="python")

        self.assertEqual([item.locator for item in chunks], ["L1-2", "L4-5"])
        self.assertIn("import os", chunks[0].text)
        self.assertIn("LIMIT = 3", chunks[0].text)

    def test_decorators_are_included_in_the_function_chunk(self) -> None:
        source = "@decorator\ndef alpha():\n    return 1\n"

        chunks = chunk("code", [_unit(source)], Config(), language="python")

        self.assertEqual([item.locator for item in chunks], ["L1-3"])
        self.assertIn("@decorator", chunks[0].text)

    def test_real_parser_reports_syntax_errors_and_falls_back(self) -> None:
        source = "def broken(:\n    return 1\n"

        with self.assertWarns(ChunkingWarning) as captured:
            chunks = chunk("code", [_unit(source)], Config(), language="python")

        self.assertEqual([item.locator for item in chunks], ["L1-2"])
        self.assertIn("parse errors", str(captured.warning).lower())


class AstParseErrorBranchTests(unittest.TestCase):
    """Exercise the AST-reported-error branch without needing a grammar."""

    def test_ast_reported_errors_raise_the_ast_fallback(self) -> None:
        source = "def broken(:\n    ???\n"
        root = _FakeNode(
            "module",
            start_byte=0,
            end_byte=len(source),
            children=[],
            has_error=True,
        )

        with self.assertRaises(_AstFallback) as raised:
            _ast_chunks(source, _FakeParser(root), "python")

        message = str(raised.exception).lower()
        self.assertIn("parse errors", message)
        self.assertIn("60-line", message)

    def test_code_chunk_uses_warned_line_fallback_for_ast_errors(self) -> None:
        source = "def broken(:\n    ???\n"
        root = _FakeNode(
            "module",
            start_byte=0,
            end_byte=len(source),
            children=[],
            has_error=True,
        )

        with mock.patch(
            "seekr.chunking._load_parser",
            return_value=(_FakeParser(root), "controlled local parser"),
        ), self.assertWarns(ChunkingWarning) as captured:
            chunks = chunk("code", [_unit(source)], Config(), language="python")

        self.assertEqual([item.locator for item in chunks], ["L1-2"])
        self.assertIn("parse errors", str(captured.warning).lower())


class CodeFallbackTests(unittest.TestCase):
    def test_unknown_syntax_falls_back_to_line_windows(self) -> None:
        source = "\n".join(f"garbage {index}" for index in range(130))

        with self.assertWarns(ChunkingWarning) as captured:
            chunks = chunk("code", [_unit(source)], Config(), language="brainfuck")

        self.assertEqual(
            [item.locator for item in chunks],
            ["L1-60", "L51-110", "L101-130"],
        )
        self.assertEqual([item.loc_start for item in chunks], [1.0, 51.0, 101.0])
        self.assertTrue(all(len(item.text.splitlines()) <= 60 for item in chunks))
        self.assertIn("60-line", str(captured.warning))
        self.assertIn("brainfuck", str(captured.warning))

    def test_empty_code_returns_empty_list(self) -> None:
        self.assertEqual(chunk("code", [_unit("  \n ")], Config(), language="python"), [])


class TranscriptChunkTests(unittest.TestCase):
    def test_transcript_groups_cues_and_keeps_first_timestamp(self) -> None:
        cfg = _cfg(chunk_target_words=10, chunk_max_words=12, chunk_overlap_words=2)
        units = [
            _unit("one two three four", locator="00:00:01", loc_start=1.0),
            _unit("five six seven eight", locator="00:00:02", loc_start=2.0),
            _unit("nine ten eleven twelve", locator="00:00:03", loc_start=3.0),
        ]

        chunks = chunk("transcript", units, cfg)

        self.assertEqual([item.locator for item in chunks], ["00:00:01", "00:00:03"])
        self.assertEqual([item.loc_start for item in chunks], [1.0, 3.0])
        self.assertEqual(chunks[0].text, "one two three four five six seven eight")
        self.assertEqual(chunks[1].text, "nine ten eleven twelve")

    def test_oversized_cue_is_kept_as_its_own_chunk(self) -> None:
        cfg = _cfg(chunk_target_words=5, chunk_max_words=10, chunk_overlap_words=1)
        units = [
            _unit(" ".join(_words("long", 12)), locator="00:00:05", loc_start=5.0),
            _unit("short cue", locator="00:00:09", loc_start=9.0),
        ]

        chunks = chunk("transcript", units, cfg)

        self.assertEqual([item.locator for item in chunks], ["00:00:05", "00:00:09"])
        self.assertEqual(len(chunks[0].text.split()), 12)

    def test_whitespace_only_cues_are_ignored(self) -> None:
        cfg = _cfg(chunk_target_words=10, chunk_max_words=12, chunk_overlap_words=2)
        units = [
            _unit("   ", locator="00:00:01", loc_start=1.0),
            _unit("real words", locator="00:00:02", loc_start=2.0),
        ]

        chunks = chunk("transcript", units, cfg)

        self.assertEqual([item.locator for item in chunks], ["00:00:02"])


class OfflineLoaderContractTests(unittest.TestCase):
    """Pin the offline loader gate with a hand-built parser, not a real grammar.

    The real-grammar tests above are the AST-coverage tests and skip when the
    dependency or a locally installed grammar is unavailable. These tests only
    verify that seekr refuses to call an on-demand loader without a local parser.
    """

    def _fake_pack(
        self,
        *,
        version: str,
        cached: list[str],
        parser: object | None,
        with_downloaded: bool = True,
    ) -> types.ModuleType:
        module = types.ModuleType("tree_sitter_language_pack")
        module.__version__ = version  # type: ignore[attr-defined]
        calls: list[str] = []

        def downloaded_languages() -> list[str]:
            return list(cached)

        def get_parser(name: str) -> object:
            calls.append(name)
            if parser is None:
                raise AssertionError("get_parser must not be called without a local grammar")
            return parser

        if with_downloaded:
            module.downloaded_languages = downloaded_languages  # type: ignore[attr-defined]
        module.get_parser = get_parser  # type: ignore[attr-defined]
        module.calls = calls  # type: ignore[attr-defined]
        return module

    def test_on_demand_pack_without_cached_grammar_is_never_asked_to_download(self) -> None:
        source = "\n".join(f"garbage {index}" for index in range(130))
        module = self._fake_pack(version="1.25.2", cached=[], parser=None)

        with mock.patch.dict(
            sys.modules, {"tree_sitter_language_pack": module}
        ), self.assertWarns(ChunkingWarning) as captured:
            chunks = chunk("code", [_unit(source)], Config(), language="python")

        self.assertEqual(module.calls, [])  # type: ignore[attr-defined]
        self.assertEqual(
            [item.locator for item in chunks],
            ["L1-60", "L51-110", "L101-130"],
        )
        message = str(captured.warning).lower()
        self.assertIn("cache", message)
        self.assertIn("pre-download", message)
        self.assertIn("60-line", message)
        self.assertIn("falling back", message)

    def test_cached_on_demand_grammar_uses_the_parser(self) -> None:
        source = "def alpha():\n    return 1\n"
        function = _FakeNode(
            "function_definition",
            start_byte=0,
            end_byte=len(source.rstrip("\n")),
            children=[],
        )
        root = _FakeNode("module", start_byte=0, end_byte=len(source), children=[function])
        module = self._fake_pack(
            version="1.25.2",
            cached=["python"],
            parser=_FakeParser(root),
        )

        with mock.patch.dict(sys.modules, {"tree_sitter_language_pack": module}):
            chunks = chunk("code", [_unit(source)], Config(), language="python")

        self.assertEqual(module.calls, ["python"])  # type: ignore[attr-defined]
        self.assertEqual([item.locator for item in chunks], ["L1-2"])
        self.assertIn("def alpha", chunks[0].text)

    def test_future_major_fails_closed_without_loading(self) -> None:
        source = "\n".join(f"garbage {index}" for index in range(130))
        module = self._fake_pack(version="2.0.0", cached=["python"], parser=None)

        with mock.patch.dict(
            sys.modules, {"tree_sitter_language_pack": module}
        ), self.assertWarns(ChunkingWarning) as captured:
            chunks = chunk("code", [_unit(source)], Config(), language="python")

        self.assertEqual(module.calls, [])  # type: ignore[attr-defined]
        self.assertEqual(
            [item.locator for item in chunks],
            ["L1-60", "L51-110", "L101-130"],
        )
        message = str(captured.warning).lower()
        self.assertIn("2.0.0", message)
        self.assertIn("unsupported major", message)
        self.assertIn("60-line", message)

    def test_unknown_version_fails_closed_without_loading(self) -> None:
        source = "\n".join(f"garbage {index}" for index in range(130))
        module = self._fake_pack(version="unknown", cached=["python"], parser=None)

        with mock.patch.dict(
            sys.modules, {"tree_sitter_language_pack": module}
        ), self.assertWarns(ChunkingWarning) as captured:
            chunks = chunk("code", [_unit(source)], Config(), language="python")

        self.assertEqual(module.calls, [])  # type: ignore[attr-defined]
        self.assertEqual(
            [item.locator for item in chunks],
            ["L1-60", "L51-110", "L101-130"],
        )
        message = str(captured.warning).lower()
        self.assertIn("cannot determine", message)
        self.assertIn("refuses", message)
        self.assertIn("60-line", message)

    def test_unverifiable_local_cache_fails_closed_without_loading(self) -> None:
        source = "\n".join(f"garbage {index}" for index in range(130))
        module = self._fake_pack(
            version="1.25.2",
            cached=["python"],
            parser=None,
            with_downloaded=False,
        )

        class BrokenRegistry:
            @staticmethod
            def new() -> BrokenRegistry:
                return BrokenRegistry()

            def has_parser(self, name: str) -> bool:
                raise RuntimeError("manifest unavailable")

        module.LanguageRegistry = BrokenRegistry  # type: ignore[attr-defined]

        with mock.patch.dict(
            sys.modules, {"tree_sitter_language_pack": module}
        ), self.assertWarns(ChunkingWarning) as captured:
            chunks = chunk("code", [_unit(source)], Config(), language="python")

        self.assertEqual(module.calls, [])  # type: ignore[attr-defined]
        self.assertEqual(
            [item.locator for item in chunks],
            ["L1-60", "L51-110", "L101-130"],
        )
        message = str(captured.warning).lower()
        self.assertIn("cannot verify", message)
        self.assertIn("manifest unavailable", message)
        self.assertIn("60-line", message)

    def test_registry_false_falls_back_to_downloaded_languages(self) -> None:
        source = "def alpha():\n    return 1\n"
        function = _FakeNode(
            "function_definition",
            start_byte=0,
            end_byte=len(source.rstrip("\n")),
            children=[],
        )
        root = _FakeNode("module", start_byte=0, end_byte=len(source), children=[function])
        module = self._fake_pack(
            version="1.25.2",
            cached=["python"],
            parser=_FakeParser(root),
        )

        class EmptyRegistry:
            @staticmethod
            def new() -> EmptyRegistry:
                return EmptyRegistry()

            def has_parser(self, name: str) -> bool:
                del name
                return False

        module.LanguageRegistry = EmptyRegistry  # type: ignore[attr-defined]

        with (
            mock.patch.dict(sys.modules, {"tree_sitter_language_pack": module}),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            chunks = chunk("code", [_unit(source)], Config(), language="python")

        self.assertEqual(caught, [])
        self.assertEqual(module.calls, ["python"])  # type: ignore[attr-defined]
        self.assertEqual([item.locator for item in chunks], ["L1-2"])
        self.assertIn("def alpha", chunks[0].text)


class _FakeNode:
    def __init__(
        self,
        node_type: str,
        *,
        start_byte: int,
        end_byte: int,
        children: list[_FakeNode],
        has_error: bool = False,
    ) -> None:
        self.type = node_type
        self.start_byte = start_byte
        self.end_byte = end_byte
        self.start_point = (0, 0)
        self.end_point = (0, 1)
        self.named_children = children
        self.has_error = has_error


class _FakeTree:
    def __init__(self, root: _FakeNode) -> None:
        self.root_node = root


class _FakeParser:
    def __init__(self, root: _FakeNode) -> None:
        self._root = root

    def parse(self, source: bytes) -> _FakeTree:
        del source
        return _FakeTree(self._root)


class EmptyInputTests(unittest.TestCase):
    def test_empty_units_return_empty_for_every_kind(self) -> None:
        for kind in ("text", "pdf", "markdown", "code", "transcript"):
            with self.subTest(kind=kind):
                self.assertEqual(chunk(kind, [], Config(), language="python"), [])


class MissingGrammarWarningTests(unittest.TestCase):
    def test_missing_grammar_emits_missing_grammar_warning(self) -> None:
        source = "def alpha():\n    return 1\n"
        with mock.patch(
            "seekr.chunking._load_parser",
            return_value=(None, "no cached 'python' parser"),
        ), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chunks = chunk("code", [_unit(source)], Config(), language="python")

        self.assertEqual([item.locator for item in chunks], ["L1-2"])
        self.assertEqual(len(caught), 1)
        warning = caught[0].message
        self.assertIsInstance(warning, chunking.MissingGrammarWarning)
        self.assertIsInstance(warning, ChunkingWarning)
        self.assertEqual(warning.language, "python")  # type: ignore[attr-defined]
        self.assertIn("no cached 'python' parser", str(warning))

    def test_missing_grammar_warns_once_across_multiple_units(self) -> None:
        units = [_unit("   "), _unit("a = 1\n"), _unit("b = 2\n")]
        with mock.patch(
            "seekr.chunking._load_parser",
            return_value=(None, "missing grammar"),
        ), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chunks = chunk("code", units, Config(), language="py")

        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(caught), 1)
        warning = caught[0].message
        self.assertIsInstance(warning, chunking.MissingGrammarWarning)
        self.assertEqual(warning.language, "python")  # type: ignore[attr-defined]

    def test_non_missing_grammar_failures_stay_plain_chunking_warning(self) -> None:
        details = [
            (
                "tree-sitter-language-pack is not installed or could not be imported, "
                "so seekr cannot build AST chunks for 'python' (No module). "
                "1.x requires parsers to be pre-downloaded into its local cache."
            ),
            "tree-sitter-language-pack 2.0.0 reports unsupported major version 2.",
            "cannot determine the installed tree-sitter-language-pack version.",
            "tree-sitter-language-pack 1.25.2 could not inspect its local parser cache.",
            "tree-sitter-language-pack 1.25.2 exposes neither API.",
            "tree-sitter-language-pack 1.25.2 could not load the 'python' parser: err.",
        ]
        for detail in details:
            with self.subTest(detail=detail), mock.patch(
                "seekr.chunking._load_parser",
                return_value=(None, detail),
            ), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                chunk("code", [_unit("x = 1\n")], Config(), language="python")

            self.assertEqual(len(caught), 1)
            self.assertIsInstance(caught[0].message, ChunkingWarning)
            self.assertNotIsInstance(caught[0].message, chunking.MissingGrammarWarning)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chunk("code", [_unit("++++")], Config(), language="brainfuck")
        self.assertEqual(len(caught), 1)
        self.assertIsInstance(caught[0].message, ChunkingWarning)
        self.assertNotIsInstance(caught[0].message, chunking.MissingGrammarWarning)

    def test_registry_missing_cached_grammar_emits_missing_grammar_warning(self) -> None:
        module = types.ModuleType("tree_sitter_language_pack")
        module.__version__ = "1.25.2"  # type: ignore[attr-defined]
        module.get_parser = lambda name: None  # type: ignore[attr-defined]

        class EmptyRegistry:
            @staticmethod
            def new() -> EmptyRegistry:
                return EmptyRegistry()

            def has_parser(self, name: str) -> bool:
                del name
                return False

        module.LanguageRegistry = EmptyRegistry  # type: ignore[attr-defined]
        with mock.patch.dict(
            sys.modules, {"tree_sitter_language_pack": module}
        ), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chunk("code", [_unit("x = 1\n")], Config(), language="python")

        self.assertEqual(len(caught), 1)
        self.assertIsInstance(caught[0].message, chunking.MissingGrammarWarning)
        self.assertEqual(caught[0].message.language, "python")  # type: ignore[attr-defined]


class OfficeAndNotebookChunkTests(unittest.TestCase):
    def test_word_chunks_use_headings_and_pilcrow_fallback(self) -> None:
        text = "Preamble paragraph.\n# Section One\nFirst section body.\n## Sub\nNested body.\n"
        chunks = chunk("word", [_unit(text)], Config())

        self.assertEqual(
            [item.locator for item in chunks],
            ["\u00b61", "\u00a7Section One", "\u00a7Section One > Sub"],
        )
        self.assertEqual([item.loc_start for item in chunks], [1.0, 2.0, 4.0])

    def test_word_headingless_long_document_uses_pilcrow_locators(self) -> None:
        lines = [f"line{idx} " + " ".join(_words(f"w{idx}_", 15)) for idx in range(25)]
        cfg = _cfg(chunk_target_words=80, chunk_max_words=100, chunk_overlap_words=10)
        chunks = chunk("word", [_unit("\n".join(lines))], cfg)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(item.locator.startswith("\u00b6") for item in chunks))
        self.assertEqual(chunks[0].locator, "\u00b61")
        self.assertGreater(int(chunks[1].locator[1:]), 1)

    def test_slides_and_notebook_use_prose_windowing_and_keep_locators(self) -> None:
        cfg = _cfg(chunk_target_words=10, chunk_max_words=10, chunk_overlap_words=0)
        slide_units = [_unit(" ".join(_words("s", 15)), locator="slide 2", loc_start=2.0)]
        slide_chunks = chunk("slides", slide_units, cfg)
        self.assertEqual([c.locator for c in slide_chunks], ["slide 2", "slide 2"])
        self.assertEqual([c.loc_start for c in slide_chunks], [2.0, 2.0])

        nb_units = [_unit(" ".join(_words("c", 15)), locator="cell 3", loc_start=3.0)]
        nb_chunks = chunk("notebook", nb_units, cfg)
        self.assertEqual([c.locator for c in nb_chunks], ["cell 3", "cell 3"])
        self.assertEqual([c.loc_start for c in nb_chunks], [3.0, 3.0])

    def test_empty_units_for_new_kinds_return_empty_list(self) -> None:
        for kind in ("word", "slides", "notebook"):
            with self.subTest(kind=kind):
                self.assertEqual(chunk(kind, [], Config()), [])
                self.assertEqual(chunk(kind, [_unit("  \n ")], Config()), [])


if __name__ == "__main__":
    unittest.main()
