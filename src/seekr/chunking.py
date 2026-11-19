"""Turn extracted text units into embeddable chunks.

Chunk kinds
-----------
``text`` / ``pdf``
    Word windows using the configured target, maximum and overlap.
``markdown``
    Sections keyed by their heading path (``§A > B``), windowed when long.
    ``loc_start`` is the source line of the first word in the chunk.
``code``
    Function and class chunks from a real tree-sitter AST, including useful
    module-level code, with ``L<start>-<end>`` locators.  Unsupported
    languages, syntax errors and unavailable parsers fall back to 60-line
    windows with a 10-line overlap.
``transcript``
    Consecutive cues grouped up to the target word count, keeping the first
    cue's timestamp.

AST dependency contract
-----------------------
Code chunking uses ``tree-sitter-language-pack`` lazily; importing seekr does
not require the package.  Two installed APIs are supported:

* ``tree-sitter-language-pack`` 0.x: wheels bundle every grammar, so
  ``get_parser(name)`` is a local operation.
* ``tree-sitter-language-pack`` 1.x: grammars are downloaded on first use.
  seekr only calls ``get_parser()`` after ``LanguageRegistry.has_parser()``
  (preferred) or ``downloaded_languages()`` proves the grammar is already on
  disk.  It never calls an on-demand loader for a missing grammar.

If the dependency is absent, its version cannot be identified without an
offline cache API, or a grammar is not cached locally, seekr emits a
``ChunkingWarning`` explaining the missing asset and the supported versions,
then uses the line-window fallback.  It does not silently claim AST coverage.
"""

from __future__ import annotations

import bisect
import importlib
import re
import warnings
from collections.abc import Iterable
from typing import Any

from .config import Config
from .models import ChunkSpec, ExtractedUnit

__all__ = ["AST_SUPPORTED_LANGUAGES", "ChunkingWarning", "ast_support", "chunk"]

#: Canonical language names accepted by the tree-sitter AST chunker.
AST_SUPPORTED_LANGUAGES = frozenset(
    {"python", "c", "cpp", "java", "javascript", "typescript", "tsx", "rust", "go"}
)

_LANGUAGE_ALIASES = {
    "py": "python",
    "python3": "python",
    "c++": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "h": "c",
    "hpp": "cpp",
    "hxx": "cpp",
    "js": "javascript",
    "jsx": "javascript",
    "node": "javascript",
    "ts": "typescript",
    "rs": "rust",
    "golang": "go",
}

_LANGUAGE_PACK_MODULE = "tree_sitter_language_pack"
_LANGUAGE_PACK_DIST = "tree-sitter-language-pack"
_FALLBACK_LINES = 60
_FALLBACK_OVERLAP = 10


class ChunkingWarning(UserWarning):
    """A chunking capability is missing and a fallback is being used."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def ast_support(language: str | None) -> tuple[bool, str]:
    """Report whether offline AST chunking is available for ``language``.

    The check never downloads a grammar.  The returned detail identifies the
    installed language-pack version and whether the grammar is bundled (0.x)
    or already cached (1.x), or gives an actionable reason for the fallback.
    """
    canonical = _canonical_language(language)
    if canonical is None:
        return False, _unsupported_language_message(language)
    parser, detail = _load_parser(canonical)
    return parser is not None, detail


def chunk(
    kind: str,
    units: list[ExtractedUnit],
    cfg: Config,
    language: str | None = None,
) -> list[ChunkSpec]:
    """Split extracted units into chunks according to ``kind``.

    ``language`` is only used for ``kind == "code"``.  It should be the
    extension-derived canonical language name supplied by the indexer.
    """
    if not units or not any(unit.text.strip() for unit in units):
        return []

    if kind in {"text", "pdf"}:
        return _prose_chunks(units, cfg)
    if kind == "markdown":
        return _markdown_chunks(units, cfg)
    if kind == "code":
        return _code_chunks(units, cfg, language)
    if kind == "transcript":
        return _transcript_chunks(units, cfg)
    raise ValueError(f"unsupported chunk kind {kind!r}")


# ---------------------------------------------------------------------------
# Prose and transcripts
# ---------------------------------------------------------------------------


def _window_ranges(count: int, cfg: Config) -> Iterable[tuple[int, int]]:
    """Yield half-open word ranges with the configured overlap."""
    size = max(1, min(cfg.chunk_target_words, cfg.chunk_max_words))
    overlap = min(max(cfg.chunk_overlap_words, 0), max(size - 1, 0))
    start = 0
    while start < count:
        end = min(start + size, count)
        yield start, end
        if end >= count:
            break
        start = end - overlap


def _prose_chunks(units: list[ExtractedUnit], cfg: Config) -> list[ChunkSpec]:
    chunks: list[ChunkSpec] = []
    for unit in units:
        words = unit.text.split()
        if not words:
            continue
        for start, end in _window_ranges(len(words), cfg):
            if start == 0 and end == len(words):
                text = unit.text
            else:
                text = " ".join(words[start:end])
            chunks.append(
                ChunkSpec(
                    text=text,
                    locator=unit.locator,
                    loc_start=unit.loc_start,
                )
            )
    return chunks


def _transcript_chunks(units: list[ExtractedUnit], cfg: Config) -> list[ChunkSpec]:
    target = max(1, cfg.chunk_target_words)
    chunks: list[ChunkSpec] = []
    current: list[str] = []
    current_words = 0
    first_locator = ""
    first_start = 0.0

    def flush() -> None:
        nonlocal current, current_words
        if current:
            chunks.append(
                ChunkSpec(
                    text=" ".join(current),
                    locator=first_locator,
                    loc_start=first_start,
                )
            )
        current = []
        current_words = 0

    for unit in units:
        words = unit.text.split()
        if not words:
            continue
        if current and current_words + len(words) > target:
            flush()
        if not current:
            first_locator = unit.locator
            first_start = unit.loc_start
        current.append(" ".join(words))
        current_words += len(words)

    flush()
    return chunks


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")
_SETEXT_UNDERLINE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_FENCE_START = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_LIST_PREFIX = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
_CLOSING_HASHES = re.compile(r"[ \t]+#+[ \t]*$")


def _fence_marker(line: str) -> tuple[str, int] | None:
    match = _FENCE_START.match(line)
    if match is None:
        return None
    marker = match.group(1)
    return marker[0], len(marker)


def _setext_level(lines: list[str], index: int) -> int | None:
    if index + 1 >= len(lines):
        return None
    title = lines[index]
    stripped = title.lstrip(" ")
    if not stripped or len(title) - len(stripped) > 3:
        return None
    if stripped.startswith(">") or _LIST_PREFIX.match(stripped):
        return None
    match = _SETEXT_UNDERLINE.match(lines[index + 1])
    if match is None:
        return None
    return 1 if match.group(1).startswith("=") else 2


def _push_heading(stack: list[tuple[int, str]], level: int, title: str) -> str:
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title or "(untitled)"))
    return "\u00a7" + " > ".join(item_title for _, item_title in stack)


def _markdown_sections(text: str) -> list[tuple[str, int, int]]:
    """Return ``(locator, start_line_index, end_line_index)`` sections."""
    lines = text.splitlines()
    if not lines:
        return []

    sections: list[tuple[str, int, int]] = []
    stack: list[tuple[int, str]] = []
    locator = ""
    section_start = 0
    fence: tuple[str, int] | None = None
    index = 0

    def flush(end: int) -> None:
        if end > section_start and any(line.strip() for line in lines[section_start:end]):
            sections.append((locator, section_start, end))

    while index < len(lines):
        line = lines[index]

        if fence is not None:
            marker = _fence_marker(line)
            if marker is not None and marker[0] == fence[0] and marker[1] >= fence[1]:
                fence = None
            index += 1
            continue

        marker = _fence_marker(line)
        if marker is not None:
            fence = marker
            index += 1
            continue

        heading = _ATX_HEADING.match(line)
        if heading is not None:
            flush(index)
            title = (heading.group(2) or "").strip()
            title = _CLOSING_HASHES.sub("", title).strip()
            locator = _push_heading(stack, len(heading.group(1)), title)
            section_start = index
            index += 1
            continue

        setext_level = _setext_level(lines, index)
        if setext_level is not None:
            flush(index)
            locator = _push_heading(stack, setext_level, lines[index].strip())
            section_start = index
            index += 2
            continue

        index += 1

    flush(len(lines))
    return sections


def _markdown_chunks(units: list[ExtractedUnit], cfg: Config) -> list[ChunkSpec]:
    chunks: list[ChunkSpec] = []
    for unit in units:
        lines = unit.text.splitlines()
        for locator, section_start, section_end in _markdown_sections(unit.text):
            section_words: list[str] = []
            word_lines: list[int] = []
            for index in range(section_start, section_end):
                line_words = lines[index].split()
                section_words.extend(line_words)
                word_lines.extend([index + 1] * len(line_words))
            if not section_words:
                continue
            for start, end in _window_ranges(len(section_words), cfg):
                line_number = word_lines[start] if start < len(word_lines) else section_start + 1
                chunks.append(
                    ChunkSpec(
                        text=" ".join(section_words[start:end]),
                        locator=locator,
                        loc_start=float(line_number),
                    )
                )
    return chunks


# ---------------------------------------------------------------------------
# Code: tree-sitter AST chunking and line fallback
# ---------------------------------------------------------------------------

_DIRECT_DEFINITIONS: dict[str, frozenset[str]] = {
    "python": frozenset({"function_definition", "class_definition", "decorated_definition"}),
    "c": frozenset(
        {"function_definition", "struct_specifier", "union_specifier", "enum_specifier"}
    ),
    "cpp": frozenset(
        {
            "function_definition",
            "class_specifier",
            "struct_specifier",
            "union_specifier",
            "enum_specifier",
            "namespace_definition",
        }
    ),
    "java": frozenset(
        {
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
            "annotation_type_declaration",
        }
    ),
    "javascript": frozenset(
        {"function_declaration", "generator_function_declaration", "class_declaration"}
    ),
    "typescript": frozenset(
        {
            "function_declaration",
            "generator_function_declaration",
            "class_declaration",
            "abstract_class_declaration",
            "interface_declaration",
            "enum_declaration",
            "type_alias_declaration",
        }
    ),
    "tsx": frozenset(
        {
            "function_declaration",
            "generator_function_declaration",
            "class_declaration",
            "abstract_class_declaration",
            "interface_declaration",
            "enum_declaration",
            "type_alias_declaration",
        }
    ),
    "rust": frozenset(
        {
            "function_item",
            "struct_item",
            "enum_item",
            "trait_item",
            "impl_item",
            "union_item",
            "mod_item",
        }
    ),
    "go": frozenset({"function_declaration", "method_declaration", "type_declaration"}),
}

_CONTAINER_DEFINITION_CHILDREN: dict[str, dict[str, frozenset[str]]] = {
    "c": {
        "declaration": frozenset({"struct_specifier", "union_specifier", "enum_specifier"}),
        "type_definition": frozenset({"struct_specifier", "union_specifier", "enum_specifier"}),
    },
    "cpp": {
        "declaration": frozenset(
            {"class_specifier", "struct_specifier", "union_specifier", "enum_specifier"}
        ),
        "type_definition": frozenset(
            {"class_specifier", "struct_specifier", "union_specifier", "enum_specifier"}
        ),
        "template_declaration": frozenset(
            {
                "function_definition",
                "class_specifier",
                "struct_specifier",
                "union_specifier",
                "enum_specifier",
            }
        ),
    },
}

_WRAPPER_DEFINITIONS = {
    "javascript": frozenset({"export_statement"}),
    "typescript": frozenset({"export_statement"}),
    "tsx": frozenset({"export_statement"}),
}

_JS_FAMILY = frozenset({"javascript", "typescript", "tsx"})
_CALLABLE_VALUES = frozenset(
    {"arrow_function", "function_expression", "generator_function", "class", "class_expression"}
)


class _AstFallback(Exception):
    """Raised internally when the AST path cannot produce chunks."""


def _canonical_language(raw: str | None) -> str | None:
    if raw is None:
        return None
    name = str(raw).strip().lower().lstrip(".")
    if not name:
        return None
    return _LANGUAGE_ALIASES.get(name, name)


def _unsupported_language_message(language: str | None) -> str:
    supported = ", ".join(sorted(AST_SUPPORTED_LANGUAGES))
    return (
        f"no tree-sitter AST chunker is configured for language {language!r} "
        f"(supported: {supported}). Falling back to {_FALLBACK_LINES}-line windows."
    )


def _load_parser(language: str) -> tuple[Any | None, str]:
    """Load a grammar without allowing an on-demand download.

    Returns ``(parser, detail)``.  ``detail`` explains the backend on success
    and is an actionable fallback message on failure.
    """
    if language not in AST_SUPPORTED_LANGUAGES:
        return None, _unsupported_language_message(language)

    try:
        module = importlib.import_module(_LANGUAGE_PACK_MODULE)
    except Exception as exc:
        return None, (
            "tree-sitter-language-pack is not installed or could not be imported, so seekr "
            f"cannot build AST chunks for {language!r} ({exc}). Install a supported release "
            "(0.x bundles grammars; 1.x requires parsers to be pre-downloaded into its "
            f"local cache) or keep the {_FALLBACK_LINES}-line fallback. seekr never "
            "downloads grammars at runtime. Falling back now."
        )

    get_parser = getattr(module, "get_parser", None)
    if not callable(get_parser):
        return None, (
            f"the installed tree-sitter-language-pack does not expose the supported "
            f"get_parser(name) API, so {language!r} AST chunks are unavailable. Supported "
            "APIs are 0.x (bundled grammars) and 1.x (pre-cached grammars). Falling back "
            f"to {_FALLBACK_LINES}-line windows."
        )

    version = _language_pack_version(module)
    major = _major_version(version)
    if major is None:
        return None, (
            "cannot determine the installed tree-sitter-language-pack version. seekr only "
            "allows the documented 0.x bundled-grammar path and 1.x locally cached-grammar "
            "path, and refuses to call get_parser() without a recognised major version "
            "because it may download a grammar. Install tree-sitter-language-pack 0.x or a "
            f"1.x release with pre-downloaded parsers. Falling back to {_FALLBACK_LINES}-line "
            "windows."
        )

    if major == 1:
        cached, cache_detail = _verify_cached_grammar(module, language, version)
        if not cached:
            return None, cache_detail
    elif major != 0:
        return None, (
            f"tree-sitter-language-pack {version} reports unsupported major version "
            f"{major}; only the documented 0.x bundled-grammar and 1.x locally cached-"
            "grammar paths are allowed, so seekr will not load it. Falling back to "
            f"{_FALLBACK_LINES}-line windows."
        )

    try:
        parser = get_parser(language)
    except Exception as exc:  # The package raises its own Error hierarchy.
        return None, (
            f"tree-sitter-language-pack {version or 'unknown version'} could not load the "
            f"{language!r} parser from local assets: {exc}. seekr did not request a "
            f"download. Falling back to {_FALLBACK_LINES}-line windows."
        )
    if parser is None:
        return None, (
            f"tree-sitter-language-pack {version or 'unknown version'} returned no parser "
            f"for {language!r}. Falling back to {_FALLBACK_LINES}-line windows."
        )

    backend = "bundled" if major == 0 else "locally cached"
    return parser, (
        f"tree-sitter-language-pack {version or 'unknown version'} provides a {backend} "
        f"{language} grammar (get_parser(name) API)."
    )


def _language_pack_version(module: Any) -> str | None:
    version = getattr(module, "__version__", None)
    if isinstance(version, str) and version.strip():
        return version.strip()
    try:
        from importlib.metadata import version as distribution_version

        return distribution_version(_LANGUAGE_PACK_DIST)
    except Exception:
        return None


def _major_version(version: str | None) -> int | None:
    if not version:
        return None
    match = re.match(r"\s*(\d+)", version)
    return int(match.group(1)) if match is not None else None


def _verify_cached_grammar(
    module: Any,
    language: str,
    version: str | None,
) -> tuple[bool, str]:
    """Check the 1.x local cache without triggering a download."""
    label = f"tree-sitter-language-pack {version or '1.x'}"
    registry_error: Exception | None = None
    registry_type = getattr(module, "LanguageRegistry", None)
    if registry_type is not None:
        try:
            factory = getattr(registry_type, "new", None)
            registry = factory() if callable(factory) else registry_type()
            has_parser = getattr(registry, "has_parser", None)
            if callable(has_parser):
                if has_parser(language):
                    return True, f"{label} has a locally cached {language} parser."
                return False, (
                    f"{label} has no locally cached {language!r} parser. seekr refuses to "
                    "call get_parser() because that could download a grammar. Pre-download "
                    f"it once (python -c \"from tree_sitter_language_pack import download; "
                    f"download(['{language}'])\") or install tree-sitter-language-pack<1, "
                    f"whose wheels bundle grammars. Falling back to {_FALLBACK_LINES}-line "
                    "windows."
                )
        except Exception as exc:
            # Fall through to the documented downloaded_languages() check, but
            # keep the failure so the final diagnostic is not silent.
            registry_error = exc

    downloaded = getattr(module, "downloaded_languages", None)
    if callable(downloaded):
        try:
            cached = {str(name).strip().lower() for name in downloaded()}
        except Exception as exc:
            return False, (
                f"{label} could not inspect its local parser cache ({exc}), so seekr cannot "
                "prove the grammar is available offline and will not call get_parser(). "
                f"Falling back to {_FALLBACK_LINES}-line windows."
            )
        if language in cached:
            return True, f"{label} has a locally cached {language} parser."
        return False, (
            f"{label} has no {language!r} parser in its local cache. seekr refuses to call "
            "get_parser() because that could download a grammar. Pre-download it once "
            f"(python -c \"from tree_sitter_language_pack import download; "
            f"download(['{language}'])\") or install tree-sitter-language-pack<1, whose "
            f"wheels bundle grammars. Falling back to {_FALLBACK_LINES}-line windows."
        )

    reason = (
        f" LanguageRegistry inspection also failed: {registry_error}."
        if registry_error is not None
        else ""
    )
    return False, (
        f"{label} loads grammars on demand, but it exposes neither "
        "LanguageRegistry.has_parser() nor downloaded_languages(), so seekr cannot verify "
        "that the grammar is already local. seekr refuses to call get_parser() because it "
        "may download. Install tree-sitter-language-pack 0.x (bundled grammars) or a 1.x "
        f"release with an offline cache API.{reason} Falling back to {_FALLBACK_LINES}-line "
        "windows."
    )


def _node_type(node: Any) -> str:
    value = getattr(node, "type", None)
    if isinstance(value, str):
        return value
    kind = getattr(node, "kind", None)
    if callable(kind):
        return str(kind())
    return str(value or "")


def _node_start_byte(node: Any) -> int | None:
    value = getattr(node, "start_byte", None)
    if callable(value):
        value = value()
    if value is None:
        return None
    return int(value)


def _node_end_byte(node: Any) -> int | None:
    value = getattr(node, "end_byte", None)
    if callable(value):
        value = value()
    if value is None:
        return None
    return int(value)


def _node_children(node: Any) -> list[Any]:
    children = getattr(node, "named_children", None)
    if callable(children):
        return list(children())
    if children is not None:
        return list(children)
    named_child = getattr(node, "named_child", None)
    count = getattr(node, "named_child_count", None)
    if callable(named_child) and count is not None:
        return [named_child(index) for index in range(int(count())) if named_child(index)]
    return []


def _node_has_error(node: Any) -> bool:
    value = getattr(node, "has_error", None)
    if callable(value):
        return bool(value())
    return bool(value)


def _is_callable_value(node: Any, depth: int = 0) -> bool:
    if depth > 3:
        return False
    for child in _node_children(node):
        if _node_type(child) in _CALLABLE_VALUES:
            return True
        if _is_callable_value(child, depth + 1):
            return True
    return False


def _is_definition_node(node: Any, language: str) -> bool:
    node_type = _node_type(node)
    if node_type in _DIRECT_DEFINITIONS.get(language, frozenset()):
        return True

    if node_type in _WRAPPER_DEFINITIONS.get(language, frozenset()):
        for child in _node_children(node):
            if _is_definition_node(child, language):
                return True
            if (
                language in _JS_FAMILY
                and _node_type(child) in {"lexical_declaration", "variable_declaration"}
                and _is_callable_value(child)
            ):
                return True
        return False

    expected_children = _CONTAINER_DEFINITION_CHILDREN.get(language, {}).get(node_type)
    if expected_children is not None:
        return any(_node_type(child) in expected_children for child in _node_children(node))

    if language in _JS_FAMILY and node_type in {"lexical_declaration", "variable_declaration"}:
        return _is_callable_value(node)
    return False


def _span_chunk(
    source_bytes: bytes,
    line_starts: list[int],
    start: int,
    end: int,
) -> ChunkSpec | None:
    if end <= start:
        return None
    text = source_bytes[start:end].decode("utf-8", errors="replace").rstrip("\n")
    if not text.strip():
        return None
    start_line = bisect.bisect_right(line_starts, start)
    end_line = bisect.bisect_right(line_starts, max(start, end - 1))
    return ChunkSpec(
        text=text,
        locator=f"L{start_line}-{end_line}",
        loc_start=float(start_line),
    )


def _line_starts(source_bytes: bytes) -> list[int]:
    starts = [0]
    starts.extend(index + 1 for index, byte in enumerate(source_bytes) if byte == 10)
    return starts


def _ast_chunks(source: str, parser: Any, language: str) -> list[ChunkSpec]:
    source_bytes = source.encode("utf-8")
    try:
        tree = parser.parse(source_bytes)
    except TypeError:
        tree = parser.parse(source)
    except Exception as exc:
        raise _AstFallback(
            f"tree-sitter could not parse the {language} source ({exc}); falling back to "
            f"{_FALLBACK_LINES}-line windows for this unit."
        ) from exc

    root = getattr(tree, "root_node", None)
    if root is None:
        raise _AstFallback(
            f"tree-sitter returned no root node for the {language} source; falling back "
            f"to {_FALLBACK_LINES}-line windows for this unit."
        )
    if _node_has_error(root):
        raise _AstFallback(
            f"tree-sitter reported parse errors in the {language} source; falling back to "
            f"{_FALLBACK_LINES}-line windows for this unit."
        )

    line_starts = _line_starts(source_bytes)
    chunks: list[ChunkSpec] = []
    pending_start: int | None = None
    pending_end: int | None = None

    def append_pending() -> None:
        nonlocal pending_start, pending_end
        if pending_start is not None and pending_end is not None:
            chunk = _span_chunk(source_bytes, line_starts, pending_start, pending_end)
            if chunk is not None:
                chunks.append(chunk)
        pending_start = None
        pending_end = None

    for node in _node_children(root):
        start = _node_start_byte(node)
        end = _node_end_byte(node)
        if _is_definition_node(node, language) and start is not None and end is not None:
            append_pending()
            chunk = _span_chunk(source_bytes, line_starts, start, end)
            if chunk is not None:
                chunks.append(chunk)
            continue

        if start is None or end is None or end <= start:
            continue
        if pending_start is None:
            pending_start = start
        pending_end = end if pending_end is None else max(pending_end, end)

    append_pending()

    if not chunks:
        raise _AstFallback(
            f"tree-sitter found no function, class or module-level code in the {language} "
            f"source; falling back to {_FALLBACK_LINES}-line windows for this unit."
        )
    return chunks


def _line_fallback(unit: ExtractedUnit) -> list[ChunkSpec]:
    lines = unit.text.splitlines()
    if not lines:
        return []
    chunks: list[ChunkSpec] = []
    start = 0
    while start < len(lines):
        end = min(start + _FALLBACK_LINES, len(lines))
        text = "\n".join(lines[start:end])
        if text.strip():
            chunks.append(
                ChunkSpec(
                    text=text,
                    locator=f"L{start + 1}-{end}",
                    loc_start=float(start + 1),
                )
            )
        if end >= len(lines):
            break
        start = end - _FALLBACK_OVERLAP
    return chunks


def _warn(message: str) -> None:
    warnings.warn(message, ChunkingWarning, stacklevel=3)


def _code_chunks(
    units: list[ExtractedUnit],
    cfg: Config,
    language: str | None,
) -> list[ChunkSpec]:
    del cfg  # The code AST path uses declaration boundaries, not word counts.
    canonical = _canonical_language(language)
    if canonical is None:
        parser: Any | None = None
        detail = _unsupported_language_message(language)
    else:
        parser, detail = _load_parser(canonical)

    chunks: list[ChunkSpec] = []
    for unit in units:
        if not unit.text.strip():
            continue
        if parser is None:
            _warn(detail)
            chunks.extend(_line_fallback(unit))
            continue
        try:
            chunks.extend(_ast_chunks(unit.text, parser, canonical or ""))
        except _AstFallback as exc:
            _warn(str(exc))
            chunks.extend(_line_fallback(unit))
        except Exception as exc:
            _warn(
                f"tree-sitter chunking failed for {canonical or language!r}: {exc}. "
                f"Falling back to {_FALLBACK_LINES}-line windows for this unit."
            )
            chunks.extend(_line_fallback(unit))
    return chunks
