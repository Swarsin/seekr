"""Text extraction for the file types seekr can index."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from . import pathnorm
from .models import ExtractedUnit

_KIND_BY_SUFFIX = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".rst": "text",
    ".py": "code",
    ".c": "code",
    ".h": "code",
    ".cpp": "code",
    ".hpp": "code",
    ".java": "code",
    ".js": "code",
    ".ts": "code",
    ".tsx": "code",
    ".rs": "code",
    ".go": "code",
    ".pdf": "pdf",
    ".vtt": "transcript",
    ".srt": "transcript",
    ".docx": "word",
    ".pptx": "slides",
    ".ipynb": "notebook",
}

_LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".java": "java",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
}

_TEXT_KINDS = frozenset({"markdown", "text", "code"})
_MIN_PDF_PAGE_WORDS = 20
_BINARY_SNIFF_BYTES = 8192
_PDF_ERRORS = (RuntimeError, ValueError, KeyError, IndexError, TypeError)

_TIMESTAMP_PATTERN = r"(?:\d+:){1,2}\d{1,2}(?:[.,]\d{1,3})?"
_CUE_TIMING_RE = re.compile(
    rf"^\s*({_TIMESTAMP_PATTERN})\s*-->\s*({_TIMESTAMP_PATTERN})(?:\s+.*)?$"
)
_TAG_RE = re.compile(r"<[^>]*>")


class ExtractionError(Exception):
    """Raised when a supported file cannot be read, decoded or parsed."""


@dataclass(frozen=True)
class ExtractionResult:
    """The text units and non-fatal warnings produced by one extraction."""

    units: list[ExtractedUnit]
    warnings: list[str]


def kind_for(path: Path) -> str | None:
    """Return the seekr file kind for *path*, or ``None`` when unsupported."""
    return _KIND_BY_SUFFIX.get(Path(path).suffix.lower())


def language_for(path: Path) -> str | None:
    """Return the canonical tree-sitter language name for a code file."""
    return _LANGUAGE_BY_SUFFIX.get(Path(path).suffix.lower())


def extract(path: Path, kind: str) -> ExtractionResult:
    """Extract location-tagged text units from *path*.

    ``FileNotFoundError`` is deliberately allowed to propagate so the indexer
    can treat a file that vanished between scanning and reading as removed.
    All other read and parse failures are reported as :class:`ExtractionError`.
    """
    source = Path(path)
    if kind in _TEXT_KINDS:
        text, warnings = _read_text(source)
        return ExtractionResult(
            units=[ExtractedUnit(text=text, locator="", loc_start=0.0)],
            warnings=warnings,
        )
    if kind == "pdf":
        return _extract_pdf(source)
    if kind == "transcript":
        return _extract_transcript(source)
    if kind == "word":
        from .extract_documents import extract_docx

        return extract_docx(source)
    if kind == "slides":
        from .extract_documents import extract_pptx

        return extract_pptx(source)
    if kind == "notebook":
        from .extract_documents import extract_notebook

        return extract_notebook(source)
    raise ExtractionError(f"unsupported file kind {kind!r} for {source}")


def os_open_path(path: Path) -> Path:
    """Return *path* in the form the OS opens (long-path prefix on Windows)."""
    return Path(pathnorm.os_path(str(path)))


def _read_bytes(path: Path) -> bytes:
    try:
        return os_open_path(path).read_bytes()
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ExtractionError(f"cannot read {path}: {exc}") from exc


def _read_text(path: Path) -> tuple[str, list[str]]:
    raw = _read_bytes(path)
    # Check UTF-16 BOM before the NUL-byte binary sniff because ASCII code points
    # in UTF-16 contain NUL bytes on every other octet.
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")) and not raw.startswith(b"\xff\xfe\x00\x00"):
        try:
            decoded = raw.decode("utf-16")
        except UnicodeDecodeError:
            decoded = ""
        else:
            if "\x00" not in decoded[:_BINARY_SNIFF_BYTES]:
                return decoded, ["file is UTF-16 encoded; decoded with UTF-16 fallback"]
    if b"\x00" in raw[:_BINARY_SNIFF_BYTES]:
        raise ExtractionError(f"binary file {path}: NUL byte in the first 8 KB")
    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            return raw.decode("utf-8-sig"), []
        except UnicodeDecodeError:
            return raw.decode("latin-1"), ["file is not valid UTF-8; used latin-1 fallback"]
    try:
        return raw.decode("utf-8"), []
    except UnicodeDecodeError:
        return raw.decode("latin-1"), ["file is not valid UTF-8; used latin-1 fallback"]


def _extract_transcript(path: Path) -> ExtractionResult:
    text, warnings = _read_text(path)
    lines = text.splitlines()
    units: list[ExtractedUnit] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if _is_ignored_transcript_block(stripped):
            index = _skip_transcript_block(lines, index)
            continue
        if "-->" not in line:
            index += 1
            continue

        line_number = index + 1
        match = _CUE_TIMING_RE.match(line)
        if match is None:
            warnings.append(f"malformed cue timestamp at line {line_number}")
            index = _skip_transcript_block(lines, index)
            continue
        try:
            start = _parse_timestamp(match.group(1))
            _parse_timestamp(match.group(2))
        except ValueError:
            warnings.append(f"malformed cue timestamp at line {line_number}")
            index = _skip_transcript_block(lines, index)
            continue

        index += 1
        cue_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            cue_lines.append(lines[index])
            index += 1

        cue_text = _clean_cue_text(" ".join(cue_lines))
        if cue_text:
            units.append(
                ExtractedUnit(
                    text=cue_text,
                    locator=_format_timestamp(start),
                    loc_start=start,
                )
            )
        else:
            warnings.append(f"cue at line {line_number} has no text")

    return ExtractionResult(units=units, warnings=warnings)


def _is_ignored_transcript_block(line: str) -> bool:
    upper = line.upper()
    return upper.startswith(("WEBVTT", "NOTE")) or upper in {"STYLE", "REGION"}


def _skip_transcript_block(lines: list[str], index: int) -> int:
    index += 1
    while index < len(lines) and lines[index].strip():
        index += 1
    return index


def _clean_cue_text(text: str) -> str:
    without_tags = _TAG_RE.sub("", text)
    return " ".join(html.unescape(without_tags).split())


def _parse_timestamp(value: str) -> float:
    clock, _, fraction = value.replace(",", ".").partition(".")
    parts = clock.split(":")
    if len(parts) == 2:
        hours = 0
        minutes, seconds = (int(part) for part in parts)
    elif len(parts) == 3:
        hours, minutes, seconds = (int(part) for part in parts)
    else:
        raise ValueError(f"invalid timestamp: {value!r}")
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid timestamp: {value!r}")
    milliseconds = int(fraction.ljust(3, "0")) if fraction else 0
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def _format_timestamp(seconds: float) -> str:
    whole_seconds = int(seconds)
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


def _load_pymupdf() -> ModuleType:
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf  # type: ignore[no-redef]
        except ImportError as exc:
            raise ExtractionError("PyMuPDF is required to extract PDF files") from exc
    return pymupdf


def _extract_pdf(path: Path) -> ExtractionResult:
    pymupdf = _load_pymupdf()
    # A failed open leaves PyMuPDF's half-built document (and its open file)
    # reachable from the error's traceback; on Windows that locks the file
    # for as long as the error lives.  Raise outside the handler, unchained.
    failure: str | None = None
    vanished = False
    try:
        document = pymupdf.open(os_open_path(path))
    except FileNotFoundError:
        raise
    except OSError as exc:
        failure = f"cannot read PDF {path}: {exc}"
    except _PDF_ERRORS as exc:
        vanished = not path.exists()
        failure = f"could not open corrupt or unreadable PDF {path}: {exc}"
    if vanished:
        raise FileNotFoundError(path)
    if failure is not None:
        raise ExtractionError(failure)

    units: list[ExtractedUnit] = []
    warnings: list[str] = []
    try:
        if getattr(document, "needs_pass", False):
            raise ExtractionError(f"encrypted PDF cannot be read: {path}")
        for number, page in enumerate(document, start=1):
            try:
                page_text = (page.get_text("text") or "").strip()
            except _PDF_ERRORS as exc:
                raise ExtractionError(f"could not read page {number} of PDF {path}: {exc}") from exc
            if len(page_text.split()) < _MIN_PDF_PAGE_WORDS:
                warnings.append(f"page {number} has no extractable text")
                continue
            units.append(
                ExtractedUnit(
                    text=page_text,
                    locator=f"p.{number}",
                    loc_start=float(number),
                )
            )
    except _PDF_ERRORS as exc:
        if not path.exists():
            raise FileNotFoundError(path) from exc
        raise ExtractionError(f"could not read PDF {path}: {exc}") from exc
    finally:
        document.close()

    return ExtractionResult(units=units, warnings=warnings)
