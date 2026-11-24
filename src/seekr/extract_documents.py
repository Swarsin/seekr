"""Text extraction for Office documents (.docx, .pptx) and Jupyter notebooks (.ipynb)."""

from __future__ import annotations

import importlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
import zlib
from pathlib import Path
from typing import Any, BinaryIO

from .extract import ExtractionError, ExtractionResult, os_open_path
from .models import ExtractedUnit

__all__ = ["extract_docx", "extract_notebook", "extract_pptx"]

_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_HEADING_STYLE_RE = re.compile(r"^heading\s+(\d+)$")
_NOTEBOOK_CELL_TYPES = frozenset({"markdown", "code", "raw"})

_BASE_OFFICE_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    EOFError,
    KeyError,
    ValueError,
    TypeError,
    IndexError,
    AttributeError,
    LookupError,
    RuntimeError,
    UnicodeError,
    zipfile.BadZipFile,
    zipfile.LargeZipFile,
    zlib.error,
    ET.ParseError,
)


_DOCX_EXCEPTION_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("docx.exceptions", ("PythonDocxError",)),
    ("docx.opc.exceptions", ("OpcError",)),
    ("docx.oxml.exceptions", ("XmlchemyError",)),
    (
        "docx.image.exceptions",
        (
            "InvalidImageStreamError",
            "UnexpectedEndOfFileError",
            "UnrecognizedImageError",
        ),
    ),
    ("lxml.etree", ("LxmlError",)),
)
_PPTX_EXCEPTION_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pptx.exc", ("PythonPptxError",)),
    ("pptx.exceptions", ("PythonPptxError",)),
    ("lxml.etree", ("LxmlError",)),
)


def _load_package(module_name: str, package_name: str, label: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ExtractionError(
            f"{package_name} is required to extract {label} files"
        ) from exc


def _load_docx_types() -> tuple[type[Any], type[Any]]:
    try:
        paragraph_mod = importlib.import_module("docx.text.paragraph")
        table_mod = importlib.import_module("docx.table")
        return paragraph_mod.Paragraph, table_mod.Table
    except (ImportError, AttributeError) as exc:
        raise ExtractionError("python-docx is required to extract Word files") from exc


def _resolve_exception_types(
    root_module: Any,
    specs: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[type[BaseException], ...]:
    resolved: list[type[BaseException]] = list(_BASE_OFFICE_ERRORS)
    seen: set[type[BaseException]] = set(resolved)
    prefix = f"{root_module.__name__}."
    for module_name, attr_names in specs:
        target: Any = None
        try:
            target = importlib.import_module(module_name)
        except ImportError:
            if module_name.startswith(prefix):
                cur: Any = root_module
                for part in module_name[len(prefix) :].split("."):
                    cur = getattr(cur, part, None)
                    if cur is None:
                        break
                target = cur
        if target is None:
            continue
        for attr_name in attr_names:
            candidate = getattr(target, attr_name, None)
            if (
                isinstance(candidate, type)
                and issubclass(candidate, BaseException)
                and candidate not in seen
            ):
                seen.add(candidate)
                resolved.append(candidate)
    return tuple(resolved)


def _reject_ole2_header(handle: BinaryIO, source: Path) -> None:
    head = handle.read(len(_OLE2_MAGIC))
    if head == _OLE2_MAGIC:
        raise ExtractionError(
            f"password-protected or binary OLE2 Office document cannot be read: {source}"
        )
    handle.seek(0)


def _clean_inline(text: str) -> str:
    return " ".join(text.split())


def _docx_heading_level(style_name: str) -> int | None:
    normalised = style_name.strip().lower()
    if normalised == "title":
        return 1
    match = _HEADING_STYLE_RE.fullmatch(normalised)
    if match is None:
        return None
    level = int(match.group(1))
    return min(level, 6) if level >= 1 else None


def _format_table_rows(table: Any) -> list[str]:
    rows: list[str] = []
    for row in table.rows:
        cells: list[str] = []
        seen_tcs: set[int] = set()
        for cell in row.cells:
            tc = getattr(cell, "_tc", None)
            if tc is not None:
                tc_id = id(tc)
                if tc_id in seen_tcs:
                    continue
                seen_tcs.add(tc_id)
            cleaned = _clean_inline(cell.text or "")
            if cleaned:
                cells.append(cleaned)
        if cells:
            rows.append(" | ".join(cells))
    return rows


def _format_docx_paragraph(paragraph: Any) -> str:
    text = _clean_inline(paragraph.text or "")
    if not text:
        return ""
    style = getattr(paragraph, "style", None)
    style_name = getattr(style, "name", None) or ""
    level = _docx_heading_level(str(style_name))
    if level is not None:
        return f"{'#' * level} {text}"
    return text


def extract_docx(path: Path) -> ExtractionResult:
    """Extract Markdown-shaped text from a ``.docx`` Word document."""
    source = Path(path)
    docx_mod = _load_package("docx", "python-docx", "Word")
    paragraph_cls, table_cls = _load_docx_types()
    error_types = _resolve_exception_types(docx_mod, _DOCX_EXCEPTION_SPECS)

    lines: list[str] = []
    try:
        with os_open_path(source).open("rb") as handle:
            _reject_ole2_header(handle, source)
            document = docx_mod.Document(handle)
            for item in document.iter_inner_content():
                if isinstance(item, paragraph_cls):
                    line = _format_docx_paragraph(item)
                    if line:
                        lines.append(line)
                elif isinstance(item, table_cls):
                    lines.extend(_format_table_rows(item))
    except FileNotFoundError:
        raise
    except ExtractionError:
        raise
    except error_types as exc:
        raise ExtractionError(f"could not read Word document {source}: {exc}") from exc

    if not lines:
        return ExtractionResult(units=[], warnings=[])
    return ExtractionResult(
        units=[ExtractedUnit(text="\n".join(lines), locator="", loc_start=0.0)],
        warnings=[],
    )


def _collect_pptx_shape_text(
    shape: Any,
    title_element: Any,
    body_parts: list[str],
    table_parts: list[str],
) -> None:
    if title_element is not None and getattr(shape, "element", None) is title_element:
        return
    grouped = getattr(shape, "shapes", None)
    if grouped is not None:
        for child in grouped:
            _collect_pptx_shape_text(child, title_element, body_parts, table_parts)
        return
    if getattr(shape, "has_text_frame", False):
        for para in shape.text_frame.paragraphs:
            cleaned = _clean_inline(para.text or "")
            if cleaned:
                body_parts.append(cleaned)
    if getattr(shape, "has_table", False):
        table_parts.extend(_format_table_rows(shape.table))


def extract_pptx(path: Path) -> ExtractionResult:
    """Extract one unit per slide (with speaker notes) from a ``.pptx`` deck."""
    source = Path(path)
    pptx_mod = _load_package("pptx", "python-pptx", "PowerPoint")
    error_types = _resolve_exception_types(pptx_mod, _PPTX_EXCEPTION_SPECS)

    units: list[ExtractedUnit] = []
    warnings: list[str] = []
    try:
        with os_open_path(source).open("rb") as handle:
            _reject_ole2_header(handle, source)
            presentation = pptx_mod.Presentation(handle)
            for number, slide in enumerate(presentation.slides, start=1):
                title_parts: list[str] = []
                body_parts: list[str] = []
                table_parts: list[str] = []
                notes_parts: list[str] = []

                title_shape = getattr(slide.shapes, "title", None)
                title_element = (
                    getattr(title_shape, "element", None)
                    if title_shape is not None
                    else None
                )
                if title_shape is not None and getattr(title_shape, "has_text_frame", False):
                    title_text = _clean_inline(title_shape.text_frame.text or "")
                    if title_text:
                        title_parts.append(title_text)

                for shape in slide.shapes:
                    _collect_pptx_shape_text(shape, title_element, body_parts, table_parts)

                if getattr(slide, "has_notes_slide", False):
                    notes_tf = getattr(slide.notes_slide, "notes_text_frame", None)
                    if notes_tf is not None:
                        notes_text = _clean_inline(notes_tf.text or "")
                        if notes_text:
                            notes_parts.append(f"Speaker notes: {notes_text}")

                slide_parts = [*title_parts, *body_parts, *table_parts, *notes_parts]
                if not slide_parts:
                    warnings.append(f"slide {number} has no text")
                    continue
                units.append(
                    ExtractedUnit(
                        text="\n".join(slide_parts),
                        locator=f"slide {number}",
                        loc_start=float(number),
                    )
                )
    except FileNotFoundError:
        raise
    except ExtractionError:
        raise
    except error_types as exc:
        raise ExtractionError(f"could not read PowerPoint file {source}: {exc}") from exc

    return ExtractionResult(units=units, warnings=warnings)


def extract_notebook(path: Path) -> ExtractionResult:
    """Extract markdown, code and raw cells from an nbformat-4 ``.ipynb`` notebook."""
    source = Path(path)
    try:
        raw = os_open_path(source).read_bytes()
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ExtractionError(f"cannot read notebook {source}: {exc}") from exc

    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ExtractionError(f"invalid notebook JSON in {source}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ExtractionError(f"invalid notebook {source}: root must be a JSON object")

    nbformat = payload.get("nbformat")
    if type(nbformat) is not int or nbformat != 4:
        raise ExtractionError(
            f"unsupported or missing notebook format in {source}: expected nbformat 4"
        )

    if "cells" not in payload or not isinstance(payload["cells"], list):
        raise ExtractionError(f"invalid notebook {source}: missing 'cells' list")

    units: list[ExtractedUnit] = []
    for index, cell in enumerate(payload["cells"], start=1):
        if not isinstance(cell, dict):
            raise ExtractionError(
                f"invalid notebook {source}: cell {index} must be a JSON object"
            )
        if "cell_type" not in cell or not isinstance(cell["cell_type"], str):
            raise ExtractionError(
                f"invalid notebook {source}: cell {index} missing string 'cell_type'"
            )
        if cell["cell_type"] not in _NOTEBOOK_CELL_TYPES:
            continue
        if "source" not in cell:
            raise ExtractionError(
                f"invalid notebook {source}: cell {index} missing 'source'"
            )
        raw_source = cell["source"]
        if isinstance(raw_source, str):
            cell_text = raw_source
        elif isinstance(raw_source, list) and all(isinstance(part, str) for part in raw_source):
            cell_text = "".join(raw_source)
        else:
            raise ExtractionError(
                f"invalid notebook {source}: cell {index} 'source' must be a string or list"
            )

        cleaned = cell_text.strip()
        if not cleaned:
            continue
        units.append(
            ExtractedUnit(
                text=cleaned,
                locator=f"cell {index}",
                loc_start=float(index),
            )
        )

    return ExtractionResult(units=units, warnings=[])
