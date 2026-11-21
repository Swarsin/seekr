"""Regenerate the demo course PDF handouts from their Markdown sources."""

from __future__ import annotations

import html
import re
from pathlib import Path

import pymupdf

DEMO_COURSE_DIR = Path(__file__).resolve().parents[1] / "src" / "seekr" / "demo_course"

PDF_SOURCES: tuple[Path, ...] = (
    Path("machine-learning/week4/ensemble-methods-handout.md"),
    Path("data-structures/week3/hashing-in-graph-algorithms.md"),
    Path("4CCS1LGC-logic/week2/structural-induction-and-soundness.md"),
)

_STYLE = """
body {
    font-family: sans-serif;
    font-size: 10.5pt;
    line-height: 1.45;
    color: #1a1a1a;
}
h1 {
    font-size: 17pt;
    margin: 0 0 10pt 0;
    color: #111111;
}
h2 {
    font-size: 12.5pt;
    margin: 12pt 0 5pt 0;
    color: #222222;
}
p {
    margin: 0 0 8pt 0;
}
ul, ol {
    margin: 4pt 0 8pt 18pt;
    padding: 0;
}
li {
    margin: 0 0 4pt 0;
}
code {
    font-family: monospace;
    font-size: 9.5pt;
}
"""

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_CODE_RE = re.compile(r"`([^`]+?)`")
_OL_ITEM_RE = re.compile(r"^\d+\.\s+(.*)$")


def _format_inline(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = _CODE_RE.sub(r"<code>\1</code>", escaped)
    escaped = _BOLD_RE.sub(r"<strong>\1</strong>", escaped)
    return escaped


def _markdown_to_html(markdown_text: str) -> tuple[str, str]:
    """Convert simple lecture-note Markdown into an HTML body and extract title."""
    lines = markdown_text.splitlines()
    blocks: list[str] = []
    paragraph: list[str] = []
    list_tag: str | None = None
    list_items: list[str] = []
    title = "Course Handout"

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(f"<p>{_format_inline(' '.join(paragraph))}</p>")
            paragraph.clear()

    def flush_list() -> None:
        nonlocal list_tag
        if list_tag and list_items:
            items_html = "".join(f"<li>{_format_inline(item)}</li>" for item in list_items)
            blocks.append(f"<{list_tag}>{items_html}</{list_tag}>")
            list_items.clear()
            list_tag = None

    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            flush_paragraph()
            flush_list()
            continue

        if stripped.startswith("# "):
            flush_paragraph()
            flush_list()
            heading = stripped[2:].strip()
            if title == "Course Handout":
                title = heading
            blocks.append(f"<h1>{_format_inline(heading)}</h1>")
            continue

        if stripped.startswith("## "):
            flush_paragraph()
            flush_list()
            blocks.append(f"<h2>{_format_inline(stripped[3:].strip())}</h2>")
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            if list_tag not in (None, "ul"):
                flush_list()
            list_tag = "ul"
            list_items.append(stripped[2:].strip())
            continue

        ol_match = _OL_ITEM_RE.match(stripped)
        if ol_match is not None:
            flush_paragraph()
            if list_tag not in (None, "ol"):
                flush_list()
            list_tag = "ol"
            list_items.append(ol_match.group(1).strip())
            continue

        if list_tag is not None and raw.startswith(("  ", "\t")):
            list_items[-1] = f"{list_items[-1]} {stripped}"
            continue

        flush_list()
        paragraph.append(stripped)

    flush_paragraph()
    flush_list()

    body = "\n".join(blocks)
    return title, f"<!doctype html><html><body>{body}</body></html>"


def render_pdf(md_path: Path, pdf_path: Path) -> None:
    """Render one Markdown file into a clean PDF using PyMuPDF Story."""
    title, html_doc = _markdown_to_html(md_path.read_text(encoding="utf-8"))
    mediabox = pymupdf.paper_rect("letter")
    where = mediabox + (54, 54, -54, -54)

    story = pymupdf.Story(html=html_doc, user_css=_STYLE)
    writer = pymupdf.DocumentWriter(str(pdf_path))
    try:
        more = True
        while more:
            device = writer.begin_page(mediabox)
            more, _ = story.place(where)
            story.draw(device, None)
            writer.end_page()
    finally:
        writer.close()

    doc = pymupdf.open(pdf_path)
    try:
        doc.set_metadata({"title": title, "author": "seekr-search demo course"})
        data = doc.tobytes(garbage=4, deflate=True)
    finally:
        doc.close()
    pdf_path.write_bytes(data)


def main() -> None:
    for rel_md in PDF_SOURCES:
        md_path = DEMO_COURSE_DIR / rel_md
        pdf_path = md_path.with_suffix(".pdf")
        render_pdf(md_path, pdf_path)
        print(f"Generated {pdf_path.relative_to(DEMO_COURSE_DIR)} ({pdf_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
