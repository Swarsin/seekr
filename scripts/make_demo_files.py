"""Regenerate the demo course PDF, Word, PowerPoint and notebook files."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import docx
import pptx
import pymupdf
from pptx.util import Inches

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


def build_docx(docx_path: Path) -> None:
    """Create the demo Word worksheet on natural deduction."""
    doc = docx.Document()
    doc.add_heading("Natural Deduction Worksheet", level=1)
    doc.add_paragraph(
        "Natural deduction formalizes everyday mathematical reasoning through "
        "pairs of introduction and elimination rules for each logical connective. "
        "Rather than memorizing large axiom lists, a Fitch-style derivation tracks "
        "which temporary assumptions remain active at each numbered line."
    )

    doc.add_heading("Fitch Subproofs and Assumption Discharge", level=2)
    doc.add_paragraph(
        "To prove an implication A implies B, open an indented subproof box "
        "with A as a temporary hypothesis. Derive B using earlier active lines, "
        "then close the box and apply implication introduction to discharge the "
        "assumption. Once discharged, formulas inside the closed box cannot be "
        "referenced directly by later steps outside that scope."
    )
    doc.add_paragraph(
        "Negation introduction follows the same subproof discipline: assume A, "
        "derive an explicit contradiction symbol bottom, and conclude not-A "
        "outside the subproof. Double negation elimination or proof by "
        "contradiction (reductio ad absurdum) separates classical derivations "
        "from purely intuitionistic reasoning."
    )

    doc.add_heading("Core Connective Rules", level=2)
    table = doc.add_table(rows=4, cols=3)
    rows = [
        ("Connective", "Introduction Rule", "Elimination Rule"),
        ("Conjunction", "Combine two proven formulas", "Project left or right conjunct"),
        ("Implication", "Open and discharge subproof", "Apply modus ponens to premise"),
        ("Disjunction", "Weaken either proven disjunct", "Split into two case subproofs"),
    ]
    for r_idx, row_vals in enumerate(rows):
        for c_idx, val in enumerate(row_vals):
            table.cell(r_idx, c_idx).text = val

    doc.add_paragraph(
        "Worksheet tip: label every justification with the exact rule name and "
        "spanning line range so automated checkers and human graders can audit "
        "each assumption discharge cleanly."
    )
    docx_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(docx_path)


def build_pptx(pptx_path: Path) -> None:
    """Create the demo PowerPoint deck on Dijkstra's shortest-path algorithm."""
    prs = pptx.Presentation()

    slide1 = prs.slides.add_slide(prs.slide_layouts[1])
    slide1.shapes.title.text = "Single-Source Shortest Paths: Dijkstra"
    slide1.placeholders[1].text = (
        "Find minimum-weight routes from one source vertex s to all reachable vertices.\n"
        "Maintain tentative distance estimates dist[v] initialized to infinity.\n"
        "Pop the unvisited vertex u with smallest dist[u] from a binary min-priority queue."
    )
    slide1.notes_slide.notes_text_frame.text = (
        "Remind students that Dijkstra commits vertices permanently in non-decreasing "
        "distance order only when all edge weights are non-negative."
    )

    slide2 = prs.slides.add_slide(prs.slide_layouts[1])
    slide2.shapes.title.text = "Greedy Edge Relaxation Mechanics"
    slide2.placeholders[1].text = (
        "For each outgoing edge (u, v) of weight w, test whether dist[u] + w < dist[v].\n"
        "If shorter, lower dist[v] and update predecessor[v] = u.\n"
        "With adjacency lists and a binary heap, runtime is O((V + E) log V)."
    )
    slide2.notes_slide.notes_text_frame.text = (
        "Lazy deletion in Python heapq avoids decrease-key bookkeeping while preserving "
        "O(E log V) asymptotic runtime."
    )

    slide3 = prs.slides.add_slide(prs.slide_layouts[1])
    slide3.shapes.title.text = "Negative Weights and Bellman-Ford"
    slide3.placeholders[1].text = (
        "A negative edge encountered after popping a neighbor breaks Dijkstra's greedy invariant."
    )
    table = slide3.shapes.add_table(3, 3, Inches(0.8), Inches(2.6), Inches(8.0), Inches(2.2)).table
    table_rows = [
        ("Algorithm", "Edge Weight Constraint", "Time Complexity"),
        ("Dijkstra", "Non-negative weights only", "O((V + E) log V)"),
        ("Bellman-Ford", "Allows negative edges, detects cycles", "O(V * E)"),
    ]
    for r_idx, row_vals in enumerate(table_rows):
        for c_idx, val in enumerate(row_vals):
            table.cell(r_idx, c_idx).text = val
    slide3.notes_slide.notes_text_frame.text = (
        "Use Bellman-Ford when negative edge weights or arbitrage cycle checks are required."
    )

    pptx_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(pptx_path)


def build_notebook(ipynb_path: Path) -> None:
    """Create the demo Jupyter notebook on stratified cross-validation."""
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Stratified K-Fold Cross-Validation Lab\n",
                    "\n",
                    (
                        "Imbalanced binary classification datasets distort plain k-fold splits "
                        "when some folds receive far fewer positive examples than others. "
                        "Stratified k-fold partitioning preserves the target class ratio inside "
                        "every training and validation fold before hyperparameter tuning begins."
                    ),
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": 1,
                "source": [
                    "def stratified_folds(labels, k=5):\n",
                    "    positives = [idx for idx, y in enumerate(labels) if y == 1]\n",
                    "    negatives = [idx for idx, y in enumerate(labels) if y == 0]\n",
                    "    folds = [[] for _ in range(k)]\n",
                    "    for pos_idx, item in enumerate(positives):\n",
                    "        folds[pos_idx % k].append(item)\n",
                    "    for neg_idx, item in enumerate(negatives):\n",
                    "        folds[neg_idx % k].append(item)\n",
                    "    return folds\n",
                ],
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": ["Fold balance verified across 5 stratified splits.\n"],
                    }
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Hyperparameter Tuning and ROC-AUC Diagnostics\n",
                    "\n",
                    (
                        "During grid search, evaluate each candidate hyperparameter tuple by "
                        "averaging validation ROC-AUC scores across all k stratified folds. "
                        "Because ROC-AUC integrates true-positive rate against false-positive "
                        "rate across every probability threshold, it ranks classifier ordering "
                        "quality without committing prematurely to a fixed decision cutoff."
                    ),
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": 2,
                "source": [
                    "def select_best_hyperparameters(grid_scores):\n",
                    "    # Pick the setting with highest mean validation ROC-AUC.\n",
                    "    return max(grid_scores, key=lambda entry: sum(entry[1]) / len(entry[1]))\n",
                ],
                "outputs": [],
            },
            {
                "cell_type": "raw",
                "metadata": {},
                "source": (
                    "Lab checklist: report both mean ROC-AUC and per-fold standard deviation "
                    "when comparing tuned candidate models on the final test split."
                ),
            },
        ],
    }
    ipynb_path.parent.mkdir(parents=True, exist_ok=True)
    ipynb_path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    for rel_md in PDF_SOURCES:
        md_path = DEMO_COURSE_DIR / rel_md
        pdf_path = md_path.with_suffix(".pdf")
        render_pdf(md_path, pdf_path)
        print(f"Generated {pdf_path.relative_to(DEMO_COURSE_DIR)} ({pdf_path.stat().st_size} bytes)")

    docx_path = DEMO_COURSE_DIR / "4CCS1LGC-logic/week2/natural-deduction-worksheet.docx"
    build_docx(docx_path)
    print(f"Generated {docx_path.relative_to(DEMO_COURSE_DIR)} ({docx_path.stat().st_size} bytes)")

    pptx_path = DEMO_COURSE_DIR / "data-structures/week3/shortest-paths-dijkstra-slides.pptx"
    build_pptx(pptx_path)
    print(f"Generated {pptx_path.relative_to(DEMO_COURSE_DIR)} ({pptx_path.stat().st_size} bytes)")

    ipynb_path = DEMO_COURSE_DIR / "machine-learning/week3/cross-validation-lab.ipynb"
    build_notebook(ipynb_path)
    print(f"Generated {ipynb_path.relative_to(DEMO_COURSE_DIR)} ({ipynb_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
