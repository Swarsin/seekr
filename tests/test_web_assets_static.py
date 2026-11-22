"""Static checks for packaged web UI assets and user documentation."""

from __future__ import annotations

import importlib.resources
import re
import shutil
import subprocess
import unittest
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = importlib.resources.files("seekr") / "web_assets"
INDEX_HTML = Path(str(ASSETS / "index.html"))
APP_JS = Path(str(ASSETS / "app.js"))
STYLES_CSS = Path(str(ASSETS / "styles.css"))
DOCS_WEB = ROOT / "docs" / "web.md"

EXPECTED_DOWNLOAD_LABEL = (
    "Download search model (about 130 MB) — the only time seekr uses the internet"
)

REQUIRED_IDS = (
    # Existing core UI elements
    "query",
    "search-form",
    "search-submit",
    "filters",
    "filter-path",
    "filter-kind",
    "filter-since",
    "filter-k",
    "filter-per-file",
    "filters-clear",
    "hit-list",
    "results-meta",
    "results-state",
    "root-list",
    "add-folder-form",
    "folder-path",
    "update-index",
    "check-model",
    "job",
    "job-body",
    # Course, preview, and first-run UI elements
    "module-filter",
    "week-from",
    "week-to",
    "spread-strip",
    "results",
    "preview-panel",
    "preview-text",
    "preview-image",
    "open-button",
    "copy-location",
    "posix-toggle",
    "first-run-panel",
    "download-model-button",
)


class _AssetHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.button_texts: dict[str, list[str]] = {}
        self.inline_script_bodies: list[str] = []
        self.inline_event_attrs: list[str] = []
        self.style_attrs: list[str] = []
        self._active_id_stack: list[tuple[str, str | None]] = []
        self._in_script = False
        self._script_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        elem_id = attr_map.get("id")
        if elem_id:
            self.ids.add(elem_id)
        for key, _val in attrs:
            lower = key.lower()
            if lower.startswith("on"):
                self.inline_event_attrs.append(f"<{tag} {key}=...>")
            if lower == "style":
                self.style_attrs.append(f"<{tag} style=...>")
        if tag.lower() == "script":
            self._in_script = True
            self._script_buf = []
        if tag.lower() not in {"input", "img", "br", "hr", "meta", "link"}:
            self._active_id_stack.append((tag.lower(), elem_id))

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower == "script":
            body = "".join(self._script_buf).strip()
            if body:
                self.inline_script_bodies.append(body)
            self._in_script = False
            self._script_buf = []
        while self._active_id_stack:
            popped_tag, _ = self._active_id_stack.pop()
            if popped_tag == lower:
                break

    def handle_data(self, data: str) -> None:
        if self._in_script:
            self._script_buf.append(data)
        for _tag, elem_id in self._active_id_stack:
            if elem_id:
                self.button_texts.setdefault(elem_id, []).append(data)


def _find_node() -> str | None:
    found = shutil.which("node")
    if found:
        return found
    candidate = Path.home() / ".local" / "bin" / "node"
    if candidate.is_file():
        return str(candidate)
    return None


class WebAssetsStaticTests(unittest.TestCase):
    def test_required_ids_present(self) -> None:
        html_text = INDEX_HTML.read_text(encoding="utf-8")
        parser = _AssetHTMLParser()
        parser.feed(html_text)
        parser.close()

        missing = [elem_id for elem_id in REQUIRED_IDS if elem_id not in parser.ids]
        self.assertEqual(missing, [], f"missing required DOM IDs in index.html: {missing}")

        button_label = " ".join(
            "".join(parser.button_texts.get("download-model-button", [])).split()
        )
        self.assertEqual(button_label, EXPECTED_DOWNLOAD_LABEL)

    def test_no_html_injection_sinks(self) -> None:
        html_text = INDEX_HTML.read_text(encoding="utf-8")
        js_text = APP_JS.read_text(encoding="utf-8")

        parser = _AssetHTMLParser()
        parser.feed(html_text)
        parser.close()
        self.assertEqual(parser.inline_script_bodies, [])
        self.assertEqual(parser.inline_event_attrs, [])
        self.assertEqual(parser.style_attrs, [])

        sinks = (
            "innerHTML",
            "outerHTML",
            "insertAdjacentHTML",
            "document.write",
            "eval(",
            "new Function",
        )
        for filename, content in (("index.html", html_text), ("app.js", js_text)):
            for sink in sinks:
                self.assertNotIn(sink, content, f"forbidden sink {sink!r} found in {filename}")

        self.assertIsNone(
            re.search(r"\?[^\n\"']*X-Seekr-Token", js_text, re.IGNORECASE),
            "session token must never appear in a URL query string",
        )

    def test_object_urls_are_revoked(self) -> None:
        js_text = APP_JS.read_text(encoding="utf-8")
        self.assertIn("URL.createObjectURL", js_text)
        self.assertIn("URL.revokeObjectURL", js_text)

        for needle in (
            "remove-root",
            "/api/roots/remove",
            "/api/course",
            "/api/preview",
            "/api/preview/page",
            "/api/open",
            "/api/setup/download",
        ):
            self.assertIn(needle, js_text)

    def test_app_js_parses(self) -> None:
        node = _find_node()
        if node is None:
            self.skipTest("node is not installed")
        result = subprocess.run(
            [node, "--check", str(APP_JS)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_docs_web_md_updated(self) -> None:
        doc = DOCS_WEB.read_text(encoding="utf-8")
        for outdated in (
            "There is no **Open file** or preview button",
            "There is no **Remove folder** button",
            "The web page never downloads a model",
        ):
            self.assertNotIn(outdated, doc)

        for expected in (
            "Appears in",
            "Module",
            "Week",
            "Open",
            "POSIX",
            "Remove",
            EXPECTED_DOWNLOAD_LABEL,
        ):
            self.assertIn(expected, doc)


if __name__ == "__main__":
    unittest.main()
