"""Tests for the BEIR SciFact conversion and opt-in download helpers.

Everything here is offline: download tests patch ``urllib.request.urlopen`` and
never touch the network.  The converter uses only local JSONL/TSV fixtures.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest import mock

from seekr import evaluation

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "eval" / "beir_scifact.py"


def _beir_module() -> object:
    spec = importlib.util.spec_from_file_location("seekr_beir_scifact_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ConvertBeirTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.corpus = self.tmp / "corpus.jsonl"
        self.queries = self.tmp / "queries.jsonl"
        self.qrels = self.tmp / "test.tsv"
        self.out = self.tmp / "converted"

        self.corpus.write_text(
            "\n".join(
                [
                    json.dumps(
                        {"_id": "d1", "title": "Doc One", "text": "First body."}
                    ),
                    json.dumps(
                        {"_id": "d2", "title": "Doc Two", "text": "Second body."}
                    ),
                    json.dumps(
                        {"_id": "d3", "title": "Doc Three", "text": "Third body."}
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        self.queries.write_text(
            "\n".join(
                [
                    json.dumps({"_id": "q1", "text": "alpha query"}),
                    json.dumps({"_id": "q2", "text": "beta query"}),
                    json.dumps({"_id": "q3", "text": "orphan query"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        self.qrels.write_text(
            "query-id\tcorpus-id\tscore\n"
            "q1\td1\t2\n"
            "q1\td2\t0\n"
            "q1\tmissing-doc\t1\n"
            "q2\td3\t1\n",
            encoding="utf-8",
        )

    def test_convert_beir_writes_files_and_filters_queries(self) -> None:
        module = _beir_module()
        module.convert_beir(self.corpus, self.queries, self.qrels, self.out)

        self.assertEqual(
            (self.out / "corpus" / "d1.txt").read_text(encoding="utf-8"),
            "Doc One\n\nFirst body.",
        )
        self.assertEqual(
            (self.out / "corpus" / "d2.txt").read_text(encoding="utf-8"),
            "Doc Two\n\nSecond body.",
        )
        self.assertEqual(
            (self.out / "corpus" / "d3.txt").read_text(encoding="utf-8"),
            "Doc Three\n\nThird body.",
        )

        rows = [
            json.loads(line)
            for line in (self.out / "queries.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual([row["id"] for row in rows], ["q1", "q2"])
        self.assertEqual(
            rows[0], {"id": "q1", "query": "alpha query", "relevant": ["d1.txt"]}
        )
        self.assertEqual(
            rows[1], {"id": "q2", "query": "beta query", "relevant": ["d3.txt"]}
        )

    def test_converted_queries_find_judged_files_through_load_queries(self) -> None:
        module = _beir_module()
        module.convert_beir(self.corpus, self.queries, self.qrels, self.out)

        queries = evaluation.load_queries(self.out / "queries.jsonl")
        by_id = {query.id: query for query in queries}
        self.assertEqual(by_id["q1"].relevant, frozenset({"d1.txt"}))
        self.assertEqual(by_id["q2"].relevant, frozenset({"d3.txt"}))

        corpus_root = self.out / "corpus"
        for query in queries:
            for key in query.relevant:
                self.assertTrue((corpus_root / key).is_file(), key)

        # A ranking of the round-trip key must score as a hit; bare doc IDs
        # (the old output) would evaluate every SciFact query to zero.
        self.assertEqual(
            evaluation.recall_at_k(["d1.txt", "d2.txt"], by_id["q1"].relevant, 2),
            1.0,
        )

    def test_convert_is_repeatable(self) -> None:
        module = _beir_module()

        module.convert_beir(self.corpus, self.queries, self.qrels, self.out)
        module.convert_beir(self.corpus, self.queries, self.qrels, self.out)

        self.assertEqual(
            len(list((self.out / "queries.jsonl").read_text(encoding="utf-8").splitlines())),
            2,
        )

    def test_document_ids_cannot_escape_the_corpus_directory(self) -> None:
        module = _beir_module()
        self.corpus.write_text(
            json.dumps({"_id": "../escape", "title": "Bad", "text": "path"}) + "\n",
            encoding="utf-8",
        )

        with self.assertRaises(ValueError):
            module.convert_beir(self.corpus, self.queries, self.qrels, self.out)

        self.assertFalse((self.tmp / "escape.txt").exists())


class DownloadAndExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _valid_archive(self, path: Path) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("scifact/corpus.jsonl", b'{"_id":"d1"}\n')
            archive.writestr("scifact/queries.jsonl", b'{"_id":"q1"}\n')
            archive.writestr("scifact/qrels/test.tsv", b"q1\td1\t1\n")

    def test_existing_valid_archive_is_reused_without_network(self) -> None:
        module = _beir_module()
        archive = self.tmp / "scifact.zip"
        self._valid_archive(archive)

        with mock.patch.object(module.urllib.request, "urlopen") as urlopen:
            result = module.download_scifact_archive(archive)

        self.assertEqual(result, archive)
        urlopen.assert_not_called()

    def test_non_https_url_is_rejected_without_network(self) -> None:
        module = _beir_module()

        with mock.patch.object(
            module.urllib.request, "urlopen"
        ) as urlopen, self.assertRaises(ValueError):
            module.download_scifact_archive(
                self.tmp / "scifact.zip", url="http://example.test/scifact.zip"
            )

        urlopen.assert_not_called()

    def test_download_failure_has_actionable_hugging_face_fallback(self) -> None:
        module = _beir_module()
        archive = self.tmp / "scifact.zip"

        with mock.patch.object(
            module.urllib.request,
            "urlopen",
            side_effect=urllib.error.URLError("offline test"),
        ), self.assertRaises(RuntimeError) as raised:
            module.download_scifact_archive(archive)

        message = str(raised.exception)
        self.assertIn("huggingface.co/datasets/BeIR/scifact", message)
        self.assertIn("manual", message.lower())
        self.assertIn("corpus.jsonl", message)
        self.assertIn("qrels/test.tsv", message)
        self.assertFalse(archive.exists())

    def test_selective_extraction_ignores_or_rejects_traversal_entries(self) -> None:
        module = _beir_module()
        archive = self.tmp / "scifact.zip"
        destination = self.tmp / "source"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("scifact/corpus.jsonl", b'{"_id":"d1"}\n')
            handle.writestr("scifact/queries.jsonl", b'{"_id":"q1"}\n')
            handle.writestr("scifact/qrels/test.tsv", b"q1\td1\t1\n")
            handle.writestr("../escape.txt", b"escaped")

        with self.assertRaises(ValueError):
            module.extract_scifact_zip(archive, destination)

        self.assertFalse((self.tmp / "escape.txt").exists())

    def test_extraction_size_limit_is_enforced(self) -> None:
        module = _beir_module()
        archive = self.tmp / "scifact.zip"
        destination = self.tmp / "source"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("scifact/corpus.jsonl", b"x" * 2048)
            handle.writestr("scifact/queries.jsonl", b'{"_id":"q1"}\n')
            handle.writestr("scifact/qrels/test.tsv", b"q1\td1\t1\n")

        with self.assertRaises(RuntimeError):
            module.extract_scifact_zip(archive, destination, max_total_bytes=100)

        self.assertFalse((destination / "corpus.jsonl").exists())

    def test_prepare_converts_an_existing_archive_without_network(self) -> None:
        module = _beir_module()
        data_dir = self.tmp / "data"
        data_dir.mkdir()
        archive = data_dir / "scifact.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr(
                "scifact/corpus.jsonl",
                json.dumps({"_id": "d1", "title": "Doc", "text": "Body"}) + "\n",
            )
            handle.writestr(
                "scifact/queries.jsonl",
                json.dumps({"_id": "q1", "text": "needle"}) + "\n",
            )
            handle.writestr("scifact/qrels/test.tsv", "q1\td1\t1\n")

        with mock.patch.object(module.urllib.request, "urlopen") as urlopen:
            output_dir = module.prepare_scifact(data_dir, allow_download=False)

        urlopen.assert_not_called()
        self.assertEqual(
            (output_dir / "corpus" / "d1.txt").read_text(encoding="utf-8"),
            "Doc\n\nBody",
        )
        rows = [
            json.loads(line)
            for line in (output_dir / "queries.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(
            rows, [{"id": "q1", "query": "needle", "relevant": ["d1.txt"]}]
        )

    def test_missing_required_member_is_reported(self) -> None:
        module = _beir_module()
        archive = self.tmp / "scifact.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("scifact/corpus.jsonl", b'{"_id":"d1"}\n')

        with self.assertRaises(RuntimeError) as raised:
            module.extract_scifact_zip(archive, self.tmp / "source")

        self.assertIn("queries.jsonl", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
