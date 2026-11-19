"""Offline tests for the BEIR NFCorpus and FiQA dataset converters."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from typing import Self
from unittest import mock

from seekr import evaluation

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BeirDatasetsConverterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.beir_common = _load_module(
            "seekr_test_beir_common", ROOT / "eval" / "beir_common.py"
        )
        self.nfcorpus = _load_module(
            "seekr_test_beir_nfcorpus", ROOT / "eval" / "beir_nfcorpus.py"
        )
        self.fiqa = _load_module(
            "seekr_test_beir_fiqa", ROOT / "eval" / "beir_fiqa.py"
        )

    def _write_archive(self, path: Path, prefix: str) -> None:
        with zipfile.ZipFile(path, "w") as handle:
            handle.writestr(
                f"{prefix}/corpus.jsonl",
                "\n".join(
                    [
                        json.dumps(
                            {"_id": "d1", "title": "Title One", "text": "Body one."}
                        ),
                        json.dumps({"_id": "d2", "title": "", "text": "Body only."}),
                    ]
                )
                + "\n",
            )
            handle.writestr(
                f"{prefix}/queries.jsonl",
                "\n".join(
                    [
                        json.dumps({"_id": "q1", "text": "first question"}),
                        json.dumps({"_id": "q2", "text": "unjudged question"}),
                    ]
                )
                + "\n",
            )
            handle.writestr(
                f"{prefix}/qrels/test.tsv",
                "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t0\n",
            )

    def test_prepare_from_existing_archive_without_network(self) -> None:
        for name, module, archive_name, prepare_name in (
            ("nfcorpus", self.nfcorpus, "nfcorpus.zip", "prepare_nfcorpus"),
            ("fiqa", self.fiqa, "fiqa.zip", "prepare_fiqa"),
        ):
            with self.subTest(dataset=name):
                data_dir = self.tmp / name
                data_dir.mkdir(parents=True)
                self._write_archive(data_dir / archive_name, name)
                with mock.patch.object(
                    self.beir_common.urllib.request, "urlopen"
                ) as urlopen:
                    out_dir = getattr(module, prepare_name)(
                        data_dir, allow_download=False
                    )
                urlopen.assert_not_called()
                self.assertEqual(
                    (out_dir / "corpus" / "d1.txt").read_text(encoding="utf-8"),
                    "Title One\n\nBody one.",
                )
                self.assertEqual(
                    (out_dir / "corpus" / "d2.txt").read_text(encoding="utf-8"),
                    "Body only.",
                )
                queries = evaluation.load_queries(out_dir / "queries.jsonl")
                self.assertEqual(len(queries), 1)
                self.assertEqual(queries[0].id, "q1")
                self.assertEqual(queries[0].relevant, frozenset({"d1.txt"}))

    def test_no_download_missing_sources_prints_hf_fallback(self) -> None:
        for name, module, hf_fragment in (
            ("nfcorpus", self.nfcorpus, "huggingface.co/datasets/BeIR/nfcorpus"),
            ("fiqa", self.fiqa, "huggingface.co/datasets/BeIR/fiqa"),
        ):
            with self.subTest(dataset=name):
                data_dir = self.tmp / f"missing-{name}"
                prepare = (
                    module.prepare_nfcorpus
                    if name == "nfcorpus"
                    else module.prepare_fiqa
                )
                with self.assertRaises(RuntimeError) as raised:
                    prepare(data_dir, allow_download=False)
                message = str(raised.exception)
                self.assertIn(hf_fragment, message)
                self.assertIn("--no-download", message)

    def test_no_download_cli_returns_nonzero_with_fallback(self) -> None:
        for name, module, hf_fragment in (
            ("nfcorpus", self.nfcorpus, "huggingface.co/datasets/BeIR/nfcorpus"),
            ("fiqa", self.fiqa, "huggingface.co/datasets/BeIR/fiqa"),
        ):
            with self.subTest(dataset=name):
                data_dir = self.tmp / f"cli-{name}"
                stderr = io.StringIO()
                with mock.patch("sys.stderr", stderr):
                    code = module.main(["--data-dir", str(data_dir), "--no-download"])
                self.assertEqual(code, 1)
                self.assertIn(hf_fragment, stderr.getvalue())

    def test_mocked_download_roundtrip(self) -> None:
        archive_bytes_io = io.BytesIO()
        with zipfile.ZipFile(archive_bytes_io, "w") as handle:
            handle.writestr(
                "nfcorpus/corpus.jsonl",
                json.dumps({"_id": "doc1", "title": "Med", "text": "Note."}) + "\n",
            )
            handle.writestr(
                "nfcorpus/queries.jsonl",
                json.dumps({"_id": "q1", "text": "med note"}) + "\n",
            )
            handle.writestr("nfcorpus/qrels/test.tsv", "q1\tdoc1\t2\n")
        payload = archive_bytes_io.getvalue()

        class _FakeResponse(io.BytesIO):
            def __init__(self, data: bytes) -> None:
                super().__init__(data)
                self.headers = {"Content-Length": str(len(data))}

            def __enter__(self) -> Self:
                return self

            def __exit__(self, *args: object) -> bool:
                self.close()
                return False

        data_dir = self.tmp / "mock-dl"
        with mock.patch.object(
            self.beir_common.urllib.request,
            "urlopen",
            return_value=_FakeResponse(payload),
        ):
            out_dir = self.nfcorpus.prepare_nfcorpus(data_dir, allow_download=True)

        self.assertTrue((data_dir / "nfcorpus.zip").is_file())
        self.assertEqual(
            (out_dir / "corpus" / "doc1.txt").read_text(encoding="utf-8"),
            "Med\n\nNote.",
        )

    def test_non_https_and_traversal_and_size_limit_rejected(self) -> None:
        with mock.patch.object(
            self.beir_common.urllib.request, "urlopen"
        ) as urlopen, self.assertRaises(ValueError):
            self.fiqa.download_fiqa_archive(
                self.tmp / "fiqa.zip", url="http://example.test/fiqa.zip"
            )
        urlopen.assert_not_called()

        bad_zip = self.tmp / "bad-traversal.zip"
        with zipfile.ZipFile(bad_zip, "w") as handle:
            handle.writestr("fiqa/corpus.jsonl", b'{"_id":"d1"}\n')
            handle.writestr("fiqa/queries.jsonl", b'{"_id":"q1"}\n')
            handle.writestr("fiqa/qrels/test.tsv", b"q1\td1\t1\n")
            handle.writestr("../escape.txt", b"boom")
        with self.assertRaises(ValueError):
            self.fiqa.extract_fiqa_zip(bad_zip, self.tmp / "out-traversal")

        big_zip = self.tmp / "big.zip"
        with zipfile.ZipFile(big_zip, "w") as handle:
            handle.writestr("fiqa/corpus.jsonl", b"x" * 2048)
            handle.writestr("fiqa/queries.jsonl", b'{"_id":"q1"}\n')
            handle.writestr("fiqa/qrels/test.tsv", b"q1\td1\t1\n")
        with self.assertRaises(RuntimeError):
            self.fiqa.extract_fiqa_zip(
                big_zip, self.tmp / "out-big", max_total_bytes=100
            )

    def test_download_network_error_includes_hf_fallback(self) -> None:
        with mock.patch.object(
            self.beir_common.urllib.request,
            "urlopen",
            side_effect=urllib.error.URLError("offline"),
        ), self.assertRaises(RuntimeError) as raised:
            self.fiqa.download_fiqa_archive(self.tmp / "fail-fiqa.zip")
        self.assertIn("huggingface.co/datasets/BeIR/fiqa", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
