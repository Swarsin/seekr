"""Tests for the ONNX Runtime embedding and reranking backends.

Every test here uses a fake session and a fake tokenizer, so no model files,
onnxruntime session or network access is needed.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest import mock

from seekr.embed import BATCH_SIZE, QUERY_PREFIX
from seekr.offline import ModelUnavailableError

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None

if _HAS_NUMPY:
    import numpy as np


@dataclass
class _Port:
    name: str
    shape: list[Any] = field(default_factory=list)


@dataclass
class _Encoding:
    ids: list[int]
    attention_mask: list[int]
    type_ids: list[int]


class FakeTokenizer:
    """Whitespace tokenizer that pads batches to the longest row.

    Token 0 is a CLS stand-in, so a text of ``n`` words has ``n + 1`` tokens.
    """

    def __init__(self) -> None:
        self.seen: list[list[Any]] = []
        self.truncation: dict[str, Any] | None = None
        self.padding = False

    def enable_truncation(self, max_length: int, **kwargs: Any) -> None:
        self.truncation = {"max_length": max_length, **kwargs}

    def enable_padding(self, **kwargs: Any) -> None:
        self.padding = True

    def encode_batch(self, inputs: Sequence[Any]) -> list[_Encoding]:
        self.seen.append(list(inputs))
        rows: list[list[int]] = []
        for item in inputs:
            text = " ".join(item) if isinstance(item, tuple) else item
            rows.append([101] + [7] * len(text.split()))
        if self.truncation is not None:
            rows = [row[: self.truncation["max_length"]] for row in rows]
        width = max(len(row) for row in rows)
        return [
            _Encoding(
                ids=row + [0] * (width - len(row)),
                attention_mask=[1] * len(row) + [0] * (width - len(row)),
                type_ids=[0] * width,
            )
            for row in rows
        ]


class FakeEmbedSession:
    """Returns 3.0 on the CLS token and 1.0 in column 0 elsewhere."""

    def __init__(self, dim: int = 4, *, shape_known: bool = True) -> None:
        self.dim = dim
        self.shape_known = shape_known
        self.calls: list[dict[str, Any]] = []

    def get_inputs(self) -> list[_Port]:
        return [_Port("input_ids"), _Port("attention_mask")]

    def get_outputs(self) -> list[_Port]:
        last = self.dim if self.shape_known else "hidden"
        return [_Port("last_hidden_state", ["batch", "seq", last])]

    def run(self, _outputs: Any, feed: dict[str, Any]) -> list[Any]:
        self.calls.append(feed)
        batch, seq = feed["input_ids"].shape
        out = np.zeros((batch, seq, self.dim), np.float32)
        out[:, 0, :] = 3.0
        out[:, 1:, 0] = 1.0
        return [out]


def _model_dir(root: Path, *, pooling: str | None = "cls", files: Sequence[str] = ()) -> Path:
    names = files or ("onnx/model.onnx", "tokenizer.json", "config.json")
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    if pooling is not None:
        pool = root / "1_Pooling" / "config.json"
        pool.parent.mkdir(parents=True, exist_ok=True)
        pool.write_text(
            json.dumps(
                {
                    "pooling_mode_cls_token": pooling == "cls",
                    "pooling_mode_mean_tokens": pooling == "mean",
                }
            ),
            encoding="utf-8",
        )
    return root


@unittest.skipUnless(_HAS_NUMPY, "numpy is not installed")
class OnnxEmbedderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _embedder(self, *, pooling: str | None = "cls", session: Any = None) -> Any:
        from seekr.onnx_backend import OnnxEmbedder

        self.session = session or FakeEmbedSession()
        self.tokenizer = FakeTokenizer()
        self.session_paths: list[Path] = []
        self.tokenizer_paths: list[Path] = []

        def session_factory(path: Path) -> Any:
            self.session_paths.append(path)
            return self.session

        def tokenizer_factory(path: Path) -> Any:
            self.tokenizer_paths.append(path)
            return self.tokenizer

        return OnnxEmbedder(
            "BAAI/bge-small-en-v1.5",
            model_dir=_model_dir(self.root, pooling=pooling),
            session_factory=session_factory,
            tokenizer_factory=tokenizer_factory,
        )

    def test_cls_pooling_and_normalisation(self) -> None:
        vecs = self._embedder(pooling="cls").embed_documents(["a", "b c"])
        self.assertEqual(vecs.dtype, np.float32)
        self.assertEqual(vecs.shape, (2, 4))
        np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, rtol=1e-6)
        np.testing.assert_allclose(vecs[0], np.full(4, 0.5), rtol=1e-6)

    def test_mean_pooling_respects_attention_mask(self) -> None:
        # "a" -> CLS + 1 token, padded to 3 tokens because "b c" has 2 words.
        vecs = self._embedder(pooling="mean").embed_documents(["a", "b c"])
        # Row 0 mean over 2 real tokens: [(3+1)/2, 3/2, 3/2, 3/2] = [2, 1.5, 1.5, 1.5]
        expected0 = np.array([2.0, 1.5, 1.5, 1.5], np.float32)
        np.testing.assert_allclose(vecs[0], expected0 / np.linalg.norm(expected0), rtol=1e-6)
        # Row 1 mean over 3 tokens: [(3+1+1)/3, 1, 1, 1]
        expected1 = np.array([5 / 3, 1.0, 1.0, 1.0], np.float32)
        np.testing.assert_allclose(vecs[1], expected1 / np.linalg.norm(expected1), rtol=1e-6)

    def test_query_prefix_applied(self) -> None:
        embedder = self._embedder()
        vec = embedder.embed_query("what is wal")
        self.assertEqual(vec.shape, (4,))
        self.assertEqual(self.tokenizer.seen[-1], [QUERY_PREFIX + "what is wal"])

    def test_batches_of_batch_size(self) -> None:
        embedder = self._embedder()
        out = embedder.embed_documents(["word"] * (BATCH_SIZE + 1))
        self.assertEqual(out.shape, (BATCH_SIZE + 1, 4))
        self.assertEqual(len(self.session.calls), 2)
        self.assertEqual(self.session.calls[0]["input_ids"].shape[0], BATCH_SIZE)
        self.assertEqual(self.session.calls[1]["input_ids"].shape[0], 1)

    def test_feed_uses_session_input_names_as_int64(self) -> None:
        embedder = self._embedder()
        embedder.embed_documents(["a b"])
        feed = self.session.calls[0]
        self.assertEqual(set(feed), {"input_ids", "attention_mask"})
        for value in feed.values():
            self.assertEqual(value.dtype, np.int64)

    def test_empty_input_returns_zero_rows(self) -> None:
        embedder = self._embedder()
        out = embedder.embed_documents([])
        self.assertEqual(out.shape, (0, 4))
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(self.session.calls, [])

    def test_truncation_enabled_at_512(self) -> None:
        from seekr.onnx_backend import MAX_SEQ_LEN

        self.assertEqual(MAX_SEQ_LEN, 512)
        embedder = self._embedder()
        out = embedder.embed_documents([" ".join(["w"] * 2000)])
        self.assertEqual(self.tokenizer.truncation["max_length"], 512)
        self.assertTrue(self.tokenizer.padding)
        self.assertEqual(self.session.calls[0]["input_ids"].shape, (1, 512))
        self.assertEqual(out.shape, (1, 4))

    def test_dim_from_output_shape_without_running(self) -> None:
        embedder = self._embedder()
        self.assertEqual(embedder.dim, 4)
        self.assertEqual(self.session.calls, [])

    def test_dim_probed_when_output_shape_is_symbolic(self) -> None:
        embedder = self._embedder(session=FakeEmbedSession(dim=6, shape_known=False))
        self.assertEqual(embedder.dim, 6)
        self.assertEqual(len(self.session.calls), 1)

    def test_lazy_loading(self) -> None:
        embedder = self._embedder()
        self.assertEqual(self.session_paths, [])
        embedder.embed_query("x")
        embedder.embed_query("y")
        self.assertEqual(len(self.session_paths), 1)
        self.assertEqual(len(self.tokenizer_paths), 1)

    def test_model_dir_used_directly(self) -> None:
        embedder = self._embedder()
        with mock.patch("huggingface_hub.try_to_load_from_cache") as cache:
            embedder.embed_query("x")
        cache.assert_not_called()
        self.assertEqual(self.session_paths, [self.root / "onnx" / "model.onnx"])
        self.assertEqual(self.tokenizer_paths, [self.root / "tokenizer.json"])

    def test_pooling_mode_from_config(self) -> None:
        from seekr.onnx_backend import read_pooling_mode

        cls_dir = _model_dir(self.root / "cls", pooling="cls")
        mean_dir = _model_dir(self.root / "mean", pooling="mean")
        self.assertEqual(read_pooling_mode(cls_dir / "1_Pooling" / "config.json"), "cls")
        self.assertEqual(read_pooling_mode(mean_dir / "1_Pooling" / "config.json"), "mean")
        self.assertEqual(read_pooling_mode(None), "mean")

    def test_missing_files_raise_without_network(self) -> None:
        from seekr.onnx_backend import OnnxEmbedder

        def no_network(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("network download attempted")

        with (
            mock.patch("huggingface_hub.try_to_load_from_cache", return_value=None),
            mock.patch("huggingface_hub.hf_hub_download", side_effect=no_network),
            mock.patch("huggingface_hub.snapshot_download", side_effect=no_network),
            self.assertRaisesRegex(ModelUnavailableError, "seekr models pull") as ctx,
        ):
            _ = OnnxEmbedder("BAAI/bge-small-en-v1.5").dim
        self.assertIn("onnx/model.onnx", str(ctx.exception))

    def test_missing_file_in_model_dir_raises(self) -> None:
        from seekr.onnx_backend import OnnxEmbedder

        root = _model_dir(self.root, files=("tokenizer.json", "config.json"))
        with self.assertRaisesRegex(ModelUnavailableError, "onnx/model.onnx"):
            _ = OnnxEmbedder("m", model_dir=root, session_factory=lambda p: None).dim


@unittest.skipUnless(_HAS_NUMPY, "numpy is not installed")
class ResolveModelFilesTests(unittest.TestCase):
    def test_cache_lookup_per_file_after_offline_env(self) -> None:
        from seekr.onnx_backend import resolve_model_files

        seen: list[tuple[str, str]] = []
        env_at_call: list[str | None] = []

        def fake_cache(repo: str, filename: str) -> str | None:
            import os

            env_at_call.append(os.environ.get("HF_HUB_OFFLINE"))
            seen.append((repo, filename))
            return None if filename == "1_Pooling/config.json" else f"/cache/{filename}"

        with mock.patch("huggingface_hub.try_to_load_from_cache", side_effect=fake_cache):
            files = resolve_model_files(
                "org/model",
                ("onnx/model.onnx", "tokenizer.json"),
                optional=("1_Pooling/config.json",),
            )
        self.assertEqual(
            files,
            {
                "onnx/model.onnx": Path("/cache/onnx/model.onnx"),
                "tokenizer.json": Path("/cache/tokenizer.json"),
            },
        )
        self.assertEqual(seen[-1], ("org/model", "1_Pooling/config.json"))
        self.assertEqual(set(env_at_call), {"1"})

    def test_non_string_cache_sentinel_counts_as_missing(self) -> None:
        from seekr.onnx_backend import resolve_model_files

        # huggingface_hub returns a sentinel object for "known to be absent".
        with (
            mock.patch("huggingface_hub.try_to_load_from_cache", return_value=object()),
            self.assertRaisesRegex(ModelUnavailableError, "tokenizer.json"),
        ):
            resolve_model_files("org/model", ("tokenizer.json",))

    def test_local_directory_model_name_is_used_as_model_dir(self) -> None:
        from seekr.onnx_backend import resolve_model_files

        with tempfile.TemporaryDirectory() as tmp:
            root = _model_dir(Path(tmp), pooling=None)
            with mock.patch("huggingface_hub.try_to_load_from_cache") as cache:
                files = resolve_model_files(str(root), ("tokenizer.json",))
            cache.assert_not_called()
            self.assertEqual(files["tokenizer.json"], root / "tokenizer.json")


if __name__ == "__main__":
    unittest.main()
