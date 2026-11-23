"""Tests for embedding backends and the offline model adapter."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from seekr import offline
from seekr.config import Config
from seekr.embed import (
    QUERY_PREFIX,
    Embedder,
    FakeEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
)

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_NEEDS_NUMPY = unittest.skipUnless(_HAS_NUMPY, "numpy is not installed")


class _FakeSentenceTransformer:
    def __init__(self, source: str, **kwargs: object) -> None:
        self.source = source
        self.kwargs = kwargs


class _FakeCrossEncoder:
    def __init__(self, source: str, **kwargs: object) -> None:
        self.source = source
        self.kwargs = kwargs


def _fake_sentence_transformers() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        SentenceTransformer=_FakeSentenceTransformer,
        CrossEncoder=_FakeCrossEncoder,
    )


class OfflineEnvironmentTests(unittest.TestCase):
    def test_offline_environment_is_forced_on(self) -> None:
        env = {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}

        offline.configure_offline_environment(env)

        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(env["HF_HUB_DISABLE_TELEMETRY"], "1")
        self.assertEqual(env["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "1")
        self.assertEqual(env["DO_NOT_TRACK"], "1")

    def test_offline_environment_defaults_to_process_environment(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            offline.configure_offline_environment()
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
            self.assertEqual(os.environ["HF_HUB_DISABLE_TELEMETRY"], "1")


class LoadLocalModelTests(unittest.TestCase):
    def test_cached_model_id_uses_offline_constructor_options(self) -> None:
        imported = _fake_sentence_transformers()

        def import_with_offline_env() -> types.SimpleNamespace:
            self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
            self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")
            self.assertEqual(os.environ.get("HF_HUB_DISABLE_TELEMETRY"), "1")
            return imported

        with (
            mock.patch.object(
                offline,
                "_import_sentence_transformers",
                side_effect=import_with_offline_env,
            ),
            mock.patch.object(offline, "_select_device", return_value="cpu"),
        ):
            model = offline.load_local_model("BAAI/bge-small-en-v1.5")

        self.assertIsInstance(model, _FakeSentenceTransformer)
        self.assertEqual(model.source, "BAAI/bge-small-en-v1.5")
        self.assertEqual(
            model.kwargs,
            {
                "device": "cpu",
                "local_files_only": True,
                "trust_remote_code": False,
            },
        )

    def test_existing_model_directory_is_used_directly(self) -> None:
        imported = _fake_sentence_transformers()
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(
                offline,
                "_import_sentence_transformers",
                return_value=imported,
            ),
            mock.patch.object(offline, "_select_device", return_value="cpu"),
        ):
            model = offline.load_local_model(tmp)

        self.assertEqual(model.source, str(Path(tmp).resolve()))
        self.assertTrue(model.kwargs["local_files_only"])
        self.assertFalse(model.kwargs["trust_remote_code"])

    def test_missing_explicit_directory_has_a_clear_offline_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "not-there"
            with (
                mock.patch.object(offline, "_import_sentence_transformers") as load,
                mock.patch.object(offline, "_select_device", return_value="cpu"),
                self.assertRaises(offline.ModelUnavailableError) as raised,
            ):
                offline.load_local_model(missing)

        message = str(raised.exception)
        self.assertIn(str(missing), message)
        self.assertIn("does not exist", message)
        self.assertIn("never downloads", message)
        load.assert_not_called()

    def test_cached_model_failure_is_wrapped_clearly(self) -> None:
        class BrokenSentenceTransformer:
            def __init__(self, source: str, **kwargs: object) -> None:
                raise OSError("boom")

        imported = types.SimpleNamespace(SentenceTransformer=BrokenSentenceTransformer)
        with (
            mock.patch.object(
                offline,
                "_import_sentence_transformers",
                return_value=imported,
            ),
            mock.patch.object(offline, "_select_device", return_value="cpu"),
            self.assertRaises(offline.ModelUnavailableError) as raised,
        ):
            offline.load_local_model("org/model")

        message = str(raised.exception)
        self.assertIn("org/model", message)
        self.assertIn("local files only", message)
        self.assertIn("never downloads", message)
        self.assertIn("Hugging Face cache", message)
        self.assertIn("boom", message)
        self.assertIsInstance(raised.exception.__cause__, OSError)

    def test_missing_sentence_transformers_import_is_wrapped(self) -> None:
        error = ModuleNotFoundError("No module named 'sentence_transformers'")
        with (
            mock.patch.object(
                offline,
                "_import_sentence_transformers",
                side_effect=error,
            ),
            self.assertRaises(offline.ModelUnavailableError) as raised,
        ):
            offline.load_local_model("BAAI/bge-small-en-v1.5")

        message = str(raised.exception)
        self.assertIn("sentence-transformers", message)
        self.assertIn("install", message.lower())
        self.assertIn("pipx install 'seekr-search[torch]'", message)
        self.assertIn("pipx inject seekr-search sentence-transformers", message)
        self.assertIn("No module named", message)
        self.assertIn("never downloads", message)
        self.assertIs(raised.exception.__cause__, error)

    def test_cross_encoder_uses_the_same_adapter_contract(self) -> None:
        imported = _fake_sentence_transformers()
        with (
            mock.patch.object(
                offline,
                "_import_sentence_transformers",
                return_value=imported,
            ),
            mock.patch.object(offline, "_select_device", return_value="cuda"),
        ):
            model = offline.load_local_model("org/cross-encoder", kind="cross_encoder")

        self.assertIsInstance(model, _FakeCrossEncoder)
        self.assertEqual(model.source, "org/cross-encoder")
        self.assertEqual(
            model.kwargs,
            {
                "device": "cuda",
                "local_files_only": True,
                "trust_remote_code": False,
            },
        )

    def test_unknown_model_kind_raises(self) -> None:
        with self.assertRaises(ValueError):
            offline.load_local_model("org/model", kind="other")  # type: ignore[arg-type]

    def test_empty_model_name_raises(self) -> None:
        with self.assertRaises(offline.ModelUnavailableError):
            offline.load_local_model("   ")


@_NEEDS_NUMPY
class FakeEmbedderTests(unittest.TestCase):
    def setUp(self) -> None:
        import numpy as np

        self.np = np

    def test_fake_is_deterministic_and_normalised(self) -> None:
        embedder = FakeEmbedder()

        vectors = embedder.embed_documents(["red apple pie", "red apple pie"])
        query = embedder.embed_query("red apple pie")

        self.assertEqual(embedder.model_name, "fake")
        self.assertEqual(embedder.dim, 64)
        self.assertIsInstance(embedder, Embedder)
        self.assertEqual(vectors.shape, (2, 64))
        self.assertEqual(vectors.dtype, self.np.float32)
        self.assertEqual(query.shape, (64,))
        self.assertEqual(query.dtype, self.np.float32)
        self.assertTrue(self.np.array_equal(vectors[0], vectors[1]))
        self.assertTrue(self.np.array_equal(vectors[0], query))
        self.assertAlmostEqual(float(self.np.linalg.norm(vectors[0])), 1.0, places=5)
        self.assertAlmostEqual(float(self.np.linalg.norm(query)), 1.0, places=5)

    def test_fake_shared_words_are_closer(self) -> None:
        embedder = FakeEmbedder()
        a, b, c = embedder.embed_documents(
            [
                "red apple pie",
                "apple pie recipe",
                "quantum chromodynamics",
            ]
        )

        self.assertGreater(float(a @ b), float(a @ c))

    def test_fake_records_calls(self) -> None:
        embedder = FakeEmbedder()
        texts = ["a", "b"]

        embedder.embed_documents(texts)
        embedder.embed_documents(["c"])
        texts.append("mutated after the call")

        self.assertEqual(embedder.calls, [["a", "b"], ["c"]])

    def test_fake_handles_empty_batches_and_empty_text(self) -> None:
        embedder = FakeEmbedder(dim=8)

        empty = embedder.embed_documents([])
        vectors = embedder.embed_documents(["", "   "])

        self.assertEqual(empty.shape, (0, 8))
        self.assertEqual(empty.dtype, self.np.float32)
        self.assertTrue(self.np.array_equal(vectors[0], vectors[1]))
        self.assertAlmostEqual(float(self.np.linalg.norm(vectors[0])), 1.0, places=5)
        self.assertEqual(embedder.calls, [[], ["", "   "]])

    def test_fake_is_stable_across_processes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (
            "from seekr.embed import FakeEmbedder\n"
            "vector = FakeEmbedder(16).embed_documents(['red apple pie'])[0]\n"
            "print(vector.tobytes().hex())\n"
        )
        outputs: list[str] = []
        for seed in ("1", "2"):
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = seed
            env["PYTHONPATH"] = str(root / "src")
            result = subprocess.run(
                [sys.executable, "-B", "-c", script],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs.append(result.stdout.strip().splitlines()[-1])

        local = FakeEmbedder(16).embed_documents(["red apple pie"])[0]
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], local.tobytes().hex())


class SentenceTransformerEmbedderTests(unittest.TestCase):
    def test_constructor_is_lazy(self) -> None:
        with mock.patch.object(offline, "load_local_model") as load:
            embedder = SentenceTransformerEmbedder("org/model")

        load.assert_not_called()
        self.assertEqual(embedder.model_name, "org/model")

    def test_query_prefix_matches_global_constraints(self) -> None:
        self.assertEqual(
            QUERY_PREFIX,
            "Represent this sentence for searching relevant passages: ",
        )

    def test_dim_uses_the_shared_adapter(self) -> None:
        class Model:
            @staticmethod
            def get_sentence_embedding_dimension() -> int:
                return 4

        with mock.patch.object(offline, "load_local_model", return_value=Model()) as load:
            embedder = SentenceTransformerEmbedder("org/model", device="cpu")
            self.assertEqual(embedder.dim, 4)
            self.assertEqual(embedder.dim, 4)

        load.assert_called_once_with("org/model", kind="embedding", device="cpu")

    def test_embed_query_reports_model_load_failure_before_numpy_import(self) -> None:
        """A missing model dependency must not surface as a raw numpy ImportError."""
        unavailable = offline.ModelUnavailableError("model is not available locally")
        embedder = SentenceTransformerEmbedder("org/model")
        with (
            mock.patch.object(offline, "load_local_model", side_effect=unavailable),
            mock.patch.dict(sys.modules, {"numpy": None}),
            self.assertRaises(offline.ModelUnavailableError) as raised,
        ):
            embedder.embed_query("query")

        self.assertIs(raised.exception, unavailable)

    @_NEEDS_NUMPY
    def test_documents_use_batch_size_32_and_queries_get_the_prefix(self) -> None:
        import numpy as np

        class Model:
            def __init__(self) -> None:
                self.calls: list[tuple[list[str], dict[str, object]]] = []

            @staticmethod
            def get_sentence_embedding_dimension() -> int:
                return 4

            def encode(self, texts: list[str], **kwargs: object):
                self.calls.append((list(texts), kwargs))
                raw = np.arange(1, len(texts) * 4 + 1, dtype=np.float32).reshape(
                    len(texts), 4
                )
                return raw

        model = Model()
        with mock.patch.object(offline, "load_local_model", return_value=model):
            embedder = SentenceTransformerEmbedder("org/model")
            documents = embedder.embed_documents(["alpha", "beta"])
            query = embedder.embed_query("alpha")

        self.assertEqual(documents.shape, (2, 4))
        self.assertEqual(documents.dtype, np.float32)
        self.assertEqual(query.shape, (4,))
        self.assertTrue(np.allclose(np.linalg.norm(documents, axis=1), 1.0))
        self.assertAlmostEqual(float(np.linalg.norm(query)), 1.0, places=5)

        self.assertEqual(
            [call[0] for call in model.calls],
            [["alpha", "beta"], [QUERY_PREFIX + "alpha"]],
        )
        for _, kwargs in model.calls:
            self.assertEqual(kwargs["batch_size"], 32)
            self.assertTrue(kwargs["convert_to_numpy"])
            self.assertTrue(kwargs["normalize_embeddings"])
            self.assertFalse(kwargs["show_progress_bar"])

    @_NEEDS_NUMPY
    def test_empty_documents_return_an_empty_matrix_without_encoding(self) -> None:
        import numpy as np

        class Model:
            def __init__(self) -> None:
                self.encoded: list[list[str]] = []

            @staticmethod
            def get_sentence_embedding_dimension() -> int:
                return 4

            def encode(self, texts: list[str], **kwargs: object):
                self.encoded.append(list(texts))
                return np.zeros((len(texts), 4), dtype=np.float32)

        model = Model()
        with mock.patch.object(offline, "load_local_model", return_value=model):
            embedder = SentenceTransformerEmbedder("org/model")
            vectors = embedder.embed_documents([])

        self.assertEqual(vectors.shape, (0, 4))
        self.assertEqual(vectors.dtype, np.float32)
        self.assertEqual(model.encoded, [])

    @_NEEDS_NUMPY
    def test_wrong_model_output_shape_raises(self) -> None:
        import numpy as np

        class Model:
            @staticmethod
            def get_sentence_embedding_dimension() -> int:
                return 4

            @staticmethod
            def encode(texts: list[str], **kwargs: object):
                return np.zeros((len(texts), 3), dtype=np.float32)

        with mock.patch.object(offline, "load_local_model", return_value=Model()):
            embedder = SentenceTransformerEmbedder("org/model")
            with self.assertRaises(ValueError):
                embedder.embed_documents(["alpha"])

    def test_concurrent_first_use_loads_model_once(self) -> None:
        load_count = 0
        count_lock = threading.Lock()

        class Model:
            @staticmethod
            def get_sentence_embedding_dimension() -> int:
                return 4

        def loader(*args: object, **kwargs: object) -> Model:
            nonlocal load_count
            with count_lock:
                load_count += 1
            time.sleep(0.05)
            return Model()

        embedder = SentenceTransformerEmbedder("org/model")
        barrier = threading.Barrier(8)

        def read_dim(_: int) -> int:
            barrier.wait(timeout=5)
            return embedder.dim

        with (
            mock.patch.object(offline, "load_local_model", side_effect=loader) as load,
            ThreadPoolExecutor(max_workers=8) as pool,
        ):
            dimensions = list(pool.map(read_dim, range(8)))

        self.assertEqual(dimensions, [4] * 8)
        self.assertEqual(load_count, 1)
        load.assert_called_once_with("org/model", kind="embedding", device=None)

    def test_model_inference_is_serialized(self) -> None:
        class Model:
            def __init__(self) -> None:
                self.guard = threading.Lock()
                self.active = 0
                self.max_active = 0
                self.calls = 0

            def encode(self, texts: list[str], **kwargs: object) -> object:
                with self.guard:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.05)
                with self.guard:
                    self.active -= 1
                    self.calls += 1
                return object()

        model = Model()
        embedder = SentenceTransformerEmbedder("org/model")

        def infer(_: int) -> object:
            return embedder._encode_with_model(model, ["x"])

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(infer, range(8)))

        self.assertEqual(model.max_active, 1)
        self.assertEqual(model.calls, 8)

    @_NEEDS_NUMPY
    def test_public_document_inference_is_serialized(self) -> None:
        import numpy as np

        class Model:
            def __init__(self) -> None:
                self.guard = threading.Lock()
                self.active = 0
                self.max_active = 0

            @staticmethod
            def get_sentence_embedding_dimension() -> int:
                return 4

            def encode(self, texts: list[str], **kwargs: object):
                with self.guard:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.05)
                with self.guard:
                    self.active -= 1
                return np.zeros((len(texts), 4), dtype=np.float32)

        model = Model()
        with mock.patch.object(offline, "load_local_model", return_value=model):
            embedder = SentenceTransformerEmbedder("org/model")

            def embed_one(index: int):
                if index % 2:
                    return embedder.embed_query("query")
                return embedder.embed_documents(["document"])

            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(embed_one, range(8)))

        self.assertEqual(model.max_active, 1)

    def test_build_embedder_returns_a_lazy_sentence_transformer_embedder(self) -> None:
        with mock.patch.object(offline, "load_local_model") as load:
            embedder = build_embedder(Config(embed_model="org/model", embed_backend="torch"))

        self.assertIsInstance(embedder, SentenceTransformerEmbedder)
        self.assertEqual(embedder.model_name, "org/model")
        load.assert_not_called()

    def test_build_embedder_dispatch(self) -> None:
        from seekr.onnx_backend import OnnxEmbedder

        with (
            mock.patch.object(offline, "load_local_model") as load,
            mock.patch("seekr.onnx_backend.resolve_model_files") as resolve,
        ):
            onnx = build_embedder(Config(embed_model="org/model", embed_backend="onnx"))
            torch = build_embedder(Config(embed_model="org/model", embed_backend="torch"))

        self.assertIsInstance(onnx, OnnxEmbedder)
        self.assertEqual(onnx.model_name, "org/model")
        self.assertIsInstance(torch, SentenceTransformerEmbedder)
        load.assert_not_called()
        resolve.assert_not_called()

    def test_torch_backend_without_extra_explains_install(self) -> None:
        error = ImportError("No module named 'sentence_transformers'")
        with (
            mock.patch.object(offline, "_import_sentence_transformers", side_effect=error),
            self.assertRaisesRegex(offline.ModelUnavailableError, r"seekr-search\[torch\]"),
        ):
            _ = build_embedder(Config(embed_backend="torch")).dim

    def test_import_does_not_eagerly_import_heavy_modules(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (
            "import sys\n"
            "import seekr.embed\n"
            "assert 'sentence_transformers' not in sys.modules\n"
            "assert 'numpy' not in sys.modules\n"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root / "src")
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
