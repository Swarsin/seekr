"""Tests for model and code-grammar provisioning and ``seekr models`` CLI."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from seekr.config import Config

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
_HAS_TYPER = importlib.util.find_spec("typer") is not None


class ProvisionUnitTests(unittest.TestCase):
    def test_assets_for_default_config(self) -> None:
        from seekr import provision

        cfg = Config(embed_backend="torch")
        self.assertEqual(
            provision.TORCH_PATTERNS,
            ("*.json", "*.txt", "model.safetensors", "1_Pooling/*"),
        )

        embed_only = provision.assets_for(cfg, rerank=False)
        self.assertEqual(
            embed_only,
            [
                provision.ModelAsset(
                    repo_id=cfg.embed_model,
                    allow_patterns=provision.TORCH_PATTERNS,
                    purpose="embedding",
                )
            ],
        )

        with_rerank = provision.assets_for(cfg, rerank=True)
        self.assertEqual(
            with_rerank,
            [
                provision.ModelAsset(
                    repo_id=cfg.embed_model,
                    allow_patterns=provision.TORCH_PATTERNS,
                    purpose="embedding",
                ),
                provision.ModelAsset(
                    repo_id=cfg.rerank_model,
                    allow_patterns=provision.TORCH_PATTERNS,
                    purpose="rerank",
                ),
            ],
        )

    def test_is_cached_checks_local_dir_and_hf_cache_offline(self) -> None:
        from seekr import provision

        with tempfile.TemporaryDirectory() as tmp:
            local_dir = Path(tmp) / "local-model"
            local_dir.mkdir()
            asset = provision.ModelAsset(
                repo_id=str(local_dir),
                allow_patterns=provision.TORCH_PATTERNS,
                purpose="embedding",
            )
            self.assertFalse(provision.is_cached(asset))

            (local_dir / "config.json").write_text("{}", encoding="utf-8")
            self.assertTrue(provision.is_cached(asset))

        hub_asset = provision.ModelAsset(
            repo_id="BAAI/bge-small-en-v1.5",
            allow_patterns=provision.TORCH_PATTERNS,
            purpose="embedding",
        )
        with mock.patch(
            "huggingface_hub.try_to_load_from_cache",
            side_effect=lambda repo, name: f"/cache/snapshots/abc/{name}",
        ) as cached_lookup:
            self.assertTrue(provision.is_cached(hub_asset))
        self.assertEqual(
            [c.args for c in cached_lookup.call_args_list],
            [
                ("BAAI/bge-small-en-v1.5", "config.json"),
                ("BAAI/bge-small-en-v1.5", "model.safetensors"),
            ],
        )

        # ONNX-only cache (no safetensors) does not satisfy the torch backend.
        with mock.patch(
            "huggingface_hub.try_to_load_from_cache",
            side_effect=lambda repo, name: None if name == "model.safetensors" else name,
        ):
            self.assertFalse(provision.is_cached(hub_asset))

        with mock.patch("huggingface_hub.try_to_load_from_cache", return_value=None):
            self.assertFalse(provision.is_cached(hub_asset))

        with mock.patch(
            "huggingface_hub.try_to_load_from_cache",
            side_effect=ValueError("invalid repo id"),
        ):
            self.assertFalse(provision.is_cached(hub_asset))

    def test_onnx_assets_use_onnx_patterns(self) -> None:
        from seekr import provision

        self.assertEqual(
            provision.ONNX_PATTERNS,
            (
                "onnx/model.onnx",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "config.json",
                "1_Pooling/config.json",
            ),
        )
        cfg = Config(embed_backend="onnx")
        assets = provision.assets_for(cfg, rerank=True)
        self.assertEqual([a.repo_id for a in assets], [cfg.embed_model, cfg.rerank_model])
        for asset in assets:
            self.assertEqual(asset.allow_patterns, provision.ONNX_PATTERNS)
            self.assertEqual(asset.backend, "onnx")
        torch_assets = provision.assets_for(Config(embed_backend="torch"), rerank=True)
        self.assertEqual({a.backend for a in torch_assets}, {"torch"})

    def test_is_cached_onnx_requires_model_onnx(self) -> None:
        from seekr import provision

        asset = provision.ModelAsset(
            repo_id="BAAI/bge-small-en-v1.5",
            allow_patterns=provision.ONNX_PATTERNS,
            purpose="embedding",
            backend="onnx",
        )
        # Torch-only cache: config.json present, no onnx/model.onnx.
        with mock.patch(
            "huggingface_hub.try_to_load_from_cache",
            side_effect=lambda repo, name: f"/c/{name}" if name == "config.json" else None,
        ):
            self.assertFalse(provision.is_cached(asset))
        with mock.patch(
            "huggingface_hub.try_to_load_from_cache",
            side_effect=lambda repo, name: f"/c/{name}",
        ) as lookup:
            self.assertTrue(provision.is_cached(asset))
        self.assertIn(("BAAI/bge-small-en-v1.5", "onnx/model.onnx"), [c.args for c in lookup.call_args_list])

        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp)
            (local / "config.json").write_text("{}", encoding="utf-8")
            local_asset = provision.ModelAsset(
                repo_id=str(local),
                allow_patterns=provision.ONNX_PATTERNS,
                purpose="embedding",
                backend="onnx",
            )
            self.assertFalse(provision.is_cached(local_asset))
            (local / "onnx").mkdir()
            (local / "onnx" / "model.onnx").write_bytes(b"x")
            self.assertTrue(provision.is_cached(local_asset))

    def test_estimate_bytes_filters_by_allow_patterns_and_handles_errors(self) -> None:
        from seekr import provision

        asset = provision.ModelAsset(
            repo_id="BAAI/bge-small-en-v1.5",
            allow_patterns=provision.TORCH_PATTERNS,
            purpose="embedding",
        )
        fake_info = types.SimpleNamespace(
            siblings=[
                types.SimpleNamespace(rfilename="config.json", size=100),
                types.SimpleNamespace(rfilename="tokenizer.txt", size=200),
                types.SimpleNamespace(rfilename="model.safetensors", size=133_000_000),
                types.SimpleNamespace(rfilename="1_Pooling/config.json", size=300),
                types.SimpleNamespace(rfilename="pytorch_model.bin", size=999_999_999),
                types.SimpleNamespace(rfilename="onnx/model.onnx", size=888_888_888),
            ]
        )
        fake_api = mock.MagicMock()
        fake_api.model_info.return_value = fake_info

        with (
            mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", False),
            mock.patch("huggingface_hub.HfApi", return_value=fake_api),
        ):
            total = provision.estimate_bytes(asset)

        self.assertEqual(total, 133_000_600)
        fake_api.model_info.assert_called_once_with(
            "BAAI/bge-small-en-v1.5", files_metadata=True
        )

        failing_api = mock.MagicMock()
        failing_api.model_info.side_effect = OSError("network unavailable")
        with (
            mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", False),
            mock.patch("huggingface_hub.HfApi", return_value=failing_api),
        ):
            self.assertIsNone(provision.estimate_bytes(asset))

    def test_download_refuses_in_offline_process(self) -> None:
        from seekr import provision

        asset = provision.ModelAsset(
            repo_id="BAAI/bge-small-en-v1.5",
            allow_patterns=provision.TORCH_PATTERNS,
            purpose="embedding",
        )
        with (
            mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", True),
            mock.patch("huggingface_hub.snapshot_download") as snap,
            self.assertRaises(provision.ProvisionError) as raised,
        ):
            provision.download(asset)

        self.assertEqual(
            str(raised.exception),
            "downloads are disabled in this process (offline mode); run `seekr models pull` in a new shell",
        )
        snap.assert_not_called()

        # Verify in a real child process that configure_offline_environment()
        # blocks provision.download() before huggingface_hub is imported.
        script = (
            "from seekr.offline import configure_offline_environment\n"
            "from seekr import provision\n"
            "configure_offline_environment()\n"
            "asset = provision.ModelAsset('BAAI/bge-small-en-v1.5', provision.TORCH_PATTERNS, 'embedding')\n"
            "try:\n"
            "    provision.download(asset)\n"
            "except provision.ProvisionError as exc:\n"
            "    print(exc)\n"
            "    raise SystemExit(0)\n"
            "raise SystemExit(1)\n"
        )
        env = {k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"}
        env["PYTHONPATH"] = os.pathsep.join([str(SRC), env.get("PYTHONPATH", "")]).rstrip(
            os.pathsep
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        self.assertIn("downloads are disabled in this process (offline mode)", proc.stdout)

    def test_download_calls_snapshot_download_and_wraps_errors(self) -> None:
        from seekr import provision

        asset = provision.ModelAsset(
            repo_id="BAAI/bge-small-en-v1.5",
            allow_patterns=provision.TORCH_PATTERNS,
            purpose="embedding",
        )
        with (
            mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", False),
            mock.patch(
                "huggingface_hub.snapshot_download",
                return_value="/tmp/hf/models--BAAI--bge-small-en-v1.5/snapshots/1",
            ) as snap,
        ):
            result = provision.download(asset)

        self.assertEqual(
            result,
            Path("/tmp/hf/models--BAAI--bge-small-en-v1.5/snapshots/1"),
        )
        snap.assert_called_once_with(
            "BAAI/bge-small-en-v1.5",
            allow_patterns=list(provision.TORCH_PATTERNS),
        )

        with (
            mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", False),
            mock.patch(
                "huggingface_hub.snapshot_download",
                side_effect=OSError("connection refused"),
            ),
            self.assertRaises(provision.ProvisionError),
        ):
            provision.download(asset)

    def test_grammar_languages(self) -> None:
        from seekr import extract, provision

        langs = provision.grammar_languages()
        expected = sorted(set(extract._LANGUAGE_BY_SUFFIX.values()))
        self.assertEqual(langs, expected)
        self.assertEqual(
            langs,
            ["c", "cpp", "go", "java", "javascript", "python", "rust", "tsx", "typescript"],
        )

        with mock.patch(
            "seekr.chunking.ast_support",
            side_effect=lambda lang: (lang in {"python", "go"}, "detail"),
        ):
            missing = provision.missing_grammars()
        self.assertEqual(
            missing,
            ["c", "cpp", "java", "javascript", "rust", "tsx", "typescript"],
        )

        with mock.patch("tree_sitter_language_pack.download", return_value=2) as dl:
            provision.download_grammars(["python", "rust"])
        dl.assert_called_once_with(["python", "rust"])

        with mock.patch("tree_sitter_language_pack.download") as dl_empty:
            provision.download_grammars([])
        dl_empty.assert_not_called()


@unittest.skipUnless(_HAS_TYPER, "typer is not installed")
class ModelsCliTests(unittest.TestCase):
    @staticmethod
    def _runner():
        from typer.testing import CliRunner

        return CliRunner()

    def test_pull_prompts_and_aborts_on_no(self) -> None:
        from seekr import provision
        from seekr.cli import app

        cfg = Config()
        with (
            mock.patch("seekr.config.load_config", return_value=cfg),
            mock.patch.object(provision, "is_cached", return_value=False),
            mock.patch.object(provision, "estimate_bytes", return_value=133_000_000),
            mock.patch.object(provision, "download") as dl,
        ):
            result = self._runner().invoke(app, ["models", "pull"], input="n\n")

        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("BAAI/bge-small-en-v1.5", result.output)
        self.assertIn("~133 MB", result.output)
        self.assertIn("Download 1 item(s), about 133 MB? [y/N]", result.output)
        dl.assert_not_called()

    def test_pull_yes_downloads_without_prompt(self) -> None:
        from seekr import provision
        from seekr.cli import app

        cfg = Config()
        with (
            mock.patch("seekr.config.load_config", return_value=cfg),
            mock.patch.object(provision, "is_cached", return_value=False),
            mock.patch.object(provision, "estimate_bytes", return_value=133_000_000),
            mock.patch.object(provision, "download", return_value=Path("/tmp/model")) as dl,
            mock.patch.object(provision, "download_grammars") as dl_grammars,
        ):
            result = self._runner().invoke(app, ["models", "pull", "--yes"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("[y/N]", result.output)
        dl.assert_called_once_with(
            provision.ModelAsset(
                repo_id=cfg.embed_model,
                allow_patterns=provision.ONNX_PATTERNS,
                purpose="embedding",
                backend="onnx",
            )
        )
        dl_grammars.assert_not_called()

        with (
            mock.patch("seekr.config.load_config", return_value=cfg),
            mock.patch.object(provision, "is_cached", return_value=False),
            mock.patch.object(provision, "estimate_bytes", return_value=90_000_000),
            mock.patch.object(provision, "missing_grammars", return_value=["python", "rust"]),
            mock.patch.object(provision, "download", return_value=Path("/tmp/model")) as dl_all,
            mock.patch.object(provision, "download_grammars") as dl_all_grammars,
        ):
            all_result = self._runner().invoke(app, ["models", "pull", "--all", "--yes"])

        self.assertEqual(all_result.exit_code, 0, all_result.output)
        self.assertEqual(dl_all.call_count, 2)
        dl_all_grammars.assert_called_once_with(["python", "rust"])

        # When embedding is already cached, --grammars --yes pulls only missing grammars.
        with (
            mock.patch("seekr.config.load_config", return_value=cfg),
            mock.patch.object(provision, "is_cached", return_value=True),
            mock.patch.object(provision, "estimate_bytes") as est_skip,
            mock.patch.object(provision, "missing_grammars", return_value=["python"]),
            mock.patch.object(provision, "download") as dl_skip,
            mock.patch.object(provision, "download_grammars") as dl_only_grammars,
        ):
            gram_result = self._runner().invoke(app, ["models", "pull", "--grammars", "--yes"])

        self.assertEqual(gram_result.exit_code, 0, gram_result.output)
        est_skip.assert_not_called()
        dl_skip.assert_not_called()
        dl_only_grammars.assert_called_once_with(["python"])

    def test_models_status(self) -> None:
        from seekr import provision
        from seekr.cli import app

        cfg = Config()

        def fake_is_cached(asset):
            return asset.purpose == "embedding"

        with (
            mock.patch("seekr.config.load_config", return_value=cfg),
            mock.patch.object(provision, "is_cached", side_effect=fake_is_cached),
            mock.patch.object(provision, "missing_grammars", return_value=["go", "rust"]),
        ):
            result = self._runner().invoke(app, ["models", "status"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn(cfg.embed_model, result.output)
        self.assertIn(cfg.rerank_model, result.output)
        self.assertIn("cached", result.output)
        self.assertIn("missing", result.output)
        self.assertIn("python: cached", result.output)
        self.assertIn("go: missing", result.output)
        self.assertIn("rust: missing", result.output)

    def test_python_dash_m_seekr(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"}
        env["PYTHONPATH"] = os.pathsep.join([str(SRC), env.get("PYTHONPATH", "")]).rstrip(
            os.pathsep
        )
        proc = subprocess.run(
            [sys.executable, "-m", "seekr", "--version"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "seekr 0.2.0")

        doctor_proc = subprocess.run(
            [sys.executable, "-m", "seekr", "doctor", "--env-only"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(doctor_proc.returncode, 0, doctor_proc.stderr)
        self.assertIn("FTS5", doctor_proc.stdout)

        # CLI startup and `models pull` must not force HF_HUB_OFFLINE in os.environ.
        check_offline_script = (
            "import os, sys\n"
            "import seekr.cli\n"
            "assert 'huggingface_hub' not in sys.modules\n"
            "assert 'tree_sitter_language_pack' not in sys.modules\n"
            "assert 'HF_HUB_OFFLINE' not in os.environ\n"
        )
        offline_proc = subprocess.run(
            [sys.executable, "-c", check_offline_script],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(offline_proc.returncode, 0, offline_proc.stderr)

    def test_pull_edge_cases_unknown_size_cached_rerank_and_errors(self) -> None:
        from seekr import provision
        from seekr.cli import app

        cfg = Config()
        # Unknown size prints "size unknown" and confirms without MB count.
        with (
            mock.patch("seekr.config.load_config", return_value=cfg),
            mock.patch.object(provision, "is_cached", return_value=False),
            mock.patch.object(provision, "estimate_bytes", return_value=None),
            mock.patch.object(provision, "download", return_value=Path("/tmp/model")) as dl,
        ):
            res_unknown = self._runner().invoke(app, ["models", "pull"], input="y\n")
        self.assertEqual(res_unknown.exit_code, 0, res_unknown.output)
        self.assertIn("size unknown", res_unknown.output)
        self.assertIn("Download 1 item(s)? [y/N]", res_unknown.output)
        dl.assert_called_once()

        # Already cached exits 0 cleanly without downloading.
        with (
            mock.patch("seekr.config.load_config", return_value=cfg),
            mock.patch.object(provision, "is_cached", return_value=True),
            mock.patch.object(provision, "download") as dl_none,
        ):
            res_cached = self._runner().invoke(app, ["models", "pull"])
        self.assertEqual(res_cached.exit_code, 0, res_cached.output)
        self.assertIn("already cached", res_cached.output)
        dl_none.assert_not_called()

        # --rerank --yes downloads embedding + rerank only.
        with (
            mock.patch("seekr.config.load_config", return_value=cfg),
            mock.patch.object(provision, "is_cached", return_value=False),
            mock.patch.object(provision, "estimate_bytes", return_value=90_000_000),
            mock.patch.object(provision, "download", return_value=Path("/tmp/m")) as dl_rerank,
            mock.patch.object(provision, "download_grammars") as dl_gram,
        ):
            res_rerank = self._runner().invoke(app, ["models", "pull", "--rerank", "--yes"])
        self.assertEqual(res_rerank.exit_code, 0, res_rerank.output)
        self.assertEqual(
            [c.args[0].purpose for c in dl_rerank.call_args_list],
            ["embedding", "rerank"],
        )
        dl_gram.assert_not_called()

        # ProvisionError exits 1 with message.
        with (
            mock.patch("seekr.config.load_config", return_value=cfg),
            mock.patch.object(provision, "is_cached", return_value=False),
            mock.patch.object(provision, "estimate_bytes", return_value=None),
            mock.patch.object(
                provision,
                "download",
                side_effect=provision.ProvisionError("offline"),
            ),
        ):
            res_err = self._runner().invoke(app, ["models", "pull", "--yes"])
        self.assertEqual(res_err.exit_code, 1, res_err.output)
        self.assertIn("offline", res_err.output)


if __name__ == "__main__":
    unittest.main()
