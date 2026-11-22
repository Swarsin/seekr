"""Tests for configuration loading, validation and shared models."""

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

from seekr.config import Config, ConfigError, load_config
from seekr.models import ChunkSpec, ExtractedUnit, Hit, text_sha


class ConfigLoadingTests(unittest.TestCase):
    def _write(self, directory: Path, contents: str) -> Path:
        path = directory / "config.toml"
        path.write_text(contents, encoding="utf-8")
        return path

    def test_load_config_missing_file_returns_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(Path(tmp) / "nope.toml")

        self.assertEqual(cfg.roots, ())
        self.assertEqual(cfg.embed_model, "BAAI/bge-small-en-v1.5")

    def test_defaults_match_global_constraints(self) -> None:
        cfg = Config()

        self.assertEqual(cfg.roots, ())
        self.assertEqual(cfg.ignore_globs, (".git", "node_modules", "__pycache__", ".venv", "venv"))
        self.assertEqual(cfg.max_file_bytes, 20_000_000)
        self.assertEqual(cfg.embed_model, "BAAI/bge-small-en-v1.5")
        self.assertEqual(cfg.rerank_model, "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.assertEqual(cfg.vector_quant, "int8")
        self.assertEqual(cfg.chunk_target_words, 300)
        self.assertEqual(cfg.chunk_max_words, 450)
        self.assertEqual(cfg.chunk_overlap_words, 50)

    def test_load_config_overrides_and_tilde(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = self._write(
                Path(tmp),
                'roots = ["~/notes"]\nchunk_target_words = 200\n',
            )
            with mock.patch.dict(os.environ, {"HOME": tmp}, clear=False):
                cfg = load_config(cfg_path)
                self.assertEqual(cfg.roots, (Path(tmp) / "notes",))

        self.assertEqual(cfg.chunk_target_words, 200)
        self.assertEqual(cfg.chunk_max_words, 450)

    def test_vector_quant_override_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = self._write(Path(tmp), 'vector_quant = "float32"\n')
            cfg = load_config(cfg_path)
            self.assertEqual(cfg.vector_quant, "float32")

            bad_path = self._write(Path(tmp), 'vector_quant = "binary"\n')
            with self.assertRaises(ConfigError):
                load_config(bad_path)

    def test_db_path_expands_tilde(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = self._write(Path(tmp), 'db_path = "~/data/index.db"\n')
            with mock.patch.dict(os.environ, {"HOME": tmp}, clear=False):
                cfg = load_config(cfg_path)
                self.assertEqual(cfg.db_path, Path(tmp) / "data" / "index.db")

    def test_partial_override_keeps_other_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = self._write(
                Path(tmp),
                'roots = []\nembed_model = "custom/model"\n',
            )
            cfg = load_config(cfg_path)

        self.assertEqual(cfg.roots, ())
        self.assertEqual(cfg.embed_model, "custom/model")
        self.assertEqual(cfg.rerank_model, "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.assertEqual(cfg.chunk_target_words, 300)

    def test_load_config_unknown_key_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = self._write(Path(tmp), "bogus = 1\n")
            with self.assertRaises(ConfigError) as raised:
                load_config(cfg_path)

        self.assertIn("bogus", str(raised.exception))

    def test_invalid_toml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = self._write(Path(tmp), "roots = [\n")
            with self.assertRaises(ConfigError):
                load_config(cfg_path)

    def test_default_db_path_uses_xdg_data_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {"XDG_DATA_HOME": tmp, "HOME": tmp},
                clear=False,
            ):
                cfg = Config()
            self.assertEqual(cfg.db_path, Path(tmp) / "seekr" / "index.db")

    def test_default_db_path_falls_back_to_local_share(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"HOME": tmp}, clear=True):
                cfg = Config()
            self.assertEqual(
                cfg.db_path,
                Path(tmp) / ".local" / "share" / "seekr" / "index.db",
            )

    def test_default_path_uses_xdg_config_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "seekr"
            config_dir.mkdir()
            self._write(config_dir, 'embed_model = "xdg/model"\n')
            with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": tmp}, clear=False):
                cfg = load_config()

        self.assertEqual(cfg.embed_model, "xdg/model")

    def test_wrong_value_types_raise(self) -> None:
        cases = [
            {"roots": [Path(".")]},
            {"db_path": "index.db"},
            {"ignore_globs": [".git"]},
            {"max_file_bytes": "big"},
            {"embed_model": ""},
            {"vector_quant": 8},
            {"chunk_target_words": "300"},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigError):
                Config(**kwargs)

    def test_invalid_ranges_raise(self) -> None:
        cases = [
            {"max_file_bytes": 0},
            {"max_file_bytes": -1},
            {"chunk_target_words": 0},
            {"chunk_max_words": 299},
            {"chunk_overlap_words": -1},
            {"chunk_overlap_words": 300},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigError):
                Config(**kwargs)

    def test_config_is_frozen(self) -> None:
        cfg = Config()
        with self.assertRaises(FrozenInstanceError):
            cfg.embed_model = "other"  # type: ignore[misc]

    def test_week_pattern_and_course_structure_load(self) -> None:
        defaults = Config()
        self.assertEqual(defaults.week_pattern, r"^(?:week|wk|w)[\s_-]*0*(\d{1,2})\b")
        self.assertIs(defaults.course_structure, True)

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = self._write(
                Path(tmp),
                'week_pattern = "^semana(\\\\d+)$"\ncourse_structure = false\n',
            )
            cfg = load_config(cfg_path)

        self.assertEqual(cfg.week_pattern, r"^semana(\d+)$")
        self.assertIs(cfg.course_structure, False)

    def test_invalid_week_pattern_is_config_error(self) -> None:
        bad_configs = (
            'week_pattern = "("\n',
            'week_pattern = "week\\\\d+"\n',
            "week_pattern = 1\n",
        )
        with tempfile.TemporaryDirectory() as tmp:
            for bad in bad_configs:
                cfg_path = self._write(Path(tmp), bad)
                with self.subTest(bad=bad):
                    with self.assertRaises(ConfigError) as raised:
                        load_config(cfg_path)
                    self.assertIn("week_pattern", str(raised.exception))

        with self.assertRaises(ConfigError) as raised:
            Config(week_pattern="(")
        self.assertIn("week_pattern", str(raised.exception))

    def test_course_structure_must_be_bool(self) -> None:
        for bad in (1, 0, "false", "true", None):
            with self.subTest(bad=bad):
                with self.assertRaises(ConfigError) as raised:
                    Config(course_structure=bad)  # type: ignore[arg-type]
                self.assertIn("course_structure", str(raised.exception))

        with tempfile.TemporaryDirectory() as tmp:
            for toml in ('course_structure = "false"\n', "course_structure = 1\n"):
                cfg_path = self._write(Path(tmp), toml)
                with self.subTest(toml=toml), self.assertRaises(ConfigError):
                    load_config(cfg_path)


class ModelTests(unittest.TestCase):
    def test_text_sha_ignores_case_and_whitespace(self) -> None:
        self.assertEqual(text_sha("Hello  World\n"), text_sha("hello world"))
        self.assertNotEqual(text_sha("hello world"), text_sha("hello there"))
        self.assertEqual(len(text_sha("anything")), 64)

    def test_models_keep_the_expected_fields(self) -> None:
        unit = ExtractedUnit(text="body", locator="p.1", loc_start=1.0)
        chunk = ChunkSpec(text="body", locator="L1-2", loc_start=1.0)
        hit = Hit(
            chunk_id=7,
            path="/notes/a.md",
            locator="\u00a7Intro",
            loc_start=0.0,
            snippet="body",
            score=0.5,
            kind="markdown",
        )

        self.assertEqual(unit.text, "body")
        self.assertEqual(chunk.locator, "L1-2")
        self.assertEqual(hit.chunk_id, 7)
        self.assertEqual(hit.kind, "markdown")


if __name__ == "__main__":
    unittest.main()
