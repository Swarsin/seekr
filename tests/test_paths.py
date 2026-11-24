"""Platform directory resolution."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

from seekr import config, paths


class PathsTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith("linux"), "linux paths")
    def test_linux_paths_unchanged(self) -> None:
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": "/x/cfg", "XDG_DATA_HOME": "/x/data"}):
            self.assertEqual(paths.config_dir(), Path("/x/cfg/seekr"))
            self.assertEqual(config.default_db_path(), Path("/x/data/seekr/index.db"))
            self.assertEqual(config.default_data_dir(), Path("/x/data/seekr"))
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("pathlib.Path.home", return_value=Path("/h")),
        ):
            self.assertEqual(paths.config_dir(), Path("/h/.config/seekr"))
            self.assertEqual(paths.data_dir(), Path("/h/.local/share/seekr"))

    def test_seekr_config_takes_precedence(self) -> None:
        with mock.patch.dict(os.environ, {"SEEKR_CONFIG": "/etc/s.toml"}):
            self.assertEqual(config.default_config_path(), Path("/etc/s.toml"))

    @unittest.skipUnless(sys.platform == "win32", "windows paths")
    def test_windows_paths_under_localappdata(self) -> None:
        self.assertIn("AppData\\Local", str(paths.data_dir()))

    def test_service_file_in_state_dir(self) -> None:
        with mock.patch.dict(os.environ, {"XDG_STATE_HOME": "/x/state"}):
            self.assertEqual(
                paths.service_file(platform="linux"), Path("/x/state/seekr/service.json")
            )
            self.assertEqual(paths.state_dir(platform="linux"), Path("/x/state/seekr"))

    def test_windows_dirs_are_local_not_roaming(self) -> None:
        def fake_folder(csidl: str) -> str:
            return {
                "CSIDL_LOCAL_APPDATA": "C:\\Users\\u\\AppData\\Local",
                "CSIDL_APPDATA": "C:\\Users\\u\\AppData\\Roaming",
            }[csidl]

        with mock.patch("platformdirs.windows.get_win_folder", fake_folder):
            for result in (
                paths.config_dir(platform="win32"),
                paths.data_dir(platform="win32"),
                paths.state_dir(platform="win32"),
                paths.cache_dir(platform="win32"),
                paths.service_file(platform="win32"),
            ):
                text = str(result).replace("/", "\\")
                self.assertIn("AppData\\Local\\seekr", text)
                self.assertNotIn("Roaming", text)
            self.assertEqual(paths.service_file(platform="win32").name, "service.json")

    def test_darwin_dirs_under_library(self) -> None:
        with (
            mock.patch("pathlib.Path.home", return_value=Path("/Users/u")),
            mock.patch("os.path.expanduser", lambda p: p.replace("~", "/Users/u", 1)),
        ):
            self.assertEqual(
                paths.data_dir(platform="darwin"),
                Path("/Users/u/Library/Application Support/seekr"),
            )
            self.assertEqual(
                paths.service_file(platform="darwin").parent,
                paths.state_dir(platform="darwin"),
            )
            self.assertIn("Library", str(paths.state_dir(platform="darwin")))


if __name__ == "__main__":
    unittest.main()
