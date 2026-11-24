"""Tests for WSL and slow-filesystem detection."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from seekr.platform_info import (
    SLOW_FSTYPES,
    Mount,
    is_slow_path,
    is_wsl,
    mount_for,
    parse_mounts,
    read_mounts,
    split_roots,
    windows_path,
    wslpath_windows,
)

MOUNTS = "C:\\134 /mnt/c 9p rw,dirsync,aname=drvfs 0 0\n/dev/sdc / ext4 rw 0 0\n"


class PlatformInfoTests(unittest.TestCase):
    def test_is_wsl(self) -> None:
        self.assertTrue(is_wsl("5.15.133.1-microsoft-standard-WSL2"))
        self.assertTrue(is_wsl("4.4.0-19041-Microsoft"))
        self.assertFalse(is_wsl("6.8.0-45-generic"))
        self.assertFalse(is_wsl(""))

        with mock.patch.object(
            Path,
            "read_text",
            return_value="6.6.36.3-microsoft-standard-WSL2\n",
        ):
            self.assertTrue(is_wsl())

        with mock.patch.object(Path, "read_text", side_effect=OSError("missing")):
            self.assertFalse(is_wsl())

    def test_parse_mounts_decodes_escapes(self) -> None:
        mounts = parse_mounts(MOUNTS)
        self.assertEqual(
            mounts,
            [
                Mount(source="C:\\", mount_point="/mnt/c", fstype="9p"),
                Mount(source="/dev/sdc", mount_point="/", fstype="ext4"),
            ],
        )

        escaped = (
            "\\134\\134server\\134share\\040name /mnt/My\\040Drive\\011tab\\012nl "
            "fuse.sshfs\\040ro rw 0 0\n"
            "C:\\134040 /mnt/literal\\134040 9p rw 0 0\n"
            "\n"
            "# comment line\n"
            "malformed_only_two_fields /mnt/bad\n"
        )
        parsed = parse_mounts(escaped)
        self.assertEqual(
            parsed,
            [
                Mount(
                    source="\\\\server\\share name",
                    mount_point="/mnt/My Drive\ttab\nnl",
                    fstype="fuse.sshfs ro",
                ),
                Mount(
                    source="C:\\040",
                    mount_point="/mnt/literal\\040",
                    fstype="9p",
                ),
            ],
        )

    def test_read_mounts_returns_empty_on_unreadable_file(self) -> None:
        with mock.patch.object(Path, "read_text", side_effect=OSError("denied")):
            self.assertEqual(read_mounts(), [])

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing-mounts"
            with mock.patch("seekr.platform_info._MOUNTS_PATH", missing):
                self.assertEqual(read_mounts(), [])

            mounts_file = Path(tmp) / "mounts"
            mounts_file.write_text(MOUNTS, encoding="utf-8")
            with mock.patch("seekr.platform_info._MOUNTS_PATH", mounts_file):
                self.assertEqual(
                    read_mounts(),
                    [
                        Mount(source="C:\\", mount_point="/mnt/c", fstype="9p"),
                        Mount(source="/dev/sdc", mount_point="/", fstype="ext4"),
                    ],
                )

    def test_mount_for_longest_prefix_and_boundaries(self) -> None:
        mounts = parse_mounts(MOUNTS)

        with mock.patch.object(
            Path,
            "resolve",
            side_effect=AssertionError("mount_for must not touch disk"),
        ):
            self.assertEqual(
                mount_for(Path("/mnt/c/Users/alice/notes.md"), mounts),
                Mount(source="C:\\", mount_point="/mnt/c", fstype="9p"),
            )
            self.assertEqual(
                mount_for(Path("/mnt/c"), mounts),
                Mount(source="C:\\", mount_point="/mnt/c", fstype="9p"),
            )
            self.assertEqual(
                mount_for(Path("/mnt/cd/notes.md"), mounts),
                Mount(source="/dev/sdc", mount_point="/", fstype="ext4"),
            )
            self.assertEqual(
                mount_for(Path("/mnt/c/.."), mounts),
                Mount(source="/dev/sdc", mount_point="/", fstype="ext4"),
            )
            self.assertEqual(
                mount_for(Path("/home/alice/notes.md"), mounts),
                Mount(source="/dev/sdc", mount_point="/", fstype="ext4"),
            )
            self.assertEqual(
                mount_for(Path("/"), mounts),
                Mount(source="/dev/sdc", mount_point="/", fstype="ext4"),
            )
            self.assertIsNone(mount_for(Path("relative/notes.md"), mounts))
            self.assertIsNone(mount_for(Path("/mnt/c/notes.md"), ()))

    def test_mount_for_prefers_latest_stacked_mount_at_same_point(self) -> None:
        mounts = [
            Mount(source="rootfs", mount_point="/", fstype="rootfs"),
            Mount(source="/dev/sdd", mount_point="/", fstype="ext4"),
        ]
        self.assertEqual(
            mount_for(Path("/home/notes"), mounts),
            Mount(source="/dev/sdd", mount_point="/", fstype="ext4"),
        )

    def test_is_slow_path(self) -> None:
        mounts = parse_mounts(MOUNTS)
        self.assertTrue(is_slow_path(Path("/mnt/c/Users/alice/notes.md"), mounts))
        self.assertTrue(is_slow_path(Path("/mnt/c"), mounts))
        self.assertFalse(is_slow_path(Path("/mnt/cd/notes.md"), mounts))
        self.assertFalse(is_slow_path(Path("/home/alice/notes.md"), mounts))
        self.assertFalse(is_slow_path(Path("relative/notes.md"), mounts))

        for fstype in sorted(SLOW_FSTYPES):
            with self.subTest(fstype=fstype):
                slow_mounts = [
                    Mount(source="/dev/sda1", mount_point="/", fstype="ext4"),
                    Mount(source="remote", mount_point="/mnt/remote", fstype=fstype),
                ]
                self.assertTrue(is_slow_path(Path("/mnt/remote/file.md"), slow_mounts))

    def test_windows_path_pure(self) -> None:
        mounts = [
            Mount(source="C:\\", mount_point="/mnt/c", fstype="9p"),
            Mount(source="d:", mount_point="/mnt/d", fstype="drvfs"),
            Mount(source="/dev/sdc", mount_point="/", fstype="ext4"),
            Mount(source="\\\\server\\share", mount_point="/mnt/share", fstype="cifs"),
        ]

        self.assertEqual(
            windows_path("/mnt/c/Users/me/a.pdf", mounts),
            r"C:\Users\me\a.pdf",
        )
        self.assertEqual(windows_path("/mnt/c", mounts), "C:\\")
        self.assertEqual(windows_path("/mnt/c/", mounts), "C:\\")
        self.assertEqual(
            windows_path("/mnt/c/Users/../Users/me/a.pdf", mounts),
            r"C:\Users\me\a.pdf",
        )
        self.assertEqual(
            windows_path("/mnt/d/slides/deck.pptx", mounts),
            r"d:\slides\deck.pptx",
        )
        self.assertEqual(windows_path("/mnt/d", mounts), "d:\\")

        self.assertIsNone(windows_path("/home/alice/notes.md", mounts))
        self.assertIsNone(windows_path("/mnt/cd/notes.md", mounts))
        self.assertIsNone(windows_path("/mnt/c/..", mounts))
        self.assertIsNone(windows_path("relative/notes.md", mounts))
        self.assertIsNone(windows_path("", mounts))
        self.assertIsNone(windows_path("/mnt/share/file.pdf", mounts))
        self.assertIsNone(windows_path("/mnt/c/Users/me/a.pdf", ()))

    def test_wslpath_windows(self) -> None:
        calls: list[tuple[list[str], dict[str, object]]] = []

        def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append((argv, kwargs))
            return subprocess.CompletedProcess(
                args=argv,
                returncode=0,
                stdout="\\\\wsl.localhost\\Ubuntu\\tmp\\a.pdf\n",
                stderr="",
            )

        self.assertEqual(
            wslpath_windows("/tmp/a.pdf", run=fake_run),
            r"\\wsl.localhost\Ubuntu\tmp\a.pdf",
        )
        self.assertEqual(
            calls,
            [
                (
                    ["wslpath", "-w", "/tmp/a.pdf"],
                    {"capture_output": True, "text": True, "timeout": 2, "check": False},
                )
            ],
        )

        self.assertIsNone(
            wslpath_windows(
                "/tmp/a.pdf",
                run=lambda argv, **_: subprocess.CompletedProcess(
                    args=argv, returncode=1, stdout="", stderr="error"
                ),
            )
        )
        self.assertIsNone(
            wslpath_windows(
                "/tmp/a.pdf",
                run=lambda argv, **_: subprocess.CompletedProcess(
                    args=argv, returncode=0, stdout="   \n", stderr=""
                ),
            )
        )

        def raise_oserror(*_args: object, **_kwargs: object) -> object:
            raise FileNotFoundError("wslpath")

        def raise_timeout(*_args: object, **_kwargs: object) -> object:
            raise subprocess.TimeoutExpired(cmd=["wslpath", "-w", "/tmp/a.pdf"], timeout=2)

        self.assertIsNone(wslpath_windows("/tmp/a.pdf", run=raise_oserror))
        self.assertIsNone(wslpath_windows("/tmp/a.pdf", run=raise_timeout))


if __name__ == "__main__":
    unittest.main()


class SplitRootsTests(unittest.TestCase):
    def test_split_roots_separates_native_and_polled_in_order(self) -> None:
        mounts = parse_mounts(MOUNTS)

        native, polled = split_roots(
            [Path("/home/a/notes"), Path("/mnt/c/Users/b"), Path("/srv/x")], mounts
        )

        self.assertEqual(polled, [Path("/mnt/c/Users/b")])
        self.assertEqual(native, [Path("/home/a/notes"), Path("/srv/x")])
