"""Offline tests for additive, byte-preserving config roots persistence."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from unittest import mock

import pytest

from seekr.config import load_config
from seekr.web_config import (
    ConfigInspection,
    add_root,
    config_revision,
    inspect_config,
    read_roots,
    remove_root,
    set_roots,
    toml_string,
)
from seekr.web_contract import ApiError


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")  # keep CRLF exactly
    return path


def _root(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir()
    return path


def _toml_path(path: Path) -> str:
    return json.dumps(str(path))


def _config_with_roots(tmp_path: Path, roots: list[Path], *, db_path: Path | None = None) -> Path:
    path = tmp_path / "config.toml"
    lines = [
        'embed_model = "fake"',
        f"db_path = {_toml_path(db_path or (tmp_path / 'index.db'))}",
        "roots = [" + ", ".join(_toml_path(root) for root in roots) + "]",
    ]
    return _write(path, "\n".join(lines) + "\n")


def _code(exc: ApiError) -> str:
    return exc.code


def test_missing_config_is_empty_with_empty_revision(tmp_path: Path) -> None:
    config = tmp_path / "missing" / "config.toml"

    snapshot = read_roots(config)

    assert snapshot.roots == ()
    assert snapshot.revision == hashlib.sha256(b"").hexdigest()
    assert snapshot.path == config
    assert config_revision(config) == snapshot.revision
    assert not config.parent.exists()


def test_absent_roots_appended_to_flat_file_without_touching_prefix(tmp_path: Path) -> None:
    root = _root(tmp_path, "docs")
    config = _write(
        tmp_path / "config.toml",
        '# keep this comment\nembed_model = "fake"  # trailing\ndb_path = "/tmp/x.db"\n',
    )
    before = config.read_bytes()

    snapshot = add_root(config, str(root), config_revision(config))

    after = config.read_bytes()
    assert after.startswith(before)
    assert after.endswith(f"roots = [{_toml_path(root)}]\n".encode())
    assert snapshot.roots == (str(root),)
    assert load_config(config).roots == (root,)
    assert snapshot.revision == hashlib.sha256(after).hexdigest()


def test_absent_roots_preserves_crlf_and_adds_one_line(tmp_path: Path) -> None:
    root = _root(tmp_path, "docs")
    config = _write(
        tmp_path / "config.toml",
        'embed_model = "fake"\r\ndb_path = "/tmp/x.db"',
    )

    add_root(config, str(root), config_revision(config))

    raw = config.read_bytes()
    assert b"db_path = \"/tmp/x.db\"\r\nroots =" in raw
    assert b"\n" not in raw.replace(b"\r\n", b"")


def test_single_line_array_replaced_preserving_key_spacing_comment_and_crlf(
    tmp_path: Path,
) -> None:
    first = _root(tmp_path, "first")
    second = _root(tmp_path, "second")
    config = _write(
        tmp_path / "config.toml",
        f'embed_model = "fake"\r\nroots  =  [ {_toml_path(first)} ]  # keep me\r\n',
    )
    before = config.read_bytes()

    add_root(config, str(second), config_revision(config))

    after = config.read_bytes()
    expected = (
        f'embed_model = "fake"\r\nroots  =  [ {_toml_path(first)}, {_toml_path(second)} ]'
        "  # keep me\r\n"
    ).encode()
    assert after == expected
    assert before.split(b"roots")[0] in after
    assert load_config(config).roots == (first, second)


def test_empty_array_shape_and_roundtrip(tmp_path: Path) -> None:
    root = _root(tmp_path, "docs")
    config = _write(tmp_path / "config.toml", "roots = []  # empty\n")

    snapshot = add_root(config, str(root), config_revision(config))

    assert config.read_text(encoding="utf-8") == f"roots = [{_toml_path(root)}]  # empty\n"
    assert snapshot.roots == (str(root),)
    assert load_config(config).roots == (root,)


def test_byte_preservation_outside_the_replaced_span(tmp_path: Path) -> None:
    first = _root(tmp_path, "first")
    second = _root(tmp_path, "second")
    prefix = (
        "# leading comment\n"
        'ignore_globs = [".git", "node_modules"]\n'
        'embed_model = "fake"\n'
    )
    suffix = '\n# trailing comment\nchunk_target_words = 120\n'
    config = _write(
        tmp_path / "config.toml",
        prefix + f"roots = [{_toml_path(first)}]\n" + suffix,
    )

    add_root(config, str(second), config_revision(config))

    text = config.read_text(encoding="utf-8")
    assert text.startswith(prefix)
    assert text.endswith(suffix)
    assert load_config(config).roots == (first, second)


def test_toml_safe_escaping_for_hostile_directory_names(tmp_path: Path) -> None:
    names = ["hyphen-dir", "unicodé", "apos'trophe", "semi;colon"]
    if os.name == "posix":
        # Windows forbids these characters in file names (and treats the
        # backslash as a separator, which every Windows root already has).
        names += ['quote"dir', "back\\slash", "tab\tdir", "new\nline", "control\x01dir"]
    config = _write(tmp_path / "config.toml", 'embed_model = "fake"\n')
    added: list[Path] = []
    for name in names:
        root = _root(tmp_path, name)
        snapshot = add_root(config, str(root), config_revision(config))
        added.append(root)
        assert snapshot.roots == tuple(str(item) for item in added)

    loaded = load_config(config)
    assert loaded.roots == tuple(added)
    text = config.read_text(encoding="utf-8")
    if os.name == "posix":
        assert "\\n" in text and "\\t" in text and "\\\\" in text and '\\"' in text
    else:
        assert "\\\\" in text  # escaped path separators


def test_unicode_line_separator_is_not_a_toml_newline(tmp_path: Path) -> None:
    first = _root(tmp_path, "line\u2028dir")
    second = _root(tmp_path, "second")
    config = _write(
        tmp_path / "config.toml",
        "# comment\u2028still a TOML comment\nroots = []\n",
    )

    after_first = add_root(config, str(first), config_revision(config))
    after_second = add_root(config, str(second), after_first.revision)

    assert after_second.roots == (str(first), str(second))
    assert load_config(config).roots == (first, second)
    text = config.read_text(encoding="utf-8")
    assert "\u2028" in text
    assert text.startswith("# comment\u2028still a TOML comment\n")


def test_submitted_root_validation(tmp_path: Path) -> None:
    config = _write(tmp_path / "config.toml", 'embed_model = "fake"\n')
    revision = config_revision(config)
    file_root = tmp_path / "file.txt"
    file_root.write_text("not a directory", encoding="utf-8")
    too_long = "a" * 4097

    cases = [
        (str(tmp_path / "missing"), "invalid_root"),
        (str(file_root), "invalid_root"),
        ("bad\x00path", "invalid_root"),
        (too_long, "invalid_root"),
        ("\ud800", "invalid_root"),
        ("", "invalid_root"),
    ]
    for value, expected in cases:
        with pytest.raises(ApiError) as raised:
            add_root(config, value, revision)
        assert _code(raised.value) == expected

    assert config.read_text(encoding="utf-8") == 'embed_model = "fake"\n'


def test_duplicate_root_uses_canonical_path(tmp_path: Path) -> None:
    root = _root(tmp_path, "docs")
    config = _config_with_roots(tmp_path, [root])
    revision = config_revision(config)

    with pytest.raises(ApiError) as raised:
        add_root(config, str(root), revision)

    assert _code(raised.value) == "duplicate_root"
    assert config_revision(config) == revision


def test_roots_limit(tmp_path: Path) -> None:
    roots = [_root(tmp_path, f"root-{index:02d}") for index in range(50)]
    extra = _root(tmp_path, "extra")
    config = _config_with_roots(tmp_path, roots)

    with pytest.raises(ApiError) as raised:
        add_root(config, str(extra), config_revision(config))

    assert _code(raised.value) == "roots_limit"


def test_stale_revision_leaves_file_untouched(tmp_path: Path) -> None:
    root = _root(tmp_path, "docs")
    config = _config_with_roots(tmp_path, [root])
    before = config.read_bytes()

    with pytest.raises(ApiError) as raised:
        add_root(config, str(_root(tmp_path, "new")), "0" * 64)

    assert _code(raised.value) == "stale_config"
    assert config.read_bytes() == before


def test_external_edit_detected_immediately_before_replace(tmp_path: Path) -> None:
    root = _root(tmp_path, "docs")
    config = _config_with_roots(tmp_path, [root])
    original = config.read_bytes()
    real_load = load_config

    def edit_after_validation(path: Path):
        loaded = real_load(path)
        if path.name.endswith(".tmp"):
            config.write_bytes(original + b"# external edit\n")
        return loaded

    with (
        mock.patch("seekr.web_config.load_config", side_effect=edit_after_validation),
        pytest.raises(ApiError) as raised,
    ):
        add_root(config, str(_root(tmp_path, "new")), config_revision(config))

    assert _code(raised.value) == "stale_config"
    assert config.read_bytes() == original + b"# external edit\n"


def test_symlink_config_target_is_refused(tmp_path: Path) -> None:
    root = _root(tmp_path, "docs")
    target = _write(tmp_path / "real.toml", 'embed_model = "fake"\n')
    link = tmp_path / "config.toml"
    link.symlink_to(target)
    before = target.read_bytes()

    with pytest.raises(ApiError) as raised:
        add_root(link, str(root), config_revision(link))

    assert _code(raised.value) == "config_invalid"
    assert target.read_bytes() == before


def test_atomic_replace_failure_preserves_existing_file(tmp_path: Path) -> None:
    root = _root(tmp_path, "docs")
    config = _config_with_roots(tmp_path, [])
    before = config.read_bytes()

    with (
        mock.patch("seekr.web_config.os.replace", side_effect=OSError("replace failed")),
        pytest.raises(ApiError) as raised,
    ):
        add_root(config, str(root), config_revision(config))

    assert _code(raised.value) == "config_write_failed"
    assert config.read_bytes() == before
    assert list(tmp_path.glob(".config.toml.*.tmp")) == []


def test_add_root_works_without_fchmod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Windows Python before 3.13 has no os.fchmod.
    monkeypatch.delattr(os, "fchmod", raising=False)
    root = _root(tmp_path, "docs")
    config = _config_with_roots(tmp_path, [])

    snapshot = add_root(config, str(root), config_revision(config))

    assert snapshot.roots == (str(root),)
    assert load_config(config).roots == (root,)
    assert list(tmp_path.glob(".config.toml.*.tmp")) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes do not exist on Windows")
def test_existing_mode_is_preserved_and_new_files_are_private(tmp_path: Path) -> None:
    root = _root(tmp_path, "docs")
    config = _config_with_roots(tmp_path, [])
    os.chmod(config, 0o640)

    add_root(config, str(root), config_revision(config))
    assert stat.S_IMODE(config.stat().st_mode) == 0o640

    new_root = _root(tmp_path, "new-root")
    nested = tmp_path / "nested" / "config" / "config.toml"
    add_root(nested, str(new_root), config_revision(nested))

    assert stat.S_IMODE(nested.stat().st_mode) == 0o600
    assert stat.S_IMODE(nested.parent.stat().st_mode) == 0o700
    assert load_config(nested).roots == (new_root,)


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes do not exist on Windows")
def test_existing_parent_mode_is_not_changed(tmp_path: Path) -> None:
    root = _root(tmp_path, "docs")
    parent = tmp_path / "existing"
    parent.mkdir()
    os.chmod(parent, 0o755)
    config = parent / "config.toml"

    add_root(config, str(root), config_revision(config))

    assert stat.S_IMODE(parent.stat().st_mode) == 0o755


def test_invalid_config_is_untouched(tmp_path: Path) -> None:
    root = _root(tmp_path, "docs")
    config = _write(tmp_path / "config.toml", "roots = [broken\n")
    before = config.read_bytes()

    for operation in (
        lambda: read_roots(config),
        lambda: add_root(config, str(root), config_revision(config)),
    ):
        with pytest.raises(ApiError) as raised:
            operation()
        assert _code(raised.value) == "config_invalid"
    assert config.read_bytes() == before


@pytest.mark.parametrize(
    "text",
    [
        'roots = [\n  "/tmp/a",\n]\n',
        '"roots" = ["/tmp/a"]\n',
        "'roots' = [\"/tmp/a\"]\n",
        'roots.value = ["/tmp/a"]\n',
        "[roots]\nvalue = ['/tmp/a']\n",
        "[[roots]]\nvalue = '/tmp/a'\n",
        "roots = [],\nroots = []\n",
        "roots = [ # comment\n]\n",
    ],
)
def test_ambiguous_or_unsupported_roots_shapes_are_refused(tmp_path: Path, text: str) -> None:
    root = _root(tmp_path, "docs")
    config = _write(tmp_path / "config.toml", text)

    with pytest.raises(ApiError) as raised:
        add_root(config, str(root), config_revision(config))

    assert _code(raised.value) == "config_invalid"


def test_absent_roots_with_table_header_is_refused(tmp_path: Path) -> None:
    root = _root(tmp_path, "docs")
    config = _write(
        tmp_path / "config.toml",
        'embed_model = "fake"\n[unrelated]\nvalue = 1\n',
    )
    before = config.read_bytes()

    with pytest.raises(ApiError) as raised:
        add_root(config, str(root), config_revision(config))

    assert _code(raised.value) == "config_invalid"
    assert config.read_bytes() == before


def test_inspect_config_reports_invalid_but_parseable_file(tmp_path: Path) -> None:
    config = _write(tmp_path / "config.toml", 'roots = ["/tmp/a"]\nunknown = 1\n')

    inspection = inspect_config(config)

    assert isinstance(inspection, ConfigInspection)
    assert inspection.state == "invalid"
    assert inspection.revision == hashlib.sha256(config.read_bytes()).hexdigest()
    assert inspection.roots == 1
    assert inspection.message is not None


def test_inspect_config_unreadable_path_has_no_revision(tmp_path: Path) -> None:
    inspection = inspect_config(tmp_path)

    assert inspection.state == "invalid"
    assert inspection.revision is None
    assert inspection.roots is None


def test_read_roots_displays_stored_order_and_form(tmp_path: Path) -> None:
    config = _write(
        tmp_path / "config.toml",
        'roots = ["relative/path", "~/notes", "/absolute"]\n',
    )

    snapshot = read_roots(config)

    assert snapshot.roots == ("relative/path", "~/notes", "/absolute")
    assert snapshot.revision == hashlib.sha256(config.read_bytes()).hexdigest()


def test_toml_string_roundtrips_tricky_values() -> None:
    import tomllib

    values = [
        'quote"dir',
        "back\\slash",
        "tab\tnew\nline",
        "unicodé/日本語",
    ]
    for value in values:
        parsed = tomllib.loads(f"v = {toml_string(value)}\n")
        assert parsed["v"] == value


def test_set_roots_preserves_comments_and_other_keys(tmp_path: Path) -> None:
    old = _root(tmp_path, "old")
    new1 = _root(tmp_path, "new1")
    new2 = _root(tmp_path, "new2")
    config = _write(
        tmp_path / "config.toml",
        (
            "# top comment\n"
            "poll_interval_seconds = 30\n"
            f"roots = [{_toml_path(old)}]  # inline comment\n"
            "# bottom comment\n"
            'embed_model = "custom/model"\n'
        ),
    )

    snapshot = set_roots(config, [new1, new2], revision=config_revision(config))

    text = config.read_text(encoding="utf-8")
    assert "# top comment\n" in text
    assert "poll_interval_seconds = 30\n" in text
    assert "# inline comment\n" in text
    assert "# bottom comment\n" in text
    assert 'embed_model = "custom/model"\n' in text
    assert snapshot.roots == (str(new1), str(new2))
    loaded = load_config(config)
    assert loaded.roots == (new1, new2)
    assert loaded.poll_interval_seconds == 30
    assert loaded.embed_model == "custom/model"


def test_set_roots_appends_when_absent_and_checks_revision(tmp_path: Path) -> None:
    root = _root(tmp_path, "docs")
    config = _write(tmp_path / "config.toml", "poll_interval_seconds = 45\n")
    before = config.read_bytes()

    with pytest.raises(ApiError) as raised:
        set_roots(config, [root], revision="0" * 64)
    assert _code(raised.value) == "stale_config"
    assert config.read_bytes() == before

    snapshot = set_roots(config, [root])
    assert snapshot.roots == (str(root),)
    loaded = load_config(config)
    assert loaded.roots == (root,)
    assert loaded.poll_interval_seconds == 45

    cleared = set_roots(config, [])
    assert cleared.roots == ()
    assert load_config(config).roots == ()


def test_remove_root_preserves_comments_other_keys_crlf_and_trailing_comment(
    tmp_path: Path,
) -> None:
    first = _root(tmp_path, "first")
    second = _root(tmp_path, "second")
    config = _write(
        tmp_path / "config.toml",
        (
            "# leading comment\r\n"
            'embed_model = "fake"\r\n'
            f"roots  =  [ {_toml_path(first)}, {_toml_path(second)} ]  # keep me\r\n"
            "# trailing comment\r\n"
        ),
    )

    snapshot = remove_root(config, str(first), config_revision(config))

    expected = (
        "# leading comment\r\n"
        'embed_model = "fake"\r\n'
        f"roots  =  [{_toml_path(second)}]  # keep me\r\n"
        "# trailing comment\r\n"
    ).encode()
    assert config.read_bytes() == expected
    assert snapshot.roots == (str(second),)
    assert snapshot.revision == hashlib.sha256(expected).hexdigest()
    assert load_config(config).roots == (second,)


def test_remove_root_matches_canonically_and_allows_missing_dir_on_disk(
    tmp_path: Path,
) -> None:
    first = _root(tmp_path, "first")
    second = _root(tmp_path, "second")
    alias = tmp_path / "second-link"
    alias.symlink_to(second)
    config = _config_with_roots(tmp_path, [first, second])

    # Remove a folder that was already deleted from disk, passing a trailing slash.
    first.rmdir()
    after_first = remove_root(config, f"{first}/", config_revision(config))
    assert after_first.roots == (str(second),)

    # Remove via a symlink alias to the remaining root, leaving roots = [].
    after_second = remove_root(config, str(alias), after_first.revision)
    assert after_second.roots == ()
    assert load_config(config).roots == ()


def test_remove_root_unknown_or_invalid_raises_invalid_root_without_touching_file(
    tmp_path: Path,
) -> None:
    first = _root(tmp_path, "first")
    other = _root(tmp_path, "other")
    config = _config_with_roots(tmp_path, [first])
    before = config.read_bytes()
    revision = config_revision(config)

    for bad_path in (str(other), "", "   ", "bad\x00path", "\ud800", "a" * 4097):
        with pytest.raises(ApiError) as raised:
            remove_root(config, bad_path, revision)
        assert _code(raised.value) == "invalid_root"
        assert raised.value.status == 400
        assert "path" in (raised.value.fields or {})

    assert config.read_bytes() == before

    missing_config = tmp_path / "missing" / "config.toml"
    with pytest.raises(ApiError) as raised:
        remove_root(missing_config, str(other), config_revision(missing_config))
    assert _code(raised.value) == "invalid_root"
    assert not missing_config.exists()


def test_remove_root_stale_revision_and_external_edit_raise_stale_config(
    tmp_path: Path,
) -> None:
    first = _root(tmp_path, "first")
    config = _config_with_roots(tmp_path, [first])
    before = config.read_bytes()

    with pytest.raises(ApiError) as raised:
        remove_root(config, str(first), "0" * 64)
    assert _code(raised.value) == "stale_config"
    assert config.read_bytes() == before

    real_load = load_config

    def edit_after_validation(path: Path):
        loaded = real_load(path)
        if path.name.endswith(".tmp"):
            config.write_bytes(before + b"# external edit\n")
        return loaded

    with (
        mock.patch("seekr.web_config.load_config", side_effect=edit_after_validation),
        pytest.raises(ApiError) as raised,
    ):
        remove_root(config, str(first), config_revision(config))

    assert _code(raised.value) == "stale_config"
    assert config.read_bytes() == before + b"# external edit\n"


def test_remove_root_refuses_symlink_and_unsupported_toml_shapes(tmp_path: Path) -> None:
    first = _root(tmp_path, "first")
    target = _write(tmp_path / "real.toml", f"roots = [{_toml_path(first)}]\n")
    link = tmp_path / "config.toml"
    link.symlink_to(target)
    before = target.read_bytes()

    with pytest.raises(ApiError) as raised:
        remove_root(link, str(first), config_revision(link))
    assert _code(raised.value) == "config_invalid"
    assert target.read_bytes() == before

    multiline = _write(
        tmp_path / "multiline.toml",
        f'embed_model = "fake"\nroots = [\n  {_toml_path(first)},\n]\n',
    )
    with pytest.raises(ApiError) as raised:
        remove_root(multiline, str(first), config_revision(multiline))
    assert _code(raised.value) == "config_invalid"


NFC = "M\u00f3dulo"
NFD = "Mo\u0301dulo"


def test_darwin_case_and_unicode_variants_are_duplicate_roots(tmp_path: Path) -> None:
    root = _root(tmp_path, NFC.lower())  # exists as "módulo" (NFC)
    for spelling in (tmp_path / NFD.upper(), tmp_path / NFD.lower()):
        config = _config_with_roots(tmp_path, [spelling])
        with mock.patch("seekr.pathnorm._PLATFORM", "darwin"), pytest.raises(
            ApiError
        ) as raised:
            add_root(config, str(root), config_revision(config))
        assert _code(raised.value) == "duplicate_root"


def test_linux_case_and_unicode_variants_stay_distinct_roots(tmp_path: Path) -> None:
    root = _root(tmp_path, NFC.lower())
    config = _config_with_roots(tmp_path, [tmp_path / NFD.upper()])
    with mock.patch("seekr.pathnorm._PLATFORM", "linux"):
        snapshot = add_root(config, str(root), config_revision(config))
    assert len(snapshot.roots) == 2


def test_darwin_remove_root_accepts_case_and_unicode_variants(tmp_path: Path) -> None:
    configured = tmp_path / NFD.upper()
    config = _config_with_roots(tmp_path, [configured])
    with mock.patch("seekr.pathnorm._PLATFORM", "linux"), pytest.raises(ApiError):
        remove_root(config, str(tmp_path / NFC.lower()), config_revision(config))
    with mock.patch("seekr.pathnorm._PLATFORM", "darwin"):
        snapshot = remove_root(config, str(tmp_path / NFC.lower()), config_revision(config))
    assert snapshot.roots == ()
