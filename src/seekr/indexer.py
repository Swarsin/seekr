"""Incremental indexing with chunk-level vector reuse and safe sync semantics.

The indexer talks to an object that satisfies the small subset of the
:class:`seekr.store.Store` API it needs, so it can be tested without the
sqlite-vec extension.  Paths are stored as absolute resolved strings, which
makes rename detection and prune comparisons component-safe.
"""

from __future__ import annotations

import errno
import fnmatch
import hashlib
import os
import stat as stat_module
import warnings
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal

from . import extract as _extract_module
from .chunking import ChunkingWarning, chunk
from .config import Config
from .extract import ExtractionError, ExtractionResult
from .models import text_sha
from .store import StoredChunk

if TYPE_CHECKING:
    from .embed import Embedder
    from .store import Store

__all__ = ["MAX_FILE_ATTEMPTS", "IndexResult", "Indexer", "SyncReport"]

IndexResult = Literal["added", "updated", "unchanged", "skipped", "removed"]

#: Number of stat/hash/extract/re-stat rounds before a changing file is skipped.
MAX_FILE_ATTEMPTS = 3
_READ_BLOCK = 1024 * 1024


@dataclass
class SyncReport:
    """Counts and per-path diagnostics produced by one synchronous scan."""

    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    skipped: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[tuple[str, str]] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)


@dataclass
class _Outcome:
    result: IndexResult
    errors: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[tuple[str, str]] = field(default_factory=list)


def _error_text(exc: BaseException) -> str:
    message = str(exc).strip()
    return message or exc.__class__.__name__


# Thin proxies keep the extractor patchable both as ``seekr.indexer.extract``
# (tests) and as ``seekr.extract.extract`` (the module's public location).
def extract(path: Path, kind: str) -> ExtractionResult:
    return _extract_module.extract(path, kind)


def kind_for(path: Path) -> str | None:
    return _extract_module.kind_for(path)


def language_for(path: Path) -> str | None:
    return _extract_module.language_for(path)


def _is_within(path: Path, root: Path) -> bool:
    """Return whether *path* is *root* or below it using path components."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _signature(stat_result: os.stat_result) -> tuple[int, int]:
    return int(stat_result.st_mtime_ns), int(stat_result.st_size)


def _is_symlink_loop(exc: BaseException) -> bool:
    return getattr(exc, "errno", None) == errno.ELOOP


class Indexer:
    """Synchronise files into a Store without re-embedding unchanged text."""

    def __init__(self, store: Store, embedder: Embedder, cfg: Config) -> None:
        self.store = store
        self.embedder = embedder
        self.cfg = cfg
        self._last_report: SyncReport | None = None
        self._last_outcome: _Outcome | None = None

    # -- public API --------------------------------------------------------

    def get_report(self) -> SyncReport | None:
        """Return the report from the most recent completed :meth:`sync`."""
        return self._last_report

    def index_path(self, path: Path) -> IndexResult:
        """Index one path, recording failures instead of raising them."""
        outcome = self._index_path(Path(path))
        self._last_outcome = outcome
        return outcome.result

    def remove_path(self, path: Path) -> None:
        """Remove one indexed document.  A missing row is a no-op."""
        resolved, _ = self._resolve(Path(path))
        if resolved is not None:
            self.store.delete_document(str(resolved))

    def move_path(self, src: Path, dst: Path) -> IndexResult:
        """Move an indexed document after a watcher move event.

        The source document and its vectors are retained until the destination
        has been prepared successfully.  An unchanged ordinary rename is a
        metadata-only operation with no embedding calls.
        """
        outcome = self._move_path(Path(src), Path(dst))
        self._last_outcome = outcome
        return outcome.result

    def sync(self, roots: list[Path]) -> SyncReport:
        """Index every supported file below *roots*, then prune vanished files."""
        report = SyncReport()
        seen: set[str] = set()
        healthy_roots: list[Path] = []

        try:
            for raw_root in list(roots):
                root, issue = self._resolve(Path(raw_root))
                if root is None:
                    report.errors.append((str(raw_root), issue or "cannot resolve root"))
                    continue
                try:
                    root_stat = root.stat()
                except FileNotFoundError as exc:
                    report.errors.append(
                        (str(root), f"root is missing or offline: {_error_text(exc)}")
                    )
                    continue
                except OSError as exc:
                    report.errors.append(
                        (str(root), f"cannot stat root: {_error_text(exc)}")
                    )
                    continue

                if stat_module.S_ISREG(root_stat.st_mode):
                    self._index_walked_file(root, root, seen, report)
                    healthy_roots.append(root)
                elif stat_module.S_ISDIR(root_stat.st_mode):
                    files, walk_ok = self._walk_root(root, report)
                    for file_path in files:
                        self._index_walked_file(file_path, root, seen, report)
                    if walk_ok:
                        healthy_roots.append(root)
                else:
                    report.errors.append(
                        (str(root), "root is not a regular file or directory")
                    )

            for root in self._unique_roots(healthy_roots):
                self._prune_root(root, seen, report)
        finally:
            self._last_report = report
            try:
                self.store.set_last_error_count(len(report.errors))
            except Exception as exc:  # Persisting diagnostics must not hide the report.
                report.warnings.append(
                    (
                        str(self.cfg.db_path),
                        f"could not persist last error count: {_error_text(exc)}",
                    )
                )
        return report

    # -- scanning ----------------------------------------------------------

    def _walk_root(self, root: Path, report: SyncReport) -> tuple[list[Path], bool]:
        files: list[Path] = []
        walk_ok = True

        def onerror(exc: OSError) -> None:
            nonlocal walk_ok
            walk_ok = False
            filename = getattr(exc, "filename", None) or str(root)
            report.errors.append(
                (str(filename), f"cannot read directory: {_error_text(exc)}")
            )

        try:
            for dirpath, dirnames, filenames in os.walk(
                root, topdown=True, onerror=onerror, followlinks=False
            ):
                # os.walk lists symlinked directories in dirnames even when it
                # does not follow them; drop them so loops and escapes stay out.
                dirnames[:] = [
                    name for name in dirnames if not os.path.islink(os.path.join(dirpath, name))
                ]
                for name in filenames:
                    files.append(Path(dirpath) / name)
        except OSError as exc:
            walk_ok = False
            report.errors.append((str(root), f"cannot walk root: {_error_text(exc)}"))

        files.sort(key=lambda item: str(item))
        return files, walk_ok

    def _index_walked_file(
        self, path: Path, root: Path, seen: set[str], report: SyncReport
    ) -> None:
        if self._is_ignored(path, root):
            report.skipped += 1
            return

        resolved, issue = self._resolve(path)
        if resolved is None:
            report.skipped += 1
            report.warnings.append((str(path), issue or "cannot resolve path"))
            return
        if not _is_within(resolved, root):
            report.skipped += 1
            report.warnings.append(
                (str(path), "path resolves outside the indexed root; skipped")
            )
            return

        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        self._merge_outcome(report, self._index_path(resolved))

    def _prune_root(self, root: Path, seen: set[str], report: SyncReport) -> None:
        try:
            indexed_paths = self.store.list_paths(str(root))
        except Exception as exc:
            report.errors.append(
                (str(root), f"cannot list indexed paths: {_error_text(exc)}")
            )
            return

        for doc_path in indexed_paths:
            candidate = Path(doc_path)
            if doc_path in seen or not _is_within(candidate, root):
                continue
            try:
                os.lstat(candidate)
            except FileNotFoundError:
                try:
                    self.store.delete_document(doc_path)
                except Exception as exc:
                    report.errors.append(
                        (doc_path, f"cannot remove vanished document: {_error_text(exc)}")
                    )
                else:
                    report.removed += 1
            except OSError as exc:
                # Permission problems and transient I/O errors must never be
                # interpreted as deletion.
                report.warnings.append(
                    (doc_path, f"cannot verify path ({_error_text(exc)}); keeping document")
                )

    @staticmethod
    def _unique_roots(roots: list[Path]) -> list[Path]:
        unique: list[Path] = []
        for root in roots:
            if root not in unique:
                unique.append(root)
        return unique

    # -- one-file indexing -------------------------------------------------

    def _index_path(self, raw_path: Path, *, force: bool = False) -> _Outcome:
        outcome = _Outcome(result="skipped")
        try:
            return self._index_path_attempts(Path(raw_path), outcome, force=force)
        except FileNotFoundError:
            outcome.result = self._remove_vanished(Path(raw_path), outcome)
            return outcome
        except Exception as exc:  # Isolate every malformed or unreadable file.
            outcome.result = "skipped"
            outcome.errors.append((str(raw_path), _error_text(exc)))
            return outcome

    def _index_path_attempts(
        self, raw_path: Path, outcome: _Outcome, *, force: bool = False
    ) -> _Outcome:
        path, issue = self._resolve(raw_path)
        if path is None:
            outcome.warnings.append((str(raw_path), issue or "cannot resolve path"))
            return outcome
        key = str(path)

        if self._is_ignored(path):
            return outcome
        kind = kind_for(path)
        if kind is None:
            return outcome

        for _attempt in range(MAX_FILE_ATTEMPTS):
            try:
                before = path.stat()
            except FileNotFoundError:
                outcome.result = self._remove_vanished(path, outcome)
                return outcome
            except OSError as exc:
                if _is_symlink_loop(exc):
                    outcome.warnings.append(
                        (str(path), f"symlink loop; skipped: {_error_text(exc)}")
                    )
                else:
                    outcome.errors.append(
                        (str(path), f"cannot stat file: {_error_text(exc)}")
                    )
                return outcome

            if not stat_module.S_ISREG(before.st_mode):
                return outcome
            if before.st_size > self.cfg.max_file_bytes:
                outcome.warnings.append(
                    (
                        str(path),
                        f"file is larger than max_file_bytes ({self.cfg.max_file_bytes}); skipped",
                    )
                )
                return outcome

            try:
                doc = self.store.get_document(key)
            except Exception as exc:
                outcome.errors.append(
                    (str(path), f"cannot read document metadata: {_error_text(exc)}")
                )
                return outcome

            signature = _signature(before)
            if (
                not force
                and doc is not None
                and (doc.mtime_ns, doc.size) == signature
                and doc.kind == kind
            ):
                outcome.result = "unchanged"
                return outcome

            try:
                sha256 = self._sha256_file(path)
            except FileNotFoundError:
                outcome.result = self._remove_vanished(path, outcome)
                return outcome
            except OSError as exc:
                self._record_read_error(path, exc, outcome, "cannot hash file")
                return outcome

            if doc is not None and doc.sha256 == sha256 and doc.kind == kind:
                after = self._restat_after_read(path, outcome)
                if after is None:
                    return outcome
                if signature != _signature(after):
                    continue
                try:
                    with self.store.transaction(write=True):
                        self.store.upsert_document(
                            key,
                            before.st_mtime_ns,
                            before.st_size,
                            sha256,
                            kind,
                        )
                except Exception as exc:
                    outcome.errors.append(
                        (str(path), f"cannot update metadata: {_error_text(exc)}")
                    )
                    return outcome
                outcome.result = "unchanged"
                return outcome

            try:
                extraction = extract(path, kind)
            except FileNotFoundError:
                outcome.result = self._remove_vanished(path, outcome)
                return outcome
            except ExtractionError as exc:
                outcome.errors.append((str(path), _error_text(exc)))
                return outcome
            except (OSError, UnicodeError) as exc:
                self._record_read_error(path, exc, outcome, "cannot read file")
                return outcome
            except Exception as exc:
                outcome.errors.append(
                    (str(path), f"extraction failed: {_error_text(exc)}")
                )
                return outcome

            after = self._restat_after_read(path, outcome)
            if after is None:
                return outcome
            if signature != _signature(after):
                continue
            if after.st_size > self.cfg.max_file_bytes:
                outcome.warnings.append(
                    (
                        str(path),
                        f"file is larger than max_file_bytes ({self.cfg.max_file_bytes}); skipped",
                    )
                )
                return outcome

            try:
                specs, chunk_warnings = self._make_chunks(kind, extraction, path)
            except Exception as exc:
                outcome.errors.append(
                    (str(path), f"chunking failed: {_error_text(exc)}")
                )
                return outcome

            try:
                stored = self._prepare_chunks(specs)
            except Exception as exc:
                outcome.errors.append(
                    (str(path), f"cannot prepare chunks: {_error_text(exc)}")
                )
                return outcome

            try:
                with self.store.transaction(write=True):
                    doc_id = self.store.upsert_document(
                        key,
                        after.st_mtime_ns,
                        after.st_size,
                        sha256,
                        kind,
                    )
                    self.store.replace_chunks(doc_id, stored)
            except Exception as exc:
                outcome.errors.append(
                    (str(path), f"cannot write document: {_error_text(exc)}")
                )
                return outcome

            outcome.result = "updated" if doc is not None else "added"
            outcome.warnings.extend((str(path), warning) for warning in extraction.warnings)
            outcome.warnings.extend((str(path), warning) for warning in chunk_warnings)
            return outcome

        outcome.errors.append(
            (
                str(path),
                f"file kept changing while indexing; gave up after {MAX_FILE_ATTEMPTS} attempts",
            )
        )
        return outcome

    def _restat_after_read(
        self, path: Path, outcome: _Outcome
    ) -> os.stat_result | None:
        try:
            return path.stat()
        except FileNotFoundError:
            outcome.result = self._remove_vanished(path, outcome)
            return None
        except OSError as exc:
            self._record_read_error(path, exc, outcome, "cannot re-stat file")
            return None

    def _record_read_error(
        self, path: Path, exc: OSError, outcome: _Outcome, action: str
    ) -> None:
        if _is_symlink_loop(exc):
            outcome.warnings.append(
                (str(path), f"symlink loop; skipped: {_error_text(exc)}")
            )
        else:
            outcome.errors.append((str(path), f"{action}: {_error_text(exc)}"))

    def _remove_vanished(self, path: Path, outcome: _Outcome) -> IndexResult:
        resolved, issue = self._resolve(path)
        if resolved is None:
            outcome.warnings.append((str(path), issue or "cannot resolve path"))
            return "skipped"
        key = str(resolved)
        try:
            if self.store.get_document(key) is not None:
                self.store.delete_document(key)
            return "removed"
        except Exception as exc:
            outcome.errors.append(
                (str(path), f"cannot remove vanished file: {_error_text(exc)}")
            )
            return "skipped"

    def _make_chunks(
        self, kind: str, extraction: object, path: Path
    ) -> tuple[list[object], list[str]]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            specs = chunk(
                kind,
                extraction.units,  # type: ignore[attr-defined]
                self.cfg,
                language=language_for(path) if kind == "code" else None,
            )
        chunk_warnings = [
            str(item.message)
            for item in caught
            if issubclass(item.category, ChunkingWarning)
        ]
        return list(specs), chunk_warnings

    def _prepare_chunks(self, specs: list[object]) -> list[StoredChunk]:
        shas = [text_sha(spec.text) for spec in specs]  # type: ignore[attr-defined]
        found: dict[str, object] = {}
        missing_shas: list[str] = []
        missing_texts: list[str] = []

        for spec, sha in zip(specs, shas):
            if sha in found or sha in missing_shas:
                continue
            try:
                vector = self.store.find_vector_by_text_sha(sha)
            except Exception as exc:
                raise RuntimeError(f"vector lookup failed: {_error_text(exc)}") from exc
            if vector is not None:
                found[sha] = vector
            else:
                missing_shas.append(sha)
                missing_texts.append(spec.text)  # type: ignore[attr-defined]

        if missing_texts:
            try:
                vectors = self.embedder.embed_documents(missing_texts)
            except Exception as exc:
                raise RuntimeError(f"embedding failed: {_error_text(exc)}") from exc
            try:
                vector_count = len(vectors)  # type: ignore[arg-type]
            except TypeError as exc:
                raise RuntimeError("embedder returned a batch without a length") from exc
            if vector_count != len(missing_texts):
                raise RuntimeError(
                    "embedder returned "
                    f"{vector_count} vectors for {len(missing_texts)} texts"
                )
            for index, sha in enumerate(missing_shas):
                found[sha] = vectors[index]  # type: ignore[index]

        return [
            StoredChunk(spec=spec, text_sha=sha, vector=found[sha])
            for spec, sha in zip(specs, shas)
        ]

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(_READ_BLOCK), b""):
                digest.update(block)
        return digest.hexdigest()

    # -- rename / move -----------------------------------------------------

    def _remove_moved_source(
        self, src_path: Path, src_key: str, outcome: _Outcome
    ) -> _Outcome:
        """Drop the old row for a completed move whose destination is unusable."""
        try:
            self.store.delete_document(src_key)
        except Exception as exc:
            outcome.errors.append(
                (str(src_path), f"cannot remove moved source: {_error_text(exc)}")
            )
            return outcome
        outcome.result = "removed"
        return outcome

    def _move_path(self, src: Path, dst: Path) -> _Outcome:
        outcome = _Outcome(result="skipped")
        try:
            return self._move_path_attempts(src, dst, outcome)
        except Exception as exc:
            outcome.result = "skipped"
            outcome.errors.append((str(src), f"move failed: {_error_text(exc)}"))
            return outcome

    def _move_path_attempts(
        self, src: Path, dst: Path, outcome: _Outcome
    ) -> _Outcome:
        src_resolved, src_issue = self._resolve(src)
        dst_resolved, dst_issue = self._resolve(dst)
        if src_resolved is None:
            outcome.warnings.append((str(src), src_issue or "cannot resolve source"))
            return outcome
        if dst_resolved is None:
            outcome.warnings.append((str(dst), dst_issue or "cannot resolve destination"))
            return outcome

        src_key = str(src_resolved)
        dst_key = str(dst_resolved)
        if src_key == dst_key:
            return self._index_path(dst_resolved)

        try:
            src_doc = self.store.get_document(src_key)
        except Exception as exc:
            outcome.errors.append(
                (str(src), f"cannot read source metadata: {_error_text(exc)}")
            )
            return outcome
        if src_doc is None:
            return self._index_path(dst_resolved)

        if self._is_ignored(dst_resolved):
            # The move completed, so the source path no longer exists.  An
            # ignored destination must not leave the old path searchable.
            return self._remove_moved_source(src_resolved, src_key, outcome)
        dst_kind = kind_for(dst_resolved)
        if dst_kind is None:
            return self._remove_moved_source(src_resolved, src_key, outcome)

        dst_stat: os.stat_result | None = None
        for attempt in range(MAX_FILE_ATTEMPTS):
            try:
                dst_stat = dst_resolved.stat()
            except FileNotFoundError:
                if attempt + 1 >= MAX_FILE_ATTEMPTS:
                    outcome.warnings.append(
                        (
                            str(dst),
                            "destination is missing; removing stale source document",
                        )
                    )
                    return self._remove_moved_source(
                        src_resolved, src_key, outcome
                    )
                continue
            except OSError as exc:
                outcome.errors.append(
                    (str(dst), f"cannot stat destination: {_error_text(exc)}")
                )
                return outcome
            break
        assert dst_stat is not None

        if not stat_module.S_ISREG(dst_stat.st_mode):
            return self._remove_moved_source(src_resolved, src_key, outcome)
        if dst_stat.st_size > self.cfg.max_file_bytes:
            outcome.warnings.append(
                (
                    str(dst),
                    (
                        f"destination is larger than max_file_bytes "
                        f"({self.cfg.max_file_bytes}); removing stale source document"
                    ),
                )
            )
            return self._remove_moved_source(src_resolved, src_key, outcome)

        try:
            dst_doc = self.store.get_document(dst_key)
        except Exception as exc:
            outcome.errors.append(
                (str(dst), f"cannot read destination metadata: {_error_text(exc)}")
            )
            return outcome

        try:
            dst_sha = self._sha256_file(dst_resolved)
        except FileNotFoundError:
            outcome.warnings.append(
                (str(dst), "destination vanished; removing stale source document")
            )
            return self._remove_moved_source(src_resolved, src_key, outcome)
        except OSError as exc:
            outcome.errors.append(
                (str(dst), f"cannot hash destination: {_error_text(exc)}")
            )
            return outcome

        same_content_and_kind = (
            src_doc.sha256 == dst_sha
            and src_doc.kind == dst_kind
            and self._same_chunking_language(src_doc.path, dst_resolved, dst_kind)
        )

        if same_content_and_kind and dst_doc is None:
            try:
                with self.store.transaction(write=True):
                    self.store.move_document(
                        src_key,
                        dst_key,
                        mtime_ns=dst_stat.st_mtime_ns,
                        size=dst_stat.st_size,
                    )
            except KeyError:
                return self._index_path(dst_resolved)
            except ValueError:
                # A destination row appeared concurrently; re-index it below
                # so the newest file content wins.
                pass
            except Exception as exc:
                outcome.errors.append(
                    (str(dst), f"cannot move document: {_error_text(exc)}")
                )
                return outcome
            else:
                outcome.result = "unchanged"
                return outcome

        if (
            same_content_and_kind
            and dst_doc is not None
            and dst_doc.sha256 == dst_sha
            and dst_doc.kind == dst_kind
        ):
            try:
                with self.store.transaction(write=True):
                    self.store.upsert_document(
                        dst_key,
                        dst_stat.st_mtime_ns,
                        dst_stat.st_size,
                        dst_doc.sha256,
                        dst_doc.kind,
                    )
                    self.store.delete_document(src_key)
            except Exception as exc:
                outcome.errors.append(
                    (str(dst), f"cannot finalise move: {_error_text(exc)}")
                )
                return outcome
            outcome.result = "removed"
            return outcome

        # Destination content or kind differs: prepare it while the source
        # document (and therefore its vectors) still exists for chunk reuse.
        # If a stale destination row claims a different hash but the file still
        # happens to match its old stat, force a real re-read rather than
        # trusting the metadata cache.
        stale_destination = dst_doc is not None and dst_doc.sha256 != dst_sha
        destination = self._index_path(dst_resolved, force=stale_destination)
        if destination.result not in {"added", "updated", "unchanged"}:
            return destination

        try:
            self.store.delete_document(src_key)
        except Exception as exc:
            destination.errors.append(
                (str(src), f"destination prepared but source cleanup failed: {_error_text(exc)}")
            )
        return destination

    # -- path helpers ------------------------------------------------------

    @staticmethod
    def _same_chunking_language(
        old_path: str, new_path: Path, kind: str
    ) -> bool:
        if kind != "code":
            return True
        return language_for(Path(old_path)) == language_for(new_path)

    @staticmethod
    def _resolve(path: Path) -> tuple[Path | None, str | None]:
        try:
            expanded = Path(path).expanduser()
        except (OSError, RuntimeError) as exc:
            return None, f"cannot expand path: {_error_text(exc)}"
        try:
            return expanded.resolve(strict=False), None
        except (OSError, RuntimeError) as exc:
            return None, f"cannot resolve path: {_error_text(exc)}"

    def _is_ignored(self, path: Path, root: Path | None = None) -> bool:
        patterns = self.cfg.ignore_globs
        if not patterns:
            return False

        candidate = Path(path)
        relatives: list[str] = []
        if root is not None:
            try:
                relatives.append(candidate.relative_to(root).as_posix())
            except ValueError:
                pass
        relatives.append(candidate.as_posix())

        for raw_pattern in patterns:
            pattern = str(raw_pattern).strip()
            pattern = pattern.removeprefix("./")
            if not pattern:
                continue
            if any(
                fnmatch.fnmatchcase(part, pattern)
                for part in candidate.parts
                if part
            ):
                return True
            for relative in relatives:
                if fnmatch.fnmatchcase(relative, pattern):
                    return True
                try:
                    if PurePosixPath(relative).match(pattern):
                        return True
                except ValueError:
                    continue
        return False

    @staticmethod
    def _merge_outcome(report: SyncReport, outcome: _Outcome) -> None:
        if outcome.result == "added":
            report.added += 1
        elif outcome.result == "updated":
            report.updated += 1
        elif outcome.result == "unchanged":
            report.unchanged += 1
        elif outcome.result == "removed":
            report.removed += 1
        else:
            report.skipped += 1
        report.errors.extend(outcome.errors)
        report.warnings.extend(outcome.warnings)
