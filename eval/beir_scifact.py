#!/usr/bin/env python3
"""Convert the public BEIR SciFact dataset into seekr's evaluation layout.

Data source
-----------
The primary archive is the public BEIR SciFact dataset published by the
UKP Lab at TU Darmstadt::

    https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip

Downloading is opt-in: importing this module has no side effects, and network
access happens only when this script is explicitly run without an existing
local archive/source.  The archive has no trusted checksum here, so the script
only accepts HTTPS from the documented host and validates the ZIP structure
before selectively extracting the three required files.

If the primary URL is unavailable, the error message prints the manual
Hugging Face fallback described in ``docs/eval.md``.

Output layout (under ``eval/data`` by default)::

    scifact.zip                 cached primary archive
    scifact-source/             selectively extracted corpus/queries/qrels
    scifact/corpus/<id>.txt     title + blank line + text
    scifact/queries.jsonl       file-level queries whose ``relevant`` entries
                                are root-relative ``<id>.txt`` corpus paths
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

SCIFACT_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)
SCIFACT_HF_URL = "https://huggingface.co/datasets/BeIR/scifact/tree/main"
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_EXTRACT_BYTES = 512 * 1024 * 1024
_READ_BLOCK = 1024 * 1024

_ARCHIVE_NAME = "scifact.zip"
_SOURCE_NAME = "scifact-source"
_OUTPUT_NAME = "scifact"
_REQUIRED_MEMBERS = {
    "scifact/corpus.jsonl": Path("corpus.jsonl"),
    "scifact/queries.jsonl": Path("queries.jsonl"),
    "scifact/qrels/test.tsv": Path("qrels") / "test.tsv",
}


def _manual_fallback_message(reason: object, source_dir: Path | None = None) -> str:
    source = source_dir or Path("<eval/data>/scifact-source")
    return (
        f"cannot acquire the SciFact dataset: {reason}. "
        "Manual fallback: open "
        f"{SCIFACT_HF_URL} , download corpus.jsonl, queries.jsonl and "
        "qrels/test.tsv, then place them at "
        f"{source / 'corpus.jsonl'}, {source / 'queries.jsonl'} and "
        f"{source / 'qrels' / 'test.tsv'} (or download the official scifact.zip "
        "from the trusted UKP URL and place it next to the source directory). "
        "Re-run `python eval/beir_scifact.py --no-download`; no network access "
        "is attempted with that flag."
    )


def _require_https(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"refusing non-HTTPS dataset URL: {url!r}")


def download_scifact_archive(
    destination: Path,
    *,
    url: str = SCIFACT_URL,
    timeout: float = 60.0,
) -> Path:
    """Return a valid local SciFact archive, downloading it only if needed.

    An existing valid ZIP is reused without network access.  An existing
    non-ZIP file is reported as an actionable error instead of being silently
    overwritten.  Downloads are streamed to a ``.part`` file, size-capped and
    atomically renamed only after ZIP validation.
    """
    destination = Path(destination)
    if destination.exists():
        if not destination.is_file():
            raise RuntimeError(f"expected a ZIP file at {destination}")
        if zipfile.is_zipfile(destination):
            return destination
        raise RuntimeError(
            _manual_fallback_message(
                f"existing file {destination} is not a valid ZIP archive; "
                "remove or replace it",
                destination.parent / _SOURCE_NAME,
            )
        )

    _require_https(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "seekr-eval/0.1 (+offline local evaluation)"},
    )
    try:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                length_header = response.headers.get("Content-Length")
                if length_header:
                    try:
                        declared = int(length_header)
                    except ValueError:
                        declared = 0
                    if declared > MAX_ARCHIVE_BYTES:
                        raise RuntimeError(
                            _manual_fallback_message(
                                f"archive is larger than {MAX_ARCHIVE_BYTES} bytes",
                                destination.parent / _SOURCE_NAME,
                            )
                        )
                total = 0
                with partial.open("wb") as handle:
                    while True:
                        chunk = response.read(_READ_BLOCK)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > MAX_ARCHIVE_BYTES:
                            raise RuntimeError(
                                _manual_fallback_message(
                                    "download exceeded the archive size limit",
                                    destination.parent / _SOURCE_NAME,
                                )
                            )
                        handle.write(chunk)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                _manual_fallback_message(exc, destination.parent / _SOURCE_NAME)
            ) from exc

        if total == 0:
            raise RuntimeError(
                _manual_fallback_message(
                    "downloaded archive was empty",
                    destination.parent / _SOURCE_NAME,
                )
            )
        if not zipfile.is_zipfile(partial):
            raise RuntimeError(
                _manual_fallback_message(
                    "downloaded file is not a ZIP archive",
                    destination.parent / _SOURCE_NAME,
                )
            )
        os.replace(partial, destination)
        return destination
    finally:
        partial.unlink(missing_ok=True)


def _validate_member_name(name: str) -> None:
    pure = PurePosixPath(name)
    if (
        not name
        or "\x00" in name
        or "\\" in name
        or pure.is_absolute()
        or ".." in pure.parts
    ):
        raise ValueError(f"unsafe path in SciFact ZIP archive: {name!r}")


def _validate_zip_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    by_name: dict[str, zipfile.ZipInfo] = {}
    for member in archive.infolist():
        _validate_member_name(member.filename)
        if member.flag_bits & 0x1:
            raise ValueError(
                f"encrypted entry in SciFact ZIP archive: {member.filename!r}"
            )
        by_name[member.filename] = member

    missing = sorted(set(_REQUIRED_MEMBERS) - set(by_name))
    if missing:
        raise RuntimeError(
            "SciFact ZIP archive is missing required member(s): "
            + ", ".join(missing)
            + "; download the official archive or use the manual Hugging Face fallback"
        )
    return by_name


def extract_scifact_zip(
    archive: Path,
    destination: Path,
    *,
    max_total_bytes: int = MAX_EXTRACT_BYTES,
) -> None:
    """Selectively extract corpus/queries/test qrels, never arbitrary entries.

    Every ZIP entry name is validated first, traversal and absolute paths are
    rejected, and the declared plus actual uncompressed sizes are capped.  Only
    the three files named in :data:`_REQUIRED_MEMBERS` are written.
    """
    if max_total_bytes <= 0:
        raise ValueError("max_total_bytes must be positive")

    archive = Path(archive)
    destination = Path(destination)
    try:
        with zipfile.ZipFile(archive) as handle:
            by_name = _validate_zip_members(handle)
            members = {_REQUIRED_MEMBERS[name]: by_name[name] for name in _REQUIRED_MEMBERS}
            declared_total = sum(member.file_size for member in members.values())
            if declared_total > max_total_bytes:
                raise RuntimeError(
                    "SciFact ZIP archive expands beyond the extraction size limit: "
                    f"{declared_total} > {max_total_bytes} bytes"
                )

            destination.mkdir(parents=True, exist_ok=True)
            written = 0
            for relative, member in members.items():
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".part")
                partial.unlink(missing_ok=True)
                try:
                    with handle.open(member) as source, partial.open("wb") as sink:
                        while True:
                            chunk = source.read(_READ_BLOCK)
                            if not chunk:
                                break
                            written += len(chunk)
                            if written > max_total_bytes:
                                raise RuntimeError(
                                    "SciFact ZIP archive expands beyond the "
                                    "extraction size limit"
                                )
                            sink.write(chunk)
                    os.replace(partial, target)
                finally:
                    partial.unlink(missing_ok=True)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"invalid SciFact ZIP archive {archive}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in {path} at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(payload, dict):
                    # Public converter API reports malformed input as ValueError.
                    raise ValueError(  # noqa: TRY004
                        f"expected a JSON object in {path} at line {line_number}"
                    )
                rows.append(payload)
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    return rows


def _read_qrels(path: Path) -> dict[str, set[str]]:
    relevant: dict[str, set[str]] = {}
    try:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.rstrip("\r\n")
                if not line.strip():
                    continue
                fields = line.split("\t")
                if len(fields) != 3:
                    raise ValueError(
                        f"malformed qrels line in {path} at line {line_number}: "
                        "expected three tab-separated fields"
                    )
                query_id, doc_id, score_text = (field.strip() for field in fields)
                try:
                    score = float(score_text)
                except ValueError as exc:
                    if line_number == 1 and score_text.strip().lower() == "score":
                        continue  # Standard BEIR header row.
                    raise ValueError(
                        f"invalid qrels score in {path} at line {line_number}: "
                        f"{score_text!r}"
                    ) from exc
                if not query_id or not doc_id:
                    raise ValueError(
                        f"empty qrels id in {path} at line {line_number}"
                    )
                if score > 0:
                    relevant.setdefault(query_id, set()).add(doc_id)
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    return relevant


def _document_id(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("corpus '_id' must be a non-empty string")
    if (
        value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(
            f"corpus document id cannot be used as a safe file name: {value!r}"
        )
    return value


def _query_id(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("query '_id' must be a non-empty string")
    return value


def _corpus_rows(corpus_jsonl: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for payload in _read_jsonl(corpus_jsonl):
        doc_id = _document_id(payload.get("_id"))
        if doc_id in seen:
            raise ValueError(f"duplicate corpus document id: {doc_id!r}")
        seen.add(doc_id)
        title = payload.get("title", "")
        text = payload.get("text", "")
        title = "" if title is None else title
        text = "" if text is None else text
        if not isinstance(title, str) or not isinstance(text, str):
            # Public converter API reports malformed input as ValueError.
            raise ValueError(  # noqa: TRY004
                f"corpus document {doc_id!r} has non-string title/text"
            )
        rows.append((doc_id, title, text))
    return rows


def _query_rows(queries_jsonl: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for payload in _read_jsonl(queries_jsonl):
        query_id = _query_id(payload.get("_id", payload.get("id")))
        if query_id in seen:
            raise ValueError(f"duplicate query id: {query_id!r}")
        seen.add(query_id)
        text = payload.get("text", payload.get("query"))
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"query {query_id!r} has no text")
        rows.append((query_id, text))
    return rows


def _evaluation_path(doc_id: str) -> str:
    """Return the root-relative corpus path used by the evaluation harness."""
    return f"{doc_id}.txt"


def convert_beir(
    corpus_jsonl: Path,
    queries_jsonl: Path,
    qrels_tsv: Path,
    out_dir: Path,
) -> None:
    """Convert BEIR SciFact files into seekr's file-level evaluation format.

    Writes ``out_dir/corpus/<doc_id>.txt`` containing ``title + two newlines +
    text`` and ``out_dir/queries.jsonl`` with only queries that have at least
    one positive test-split qrel whose document exists in the corpus.  A zero
    qrel score is not relevance.  Relevance labels are root-relative
    ``<doc_id>.txt`` paths, matching ``evaluation.py``'s path contract.
    """
    corpus = _corpus_rows(Path(corpus_jsonl))
    queries = _query_rows(Path(queries_jsonl))
    qrels = _read_qrels(Path(qrels_tsv))
    corpus_ids = {doc_id for doc_id, _, _ in corpus}

    output_queries: list[dict[str, object]] = []
    for query_id, query_text in queries:
        relevant = sorted(
            _evaluation_path(doc_id)
            for doc_id in qrels.get(query_id, set()) & corpus_ids
        )
        if not relevant:
            continue
        output_queries.append(
            {"id": query_id, "query": query_text, "relevant": relevant}
        )

    out_dir = Path(out_dir)
    corpus_dir = out_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, title, text in corpus:
        (corpus_dir / f"{doc_id}.txt").write_text(
            f"{title}\n\n{text}",
            encoding="utf-8",
        )

    queries_path = out_dir / "queries.jsonl"
    with queries_path.open("w", encoding="utf-8") as handle:
        for payload in output_queries:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _source_paths(source_dir: Path) -> tuple[Path, Path, Path]:
    source_dir = Path(source_dir)
    return (
        source_dir / "corpus.jsonl",
        source_dir / "queries.jsonl",
        source_dir / "qrels" / "test.tsv",
    )


def prepare_scifact(
    data_dir: Path,
    *,
    allow_download: bool = True,
    timeout: float = 60.0,
) -> Path:
    """Ensure converted SciFact data exists, downloading only when allowed."""
    data_dir = Path(data_dir)
    archive = data_dir / _ARCHIVE_NAME
    source_dir = data_dir / _SOURCE_NAME
    output_dir = data_dir / _OUTPUT_NAME
    corpus, queries, qrels = _source_paths(source_dir)

    if not (corpus.is_file() and queries.is_file() and qrels.is_file()):
        if archive.exists():
            extract_scifact_zip(archive, source_dir)
        elif allow_download:
            download_scifact_archive(archive, timeout=timeout)
            extract_scifact_zip(archive, source_dir)
        else:
            missing = [
                str(path)
                for path in (corpus, queries, qrels)
                if not path.is_file()
            ]
            raise RuntimeError(
                _manual_fallback_message(
                    "downloads are disabled and source files are missing: "
                    + ", ".join(missing),
                    source_dir,
                )
            )

    convert_beir(corpus, queries, qrels, output_dir)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    """Download (opt-in), extract and convert SciFact for ``seekr eval``."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert the public BEIR SciFact dataset. Network access happens "
            "only when this script is explicitly run and no local archive/"
            "source is present; use --no-download for a fully offline run."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory (default: <repo>/eval/data, git-ignored).",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Never touch the network; require an existing archive/source tree.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTPS connect/read timeout in seconds (default: 60).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir if args.data_dir is not None else repo_root / "eval" / "data"
    try:
        output_dir = prepare_scifact(
            data_dir,
            allow_download=not args.no_download,
            timeout=args.timeout,
        )
    except (RuntimeError, ValueError, OSError, zipfile.BadZipFile) as exc:
        print(f"beir_scifact: {exc}", file=sys.stderr)
        return 1

    print(f"converted SciFact into {output_dir}")
    print(
        "run: seekr eval --root "
        f"{output_dir / 'corpus'} --queries {output_dir / 'queries.jsonl'} "
        "--out bench/scifact-results.md"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
