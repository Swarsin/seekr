"""Shared opt-in download, validation, and conversion helpers for BEIR datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_EXTRACT_BYTES = 512 * 1024 * 1024
_READ_BLOCK = 1024 * 1024


@dataclass(frozen=True)
class BeirDataset:
    """Description of one public BEIR dataset archive and output layout."""

    name: str
    label: str
    url: str
    hf_url: str
    script_name: str
    archive_prefix: str | None = None
    max_archive_bytes: int = MAX_ARCHIVE_BYTES
    max_extract_bytes: int = MAX_EXTRACT_BYTES

    @property
    def prefix(self) -> str:
        return self.archive_prefix or self.name

    @property
    def archive_name(self) -> str:
        return f"{self.name}.zip"

    @property
    def source_name(self) -> str:
        return f"{self.name}-source"

    @property
    def output_name(self) -> str:
        return self.name

    @property
    def required_members(self) -> dict[str, Path]:
        prefix = self.prefix
        return {
            f"{prefix}/corpus.jsonl": Path("corpus.jsonl"),
            f"{prefix}/queries.jsonl": Path("queries.jsonl"),
            f"{prefix}/qrels/test.tsv": Path("qrels") / "test.tsv",
        }


def manual_fallback_message(
    dataset: BeirDataset,
    reason: object,
    source_dir: Path | None = None,
) -> str:
    """Return an actionable manual Hugging Face fallback error message."""
    source = source_dir or Path(f"<eval/data>/{dataset.source_name}")
    return (
        f"cannot acquire the {dataset.label} dataset: {reason}. "
        "Manual fallback: open "
        f"{dataset.hf_url} , download corpus.jsonl, queries.jsonl and "
        "qrels/test.tsv, then place them at "
        f"{source / 'corpus.jsonl'}, {source / 'queries.jsonl'} and "
        f"{source / 'qrels' / 'test.tsv'} (or download the official {dataset.archive_name} "
        "from the trusted UKP URL and place it next to the source directory). "
        f"Re-run `python eval/{dataset.script_name} --no-download`; no network access "
        "is attempted with that flag."
    )


def _require_https(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"refusing non-HTTPS dataset URL: {url!r}")


def download_archive(
    dataset: BeirDataset,
    destination: Path,
    *,
    url: str | None = None,
    timeout: float = 60.0,
) -> Path:
    """Return a valid local BEIR archive, downloading it only when absent."""
    target_url = dataset.url if url is None else url
    destination = Path(destination)
    source_dir = destination.parent / dataset.source_name

    if destination.exists():
        if not destination.is_file():
            raise RuntimeError(f"expected a ZIP file at {destination}")
        if zipfile.is_zipfile(destination):
            return destination
        raise RuntimeError(
            manual_fallback_message(
                dataset,
                f"existing file {destination} is not a valid ZIP archive; "
                "remove or replace it",
                source_dir,
            )
        )

    _require_https(target_url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)

    request = urllib.request.Request(
        target_url,
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
                    if declared > dataset.max_archive_bytes:
                        raise RuntimeError(
                            manual_fallback_message(
                                dataset,
                                f"archive is larger than {dataset.max_archive_bytes} bytes",
                                source_dir,
                            )
                        )
                total = 0
                with partial.open("wb") as handle:
                    while True:
                        chunk = response.read(_READ_BLOCK)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > dataset.max_archive_bytes:
                            raise RuntimeError(
                                manual_fallback_message(
                                    dataset,
                                    "download exceeded the archive size limit",
                                    source_dir,
                                )
                            )
                        handle.write(chunk)
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            OSError,
        ) as exc:
            raise RuntimeError(
                manual_fallback_message(dataset, exc, source_dir)
            ) from exc

        if total == 0:
            raise RuntimeError(
                manual_fallback_message(
                    dataset, "downloaded archive was empty", source_dir
                )
            )
        if not zipfile.is_zipfile(partial):
            raise RuntimeError(
                manual_fallback_message(
                    dataset, "downloaded file is not a ZIP archive", source_dir
                )
            )
        os.replace(partial, destination)
        return destination
    finally:
        partial.unlink(missing_ok=True)


def _validate_member_name(dataset: BeirDataset, name: str) -> None:
    pure = PurePosixPath(name)
    if (
        not name
        or "\x00" in name
        or "\\" in name
        or pure.is_absolute()
        or ".." in pure.parts
    ):
        raise ValueError(f"unsafe path in {dataset.label} ZIP archive: {name!r}")


def _validate_zip_members(
    dataset: BeirDataset,
    archive: zipfile.ZipFile,
) -> dict[str, zipfile.ZipInfo]:
    by_name: dict[str, zipfile.ZipInfo] = {}
    for member in archive.infolist():
        _validate_member_name(dataset, member.filename)
        if member.flag_bits & 0x1:
            raise ValueError(
                f"encrypted entry in {dataset.label} ZIP archive: {member.filename!r}"
            )
        by_name[member.filename] = member

    required = dataset.required_members
    missing = sorted(set(required) - set(by_name))
    if missing:
        raise RuntimeError(
            f"{dataset.label} ZIP archive is missing required member(s): "
            + ", ".join(missing)
            + "; download the official archive or use the manual Hugging Face fallback"
        )
    return by_name


def extract_zip(
    dataset: BeirDataset,
    archive: Path,
    destination: Path,
    *,
    max_total_bytes: int | None = None,
) -> None:
    """Selectively extract corpus, queries, and test qrels after validating ZIP entries."""
    limit = dataset.max_extract_bytes if max_total_bytes is None else max_total_bytes
    if limit <= 0:
        raise ValueError("max_total_bytes must be positive")

    archive = Path(archive)
    destination = Path(destination)
    required = dataset.required_members
    try:
        with zipfile.ZipFile(archive) as handle:
            by_name = _validate_zip_members(dataset, handle)
            members = {relative: by_name[name] for name, relative in required.items()}
            declared_total = sum(member.file_size for member in members.values())
            if declared_total > limit:
                raise RuntimeError(
                    f"{dataset.label} ZIP archive expands beyond the extraction size "
                    f"limit: {declared_total} > {limit} bytes"
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
                            if written > limit:
                                raise RuntimeError(
                                    f"{dataset.label} ZIP archive expands beyond the "
                                    "extraction size limit"
                                )
                            sink.write(chunk)
                    os.replace(partial, target)
                finally:
                    partial.unlink(missing_ok=True)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"invalid {dataset.label} ZIP archive {archive}: {exc}") from exc


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
                        continue
                    raise ValueError(
                        f"invalid qrels score in {path} at line {line_number}: "
                        f"{score_text!r}"
                    ) from exc
                if not query_id or not doc_id:
                    raise ValueError(f"empty qrels id in {path} at line {line_number}")
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
    if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
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


def convert_beir(
    corpus_jsonl: Path,
    queries_jsonl: Path,
    qrels_tsv: Path,
    out_dir: Path,
) -> None:
    """Convert BEIR JSONL/TSV files into seekr's file-level evaluation layout."""
    corpus = _corpus_rows(Path(corpus_jsonl))
    queries = _query_rows(Path(queries_jsonl))
    qrels = _read_qrels(Path(qrels_tsv))
    corpus_ids = {doc_id for doc_id, _, _ in corpus}

    output_queries: list[dict[str, object]] = []
    for query_id, query_text in queries:
        relevant = sorted(
            f"{doc_id}.txt" for doc_id in qrels.get(query_id, set()) & corpus_ids
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
        body = f"{title}\n\n{text}" if title else text
        (corpus_dir / f"{doc_id}.txt").write_text(body, encoding="utf-8")

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


def prepare_dataset(
    dataset: BeirDataset,
    data_dir: Path,
    *,
    allow_download: bool = True,
    timeout: float = 60.0,
) -> Path:
    """Ensure converted BEIR dataset files exist, downloading only when allowed."""
    data_dir = Path(data_dir)
    archive = data_dir / dataset.archive_name
    source_dir = data_dir / dataset.source_name
    output_dir = data_dir / dataset.output_name
    corpus, queries, qrels = _source_paths(source_dir)

    if not (corpus.is_file() and queries.is_file() and qrels.is_file()):
        if archive.exists():
            extract_zip(dataset, archive, source_dir)
        elif allow_download:
            download_archive(dataset, archive, timeout=timeout)
            extract_zip(dataset, archive, source_dir)
        else:
            missing = [
                str(path)
                for path in (corpus, queries, qrels)
                if not path.is_file()
            ]
            raise RuntimeError(
                manual_fallback_message(
                    dataset,
                    "downloads are disabled and source files are missing: "
                    + ", ".join(missing),
                    source_dir,
                )
            )

    convert_beir(corpus, queries, qrels, output_dir)
    return output_dir


def run_cli(dataset: BeirDataset, argv: list[str] | None = None) -> int:
    """Download (opt-in), extract, and convert a BEIR dataset for ``seekr eval``."""
    parser = argparse.ArgumentParser(
        description=(
            f"Convert the public BEIR {dataset.label} dataset. Network access happens "
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
        output_dir = prepare_dataset(
            dataset,
            data_dir,
            allow_download=not args.no_download,
            timeout=args.timeout,
        )
    except (RuntimeError, ValueError, OSError, zipfile.BadZipFile) as exc:
        print(f"{Path(dataset.script_name).stem}: {exc}", file=sys.stderr)
        return 1

    print(f"converted {dataset.label} into {output_dir}")
    print(
        "run: seekr eval --root "
        f"{output_dir / 'corpus'} --queries {output_dir / 'queries.jsonl'} "
        f"--out bench/{dataset.name}-results.md"
    )
    return 0
