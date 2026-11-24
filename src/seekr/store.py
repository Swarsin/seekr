"""SQLite persistence for seekr documents, chunks, keyword search and vectors."""

from __future__ import annotations

import logging
import math
import sqlite3
import string
import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import pathnorm
from .logging_config import log_event
from .models import ChunkSpec

_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np

_DELETE_BATCH = 500
_FETCH_BATCH = 800
# sqlite-vec's vec0 KNN scan accepts k values up to this limit.
_VEC0_MAX_K = 4096
VECTOR_QUANTS = frozenset({"float32", "int8"})
SCHEMA_VERSION = 3

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    mtime_ns INTEGER NOT NULL,
    size INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    kind TEXT NOT NULL,
    module TEXT,
    week INTEGER
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    text_sha TEXT NOT NULL,
    locator TEXT NOT NULL,
    loc_start REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS chunks_text_sha_idx ON chunks(text_sha);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class ModelMismatchError(RuntimeError):
    """Raised when the index was built with a different embedding model or dimension."""


class SchemaVersionError(RuntimeError):
    """Raised when the database schema version is unsupported or migration fails."""


def validate_schema_version(rows: dict[str, str]) -> int:
    """Parse and validate ``schema_version`` from metadata rows (default 0)."""
    raw = rows.get("schema_version")
    if raw is None:
        return 0
    try:
        version = int(raw)
    except (TypeError, ValueError) as exc:
        raise SchemaVersionError(
            f"index database has invalid schema_version {raw!r}; "
            "run `seekr reindex` to rebuild it"
        ) from exc
    if version < 0:
        raise SchemaVersionError(
            f"index database has invalid negative schema_version {version}; "
            "run `seekr reindex` to rebuild it"
        )
    if version > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"index database uses schema version {version}, which is newer than "
            f"supported version {SCHEMA_VERSION}; upgrade seekr or run "
            "`seekr reindex` to rebuild it"
        )
    return version


def _migrate_0_to_1(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES ('vector_quant', 'float32')"
    )


def _migrate_1_to_2(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE INDEX IF NOT EXISTS documents_kind_mtime_idx ON documents(kind, mtime_ns)"
    )


def _create_module_week_index(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE INDEX IF NOT EXISTS documents_module_week_idx ON documents(module, week)"
    )


def _migrate_2_to_3(conn: sqlite3.Connection) -> None:
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(documents)")}
    if "module" not in columns:
        conn.execute("ALTER TABLE documents ADD COLUMN module TEXT")
    if "week" not in columns:
        conn.execute("ALTER TABLE documents ADD COLUMN week INTEGER")
    _create_module_week_index(conn)


_MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {
    0: _migrate_0_to_1,
    1: _migrate_1_to_2,
    2: _migrate_2_to_3,
}


@dataclass(frozen=True)
class StoredChunk:
    """A chunk, its content hash and the vector to persist for it."""

    spec: ChunkSpec
    text_sha: str
    vector: np.ndarray


@dataclass(frozen=True)
class DocRow:
    """A row from the documents table."""

    id: int
    path: str
    mtime_ns: int
    size: int
    sha256: str
    kind: str
    module: str | None = None
    week: int | None = None


@dataclass(frozen=True)
class ChunkRow:
    """A chunk joined with its owning document."""

    id: int
    doc_id: int
    path: str
    kind: str
    text: str
    locator: str
    loc_start: float
    module: str | None = None
    week: int | None = None


def _load_vector_extension(conn: sqlite3.Connection) -> None:
    import sqlite_vec

    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        try:
            conn.enable_load_extension(False)
        except (AttributeError, sqlite3.Error):
            pass


def _create_vector_table(conn: sqlite3.Connection, dim: int) -> None:
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(embedding float[{int(dim)}])"
    )


_DEFAULT_LOAD_VECTOR_EXTENSION = _load_vector_extension
_DEFAULT_CREATE_VECTOR_TABLE = _create_vector_table


def _int8_scale(dim: int) -> float:
    return float(min(500.0, max(127.0, 25.0 * math.sqrt(dim))))


def _create_vector_table_int8(conn: sqlite3.Connection, dim: int) -> None:
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(embedding int8[{int(dim)}])"
    )


def _serialize_vector(vector: object) -> bytes:
    import numpy as np

    return np.ascontiguousarray(vector, dtype="<f4").tobytes()


def _serialize_vector_int8(vector: object, dim: int) -> bytes:
    import numpy as np

    scale = np.float32(_int8_scale(dim))
    scaled = np.rint(np.asarray(vector, dtype=np.float32) * scale)
    clamped = np.clip(scaled, -127.0, 127.0).astype(np.int8)
    return np.ascontiguousarray(clamped).tobytes()


def _deserialize_vector(payload: bytes | memoryview) -> np.ndarray:
    import numpy as np

    return np.frombuffer(payload, dtype=np.float32).copy()


def _deserialize_vector_int8(payload: bytes | memoryview, dim: int) -> np.ndarray:
    import numpy as np

    scale = np.float32(_int8_scale(dim))
    return np.frombuffer(payload, dtype=np.int8).astype(np.float32) / scale


_ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)
_LIKE_ESCAPE = "!"


def _like_escape(value: str) -> str:
    return (
        value.replace(_LIKE_ESCAPE, _LIKE_ESCAPE * 2)
        .replace("%", _LIKE_ESCAPE + "%")
        .replace("_", _LIKE_ESCAPE + "_")
    )


def _path_prefix_clause(
    prefix: str | None,
    *,
    column: str = "path",
    case_insensitive: bool = False,
    separator: str = "/",
) -> tuple[str, tuple[object, ...]]:
    """Return a component-aware path-prefix SQL predicate and its parameters.

    The prefix matches itself or anything below it on a *separator* boundary,
    so ``C:\\notes`` never matches ``C:\\notes2``.  With ``\\`` as the
    separator a ``/`` in the prefix is read as ``\\``.  ``case_insensitive``
    compares ``lower(column)`` with a lowered, LIKE-escaped prefix; like
    SQLite's ``lower()`` it folds ASCII letters only.
    """
    if prefix is None:
        return "", ()
    value = str(prefix)
    if separator != "/":
        value = value.replace("/", separator)
    value = value.rstrip(separator)
    if not value:
        return "", ()
    if not case_insensitive:
        return (
            f"({column} = ? OR substr({column}, 1, ?) = ?)",
            (value, len(value) + 1, value + separator),
        )
    folded = value.translate(_ASCII_LOWER)
    return (
        f"(lower({column}) = ? OR lower({column}) LIKE ? ESCAPE '{_LIKE_ESCAPE}')",
        (folded, _like_escape(folded + separator) + "%"),
    )


def _document_filter_clause(
    *,
    path_prefix: str | None = None,
    kind: str | None = None,
    since_mtime_ns: int | None = None,
    exclude_path: str | None = None,
    module: str | None = None,
    week_range: tuple[int, int] | None = None,
    alias: str = "d",
    case_insensitive: bool = False,
    separator: str = "/",
) -> tuple[str, tuple[object, ...]]:
    """Combine document-level path, kind, mtime, label and exclusion SQL predicates."""
    clauses: list[str] = []
    params: list[object] = []
    prefix_sql, prefix_params = _path_prefix_clause(
        path_prefix,
        column=f"{alias}.path",
        case_insensitive=case_insensitive,
        separator=separator,
    )
    if prefix_sql:
        clauses.append(prefix_sql)
        params.extend(prefix_params)
    if kind is not None:
        clauses.append(f"{alias}.kind = ?")
        params.append(str(kind))
    if since_mtime_ns is not None:
        clauses.append(f"{alias}.mtime_ns >= ?")
        params.append(int(since_mtime_ns))
    if exclude_path is not None:
        clauses.append(f"{alias}.path != ?")
        params.append(str(exclude_path))
    if module is not None:
        clauses.append(f"{alias}.module = ?")
        params.append(str(module))
    if week_range is not None:
        start_week, end_week = week_range
        clauses.append(f"{alias}.week BETWEEN ? AND ?")
        params.extend((int(start_week), int(end_week)))
    return " AND ".join(clauses), tuple(params)


def _doc_row_from_sql(row: tuple[Any, ...]) -> DocRow:
    return DocRow(
        id=int(row[0]),
        path=str(row[1]),
        mtime_ns=int(row[2]),
        size=int(row[3]),
        sha256=str(row[4]),
        kind=str(row[5]),
        module=str(row[6]) if row[6] is not None else None,
        week=int(row[7]) if row[7] is not None else None,
    )


def _rollback(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ROLLBACK")
    except sqlite3.Error:
        pass


class Store:
    """One SQLite connection, shared by threads and guarded by an RLock.

    The vector table is a sqlite-vec ``vec0`` table.  There is deliberately no
    fallback backend: if the extension is missing, construction fails.
    """

    def __init__(
        self,
        db_path: Path,
        embed_model: str,
        dim: int,
        *,
        vector_quant: str | None = None,
        read_only: bool = False,
        embed_backend: str | None = None,
        case_insensitive_paths: bool | None = None,
    ) -> None:
        """Open (or create) the index.

        ``case_insensitive_paths`` makes path-prefix filters fold case; it
        defaults to :func:`seekr.pathnorm.case_insensitive` for this OS.
        """
        if not isinstance(embed_model, str) or not embed_model.strip():
            raise ValueError("embed_model must be a non-empty string")
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        if vector_quant is not None and (
            not isinstance(vector_quant, str) or vector_quant not in VECTOR_QUANTS
        ):
            choices = ", ".join(sorted(VECTOR_QUANTS))
            raise ValueError(f"vector_quant must be one of {choices}")

        if embed_backend is not None and (
            not isinstance(embed_backend, str) or not embed_backend.strip()
        ):
            raise ValueError("embed_backend must be a non-empty string")

        self._db_path = Path(db_path)
        self._embed_model = embed_model
        # Informational only: indexes are compatible across backends that run
        # the same model, so the recorded backend is never compared.
        self._embed_backend = embed_backend
        self._dim = dim
        self._read_only = bool(read_only)
        self._case_insensitive_paths = (
            pathnorm.case_insensitive()
            if case_insensitive_paths is None
            else bool(case_insensitive_paths)
        )
        self._path_separator = pathnorm.path_separator()
        self._stubbed_backend = (
            _load_vector_extension is not _DEFAULT_LOAD_VECTOR_EXTENSION
            or _create_vector_table is not _DEFAULT_CREATE_VECTOR_TABLE
        )
        self._requested_vector_quant = vector_quant
        self._vector_quant = "float32" if vector_quant is None else vector_quant
        self._schema_version = SCHEMA_VERSION
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._tx_depth = 0

        if self._read_only:
            self._open_read_only()
        else:
            self._open_writable(dim)

        log_event(
            _LOGGER,
            logging.DEBUG,
            "store.open",
            db_path=self._db_path,
            model=self._embed_model,
            dim=self._dim,
            quant=self._vector_quant,
            schema_version=self._schema_version,
            read_only=self._read_only,
        )

    @property
    def case_insensitive_paths(self) -> bool:
        """Whether path-prefix filters ignore (ASCII) letter case."""
        return self._case_insensitive_paths

    def _path_options(self) -> dict[str, Any]:
        return {
            "case_insensitive": self._case_insensitive_paths,
            "separator": self._path_separator,
        }

    @staticmethod
    def _canonical_prefix(prefix: str | None) -> str | None:
        """Bring a caller's prefix to the stored form (NFC on macOS)."""
        return None if prefix is None else pathnorm.normalize_unicode(str(prefix))

    def _prefix_clause(self, prefix: str | None) -> tuple[str, tuple[object, ...]]:
        return _path_prefix_clause(self._canonical_prefix(prefix), **self._path_options())

    def _open_writable(self, dim: int) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA cache_size=-32000")
            conn.execute("PRAGMA foreign_keys=ON")
            foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()
            if not foreign_keys or int(foreign_keys[0]) != 1:
                raise RuntimeError("SQLite foreign key enforcement could not be enabled")

            self._check_existing_schema_version(conn)
            conn.executescript(_SCHEMA_SQL)
            needs_metadata = self._check_metadata(conn)
            _load_vector_extension(conn)
            if self._use_int8_sql:
                _create_vector_table_int8(conn, dim)
            else:
                _create_vector_table(conn, dim)
            if needs_metadata:
                self._write_metadata(conn)
            if self._embed_backend is not None:
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('embed_backend', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (self._embed_backend,),
                )
        except BaseException:
            conn.close()
            raise
        self._conn = conn

    def _open_read_only(self) -> None:
        """Open an existing index without writing, migrating or loading vectors."""
        uri = f"{self._db_path.resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(
            uri,
            uri=True,
            timeout=2.0,
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            conn.execute("PRAGMA query_only=ON")
            conn.execute("PRAGMA busy_timeout=2000")
            self._validate_read_only_metadata(conn)
        except BaseException:
            conn.close()
            raise
        self._conn = conn

    def _validate_read_only_metadata(self, conn: sqlite3.Connection) -> None:
        """Validate schema/metadata with SELECTs only; never migrate."""
        names = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        required = {"documents", "chunks", "chunks_fts", "meta"}
        missing = sorted(required - names)
        if missing:
            raise sqlite3.OperationalError(
                "index database is missing required table(s): " + ", ".join(missing)
            )

        rows = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        stored_version = validate_schema_version(rows)
        if stored_version < SCHEMA_VERSION:
            raise SchemaVersionError(
                f"index database uses schema version {stored_version}, which is older "
                f"than supported version {SCHEMA_VERSION}; run `seekr index` once to "
                "upgrade it (read-only mode does not migrate)"
            )

        stored_model = rows.get("embed_model")
        stored_dim = rows.get("dim")
        if stored_model is None and stored_dim is None:
            if self._has_indexed_data(conn):
                raise ModelMismatchError(
                    "index database has no recorded embedding model; "
                    "run `seekr reindex` to rebuild it"
                )
        elif stored_model is None or stored_dim is None:
            raise ModelMismatchError(
                "index database has incomplete embedding metadata; "
                "run `seekr reindex` to rebuild it"
            )
        else:
            try:
                stored_dim_value = int(stored_dim)
            except (TypeError, ValueError):
                stored_dim_value = 0
            if stored_model != self._embed_model or stored_dim_value != self._dim:
                raise ModelMismatchError(
                    "index database was built with embedding model "
                    f"{stored_model!r} (dim {stored_dim}), but this configuration uses "
                    f"{self._embed_model!r} (dim {self._dim}); "
                    "run `seekr reindex` to rebuild it"
                )
            stored_quant = rows.get("vector_quant", "float32")
            if stored_quant not in VECTOR_QUANTS:
                raise ModelMismatchError(
                    f"index database has unsupported vector quantization {stored_quant!r}; "
                    "run `seekr reindex` to rebuild it"
                )
            if self._requested_vector_quant is None:
                self._vector_quant = stored_quant
            elif self._requested_vector_quant != stored_quant:
                raise ModelMismatchError(
                    f"index database uses vector quantization {stored_quant!r}, but "
                    f"this configuration uses {self._requested_vector_quant!r}; "
                    "run `seekr reindex` to rebuild it"
                )

        self._schema_version = stored_version

    @property
    def read_only(self) -> bool:
        """Return whether this Store was opened in read-only mode."""
        return self._read_only

    def _reject_write(self) -> None:
        if self._read_only:
            raise RuntimeError("store is read-only")

    @property
    def vector_quant(self) -> str:
        """Return the active vector quantization mode ('float32' or 'int8')."""
        return self._vector_quant

    @property
    def schema_version(self) -> int:
        """Return the database schema version."""
        return self._schema_version

    @property
    def _use_int8_sql(self) -> bool:
        return self._vector_quant == "int8" and not self._stubbed_backend

    def _serialize(self, vector: object) -> bytes:
        if self._use_int8_sql:
            return _serialize_vector_int8(vector, self._dim)
        return _serialize_vector(vector)

    def _deserialize(self, payload: bytes | memoryview) -> np.ndarray:
        if self._use_int8_sql:
            return _deserialize_vector_int8(payload, self._dim)
        return _deserialize_vector(payload)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the underlying connection; safe to call more than once."""
        with self._lock:
            conn = self._conn
            self._conn = None
            if conn is not None:
                conn.close()

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("store is closed")
        return self._conn

    # -- transactions ------------------------------------------------------

    @contextmanager
    def transaction(self, *, write: bool = False) -> Iterator[None]:
        """Run a block atomically, holding the store lock for a stable snapshot.

        Top-level transactions use ``BEGIN IMMEDIATE`` for writes and ``BEGIN``
        for reads.  Nested transactions use savepoints, so an inner failure can
        be rolled back without discarding successful outer work, and an outer
        failure discards everything.
        """
        if write:
            self._reject_write()
        with self._lock:
            conn = self._connection()
            depth = self._tx_depth
            savepoint = f"seekr_tx_{depth}"
            if depth == 0:
                conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            else:
                conn.execute(f"SAVEPOINT {savepoint}")
            self._tx_depth = depth + 1
            try:
                yield
            except BaseException:
                self._tx_depth = depth
                if depth == 0:
                    _rollback(conn)
                else:
                    try:
                        conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                    except sqlite3.Error:
                        _rollback(conn)
                raise
            else:
                self._tx_depth = depth
                if depth == 0:
                    try:
                        conn.execute("COMMIT")
                    except BaseException:
                        _rollback(conn)
                        raise
                else:
                    conn.execute(f"RELEASE SAVEPOINT {savepoint}")

    # -- schema / metadata -------------------------------------------------

    @staticmethod
    def _has_indexed_data(conn: sqlite3.Connection) -> bool:
        row = conn.execute(
            "SELECT EXISTS(SELECT 1 FROM documents) OR EXISTS(SELECT 1 FROM chunks)"
        ).fetchone()
        return bool(row[0]) if row else False

    @staticmethod
    def _check_existing_schema_version(conn: sqlite3.Connection) -> None:
        has_meta = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()
        if not has_meta:
            return
        rows = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        validate_schema_version(rows)

    @staticmethod
    def _migrate_schema(
        conn: sqlite3.Connection, current_version: int, target_version: int
    ) -> None:
        log_event(
            _LOGGER,
            logging.INFO,
            "store.migrate",
            from_version=current_version,
            to_version=target_version,
        )
        conn.execute("BEGIN IMMEDIATE")
        try:
            for version in range(current_version, target_version):
                step = _MIGRATIONS.get(version)
                if step is None:
                    raise SchemaVersionError(
                        f"no migration path from schema version {version} to "
                        f"{version + 1}; run `seekr reindex` to rebuild the index"
                    )
                step(conn)
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (str(version + 1),),
                )
            conn.execute("COMMIT")
        except SchemaVersionError:
            _rollback(conn)
            raise
        except (sqlite3.Error, RuntimeError, ValueError, TypeError) as exc:
            _rollback(conn)
            raise SchemaVersionError(
                f"cannot migrate index database from schema version {current_version} "
                f"to {target_version}: {exc}; run `seekr reindex` to rebuild it"
            ) from exc

    def _check_metadata(self, conn: sqlite3.Connection) -> bool:
        rows = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        stored_version = validate_schema_version(rows)
        stored_model = rows.get("embed_model")
        stored_dim = rows.get("dim")
        if stored_model is None and stored_dim is None:
            if self._has_indexed_data(conn):
                raise ModelMismatchError(
                    "index database has no recorded embedding model; "
                    "run `seekr reindex` to rebuild it"
                )
            return True
        if stored_model is None or stored_dim is None:
            raise ModelMismatchError(
                "index database has incomplete embedding metadata; "
                "run `seekr reindex` to rebuild it"
            )
        dim_matches = False
        try:
            dim_matches = int(stored_dim) == self._dim
        except (TypeError, ValueError):
            dim_matches = False
        if stored_model != self._embed_model or not dim_matches:
            raise ModelMismatchError(
                "index database was built with embedding model "
                f"{stored_model!r} (dim {stored_dim}), but this configuration uses "
                f"{self._embed_model!r} (dim {self._dim}); "
                "run `seekr reindex` to rebuild it"
            )
        stored_quant = rows.get("vector_quant", "float32")
        if stored_quant not in VECTOR_QUANTS:
            raise ModelMismatchError(
                f"index database has unsupported vector quantization {stored_quant!r}; "
                "run `seekr reindex` to rebuild it"
            )
        if self._requested_vector_quant is None:
            self._vector_quant = stored_quant
        elif self._requested_vector_quant != stored_quant:
            raise ModelMismatchError(
                f"index database uses vector quantization {stored_quant!r}, but "
                f"this configuration uses {self._requested_vector_quant!r}; "
                "run `seekr reindex` to rebuild it"
            )
        if stored_version < SCHEMA_VERSION:
            self._migrate_schema(conn, stored_version, SCHEMA_VERSION)
        self._schema_version = SCHEMA_VERSION
        return False

    def _write_metadata(self, conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.executemany(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (
                    ("embed_model", self._embed_model),
                    ("dim", str(self._dim)),
                    ("vector_quant", self._vector_quant),
                    ("schema_version", str(SCHEMA_VERSION)),
                ),
            )
            _migrate_1_to_2(conn)
            _create_module_week_index(conn)
            conn.execute("COMMIT")
        except BaseException:
            _rollback(conn)
            raise
        self._schema_version = SCHEMA_VERSION

    # -- documents ---------------------------------------------------------

    def upsert_document(
        self,
        path: str,
        mtime_ns: int,
        size: int,
        sha256: str,
        kind: str,
        *,
        module: str | None = None,
        week: int | None = None,
    ) -> int:
        """Insert a document or update its metadata, returning its stable id."""
        self._reject_write()
        module_val = None if module is None else str(module)
        week_val = None if week is None else int(week)
        with self._lock:
            conn = self._connection()
            with self.transaction(write=True):
                conn.execute(
                    """
                    INSERT INTO documents(path, mtime_ns, size, sha256, kind, module, week)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        mtime_ns = excluded.mtime_ns,
                        size = excluded.size,
                        sha256 = excluded.sha256,
                        kind = excluded.kind,
                        module = excluded.module,
                        week = excluded.week
                    """,
                    (
                        str(path),
                        int(mtime_ns),
                        int(size),
                        sha256,
                        kind,
                        module_val,
                        week_val,
                    ),
                )
                row = conn.execute("SELECT id FROM documents WHERE path = ?", (str(path),))
                found = row.fetchone()
            if found is None:
                raise RuntimeError("document disappeared during upsert")
            return int(found[0])

    def set_document_labels(
        self, labels: Iterable[tuple[str, str | None, int | None]]
    ) -> int:
        """Update module and week labels for existing documents in one write transaction."""
        self._reject_write()
        items = [
            (
                None if module is None else str(module),
                None if week is None else int(week),
                str(path),
            )
            for path, module, week in labels
        ]
        if not items:
            with self._lock:
                self._connection()
            return 0
        updated = 0
        with self._lock, self.transaction(write=True):
            conn = self._connection()
            for module_val, week_val, path_val in items:
                cursor = conn.execute(
                    "UPDATE documents SET module = ?, week = ? WHERE path = ?",
                    (module_val, week_val, path_val),
                )
                if cursor.rowcount > 0:
                    updated += int(cursor.rowcount)
        return updated

    def get_document(self, path: str) -> DocRow | None:
        """Return the document stored at ``path``, if any."""
        with self._lock:
            conn = self._connection()
            row = conn.execute(
                "SELECT id, path, mtime_ns, size, sha256, kind, module, week "
                "FROM documents WHERE path = ?",
                (str(path),),
            ).fetchone()
        if row is None:
            return None
        return _doc_row_from_sql(row)

    def get_documents_metadata(self, prefix: str | None = None) -> dict[str, DocRow]:
        """Return indexed document rows keyed by path, optionally under *prefix*."""
        clause, params = self._prefix_clause(prefix)
        sql = "SELECT id, path, mtime_ns, size, sha256, kind, module, week FROM documents"
        if clause:
            sql += f" WHERE {clause}"
        with self._lock:
            conn = self._connection()
            rows = conn.execute(sql, params).fetchall()
        docs: dict[str, DocRow] = {}
        for row in rows:
            doc = _doc_row_from_sql(row)
            docs[doc.path] = doc
        return docs

    def move_document(self, src: str, dst: str, *, mtime_ns: int, size: int) -> int:
        """Move an unchanged document to a new path and update its stat fields.

        Chunk ids, vectors and FTS rows are preserved.  The caller is expected
        to have verified that the file content is unchanged.  A missing source
        raises ``KeyError``; an already indexed destination raises ``ValueError``.
        """
        self._reject_write()
        source = str(src)
        destination = str(dst)
        with self._lock:
            conn = self._connection()
            with self.transaction(write=True):
                row = conn.execute(
                    "SELECT id FROM documents WHERE path = ?", (source,)
                ).fetchone()
                if row is None:
                    raise KeyError(f"document not found: {source}")
                doc_id = int(row[0])
                if source != destination:
                    existing = conn.execute(
                        "SELECT id FROM documents WHERE path = ?", (destination,)
                    ).fetchone()
                    if existing is not None:
                        raise ValueError(f"destination is already indexed: {destination}")
                conn.execute(
                    "UPDATE documents SET path = ?, mtime_ns = ?, size = ? WHERE id = ?",
                    (destination, int(mtime_ns), int(size), doc_id),
                )
            return doc_id

    def replace_chunks(self, doc_id: int, chunks: list[StoredChunk]) -> None:
        """Atomically replace every chunk, FTS row and vector for a document."""
        self._reject_write()
        items = list(chunks)
        with self._lock:
            conn = self._connection()
            with self.transaction(write=True):
                old_ids = [
                    int(row[0])
                    for row in conn.execute(
                        "SELECT id FROM chunks WHERE doc_id = ? ORDER BY id", (int(doc_id),)
                    )
                ]
                if old_ids:
                    self._delete_vectors(conn, old_ids)
                    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (int(doc_id),))
                if not items:
                    return
                new_ids: list[int] = []
                for chunk in items:
                    cursor = conn.execute(
                        """
                        INSERT INTO chunks(doc_id, text, text_sha, locator, loc_start)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            int(doc_id),
                            chunk.spec.text,
                            chunk.text_sha,
                            chunk.spec.locator,
                            float(chunk.spec.loc_start),
                        ),
                    )
                    new_ids.append(int(cursor.lastrowid))
                value_expr = "vec_int8(?)" if self._use_int8_sql else "?"
                conn.executemany(
                    f"INSERT INTO chunks_vec(rowid, embedding) VALUES (?, {value_expr})",
                    [
                        (chunk_id, self._serialize(chunk.vector))
                        for chunk_id, chunk in zip(new_ids, items)
                    ],
                )

    def delete_document(self, path: str) -> None:
        """Delete a document and explicitly purge its chunks, FTS and vector rows."""
        self._reject_write()
        with self._lock:
            conn = self._connection()
            with self.transaction(write=True):
                row = conn.execute(
                    "SELECT id FROM documents WHERE path = ?", (str(path),)
                ).fetchone()
                if row is None:
                    return
                doc_id = int(row[0])
                old_ids = [
                    int(chunk_row[0])
                    for chunk_row in conn.execute(
                        "SELECT id FROM chunks WHERE doc_id = ? ORDER BY id", (doc_id,)
                    )
                ]
                self._delete_vectors(conn, old_ids)
                conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def delete_documents_under(self, prefix: str) -> int:
        """Delete documents under *prefix* along with their chunks, FTS and vectors."""
        self._reject_write()
        clause, params = self._prefix_clause(prefix)
        if not clause:
            with self._lock:
                self._connection()
            return 0
        with self._lock, self.transaction(write=True):
            conn = self._connection()
            doc_ids = [
                int(row[0])
                for row in conn.execute(f"SELECT id FROM documents WHERE {clause}", params)
            ]
            if not doc_ids:
                return 0
            chunk_ids: list[int] = []
            for start in range(0, len(doc_ids), _FETCH_BATCH):
                batch = doc_ids[start : start + _FETCH_BATCH]
                placeholders = ", ".join("?" for _ in batch)
                chunk_ids.extend(
                    int(row[0])
                    for row in conn.execute(
                        f"SELECT id FROM chunks WHERE doc_id IN ({placeholders}) ORDER BY id",
                        batch,
                    )
                )
            if chunk_ids:
                self._delete_vectors(conn, chunk_ids)
            for start in range(0, len(doc_ids), _DELETE_BATCH):
                batch = doc_ids[start : start + _DELETE_BATCH]
                placeholders = ", ".join("?" for _ in batch)
                conn.execute(
                    f"DELETE FROM chunks WHERE doc_id IN ({placeholders})",
                    batch,
                )
                conn.execute(
                    f"DELETE FROM documents WHERE id IN ({placeholders})",
                    batch,
                )
        return len(doc_ids)

    @staticmethod
    def _delete_vectors(conn: sqlite3.Connection, chunk_ids: list[int]) -> None:
        for start in range(0, len(chunk_ids), _DELETE_BATCH):
            batch = chunk_ids[start : start + _DELETE_BATCH]
            conn.executemany(
                "DELETE FROM chunks_vec WHERE rowid = ?",
                [(chunk_id,) for chunk_id in batch],
            )

    def list_paths(self, prefix: str | None = None) -> list[str]:
        """Return paths in stable order, optionally below a path component prefix."""
        with self._lock:
            conn = self._connection()
            clause, params = self._prefix_clause(prefix)
            sql = "SELECT path FROM documents"
            if clause:
                sql += f" WHERE {clause}"
            sql += " ORDER BY path"
            rows = conn.execute(sql, params).fetchall()
        return [str(row[0]) for row in rows]

    def list_modules(self) -> list[str]:
        """Return distinct non-null module labels sorted case-insensitively."""
        with self._lock:
            conn = self._connection()
            rows = conn.execute(
                "SELECT DISTINCT module FROM documents WHERE module IS NOT NULL"
            ).fetchall()
        modules = [str(row[0]) for row in rows]
        modules.sort(key=lambda name: (name.casefold(), name))
        return modules

    def course_summary(self) -> list[tuple[str | None, int | None, int]]:
        """Return ``(module, week, file_count)`` sorted by module then week, NULLs last."""
        with self._lock:
            conn = self._connection()
            rows = conn.execute(
                "SELECT module, week, COUNT(*) FROM documents GROUP BY module, week"
            ).fetchall()
        summary = [
            (
                str(row[0]) if row[0] is not None else None,
                int(row[1]) if row[1] is not None else None,
                int(row[2]),
            )
            for row in rows
        ]
        summary.sort(
            key=lambda item: (
                item[0] is None,
                (item[0] or "").casefold(),
                item[0] or "",
                item[1] is None,
                item[1] if item[1] is not None else 0,
            )
        )
        return summary

    # -- search primitives -------------------------------------------------

    def find_vector_by_text_sha(self, sha: str) -> np.ndarray | None:
        """Return a stored vector for a known content hash, or ``None``."""
        with self._lock:
            conn = self._connection()
            row = conn.execute(
                "SELECT id FROM chunks WHERE text_sha = ? ORDER BY id LIMIT 1", (sha,)
            ).fetchone()
            if row is None:
                return None
            vector_row = conn.execute(
                "SELECT embedding FROM chunks_vec WHERE rowid = ?", (int(row[0]),)
            ).fetchone()
        if vector_row is None or vector_row[0] is None:
            return None
        return self._deserialize(vector_row[0])

    def find_vectors_by_text_shas(self, shas: Iterable[str]) -> dict[str, np.ndarray]:
        """Return stored vectors keyed by content hash for any known chunks."""
        unique_shas = list(dict.fromkeys(str(sha) for sha in shas))
        vectors: dict[str, np.ndarray] = {}
        if not unique_shas:
            return vectors
        with self._lock:
            conn = self._connection()
            for start in range(0, len(unique_shas), _FETCH_BATCH):
                batch = unique_shas[start : start + _FETCH_BATCH]
                placeholders = ", ".join("?" for _ in batch)
                rows = conn.execute(
                    f"""
                    SELECT c.text_sha, v.embedding
                    FROM chunks AS c
                    JOIN chunks_vec AS v ON v.rowid = c.id
                    WHERE c.text_sha IN ({placeholders})
                    ORDER BY c.id
                    """,
                    batch,
                ).fetchall()
                for row in rows:
                    sha = str(row[0])
                    if sha not in vectors and row[1] is not None:
                        vectors[sha] = self._deserialize(row[1])
        return vectors

    def keyword_search(
        self,
        match_query: str,
        limit: int,
        *,
        path_prefix: str | None = None,
        kind: str | None = None,
        since_mtime_ns: int | None = None,
        module: str | None = None,
        week_range: tuple[int, int] | None = None,
    ) -> list[tuple[int, float]]:
        """Return ``(chunk_id, bm25)`` pairs, best (lowest bm25) first.

        Document filters (``path_prefix``, ``kind``, ``since_mtime_ns``,
        ``module``, ``week_range``) are applied before the SQL limit, so
        returned rows are the best matching hits rather than global hits
        filtered afterwards.
        """
        if limit <= 0 or not isinstance(match_query, str) or not match_query.strip():
            return []
        clause, filter_params = _document_filter_clause(
            path_prefix=self._canonical_prefix(path_prefix),
            kind=kind,
            since_mtime_ns=since_mtime_ns,
            module=module,
            week_range=week_range,
            alias="d",
            **self._path_options(),
        )
        with self._lock:
            conn = self._connection()
            try:
                if clause:
                    rows = conn.execute(
                        f"""
                        SELECT chunks_fts.rowid, bm25(chunks_fts) AS score
                        FROM chunks_fts
                        JOIN chunks AS c ON c.id = chunks_fts.rowid
                        JOIN documents AS d ON d.id = c.doc_id
                        WHERE chunks_fts MATCH ? AND {clause}
                        ORDER BY score ASC, chunks_fts.rowid ASC
                        LIMIT ?
                        """,
                        (match_query, *filter_params, int(limit)),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT rowid, bm25(chunks_fts) AS score
                        FROM chunks_fts
                        WHERE chunks_fts MATCH ?
                        ORDER BY score ASC, rowid ASC
                        LIMIT ?
                        """,
                        (match_query, int(limit)),
                    ).fetchall()
            except sqlite3.OperationalError as exc:
                message = str(exc).lower()
                query_error_markers = (
                    "fts5",
                    "syntax",
                    "unterminated string",
                    "no such column:",
                    "unknown special query",
                    "malformed match",
                )
                if any(marker in message for marker in query_error_markers):
                    return []
                raise
        return [(int(row[0]), float(row[1])) for row in rows]

    def _vector_scalar_search(
        self,
        conn: sqlite3.Connection,
        blob: bytes,
        limit: int,
        path_prefix: str | None,
        *,
        kind: str | None = None,
        since_mtime_ns: int | None = None,
        exclude_path: str | None = None,
        module: str | None = None,
        week_range: tuple[int, int] | None = None,
        value_expr: str = "?",
    ) -> list[tuple[object, ...]]:
        """Exact vector search using sqlite-vec's scalar L2 distance."""
        clause, filter_params = _document_filter_clause(
            path_prefix=self._canonical_prefix(path_prefix),
            kind=kind,
            since_mtime_ns=since_mtime_ns,
            exclude_path=exclude_path,
            module=module,
            week_range=week_range,
            alias="d",
            **self._path_options(),
        )
        if clause:
            return conn.execute(
                f"""
                SELECT v.rowid, vec_distance_L2(v.embedding, {value_expr}) AS distance
                FROM chunks_vec AS v
                JOIN chunks AS c ON c.id = v.rowid
                JOIN documents AS d ON d.id = c.doc_id
                WHERE {clause}
                ORDER BY distance ASC, v.rowid ASC
                LIMIT ?
                """,
                (blob, *filter_params, limit),
            ).fetchall()
        return conn.execute(
            f"""
            SELECT v.rowid, vec_distance_L2(v.embedding, {value_expr}) AS distance
            FROM chunks_vec AS v
            JOIN chunks AS c ON c.id = v.rowid
            ORDER BY distance ASC, v.rowid ASC
            LIMIT ?
            """,
            (blob, limit),
        ).fetchall()

    def vector_search(
        self,
        vec: np.ndarray,
        limit: int,
        *,
        path_prefix: str | None = None,
        kind: str | None = None,
        since_mtime_ns: int | None = None,
        exclude_path: str | None = None,
        module: str | None = None,
        week_range: tuple[int, int] | None = None,
    ) -> list[tuple[int, float]]:
        """Return ``(chunk_id, distance)`` pairs, nearest first.

        The canonical unfiltered path uses a vec0 KNN ``MATCH`` query.  Any
        document filter, or a limit above sqlite-vec's KNN cap, uses the real
        ``vec_distance_L2`` scalar over the joined chunk/document rows so the
        filter is applied before top-k selection.
        """
        if self._read_only:
            raise RuntimeError("store is read-only: vector search unavailable")
        if limit <= 0:
            return []
        import numpy as np

        array = np.asarray(vec, dtype=np.float32)
        if array.ndim != 1 or array.shape[0] != self._dim:
            raise ValueError(f"expected a {self._dim}-dimensional vector, got {array.shape}")
        blob = self._serialize(array)
        value_expr = "vec_int8(?)" if self._use_int8_sql else "?"
        has_filter = (
            path_prefix is not None
            or kind is not None
            or since_mtime_ns is not None
            or exclude_path is not None
            or module is not None
            or week_range is not None
        )
        with self._lock:
            conn = self._connection()
            if has_filter or int(limit) > _VEC0_MAX_K:
                rows = self._vector_scalar_search(
                    conn,
                    blob,
                    int(limit),
                    path_prefix,
                    kind=kind,
                    since_mtime_ns=since_mtime_ns,
                    exclude_path=exclude_path,
                    module=module,
                    week_range=week_range,
                    value_expr=value_expr,
                )
            else:
                rows = conn.execute(
                    f"""
                    SELECT rowid, distance
                    FROM chunks_vec
                    WHERE embedding MATCH {value_expr} AND k = ?
                    ORDER BY distance
                    """,
                    (blob, int(limit)),
                ).fetchall()
        pairs = [(int(row[0]), float(row[1])) for row in rows]
        pairs.sort(key=lambda item: (item[1], item[0]))
        return pairs

    def get_document_vectors(self, path: str) -> list[np.ndarray]:
        """Return stored chunk vectors for ``path`` ordered by chunk id."""
        with self._lock:
            conn = self._connection()
            rows = conn.execute(
                """
                SELECT v.embedding
                FROM chunks AS c
                JOIN documents AS d ON d.id = c.doc_id
                JOIN chunks_vec AS v ON v.rowid = c.id
                WHERE d.path = ?
                ORDER BY c.id
                """,
                (str(path),),
            ).fetchall()
        return [self._deserialize(row[0]) for row in rows if row[0] is not None]

    def fetch_chunks(self, ids: list[int]) -> dict[int, ChunkRow]:
        """Fetch chunk rows in stable id order, ignoring unknown ids."""
        unique_ids = sorted({int(chunk_id) for chunk_id in ids})
        chunks: dict[int, ChunkRow] = {}
        if not unique_ids:
            return chunks
        with self._lock:
            conn = self._connection()
            for start in range(0, len(unique_ids), _FETCH_BATCH):
                batch = unique_ids[start : start + _FETCH_BATCH]
                placeholders = ", ".join("?" for _ in batch)
                rows = conn.execute(
                    f"""
                    SELECT
                        c.id, c.doc_id, d.path, d.kind, c.text, c.locator, c.loc_start,
                        d.module, d.week
                    FROM chunks AS c
                    JOIN documents AS d ON d.id = c.doc_id
                    WHERE c.id IN ({placeholders})
                    ORDER BY c.id
                    """,
                    batch,
                ).fetchall()
                for row in rows:
                    chunk_id = int(row[0])
                    chunks[chunk_id] = ChunkRow(
                        id=chunk_id,
                        doc_id=int(row[1]),
                        path=str(row[2]),
                        kind=str(row[3]),
                        text=str(row[4]),
                        locator=str(row[5]),
                        loc_start=float(row[6]),
                        module=str(row[7]) if row[7] is not None else None,
                        week=int(row[8]) if row[8] is not None else None,
                    )
        return chunks

    # -- status ------------------------------------------------------------

    def has_chunks(self) -> bool:
        """Return whether the store currently holds at least one chunk."""
        with self._lock:
            conn = self._connection()
            row = conn.execute("SELECT EXISTS(SELECT 1 FROM chunks)").fetchone()
        return bool(row[0]) if row else False

    def stats(self) -> dict[str, int]:
        """Return the document and chunk counts; no other keys are included."""
        with self._lock:
            conn = self._connection()
            documents = int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
            chunks = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        return {"documents": documents, "chunks": chunks}

    def detailed_stats(self) -> dict[str, Any]:
        """Return document/chunk counts, per-kind counts and store metadata."""
        with self._lock:
            conn = self._connection()
            documents = int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
            chunks = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            kind_rows = conn.execute(
                "SELECT kind, COUNT(*) FROM documents GROUP BY kind ORDER BY kind"
            ).fetchall()
        return {
            "documents": documents,
            "chunks": chunks,
            "kinds": {str(row[0]): int(row[1]) for row in kind_rows},
            "embed_model": self._embed_model,
            "dim": self._dim,
            "vector_quant": self._vector_quant,
            "schema_version": self._schema_version,
        }

    def get_meta_value(self, key: str) -> str | None:
        """Return a metadata value from the ``meta`` table, or ``None``."""
        with self._lock:
            conn = self._connection()
            row = conn.execute(
                "SELECT value FROM meta WHERE key = ?",
                (str(key),),
            ).fetchone()
        if row is None:
            return None
        return str(row[0])

    def set_meta_value(self, key: str, value: str) -> None:
        """Insert or update a metadata key/value pair in the ``meta`` table."""
        self._reject_write()
        with self._lock, self.transaction(write=True):
            conn = self._connection()
            conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(key), str(value)),
            )

    def set_last_error_count(self, n: int) -> None:
        """Persist the number of errors from the most recent indexing run."""
        self._reject_write()
        count = int(n)
        if count < 0:
            raise ValueError("last error count must be non-negative")
        with self._lock:
            conn = self._connection()
            with self.transaction(write=True):
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('last_error_count', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (str(count),),
                )

    def get_last_error_count(self) -> int:
        """Return the persisted last indexing error count (default 0)."""
        with self._lock:
            conn = self._connection()
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'last_error_count'"
            ).fetchone()
        if row is None:
            return 0
        try:
            return int(row[0])
        except (TypeError, ValueError):
            return 0
