"""
sqlite_store.py — SQLite-backed enrollment database.

Drop-in alternative to `FaceDatabase` (the JSON store) exposing the same
interface, for when more than one process serves the application.

Why it exists
-------------
The JSON store holds the whole database in memory and rewrites the file
wholesale on every change. That is correct for a single process and silently
destructive with two:

    worker A enrols Alice   -> writes {Alice}
    worker B enrols Bob     -> writes {Bob}, from its own stale snapshot
    on disk                 -> {Bob}.  Alice is gone, with no error anywhere.

Gunicorn defaults to several workers, so any real deployment hits this on the
second request. SQLite fixes it properly: one writer at a time, enforced by the
database rather than by hoping, and every read sees committed state instead of
a snapshot taken at start-up.

It is deliberately SQLite and not Postgres or a vector database. Matching 9,000
enrolled people takes ~21 ms against ~200 ms to embed the query face, so search
is not the bottleneck and an index would optimise the wrong 10%. SQLite adds
transactions and multi-process safety without adding a service to run.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 128
_BYTES_PER_EMBEDDING = EMBEDDING_DIM * 8   # float64

SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    name        TEXT PRIMARY KEY,
    enrolled_at TEXT NOT NULL,
    engine      TEXT NOT NULL DEFAULT 'dlib'
);

CREATE TABLE IF NOT EXISTS embeddings (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name   TEXT NOT NULL REFERENCES persons(name) ON DELETE CASCADE,
    vector BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_embeddings_name ON embeddings(name);
"""


class SqliteFaceDatabase:
    """Enrollment store backed by SQLite. Interface-compatible with FaceDatabase."""

    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        # Connections are per-thread: SQLite objects are not shareable across
        # threads, and Flask serves requests on several.
        self._local = threading.local()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")    # readers don't block the writer
            conn.execute("PRAGMA foreign_keys=ON")     # make ON DELETE CASCADE real
            conn.execute("PRAGMA busy_timeout=30000")  # wait rather than fail under contention
            self._local.conn = conn
        return conn

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def enroll(self, name: str, embedding: np.ndarray, engine: str = "dlib") -> None:
        self.enroll_batch(name, [embedding], engine=engine)

    def enroll_batch(self, name: str, embeddings: list[np.ndarray], engine: str = "dlib") -> None:
        name = (name or "").strip()
        if not name:
            raise ValueError("Person name must not be empty.")
        if not embeddings:
            return
        for embedding in embeddings:
            self._validate(embedding)

        conn = self._connect()
        conn.execute("BEGIN IMMEDIATE")     # take the write lock up front
        try:
            row = conn.execute("SELECT engine FROM persons WHERE name = ?", (name,)).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO persons (name, enrolled_at, engine) VALUES (?, ?, ?)",
                    (name, datetime.now().isoformat(), engine),
                )
            elif row[0] != engine:
                raise ValueError(
                    f"'{name}' was enrolled with the '{row[0]}' engine; refusing to add "
                    f"'{engine}' embeddings. Remove the person first, or switch back."
                )

            conn.executemany(
                "INSERT INTO embeddings (name, vector) VALUES (?, ?)",
                [(name, np.asarray(e, dtype=np.float64).tobytes()) for e in embeddings],
            )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

        logger.info("Enrolled %d embedding(s) for '%s'", len(embeddings), name)

    def remove(self, name: str) -> bool:
        conn = self._connect()
        cursor = conn.execute("DELETE FROM persons WHERE name = ?", (name,))
        if cursor.rowcount:
            logger.info("Removed '%s' from database.", name)
            return True
        return False

    def list_enrolled(self) -> list[str]:
        return [r[0] for r in self._connect().execute(
            "SELECT name FROM persons ORDER BY name"
        )]

    def find_name(self, name: str) -> str | None:
        """Stored spelling of *name*, matched case-insensitively."""
        target = (name or "").strip().casefold()
        for (existing,) in self._connect().execute("SELECT name FROM persons"):
            if existing.casefold() == target:
                return existing
        return None

    def get_embeddings(self, name: str) -> list[np.ndarray]:
        return [
            self._decode(blob) for (blob,) in self._connect().execute(
                "SELECT vector FROM embeddings WHERE name = ? ORDER BY id", (name,)
            )
        ]

    def get_all_embeddings(self) -> dict[str, list[np.ndarray]]:
        out: dict[str, list[np.ndarray]] = {}
        for name, blob in self._connect().execute(
            "SELECT name, vector FROM embeddings ORDER BY name, id"
        ):
            out.setdefault(name, []).append(self._decode(blob))
        # Include people whose embeddings were all removed, for parity with JSON.
        for name in self.list_enrolled():
            out.setdefault(name, [])
        return out

    def get_info(self, name: str) -> dict | None:
        row = self._connect().execute(
            "SELECT enrolled_at, engine, "
            "(SELECT COUNT(*) FROM embeddings WHERE name = persons.name) "
            "FROM persons WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        return {"enrolled_at": row[0], "engine": row[1], "num_images": row[2]}

    def stats(self) -> dict:
        conn = self._connect()
        counts = {
            name: count for name, count in conn.execute(
                "SELECT p.name, COUNT(e.id) FROM persons p "
                "LEFT JOIN embeddings e ON e.name = p.name GROUP BY p.name"
            )
        }
        engines = sorted({r[0] for r in conn.execute("SELECT DISTINCT engine FROM persons")})
        return {
            "total_persons": len(counts),
            "total_embeddings": sum(counts.values()),
            "persons": counts,
            "engines": engines,
        }

    def engines_in_use(self) -> set[str]:
        return {r[0] for r in self._connect().execute("SELECT DISTINCT engine FROM persons")}

    def clear(self) -> None:
        conn = self._connect()
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("DELETE FROM embeddings")
            conn.execute("DELETE FROM persons")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        logger.warning("Database cleared.")

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def import_json(self, json_path: str) -> int:
        """Copy an existing JSON store into this database. Returns people imported."""
        from src.database import FaceDatabase

        if not os.path.exists(json_path):
            return 0
        source = FaceDatabase(json_path)
        imported = 0
        for name in source.list_enrolled():
            vectors = source.get_embeddings(name)
            if not vectors:
                continue
            info = source.get_info(name) or {}
            self.enroll_batch(name, vectors, engine=info.get("engine", "dlib"))
            imported += 1
        logger.info("Imported %d person(s) from %s", imported, json_path)
        return imported

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(embedding: np.ndarray) -> None:
        arr = np.asarray(embedding)
        if arr.shape != (EMBEDDING_DIM,):
            raise ValueError(f"Embedding must have shape ({EMBEDDING_DIM},), got {arr.shape}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Embedding contains NaN or infinity.")

    @staticmethod
    def _decode(blob: bytes) -> np.ndarray:
        if len(blob) != _BYTES_PER_EMBEDDING:
            raise ValueError(
                f"Corrupt embedding: expected {_BYTES_PER_EMBEDDING} bytes, got {len(blob)}."
            )
        # frombuffer aliases immutable bytes, so the result would be read-only.
        return np.frombuffer(blob, dtype=np.float64).copy()


def open_database(db_path: str):
    """
    Return the right store for a path: SQLite for .db/.sqlite, JSON otherwise.

    Lets deployment select a multi-process-safe backend by filename alone,
    without every caller growing a backend argument.
    """
    from src.database import FaceDatabase

    if os.path.splitext(db_path)[1].lower() in (".db", ".sqlite", ".sqlite3"):
        return SqliteFaceDatabase(db_path)
    return FaceDatabase(db_path)
