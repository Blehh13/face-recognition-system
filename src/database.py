"""
database.py — Persistent face enrollment database.

Stores name → list-of-embeddings mappings in a JSON file (base64-encoded
numpy arrays). No external DB required — zero cost, zero setup.

Concurrency: the Flask app serves requests on multiple threads, so every
read and write goes through a re-entrant lock and writes land atomically
(temp file + os.replace). Without both, two simultaneous enrolments could
interleave and leave a truncated, unparseable JSON file on disk.
"""

import json
import os
import base64
import threading
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# The JSON store remains for inspection and for single-process use; the
# default path now ends in .sqlite so src.sqlite_store.open_database selects
# the multi-process-safe backend. A JSON path still works and still gets this
# class.
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "enrolled_faces.json")
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "enrolled_faces.sqlite")

EMBEDDING_DIM = 128          # dlib ResNet output size
_BYTES_PER_EMBEDDING = EMBEDDING_DIM * 8   # float64


class FaceDatabase:
    """
    Key-value store for enrolled face embeddings.

    Schema (JSON):
    {
        "person_name": {
            "embeddings": ["<base64-encoded float64 array>", ...],
            "enrolled_at": "<ISO timestamp>",
            "num_images": <int>
        },
        ...
    }
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.RLock()
        self._data: dict = self._load()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def enroll(self, name: str, embedding: np.ndarray, engine: str = "dlib") -> None:
        """Add one embedding for *name*. Accumulates multiple embeddings."""
        self.enroll_batch(name, [embedding], engine=engine)

    def enroll_batch(self, name: str, embeddings: list[np.ndarray], engine: str = "dlib") -> None:
        """
        Enroll several embeddings for *name* with a single write.

        `engine` records which model produced the vectors. Embeddings from
        different engines occupy different spaces and are meaningless to
        compare, so the name is stored and checked rather than assumed.

        The old implementation rewrote the entire JSON file once per
        embedding, so a 5-photo enrolment meant 5 full-database writes.
        """
        name = (name or "").strip()
        if not name:
            raise ValueError("Person name must not be empty.")
        if not embeddings:
            return

        for emb in embeddings:
            self._validate(emb)

        with self._lock:
            entry = self._data.get(name)
            if entry is None:
                entry = {
                    "embeddings": [],
                    "enrolled_at": datetime.now().isoformat(),
                    "num_images": 0,
                    "engine": engine,
                }
                self._data[name] = entry
            entry.setdefault("engine", "dlib")
            if entry["engine"] != engine:
                raise ValueError(
                    f"'{name}' was enrolled with the '{entry['engine']}' engine; "
                    f"refusing to add '{engine}' embeddings. Remove the person first, "
                    f"or switch back to '{entry['engine']}'."
                )

            entry["embeddings"].extend(self._encode(e) for e in embeddings)
            entry["num_images"] = len(entry["embeddings"])
            self._save()

        logger.info(
            "Enrolled %d embedding(s) for '%s' (total: %d)",
            len(embeddings), name, self._data[name]["num_images"],
        )

    def remove(self, name: str) -> bool:
        """Delete a person from the database. Returns True if found."""
        with self._lock:
            if name in self._data:
                del self._data[name]
                self._save()
                logger.info("Removed '%s' from database.", name)
                return True
        return False

    def list_enrolled(self) -> list[str]:
        """Return sorted list of enrolled names."""
        with self._lock:
            return sorted(self._data.keys())

    def find_name(self, name: str) -> str | None:
        """
        Return the stored spelling of *name*, matched case-insensitively.

        Lets callers warn about "alice" vs "Alice" instead of silently
        creating two people who are the same person.
        """
        target = (name or "").strip().casefold()
        with self._lock:
            for existing in self._data:
                if existing.casefold() == target:
                    return existing
        return None

    def get_embeddings(self, name: str) -> list[np.ndarray]:
        """Return all embeddings for *name*, or [] if not found."""
        with self._lock:
            entry = self._data.get(name)
            if entry is None:
                return []
            encoded = list(entry["embeddings"])
        return [self._decode(e) for e in encoded]

    def get_all_embeddings(self) -> dict[str, list[np.ndarray]]:
        """Return {name: [embedding, ...]} for every enrolled person."""
        with self._lock:
            snapshot = {n: list(e["embeddings"]) for n, e in self._data.items()}
        return {
            name: [self._decode(e) for e in encoded]
            for name, encoded in snapshot.items()
        }

    def get_info(self, name: str) -> dict | None:
        """Return metadata for *name*."""
        with self._lock:
            entry = self._data.get(name)
            if entry is None:
                return None
            info = dict(entry)
        info.pop("embeddings", None)  # exclude raw bytes from info
        return info

    def stats(self) -> dict:
        """Return summary statistics of the database."""
        with self._lock:
            counts = {n: e["num_images"] for n, e in self._data.items()}
            engines = sorted({e.get("engine", "dlib") for e in self._data.values()})
        return {
            "total_persons": len(counts),
            "total_embeddings": sum(counts.values()),
            "persons": counts,
            "engines": engines,
        }

    def engines_in_use(self) -> set[str]:
        """Which embedding engines produced the stored vectors."""
        with self._lock:
            return {e.get("engine", "dlib") for e in self._data.values()}

    def clear(self) -> None:
        """Wipe the entire database."""
        with self._lock:
            self._data = {}
            self._save()
        logger.warning("Database cleared.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
                logger.error("Database root is %s, expected object. Starting empty.", type(data).__name__)
            except (json.JSONDecodeError, IOError) as exc:
                logger.error("Failed to load database: %s", exc)
        return {}

    def _save(self) -> None:
        """
        Write atomically: a crash or a concurrent reader can never observe a
        half-written database, because the rename is the only visible step.
        """
        tmp_path = f"{self.db_path}.{os.getpid()}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.db_path)
        except BaseException:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(embedding: np.ndarray) -> None:
        arr = np.asarray(embedding)
        if arr.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"Embedding must have shape ({EMBEDDING_DIM},), got {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Embedding contains NaN or infinity.")

    @staticmethod
    def _encode(embedding: np.ndarray) -> str:
        """Encode numpy float64 array → base64 string."""
        return base64.b64encode(np.asarray(embedding, dtype=np.float64).tobytes()).decode("ascii")

    @staticmethod
    def _decode(encoded: str) -> np.ndarray:
        """
        Decode base64 string → numpy float64 array.

        np.frombuffer aliases the immutable bytes object, so the result is
        read-only and any in-place maths downstream raises. Copy it.
        """
        raw = base64.b64decode(encoded)
        if len(raw) != _BYTES_PER_EMBEDDING:
            raise ValueError(
                f"Corrupt embedding: expected {_BYTES_PER_EMBEDDING} bytes, got {len(raw)}."
            )
        return np.frombuffer(raw, dtype=np.float64).copy()
