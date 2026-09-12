"""
database.py — Persistent face enrollment database.

Stores name → list-of-embeddings mappings in a JSON file (base64-encoded
numpy arrays). No external DB required — zero cost, zero setup.
"""

import json
import os
import base64
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "enrolled_faces.json")


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
        self._data: dict = self._load()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def enroll(self, name: str, embedding: np.ndarray) -> None:
        """Add one embedding for *name*. Accumulates multiple embeddings."""
        name = name.strip()
        if name not in self._data:
            self._data[name] = {
                "embeddings": [],
                "enrolled_at": datetime.now().isoformat(),
                "num_images": 0,
            }
        self._data[name]["embeddings"].append(self._encode(embedding))
        self._data[name]["num_images"] += 1
        self._save()
        logger.info("Enrolled '%s' (total embeddings: %d)", name, self._data[name]["num_images"])

    def enroll_batch(self, name: str, embeddings: list[np.ndarray]) -> None:
        """Enroll multiple embeddings for *name* in one call."""
        for emb in embeddings:
            self.enroll(name, emb)

    def remove(self, name: str) -> bool:
        """Delete a person from the database. Returns True if found."""
        if name in self._data:
            del self._data[name]
            self._save()
            logger.info("Removed '%s' from database.", name)
            return True
        return False

    def list_enrolled(self) -> list[str]:
        """Return sorted list of enrolled names."""
        return sorted(self._data.keys())

    def get_embeddings(self, name: str) -> list[np.ndarray]:
        """Return all embeddings for *name*, or [] if not found."""
        if name not in self._data:
            return []
        return [self._decode(e) for e in self._data[name]["embeddings"]]

    def get_all_embeddings(self) -> dict[str, list[np.ndarray]]:
        """Return {name: [embedding, ...]} for every enrolled person."""
        return {
            name: [self._decode(e) for e in entry["embeddings"]]
            for name, entry in self._data.items()
        }

    def get_info(self, name: str) -> dict | None:
        """Return metadata for *name*."""
        if name not in self._data:
            return None
        info = dict(self._data[name])
        info.pop("embeddings", None)  # exclude raw bytes from info
        return info

    def stats(self) -> dict:
        """Return summary statistics of the database."""
        names = list(self._data.keys())
        counts = {n: self._data[n]["num_images"] for n in names}
        return {
            "total_persons": len(names),
            "total_embeddings": sum(counts.values()),
            "persons": counts,
        }

    def clear(self) -> None:
        """⚠ Wipe the entire database."""
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
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as exc:
                logger.error("Failed to load database: %s", exc)
        return {}

    def _save(self) -> None:
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode(embedding: np.ndarray) -> str:
        """Encode numpy float64 array → base64 string."""
        return base64.b64encode(embedding.astype(np.float64).tobytes()).decode("ascii")

    @staticmethod
    def _decode(encoded: str) -> np.ndarray:
        """Decode base64 string → numpy float64 array."""
        raw = base64.b64decode(encoded)
        return np.frombuffer(raw, dtype=np.float64)
