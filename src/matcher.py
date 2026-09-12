"""
matcher.py — Similarity-based face matching with unknown rejection.

Two distance metrics are supported:
  - Euclidean distance (default, used by dlib/face_recognition)
  - Cosine similarity (alternative)

The **rejection threshold** controls the "Unknown" gate:
  - Euclidean: distance > threshold → Unknown  (recommended: 0.55–0.65)
  - Cosine:    similarity < threshold → Unknown (recommended: 0.40–0.50)

The matcher aggregates multiple enrolled embeddings per person by picking the
minimum distance (nearest-neighbour strategy). This naturally handles
intra-class variation across enrolment images.
"""

import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Defaults (tuned on LFW-style benchmarks)
# -------------------------------------------------------------------
DEFAULT_EUCLIDEAN_THRESHOLD = 0.60   # face_recognition default is 0.6
DEFAULT_COSINE_THRESHOLD    = 0.40


@dataclass
class MatchResult:
    """Result of a single face identification query."""
    name: str                  # Matched person name or "Unknown"
    distance: float            # Best (minimum) distance to enrolled faces
    similarity: float          # Cosine similarity (0–1)
    is_known: bool             # True if accepted, False if rejected
    threshold_used: float      # Threshold applied
    metric: str                # 'euclidean' or 'cosine'
    scores: dict[str, float]   # {name: best_distance} for all candidates

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "distance": round(self.distance, 4),
            "similarity": round(self.similarity, 4),
            "is_known": self.is_known,
            "confidence": round(self._confidence(), 4),
            "threshold_used": self.threshold_used,
            "metric": self.metric,
        }

    def _confidence(self) -> float:
        """
        Map distance to a [0, 1] confidence score.
        Higher = more confident.
        """
        if self.metric == "euclidean":
            # Sigmoid-like decay: 0.0 distance → 1.0, threshold → ~0.5
            return float(np.exp(-2.5 * self.distance))
        else:
            return self.similarity


class FaceMatcher:
    """
    Match a query embedding against the enrolled database.

    Parameters
    ----------
    threshold : float | None
        Rejection threshold. None → use the metric default.
    metric : str
        'euclidean' (default) or 'cosine'.
    """

    def __init__(
        self,
        threshold: float | None = None,
        metric: str = "euclidean",
    ):
        if metric not in ("euclidean", "cosine"):
            raise ValueError("metric must be 'euclidean' or 'cosine'")
        self.metric = metric
        if threshold is None:
            self.threshold = (
                DEFAULT_EUCLIDEAN_THRESHOLD
                if metric == "euclidean"
                else DEFAULT_COSINE_THRESHOLD
            )
        else:
            self.threshold = threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self,
        query_embedding: np.ndarray,
        database: dict[str, list[np.ndarray]],
    ) -> MatchResult:
        """
        Identify the person closest to *query_embedding*.

        Parameters
        ----------
        query_embedding : np.ndarray  shape (128,)
        database : {name: [embedding, ...]}

        Returns
        -------
        MatchResult
        """
        if not database:
            return MatchResult(
                name="Unknown",
                distance=float("inf"),
                similarity=0.0,
                is_known=False,
                threshold_used=self.threshold,
                metric=self.metric,
                scores={},
            )

        scores: dict[str, float] = {}

        for name, embeddings in database.items():
            dists = [self._distance(query_embedding, e) for e in embeddings]
            scores[name] = min(dists)  # nearest-neighbour per person

        best_name = min(scores, key=scores.__getitem__)
        best_dist = scores[best_name]
        best_sim  = self._cosine_similarity(query_embedding, database[best_name])

        if self.metric == "euclidean":
            is_known = best_dist <= self.threshold
        else:
            is_known = best_sim >= self.threshold

        return MatchResult(
            name=best_name if is_known else "Unknown",
            distance=best_dist,
            similarity=best_sim,
            is_known=is_known,
            threshold_used=self.threshold,
            metric=self.metric,
            scores=scores,
        )

    def match_many(
        self,
        query_embeddings: list[np.ndarray],
        database: dict[str, list[np.ndarray]],
    ) -> list[MatchResult]:
        """Identify multiple faces (e.g., from a group photo)."""
        return [self.match(q, database) for q in query_embeddings]

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.metric == "euclidean":
            return float(np.linalg.norm(a - b))
        else:
            return 1.0 - self._dot_similarity(a, b)

    @staticmethod
    def _dot_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a_n = a / (np.linalg.norm(a) + 1e-10)
        b_n = b / (np.linalg.norm(b) + 1e-10)
        return float(np.dot(a_n, b_n))

    @staticmethod
    def _cosine_similarity(
        query: np.ndarray, enrolled_embeddings: list[np.ndarray]
    ) -> float:
        """Return the maximum cosine similarity across enrolled embeddings."""
        sims = [FaceMatcher._dot_similarity(query, e) for e in enrolled_embeddings]
        return max(sims)
