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

import json
import logging
import math
import os
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Defaults
# -------------------------------------------------------------------
# 0.60 is dlib's documented default. It is also, independently, what falls out
# of fitting the threshold on LFW: `python -m ml.calibrate` selects 0.605 on
# validation identities and scores 97.10% accuracy / 0.63% FAR / 5.17% FRR at
# 0.60 on 462 held-out identities. See ml/results/threshold.json.
#
# The distance is measured on dlib's **raw** 128-d vectors, which are not unit
# length (‖v‖ ≈ 1.42). Normalising first shrinks every distance by that factor
# and makes 0.60 far too permissive — if you change the metric, re-fit the
# threshold with it.
DEFAULT_EUCLIDEAN_THRESHOLD = 0.60
DEFAULT_COSINE_THRESHOLD    = 0.40

# Optional override written by `python -m ml.calibrate`, so a threshold fitted
# on your own population takes effect without editing code. Absent by default.
_CALIBRATION_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ml", "results", "threshold.json",
)


def calibrated_threshold(metric: str = "euclidean") -> float | None:
    """
    Return the fitted threshold for *metric*, or None if no calibration exists.

    Reads the artefact `ml/calibrate.py` writes. Never raises: a missing or
    malformed file simply means "no calibration", and the shipped default
    stands.
    """
    try:
        with open(_CALIBRATION_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("metric") != metric:
        return None
    value = payload.get("fitted_threshold")
    return float(value) if isinstance(value, (int, float)) else None


def _json_safe(value: float) -> float | None:
    """
    JSON has no literal for infinity or NaN. Python's encoder emits the bare
    tokens `Infinity`/`NaN` anyway, which `JSON.parse` in a browser rejects —
    so a single unmatched face used to break the whole web response. Anything
    non-finite becomes null, which every client can read.
    """
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


@dataclass
class MatchResult:
    """Result of a single face identification query."""
    name: str                  # Matched person name or "Unknown"
    distance: float            # Best (minimum) distance to enrolled faces
    similarity: float          # Cosine similarity (0–1)
    is_known: bool             # True if accepted, False if rejected
    threshold_used: float      # Threshold applied
    metric: str                # 'euclidean' or 'cosine'
    scores: dict[str, float] = field(default_factory=dict)  # {name: best_distance}

    def to_dict(self, max_candidates: int = 5) -> dict:
        """JSON-serialisable view. Non-finite numbers are emitted as null."""
        distance = _json_safe(self.distance)
        return {
            "name": self.name,
            "distance": round(distance, 4) if distance is not None else None,
            "similarity": round(_json_safe(self.similarity) or 0.0, 4),
            "is_known": self.is_known,
            "confidence": round(self.confidence(), 4),
            "threshold_used": self.threshold_used,
            "metric": self.metric,
            # True when there was simply nobody to compare against.
            "no_candidates": not self.scores,
            "candidates": self.ranked_candidates(max_candidates),
        }

    def ranked_candidates(self, limit: int = 5) -> list[dict]:
        """
        Every enrolled person this face was compared against, nearest first.

        These distances are computed anyway to find the winner; discarding them
        threw away the answer to the only question a user actually asks when a
        result looks wrong — "who else did it consider, and by how much?". A
        match at 0.41 with the runner-up at 0.43 is a coin toss worth
        distrusting; the same match with the runner-up at 0.85 is not.
        """
        ordered = sorted(self.scores.items(), key=lambda kv: kv[1])[:limit]
        return [
            {
                "name": name,
                "distance": round(_json_safe(d), 4) if _json_safe(d) is not None else None,
                "accepted": bool(self._accepts(d)),
            }
            for name, d in ordered
        ]

    def _accepts(self, distance: float) -> bool:
        """Whether *distance* clears this result's own gate, in its own metric."""
        if not math.isfinite(distance):
            return False
        if self.metric == "euclidean":
            return distance <= self.threshold_used
        # In cosine mode `scores` holds 1 - similarity.
        return (1.0 - distance) >= self.threshold_used

    def confidence(self) -> float:
        """
        Map the match onto a [0, 1] confidence score, calibrated so that the
        rejection threshold always sits at exactly 0.5:

            distance 0          → 1.0   (identical embeddings)
            distance threshold  → 0.5   (right on the accept/reject line)
            distance 2·threshold→ 0.25

        That makes the number readable without knowing the metric: above 50%
        means accepted, below means rejected. (The previous exp(-2.5·d) curve
        put the threshold at 0.22, so a solid match reported 66% and the
        docstring's promise of ~0.5 at the threshold was simply untrue.)
        """
        threshold = self.threshold_used

        if self.metric == "euclidean":
            if not math.isfinite(self.distance):
                return 0.0
            if threshold <= 0:
                return 1.0 if self.distance <= 0 else 0.0
            return float(0.5 ** (self.distance / threshold))

        # Cosine: higher similarity is better, so the scale runs the other way.
        similarity = self.similarity
        if not math.isfinite(similarity):
            return 0.0
        if similarity >= threshold:
            head = 1.0 - threshold
            return float(0.5 + 0.5 * (similarity - threshold) / head) if head > 0 else 1.0
        return float(0.5 * max(similarity, 0.0) / threshold) if threshold > 0 else 0.0

    # Backwards-compatible alias for the old private name.
    _confidence = confidence


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
            self.threshold = float(threshold)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self,
        query_embedding: np.ndarray,
        database: dict[str, list[np.ndarray]],
        threshold: float | None = None,
    ) -> MatchResult:
        """
        Identify the person closest to *query_embedding*.

        Parameters
        ----------
        query_embedding : np.ndarray  shape (128,)
        database : {name: [embedding, ...]}
        threshold : optional per-call override of the rejection threshold.

        Returns
        -------
        MatchResult
        """
        active_threshold = self.threshold if threshold is None else float(threshold)

        if not database:
            # Nothing enrolled: there is no distance to report, and inf is not
            # representable in JSON. `no_candidates` tells the caller why.
            return MatchResult(
                name="Unknown",
                distance=float("inf"),
                similarity=0.0,
                is_known=False,
                threshold_used=active_threshold,
                metric=self.metric,
                scores={},
            )

        scores: dict[str, float] = {}
        for name, embeddings in database.items():
            if not embeddings:
                continue
            scores[name] = min(self._distance(query_embedding, e) for e in embeddings)

        if not scores:
            return MatchResult(
                name="Unknown",
                distance=float("inf"),
                similarity=0.0,
                is_known=False,
                threshold_used=active_threshold,
                metric=self.metric,
                scores={},
            )

        best_name = min(scores, key=scores.__getitem__)
        best_dist = scores[best_name]
        best_sim  = self._cosine_similarity(query_embedding, database[best_name])

        if self.metric == "euclidean":
            is_known = best_dist <= active_threshold
        else:
            is_known = best_sim >= active_threshold

        return MatchResult(
            name=best_name if is_known else "Unknown",
            distance=best_dist,
            similarity=best_sim,
            is_known=is_known,
            threshold_used=active_threshold,
            metric=self.metric,
            scores=scores,
        )

    def match_many(
        self,
        query_embeddings: list[np.ndarray],
        database: dict[str, list[np.ndarray]],
        threshold: float | None = None,
    ) -> list[MatchResult]:
        """Identify multiple faces (e.g., from a group photo)."""
        return [self.match(q, database, threshold=threshold) for q in query_embeddings]

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.metric == "euclidean":
            return float(np.linalg.norm(a - b))
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
        if not enrolled_embeddings:
            return 0.0
        return max(FaceMatcher._dot_similarity(query, e) for e in enrolled_embeddings)
