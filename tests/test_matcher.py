"""
Matching, thresholding and JSON-serialisation behaviour.

These run without dlib: the matcher works on plain 128-d vectors.
"""

import json
import math

import numpy as np
import pytest

from src.matcher import FaceMatcher, MatchResult


def vec(seed: int, dim: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)


@pytest.fixture
def database():
    return {"Alice": [vec(1)], "Bob": [vec(2)]}


# ----------------------------------------------------------------- matching

def test_exact_match_is_accepted(database):
    result = FaceMatcher(threshold=0.6).match(vec(1), database)
    assert result.name == "Alice"
    assert result.is_known
    assert result.distance == pytest.approx(0.0, abs=1e-9)


def test_distant_face_is_rejected(database):
    result = FaceMatcher(threshold=0.6).match(vec(99), database)
    assert result.name == "Unknown"
    assert not result.is_known


def test_threshold_gates_the_decision(database):
    query = vec(1) * 0.75 + vec(2) * 0.25
    loose = FaceMatcher(threshold=2.0).match(query, database)
    strict = FaceMatcher(threshold=0.01).match(query, database)
    assert loose.is_known
    assert not strict.is_known


def test_per_call_threshold_overrides_the_default(database):
    matcher = FaceMatcher(threshold=0.01)
    assert not matcher.match(vec(1) * 0.8 + vec(2) * 0.2, database).is_known
    assert matcher.match(vec(1) * 0.8 + vec(2) * 0.2, database, threshold=2.0).is_known
    # The override must not mutate the matcher.
    assert matcher.threshold == 0.01


def test_nearest_neighbour_across_multiple_embeddings():
    db = {"Alice": [vec(50), vec(1)]}
    result = FaceMatcher(threshold=0.6).match(vec(1), db)
    assert result.is_known
    assert result.distance == pytest.approx(0.0, abs=1e-9)


def test_person_with_no_embeddings_is_skipped():
    db = {"Ghost": [], "Alice": [vec(1)]}
    result = FaceMatcher(threshold=0.6).match(vec(1), db)
    assert result.name == "Alice"


# ------------------------------------------------------- empty database/JSON

def test_empty_database_yields_unknown():
    result = FaceMatcher(threshold=0.6).match(vec(1), {})
    assert result.name == "Unknown"
    assert not result.is_known
    assert not result.scores


def test_to_dict_never_emits_non_finite_json():
    """
    Regression: identifying against an empty database produced
    {"distance": Infinity}, which is not valid JSON. Browsers threw on
    JSON.parse, so the very first thing a new user did broke the web UI.
    """
    result = FaceMatcher(threshold=0.6).match(vec(1), {})
    payload = result.to_dict()

    assert payload["distance"] is None
    assert payload["no_candidates"] is True

    encoded = json.dumps(payload, allow_nan=False)      # would raise on inf/nan
    assert "Infinity" not in encoded and "NaN" not in encoded
    assert json.loads(encoded)["distance"] is None


def test_to_dict_is_finite_for_a_normal_match(database):
    payload = FaceMatcher(threshold=0.6).match(vec(1), database).to_dict()
    assert math.isfinite(payload["distance"])
    assert math.isfinite(payload["similarity"])
    assert math.isfinite(payload["confidence"])
    assert payload["no_candidates"] is False


# -------------------------------------------------------------- confidence

def _result(distance: float, threshold: float = 0.6) -> MatchResult:
    return MatchResult("x", distance, 0.9, True, threshold, "euclidean", {"x": distance})


def test_confidence_is_calibrated_to_the_threshold():
    """A match exactly on the threshold must read 50%, not 22%."""
    assert _result(0.0).confidence() == pytest.approx(1.0)
    assert _result(0.6).confidence() == pytest.approx(0.5)
    assert _result(1.2).confidence() == pytest.approx(0.25)


def test_confidence_tracks_the_threshold_that_was_used():
    assert _result(0.5, threshold=0.5).confidence() == pytest.approx(0.5)
    assert _result(0.5, threshold=1.0).confidence() == pytest.approx(0.5 ** 0.5)


def test_confidence_decreases_with_distance():
    values = [_result(d).confidence() for d in (0.1, 0.3, 0.6, 0.9)]
    assert values == sorted(values, reverse=True)


def test_confidence_is_zero_without_candidates():
    assert FaceMatcher(threshold=0.6).match(vec(1), {}).confidence() == 0.0


def test_confidence_above_half_iff_accepted(database):
    for seed in range(10):
        result = FaceMatcher(threshold=0.6).match(vec(seed), database)
        if result.distance == float("inf"):
            continue
        assert result.is_known == (result.confidence() >= 0.5)


def test_legacy_private_alias_still_works():
    assert _result(0.6)._confidence() == pytest.approx(_result(0.6).confidence())


# ------------------------------------------------------------------ cosine

def test_cosine_metric_accepts_and_rejects():
    db = {"Alice": [vec(1)]}
    matcher = FaceMatcher(threshold=0.4, metric="cosine")
    assert matcher.match(vec(1), db).is_known
    assert not matcher.match(vec(99), db).is_known


def test_cosine_confidence_is_half_at_threshold():
    r = MatchResult("x", 0.6, 0.4, True, 0.4, "cosine", {"x": 0.6})
    assert r.confidence() == pytest.approx(0.5)


def test_invalid_metric_rejected():
    with pytest.raises(ValueError):
        FaceMatcher(metric="manhattan")
