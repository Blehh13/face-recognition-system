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


def test_nearest_aggregation_uses_the_closest_photo():
    """aggregation="nearest" keeps the pre-centroid behaviour."""
    db = {"Alice": [vec(50), vec(1)]}
    result = FaceMatcher(threshold=0.6, aggregation="nearest").match(vec(1), db)
    assert result.is_known
    assert result.distance == pytest.approx(0.0, abs=1e-9)


def test_centroid_is_the_default_and_averages_the_set():
    """
    The default compares against the mean of a person's photographs, so an
    exact match to one enrolled photo no longer gives distance zero.
    """
    from src.matcher import representative

    db = {"Alice": [vec(50), vec(1)]}
    result = FaceMatcher(threshold=2.0).match(vec(1), db)
    assert result.name == "Alice"
    expected = float(np.linalg.norm(vec(1) - representative(db["Alice"])))
    assert result.distance == pytest.approx(expected)
    assert result.distance > 0


def test_centroid_beats_nearest_on_a_noisy_enrolment():
    """
    Averaging suppresses one bad enrolment photograph; nearest-neighbour is
    only ever as good as the worst photo someone enrolled. This is the effect
    measured on LFW in ml/aggregation.py.
    """
    true = vec(1)
    noisy = [true + 0.25 * vec(s) for s in (11, 12, 13)]
    impostor = vec(99)

    centroid = FaceMatcher(threshold=10.0, aggregation="centroid")
    nearest = FaceMatcher(threshold=10.0, aggregation="nearest")

    # Margin between a genuine query and an impostor, under each strategy.
    def margin(matcher):
        db = {"Alice": noisy}
        return (matcher.match(impostor, db).distance
                - matcher.match(true, db).distance)

    assert margin(centroid) > margin(nearest)


def test_aggregation_is_validated():
    with pytest.raises(ValueError):
        FaceMatcher(aggregation="median")


def test_representative_of_one_embedding_is_itself():
    from src.matcher import representative
    v = vec(3)
    np.testing.assert_allclose(representative([v]), v)


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


# ------------------------------------------------------- match explanation

def test_candidates_are_ranked_nearest_first(database):
    result = FaceMatcher(threshold=0.6).match(vec(1), database)
    candidates = result.ranked_candidates()
    assert [c["name"] for c in candidates] == ["Alice", "Bob"]
    assert candidates[0]["distance"] <= candidates[1]["distance"]


def test_candidates_mark_which_cleared_the_threshold(database):
    result = FaceMatcher(threshold=0.6).match(vec(1), database)
    candidates = result.ranked_candidates()
    assert candidates[0]["accepted"] is True     # exact match
    assert candidates[1]["accepted"] is False    # unrelated vector


def test_candidates_respect_the_limit():
    db = {f"p{i}": [vec(i)] for i in range(20)}
    assert len(FaceMatcher(threshold=0.6).match(vec(0), db).ranked_candidates(limit=3)) == 3


def test_candidates_are_empty_without_a_database():
    assert FaceMatcher(threshold=0.6).match(vec(1), {}).ranked_candidates() == []


def test_candidates_are_json_safe(database):
    payload = FaceMatcher(threshold=0.6).match(vec(1), database).to_dict()
    json.dumps(payload, allow_nan=False)
    assert payload["candidates"][0]["name"] == "Alice"
