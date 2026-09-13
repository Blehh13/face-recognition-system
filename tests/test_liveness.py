"""
Tests for the advisory liveness (presentation-attack) detector.

What these can and cannot establish: they confirm the model loads, produces a
calibrated probability, crops correctly, and never breaks a match when it
fails. They do **not** establish that it catches real attacks — that needs
printed-photo and screen-replay captures from the target camera, which we do
not have. The false-accept side is therefore untested by design rather than by
oversight, and the feature is wired as advisory for exactly that reason.
"""

import numpy as np
import pytest

liveness = pytest.importorskip("src.liveness")

requires_model = pytest.mark.skipif(
    not liveness.available(), reason="MiniFASNet ONNX weights not downloaded"
)


@pytest.fixture(scope="module")
def face_photo():
    import os
    matplotlib = pytest.importorskip("matplotlib")
    from PIL import Image
    path = os.path.join(os.path.dirname(matplotlib.__file__),
                        "mpl-data", "sample_data", "grace_hopper.jpg")
    if not os.path.exists(path):
        pytest.skip("sample photograph unavailable")
    return np.array(Image.open(path).convert("RGB"))


# ------------------------------------------------------------------ result

def test_result_serialises_and_is_marked_advisory():
    r = liveness.LivenessResult(live_score=0.31, suspicious=True, label="possible spoof")
    payload = r.to_dict()
    assert payload["live_score"] == pytest.approx(0.31)
    assert payload["suspicious"] is True
    # A client must never be able to mistake this for a verdict.
    assert payload["advisory"] is True


def test_softmax_is_a_distribution():
    p = liveness._softmax(np.array([3.0, -1.0, 0.5]))
    assert p.sum() == pytest.approx(1.0)
    assert np.all(p >= 0)
    assert int(np.argmax(p)) == 0


def test_softmax_survives_large_logits():
    """Naive exp() overflows here; the shift-by-max form must not."""
    p = liveness._softmax(np.array([10000.0, 9999.0, 1.0]))
    assert np.all(np.isfinite(p))
    assert p.sum() == pytest.approx(1.0)


# -------------------------------------------------------------------- crop

def test_crop_is_square_and_correctly_sized():
    img = np.random.default_rng(0).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    patch = liveness.LivenessDetector._crop(img, (150, 250, 250, 150))
    assert patch.shape == (liveness.INPUT_SIZE, liveness.INPUT_SIZE, 3)


def test_crop_handles_a_face_against_the_edge():
    """The 2.7x context box runs past the frame for a face near a corner."""
    img = np.random.default_rng(1).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    patch = liveness.LivenessDetector._crop(img, (0, 40, 40, 0))
    assert patch.shape == (liveness.INPUT_SIZE, liveness.INPUT_SIZE, 3)


def test_crop_never_returns_empty_for_a_degenerate_box():
    img = np.random.default_rng(2).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    patch = liveness.LivenessDetector._crop(img, (50, 50, 50, 50))
    assert patch.size > 0


# ------------------------------------------------------------- live model

@requires_model
def test_scores_a_real_photograph_as_a_probability(face_photo):
    detector = liveness.LivenessDetector()
    result = detector.score(face_photo, (139, 345, 325, 159))
    assert 0.0 <= result.live_score <= 1.0
    assert result.label in ("live", "possible spoof")
    assert result.suspicious == (result.live_score < liveness.SUSPICION_THRESHOLD)


@requires_model
def test_scoring_is_deterministic(face_photo):
    detector = liveness.LivenessDetector()
    box = (139, 345, 325, 159)
    assert detector.score(face_photo, box).live_score == pytest.approx(
        detector.score(face_photo, box).live_score
    )


@requires_model
def test_genuine_photograph_is_not_flagged(face_photo):
    """
    A normal photograph should clear the caution line.

    This is the false-reject side, and it is the only side we can measure
    without real attack samples. If this starts failing, the threshold is
    flagging real people.
    """
    detector = liveness.LivenessDetector()
    assert not detector.score(face_photo, (139, 345, 325, 159)).suspicious


# ----------------------------------------------------------------- wiring

def test_identify_still_works_when_liveness_is_unavailable(monkeypatch):
    """An advisory signal must never be able to break identification."""
    import app as flask_app

    monkeypatch.setattr(flask_app, "_liveness", None, raising=False)
    monkeypatch.setattr(flask_app, "_liveness_failed", False, raising=False)
    monkeypatch.setattr(
        "src.liveness.LivenessDetector",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("weights missing")),
    )
    assert flask_app.get_liveness() is None
    # and it must not retry on every request
    assert flask_app.get_liveness() is None
