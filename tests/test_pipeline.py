"""
End-to-end tests against the real dlib models, on a real photograph.

Skipped automatically when the models or a sample face are unavailable, so
the rest of the suite still runs on a bare checkout.
"""

import numpy as np
import pytest

pytest.importorskip("face_recognition", reason="dlib/face_recognition not installed")

from src.system import COLOR_KNOWN, COLOR_UNKNOWN, FaceRecognitionSystem  # noqa: E402


@pytest.fixture(scope="module")
def face_photo():
    """A real photograph bundled with matplotlib (Grace Hopper, public domain)."""
    import os

    matplotlib = pytest.importorskip("matplotlib")
    from PIL import Image

    path = os.path.join(
        os.path.dirname(matplotlib.__file__), "mpl-data", "sample_data", "grace_hopper.jpg"
    )
    if not os.path.exists(path):
        pytest.skip("sample photograph not available")
    return np.array(Image.open(path).convert("RGB"))


@pytest.fixture(scope="module")
def variant_photo(face_photo):
    """
    The same person, mirrored.

    Re-querying the *identical* array gives a distance of exactly 0.0, which
    no threshold above zero can reject — so rejection tests need a query that
    is genuinely a little way away from what was enrolled.
    """
    return np.ascontiguousarray(face_photo[:, ::-1])


@pytest.fixture
def system(tmp_path):
    try:
        return FaceRecognitionSystem(db_path=str(tmp_path / "db.json"))
    except Exception as exc:  # noqa: BLE001 - missing model files
        pytest.skip(f"dlib models unavailable: {exc}")


@pytest.fixture
def enrolled(system, face_photo):
    outcome = system.enroll_from_array("Grace Hopper", face_photo)
    if not outcome.enrolled:
        pytest.skip(f"detector found no face: {outcome.reason}")
    return system


# ------------------------------------------------------------------ enrol

def test_enroll_real_face(system, face_photo):
    outcome = system.enroll_from_array("Grace Hopper", face_photo)
    assert outcome.enrolled == 1
    assert outcome.faces_found == 1
    assert outcome.reason is None
    assert system.list_enrolled() == ["Grace Hopper"]


def test_enroll_reports_no_face(system):
    blank = np.full((120, 120, 3), 200, dtype=np.uint8)
    outcome = system.enroll_from_array("Nobody", blank)
    assert outcome.enrolled == 0
    assert outcome.faces_found == 0
    assert "No face" in outcome.reason
    assert system.list_enrolled() == []


def test_group_photo_is_refused(system, face_photo):
    """Two faces must not both be stored under one name."""
    group = np.concatenate([face_photo, face_photo[:, ::-1]], axis=1)
    outcome = system.enroll_from_array("Alice", group)
    if outcome.faces_found < 2:
        pytest.skip("detector did not find two faces in the composite")
    assert outcome.enrolled == 0
    assert "just this person" in outcome.reason
    assert system.list_enrolled() == []


def test_group_photo_can_be_forced(system, face_photo):
    group = np.concatenate([face_photo, face_photo[:, ::-1]], axis=1)
    outcome = system.enroll_from_array("Alice", group, require_single_face=False)
    if outcome.faces_found < 2:
        pytest.skip("detector did not find two faces in the composite")
    assert outcome.enrolled == outcome.faces_found


# --------------------------------------------------------------- identify

def test_identifies_the_same_person(enrolled, face_photo):
    results = enrolled.identify_from_array(face_photo)
    assert len(results) == 1
    _loc, result = results[0]
    assert result.name == "Grace Hopper"
    assert result.is_known
    assert result.confidence() >= 0.5


def test_strict_threshold_rejects_a_true_match(enrolled, variant_photo):
    _loc, result = enrolled.identify_from_array(variant_photo, threshold=0.01)[0]
    assert not result.is_known
    assert result.name == "Unknown"
    assert result.confidence() < 0.5


def test_per_call_threshold_does_not_leak(enrolled, variant_photo):
    enrolled.identify_from_array(variant_photo, threshold=0.01)
    _loc, result = enrolled.identify_from_array(variant_photo)[0]
    assert result.is_known


def test_identify_against_empty_database_is_serialisable(system, face_photo):
    """The first thing a new user does: identify before enrolling anyone."""
    import json

    results = system.identify_from_array(face_photo)
    if not results:
        pytest.skip("no face detected")
    payload = results[0][1].to_dict()
    assert payload["distance"] is None
    assert payload["no_candidates"] is True
    json.dumps(payload, allow_nan=False)


def test_no_faces_returns_empty(system):
    blank = np.full((120, 120, 3), 200, dtype=np.uint8)
    assert system.identify_from_array(blank) == []


# --------------------------------------------------------------- annotate

def test_annotation_uses_ui_colours(enrolled, face_photo, variant_photo):
    """
    Regression: the colours were written BGR-first but drawn onto an RGB
    array, so the "unknown" box rendered blue instead of the intended red.
    """
    results = enrolled.identify_from_array(face_photo)
    known = enrolled.annotate_image(face_photo, results)
    assert _contains_colour(known, COLOR_KNOWN)

    rejected = enrolled.identify_from_array(variant_photo, threshold=0.01)
    unknown = enrolled.annotate_image(variant_photo, rejected)
    assert _contains_colour(unknown, COLOR_UNKNOWN)
    assert not _contains_colour(unknown, (0, 0, 220))  # the old blue


def test_annotation_does_not_mutate_the_input(enrolled, face_photo):
    before = face_photo.copy()
    enrolled.annotate_image(face_photo, enrolled.identify_from_array(face_photo))
    np.testing.assert_array_equal(face_photo, before)


def test_annotation_handles_face_at_top_edge(enrolled, face_photo):
    """A label strip drawn above y=0 used to fall outside the image."""
    results = enrolled.identify_from_array(face_photo)
    (top, right, bottom, left), result = results[0]
    shifted = [((0, right, bottom - top, left), result)]
    out = enrolled.annotate_image(face_photo, shifted)
    assert out.shape == face_photo.shape


def _contains_colour(image: np.ndarray, rgb: tuple) -> bool:
    return bool(np.any(np.all(image == np.array(rgb, dtype=image.dtype), axis=-1)))
