"""
Tests for the YuNet + SFace backend and the engine-isolation guard.

The model files are a one-off download, so tests needing them skip when they
are absent. The guard tests do not need models and always run.
"""

import os
import tempfile

import numpy as np
import pytest

from src.database import FaceDatabase


def emb(seed: int, dim: int = 128) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=dim)


# ------------------------------------------------------- engine isolation
# These matter most: comparing vectors from two different models produces
# confident nonsense rather than an obvious failure.

def test_database_records_the_engine(tmp_path):
    db = FaceDatabase(str(tmp_path / "db.json"))
    db.enroll("Alice", emb(1), engine="opencv-sface")
    assert db.get_info("Alice")["engine"] == "opencv-sface"
    assert db.engines_in_use() == {"opencv-sface"}


def test_database_defaults_to_dlib_for_legacy_rows(tmp_path):
    """A database written before engines were tracked must still load."""
    import json
    path = tmp_path / "db.json"
    path.write_text(json.dumps({
        "Alice": {"embeddings": [], "num_images": 0, "enrolled_at": "2026-01-01T00:00:00"}
    }))
    db = FaceDatabase(str(path))
    assert db.engines_in_use() == {"dlib"}


def test_database_refuses_to_mix_engines(tmp_path):
    db = FaceDatabase(str(tmp_path / "db.json"))
    db.enroll("Alice", emb(1), engine="dlib")
    with pytest.raises(ValueError, match="enrolled with the 'dlib' engine"):
        db.enroll("Alice", emb(2), engine="opencv-sface")


def test_different_people_may_not_disagree_on_engine(tmp_path):
    """Two engines in one database is exactly the state the system rejects."""
    db = FaceDatabase(str(tmp_path / "db.json"))
    db.enroll("Alice", emb(1), engine="dlib")
    db.enroll("Bob", emb(2), engine="opencv-sface")
    assert db.engines_in_use() == {"dlib", "opencv-sface"}

    from src.system import FaceRecognitionSystem
    with pytest.raises(ValueError, match="Database contains embeddings"):
        FaceRecognitionSystem(db_path=str(tmp_path / "db.json"))


def test_stats_reports_engines(tmp_path):
    db = FaceDatabase(str(tmp_path / "db.json"))
    db.enroll("Alice", emb(1), engine="dlib")
    assert db.stats()["engines"] == ["dlib"]


def test_unknown_engine_rejected():
    from src.system import FaceRecognitionSystem
    with pytest.raises(ValueError, match="engine must be"):
        FaceRecognitionSystem(engine="magic")


# ------------------------------------------------------------ live models

requires_models = pytest.mark.skipif(
    not __import__("src.opencv_engine", fromlist=["available"]).available(),
    reason="YuNet/SFace ONNX models not downloaded",
)


@pytest.fixture(scope="module")
def face_photo():
    matplotlib = pytest.importorskip("matplotlib")
    from PIL import Image
    path = os.path.join(os.path.dirname(matplotlib.__file__),
                        "mpl-data", "sample_data", "grace_hopper.jpg")
    if not os.path.exists(path):
        pytest.skip("sample photograph unavailable")
    return np.array(Image.open(path).convert("RGB"))


@requires_models
def test_yunet_detects_a_real_face(face_photo):
    from src.opencv_engine import YuNetDetector
    boxes = YuNetDetector().detect(face_photo)
    assert len(boxes) == 1
    top, right, bottom, left = boxes[0]
    assert 0 <= top < bottom <= face_photo.shape[0]
    assert 0 <= left < right <= face_photo.shape[1]


@requires_models
def test_yunet_finds_nothing_in_a_blank_frame():
    from src.opencv_engine import YuNetDetector
    assert YuNetDetector().detect(np.full((200, 200, 3), 200, np.uint8)) == []


@requires_models
def test_sface_embeddings_are_unit_length(face_photo):
    from src.opencv_engine import SFaceEmbedder
    vectors = SFaceEmbedder().embed(face_photo)
    assert len(vectors) == 1
    assert vectors[0].shape == (128,)
    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0, abs=1e-6)


@requires_models
def test_sface_is_stable_for_the_same_image(face_photo):
    from src.opencv_engine import SFaceEmbedder
    e = SFaceEmbedder()
    np.testing.assert_allclose(e.embed(face_photo)[0], e.embed(face_photo)[0], atol=1e-6)


@requires_models
def test_sface_separates_a_person_from_a_distorted_stranger(face_photo):
    """A mirrored face should stay much closer than an unrelated crop."""
    from src.opencv_engine import SFaceEmbedder
    e = SFaceEmbedder()
    original = e.embed(face_photo)[0]
    mirrored = e.embed(np.ascontiguousarray(face_photo[:, ::-1]))[0]
    assert np.linalg.norm(original - mirrored) < 1.17   # the fitted threshold


@requires_models
def test_end_to_end_enroll_and_identify(face_photo):
    from src.system import FaceRecognitionSystem
    with tempfile.TemporaryDirectory() as tmp:
        system = FaceRecognitionSystem(engine="opencv", db_path=os.path.join(tmp, "db.json"))
        assert system.engine_name == "opencv-sface"
        assert system.matcher.threshold == pytest.approx(1.17)

        assert system.enroll_from_array("Grace Hopper", face_photo).enrolled == 1
        (_loc, result), = system.identify_from_array(face_photo)
        assert result.name == "Grace Hopper"
        assert result.is_known


@requires_models
def test_opencv_engine_refuses_group_photo(face_photo):
    from src.system import FaceRecognitionSystem
    group = np.concatenate([face_photo, face_photo[:, ::-1]], axis=1)
    with tempfile.TemporaryDirectory() as tmp:
        system = FaceRecognitionSystem(engine="opencv", db_path=os.path.join(tmp, "db.json"))
        outcome = system.enroll_from_array("Team", group)
        if outcome.faces_found < 2:
            pytest.skip("detector did not find two faces")
        assert outcome.enrolled == 0
        assert "just this person" in outcome.reason
