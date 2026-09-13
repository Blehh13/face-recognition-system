"""
HTTP contract of the Flask API.

The recognition system is stubbed so these stay fast and run without dlib
models installed; test_pipeline.py covers the real thing.
"""

import io
import json

import numpy as np
import pytest
from PIL import Image

import app as flask_app
from src.system import EnrollOutcome


class FakeMatcher:
    """The parts of FaceMatcher that app.py reads."""
    threshold = 1.012
    metric = "euclidean"
    aggregation = "centroid"


class FakeSystem:
    """
    Stands in for FaceRecognitionSystem with predictable behaviour.

    It has to expose everything app.py touches — including `matcher`, since
    /identify falls back to the engine's own threshold when the request does
    not supply one.
    """

    engine_name = "opencv-sface"

    def __init__(self):
        self.people: dict[str, int] = {}
        self.faces_in_next_photo = 1
        self.matcher = FakeMatcher()

    # -- enrolment ----------------------------------------------------
    def enroll_from_array(self, name, image, require_single_face=True):
        if self.faces_in_next_photo == 0:
            return EnrollOutcome(0, 0, "No face found in the photo.")
        if require_single_face and self.faces_in_next_photo > 1:
            return EnrollOutcome(0, self.faces_in_next_photo, "Found 2 faces. Use a photo of just this person.")
        self.people[name] = self.people.get(name, 0) + 1
        return EnrollOutcome(1, 1)

    # -- identification -----------------------------------------------
    def identify_from_array(self, image, threshold=None):
        self.last_threshold = threshold
        return []

    def annotate_image(self, image, results):
        return image

    # -- database -----------------------------------------------------
    def list_enrolled(self):
        return sorted(self.people)

    def remove_person(self, name):
        return self.people.pop(name, None) is not None

    def db_stats(self):
        return {"total_persons": len(self.people), "total_embeddings": sum(self.people.values())}

    @property
    def database(self):
        outer = self

        class _DB:
            def find_name(self, name):
                target = name.strip().casefold()
                return next((n for n in outer.people if n.casefold() == target), None)

        return _DB()


@pytest.fixture
def fake(monkeypatch):
    system = FakeSystem()
    monkeypatch.setattr(flask_app, "get_system", lambda: system)
    return system


@pytest.fixture
def client():
    flask_app.app.config["TESTING"] = True
    return flask_app.app.test_client()


JSON = {"Accept": "application/json"}


def photo(color=(120, 120, 120), size=(64, 64), fmt="JPEG") -> io.BytesIO:
    buf = io.BytesIO()
    Image.fromarray(np.full((*size, 3), color, dtype=np.uint8)).save(buf, format=fmt)
    buf.seek(0)
    return buf


def post_enroll(client, name="Alice", files=None, headers=JSON):
    data = {"name": name}
    if files is None:
        files = [(photo(), "a.jpg")]
    if files:
        data["images"] = files if len(files) > 1 else files[0]
    return client.post("/enroll", headers=headers, data=data, content_type="multipart/form-data")


# ------------------------------------------------------------------- health

def test_health(client):
    assert client.get("/health").get_json() == {"status": "ok"}


# ------------------------------------------------------------------ enroll

def test_enroll_success_returns_json(client, fake):
    res = post_enroll(client)
    assert res.status_code == 200
    body = res.get_json()
    assert body["name"] == "Alice"
    assert body["faces_enrolled"] == 1
    assert body["warnings"] == []


def test_enroll_requires_a_name(client, fake):
    res = post_enroll(client, name="   ")
    assert res.status_code == 400
    assert "name" in res.get_json()["error"].lower()


def test_enroll_requires_a_file(client, fake):
    res = client.post("/enroll", headers=JSON, data={"name": "Alice"},
                      content_type="multipart/form-data")
    assert res.status_code == 400


def test_enroll_reports_no_face_found(client, fake):
    fake.faces_in_next_photo = 0
    body = post_enroll(client).get_json()
    assert body["faces_enrolled"] == 0
    assert any("No face" in w for w in body["warnings"])


def test_enroll_refuses_group_photo(client, fake):
    """
    Regression: every face in a group photo used to be stored under the one
    name, silently poisoning that identity.
    """
    fake.faces_in_next_photo = 2
    body = post_enroll(client).get_json()
    assert body["faces_enrolled"] == 0
    assert any("just this person" in w for w in body["warnings"])
    assert fake.people == {}


def test_enroll_rejects_non_image(client, fake):
    body = post_enroll(client, files=[(io.BytesIO(b"definitely not an image"), "x.jpg")]).get_json()
    assert body["faces_enrolled"] == 0
    assert body["warnings"]


def test_enroll_folds_into_existing_name_case_insensitively(client, fake):
    post_enroll(client, name="Alice Johnson")
    body = post_enroll(client, name="alice johnson").get_json()
    assert body["name"] == "Alice Johnson"
    assert fake.list_enrolled() == ["Alice Johnson"]


def test_legacy_form_post_still_redirects(client, fake):
    res = post_enroll(client, headers={})
    assert res.status_code == 302


# ---------------------------------------------------------------- identify

def test_identify_requires_an_image(client, fake):
    res = client.post("/identify", headers=JSON, data={}, content_type="multipart/form-data")
    assert res.status_code == 400


def test_identify_rejects_non_image(client, fake):
    res = client.post("/identify", headers=JSON,
                      data={"image": (io.BytesIO(b"nope"), "x.jpg")},
                      content_type="multipart/form-data")
    assert res.status_code == 400


def test_identify_with_no_faces(client, fake):
    res = client.post("/identify", headers=JSON,
                      data={"image": (photo(), "a.jpg"), "threshold": "0.6"},
                      content_type="multipart/form-data")
    body = res.get_json()
    assert body["num_faces"] == 0
    assert body["faces"] == []
    assert body["threshold_used"] == 0.6


def test_identify_passes_threshold_through(client, fake):
    client.post("/identify", headers=JSON,
                data={"image": (photo(), "a.jpg"), "threshold": "0.45"},
                content_type="multipart/form-data")
    assert fake.last_threshold == 0.45


def test_identify_clamps_absurd_threshold(client, fake):
    client.post("/identify", headers=JSON,
                data={"image": (photo(), "a.jpg"), "threshold": "-5"},
                content_type="multipart/form-data")
    assert fake.last_threshold >= 0.0


def test_identify_tolerates_bad_threshold(client, fake):
    """An unparseable threshold falls back to the engine's own, not a literal."""
    res = client.post("/identify", headers=JSON,
                      data={"image": (photo(), "a.jpg"), "threshold": "banana"},
                      content_type="multipart/form-data")
    assert res.status_code == 200
    assert res.get_json()["threshold_used"] == pytest.approx(FakeMatcher.threshold)


def test_identify_without_a_threshold_uses_the_engine_default(client, fake):
    """
    Regression: /identify hardcoded 0.60, left over from the dlib engine. With
    SFace gating at 1.012 that silently rejected valid matches at 0.70.
    """
    res = client.post("/identify", headers=JSON,
                      data={"image": (photo(), "a.jpg")},
                      content_type="multipart/form-data")
    assert res.get_json()["threshold_used"] == pytest.approx(FakeMatcher.threshold)
    assert fake.last_threshold is None      # passed through as "use your own"


def test_config_reports_engine_facts(client, fake):
    body = client.get("/api/config", headers=JSON).get_json()
    assert body["engine"] == "opencv-sface"
    assert body["threshold"] == pytest.approx(FakeMatcher.threshold)
    assert body["aggregation"] == "centroid"


def test_identify_response_is_strict_json(client, fake):
    """The browser's JSON.parse rejects Infinity/NaN, so the body must not contain them."""
    res = client.post("/identify", headers=JSON,
                      data={"image": (photo(), "a.jpg")},
                      content_type="multipart/form-data")
    raw = res.get_data(as_text=True)
    json.loads(raw)  # strict by default: raises on bare Infinity/NaN
    assert "Infinity" not in raw
    assert "NaN" not in raw


# ------------------------------------------------------------------- persons

def test_persons_listing(client, fake):
    post_enroll(client, name="Alice")
    body = client.get("/api/persons", headers=JSON).get_json()
    assert body["persons"] == ["Alice"]
    assert body["stats"]["total_persons"] == 1


def test_remove(client, fake):
    post_enroll(client, name="Alice")
    assert client.post("/api/remove", json={"name": "Alice"}).get_json()["removed"] is True
    assert client.post("/api/remove", json={"name": "Alice"}).get_json()["removed"] is False


def test_remove_requires_name(client, fake):
    assert client.post("/api/remove", json={}).status_code == 400
    assert client.post("/api/remove", json={"name": "  "}).status_code == 400
