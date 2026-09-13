"""
Persistence, concurrency and serialisation behaviour of FaceDatabase.
"""

import json
import os
import threading

import numpy as np
import pytest

from src.database import EMBEDDING_DIM, FaceDatabase


@pytest.fixture
def db(tmp_path):
    return FaceDatabase(str(tmp_path / "db.json"))


def emb(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=EMBEDDING_DIM)


# ------------------------------------------------------------------- basics

def test_enroll_and_read_back(db):
    original = emb(1)
    db.enroll("Alice", original)
    stored = db.get_embeddings("Alice")
    assert len(stored) == 1
    np.testing.assert_allclose(stored[0], original)


def test_round_trip_survives_reopen(tmp_path):
    path = str(tmp_path / "db.json")
    original = emb(2)
    FaceDatabase(path).enroll("Alice", original)
    np.testing.assert_allclose(FaceDatabase(path).get_embeddings("Alice")[0], original)


def test_remove(db):
    db.enroll("Alice", emb(1))
    assert db.remove("Alice") is True
    assert db.remove("Alice") is False
    assert db.list_enrolled() == []


def test_stats(db):
    db.enroll_batch("Alice", [emb(1), emb(2)])
    db.enroll("Bob", emb(3))
    stats = db.stats()
    assert stats["total_persons"] == 2
    assert stats["total_embeddings"] == 3
    assert stats["persons"]["Alice"] == 2


def test_list_is_sorted(db):
    for name in ("Carol", "alice", "Bob"):
        db.enroll(name, emb(1))
    assert db.list_enrolled() == sorted(["Carol", "alice", "Bob"])


# ------------------------------------------------------------ decoded arrays

def test_decoded_embeddings_are_writable(db):
    """
    Regression: np.frombuffer aliases immutable bytes, so decoded embeddings
    came back read-only and any in-place maths downstream raised.
    """
    db.enroll("Alice", emb(1))
    restored = db.get_embeddings("Alice")[0]
    assert restored.flags.writeable
    restored += 1.0  # must not raise


def test_mutating_a_result_does_not_corrupt_the_store(db):
    db.enroll("Alice", emb(1))
    db.get_embeddings("Alice")[0][:] = 999.0
    assert not np.allclose(db.get_embeddings("Alice")[0], 999.0)


# ---------------------------------------------------------------- validation

def test_rejects_wrong_dimension(db):
    with pytest.raises(ValueError):
        db.enroll("Alice", np.zeros(64))


def test_rejects_non_finite(db):
    bad = emb(1)
    bad[0] = np.inf
    with pytest.raises(ValueError):
        db.enroll("Alice", bad)


def test_rejects_empty_name(db):
    with pytest.raises(ValueError):
        db.enroll("   ", emb(1))


def test_detects_corrupt_payload(tmp_path):
    path = tmp_path / "db.json"
    path.write_text(json.dumps({"Alice": {"embeddings": ["dHJ1bmNhdGVk"], "num_images": 1}}))
    with pytest.raises(ValueError, match="Corrupt embedding"):
        FaceDatabase(str(path)).get_embeddings("Alice")


def test_unreadable_file_starts_empty(tmp_path):
    path = tmp_path / "db.json"
    path.write_text("{ not json")
    assert FaceDatabase(str(path)).list_enrolled() == []


# ------------------------------------------------------- case-insensitive name

def test_find_name_is_case_insensitive(db):
    db.enroll("Alice Johnson", emb(1))
    assert db.find_name("alice johnson") == "Alice Johnson"
    assert db.find_name("ALICE JOHNSON") == "Alice Johnson"
    assert db.find_name("Bob") is None


# ------------------------------------------------------------------- writes

def test_batch_writes_once(db, monkeypatch):
    """enroll_batch used to rewrite the whole file once per embedding."""
    calls = []
    original = FaceDatabase._save
    monkeypatch.setattr(
        FaceDatabase, "_save", lambda self: (calls.append(1), original(self))[1]
    )
    db.enroll_batch("Alice", [emb(i) for i in range(5)])
    assert len(calls) == 1
    assert db.stats()["total_embeddings"] == 5


def test_save_is_atomic_and_leaves_no_temp_files(db, tmp_path):
    db.enroll_batch("Alice", [emb(1), emb(2)])
    leftovers = [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]
    assert leftovers == []
    json.loads((tmp_path / "db.json").read_text())  # always parseable


def test_concurrent_enrolments_do_not_lose_writes(db):
    """
    Flask serves on multiple threads; without a lock, concurrent enrolments
    interleaved on the shared dict and lost updates.
    """
    people = [f"P{i}" for i in range(12)]
    barrier = threading.Barrier(len(people))
    errors = []

    def work(name):
        try:
            barrier.wait()
            db.enroll_batch(name, [emb(hash(name) % 1000)])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(n,)) for n in people]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert sorted(db.list_enrolled()) == sorted(people)
    assert db.stats()["total_embeddings"] == len(people)
