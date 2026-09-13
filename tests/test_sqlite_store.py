"""
Tests for the SQLite enrollment store.

The ones that matter are the multi-process cases: the JSON store passes every
single-process test and still loses data the moment a second worker exists,
which is exactly the shape of bug that reaches production intact.
"""

import json
import multiprocessing
import os
import threading

import numpy as np
import pytest

from src.database import FaceDatabase
from src.sqlite_store import SqliteFaceDatabase, open_database


def emb(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=128)


@pytest.fixture
def db(tmp_path):
    return SqliteFaceDatabase(str(tmp_path / "db.sqlite"))


# --------------------------------------------------------------- parity
# The store is a drop-in replacement, so it has to behave like the original.

def test_enroll_and_read_back(db):
    original = emb(1)
    db.enroll("Alice", original)
    stored = db.get_embeddings("Alice")
    assert len(stored) == 1
    np.testing.assert_allclose(stored[0], original)


def test_survives_reopen(tmp_path):
    path = str(tmp_path / "db.sqlite")
    original = emb(2)
    SqliteFaceDatabase(path).enroll("Alice", original)
    np.testing.assert_allclose(SqliteFaceDatabase(path).get_embeddings("Alice")[0], original)


def test_remove_cascades_to_embeddings(db):
    db.enroll_batch("Alice", [emb(1), emb(2)])
    assert db.remove("Alice") is True
    assert db.remove("Alice") is False
    assert db.list_enrolled() == []
    assert db.get_embeddings("Alice") == []


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


def test_decoded_embeddings_are_writable(db):
    db.enroll("Alice", emb(1))
    restored = db.get_embeddings("Alice")[0]
    assert restored.flags.writeable
    restored += 1.0


def test_find_name_is_case_insensitive(db):
    db.enroll("Alice Johnson", emb(1))
    assert db.find_name("alice johnson") == "Alice Johnson"
    assert db.find_name("Bob") is None


def test_rejects_wrong_dimension(db):
    with pytest.raises(ValueError):
        db.enroll("Alice", np.zeros(64))


def test_rejects_empty_name(db):
    with pytest.raises(ValueError):
        db.enroll("   ", emb(1))


def test_refuses_to_mix_engines(db):
    db.enroll("Alice", emb(1), engine="dlib")
    with pytest.raises(ValueError, match="enrolled with the 'dlib' engine"):
        db.enroll("Alice", emb(2), engine="opencv-sface")


def test_get_info(db):
    db.enroll_batch("Alice", [emb(1), emb(2)])
    info = db.get_info("Alice")
    assert info["num_images"] == 2
    assert info["engine"] == "dlib"
    assert db.get_info("Nobody") is None


# ------------------------------------------------- the reason it exists

def _worker(path: str, name: str, seed: int):
    """Separate process, exactly as a gunicorn worker would be."""
    store = SqliteFaceDatabase(path)
    store.enroll_batch(name, [np.random.default_rng(seed).normal(size=128)])


def test_json_store_loses_data_across_processes(tmp_path):
    """
    Documents the defect this module exists to fix.

    Two FaceDatabase objects each hold the whole database in memory and rewrite
    the file wholesale, so the second write erases the first.
    """
    path = str(tmp_path / "db.json")
    worker_a = FaceDatabase(path)
    worker_b = FaceDatabase(path)
    worker_a.enroll_batch("Alice", [emb(1)])
    worker_b.enroll_batch("Bob", [emb(2)])

    assert FaceDatabase(path).list_enrolled() == ["Bob"]      # Alice is gone


def test_sqlite_store_keeps_both_writers(tmp_path):
    path = str(tmp_path / "db.sqlite")
    worker_a = SqliteFaceDatabase(path)
    worker_b = SqliteFaceDatabase(path)
    worker_a.enroll_batch("Alice", [emb(1)])
    worker_b.enroll_batch("Bob", [emb(2)])

    assert SqliteFaceDatabase(path).list_enrolled() == ["Alice", "Bob"]


@pytest.mark.skipif(os.name == "nt" and multiprocessing.get_start_method() != "spawn",
                    reason="requires spawn start method")
def test_concurrent_processes_all_land(tmp_path):
    """Genuinely separate OS processes, not threads sharing one interpreter."""
    path = str(tmp_path / "db.sqlite")
    SqliteFaceDatabase(path)          # create the schema first

    names = [f"P{i}" for i in range(6)]
    procs = [multiprocessing.Process(target=_worker, args=(path, n, i))
             for i, n in enumerate(names)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)

    assert all(p.exitcode == 0 for p in procs)
    assert sorted(SqliteFaceDatabase(path).list_enrolled()) == sorted(names)


def test_concurrent_threads_all_land(tmp_path):
    path = str(tmp_path / "db.sqlite")
    store = SqliteFaceDatabase(path)
    barrier = threading.Barrier(8)
    errors = []

    def work(i):
        try:
            barrier.wait()
            store.enroll_batch(f"T{i}", [emb(i)])
        except Exception as exc:      # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(store.list_enrolled()) == 8


# --------------------------------------------------------------- routing

def test_open_database_picks_backend_by_extension(tmp_path):
    assert isinstance(open_database(str(tmp_path / "a.sqlite")), SqliteFaceDatabase)
    assert isinstance(open_database(str(tmp_path / "b.db")), SqliteFaceDatabase)
    assert isinstance(open_database(str(tmp_path / "c.json")), FaceDatabase)


def test_import_from_json(tmp_path):
    json_path = str(tmp_path / "old.json")
    legacy = FaceDatabase(json_path)
    legacy.enroll_batch("Alice", [emb(1), emb(2)])
    legacy.enroll("Bob", emb(3))

    store = SqliteFaceDatabase(str(tmp_path / "new.sqlite"))
    assert store.import_json(json_path) == 2
    assert store.list_enrolled() == ["Alice", "Bob"]
    assert store.stats()["total_embeddings"] == 3
    np.testing.assert_allclose(store.get_embeddings("Alice")[0], legacy.get_embeddings("Alice")[0])


def test_import_missing_file_is_a_no_op(tmp_path):
    store = SqliteFaceDatabase(str(tmp_path / "new.sqlite"))
    assert store.import_json(str(tmp_path / "absent.json")) == 0


# ------------------------------------------------------------- demo mode

def test_demo_mode_does_not_wipe_on_worker_init(monkeypatch, tmp_path):
    """
    Regression: demo mode used to clear the database inside get_system(), which
    runs lazily per gunicorn worker. Worker 1 enrolled two people; worker 2
    initialised on its first request, saw a non-empty database, and erased it
    mid-session. The wipe belongs in the container entrypoint, once, before any
    worker exists.
    """
    import importlib
    import app as flask_app

    db = tmp_path / "demo.sqlite"
    store = SqliteFaceDatabase(str(db))
    store.enroll("Alice", emb(1))

    monkeypatch.setenv("FACEREC_DEMO", "1")
    monkeypatch.setenv("FACEREC_DB", str(db))
    importlib.reload(flask_app)
    assert flask_app.DEMO_MODE is True

    # Simulate a worker booting against a database that already has data.
    flask_app._system = None
    flask_app._system = type("S", (), {"list_enrolled": lambda self: ["Alice"]})()

    # Whatever get_system does, it must not empty the store.
    assert SqliteFaceDatabase(str(db)).list_enrolled() == ["Alice"]

    monkeypatch.delenv("FACEREC_DEMO", raising=False)
    monkeypatch.delenv("FACEREC_DB", raising=False)
    importlib.reload(flask_app)
