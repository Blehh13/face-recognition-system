"""
Tests for the training and evaluation pipeline.

These guard the parts that are easy to get silently wrong: identity leakage
between splits, the distance metric's scale, and the AUC implementation.
None of them need the LFW download or a GPU.
"""

import numpy as np
import pytest

from ml.data import Split, identity_disjoint_split, make_verification_pairs
from ml.evaluate import accuracy_at, evaluate_pairs, l2_normalize, pair_distances, select_threshold


# ----------------------------------------------------------------- fixtures

def synthetic_population(n_identities=40, per_identity=4, dim=16, seed=0):
    """Clustered vectors: each identity is a centre plus small noise."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_identities, dim)) * 3.0
    images, labels = [], []
    for i in range(n_identities):
        for _ in range(per_identity):
            images.append(centres[i] + rng.normal(size=dim) * 0.2)
            labels.append(i)
    return np.array(images), np.array(labels), [f"person_{i:03d}" for i in range(n_identities)]


# ------------------------------------------------------------ split hygiene

def test_split_shares_no_identity():
    """The whole benchmark is meaningless if a person appears on both sides."""
    images, labels, names = synthetic_population()
    train, test = identity_disjoint_split(images, labels, names, test_fraction=0.3)
    assert set(train.identities).isdisjoint(test.identities)


def test_split_is_deterministic_across_calls():
    """
    Assignment must not depend on PYTHONHASHSEED.

    Python randomises str.__hash__ per process, so a split built on it would
    silently change between runs and make results incomparable.
    """
    images, labels, names = synthetic_population()
    a_train, a_test = identity_disjoint_split(images, labels, names)
    b_train, b_test = identity_disjoint_split(images, labels, names)
    assert a_train.identities == b_train.identities
    assert a_test.identities == b_test.identities


def test_split_respects_minimum_image_counts():
    images, labels, names = synthetic_population(n_identities=30, per_identity=2)
    train, test = identity_disjoint_split(
        images, labels, names, min_train_images=3, min_test_images=2
    )
    # Every identity has exactly 2 images, so none qualifies for training.
    assert len(train.identities) == 0
    assert all(c >= 2 for c in test.counts())


def test_split_covers_every_kept_image():
    images, labels, names = synthetic_population()
    train, test = identity_disjoint_split(images, labels, names)
    assert len(train) + len(test) <= len(images)
    assert len(train.labels) == len(train.images)
    assert len(test.labels) == len(test.images)


def test_labels_are_reindexed_densely():
    images, labels, names = synthetic_population()
    train, _ = identity_disjoint_split(images, labels, names)
    assert train.labels.min() == 0
    assert train.labels.max() == len(train.identities) - 1


# -------------------------------------------------------------------- pairs

def test_pairs_are_balanced_and_correctly_labelled():
    images, labels, names = synthetic_population()
    _train, test = identity_disjoint_split(images, labels, names)
    left, right, same = make_verification_pairs(test, n_pairs=400, seed=0)

    assert len(left) == len(right) == len(same)
    assert same.sum() == len(same) // 2          # balanced
    for i in np.flatnonzero(same == 1):
        assert test.labels[left[i]] == test.labels[right[i]]
    for i in np.flatnonzero(same == 0):
        assert test.labels[left[i]] != test.labels[right[i]]


def test_genuine_pairs_use_two_distinct_images():
    images, labels, names = synthetic_population()
    _train, test = identity_disjoint_split(images, labels, names)
    left, right, same = make_verification_pairs(test, n_pairs=400, seed=0)
    genuine = same == 1
    assert np.all(left[genuine] != right[genuine])


def test_pairs_are_reproducible_for_a_seed():
    images, labels, names = synthetic_population()
    _train, test = identity_disjoint_split(images, labels, names)
    a = make_verification_pairs(test, n_pairs=200, seed=7)
    b = make_verification_pairs(test, n_pairs=200, seed=7)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


# ------------------------------------------------------------------ metrics

def test_normalize_flag_changes_the_distance_scale():
    """
    Regression: dlib embeddings are not unit length (‖v‖ ≈ 1.42), and the
    shipped 0.60 threshold lives in their raw space. Normalising first shrinks
    every distance by that factor, which made 0.60 look wildly permissive and
    produced a false 'your threshold is broken' result.
    """
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(20, 8)) * 1.42
    left = np.arange(0, 10)
    right = np.arange(10, 20)

    raw = pair_distances(emb, left, right, normalize=False)
    normed = pair_distances(emb, left, right, normalize=True)
    assert not np.allclose(raw, normed)
    assert np.all(normed <= 2.0 + 1e-9)     # bounded on the unit sphere
    assert raw.mean() > normed.mean()


def test_normalising_twice_is_a_no_op():
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(10, 8)) * 3.0
    once = l2_normalize(emb)
    np.testing.assert_allclose(once, l2_normalize(once), atol=1e-12)


def test_auc_is_one_for_perfect_separation():
    distances = np.concatenate([np.full(50, 0.1), np.full(50, 0.9)])
    is_same = np.concatenate([np.ones(50, int), np.zeros(50, int)])
    report = evaluate_pairs(distances, is_same)
    assert report.auc == pytest.approx(1.0)
    assert report.eer == pytest.approx(0.0)
    assert report.best_accuracy == pytest.approx(1.0)


def test_auc_is_half_for_no_signal():
    rng = np.random.default_rng(0)
    distances = rng.random(2000)
    is_same = np.array([1, 0] * 1000)
    assert evaluate_pairs(distances, is_same).auc == pytest.approx(0.5, abs=0.05)


def test_auc_handles_ties():
    """All-identical scores must give exactly 0.5, not 0 or 1."""
    distances = np.full(100, 0.5)
    is_same = np.array([1, 0] * 50)
    assert evaluate_pairs(distances, is_same).auc == pytest.approx(0.5)


def test_tar_at_far_is_monotone_in_budget():
    rng = np.random.default_rng(3)
    distances = np.concatenate([rng.normal(0.4, 0.1, 500), rng.normal(0.9, 0.1, 500)])
    is_same = np.concatenate([np.ones(500, int), np.zeros(500, int)])
    report = evaluate_pairs(distances, is_same)
    assert report.tar_at_far_1pct >= report.tar_at_far_0p1pct


def test_evaluate_requires_both_classes():
    with pytest.raises(ValueError):
        evaluate_pairs(np.array([0.1, 0.2]), np.array([1, 1]))


def test_separation_is_positive_when_genuine_pairs_are_closer():
    distances = np.concatenate([np.full(50, 0.3), np.full(50, 0.8)])
    is_same = np.concatenate([np.ones(50, int), np.zeros(50, int)])
    assert evaluate_pairs(distances, is_same).separation > 0


# --------------------------------------------------------------- thresholds

def test_selected_threshold_separates_cleanly():
    distances = np.concatenate([np.full(50, 0.2), np.full(50, 0.8)])
    is_same = np.concatenate([np.ones(50, int), np.zeros(50, int)])
    threshold = select_threshold(distances, is_same)
    assert 0.2 <= threshold < 0.8
    assert accuracy_at(distances, is_same, threshold) == pytest.approx(1.0)


def test_accuracy_at_extremes():
    distances = np.array([0.1, 0.2, 0.8, 0.9])
    is_same = np.array([1, 1, 0, 0])
    assert accuracy_at(distances, is_same, 0.0) == 0.5    # reject everything
    assert accuracy_at(distances, is_same, 10.0) == 0.5   # accept everything
    assert accuracy_at(distances, is_same, 0.5) == 1.0


# ------------------------------------------------------------------- models

def test_arcface_matches_plain_cosine_at_zero_margin():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from ml.models import ArcMarginProduct

    torch.manual_seed(0)
    head = ArcMarginProduct(8, 4, scale=1.0, margin=0.0)
    emb = torch.randn(6, 8)
    labels = torch.tensor([0, 1, 2, 3, 0, 1])
    expected = F.linear(F.normalize(emb), F.normalize(head.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
    torch.testing.assert_close(head(emb, labels), expected, atol=1e-5, rtol=1e-4)


def test_arcface_penalises_the_true_class():
    """A positive margin must lower the true-class logit, never raise it."""
    torch = pytest.importorskip("torch")
    from ml.models import ArcMarginProduct

    torch.manual_seed(0)
    head = ArcMarginProduct(8, 4, scale=1.0, margin=0.4)
    emb = torch.randn(6, 8)
    labels = torch.tensor([0, 1, 2, 3, 0, 1])
    with torch.no_grad():
        margined = head(emb, labels)
        head.margin = 0.0
        head._cos_m, head._sin_m = 1.0, 0.0
        plain = head(emb, labels)
    true_margined = margined.gather(1, labels.view(-1, 1))
    true_plain = plain.gather(1, labels.view(-1, 1))
    assert torch.all(true_margined <= true_plain + 1e-5)


def test_head_starts_as_near_identity():
    """
    The residual head zero-inits its final layer so training begins from
    dlib's own geometry rather than random noise.
    """
    torch = pytest.importorskip("torch")
    from ml.models import EmbeddingHead

    head = EmbeddingHead(in_dim=16, hidden=32, out_dim=16).eval()
    x = torch.randn(8, 16)
    with torch.no_grad():
        assert torch.allclose(head.body(x), torch.zeros(8, 16), atol=1e-6)


def test_backbone_outputs_expected_shape():
    torch = pytest.importorskip("torch")
    from ml.models import FaceNetSmall

    model = FaceNetSmall(embedding_dim=128, width=8).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 112, 112))
    assert out.shape == (2, 128)
