"""
evaluate.py — Verification metrics for face embeddings.

Reports the measures the face-recognition literature uses, rather than bare
accuracy:

  ROC AUC     threshold-free ranking quality
  EER         the rate where false accepts equal false rejects
  TAR @ FAR   how many genuine pairs are accepted when impostor accepts are
              held to a fixed budget (0.1% / 1%) — the number that matters
              operationally, because deployments fix a FAR and live with the
              resulting TAR
  best accuracy and the threshold that produces it

A threshold chosen on the evaluation set is optimistic by construction, so
`select_threshold` fits it on a held-out validation slice instead.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class VerificationReport:
    n_pairs: int
    n_genuine: int
    n_impostor: int
    auc: float
    eer: float
    eer_threshold: float
    best_accuracy: float
    best_threshold: float
    tar_at_far_1pct: float
    tar_at_far_0p1pct: float
    mean_genuine_distance: float
    mean_impostor_distance: float
    separation: float          # impostor mean - genuine mean, in std units

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"AUC {self.auc:.4f} | EER {self.eer * 100:.2f}% | "
            f"acc {self.best_accuracy * 100:.2f}% @ d<{self.best_threshold:.3f} | "
            f"TAR@FAR=1% {self.tar_at_far_1pct * 100:.1f}% | "
            f"TAR@FAR=0.1% {self.tar_at_far_0p1pct * 100:.1f}% | "
            f"d' {self.separation:.2f}"
        )


def pair_distances(
    embeddings: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Euclidean distance between paired embeddings.

    `normalize` must match the space the threshold lives in, and getting it
    wrong silently rescales every distance:

      * dlib's 128-d vectors are **not** unit length (‖v‖ ≈ 1.42), and both
        `src.matcher` and dlib's documented 0.6 default operate on the raw
        vectors. Score them with normalize=False, or 0.6 gets compared against
        distances shrunk by ~1.42× and looks absurdly permissive.
      * the ArcFace-trained models here are explicitly angular — their loss
        lives on the unit hypersphere — so they are scored with normalize=True.

    Thresholds are therefore comparable *within* a row of the benchmark, not
    across rows; AUC, EER and TAR@FAR are unaffected by the choice of scale
    for a given model, so cross-model comparison uses those.
    """
    a, b = embeddings[left], embeddings[right]
    if normalize:
        a, b = l2_normalize(a), l2_normalize(b)
    return np.linalg.norm(a - b, axis=1)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)


def evaluate_pairs(distances: np.ndarray, is_same: np.ndarray) -> VerificationReport:
    """Score a set of pair distances against their ground-truth labels."""
    genuine = distances[is_same == 1]
    impostor = distances[is_same == 0]
    if len(genuine) == 0 or len(impostor) == 0:
        raise ValueError("Need both genuine and impostor pairs to evaluate.")

    # Smaller distance should mean "same", so negate to make it a score.
    auc = _roc_auc(-distances, is_same)

    thresholds = np.unique(np.concatenate([distances, [distances.min() - 1e-6, distances.max() + 1e-6]]))
    accepted = distances[None, :] <= thresholds[:, None]
    tar = (accepted & (is_same == 1)).sum(axis=1) / max(len(genuine), 1)   # true accept rate
    far = (accepted & (is_same == 0)).sum(axis=1) / max(len(impostor), 1)  # false accept rate
    frr = 1.0 - tar

    eer_idx = int(np.argmin(np.abs(far - frr)))
    accuracy = ((accepted & (is_same == 1)).sum(axis=1) + ((~accepted) & (is_same == 0)).sum(axis=1)) / len(is_same)
    best_idx = int(np.argmax(accuracy))

    pooled = np.sqrt((genuine.var() + impostor.var()) / 2) + 1e-10

    return VerificationReport(
        n_pairs=len(is_same),
        n_genuine=int(len(genuine)),
        n_impostor=int(len(impostor)),
        auc=float(auc),
        eer=float((far[eer_idx] + frr[eer_idx]) / 2),
        eer_threshold=float(thresholds[eer_idx]),
        best_accuracy=float(accuracy[best_idx]),
        best_threshold=float(thresholds[best_idx]),
        tar_at_far_1pct=float(_tar_at_far(tar, far, 0.01)),
        tar_at_far_0p1pct=float(_tar_at_far(tar, far, 0.001)),
        mean_genuine_distance=float(genuine.mean()),
        mean_impostor_distance=float(impostor.mean()),
        separation=float((impostor.mean() - genuine.mean()) / pooled),
    )


def select_threshold(distances: np.ndarray, is_same: np.ndarray) -> float:
    """
    Pick the accuracy-maximising threshold.

    Call this on a validation split, never on the split you report — a
    threshold tuned on the test set inflates the score it is scored against.
    """
    thresholds = np.unique(distances)
    accepted = distances[None, :] <= thresholds[:, None]
    accuracy = ((accepted & (is_same == 1)).sum(axis=1) + ((~accepted) & (is_same == 0)).sum(axis=1)) / len(is_same)
    return float(thresholds[int(np.argmax(accuracy))])


def accuracy_at(distances: np.ndarray, is_same: np.ndarray, threshold: float) -> float:
    predicted_same = distances <= threshold
    return float((predicted_same == (is_same == 1)).mean())


def _tar_at_far(tar: np.ndarray, far: np.ndarray, target_far: float) -> float:
    """Highest true-accept rate whose false-accept rate stays within budget."""
    allowed = far <= target_far
    return float(tar[allowed].max()) if allowed.any() else 0.0


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    AUC via the rank-sum identity, with ties averaged.

    Equivalent to the probability that a random genuine pair outranks a random
    impostor pair.
    """
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1

    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def plot_comparison(reports: dict[str, VerificationReport],
                    curves: dict[str, tuple[np.ndarray, np.ndarray]],
                    save_path: str) -> None:
    """Write an ROC comparison plus a distance-distribution panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for name, (distances, is_same) in curves.items():
        thresholds = np.unique(distances)
        accepted = distances[None, :] <= thresholds[:, None]
        tar = (accepted & (is_same == 1)).sum(axis=1) / max((is_same == 1).sum(), 1)
        far = (accepted & (is_same == 0)).sum(axis=1) / max((is_same == 0).sum(), 1)
        axes[0].plot(far, tar, label=f"{name} (AUC {reports[name].auc:.3f})")

    axes[0].set_xscale("log")
    axes[0].set_xlim(1e-4, 1.0)
    axes[0].set_xlabel("False accept rate")
    axes[0].set_ylabel("True accept rate")
    axes[0].set_title("ROC — identity-disjoint test split")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=9)

    best = max(reports, key=lambda k: reports[k].auc)
    distances, is_same = curves[best]
    axes[1].hist(distances[is_same == 1], bins=50, alpha=0.65, label="same person", color="#16a34a")
    axes[1].hist(distances[is_same == 0], bins=50, alpha=0.65, label="different people", color="#d97706")
    axes[1].axvline(reports[best].best_threshold, color="#2563eb", linestyle="--",
                    label=f"threshold {reports[best].best_threshold:.3f}")
    axes[1].set_xlabel("Embedding distance")
    axes[1].set_ylabel("Pairs")
    axes[1].set_title(f"Distance distributions — {best}")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
