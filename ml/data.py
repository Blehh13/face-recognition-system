"""
data.py — LFW loading, identity-disjoint splits, and verification pairs.

The one rule this module exists to enforce: **no identity appears in both the
training and the evaluation split.**

Face recognition is an open-set problem. The deployed system meets people it
was never trained on, so a benchmark that shares identities between train and
test measures memorisation and reports a number that will not survive contact
with reality. Splitting by identity — not by image — is what makes the
reported figures mean something.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# LFW images are 250x250 with the face roughly centred by the funnelling
# process. This window trims the background without cutting off chins.
LFW_CROP = (slice(61, 189), slice(61, 189))


@dataclass
class Split:
    """One identity-disjoint partition of the dataset."""
    name: str
    images: np.ndarray          # uint8, (N, H, W, 3), RGB
    labels: np.ndarray          # int, (N,) — index into `identities`
    identities: list[str]

    def __len__(self) -> int:
        return len(self.labels)

    def counts(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=len(self.identities))

    def describe(self) -> str:
        c = self.counts()
        return (
            f"{self.name}: {len(self)} images, {len(self.identities)} identities, "
            f"{c.min()}–{c.max()} images each (median {int(np.median(c))})"
        )


def load_lfw(min_faces_per_person: int = 2, color: bool = True) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Fetch LFW through scikit-learn and return (images uint8 RGB, labels, names).

    The first call downloads ~200 MB into scikit-learn's data cache; later
    calls read from disk.
    """
    from sklearn.datasets import fetch_lfw_people

    logger.info("Loading LFW (min_faces_per_person=%d) …", min_faces_per_person)
    lfw = fetch_lfw_people(
        min_faces_per_person=min_faces_per_person,
        color=color,
        resize=1.0,
        slice_=(slice(0, 250), slice(0, 250)),
    )
    images = (lfw.images * 255).round().astype(np.uint8)
    return images, lfw.target.astype(np.int64), list(lfw.target_names)


def _stable_hash(name: str) -> int:
    """
    Deterministic hash that does not depend on PYTHONHASHSEED.

    Python's built-in hash() is randomised per process, so using it to assign
    identities to splits would silently reshuffle the benchmark between runs
    and make results incomparable.
    """
    return int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:12], 16)


def identity_disjoint_split(
    images: np.ndarray,
    labels: np.ndarray,
    identities: list[str],
    test_fraction: float = 0.25,
    min_train_images: int = 3,
    min_test_images: int = 2,
) -> tuple[Split, Split]:
    """
    Partition by identity so that no person appears in both splits.

    Assignment is by a stable hash of the person's name, so the split is
    reproducible across runs and machines without storing an index file.

    Parameters
    ----------
    test_fraction : share of identities held out for evaluation.
    min_train_images : identities below this are dropped from training — a
        single photo teaches a metric-learning loss nothing about
        intra-person variation.
    min_test_images : identities below this are dropped from evaluation, since
        a genuine (same-person) pair needs at least two photographs.
    """
    counts = np.bincount(labels, minlength=len(identities))
    cutoff = test_fraction * (2 ** 48)

    train_ids, test_ids = [], []
    for idx, name in enumerate(identities):
        if counts[idx] == 0:
            continue
        if _stable_hash(name) % (2 ** 48) < cutoff:
            if counts[idx] >= min_test_images:
                test_ids.append(idx)
        elif counts[idx] >= min_train_images:
            train_ids.append(idx)

    overlap = set(train_ids) & set(test_ids)
    assert not overlap, f"identity leaked across splits: {overlap}"

    def build(name: str, id_list: list[int]) -> Split:
        id_list = sorted(id_list)
        remap = {old: new for new, old in enumerate(id_list)}
        mask = np.isin(labels, id_list)
        return Split(
            name=name,
            images=images[mask],
            labels=np.array([remap[v] for v in labels[mask]], dtype=np.int64),
            identities=[identities[i] for i in id_list],
        )

    return build("train", train_ids), build("test", test_ids)


def make_verification_pairs(
    split: Split,
    n_pairs: int = 6000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a balanced same/different pair list for verification.

    Returns (left_index, right_index, is_same) as arrays of equal length, with
    an even number of genuine and impostor pairs. Pairs are drawn from a fixed
    seed so every model is scored on exactly the same comparisons.
    """
    rng = np.random.default_rng(seed)
    by_identity: dict[int, np.ndarray] = {
        label: np.flatnonzero(split.labels == label) for label in np.unique(split.labels)
    }
    eligible = [label for label, idx in by_identity.items() if len(idx) >= 2]
    if not eligible:
        raise ValueError("No identity has two or more images; cannot form genuine pairs.")

    half = n_pairs // 2
    left, right, same = [], [], []

    # Genuine pairs: two different photographs of one person.
    for _ in range(half):
        label = eligible[rng.integers(len(eligible))]
        a, b = rng.choice(by_identity[label], size=2, replace=False)
        left.append(a)
        right.append(b)
        same.append(1)

    # Impostor pairs: photographs of two different people.
    labels_present = list(by_identity)
    for _ in range(half):
        la, lb = rng.choice(labels_present, size=2, replace=False)
        left.append(rng.choice(by_identity[la]))
        right.append(rng.choice(by_identity[lb]))
        same.append(0)

    return np.array(left), np.array(right), np.array(same, dtype=np.int64)


def crop_faces(images: np.ndarray, size: int = 112) -> np.ndarray:
    """
    Centre-crop LFW's funnelled frames to the face and resize to `size`.

    LFW is already roughly aligned, so a fixed window is enough and avoids
    making the whole pipeline depend on a detector that may miss a frame.
    """
    import cv2

    cropped = images[:, LFW_CROP[0], LFW_CROP[1], :]
    if cropped.shape[1] == size and cropped.shape[2] == size:
        return np.ascontiguousarray(cropped)
    out = np.empty((len(cropped), size, size, 3), dtype=np.uint8)
    for i, frame in enumerate(cropped):
        out[i] = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    return out
