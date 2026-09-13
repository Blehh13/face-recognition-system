"""
prepare.py — Download LFW, build the identity-disjoint split, cache embeddings.

Run once before training:

    python -m ml.prepare

Everything downstream reads the artefacts this writes into ml/cache/, so the
expensive work (a ~200 MB download and ~9k dlib forward passes) happens once.

Note on Windows: the embedding step uses a process pool, and Windows spawns
fresh interpreters that re-import the __main__ module. All work therefore sits
behind `if __name__ == "__main__"` — without that guard each child re-runs the
download and the pool dies with BrokenProcessPool.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np

from ml.data import crop_faces, identity_disjoint_split, load_lfw, make_verification_pairs

logger = logging.getLogger("prepare")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def paths() -> dict[str, str]:
    return {
        "crops": os.path.join(CACHE_DIR, "crops_112.npy"),
        "labels": os.path.join(CACHE_DIR, "labels.npy"),
        "identities": os.path.join(CACHE_DIR, "identities.json"),
        "dlib": os.path.join(CACHE_DIR, "dlib_embeddings.npy"),
        "split": os.path.join(CACHE_DIR, "split.json"),
        "pairs": os.path.join(CACHE_DIR, "test_pairs.npz"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the LFW training artefacts.")
    parser.add_argument("--min-faces", type=int, default=2,
                        help="Drop identities with fewer photographs than this.")
    parser.add_argument("--test-fraction", type=float, default=0.25,
                        help="Share of identities held out for evaluation.")
    parser.add_argument("--pairs", type=int, default=6000,
                        help="Verification pairs to draw from the test split.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Processes for dlib embedding (default: min(cores, 12)).")
    parser.add_argument("--force", action="store_true", help="Recompute even if cached.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = paths()

    # ---------------------------------------------------------------- data
    images, labels, identities = load_lfw(min_faces_per_person=args.min_faces)
    logger.info("LFW: %d images, %d identities", len(images), len(identities))

    train, test = identity_disjoint_split(
        images, labels, identities, test_fraction=args.test_fraction
    )
    logger.info(train.describe())
    logger.info(test.describe())

    overlap = set(train.identities) & set(test.identities)
    if overlap:
        raise AssertionError(f"identity leaked across splits: {sorted(overlap)[:5]}")
    logger.info("Verified: no identity appears in both splits.")

    # ------------------------------------------------------------- crops
    if args.force or not os.path.exists(p["crops"]):
        logger.info("Cropping and resizing to 112x112 …")
        np.save(p["crops"], crop_faces(images, size=112))
    np.save(p["labels"], labels)
    with open(p["identities"], "w", encoding="utf-8") as f:
        json.dump(identities, f)

    with open(p["split"], "w", encoding="utf-8") as f:
        json.dump({"train": train.identities, "test": test.identities}, f, indent=2)

    # -------------------------------------------------------------- pairs
    left, right, same = make_verification_pairs(test, n_pairs=args.pairs, seed=0)
    # Store pair indices against the *global* image array so every model can be
    # scored on exactly the same comparisons regardless of how it loads data.
    test_positions = np.flatnonzero(np.isin(labels, [identities.index(n) for n in test.identities]))
    np.savez(
        p["pairs"],
        left=test_positions[left],
        right=test_positions[right],
        same=same,
        test_positions=test_positions,
    )
    logger.info("Wrote %d verification pairs (%d genuine / %d impostor)",
                len(same), int(same.sum()), int((1 - same).sum()))

    # --------------------------------------------------------- embeddings
    from ml.embed import load_or_compute

    if args.force and os.path.exists(p["dlib"]):
        os.remove(p["dlib"])
    embeddings = load_or_compute(p["dlib"], images, workers=args.workers)
    logger.info("dlib embeddings: %s", embeddings.shape)

    logger.info("Artefacts ready in %s", CACHE_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
