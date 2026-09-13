"""
calibrate.py — Fit the rejection threshold from data instead of folklore.

    python -m ml.calibrate

The system ships a threshold of 0.60, inherited from `face_recognition`'s
documented default. That default is a general-purpose suggestion, not a
measurement. This module fits the threshold on validation identities, then
reports — once — what it does on the untouched test identities, alongside what
0.60 does on the same pairs.

Fitting on validation and reporting on test is the whole point: a threshold
chosen on the set it is scored against always looks better than it will in
production.

The chosen value is written to ml/results/threshold.json, which
`src.matcher` loads at import so the deployed default follows the evidence.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

from ml.evaluate import accuracy_at, evaluate_pairs, pair_distances, select_threshold
from ml.prepare import paths
from ml.train import build_val_pairs, split_train_val

logger = logging.getLogger("calibrate")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SHIPPED_DEFAULT = 0.60


def rates_at(distances: np.ndarray, is_same: np.ndarray, threshold: float) -> dict:
    """Accuracy plus the two error rates that actually matter operationally."""
    accepted = distances <= threshold
    genuine = is_same == 1
    impostor = ~genuine
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_at(distances, is_same, threshold),
        # FAR: strangers wrongly accepted as an enrolled person.
        "far": float((accepted & impostor).sum() / max(impostor.sum(), 1)),
        # FRR: the enrolled person wrongly rejected.
        "frr": float(((~accepted) & genuine).sum() / max(genuine.sum(), 1)),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    p = paths()

    labels_all = np.load(p["labels"])
    embeddings = np.load(p["dlib"])
    with open(p["split"], encoding="utf-8") as f:
        split = json.load(f)
    with open(p["identities"], encoding="utf-8") as f:
        identities = json.load(f)

    # ---- fit on validation identities carved out of the training split ----
    train_ids = [identities.index(n) for n in split["train"]]
    train_mask = np.isin(labels_all, train_ids)
    raw = labels_all[train_mask]
    remap = {old: new for new, old in enumerate(sorted(set(raw.tolist())))}
    y = np.array([remap[v] for v in raw], dtype=np.int64)

    _fit_mask, val_mask = split_train_val(y, seed=0)
    val_features = embeddings[train_mask][val_mask]
    vl, vr, vs = build_val_pairs(y[val_mask], n_pairs=4000)
    val_distances = pair_distances(val_features, vl, vr, normalize=False)

    fitted = select_threshold(val_distances, vs)
    logger.info("Threshold fitted on %d validation identities: %.3f",
                len(np.unique(y[val_mask])), fitted)

    # ---- report once on the untouched test identities ----
    pairs = np.load(p["pairs"])
    positions = pairs["test_positions"]
    pos_of = {int(g): i for i, g in enumerate(positions)}
    tl = np.array([pos_of[int(v)] for v in pairs["left"]])
    tr = np.array([pos_of[int(v)] for v in pairs["right"]])
    ts = pairs["same"]

    test_features = embeddings[positions]
    test_distances = pair_distances(test_features, tl, tr, normalize=False)
    report = evaluate_pairs(test_distances, ts)

    shipped = rates_at(test_distances, ts, SHIPPED_DEFAULT)
    calibrated = rates_at(test_distances, ts, fitted)
    oracle = rates_at(test_distances, ts, report.best_threshold)

    payload = {
        "fitted_threshold": float(fitted),
        "shipped_threshold": SHIPPED_DEFAULT,
        "metric": "euclidean",
        "fitted_on": {
            "identities": int(len(np.unique(y[val_mask]))),
            "pairs": int(len(vs)),
            "source": "LFW validation identities (disjoint from train and test)",
        },
        "test": {
            "identities": len(split["test"]),
            "pairs": int(len(ts)),
            "shipped": shipped,
            "calibrated": calibrated,
            "oracle_best_on_test": oracle,
            "auc": report.auc,
            "eer": report.eer,
            "eer_threshold": report.eer_threshold,
        },
    }
    out_path = os.path.join(RESULTS_DIR, "threshold.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print()
    print("=" * 82)
    print("Threshold calibration — dlib embeddings, LFW, identity-disjoint test split")
    print("=" * 82)
    print(f"  fitted on validation identities : {fitted:.3f}")
    print(f"  shipped default                 : {SHIPPED_DEFAULT:.3f}")
    print()
    print(f"  {'':<26}{'threshold':>11}{'accuracy':>11}{'FAR':>10}{'FRR':>10}")
    print("  " + "-" * 68)
    for label, row in (("shipped (0.60)", shipped),
                       ("calibrated (fitted)", calibrated),
                       ("oracle (best on test)", oracle)):
        print(f"  {label:<26}{row['threshold']:>11.3f}{row['accuracy'] * 100:>10.2f}%"
              f"{row['far'] * 100:>9.2f}%{row['frr'] * 100:>9.2f}%")
    print("  " + "-" * 68)
    far_drop = shipped["far"] - calibrated["far"]
    print(f"  Calibrating cuts false accepts by {far_drop * 100:.2f} points "
          f"({shipped['far'] * 100:.2f}% -> {calibrated['far'] * 100:.2f}%)")
    print(f"  at the cost of {(calibrated['frr'] - shipped['frr']) * 100:+.2f} points of false rejects.")
    print("=" * 82)
    print(f"  Written to {out_path}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
