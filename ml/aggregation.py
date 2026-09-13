"""
aggregation.py — How should several photographs of one person be combined?

    python -m ml.aggregation

Two strategies are in common use when a person is enrolled from k photographs:

  nearest      keep every embedding; a query's distance to the person is the
               distance to their *closest* stored photograph. This is what the
               system does today.
  centroid     average the L2-normalised embeddings into one representative
               vector per person, and compare against that. Cheaper, and the
               approach most tutorials recommend.

Which is better is an empirical question, not a matter of taste, and the answer
depends on k. This script settles it on identities the model never saw, at
k = 1..5, for both engines, and reports EER and TAR at a fixed false-accept
budget.

Protocol: LFW test-split identities with at least k+2 photographs. For each,
k are enrolled and the rest become genuine probes; impostor probes are drawn
from other identities. Identical probe sets are used for both strategies, so
the comparison isolates the aggregation choice.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np

from ml.evaluate import evaluate_pairs, l2_normalize
from ml.prepare import paths

logger = logging.getLogger("aggregation")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def person_scores(query: np.ndarray, gallery: np.ndarray, strategy: str, normalize: bool) -> float:
    """Distance from one query embedding to one person's enrolled set."""
    q = l2_normalize(query) if normalize else query
    g = l2_normalize(gallery) if normalize else gallery

    if strategy == "centroid":
        # Mean of the normalised vectors, renormalised: the standard
        # "representative embedding" construction.
        centre = g.mean(axis=0)
        centre = centre / (np.linalg.norm(centre) + 1e-10) if normalize else centre
        return float(np.linalg.norm(q - centre))

    # nearest: distance to the closest enrolled photograph
    return float(np.min(np.linalg.norm(g - q, axis=1)))


def run(embeddings: np.ndarray, labels: np.ndarray, positions: np.ndarray,
        k: int, normalize: bool, seed: int = 0) -> dict:
    """Score both strategies at enrolment size k on the same probe set."""
    rng = np.random.default_rng(seed)

    by_person: dict[int, list[int]] = {}
    for idx in positions:
        by_person.setdefault(int(labels[idx]), []).append(int(idx))

    eligible = {p: idx for p, idx in by_person.items() if len(idx) >= k + 2}
    if len(eligible) < 10:
        return {}

    gallery: dict[int, np.ndarray] = {}
    genuine_probes: list[tuple[int, int]] = []     # (probe index, true person)
    for person, idx in eligible.items():
        shuffled = list(rng.permutation(idx))
        gallery[person] = embeddings[shuffled[:k]]
        for probe in shuffled[k:k + 2]:
            genuine_probes.append((probe, person))

    # Impostors: identities excluded from the gallery entirely.
    outsiders = [p for p in by_person if p not in eligible]
    impostor_probes: list[int] = []
    for person in outsiders:
        impostor_probes.extend(by_person[person][:1])
    impostor_probes = impostor_probes[:len(genuine_probes)]
    if not impostor_probes:
        return {}

    out = {}
    for strategy in ("nearest", "centroid"):
        distances, is_same = [], []
        for probe, truth in genuine_probes:
            best = min(person_scores(embeddings[probe], g, strategy, normalize)
                       for g in gallery.values())
            # Genuine: the distance to the *correct* person is what a
            # verification decision would use.
            distances.append(person_scores(embeddings[probe], gallery[truth], strategy, normalize))
            is_same.append(1)
            del best
        for probe in impostor_probes:
            distances.append(min(person_scores(embeddings[probe], g, strategy, normalize)
                                 for g in gallery.values()))
            is_same.append(0)

        report = evaluate_pairs(np.array(distances), np.array(is_same))
        out[strategy] = {
            "auc": round(report.auc, 4),
            "eer": round(report.eer, 4),
            "accuracy": round(report.best_accuracy, 4),
            "tar_at_far_1pct": round(report.tar_at_far_1pct, 4),
            "threshold": round(report.best_threshold, 4),
        }

    out["_meta"] = {"k": k, "identities": len(eligible),
                    "genuine": len(genuine_probes), "impostor": len(impostor_probes)}
    return out


def sface_embeddings(positions: np.ndarray) -> np.ndarray | None:
    """Embed the test split with YuNet + SFace, if the models are present."""
    from src.opencv_engine import available
    if not available():
        logger.warning("YuNet/SFace models absent; skipping the opencv engine.")
        return None

    import cv2
    from src.opencv_engine import SFaceEmbedder, YuNetDetector, model_path

    detector = YuNetDetector()
    recognizer = cv2.FaceRecognizerSF.create(model_path("recognizer"), "")
    del SFaceEmbedder

    from ml.data import load_lfw
    images, _labels, _names = load_lfw(min_faces_per_person=2)
    out = np.zeros((len(positions), 128), dtype=np.float64)
    for i, pos in enumerate(positions):
        rgb = images[pos]
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        faces = detector.detect_faces(rgb)
        if len(faces) == 0:
            continue
        face = faces[np.argmax(faces[:, 2] * faces[:, 3])]
        out[i] = recognizer.feature(recognizer.alignCrop(bgr, face)).flatten()
        if (i + 1) % 500 == 0:
            logger.info("  embedded %d/%d", i + 1, len(positions))
    return out


def validation_positions(labels: np.ndarray) -> np.ndarray:
    """
    Image indices for identities held out of training but *not* in the test
    split — the only data a shipped threshold may be fitted on.
    """
    import json as _json
    from ml.train import split_train_val

    p = paths()
    with open(p["split"], encoding="utf-8") as f:
        split = _json.load(f)
    with open(p["identities"], encoding="utf-8") as f:
        identities = _json.load(f)

    train_ids = [identities.index(n) for n in split["train"]]
    train_mask = np.isin(labels, train_ids)
    raw = labels[train_mask]
    remap = {old: new for new, old in enumerate(sorted(set(raw.tolist())))}
    dense = np.array([remap[v] for v in raw], dtype=np.int64)
    _fit_mask, val_mask = split_train_val(dense, seed=0)
    return np.flatnonzero(train_mask)[val_mask]


def error_curve(embeddings, labels, positions, normalize, ks=(1, 2, 3, 4, 5)) -> list:
    """
    [threshold, FAR, FRR] over the useful range, pooled across enrolment sizes.

    Shipped to the frontend so the strictness control can state what a setting
    costs — "about 1 in 30 strangers may match" — instead of showing a number
    whose meaning depends on the engine.
    """
    distances, is_same = [], []
    for k in ks:
        by_person: dict[int, list[int]] = {}
        for idx in positions:
            by_person.setdefault(int(labels[idx]), []).append(int(idx))
        eligible = {p: i for p, i in by_person.items() if len(i) >= k + 2}
        if len(eligible) < 10:
            continue

        rng = np.random.default_rng(0)
        gallery, genuine = {}, []
        for person, idx in eligible.items():
            order = list(rng.permutation(idx))
            gallery[person] = embeddings[order[:k]]
            genuine.extend((pr, person) for pr in order[k:k + 2])
        outsiders = [by_person[p][0] for p in by_person if p not in eligible][:len(genuine)]

        for probe, truth in genuine:
            distances.append(person_scores(embeddings[probe], gallery[truth], "centroid", normalize))
            is_same.append(1)
        for probe in outsiders:
            distances.append(min(person_scores(embeddings[probe], g, "centroid", normalize)
                                 for g in gallery.values()))
            is_same.append(0)

    distances = np.array(distances)
    is_same = np.array(is_same)
    if len(distances) == 0:
        return []

    lo, hi = float(np.percentile(distances, 1)), float(np.percentile(distances, 99))
    out = []
    for t in np.linspace(lo * 0.6, hi * 1.15, 26):
        accepted = distances <= t
        far = float((accepted & (is_same == 0)).sum() / max((is_same == 0).sum(), 1))
        frr = float(((~accepted) & (is_same == 1)).sum() / max((is_same == 1).sum(), 1))
        out.append([round(float(t), 4), round(far, 4), round(frr, 4)])
    return out


def fit_threshold(embeddings, labels, positions, normalize, ks=(1, 2, 3, 4, 5)) -> float:
    """
    One threshold that serves every enrolment size.

    A person may be enrolled from one photograph or five, and the system
    cannot carry a different gate for each. Averaging the per-k optima weights
    them equally rather than letting whichever k happened to have most
    identities dominate.
    """
    chosen = []
    for k in ks:
        result = run(embeddings, labels, positions, k, normalize)
        if result:
            chosen.append(result["centroid"]["threshold"])
    return float(np.mean(chosen)) if chosen else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare enrolment aggregation strategies.")
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--fit", action="store_true",
                        help="Fit the shipped threshold on validation identities.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    p = paths()
    labels = np.load(p["labels"])
    pairs = np.load(p["pairs"])
    positions = pairs["test_positions"]

    engines: dict[str, tuple[np.ndarray, np.ndarray, bool]] = {}

    dlib_all = np.load(p["dlib"])
    # dlib vectors are not unit length and the shipped threshold lives in their
    # raw space, so they are compared unnormalised.
    engines["dlib"] = (dlib_all, positions, False)

    val_positions = validation_positions(labels) if args.fit else np.array([], dtype=int)
    needed = np.unique(np.concatenate([positions, val_positions])) if len(val_positions) else positions
    sface = sface_embeddings(needed)
    if sface is not None:
        # SFace is angular by construction; its embeddings are normalised.
        remapped = np.zeros((len(labels), 128))
        remapped[needed] = sface
        engines["opencv-sface"] = (remapped, positions, True)

    findings = {}
    for engine, (emb, pos, normalize) in engines.items():
        print()
        print(f"{engine}  (normalise={normalize})")
        print(f"  {'k':>2}  {'identities':>10}  {'nearest EER':>12}  {'centroid EER':>13}  {'winner':>9}")
        print("  " + "-" * 56)
        per_k = {}
        for k in range(1, args.max_k + 1):
            result = run(emb, labels, pos, k, normalize)
            if not result:
                continue
            near, cent = result["nearest"], result["centroid"]
            winner = "nearest" if near["eer"] < cent["eer"] else (
                "centroid" if cent["eer"] < near["eer"] else "tie")
            print(f"  {k:>2}  {result['_meta']['identities']:>10}  "
                  f"{near['eer'] * 100:>11.2f}%  {cent['eer'] * 100:>12.2f}%  {winner:>9}")
            per_k[k] = result
        findings[engine] = per_k

    if args.fit:
        print()
        print("Fitting the shipped threshold on VALIDATION identities")
        print("  (disjoint from both training and the test split above)")
        val = val_positions
        fitted = {}
        for engine, (emb, _pos, normalize) in engines.items():
            value = fit_threshold(emb, labels, val, normalize)
            fitted[engine] = round(float(value), 4)
            print(f"  {engine:14s} centroid threshold = {value:.4f}")
        findings["fitted_thresholds"] = fitted

        curves = {}
        for engine, (emb, _pos, normalize) in engines.items():
            key = "opencv-sface" if engine == "opencv-sface" else "dlib"
            curves[key] = error_curve(emb, labels, val, normalize)
        with open(os.path.join(RESULTS_DIR, "error_curves.json"), "w", encoding="utf-8") as f:
            json.dump(curves, f, indent=2)
        print(f"  error curves written to {os.path.join(RESULTS_DIR, 'error_curves.json')}")

    with open(os.path.join(RESULTS_DIR, "aggregation.json"), "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2)
    print()
    print(f"  written to {os.path.join(RESULTS_DIR, 'aggregation.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
