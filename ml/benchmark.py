"""
benchmark.py — Score every approach on one identity-disjoint test split.

    python -m ml.benchmark

Compares:
  dlib-baseline   the frozen 128-d embeddings the product ships today
  head            the residual MLP trained over those embeddings
  scratch         the CNN + ArcFace trained from random initialisation

All three are scored on the *same* pairs, drawn once with a fixed seed from
identities none of them were trained on. Thresholds quoted here are the
accuracy-optimal ones on the test split and are therefore slightly optimistic
as operating points — `ml/calibrate.py` fits the deployable threshold on a
validation split instead.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import torch

from ml.evaluate import VerificationReport, evaluate_pairs, l2_normalize, pair_distances, plot_comparison
from ml.prepare import CACHE_DIR, paths

logger = logging.getLogger("benchmark")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def load_checkpoint(track: str, device: torch.device):
    """Rebuild a trained model from its checkpoint, or return None if absent."""
    path = os.path.join(CHECKPOINT_DIR, f"{track}.pt")
    if not os.path.exists(path):
        logger.warning("No checkpoint for '%s' at %s — skipping.", track, path)
        return None

    from ml.models import EmbeddingHead, FaceNetSmall

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = EmbeddingHead(in_dim=128, out_dim=128) if track == "head" else FaceNetSmall(embedding_dim=128)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


# dlib vectors are not unit length and the shipped threshold lives in their
# raw space; the ArcFace-trained models are angular by construction.
NORMALIZE = {"dlib-baseline": False, "head": True, "scratch": True}


def embeddings_for(track: str, device: torch.device, positions: np.ndarray) -> np.ndarray | None:
    """Produce test-split embeddings for one approach, in that model's own space."""
    p = paths()

    if track == "dlib-baseline":
        return np.load(p["dlib"])[positions]

    model = load_checkpoint(track, device)
    if model is None:
        return None

    from ml.train import embed_with

    if track == "head":
        source = l2_normalize(np.load(p["dlib"])[positions]).astype(np.float32)
    else:
        source = np.ascontiguousarray(np.load(p["crops"], mmap_mode="r")[positions])
    return l2_normalize(embed_with(model, source, track, device))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark all approaches on the test split.")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    p = paths()

    pairs = np.load(p["pairs"])
    positions = pairs["test_positions"]
    # Pair indices are stored against the global image array; remap them onto
    # the compact per-split arrays the models actually consume.
    position_of = {int(g): i for i, g in enumerate(positions)}
    left = np.array([position_of[int(v)] for v in pairs["left"]])
    right = np.array([position_of[int(v)] for v in pairs["right"]])
    same = pairs["same"]

    with open(p["split"], encoding="utf-8") as f:
        split = json.load(f)
    logger.info("Test split: %d identities, %d images, %d pairs (%d genuine)",
                len(split["test"]), len(positions), len(same), int(same.sum()))

    reports: dict[str, VerificationReport] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for track in ("dlib-baseline", "head", "scratch"):
        emb = embeddings_for(track, device, positions)
        if emb is None:
            continue
        distances = pair_distances(emb, left, right, normalize=NORMALIZE[track])
        report = evaluate_pairs(distances, same)
        reports[track] = report
        curves[track] = (distances, same)
        logger.info("%-14s %s", track, report.summary())

    if not reports:
        logger.error("Nothing to benchmark. Run `python -m ml.prepare` first.")
        return 1

    plot_path = os.path.join(RESULTS_DIR, "comparison.png")
    plot_comparison(reports, curves, plot_path)

    payload = {
        "test_identities": len(split["test"]),
        "test_images": int(len(positions)),
        "pairs": int(len(same)),
        "results": {k: v.to_dict() for k, v in reports.items()},
    }
    with open(os.path.join(RESULTS_DIR, "benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print()
    print("=" * 108)
    print(f"{'approach':<16}{'AUC':>8}{'EER':>9}{'accuracy':>11}{'thresh':>9}"
          f"{'TAR@FAR=1%':>13}{'TAR@FAR=0.1%':>15}{'d-prime':>10}")
    print("-" * 108)
    for name, r in sorted(reports.items(), key=lambda kv: -kv[1].auc):
        print(f"{name:<16}{r.auc:>8.4f}{r.eer * 100:>8.2f}%{r.best_accuracy * 100:>10.2f}%"
              f"{r.best_threshold:>9.3f}{r.tar_at_far_1pct * 100:>12.1f}%"
              f"{r.tar_at_far_0p1pct * 100:>14.1f}%{r.separation:>10.2f}")
    print("=" * 108)
    print(f"Plot:  {plot_path}")
    print(f"JSON:  {os.path.join(RESULTS_DIR, 'benchmark.json')}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
