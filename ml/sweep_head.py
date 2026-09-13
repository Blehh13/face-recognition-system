"""
sweep_head.py — Does a learned head over dlib embeddings help at all?

    python -m ml.sweep_head

The first head we trained scored better than its own starting point on the
validation identities but *worse* than plain dlib on the test identities. That
is the signature of selecting on noise, not of a real gain — but it could also
just be one bad hyper-parameter choice. This sweep settles which.

Protocol:
  * every configuration is scored on validation identities only;
  * the single best-on-validation configuration is then scored once on test;
  * plain dlib is measured on the *same* validation pairs as a control, so the
    comparison is like-for-like.

Reporting the best test score across configurations would be tuning on the
test set, which is how papers accidentally claim improvements that evaporate.
"""

from __future__ import annotations

import itertools
import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.evaluate import evaluate_pairs, l2_normalize, pair_distances
from ml.models import ArcMarginProduct, EmbeddingHead
from ml.prepare import paths
from ml.train import EmbeddingDataset, build_val_pairs, embed_with, split_train_val

logger = logging.getLogger("sweep")


def run_one(features, y, fit_mask, val_mask, val_pairs, device,
            lr, weight_decay, hidden, dropout, margin, epochs, jitter):
    torch.manual_seed(0)
    np.random.seed(0)
    val_left, val_right, val_same = val_pairs

    n_classes = int(y.max()) + 1
    model = EmbeddingHead(in_dim=features.shape[1], hidden=hidden, out_dim=128, dropout=dropout).to(device)
    margin_head = ArcMarginProduct(128, n_classes, scale=32.0, margin=margin).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(margin_head.parameters()),
                            lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    loader = DataLoader(EmbeddingDataset(features[fit_mask], y[fit_mask], jitter=jitter),
                        batch_size=256, shuffle=True, drop_last=True)

    best_auc, best_state = -1.0, None
    for _ in range(epochs):
        model.train(); margin_head.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(margin_head(model(xb), yb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            emb = embed_with(model, features[val_mask].astype(np.float32), "head", device)
        auc = evaluate_pairs(pair_distances(emb, val_left, val_right), val_same).auc
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return best_auc, best_state


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cpu")  # the head is tiny; leave the GPU free
    p = paths()

    labels_all = np.load(p["labels"])
    with open(p["split"], encoding="utf-8") as f:
        split = json.load(f)
    with open(p["identities"], encoding="utf-8") as f:
        identities = json.load(f)

    train_ids = [identities.index(n) for n in split["train"]]
    train_mask = np.isin(labels_all, train_ids)
    raw = labels_all[train_mask]
    remap = {old: new for new, old in enumerate(sorted(set(raw.tolist())))}
    y = np.array([remap[v] for v in raw], dtype=np.int64)

    raw_features = np.load(p["dlib"])[train_mask].astype(np.float32)
    features = l2_normalize(raw_features).astype(np.float32)  # head input
    fit_mask, val_mask = split_train_val(y, seed=0)
    val_pairs = build_val_pairs(y[val_mask], n_pairs=4000)

    # Control: plain dlib on exactly these validation pairs.
    left, right, same = val_pairs
    dlib_val_auc = evaluate_pairs(pair_distances(raw_features[val_mask], left, right, normalize=False), same).auc
    logger.info("CONTROL — plain dlib on validation pairs: AUC %.4f", dlib_val_auc)

    grid = list(itertools.product(
        [3e-4, 1e-3],          # lr
        [5e-4, 5e-2],          # weight decay
        [256, 512],            # hidden
        [0.3],                 # dropout
        [0.1, 0.3],            # arcface margin
        [0.0, 0.05],           # input jitter
    ))
    logger.info("Sweeping %d configurations …", len(grid))

    results = []
    best = None
    for lr, wd, hidden, dropout, margin, jitter in grid:
        auc, state = run_one(features, y, fit_mask, val_mask, val_pairs, device,
                             lr, wd, hidden, dropout, margin, epochs=25, jitter=jitter)
        cfg = {"lr": lr, "weight_decay": wd, "hidden": hidden,
               "dropout": dropout, "margin": margin, "jitter": jitter}
        delta = auc - dlib_val_auc
        results.append({**cfg, "val_auc": auc, "delta_vs_dlib": delta})
        logger.info("lr=%.0e wd=%.0e h=%d m=%.1f j=%.2f -> val AUC %.4f (%+.4f vs dlib)",
                    lr, wd, hidden, margin, jitter, auc, delta)
        if best is None or auc > best[0]:
            best = (auc, cfg, state)

    best_auc, best_cfg, best_state = best
    logger.info("Best on validation: %s -> AUC %.4f (%+.4f vs dlib)",
                best_cfg, best_auc, best_auc - dlib_val_auc)

    # Score the winner once on the untouched test split.
    test_pairs = np.load(p["pairs"])
    positions = test_pairs["test_positions"]
    pos_of = {int(g): i for i, g in enumerate(positions)}
    tl = np.array([pos_of[int(v)] for v in test_pairs["left"]])
    tr = np.array([pos_of[int(v)] for v in test_pairs["right"]])
    ts = test_pairs["same"]

    raw_test = np.load(p["dlib"])[positions].astype(np.float32)
    test_features = l2_normalize(raw_test).astype(np.float32)  # head input
    dlib_test = evaluate_pairs(pair_distances(raw_test, tl, tr, normalize=False), ts)

    model = EmbeddingHead(in_dim=128, hidden=best_cfg["hidden"], out_dim=128,
                          dropout=best_cfg["dropout"]).to(device)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        head_emb = embed_with(model, test_features, "head", device)
    head_test = evaluate_pairs(pair_distances(head_emb, tl, tr, normalize=True), ts)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "head_sweep.json"), "w", encoding="utf-8") as f:
        json.dump({
            "dlib_val_auc": dlib_val_auc,
            "configurations": results,
            "best_config": best_cfg,
            "best_val_auc": best_auc,
            "test": {"dlib": dlib_test.to_dict(), "head": head_test.to_dict()},
        }, f, indent=2)

    print()
    print("=" * 78)
    print("Does a learned head over dlib embeddings help?")
    print("=" * 78)
    print(f"  configurations tried            : {len(grid)}")
    print(f"  plain dlib,  validation AUC     : {dlib_val_auc:.4f}")
    print(f"  best head,   validation AUC     : {best_auc:.4f}  ({best_auc - dlib_val_auc:+.4f})")
    print(f"  configs beating dlib on val     : {sum(r['delta_vs_dlib'] > 0 for r in results)}/{len(results)}")
    print("-" * 78)
    print("  Scored once on the untouched test identities:")
    print(f"    plain dlib   AUC {dlib_test.auc:.4f}   EER {dlib_test.eer*100:.2f}%")
    print(f"    best head    AUC {head_test.auc:.4f}   EER {head_test.eer*100:.2f}%")
    print("-" * 78)
    verdict = "HELPS" if head_test.auc > dlib_test.auc else "DOES NOT HELP"
    print(f"  VERDICT: a learned head {verdict} on held-out identities.")
    print("=" * 78)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
