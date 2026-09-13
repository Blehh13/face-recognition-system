"""
train.py — Training loops for both tracks.

    python -m ml.train --track head      # MLP over frozen dlib embeddings
    python -m ml.train --track scratch   # CNN + ArcFace from random init

Both tracks train only on the training identities and are scored only on the
held-out ones, so what gets reported is generalisation to strangers rather
than recall of the training set.

A validation slice is carved out of the *training identities* (again disjoint)
purely for early stopping and threshold selection. The test split is touched
once, at the end, by ml/benchmark.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ml.evaluate import evaluate_pairs, l2_normalize, pair_distances
from ml.models import ArcMarginProduct, EmbeddingHead, FaceNetSmall, count_parameters
from ml.prepare import CACHE_DIR, paths

logger = logging.getLogger("train")

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


# ----------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    """Frozen dlib vectors plus their identity label."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, jitter: float = 0.0):
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.jitter = jitter

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        x = self.features[i]
        if self.jitter > 0:
            # Gaussian jitter is the only augmentation available once the
            # image is gone; it discourages the head from overfitting exact
            # coordinates of the few training vectors.
            x = x + torch.randn_like(x) * self.jitter
        return x, self.labels[i]


class FaceCropDataset(Dataset):
    """112x112 RGB crops with light train-time augmentation."""

    def __init__(self, crops: np.ndarray, labels: np.ndarray, train: bool):
        self.crops = crops
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.train = train

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        img = self.crops[i]
        if self.train:
            if np.random.rand() < 0.5:
                img = img[:, ::-1]                      # mirror
            if np.random.rand() < 0.3:                  # brightness / contrast
                img = np.clip(img.astype(np.float32) * np.random.uniform(0.8, 1.2)
                              + np.random.uniform(-18, 18), 0, 255).astype(np.uint8)
            if np.random.rand() < 0.25:                 # random erase
                h = np.random.randint(12, 34)
                w = np.random.randint(12, 34)
                y = np.random.randint(0, img.shape[0] - h)
                x = np.random.randint(0, img.shape[1] - w)
                img = img.copy()
                img[y:y + h, x:x + w] = np.random.randint(0, 255)
        tensor = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32))
        return (tensor - 127.5) / 128.0, self.labels[i]


# ----------------------------------------------------------------------
# Split helpers
# ----------------------------------------------------------------------

def split_train_val(labels: np.ndarray, val_identity_fraction: float = 0.15, seed: int = 0):
    """
    Carve a validation set out of the training identities, disjoint again.

    Early stopping and threshold choice both need data the model has not been
    fitted on; using the test split for that would quietly contaminate the
    headline numbers.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(labels)
    counts = {u: int((labels == u).sum()) for u in unique}
    eligible = np.array([u for u in unique if counts[u] >= 2])
    n_val = max(1, int(len(eligible) * val_identity_fraction))
    val_ids = set(rng.choice(eligible, size=n_val, replace=False).tolist())

    val_mask = np.isin(labels, list(val_ids))
    return ~val_mask, val_mask


def build_val_pairs(labels: np.ndarray, n_pairs: int = 3000, seed: int = 1):
    """Balanced same/different pairs within a label array."""
    rng = np.random.default_rng(seed)
    by_id = {u: np.flatnonzero(labels == u) for u in np.unique(labels)}
    eligible = [u for u, idx in by_id.items() if len(idx) >= 2]
    if not eligible:
        return np.array([]), np.array([]), np.array([])

    left, right, same = [], [], []
    for _ in range(n_pairs // 2):
        u = eligible[rng.integers(len(eligible))]
        a, b = rng.choice(by_id[u], size=2, replace=False)
        left.append(a); right.append(b); same.append(1)
    ids = list(by_id)
    for _ in range(n_pairs // 2):
        ua, ub = rng.choice(ids, size=2, replace=False)
        left.append(rng.choice(by_id[ua])); right.append(rng.choice(by_id[ub])); same.append(0)
    return np.array(left), np.array(right), np.array(same)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train_model(
    track: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    p = paths()
    labels_all = np.load(p["labels"])
    with open(p["split"], encoding="utf-8") as f:
        split = json.load(f)
    with open(p["identities"], encoding="utf-8") as f:
        identities = json.load(f)

    train_ids = [identities.index(n) for n in split["train"]]
    train_mask = np.isin(labels_all, train_ids)
    labels_train_global = labels_all[train_mask]
    # Re-index identities to a dense 0..C-1 range for the classifier head.
    remap = {old: new for new, old in enumerate(sorted(set(labels_train_global.tolist())))}
    y = np.array([remap[v] for v in labels_train_global], dtype=np.int64)
    n_classes = len(remap)

    fit_mask, val_mask = split_train_val(y, seed=seed)
    logger.info("Training identities: %d | fit images: %d | val images: %d",
                n_classes, int(fit_mask.sum()), int(val_mask.sum()))

    # ------------------------------------------------------- build inputs
    if track == "head":
        features = np.load(p["dlib"])[train_mask]
        features = l2_normalize(features)
        train_ds = EmbeddingDataset(features[fit_mask], y[fit_mask], jitter=0.05)
        val_features = features[val_mask]
        model = EmbeddingHead(in_dim=features.shape[1], out_dim=128).to(device)
        num_workers = 0
    elif track == "scratch":
        crops = np.load(p["crops"], mmap_mode="r")[train_mask]
        crops = np.ascontiguousarray(crops)
        train_ds = FaceCropDataset(crops[fit_mask], y[fit_mask], train=True)
        val_crops = crops[val_mask]
        model = FaceNetSmall(embedding_dim=128).to(device)
        num_workers = 0
    else:
        raise ValueError(f"unknown track: {track}")

    logger.info("Model: %s (%s parameters)", type(model).__name__, f"{count_parameters(model):,}")

    margin_head = ArcMarginProduct(128, n_classes, scale=32.0, margin=0.3).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(margin_head.parameters()),
        lr=learning_rate, weight_decay=5e-4,
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, drop_last=True, pin_memory=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, epochs=epochs,
        steps_per_epoch=max(1, len(loader)), pct_start=0.25,
    )

    val_left, val_right, val_same = build_val_pairs(y[val_mask])
    history = []
    best_auc, best_state, best_epoch = -1.0, None, -1

    for epoch in range(1, epochs + 1):
        model.train(); margin_head.train()
        started = time.time()
        total_loss = correct = seen = 0

        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            emb = model(xb)
            logits = margin_head(emb, yb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(margin_head.parameters()), 5.0
            )
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(yb)

        # ---- validation: verification AUC on held-out identities ----
        model.eval()
        with torch.no_grad():
            source = val_features if track == "head" else val_crops
            val_emb = embed_with(model, source, track, device)
        if len(val_same):
            distances = pair_distances(val_emb, val_left, val_right)
            report = evaluate_pairs(distances, val_same)
            val_auc, val_eer = report.auc, report.eer
        else:
            val_auc = val_eer = float("nan")

        history.append({"epoch": epoch, "loss": total_loss / max(seen, 1),
                        "train_acc": correct / max(seen, 1), "val_auc": val_auc, "val_eer": val_eer})
        logger.info("epoch %2d/%d  loss %.4f  train-acc %.3f  val-AUC %.4f  val-EER %.3f  (%.1fs)",
                    epoch, epochs, history[-1]["loss"], history[-1]["train_acc"],
                    val_auc, val_eer, time.time() - started)

        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info("Best epoch %d with val AUC %.4f", best_epoch, best_auc)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{track}.pt")
    torch.save({
        "track": track,
        "state_dict": model.state_dict(),
        "embedding_dim": 128,
        "input_dim": 128 if track == "head" else None,
        "best_val_auc": best_auc,
        "best_epoch": best_epoch,
        "history": history,
    }, ckpt_path)
    logger.info("Saved %s", ckpt_path)

    with open(os.path.join(CHECKPOINT_DIR, f"{track}_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {"checkpoint": ckpt_path, "best_val_auc": best_auc, "history": history}


def embed_with(model: nn.Module, source: np.ndarray, track: str,
               device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Run a trained model over raw inputs and return embeddings as numpy."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(source), batch_size):
            chunk = source[i : i + batch_size]
            if track == "head":
                xb = torch.from_numpy(np.asarray(chunk, dtype=np.float32))
            else:
                arr = np.ascontiguousarray(np.asarray(chunk).transpose(0, 3, 1, 2)).astype(np.float32)
                xb = (torch.from_numpy(arr) - 127.5) / 128.0
            out.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a face embedding model.")
    parser.add_argument("--track", choices=["head", "scratch"], required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is present.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    defaults = {
        "head":    {"epochs": 40, "batch_size": 256, "lr": 1e-3},
        "scratch": {"epochs": 30, "batch_size": 64,  "lr": 3e-3},
    }[args.track]
    epochs = args.epochs or defaults["epochs"]
    batch_size = args.batch_size or defaults["batch_size"]
    lr = args.lr or defaults["lr"]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info("Device: %s", device if device.type == "cpu" else torch.cuda.get_device_name(0))

    train_model(args.track, epochs, batch_size, lr, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
