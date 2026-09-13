"""
evaluator.py — Evaluation utilities for the Face Recognition System.

Metrics computed:
  - True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
  - Precision, Recall, F1
  - False Accept Rate (FAR), False Reject Rate (FRR)
  - Accuracy
  - Threshold sweep (to pick an optimal threshold)

Evaluation protocol
-------------------
A *probe set* is a list of (image_path, true_label) pairs where:
  - true_label = enrolled name  → genuine probe
  - true_label = "Unknown"      → impostor probe

The system's prediction is compared against the true label.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluate a FaceRecognitionSystem on a labelled probe set.
    """

    def __init__(self, system):
        self.system = system

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        probe_set: list[tuple[str, str]],
        save_dir: str = "evaluation",
    ) -> dict:
        """
        Run evaluation on *probe_set*.

        Parameters
        ----------
        probe_set : list of (image_path, true_label)
        save_dir  : directory to save plots and report

        Returns
        -------
        dict with all metrics.
        """
        os.makedirs(save_dir, exist_ok=True)
        y_true, y_pred, details = [], [], []

        logger.info("Evaluating %d probes …", len(probe_set))
        for image_path, true_label in tqdm(probe_set, desc="Evaluating"):
            try:
                face_results = self.system.identify_from_image(image_path)
            except Exception as exc:
                logger.warning("Failed on '%s': %s", image_path, exc)
                y_true.append(true_label)
                y_pred.append("Unknown")
                details.append({"path": image_path, "true": true_label,
                                 "pred": "Unknown", "error": str(exc)})
                continue

            if not face_results:
                pred_label = "Unknown"
                distance   = float("inf")
                confidence = 0.0
            else:
                _, result    = face_results[0]  # take primary face
                pred_label   = result.name
                distance     = result.distance
                confidence   = result.confidence()

            y_true.append(true_label)
            y_pred.append(pred_label)
            details.append({
                "path": image_path,
                "true": true_label,
                "pred": pred_label,
                "distance": distance,
                "confidence": confidence,
                "correct": true_label == pred_label,
            })

        metrics = self._compute_metrics(y_true, y_pred, details, save_dir)
        self._save_report(metrics, details, save_dir)
        return metrics

    # ------------------------------------------------------------------
    # Threshold sweep
    # ------------------------------------------------------------------

    def threshold_sweep(
        self,
        probe_set: list[tuple[str, str]],
        thresholds: list[float] | None = None,
        metric: str = "euclidean",
        save_dir: str = "evaluation",
    ) -> dict:
        """
        Evaluate across multiple thresholds and plot FAR/FRR curves.
        Returns the threshold with the best F1 score.
        """
        if thresholds is None:
            thresholds = [round(t, 2) for t in np.arange(0.30, 0.90, 0.05)]

        os.makedirs(save_dir, exist_ok=True)

        # Pre-compute distances once
        logger.info("Pre-computing embeddings for %d probes …", len(probe_set))
        probe_data = []
        for image_path, true_label in tqdm(probe_set, desc="Embedding"):
            try:
                import face_recognition
                image = face_recognition.load_image_file(image_path)
                locs  = self.system.detector.detect(image)
                if not locs:
                    probe_data.append((None, true_label))
                    continue
                embs = self.system.embedder.embed(image, face_locations=locs)
                probe_data.append((embs[0], true_label))
            except Exception:
                probe_data.append((None, true_label))

        db = self.system.database.get_all_embeddings()
        results_by_threshold = {}

        for thresh in thresholds:
            from src.matcher import FaceMatcher
            matcher = FaceMatcher(threshold=thresh, metric=metric)
            y_true, y_pred = [], []
            for emb, true_label in probe_data:
                if emb is None:
                    y_true.append(true_label)
                    y_pred.append("Unknown")
                    continue
                result = matcher.match(emb, db)
                y_true.append(true_label)
                y_pred.append(result.name)
            results_by_threshold[thresh] = self._basic_metrics(y_true, y_pred)

        # Plot
        thresholds_list = sorted(results_by_threshold.keys())
        far_values  = [results_by_threshold[t]["far"]  for t in thresholds_list]
        frr_values  = [results_by_threshold[t]["frr"]  for t in thresholds_list]
        f1_values   = [results_by_threshold[t]["f1"]   for t in thresholds_list]
        acc_values  = [results_by_threshold[t]["accuracy"] for t in thresholds_list]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(thresholds_list, far_values, "r-o", label="FAR")
        axes[0].plot(thresholds_list, frr_values, "b-o", label="FRR")
        axes[0].axvline(x=0.60, color="gray", linestyle="--", label="Default (0.60)")
        axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Rate")
        axes[0].set_title("FAR vs FRR Curve"); axes[0].legend(); axes[0].grid(True)

        axes[1].plot(thresholds_list, f1_values, "g-o", label="F1")
        axes[1].plot(thresholds_list, acc_values, "m-o", label="Accuracy")
        axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("Score")
        axes[1].set_title("F1 & Accuracy vs Threshold"); axes[1].legend(); axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "threshold_sweep.png"), dpi=150)
        plt.close()

        best_thresh = max(results_by_threshold, key=lambda t: results_by_threshold[t]["f1"])
        logger.info("Best threshold by F1: %.2f", best_thresh)
        return {"best_threshold": best_thresh, "results": results_by_threshold}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        y_true: list[str],
        y_pred: list[str],
        details: list[dict],
        save_dir: str,
    ) -> dict:
        metrics = self._basic_metrics(y_true, y_pred)
        enrolled = self.system.list_enrolled()
        all_labels = sorted(set(y_true + y_pred))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        fig, ax = plt.subplots(figsize=(max(6, len(all_labels)), max(5, len(all_labels) - 1)))
        sns.heatmap(
            cm, annot=True, fmt="d", xticklabels=all_labels,
            yticklabels=all_labels, cmap="Blues", ax=ax
        )
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
        plt.close()

        # Per-class report
        report = classification_report(y_true, y_pred, labels=all_labels, output_dict=True, zero_division=0)
        metrics["classification_report"] = report
        logger.info("Accuracy: %.2f%%", metrics["accuracy"] * 100)
        logger.info("F1 (weighted): %.4f", metrics["f1"])
        return metrics

    @staticmethod
    def _basic_metrics(y_true: list[str], y_pred: list[str]) -> dict:
        correct = sum(t == p for t, p in zip(y_true, y_pred))
        n = len(y_true)
        accuracy = correct / n if n > 0 else 0.0

        # FAR: known accepted as wrong person (impostor accepted)
        # FRR: known rejected (genuine rejected)
        tp = fp = tn = fn = 0
        for t, p in zip(y_true, y_pred):
            genuine = t != "Unknown"
            accepted = p != "Unknown"
            correct_id = t == p

            if genuine and accepted and correct_id:
                tp += 1
            elif genuine and (not accepted or not correct_id):
                fn += 1
            elif not genuine and not accepted:
                tn += 1
            elif not genuine and accepted:
                fp += 1

        total_genuine   = tp + fn
        total_impostor  = tn + fp
        far = fp / total_impostor if total_impostor > 0 else 0.0
        frr = fn / total_genuine  if total_genuine  > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "far": far,
            "frr": frr,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n_samples": n,
        }

    @staticmethod
    def _save_report(metrics: dict, details: list[dict], save_dir: str) -> None:
        report = {
            "summary": {k: v for k, v in metrics.items() if k != "classification_report"},
            "details": details,
        }
        path = os.path.join(save_dir, "evaluation_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Report saved to '%s'", path)
