#!/usr/bin/env python3
"""
demo.py — End-to-end demonstration using the LFW (Labeled Faces in the Wild)
dataset subset OR synthetic images (no internet required).

What this demo does:
  1. Downloads a small subset of LFW (optional, requires internet)
  2. Falls back to synthetic cartoon faces if LFW unavailable
  3. Enrolls persons
  4. Runs identification on probe images
  5. Prints results + saves evaluation report
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("demo")


def run_demo_synthetic():
    """Fully offline demo using synthetic cartoon face images."""
    logger.info("=== Running SYNTHETIC demo (no real faces, no internet needed) ===")

    # 1. Generate sample data
    logger.info("Step 1: Generating synthetic sample data …")
    from generate_sample_data import generate
    generate(out_dir="sample_data")

    # 2. Init system
    logger.info("Step 2: Initialising Face Recognition System …")
    from src.system import FaceRecognitionSystem
    sys_ = FaceRecognitionSystem(threshold=0.60, detection_model="hog")

    # 3. Enroll
    logger.info("Step 3: Enrolling persons …")
    persons = ["Alice", "Bob", "Charlie"]
    for person in persons:
        person_dir = os.path.join("sample_data", person)
        count = sys_.enroll_from_directory(person, person_dir)
        logger.info("  Enrolled %d image(s) for '%s'", count, person)

    logger.info("DB stats: %s", sys_.db_stats())

    # 4. Identify probe images
    logger.info("Step 4: Identifying probe images …")
    probe_dir = os.path.join("sample_data", "probe")
    for fname in sorted(os.listdir(probe_dir)):
        if not fname.lower().endswith((".jpg", ".png")):
            continue
        path = os.path.join(probe_dir, fname)
        results = sys_.identify_from_image(path)
        if not results:
            logger.info("  [NO FACE]  %s", fname)
        for (top, right, bottom, left), result in results:
            status = "✓ KNOWN" if result.is_known else "✗ UNKNOWN"
            logger.info(
                "  %s → %-15s  dist=%.3f  conf=%.0f%%  (file: %s)",
                status, result.name, result.distance,
                result._confidence() * 100, fname
            )

    # 5. Evaluation
    logger.info("Step 5: Running evaluation …")
    import csv
    probe_set = []
    with open("probes.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            probe_set.append((row["image_path"], row["label"]))

    from src.evaluator import Evaluator
    ev = Evaluator(sys_)
    metrics = ev.evaluate(probe_set, save_dir="evaluation")

    logger.info("=== Evaluation Results ===")
    logger.info("  Accuracy  : %.2f%%", metrics["accuracy"] * 100)
    logger.info("  Precision : %.4f",   metrics["precision"])
    logger.info("  Recall    : %.4f",   metrics["recall"])
    logger.info("  F1 Score  : %.4f",   metrics["f1"])
    logger.info("  FAR       : %.4f",   metrics["far"])
    logger.info("  FRR       : %.4f",   metrics["frr"])
    logger.info("Plots → evaluation/confusion_matrix.png")
    logger.info("Report → evaluation/evaluation_report.json")
    logger.info("=== Demo complete ===")


if __name__ == "__main__":
    run_demo_synthetic()
