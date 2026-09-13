#!/usr/bin/env python3
"""
run_test.py — Enrol the test set, identify every probe, publish the results.

    python Test/run_test.py                  # default engine (dlib), 0.60
    python Test/run_test.py --engine opencv  # YuNet + SFace
    python Test/run_test.py --threshold 0.55

Protocol
--------
Each person in Test/enrolled/ is enrolled from two photographs. Every file in
Test/probe/ is then identified against that database:

  * a probe named <Person>_N.jpg should come back as that person — these are
    **different photographs** from the ones enrolled, so this measures
    recognition rather than recall of a stored image;
  * a probe named unknown_*.jpg is somebody who was never enrolled and must be
    rejected. Without these the accuracy figure would be meaningless: a system
    that simply accepts everyone scores perfectly on genuine probes alone.

Writes results/results.json and results/results.md, and uses a temporary
database so an existing enrolment is never touched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def expected_label(filename: str) -> str:
    """Ground truth from the filename: unknown_* is an impostor."""
    stem = os.path.splitext(filename)[0]
    if stem.lower().startswith("unknown"):
        return "Unknown"
    return stem.rsplit("_", 1)[0].replace("_", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the committed test set.")
    # No default of its own: the point of this script is to exercise whatever
    # the system actually ships with, so a default here would silently test a
    # configuration nobody runs.
    parser.add_argument("--engine", choices=["dlib", "opencv"], default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    from src.system import FaceRecognitionSystem

    enrolled_dir = os.path.join(HERE, "enrolled")
    probe_dir = os.path.join(HERE, "probe")
    if not os.path.isdir(enrolled_dir) or not os.path.isdir(probe_dir):
        print("Test set missing. Run: python Test/build_dataset.py")
        return 1

    db_path = os.path.join(tempfile.mkdtemp(prefix="facerec_test_"), "db.json")
    kwargs = {"threshold": args.threshold, "db_path": db_path}
    if args.engine:
        kwargs["engine"] = args.engine
    system = FaceRecognitionSystem(**kwargs)
    engine_used = system.engine
    threshold = system.matcher.threshold

    # ------------------------------------------------------------ enrol
    print(f"Enrolling (engine={engine_used}, threshold={threshold})")
    enrolment = {}
    for person_dir in sorted(os.listdir(enrolled_dir)):
        full = os.path.join(enrolled_dir, person_dir)
        if not os.path.isdir(full):
            continue
        name = person_dir.replace("_", " ")
        stored = 0
        for fname in sorted(os.listdir(full)):
            if fname.lower().endswith(IMAGE_EXTS):
                stored += system.enroll_from_image(name, os.path.join(full, fname)).enrolled
        enrolment[name] = stored
        print(f"  {name:22s} {stored} photo(s)")

    # ---------------------------------------------------------- identify
    print("\nIdentifying probes")
    rows = []
    for fname in sorted(os.listdir(probe_dir)):
        if not fname.lower().endswith(IMAGE_EXTS):
            continue
        truth = expected_label(fname)
        results = system.identify_from_image(os.path.join(probe_dir, fname))

        if not results:
            predicted, distance, confidence, runner_up = "No face detected", None, 0.0, None
        else:
            _loc, match = results[0]
            predicted = match.name
            distance = None if match.distance == float("inf") else round(match.distance, 4)
            confidence = round(match.confidence(), 4)
            ranked = sorted(match.scores.items(), key=lambda kv: kv[1])
            runner_up = (
                {"name": ranked[1][0], "distance": round(ranked[1][1], 4)}
                if len(ranked) > 1 else None
            )

        correct = predicted == truth
        rows.append({
            "probe": fname,
            "expected": truth,
            "predicted": predicted,
            "correct": correct,
            "distance": distance,
            "confidence": confidence,
            "runner_up": runner_up,
        })
        mark = "PASS" if correct else "FAIL"
        shown = "n/a" if distance is None else f"{distance:.3f}"
        print(f"  {mark}  {fname:26s} expected {truth:18s} got {predicted:18s} d={shown}")

    # ----------------------------------------------------------- metrics
    genuine = [r for r in rows if r["expected"] != "Unknown"]
    impostor = [r for r in rows if r["expected"] == "Unknown"]

    tp = sum(1 for r in genuine if r["correct"])
    fn = len(genuine) - tp                                   # missed or misnamed
    tn = sum(1 for r in impostor if r["correct"])
    fp = len(impostor) - tn                                  # stranger accepted

    accuracy = (tp + tn) / len(rows) if rows else 0.0
    far = fp / len(impostor) if impostor else 0.0            # false accept rate
    frr = fn / len(genuine) if genuine else 0.0              # false reject rate
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    summary = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "engine": engine_used,
        "threshold": threshold,
        "people_enrolled": len(enrolment),
        "enrolment_photos": sum(enrolment.values()),
        "probes": len(rows),
        "genuine_probes": len(genuine),
        "impostor_probes": len(impostor),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "far": round(far, 4),
        "frr": round(frr, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }

    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "probes": rows}, handle, indent=2)
    _write_markdown(os.path.join(out_dir, "results.md"), summary, rows, enrolment)

    print("\n" + "=" * 62)
    print(f"  Accuracy   {accuracy * 100:6.2f}%   ({tp + tn}/{len(rows)} probes correct)")
    print(f"  Precision  {precision:6.4f}")
    print(f"  Recall     {recall:6.4f}")
    print(f"  F1         {f1:6.4f}")
    print(f"  FAR        {far * 100:6.2f}%   (strangers wrongly accepted)")
    print(f"  FRR        {frr * 100:6.2f}%   (enrolled people wrongly rejected)")
    print("=" * 62)
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"\n  results/results.json and results/results.md written")
    return 0


def _write_markdown(path: str, summary: dict, rows: list[dict], enrolment: dict) -> None:
    lines = [
        "# Test results",
        "",
        f"Generated {summary['generated']} by `python Test/run_test.py`.",
        "",
        f"**{summary['people_enrolled']} people enrolled** from "
        f"{summary['enrolment_photos']} photographs, then "
        f"{summary['probes']} probes identified against them "
        f"({summary['genuine_probes']} genuine, {summary['impostor_probes']} impostor). "
        f"Engine `{summary['engine']}`, threshold {summary['threshold']}.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| Accuracy | **{summary['accuracy'] * 100:.2f}%** |",
        f"| Precision | {summary['precision']:.4f} |",
        f"| Recall | {summary['recall']:.4f} |",
        f"| F1 | {summary['f1']:.4f} |",
        f"| False accept rate | {summary['far'] * 100:.2f}% |",
        f"| False reject rate | {summary['frr'] * 100:.2f}% |",
        "",
        f"TP {summary['tp']} · FP {summary['fp']} · TN {summary['tn']} · FN {summary['fn']}",
        "",
        "## Enrolled",
        "",
        "| person | photos |",
        "|---|---:|",
    ]
    lines += [f"| {name} | {count} |" for name, count in sorted(enrolment.items())]
    lines += [
        "",
        "## Every probe",
        "",
        "Probe photographs are different images from the enrolment ones.",
        "`unknown_*` are people who were never enrolled and must be rejected.",
        "",
        "| probe | expected | predicted | distance | runner-up | |",
        "|---|---|---|---:|---|---|",
    ]
    for r in rows:
        distance = "n/a" if r["distance"] is None else f"{r['distance']:.3f}"
        runner = (f"{r['runner_up']['name']} ({r['runner_up']['distance']:.3f})"
                  if r["runner_up"] else "—")
        lines.append(
            f"| `{r['probe']}` | {r['expected']} | {r['predicted']} | "
            f"{distance} | {runner} | {'PASS' if r['correct'] else 'FAIL'} |"
        )
    lines += [
        "",
        "The runner-up column is the next closest enrolled person. A large gap "
        "between the match and the runner-up means the decision was not marginal.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
