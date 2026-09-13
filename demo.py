#!/usr/bin/env python3
"""
demo.py - End-to-end demonstration of the Face Recognition System.

Three modes, in descending order of how much they actually prove:

  --data-dir DIR   Real photos you supply. The honest demo.
                   Layout:  DIR/<Person Name>/*.jpg   (enrolment)
                            DIR/probe/*.jpg           (queries)
                   Probe files are labelled by filename prefix: a file named
                   "Alice_01.jpg" is expected to identify as "Alice";
                   anything starting with "unknown" is expected to be
                   rejected.

  --lfw            Download a small LFW subset via scikit-learn and evaluate
                   on real faces. Needs internet on first run (~200 MB cache).

  --smoke          Offline plumbing check on synthetic images. Verifies the
                   pipeline runs end to end. It does NOT measure accuracy -
                   see the note under "Why --smoke reports no metrics".

Why --smoke reports no metrics
------------------------------
The synthetic images are drawn with OpenCV primitives. dlib's HOG detector
finds a face in only a handful of them, and the ones it does find are nearly
identical to each other, so distinct "people" collapse to distances of ~0.1
and cross-match. Reporting accuracy on that set produces numbers that describe
the cartoon generator, not this system. Earlier versions of this file printed
those numbers anyway (F1 = 0.0000 with one person silently identified as
another); it now refuses to.
"""

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("demo")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ----------------------------------------------------------------------
# Shared reporting
# ----------------------------------------------------------------------

def _report(metrics: dict) -> None:
    print()
    print("=" * 46)
    print(f"  Accuracy  : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  FAR       : {metrics['far']:.4f}")
    print(f"  FRR       : {metrics['frr']:.4f}")
    print("=" * 46)
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  TN={metrics['tn']}  FN={metrics['fn']}")
    print()


def _identify_and_print(system, paths: list[str]) -> None:
    for path in paths:
        results = system.identify_from_image(path)
        name = os.path.basename(path)
        if not results:
            logger.info("  [no face]  %s", name)
            continue
        for _loc, result in results:
            status = "KNOWN  " if result.is_known else "UNKNOWN"
            distance = "n/a" if result.distance == float("inf") else f"{result.distance:.3f}"
            logger.info(
                "  %s -> %-18s dist=%-6s conf=%3.0f%%  (%s)",
                status, result.name, distance, result.confidence() * 100, name,
            )


# ----------------------------------------------------------------------
# Mode: real photos from a directory
# ----------------------------------------------------------------------

def run_data_dir(data_dir: str, threshold: float) -> int:
    from src.evaluator import Evaluator
    from src.system import FaceRecognitionSystem

    if not os.path.isdir(data_dir):
        logger.error("Not a directory: %s", data_dir)
        return 1

    people = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.lower() != "probe"
    )
    if not people:
        logger.error(
            "No person folders in '%s'. Expected %s/<Person Name>/*.jpg", data_dir, data_dir
        )
        return 1

    db_path = os.path.join(data_dir, "_demo_db.json")
    if os.path.exists(db_path):
        os.remove(db_path)
    system = FaceRecognitionSystem(threshold=threshold, db_path=db_path)

    logger.info("Enrolling %d people from '%s' ...", len(people), data_dir)
    for person in people:
        count = system.enroll_from_directory(person, os.path.join(data_dir, person))
        logger.info("  %-20s %d face(s)", person, count)

    if system.db_stats()["total_embeddings"] == 0:
        logger.error("Nothing enrolled - no detectable faces in those photos.")
        return 1

    probe_dir = os.path.join(data_dir, "probe")
    if not os.path.isdir(probe_dir):
        logger.warning("No '%s' folder, so there is nothing to evaluate.", probe_dir)
        return 0

    probes = []
    for fname in sorted(os.listdir(probe_dir)):
        if not fname.lower().endswith(IMAGE_EXTS):
            continue
        stem = os.path.splitext(fname)[0]
        label = "Unknown" if stem.lower().startswith("unknown") else stem.rsplit("_", 1)[0]
        probes.append((os.path.join(probe_dir, fname), label))

    if not probes:
        logger.warning("No probe images found in '%s'.", probe_dir)
        return 0

    logger.info("Identifying %d probe image(s) ...", len(probes))
    _identify_and_print(system, [p for p, _ in probes])

    logger.info("Evaluating ...")
    metrics = Evaluator(system).evaluate(probes, save_dir="evaluation")
    _report(metrics)
    logger.info("Plots and report written to 'evaluation/'.")
    return 0


# ----------------------------------------------------------------------
# Mode: LFW via scikit-learn
# ----------------------------------------------------------------------

def run_lfw(threshold: float, people: int, per_person: int) -> int:
    import tempfile

    import numpy as np
    from PIL import Image

    try:
        from sklearn.datasets import fetch_lfw_people
    except ImportError:
        logger.error("scikit-learn is not installed. Run: pip install scikit-learn")
        return 1

    logger.info("Fetching LFW (first run downloads ~200 MB) ...")
    try:
        lfw = fetch_lfw_people(
            min_faces_per_person=per_person * 2,
            color=True,
            resize=1.0,
            slice_=(slice(0, 250), slice(0, 250)),
        )
    except Exception as exc:  # noqa: BLE001 - network/cache failures are expected
        logger.error("Could not fetch LFW: %s", exc)
        logger.error("Run with --data-dir on your own photos, or --smoke for a plumbing check.")
        return 1

    images = (lfw.images * 255).astype(np.uint8)
    names = [lfw.target_names[t] for t in lfw.target]

    by_person: dict[str, list] = {}
    for img, name in zip(images, names):
        by_person.setdefault(name, []).append(img)
    chosen = sorted(by_person, key=lambda n: -len(by_person[n]))[:people]
    logger.info("Using %d identities: %s", len(chosen), ", ".join(chosen))

    workdir = tempfile.mkdtemp(prefix="lfw_demo_")
    os.makedirs(os.path.join(workdir, "probe"), exist_ok=True)
    impostor = sorted(by_person, key=lambda n: -len(by_person[n]))[people : people + 3]

    for name in chosen:
        person_dir = os.path.join(workdir, name.replace(" ", "_"))
        os.makedirs(person_dir, exist_ok=True)
        shots = by_person[name]
        for i, img in enumerate(shots[:per_person]):
            Image.fromarray(img).save(os.path.join(person_dir, f"{i}.jpg"), quality=95)
        for i, img in enumerate(shots[per_person : per_person + 2]):
            Image.fromarray(img).save(
                os.path.join(workdir, "probe", f"{name.replace(' ', '_')}_{i}.jpg"), quality=95
            )
    for j, name in enumerate(impostor):
        Image.fromarray(by_person[name][0]).save(
            os.path.join(workdir, "probe", f"unknown_{j}.jpg"), quality=95
        )

    logger.info("Prepared LFW working set in %s", workdir)
    return run_data_dir(workdir, threshold)


# ----------------------------------------------------------------------
# Mode: offline smoke test
# ----------------------------------------------------------------------

def run_smoke() -> int:
    """Verify the pipeline runs end to end. Deliberately reports no metrics."""
    import tempfile

    from generate_sample_data import generate
    from src.system import FaceRecognitionSystem

    workdir = tempfile.mkdtemp(prefix="facerec_smoke_")
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        logger.info("Generating synthetic images in %s ...", workdir)
        generate(out_dir="sample_data")

        system = FaceRecognitionSystem(threshold=0.60, db_path=os.path.join(workdir, "db.json"))

        detected = enrolled = 0
        for person in ("Alice", "Bob", "Charlie"):
            person_dir = os.path.join("sample_data", person)
            if not os.path.isdir(person_dir):
                continue
            for fname in sorted(os.listdir(person_dir)):
                if not fname.lower().endswith(IMAGE_EXTS):
                    continue
                outcome = system.enroll_from_image(person, os.path.join(person_dir, fname))
                detected += outcome.faces_found
                enrolled += outcome.enrolled

        probe_dir = os.path.join("sample_data", "probe")
        queries = 0
        if os.path.isdir(probe_dir):
            for fname in sorted(os.listdir(probe_dir)):
                if fname.lower().endswith(IMAGE_EXTS):
                    system.identify_from_image(os.path.join(probe_dir, fname))
                    queries += 1
    finally:
        os.chdir(cwd)

    print()
    print("=" * 62)
    print("  PIPELINE SMOKE TEST")
    print("=" * 62)
    print(f"  detection ran, faces found : {detected}")
    print(f"  embeddings stored          : {enrolled}")
    print(f"  identification queries ran : {queries}")
    print(f"  database round-trip        : {system.db_stats()}")
    print("=" * 62)
    print("  Pipeline is wired correctly: detect -> embed -> store -> match.")
    print()
    print("  No accuracy is reported. These are drawn shapes, not faces:")
    print("  dlib detects few of them and the ones it does are near-identical,")
    print("  so any score would measure the generator, not this system.")
    print("  For real numbers:  python demo.py --data-dir your_photos/")
    print("                or:  python demo.py --lfw")
    print("=" * 62)
    print()
    return 0


# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Face Recognition System demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--data-dir", help="Directory of real photos (see module docstring).")
    group.add_argument("--lfw", action="store_true", help="Evaluate on an LFW subset.")
    group.add_argument("--smoke", action="store_true", help="Offline pipeline check, no metrics.")
    parser.add_argument("--threshold", type=float, default=0.60, help="Rejection threshold.")
    parser.add_argument("--people", type=int, default=5, help="LFW identities to enroll.")
    parser.add_argument("--per-person", type=int, default=3, help="LFW photos per identity.")
    args = parser.parse_args()

    if args.data_dir:
        return run_data_dir(args.data_dir, args.threshold)
    if args.lfw:
        return run_lfw(args.threshold, args.people, args.per_person)
    if args.smoke:
        return run_smoke()

    parser.print_help()
    print()
    print("Pick a mode. --data-dir with your own photos gives the most meaningful result;")
    print("--smoke needs no network but reports no accuracy.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
