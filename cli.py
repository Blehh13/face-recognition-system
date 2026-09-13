#!/usr/bin/env python3
"""
cli.py - Command-line interface for the Face Recognition System.

Commands:
  enroll    Enroll a person from one or more images
  identify  Identify face(s) in an image
  list      List enrolled persons
  remove    Remove a person from the database
  stats     Show database statistics
  evaluate  Run evaluation on a labelled probe set (CSV)
"""

import os
import sys
import json
import csv
import logging

import click
import cv2

# --------------------------------------------------
# Logging setup
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cli")


def get_system(threshold, metric):
    """Lazy-import to avoid slow imports at startup."""
    from src.system import FaceRecognitionSystem
    return FaceRecognitionSystem(threshold=threshold, distance_metric=metric)


# --------------------------------------------------
# CLI group
# --------------------------------------------------
@click.group()
def cli():
    """Face Recognition Identification System - CLI"""


# --------------------------------------------------
# enroll
# --------------------------------------------------
@cli.command()
@click.argument("name")
@click.argument("images", nargs=-1, required=True)
@click.option("--allow-multi-face", is_flag=True, default=False,
              help="Enroll every face in the photo (unsafe: poisons the identity).")
def enroll(name, images, allow_multi_face):
    """
    Enroll NAME from one or more IMAGE paths.

    \b
    Example:
        python cli.py enroll "Alice" photos/alice1.jpg photos/alice2.jpg
    """
    sys_ = get_system(None, "euclidean")
    total = 0
    for img_path in images:
        if not os.path.exists(img_path):
            click.echo(f"  !  File not found: {img_path}", err=True)
            continue
        outcome = sys_.enroll_from_image(name, img_path, require_single_face=not allow_multi_face)
        if outcome.enrolled:
            click.echo(f"  OK    Enrolled {outcome.enrolled} face(s) from '{img_path}'")
        else:
            click.echo(f"  SKIP  {img_path}: {outcome.reason}")
        total += outcome.enrolled

    click.echo(f"\nTotal faces enrolled for '{name}': {total}")


# --------------------------------------------------
# identify
# --------------------------------------------------
@cli.command()
@click.argument("image")
@click.option("--threshold", "-t", default=0.60, type=float,
              help="Rejection threshold (default: 0.60).")
@click.option("--metric", "-m", default="euclidean",
              type=click.Choice(["euclidean", "cosine"]),
              help="Distance metric (default: euclidean).")
@click.option("--show", is_flag=True, default=False,
              help="Display annotated image with OpenCV.")
@click.option("--save", "-s", default=None,
              help="Save annotated image to this path.")
@click.option("--json-output", is_flag=True, default=False,
              help="Print results as JSON.")
def identify(image, threshold, metric, show, save, json_output):
    """
    Identify all faces in IMAGE.

    \b
    Example:
        python cli.py identify test.jpg --show
        python cli.py identify test.jpg --threshold 0.55 --json-output
    """
    if not os.path.exists(image):
        click.echo(f"Error: file not found: {image}", err=True)
        sys.exit(1)

    sys_ = get_system(threshold, metric)
    results = sys_.identify_from_image(image)

    if not results:
        click.echo("No faces detected in the image.")
        return

    output_records = []
    for (top, right, bottom, left), result in results:
        rec = result.to_dict()
        rec["bbox"] = {"top": top, "right": right, "bottom": bottom, "left": left}
        output_records.append(rec)

        if not json_output:
            status = "KNOWN  " if result.is_known else "UNKNOWN"
            click.echo(
                f"  {status}  ->  {result.name}"
                f"  | distance={result.distance:.4f}"
                f"  | confidence={result.confidence():.1%}"
                f"  | bbox=({left},{top},{right},{bottom})"
            )

    if json_output:
        click.echo(json.dumps(output_records, indent=2))

    # Visualisation
    if show or save:
        import face_recognition
        img_rgb = face_recognition.load_image_file(image)
        annotated = sys_.annotate_image(img_rgb, results)
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        if save:
            cv2.imwrite(save, annotated_bgr)
            click.echo(f"  -> Saved annotated image to '{save}'")
        if show:
            cv2.imshow("Face Recognition - Press any key to close", annotated_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# --------------------------------------------------
# list
# --------------------------------------------------
@cli.command("list")
def list_enrolled():
    """List all enrolled persons."""
    from src.database import FaceDatabase
    db = FaceDatabase()
    names = db.list_enrolled()
    if not names:
        click.echo("Database is empty. Enroll some faces first.")
        return
    click.echo(f"Enrolled persons ({len(names)}):")
    for name in names:
        info = db.get_info(name)
        click.echo(f"  - {name}  ({info['num_images']} image(s), enrolled {info['enrolled_at'][:10]})")


# --------------------------------------------------
# remove
# --------------------------------------------------
@cli.command()
@click.argument("name")
@click.confirmation_option(prompt="Are you sure you want to remove this person?")
def remove(name):
    """Remove NAME from the database."""
    from src.database import FaceDatabase
    db = FaceDatabase()
    if db.remove(name):
        click.echo(f"OK '{name}' removed from database.")
    else:
        click.echo(f"X '{name}' not found in database.")


# --------------------------------------------------
# stats
# --------------------------------------------------
@cli.command()
def stats():
    """Show database statistics."""
    from src.database import FaceDatabase
    db = FaceDatabase()
    s = db.stats()
    click.echo(f"Total persons    : {s['total_persons']}")
    click.echo(f"Total embeddings : {s['total_embeddings']}")
    if s["persons"]:
        click.echo("\nBreakdown:")
        for name, count in sorted(s["persons"].items()):
            click.echo(f"  {name:<30} {count} embedding(s)")


# --------------------------------------------------
# evaluate
# --------------------------------------------------
@cli.command()
@click.argument("probe_csv")
@click.option("--threshold", "-t", default=0.60, type=float)
@click.option("--metric", "-m", default="euclidean",
              type=click.Choice(["euclidean", "cosine"]))
@click.option("--save-dir", default="evaluation", help="Directory for reports/plots.")
@click.option("--sweep", is_flag=True, default=False,
              help="Run a threshold sweep (slower).")
def evaluate(probe_csv, threshold, metric, save_dir, sweep):
    """
    Evaluate on a CSV file with columns: image_path,label.

    \b
    Example:
        python cli.py evaluate probes.csv --sweep
    """
    if not os.path.exists(probe_csv):
        click.echo(f"Error: file not found: {probe_csv}", err=True)
        sys.exit(1)

    probe_set = []
    with open(probe_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            probe_set.append((row["image_path"].strip(), row["label"].strip()))

    click.echo(f"Loaded {len(probe_set)} probes from '{probe_csv}'.")
    sys_ = get_system(threshold, metric)

    from src.evaluator import Evaluator
    ev = Evaluator(sys_)

    if sweep:
        click.echo("Running threshold sweep ...")
        sweep_result = ev.threshold_sweep(probe_set, metric=metric, save_dir=save_dir)
        click.echo(f"Best threshold: {sweep_result['best_threshold']:.2f}")

    click.echo("Running full evaluation ...")
    metrics = ev.evaluate(probe_set, save_dir=save_dir)

    click.echo(f"\n{'='*40}")
    click.echo(f"  Accuracy  : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    click.echo(f"  Precision : {metrics['precision']:.4f}")
    click.echo(f"  Recall    : {metrics['recall']:.4f}")
    click.echo(f"  F1 Score  : {metrics['f1']:.4f}")
    click.echo(f"  FAR       : {metrics['far']:.4f}")
    click.echo(f"  FRR       : {metrics['frr']:.4f}")
    click.echo(f"{'='*40}")
    click.echo(f"  TP={metrics['tp']} | FP={metrics['fp']} | TN={metrics['tn']} | FN={metrics['fn']}")
    click.echo(f"\nPlots and report saved to '{save_dir}/'")


if __name__ == "__main__":
    cli()
