"""
app.py — Flask web interface for the Face Recognition System.

Routes:
  GET  /              → dashboard (enrolled persons + stats)
  POST /enroll        → enroll a person from an uploaded image
  POST /identify      → identify face(s) in an uploaded image
  GET  /api/persons   → JSON list of enrolled persons
  POST /api/remove    → remove a person (JSON body: {"name": "..."})
  GET  /health        → health check
"""

import os
import io
import base64
import logging

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = Flask(__name__)
app.secret_key = "face-recog-secret-key-change-in-prod"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# Lazy-init the system (heavy import: dlib)
_system = None


def get_system():
    global _system
    if _system is None:
        from src.system import FaceRecognitionSystem
        _system = FaceRecognitionSystem()
        logger.info("FaceRecognitionSystem initialised.")
    return _system


def pil_to_rgb_array(pil_image: Image.Image) -> np.ndarray:
    return np.array(pil_image.convert("RGB"))


def array_to_base64_jpeg(arr: np.ndarray) -> str:
    """Convert RGB numpy array → base64-encoded JPEG for HTML display."""
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    return base64.b64encode(buf).decode("ascii")


# ------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------

@app.route("/")
def index():
    sys_ = get_system()
    stats = sys_.db_stats()
    enrolled = sys_.list_enrolled()
    return render_template("index.html", stats=stats, enrolled=enrolled)


# ------------------------------------------------------------------
# Enroll
# ------------------------------------------------------------------

@app.route("/enroll", methods=["POST"])
def enroll():
    sys_ = get_system()
    name = request.form.get("name", "").strip()
    files = request.files.getlist("images")

    if not name:
        flash("Please enter a person's name.", "error")
        return redirect(url_for("index"))
    if not files or all(f.filename == "" for f in files):
        flash("Please select at least one image.", "error")
        return redirect(url_for("index"))

    total_enrolled = 0
    errors = []
    for f in files:
        if f.filename == "":
            continue
        try:
            img = pil_to_rgb_array(Image.open(f.stream))
            count = sys_.enroll_from_array(name, img)
            total_enrolled += count
            if count == 0:
                errors.append(f"No face detected in '{f.filename}'.")
        except Exception as exc:
            errors.append(f"Error processing '{f.filename}': {exc}")

    if total_enrolled > 0:
        flash(f"✓ Enrolled {total_enrolled} face(s) for '{name}'.", "success")
    else:
        flash("No faces were enrolled. Make sure images contain clear, frontal faces.", "warning")
    for e in errors:
        flash(e, "warning")

    return redirect(url_for("index"))


# ------------------------------------------------------------------
# Identify
# ------------------------------------------------------------------

@app.route("/identify", methods=["POST"])
def identify():
    sys_ = get_system()
    f = request.files.get("image")

    if f is None or f.filename == "":
        return jsonify({"error": "No image uploaded"}), 400

    threshold_str = request.form.get("threshold", "0.60")
    try:
        threshold = float(threshold_str)
    except ValueError:
        threshold = 0.60

    try:
        img_rgb = pil_to_rgb_array(Image.open(f.stream))
    except Exception as exc:
        return jsonify({"error": f"Cannot open image: {exc}"}), 400

    # Re-create system with requested threshold
    from src.system import FaceRecognitionSystem
    sys_custom = FaceRecognitionSystem(threshold=threshold)

    try:
        results = sys_custom.identify_from_array(img_rgb)
    except Exception as exc:
        logger.exception("Identification failed")
        return jsonify({"error": str(exc)}), 500

    # Annotate
    annotated = sys_custom.annotate_image(img_rgb, results)
    img_b64 = array_to_base64_jpeg(annotated)

    faces = []
    for (top, right, bottom, left), result in results:
        faces.append({
            **result.to_dict(),
            "bbox": {"top": top, "right": right, "bottom": bottom, "left": left},
        })

    return jsonify({
        "faces": faces,
        "num_faces": len(faces),
        "annotated_image": f"data:image/jpeg;base64,{img_b64}",
        "threshold_used": threshold,
    })


# ------------------------------------------------------------------
# API endpoints
# ------------------------------------------------------------------

@app.route("/api/persons", methods=["GET"])
def api_persons():
    sys_ = get_system()
    return jsonify({"persons": sys_.list_enrolled(), "stats": sys_.db_stats()})


@app.route("/api/remove", methods=["POST"])
def api_remove():
    sys_ = get_system()
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400
    removed = sys_.remove_person(name)
    return jsonify({"removed": removed, "name": name})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
