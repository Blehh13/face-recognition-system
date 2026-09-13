"""
app.py — Flask backend for the Face Recognition System.

API Routes (consumed by the React frontend):
  POST /enroll        → enroll a person from an uploaded image
  POST /identify      → identify face(s) in an uploaded image
  GET  /api/persons   → JSON list of enrolled persons
  POST /api/remove    → remove a person (JSON body: {"name": "..."})
  GET  /health        → health check

Production:
  The React app is built to static/dist/ via `cd frontend && npm run build`.
  Flask serves those files as a SPA fallback.

Development:
  Run Flask (port 5000) + `cd frontend && npm run dev` (port 5173) separately.
  Vite proxies /enroll, /identify, /api/* to Flask automatically.

Configuration (environment variables):
  FACEREC_HOST    interface to bind   (default 127.0.0.1 — loopback only)
  FACEREC_PORT    port                (default 5000)
  FACEREC_SECRET  Flask secret key    (default: a dev-only value)
  FACEREC_DB      database path       (default database/enrolled_faces.json).
                  A .sqlite/.db extension selects the multi-process-safe
                  backend, which a deployment behind gunicorn requires.
  FACEREC_DEMO    "1" advertises this as a throwaway public demo via
                  /api/config. The database is emptied by the container
                  entrypoint, not here: workers initialise lazily, so clearing
                  on first use meant the second worker wiped the database
                  mid-session when it handled its first request.
"""

import base64
import json
import logging
import os
import pathlib

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, flash

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# React production build lives in static/dist/
REACT_DIST = pathlib.Path(__file__).parent / "static" / "dist"

app = Flask(__name__, static_folder=str(REACT_DIST), static_url_path="/")
app.secret_key = os.environ.get("FACEREC_SECRET", "dev-only-insecure-key")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# Emit an error rather than the bare tokens `NaN`/`Infinity`, which are not
# valid JSON and make browsers throw on parse.
app.json.allow_nan = False

# Decompression-bomb guard: a small file can decode to gigapixels and exhaust
# memory. 50 MP is far above any real photograph.
Image.MAX_IMAGE_PIXELS = 50_000_000

ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "WEBP", "GIF", "TIFF"}

# Lazy-init the system (heavy import: dlib)
_system = None
_liveness = None
_liveness_failed = False


def get_liveness():
    """
    Return the liveness detector, or None when its weights are unavailable.

    Advisory only: a low score annotates the result, it never blocks a match.
    See src/liveness.py for why that is the right call on the current numbers.
    """
    global _liveness, _liveness_failed
    if _liveness is None and not _liveness_failed:
        try:
            from src.liveness import LivenessDetector
            _liveness = LivenessDetector()
            logger.info("Liveness detector loaded (advisory).")
        except Exception as exc:  # noqa: BLE001 - absence is a normal state
            _liveness_failed = True
            logger.info("Liveness detector unavailable, continuing without it: %s", exc)
    return _liveness


DEMO_MODE = os.environ.get("FACEREC_DEMO", "").strip() == "1"


def get_system():
    global _system
    if _system is None:
        from src.system import FaceRecognitionSystem
        _system = FaceRecognitionSystem(db_path=os.environ.get("FACEREC_DB") or None)
        logger.info("FaceRecognitionSystem initialised (demo=%s).", DEMO_MODE)
    return _system


def error_curve(engine: str) -> list[list[float]]:
    """
    Measured [threshold, false-accept rate, false-reject rate] triples.

    Produced by `python -m ml.aggregation --fit` on LFW validation identities
    with centroid aggregation, and shipped so the UI can state the consequence
    of a threshold rather than presenting a bare number.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "ml", "results", "error_curves.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle).get(engine, [])
    except (OSError, json.JSONDecodeError):
        return []


class ImageError(ValueError):
    """Raised when an upload is missing, unreadable, or not an image."""


def read_upload(file_storage) -> np.ndarray:
    """
    Decode an uploaded file into an RGB numpy array, or raise ImageError.

    Pillow is asked to verify the format before decoding so that arbitrary
    bytes cannot be pushed through the decoder.
    """
    if file_storage is None or not file_storage.filename:
        raise ImageError("No image uploaded.")

    try:
        image = Image.open(file_storage.stream)
        fmt = (image.format or "").upper()
        if fmt not in ALLOWED_FORMATS:
            raise ImageError(f"Unsupported image format: {fmt or 'unknown'}.")
        return np.array(image.convert("RGB"))
    except ImageError:
        raise
    except UnidentifiedImageError:
        raise ImageError("That file is not a readable image.")
    except Image.DecompressionBombError:
        raise ImageError("That image is too large to process.")
    except Exception as exc:  # noqa: BLE001 - surface decode failures to the client
        raise ImageError(f"Could not read the image: {exc}")


def array_to_base64_jpeg(arr: np.ndarray) -> str:
    """Convert RGB numpy array → base64-encoded JPEG for HTML display."""
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    return base64.b64encode(buf).decode("ascii")


def wants_json() -> bool:
    """
    True when the caller wants a JSON reply rather than a redirect.

    The React client sends `Accept: application/json` and needs to know what
    actually happened; a plain HTML form post still gets flash-and-redirect.
    """
    return "application/json" in request.headers.get("Accept", "")


# ------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------

@app.route("/")
def index():
    """
    Serve the built React SPA.

    There used to be a second, separately-styled Jinja UI behind this route as
    a fallback. Two hand-maintained frontends for three screens is a liability,
    and the Jinja one was unreachable whenever the build existed, so it is gone
    — an unbuilt frontend now says so instead of silently rendering a different
    interface.
    """
    if REACT_DIST.exists():
        return send_from_directory(str(REACT_DIST), "index.html")
    return (
        "<h1>Frontend not built</h1>"
        "<p>Run <code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code>, "
        "then reload. For development, run <code>npm run dev</code> and use port 5173.</p>",
        503,
    )


@app.route("/<path:path>")
def static_proxy(path):
    """Serve React static assets (JS, CSS, etc.) or fall back to index.html for SPA routing."""
    if REACT_DIST.exists():
        target = REACT_DIST / path
        if target.exists():
            return send_from_directory(str(REACT_DIST), path)
        return send_from_directory(str(REACT_DIST), "index.html")
    return jsonify({"error": "Not found"}), 404


# ------------------------------------------------------------------
# Enroll
# ------------------------------------------------------------------

@app.route("/enroll", methods=["POST"])
def enroll():
    sys_ = get_system()
    name = request.form.get("name", "").strip()
    files = [f for f in request.files.getlist("images") if f and f.filename]
    as_json = wants_json()

    if not name:
        if as_json:
            return jsonify({"error": "Please enter a person's name."}), 400
        flash("Please enter a person's name.", "error")
        return redirect(url_for("index"))

    if not files:
        if as_json:
            return jsonify({"error": "Please select at least one image."}), 400
        flash("Please select at least one image.", "error")
        return redirect(url_for("index"))

    # Fold "alice" into an existing "Alice" instead of creating a second person.
    existing = sys_.database.find_name(name)
    if existing and existing != name:
        logger.info("Name '%s' matched existing entry '%s'.", name, existing)
        name = existing

    total_enrolled = 0
    warnings = []
    for f in files:
        try:
            image = read_upload(f)
        except ImageError as exc:
            warnings.append(f"{f.filename}: {exc}")
            continue

        try:
            outcome = sys_.enroll_from_array(name, image)
        except ValueError as exc:
            warnings.append(f"{f.filename}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            logger.exception("Enrolment failed for '%s'", f.filename)
            warnings.append(f"{f.filename}: {exc}")
            continue

        total_enrolled += outcome.enrolled
        if outcome.reason:
            warnings.append(f"{f.filename}: {outcome.reason}")

    if as_json:
        return jsonify({
            "name": name,
            "faces_enrolled": total_enrolled,
            "warnings": warnings,
        })

    if total_enrolled > 0:
        flash(f"Enrolled {total_enrolled} face(s) for '{name}'.", "success")
    else:
        flash("No faces were enrolled. Use a clear, single-face photo.", "warning")
    for w in warnings:
        flash(w, "warning")

    return redirect(url_for("index"))


# ------------------------------------------------------------------
# Identify
# ------------------------------------------------------------------

@app.route("/identify", methods=["POST"])
def identify():
    sys_ = get_system()

    try:
        img_rgb = read_upload(request.files.get("image"))
    except ImageError as exc:
        return jsonify({"error": str(exc)}), 400

    # No hardcoded fallback: the right threshold depends on the engine and on
    # how enrolled photographs are aggregated. A literal here was still 0.60
    # from the dlib era and silently rejected valid SFace matches at 0.70.
    raw = request.form.get("threshold")
    threshold = None
    if raw not in (None, ""):
        try:
            threshold = min(max(float(raw), 0.05), 4.0)
        except ValueError:
            threshold = None

    try:
        # Reuse the loaded system and vary only the threshold. Rebuilding it
        # per request re-created the dlib wrappers and re-read the whole
        # database from disk on every identification.
        results = sys_.identify_from_array(img_rgb, threshold=threshold)
        effective_threshold = threshold if threshold is not None else sys_.matcher.threshold
    except Exception as exc:  # noqa: BLE001
        logger.exception("Identification failed")
        return jsonify({"error": str(exc)}), 500

    annotated = sys_.annotate_image(img_rgb, results)
    img_b64 = array_to_base64_jpeg(annotated)

    detector = get_liveness()
    faces = []
    for (top, right, bottom, left), result in results:
        entry = {
            **result.to_dict(),
            "bbox": {"top": top, "right": right, "bottom": bottom, "left": left},
        }
        if detector is not None:
            try:
                entry["liveness"] = detector.score(img_rgb, (top, right, bottom, left)).to_dict()
            except Exception:  # noqa: BLE001 - never fail a match over an advisory signal
                logger.exception("Liveness scoring failed")
        faces.append(entry)

    return jsonify({
        "faces": faces,
        "num_faces": len(faces),
        "annotated_image": f"data:image/jpeg;base64,{img_b64}",
        "threshold_used": effective_threshold,
        "enrolled_count": len(sys_.list_enrolled()),
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
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400
    removed = sys_.remove_person(name)
    return jsonify({"removed": removed, "name": name})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/config")
def api_config():
    """
    Runtime facts the frontend needs.

    The threshold and its error curve are engine-specific, so the UI reads them
    from here instead of carrying a copy. A hardcoded 0.60 in the frontend was
    correct only for the dlib engine with nearest-neighbour matching.
    """
    sys_ = get_system()
    return jsonify({
        "demo": DEMO_MODE,
        "engine": sys_.engine_name,
        "metric": sys_.matcher.metric,
        "aggregation": sys_.matcher.aggregation,
        "threshold": sys_.matcher.threshold,
        "error_curve": error_curve(sys_.engine_name),
    })


@app.errorhandler(413)
def too_large(_exc):
    return jsonify({"error": "That image is larger than the 16 MB limit."}), 413


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Loopback by default. These endpoints have no authentication, so binding
    # 0.0.0.0 would let anyone on the network enroll, identify, and delete
    # people. Opt in explicitly with FACEREC_HOST=0.0.0.0 on a trusted network.
    host = os.environ.get("FACEREC_HOST", "127.0.0.1")
    port = int(os.environ.get("FACEREC_PORT", "5000"))
    if host != "127.0.0.1":
        logger.warning("Binding %s — the API is unauthenticated. Use a trusted network.", host)
    app.run(host=host, port=port, debug=False)
