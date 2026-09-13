"""
opencv_engine.py — YuNet detection + SFace recognition, through OpenCV's own API.

An alternative to the dlib pipeline that needs **no new dependencies**: both
models ship as ONNX files consumed by `cv2.FaceDetectorYN` and
`cv2.FaceRecognizerSF`, which are already part of opencv-python.

Why it exists, measured on this repository's own LFW protocol
(462 held-out identities, 6,000 pairs — see ml/README.md):

    metric          dlib      YuNet+SFace
    EER             2.67%     1.83%
    accuracy        97.50%    98.50%
    TAR @ FAR=1%    95.6%     97.6%
    TAR @ FAR=0.1%  90.3%     96.8%
    d-prime         4.08      4.43
    throughput      ~14 img/s ~90 img/s   (dlib figure uses 12 processes;
                                           SFace is single-threaded)

dlib retains a marginally higher AUC (0.9954 vs 0.9902), which is a
ranking measure dominated by the extreme tail. At every operating point a
deployment would actually choose, SFace is ahead — most sharply at
FAR = 0.1%, the strict setting that matters when a false accept is costly.

Licences: YuNet is MIT, SFace is Apache 2.0. Both permit commercial use,
unlike InsightFace's pretrained packs.

Two things to know before switching:

1. **Embeddings are not interchangeable.** SFace produces 128-d vectors in a
   different space from dlib's. A database enrolled with one engine is
   meaningless to the other, so `FaceDatabase` records which engine wrote each
   person and `FaceRecognitionSystem` refuses to mix them.
2. **The threshold changes.** SFace is compared with cosine distance on
   L2-normalised vectors, where the fitted operating point is ~1.17 — not
   dlib's 0.60.
"""

from __future__ import annotations

import logging
import os
import urllib.request

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ENGINE_NAME = "opencv-sface"

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

_MODELS = {
    "detector": (
        "face_detection_yunet_2023mar.onnx",
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ),
    "recognizer": (
        "face_recognition_sface_2021dec.onnx",
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ),
}

# Fitted on the LFW validation identities, Euclidean distance between
# L2-normalised SFace features. See ml/README.md.
DEFAULT_SFACE_THRESHOLD = 1.17


class ModelsUnavailable(RuntimeError):
    """The ONNX weights are neither present locally nor downloadable."""


def model_path(kind: str, download: bool = True) -> str:
    """Return the local path to a model file, fetching it on first use."""
    filename, url = _MODELS[kind]
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    if not download:
        raise ModelsUnavailable(f"{filename} not found in {MODEL_DIR}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info("Downloading %s …", filename)
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 - offline is a normal condition
        if os.path.exists(tmp):
            os.remove(tmp)
        raise ModelsUnavailable(
            f"Could not download {filename}: {exc}. "
            f"Download it manually from {url} into {MODEL_DIR}/."
        ) from exc
    return path


class YuNetDetector:
    """
    Face detector matching the interface of `src.detector.FaceDetector`.

    Returns (top, right, bottom, left) boxes so it is a drop-in replacement,
    but also exposes `detect_faces` which keeps YuNet's five landmarks — those
    are what SFace's alignment needs, and throwing them away is what made a
    naive SFace port score worse than dlib in our first attempt.
    """

    def __init__(self, score_threshold: float = 0.6, nms_threshold: float = 0.3, top_k: int = 500):
        self._detector = cv2.FaceDetectorYN.create(
            model_path("detector"), "", (320, 320),
            score_threshold=score_threshold, nms_threshold=nms_threshold, top_k=top_k,
        )
        self.model = "yunet"

    def detect_faces(self, image: np.ndarray) -> np.ndarray:
        """Raw YuNet rows: [x, y, w, h, 5 landmark xy pairs, score]."""
        bgr = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(bgr)
        return np.empty((0, 15), dtype=np.float32) if faces is None else faces

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Bounding boxes as (top, right, bottom, left), clipped to the frame."""
        height, width = image.shape[:2]
        boxes = []
        for row in self.detect_faces(image):
            x, y, w, h = row[:4]
            left, top = max(int(round(x)), 0), max(int(round(y)), 0)
            right, bottom = min(int(round(x + w)), width), min(int(round(y + h)), height)
            if right > left and bottom > top:
                boxes.append((top, right, bottom, left))
        return boxes

    def count_faces(self, image: np.ndarray) -> int:
        return len(self.detect(image))


class SFaceEmbedder:
    """
    128-d SFace embeddings, matching `src.embedder.FaceEmbedder`'s interface.

    Alignment is not optional. `alignCrop` applies a similarity transform from
    YuNet's five landmarks to SFace's canonical 112x112 layout; feeding it an
    unaligned crop instead costs roughly four points of accuracy (98.5% -> 94.6%
    measured on our LFW split).
    """

    def __init__(self):
        self._recognizer = cv2.FaceRecognizerSF.create(model_path("recognizer"), "")
        self._detector = YuNetDetector()
        self.model = "sface"

    def embed(self, image: np.ndarray, face_locations: list[tuple] | None = None) -> list[np.ndarray]:
        """
        Embed every face in an RGB image.

        `face_locations` is accepted for interface compatibility and used only
        to pick which detections to keep — SFace still needs YuNet's landmarks,
        which plain boxes do not carry.
        """
        bgr = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
        faces = self._detector.detect_faces(image)
        if len(faces) == 0:
            return []

        if face_locations:
            faces = _select_matching(faces, face_locations)

        out = []
        for row in faces:
            aligned = self._recognizer.alignCrop(bgr, row)
            feature = self._recognizer.feature(aligned).flatten().astype(np.float64)
            norm = np.linalg.norm(feature)
            # Normalise here so the stored vectors live in the space the
            # threshold was fitted in.
            out.append(feature / norm if norm > 0 else feature)
        return out

    def embed_single(self, image: np.ndarray, face_locations: list[tuple] | None = None):
        embeddings = self.embed(image, face_locations)
        if not embeddings:
            logger.warning("No face detected in image.")
            return None
        if len(embeddings) > 1:
            logger.warning("Multiple faces detected; using the first one.")
        return embeddings[0]


def _select_matching(faces: np.ndarray, wanted: list[tuple]) -> np.ndarray:
    """Keep the detection whose box best overlaps each requested location."""
    keep = []
    for top, right, bottom, left in wanted:
        best, best_iou = None, 0.0
        for row in faces:
            x, y, w, h = row[:4]
            iou = _iou((left, top, right, bottom), (x, y, x + w, y + h))
            if iou > best_iou:
                best, best_iou = row, iou
        if best is not None and best_iou > 0.3:
            keep.append(best)
    return np.array(keep) if keep else faces


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def available() -> bool:
    """True when both ONNX files are already on disk (no download needed)."""
    try:
        model_path("detector", download=False)
        model_path("recognizer", download=False)
        return True
    except ModelsUnavailable:
        return False
