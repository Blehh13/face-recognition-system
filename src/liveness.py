"""
liveness.py — Presentation-attack detection (MiniFASNet, via OpenCV DNN).

The problem this addresses: without it, holding a printed photograph or a phone
screen in front of the camera authenticates as whoever is in the picture. Face
matching answers "is this the same face", never "is this a real person".

Model: MiniFASNetV2 from minivision-ai/Silent-Face-Anti-Spoofing (Apache 2.0,
commercial use permitted), exported to ONNX and run through `cv2.dnn` — so no
PyTorch is required. 1.7 MB, ~5 ms per face on CPU.

It outputs three classes; index 1 is "live", 0 and 2 are two families of
attack (print and replay).

--------------------------------------------------------------------------
IMPORTANT — this is ADVISORY, and deliberately does not block
--------------------------------------------------------------------------
The single-model score is not strong enough to gate access on:

  * genuine photographs in our own testing score around 0.62-0.64 live, not
    the 0.95+ a confident detector would give;
  * simply brightening a genuine photograph flipped the prediction to spoof,
    so the model is sensitive to capture conditions rather than purely to
    liveness;
  * the upstream project ensembles two models and we run one;
  * and crucially, **the spoof side is unvalidated here**. Verifying it needs
    real attack samples — an actual printed photo and a screen replay,
    captured on the camera being deployed. We had none, so the false-reject
    behaviour is measured and the false-accept behaviour is not.

Anti-spoofing generalises notoriously badly across cameras and lighting. Used
as a hard gate on these numbers it would reject real people. It therefore
returns a score and a flag that the UI surfaces as a caution, and matching
proceeds regardless. Turning it into a gate is a one-line change
(`require_live=True`) once it has been measured on the target hardware.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_FILE = "minifasnet_v2.onnx"
MODEL_URL = (
    "https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/"
    "resolve/main/minifasnet_v2.onnx"
)

INPUT_SIZE = 80
# The upstream pipeline crops a box 2.7x the detected face, so the model sees
# the border between a face and whatever surrounds it — the edge of a phone or
# a sheet of paper is much of the signal.
CROP_SCALE = 2.7

# Below this live-probability the result is flagged for a human to look at.
# Chosen to sit under the ~0.62 genuine photographs score, so real captures are
# not routinely flagged; it is a caution line, not an access decision.
SUSPICION_THRESHOLD = 0.50


class LivenessUnavailable(RuntimeError):
    """The ONNX weights are neither present locally nor downloadable."""


@dataclass
class LivenessResult:
    """One liveness assessment. Advisory — see the module docstring."""
    live_score: float          # P(live), 0-1
    suspicious: bool           # live_score < SUSPICION_THRESHOLD
    label: str                 # "live" | "possible spoof"

    def to_dict(self) -> dict:
        return {
            "live_score": round(float(self.live_score), 4),
            "suspicious": bool(self.suspicious),
            "label": self.label,
            "advisory": True,   # never let a client mistake this for a verdict
        }


def model_path(download: bool = True) -> str:
    path = os.path.join(MODEL_DIR, MODEL_FILE)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    if not download:
        raise LivenessUnavailable(f"{MODEL_FILE} not found in {MODEL_DIR}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp = path + ".part"
    logger.info("Downloading %s …", MODEL_FILE)
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 - offline is normal
        if os.path.exists(tmp):
            os.remove(tmp)
        raise LivenessUnavailable(
            f"Could not download {MODEL_FILE}: {exc}. Fetch it manually from "
            f"{MODEL_URL} into {MODEL_DIR}/."
        ) from exc
    return path


class LivenessDetector:
    """Scores how likely a detected face is a live person rather than a replay."""

    def __init__(self):
        self._net = cv2.dnn.readNetFromONNX(model_path())

    def score(self, image: np.ndarray, box: tuple[int, int, int, int]) -> LivenessResult:
        """
        Assess one face.

        Parameters
        ----------
        image : RGB numpy array (the convention everywhere else in src/).
        box   : (top, right, bottom, left), matching the detector's output.
        """
        bgr = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
        patch = self._crop(bgr, box)
        blob = cv2.dnn.blobFromImage(patch, size=(INPUT_SIZE, INPUT_SIZE), swapRB=False)
        self._net.setInput(blob)
        probabilities = _softmax(self._net.forward().flatten())

        live = float(probabilities[1])
        suspicious = live < SUSPICION_THRESHOLD
        return LivenessResult(
            live_score=live,
            suspicious=suspicious,
            label="possible spoof" if suspicious else "live",
        )

    @staticmethod
    def _crop(bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        top, right, bottom, left = box
        cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
        half = max(right - left, bottom - top) * CROP_SCALE / 2.0
        height, width = bgr.shape[:2]
        x1, y1 = int(max(cx - half, 0)), int(max(cy - half, 0))
        x2, y2 = int(min(cx + half, width)), int(min(cy + half, height))
        patch = bgr[y1:y2, x1:x2]
        if patch.size == 0:
            patch = bgr
        return cv2.resize(patch, (INPUT_SIZE, INPUT_SIZE))


def available() -> bool:
    """True when the weights are already on disk (no download needed)."""
    try:
        model_path(download=False)
        return True
    except LivenessUnavailable:
        return False


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()
