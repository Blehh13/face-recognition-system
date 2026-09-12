"""
detector.py — Face detection module.

Wraps face_recognition's face detection (HOG model, CPU-friendly) and
optionally a CNN model when a CUDA-capable GPU is available.
"""

import face_recognition
import numpy as np
from PIL import Image
import cv2
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Detects faces in images and returns bounding boxes.

    Parameters
    ----------
    model : str
        'hog'  — fast, CPU-only (default)
        'cnn'  — more accurate, GPU optional
    upscale : int
        Number of times to upscale image before detection.
        Higher values find smaller faces but are slower.
    """

    def __init__(self, model: str = "hog", upscale: int = 1):
        if model not in ("hog", "cnn"):
            raise ValueError("model must be 'hog' or 'cnn'")
        self.model = model
        self.upscale = upscale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detect faces in an image.

        Parameters
        ----------
        image : np.ndarray
            BGR (OpenCV) or RGB numpy array (H×W×3).

        Returns
        -------
        list of (top, right, bottom, left) tuples — face bounding boxes.
        """
        rgb = self._ensure_rgb(image)
        locations = face_recognition.face_locations(
            rgb, number_of_times_to_upsample=self.upscale, model=self.model
        )
        return locations  # list of (top, right, bottom, left)

    def detect_from_path(self, image_path: str) -> tuple[np.ndarray, list]:
        """Load an image from disk and detect faces."""
        image = face_recognition.load_image_file(image_path)  # RGB array
        locations = face_recognition.face_locations(
            image, number_of_times_to_upsample=self.upscale, model=self.model
        )
        return image, locations

    def count_faces(self, image: np.ndarray) -> int:
        return len(self.detect(image))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_rgb(image: np.ndarray) -> np.ndarray:
        """Convert BGR (OpenCV) to RGB if needed."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            # Heuristic: assume BGR if loaded by OpenCV
            # face_recognition.load_image_file gives RGB — caller should pass
            # the original when possible.
            pass
        return image

    @staticmethod
    def draw_boxes(
        image: np.ndarray,
        locations: list,
        labels: list[str] | None = None,
        color: tuple = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw bounding boxes on an image (BGR)."""
        out = image.copy()
        for idx, (top, right, bottom, left) in enumerate(locations):
            cv2.rectangle(out, (left, top), (right, bottom), color, thickness)
            if labels and idx < len(labels):
                label = labels[idx]
                cv2.putText(
                    out,
                    label,
                    (left, top - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
        return out
