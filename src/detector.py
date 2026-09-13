"""
detector.py — Face detection module.

Wraps face_recognition's face detection (HOG model, CPU-friendly) and
optionally a CNN model when a CUDA-capable GPU is available.

Colour convention: every function here takes and returns **RGB** arrays,
matching what `face_recognition` expects. Callers holding OpenCV output
(BGR) must convert before calling in — see `bgr_to_rgb`.
"""

import face_recognition
import numpy as np
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
        Detect faces in an RGB image.

        Parameters
        ----------
        image : np.ndarray
            RGB array (H×W×3), or grayscale / RGBA which is converted for you.

        Returns
        -------
        list of (top, right, bottom, left) tuples — face bounding boxes.
        """
        rgb = self.to_rgb(image)
        return face_recognition.face_locations(
            rgb, number_of_times_to_upsample=self.upscale, model=self.model
        )

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
    def to_rgb(image: np.ndarray) -> np.ndarray:
        """
        Normalise an array to 3-channel RGB.

        Grayscale and RGBA are converted. A 3-channel array is assumed to be
        RGB already and returned untouched — channel order is not detectable
        from pixels, so it is the caller's contract to honour. (The previous
        version advertised BGR→RGB conversion in its docstring but its
        3-channel branch was a bare `pass`, so it silently did nothing.)
        """
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        raise ValueError(f"Unsupported image shape for detection: {image.shape}")

    @staticmethod
    def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        """Explicit BGR→RGB conversion for callers holding OpenCV output."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
