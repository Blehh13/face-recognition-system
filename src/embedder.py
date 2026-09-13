"""
embedder.py — Face embedding (feature extraction) module.

Uses dlib's ResNet-based face recognition model (via the `face_recognition`
library) to produce 128-dimensional L2-normalised embeddings. The model was
pre-trained on ~3 million faces and achieves 99.38% accuracy on LFW.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 128  # dlib ResNet output size


class FaceEmbedder:
    """
    Generates 128-d face embeddings.

    Parameters
    ----------
    model : str
        'small' — 5-point landmark model (faster, slightly less accurate)
        'large' — 68-point landmark model (default, more accurate)
    """

    def __init__(self, model: str = "large"):
        if model not in ("small", "large"):
            raise ValueError("model must be 'small' or 'large'")
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(
        self,
        image: np.ndarray,
        face_locations: list[tuple] | None = None,
    ) -> list[np.ndarray]:
        """
        Compute 128-d embeddings for all detected faces in *image*.

        Parameters
        ----------
        image : np.ndarray  (RGB, H×W×3)
        face_locations : optional pre-computed bounding boxes.
            If None, face detection is run internally.

        Returns
        -------
        list of np.ndarray, each of shape (128,)
        """
        from src.detector import FaceDetector
        encodings = FaceDetector._dlib().face_encodings(
            image,
            known_face_locations=face_locations,
            num_jitters=1,          # 1 = fast; increase for better accuracy
            model=self.model,
        )
        return [np.array(e) for e in encodings]

    def embed_from_path(
        self,
        image_path: str,
        face_locations: list[tuple] | None = None,
    ) -> list[np.ndarray]:
        """Load image from disk and compute embeddings."""
        from src.detector import FaceDetector
        image = FaceDetector._dlib().load_image_file(image_path)  # RGB
        return self.embed(image, face_locations)

    def embed_single(
        self,
        image: np.ndarray,
        face_locations: list[tuple] | None = None,
    ) -> np.ndarray | None:
        """
        Convenience: return the first embedding or None if no face found.
        Logs a warning if multiple faces detected.
        """
        embeddings = self.embed(image, face_locations)
        if not embeddings:
            logger.warning("No face detected in image.")
            return None
        if len(embeddings) > 1:
            logger.warning(
                "Multiple faces detected; using the first one. "
                "For enrollment, use a single-face image."
            )
        return embeddings[0]

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def l2_normalize(embedding: np.ndarray) -> np.ndarray:
        """L2-normalise an embedding vector."""
        norm = np.linalg.norm(embedding)
        return embedding / (norm + 1e-10)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity ∈ [-1, 1]."""
        a = FaceEmbedder.l2_normalize(a)
        b = FaceEmbedder.l2_normalize(b)
        return float(np.dot(a, b))

    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance between two embeddings."""
        return float(np.linalg.norm(a - b))
