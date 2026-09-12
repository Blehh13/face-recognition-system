"""
system.py — High-level FaceRecognitionSystem facade.

Combines Detector + Embedder + Database + Matcher into a single easy-to-use
object for both enrollment and identification workflows.
"""

import os
import numpy as np
import cv2
import face_recognition
import logging
from typing import Optional

from src.detector import FaceDetector
from src.embedder import FaceEmbedder
from src.database import FaceDatabase
from src.matcher import FaceMatcher, MatchResult

logger = logging.getLogger(__name__)


class FaceRecognitionSystem:
    """
    End-to-end face recognition pipeline.

    Usage:
        sys = FaceRecognitionSystem()
        sys.enroll_from_image("Alice", "alice_photo.jpg")
        result = sys.identify_from_image("test_photo.jpg")
        print(result.name)  # "Alice" or "Unknown"
    """

    def __init__(
        self,
        detection_model: str = "hog",
        embedding_model: str = "large",
        distance_metric: str = "euclidean",
        threshold: float | None = None,
        db_path: str | None = None,
    ):
        self.detector = FaceDetector(model=detection_model)
        self.embedder = FaceEmbedder(model=embedding_model)
        self.database = FaceDatabase(db_path) if db_path else FaceDatabase()
        self.matcher  = FaceMatcher(threshold=threshold, metric=distance_metric)

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def enroll_from_image(
        self,
        name: str,
        image_path: str,
        require_single_face: bool = False,
    ) -> int:
        """
        Enroll a person from an image file.

        Returns
        -------
        int — number of faces enrolled from this image (0 if detection failed).
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = face_recognition.load_image_file(image_path)
        locations = face_recognition.face_locations(image, model=self.detector.model)

        if not locations:
            logger.warning("No face detected in '%s'. Skipping.", image_path)
            return 0

        if require_single_face and len(locations) > 1:
            logger.warning(
                "Multiple faces (%d) in '%s'; require_single_face=True. Skipping.",
                len(locations), image_path
            )
            return 0

        embeddings = self.embedder.embed(image, face_locations=locations)
        for emb in embeddings:
            self.database.enroll(name, emb)

        logger.info(
            "Enrolled %d face(s) for '%s' from '%s'",
            len(embeddings), name, image_path
        )
        return len(embeddings)

    def enroll_from_directory(
        self,
        name: str,
        directory: str,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    ) -> int:
        """Enroll all images in *directory* as *name*."""
        total = 0
        for fname in sorted(os.listdir(directory)):
            if fname.lower().endswith(extensions):
                path = os.path.join(directory, fname)
                total += self.enroll_from_image(name, path)
        return total

    def enroll_from_array(self, name: str, image: np.ndarray) -> int:
        """Enroll from an in-memory RGB numpy array."""
        locations = self.detector.detect(image)
        if not locations:
            return 0
        embeddings = self.embedder.embed(image, face_locations=locations)
        for emb in embeddings:
            self.database.enroll(name, emb)
        return len(embeddings)

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------

    def identify_from_image(
        self, image_path: str
    ) -> list[tuple[tuple, MatchResult]]:
        """
        Identify all faces in *image_path*.

        Returns
        -------
        list of (face_location, MatchResult) tuples.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = face_recognition.load_image_file(image_path)
        return self._identify(image)

    def identify_from_array(
        self, image: np.ndarray
    ) -> list[tuple[tuple, MatchResult]]:
        """Identify all faces in a numpy RGB image array."""
        return self._identify(image)

    def _identify(
        self, image: np.ndarray
    ) -> list[tuple[tuple, MatchResult]]:
        locations = self.detector.detect(image)
        if not locations:
            logger.info("No faces detected in query image.")
            return []

        embeddings = self.embedder.embed(image, face_locations=locations)
        db = self.database.get_all_embeddings()
        results = self.matcher.match_many(embeddings, db)
        return list(zip(locations, results))

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def annotate_image(
        self,
        image: np.ndarray,
        face_results: list[tuple[tuple, MatchResult]],
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on *image* (BGR expected for display).
        """
        out = image.copy()
        for (top, right, bottom, left), result in face_results:
            color = (0, 200, 0) if result.is_known else (0, 0, 220)
            label = (
                f"{result.name} ({result._confidence():.0%})"
                if result.is_known
                else "Unknown"
            )
            cv2.rectangle(out, (left, top), (right, bottom), color, 2)
            # Background for text
            cv2.rectangle(out, (left, top - 28), (right, top), color, -1)
            cv2.putText(
                out, label,
                (left + 4, top - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            )
        return out

    # ------------------------------------------------------------------
    # Database shortcuts
    # ------------------------------------------------------------------

    def list_enrolled(self) -> list[str]:
        return self.database.list_enrolled()

    def remove_person(self, name: str) -> bool:
        return self.database.remove(name)

    def db_stats(self) -> dict:
        return self.database.stats()
