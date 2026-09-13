"""
system.py — High-level FaceRecognitionSystem facade.

Combines Detector + Embedder + Database + Matcher into a single easy-to-use
object for both enrollment and identification workflows.

Colour convention: images are **RGB** numpy arrays throughout, including the
array returned by `annotate_image`. Convert once at the display boundary if
you need BGR for `cv2.imwrite`/`cv2.imshow`.
"""

import os
import numpy as np
import cv2
import face_recognition
import logging
from dataclasses import dataclass

from src.detector import FaceDetector
from src.embedder import FaceEmbedder
from src.database import FaceDatabase
from src.matcher import FaceMatcher, MatchResult

logger = logging.getLogger(__name__)

# Annotation colours, RGB, matching the web UI's result cards.
COLOR_KNOWN   = (22, 163, 74)    # green  — accepted
COLOR_UNKNOWN = (217, 119, 6)    # amber  — rejected
COLOR_LABEL   = (255, 255, 255)


@dataclass
class EnrollOutcome:
    """
    What actually happened during one enrolment attempt.

    The old API returned a bare int, which could not distinguish "no face
    found" from "refused a group photo" — so the caller had to guess.
    """
    enrolled: int              # embeddings actually stored
    faces_found: int           # faces detected in the image
    reason: str | None = None  # why nothing was stored, if nothing was

    def __bool__(self) -> bool:
        return self.enrolled > 0

    def __int__(self) -> int:
        return self.enrolled


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
        engine: str = "dlib",
    ):
        """
        `engine` selects the detection + embedding backend:

          "dlib"    HOG detector + dlib ResNet-128 (default, no downloads)
          "opencv"  YuNet + SFace through OpenCV's own API — more accurate at
                    every operating point and ~6x faster, at the cost of a
                    one-off 37 MB model download. See src/opencv_engine.py.

        Each engine carries its own default threshold: the two produce
        embeddings in different spaces, so dlib's 0.60 is meaningless to SFace.
        """
        if engine not in ("dlib", "opencv"):
            raise ValueError("engine must be 'dlib' or 'opencv'")
        self.engine = engine

        if engine == "opencv":
            from src.opencv_engine import (
                DEFAULT_SFACE_THRESHOLD, ENGINE_NAME, SFaceEmbedder, YuNetDetector,
            )
            self.detector = YuNetDetector()
            self.embedder = SFaceEmbedder()
            self.engine_name = ENGINE_NAME
            if threshold is None:
                threshold = DEFAULT_SFACE_THRESHOLD
        else:
            self.detector = FaceDetector(model=detection_model)
            self.embedder = FaceEmbedder(model=embedding_model)
            self.engine_name = "dlib"

        self.database = FaceDatabase(db_path) if db_path else FaceDatabase()
        self.matcher  = FaceMatcher(threshold=threshold, metric=distance_metric)

        # Comparing vectors from two different models yields nonsense, so fail
        # loudly at construction rather than silently mis-identifying people.
        stored = self.database.engines_in_use()
        foreign = stored - {self.engine_name}
        if foreign:
            raise ValueError(
                f"Database contains embeddings from {sorted(foreign)} but this system "
                f"uses '{self.engine_name}'. Re-enrol everyone with the new engine, or "
                f"construct the system with engine matching the stored data."
            )

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def enroll_from_image(
        self,
        name: str,
        image_path: str,
        require_single_face: bool = True,
    ) -> EnrollOutcome:
        """
        Enroll a person from an image file.

        Returns
        -------
        EnrollOutcome — truthy when at least one embedding was stored.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = face_recognition.load_image_file(image_path)
        return self.enroll_from_array(name, image, require_single_face=require_single_face)

    def enroll_from_directory(
        self,
        name: str,
        directory: str,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        require_single_face: bool = True,
    ) -> int:
        """Enroll all images in *directory* as *name*. Returns total enrolled."""
        total = 0
        for fname in sorted(os.listdir(directory)):
            if fname.lower().endswith(extensions):
                path = os.path.join(directory, fname)
                total += self.enroll_from_image(
                    name, path, require_single_face=require_single_face
                ).enrolled
        return total

    def enroll_from_array(
        self,
        name: str,
        image: np.ndarray,
        require_single_face: bool = True,
    ) -> EnrollOutcome:
        """
        Enroll from an in-memory RGB numpy array.

        With `require_single_face` (the default) a photo containing more than
        one face is refused rather than enrolled. Storing every face in a
        group photo under one name silently poisons the database: the person
        then matches strangers, and there is no way to tell which embedding
        was the wrong one.
        """
        locations = self.detector.detect(image)

        if not locations:
            return EnrollOutcome(0, 0, "No face found in the photo.")

        if require_single_face and len(locations) > 1:
            return EnrollOutcome(
                0,
                len(locations),
                f"Found {len(locations)} faces. Use a photo of just this person.",
            )

        embeddings = self.embedder.embed(image, face_locations=locations)
        if not embeddings:
            return EnrollOutcome(0, len(locations), "Could not read the face clearly.")

        self.database.enroll_batch(name, embeddings, engine=self.engine_name)
        logger.info("Enrolled %d face(s) for '%s'", len(embeddings), name)
        return EnrollOutcome(len(embeddings), len(locations))

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------

    def identify_from_image(
        self, image_path: str, threshold: float | None = None
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
        return self._identify(image, threshold=threshold)

    def identify_from_array(
        self, image: np.ndarray, threshold: float | None = None
    ) -> list[tuple[tuple, MatchResult]]:
        """
        Identify all faces in an RGB numpy image array.

        `threshold` overrides the rejection gate for this call only, so a web
        request can vary strictness without rebuilding the whole system (which
        re-loads the dlib models and re-reads the database from disk).
        """
        return self._identify(image, threshold=threshold)

    def _identify(
        self, image: np.ndarray, threshold: float | None = None
    ) -> list[tuple[tuple, MatchResult]]:
        locations = self.detector.detect(image)
        if not locations:
            logger.info("No faces detected in query image.")
            return []

        embeddings = self.embedder.embed(image, face_locations=locations)
        db = self.database.get_all_embeddings()
        results = self.matcher.match_many(embeddings, db, threshold=threshold)
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
        Draw bounding boxes and labels on an RGB image, returning RGB.

        Colours match the web UI: green for a match, amber for a rejection.
        (The previous colours were written BGR-first but applied to an RGB
        array, so the intended red "unknown" box rendered blue.)
        """
        out = image.copy()
        height, width = out.shape[:2]

        for (top, right, bottom, left), result in face_results:
            color = COLOR_KNOWN if result.is_known else COLOR_UNKNOWN
            label = (
                f"{result.name} ({result.confidence():.0%})"
                if result.is_known
                else "Unknown"
            )

            cv2.rectangle(out, (left, top), (right, bottom), color, 2)

            # Keep the label strip on-screen when the face touches the top edge.
            strip_bottom = top
            strip_top = top - 28
            if strip_top < 0:
                strip_top, strip_bottom = bottom, bottom + 28
            strip_top = max(0, min(strip_top, height))
            strip_bottom = max(0, min(strip_bottom, height))

            if strip_bottom > strip_top:
                cv2.rectangle(
                    out, (left, strip_top), (min(right, width), strip_bottom), color, -1
                )
                cv2.putText(
                    out, label,
                    (left + 4, strip_bottom - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_LABEL, 1,
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
