"""
embed.py — Cache dlib ResNet-128 embeddings for a whole image set.

These embeddings serve two purposes:
  1. the baseline the trained models must beat, and
  2. the input features for the Track B head.

LFW's funnelled frames are already aligned, so a fixed face box is used rather
than running the HOG detector on every image. That is both much faster and
more consistent: a detector that misses one frame would silently drop it from
the benchmark and quietly change the population being measured.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

logger = logging.getLogger(__name__)

# (top, right, bottom, left) for a 250x250 funnelled LFW frame.
LFW_FACE_BOX = (61, 189, 189, 61)

_EMBEDDING_DIM = 128


def _encode_chunk(payload: tuple[np.ndarray, tuple[int, int, int, int], str]) -> np.ndarray:
    """Worker: embed a block of images with one fixed face box each."""
    images, box, model = payload
    import face_recognition

    out = np.zeros((len(images), _EMBEDDING_DIM), dtype=np.float64)
    for i, image in enumerate(images):
        encodings = face_recognition.face_encodings(
            np.ascontiguousarray(image), known_face_locations=[box], model=model, num_jitters=1
        )
        if encodings:
            out[i] = encodings[0]
    return out


def embed_images(
    images: np.ndarray,
    box: tuple[int, int, int, int] = LFW_FACE_BOX,
    model: str = "large",
    workers: int | None = None,
    chunk_size: int = 64,
) -> np.ndarray:
    """
    Compute dlib embeddings for `images` (uint8 RGB), parallelised over processes.

    dlib releases the GIL only partially, so processes rather than threads are
    what actually use the cores here.
    """
    if workers is None:
        workers = max(1, min(os.cpu_count() or 4, 12))

    chunks = [
        (images[i : i + chunk_size], box, model)
        for i in range(0, len(images), chunk_size)
    ]

    logger.info("Embedding %d images with %d workers …", len(images), workers)
    started = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, part in enumerate(pool.map(_encode_chunk, chunks), start=1):
            results.append(part)
            if done % 20 == 0 or done == len(chunks):
                seen = min(done * chunk_size, len(images))
                rate = seen / max(time.time() - started, 1e-9)
                logger.info("  %d/%d images (%.0f/s)", seen, len(images), rate)

    embeddings = np.concatenate(results, axis=0)
    blank = int((np.abs(embeddings).sum(axis=1) == 0).sum())
    if blank:
        logger.warning("%d image(s) produced no embedding.", blank)
    logger.info("Embedded %d images in %.0fs", len(images), time.time() - started)
    return embeddings


def load_or_compute(cache_path: str, images: np.ndarray, **kwargs) -> np.ndarray:
    """Return cached embeddings, computing and saving them on a miss."""
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if len(cached) == len(images):
            logger.info("Loaded %d cached embeddings from %s", len(cached), cache_path)
            return cached
        logger.warning("Cache %s has %d rows, expected %d — recomputing.",
                       cache_path, len(cached), len(images))

    embeddings = embed_images(images, **kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.save(cache_path, embeddings)
    logger.info("Cached embeddings to %s", cache_path)
    return embeddings
