"""
generate_sample_data.py — Generate synthetic sample data using face_recognition
and OpenCV so the system can be demonstrated without any real photos.

Creates:
  sample_data/
    Alice/  → 3 synthetic face images
    Bob/    → 3 synthetic face images
    probe/  → test images for each person + 1 unknown
  probes.csv  → evaluation probe list
"""

import os
import csv
import cv2
import numpy as np


def draw_face(name: str, variation: int, size: int = 200) -> np.ndarray:
    """
    Draw a simple cartoon face with a name label.
    Each variation tweaks colour/feature positions slightly.
    """
    rng = np.random.default_rng(hash(name + str(variation)) % (2**31))
    img = np.ones((size, size, 3), dtype=np.uint8) * 240

    # Background tint
    img[:, :] = rng.integers(180, 230, size=3, dtype=np.uint8)

    # Face circle
    cx, cy = size // 2, size // 2
    r = size // 3
    face_color = tuple(int(c) for c in rng.integers(180, 220, size=3, dtype=np.uint8).tolist())
    cv2.circle(img, (cx, cy), r, face_color, -1)

    # Eyes
    ey = cy - r // 5 + rng.integers(-5, 5)
    ex1 = cx - r // 3 + rng.integers(-4, 4)
    ex2 = cx + r // 3 + rng.integers(-4, 4)
    cv2.circle(img, (ex1, ey), r // 8, (30, 30, 30), -1)
    cv2.circle(img, (ex2, ey), r // 8, (30, 30, 30), -1)

    # Mouth
    my = cy + r // 3 + rng.integers(-5, 5)
    cv2.ellipse(img, (cx, my), (r // 4, r // 8), 0, 0, 180, (50, 50, 50), 2)

    # Name label
    cv2.putText(img, f"{name[:6]} v{variation}", (4, size - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    return img


def generate(out_dir: str = "sample_data") -> list[tuple[str, str]]:
    persons = ["Alice", "Bob", "Charlie"]
    probes: list[tuple[str, str]] = []

    os.makedirs(out_dir, exist_ok=True)
    probe_dir = os.path.join(out_dir, "probe")
    os.makedirs(probe_dir, exist_ok=True)

    for person in persons:
        person_dir = os.path.join(out_dir, person)
        os.makedirs(person_dir, exist_ok=True)

        for i in range(3):  # 3 enrollment images
            img = draw_face(person, i)
            path = os.path.join(person_dir, f"{person.lower()}_{i}.jpg")
            cv2.imwrite(path, img)

        # 2 probe images per person (genuine)
        for i in range(3, 5):
            img = draw_face(person, i)
            path = os.path.join(probe_dir, f"{person.lower()}_probe_{i}.jpg")
            cv2.imwrite(path, img)
            probes.append((path, person))

    # 2 unknown probes (new person not enrolled)
    unknown_dir = os.path.join(out_dir, "Unknown")
    os.makedirs(unknown_dir, exist_ok=True)
    for i in range(2):
        img = draw_face("Dave", i + 10)
        path = os.path.join(probe_dir, f"unknown_{i}.jpg")
        cv2.imwrite(path, img)
        probes.append((path, "Unknown"))

    # Write probes.csv
    csv_path = os.path.join(os.path.dirname(out_dir), "probes.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerows([{"image_path": p, "label": l} for p, l in probes])

    print(f"Sample data written to '{out_dir}/'")
    print(f"Probe CSV: '{csv_path}'")
    return probes


if __name__ == "__main__":
    generate()
