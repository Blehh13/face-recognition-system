#!/usr/bin/env python3
"""
build_dataset.py — Assemble the test set from Wikimedia Commons.

    python Test/build_dataset.py

Fetches photographs of public figures, keeps only those whose licence is
explicitly **public domain**, and lays them out as:

    Test/enrolled/<Person Name>/*.jpg   two photographs used to enrol
    Test/probe/<Person Name>_N.jpg      a different photograph, used to query
    Test/probe/unknown_N.jpg            people who were never enrolled

Why these people: they are US federal employees (NASA, US Navy), so their
official photographs are public domain by statute rather than by permission.
That matters because this directory is committed to a public repository — a
face dataset assembled from ordinary web images would be neither licensed for
redistribution nor consented to by the people in it.

Every accepted file is recorded in MANIFEST.json with its source URL, licence
and credit, so the provenance of the dataset can be checked rather than taken
on trust.

Selection is automatic but not blind: a candidate is rejected unless the
detector finds exactly one face, since a group shot would poison an enrolment
and a miss would silently shrink the test set.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "FaceRecognitionAssignment/1.0 (educational use)"}

# Licences we will redistribute. Anything else is skipped, including CC-BY
# variants — they are usable with attribution, but keeping the rule to a single
# unambiguous category makes the dataset trivially safe to publish.
ACCEPTED_LICENCES = ("public domain", "pd-")

# Three, because averaging a person's photographs only helps once there is
# something to average: measured EER falls from 2.71% at two photographs to
# 0.44% at three (python -m ml.aggregation).
ENROL_PER_PERSON = 3
PROBES_PER_PERSON = 1

# (folder name, Commons search term)
ENROLLED_PEOPLE = [
    ("Grace Hopper",      "Grace Hopper computer scientist"),
    ("Katherine Johnson", "Katherine Johnson NASA"),
    ("Sally Ride",        "Sally Ride astronaut"),
    ("Mae Jemison",       "Mae Jemison astronaut"),
    ("Buzz Aldrin",       "Buzz Aldrin astronaut"),
]

# Never enrolled: these test that a stranger is rejected rather than
# force-matched onto the closest enrolled person.
IMPOSTORS = [
    ("unknown_1", "Alan Shepard astronaut"),
    ("unknown_2", "Gus Grissom astronaut"),
    ("unknown_3", "Christina Koch astronaut"),
]


_last_call = [0.0]


def api(params: dict, attempts: int = 7) -> dict:
    """Query the Commons API, throttled and backing off on rate limits."""
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    for attempt in range(attempts):
        # Commons returns 429 quickly if called in a tight loop.
        wait = 1.8 - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < attempts - 1:
                backoff = min(2 ** attempt * 2, 45)
                print(f"    rate limited, waiting {backoff}s")
                time.sleep(backoff)
                continue
            raise
    raise RuntimeError("unreachable")


def search_files(term: str, limit: int = 20) -> list[str]:
    result = api({
        "action": "query", "list": "search",
        "srsearch": f"{term} filetype:bitmap",
        "srnamespace": "6", "srlimit": limit,
    })
    return [hit["title"] for hit in result["query"]["search"]]


def file_details(titles: list[str]) -> dict:
    details = {}
    for start in range(0, len(titles), 10):
        chunk = titles[start:start + 10]
        result = api({
            "action": "query", "titles": "|".join(chunk),
            "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": "800",
        })
        for page in result["query"]["pages"].values():
            info = (page.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata", {})
            details[page["title"]] = {
                "licence": meta.get("LicenseShortName", {}).get("value", "unknown"),
                "credit": _strip_html(meta.get("Artist", {}).get("value", "unknown")),
                "download": info.get("thumburl") or info.get("url"),
                "page": info.get("descriptionurl", ""),
            }
    return details


def _strip_html(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", value)).strip()[:120]


def base_image(title: str) -> str:
    """
    Collapse Commons variants of one photograph to a single key.

    Commons routinely holds "X.jpg" alongside "X (cropped).jpg" and
    "X - Original.jpg". They are the same photograph, and letting a crop
    become the probe for an enrolment of the original measures whether the
    system can retrieve an image, not whether it can recognise a face — it
    produced a distance of 0.037, which is not a real recognition result.
    """
    name = title.lower()
    name = re.sub(r"^file:", "", name)
    name = re.sub(r"\.(jpe?g|png|webp|bmp)$", "", name)
    name = re.sub(r"\s*[\(\[](cropped|crop|clipped|retouched|edited)[^)\]]*[\)\]]", "", name)
    name = re.sub(r"\s*[-–]\s*(original|cropped|crop|edit).*$", "", name)
    return re.sub(r"[^a-z0-9]+", "", name)


def is_public_domain(licence: str) -> bool:
    return any(token in licence.lower() for token in ACCEPTED_LICENCES)


def download(url: str, path: str) -> bool:
    try:
        request = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        if len(data) < 5000:
            return False
        with open(path, "wb") as handle:
            handle.write(data)
        return True
    except Exception:
        return False


# A detected face smaller than this fraction of the frame means the photograph
# is a wide shot in which the subject is incidental. One such image slipped in
# as a Grace Hopper probe: a three-person White House photo where the only
# detectable face was Ronald Reagan's, at 0.75% of frame. The system correctly
# rejected him, and the test counted that as a recognition failure.
MIN_FACE_FRACTION = 0.03


def single_face(path: str, detector) -> bool:
    """Accept only portraits: exactly one face, and large enough to be the subject."""
    try:
        import numpy as np
        from PIL import Image
        image = np.array(Image.open(path).convert("RGB"))
    except Exception:
        return False
    try:
        boxes = detector.detect(image)
    except Exception:
        return False
    if len(boxes) != 1:
        return False
    top, right, bottom, left = boxes[0]
    fraction = ((right - left) * (bottom - top)) / (image.shape[0] * image.shape[1])
    return fraction >= MIN_FACE_FRACTION


USED_TITLES: set[str] = set()
USED_BASES: set[str] = set()


def collect(term: str, detector, plan: list[tuple[str, int, str]], manifest: list) -> None:
    """
    Run one search, then fill each (destination, count, prefix) slot from it.

    A single search per person keeps the API traffic low — querying twice per
    person is what tripped Commons' rate limiter — and guarantees enrolment and
    probe photographs are distinct files, since every accepted title is
    consumed from the same pool.
    """
    titles = [t for t in search_files(term, limit=40) if t not in USED_TITLES]
    details = file_details(titles)
    candidates = [(t, m) for t, m in details.items()
                  if m["download"] and is_public_domain(m["licence"])
                  and t not in USED_TITLES and base_image(t) not in USED_BASES]

    index = 0
    for dest_dir, wanted, prefix in plan:
        os.makedirs(dest_dir, exist_ok=True)
        kept = 0
        while kept < wanted and index < len(candidates):
            title, meta = candidates[index]
            index += 1
            # Re-check: an earlier slot in this same plan may have taken the
            # original that this candidate is a crop of.
            if base_image(title) in USED_BASES:
                continue

            filename = f"{prefix}_{kept + 1}.jpg"
            path = os.path.join(dest_dir, filename)
            if not download(meta["download"], path):
                continue
            if not single_face(path, detector):
                os.remove(path)
                continue

            manifest.append({
                "file": os.path.relpath(path, HERE).replace("\\", "/"),
                "person": prefix,
                "commons_title": title,
                "licence": meta["licence"],
                "credit": meta["credit"],
                "source": meta["page"],
            })
            USED_TITLES.add(title)
            USED_BASES.add(base_image(title))
            kept += 1
            print(f"    {os.path.basename(dest_dir)}/{filename}  [{meta['licence']}]")

        if kept < wanted:
            print(f"    WARNING: only {kept}/{wanted} for {prefix} in {os.path.basename(dest_dir)}")


def main() -> int:
    from src.detector import FaceDetector
    detector = FaceDetector()

    enrolled_root = os.path.join(HERE, "enrolled")
    probe_root = os.path.join(HERE, "probe")
    os.makedirs(probe_root, exist_ok=True)

    manifest: list[dict] = []

    for name, term in ENROLLED_PEOPLE:
        print()
        print(name)
        folder = name.replace(" ", "_")
        collect(term, detector, [
            (os.path.join(enrolled_root, folder), ENROL_PER_PERSON, folder),
            (probe_root, PROBES_PER_PERSON, folder),
        ], manifest)

    print()
    print("Impostors (never enrolled)")
    for label, term in IMPOSTORS:
        collect(term, detector, [(probe_root, 1, label)], manifest)

    with open(os.path.join(HERE, "MANIFEST.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "source": "Wikimedia Commons",
            "licence_policy": "public domain only",
            "files": manifest,
        }, handle, indent=2)

    print(f"\n{len(manifest)} files recorded in MANIFEST.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
