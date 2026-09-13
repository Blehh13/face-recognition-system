# Test set

A small, hand-checkable face dataset committed to the repository so the
evaluation can be reproduced and inspected rather than taken on trust.

**5 people enrolled from 3 photographs each, then 8 probes identified against
them. Accuracy 100% (8/8), precision 1.00, recall 1.00, FAR 0%, FRR 0%.**
Produced with the shipped defaults — YuNet + SFace, averaged enrolment
embeddings, threshold 1.012. Per-probe distances in
[results/results.md](results/results.md).

```bash
python Test/build_dataset.py    # fetch the images (needs internet, once)
python Test/run_test.py         # enrol, identify, write results
```

## Layout

```
Test/
  enrolled/<Person Name>/*.jpg   3 photographs each, used to enrol
  probe/<Person Name>_1.jpg      a DIFFERENT photograph, used to query
  probe/unknown_*.jpg            people who were never enrolled
  MANIFEST.json                  source URL, licence and credit per file
  results/                       results.json and results.md
```

Enrolled: Grace Hopper, Katherine Johnson, Sally Ride, Mae Jemison, Buzz Aldrin.
Impostors (never enrolled): Alan Shepard, Gus Grissom, Christina Koch.

## Licensing

Every image is **public domain**, verified against the Commons API licence
field at download time — the fetcher rejects anything else, including CC-BY,
which is usable but would need attribution handling. These people are US
federal employees (NASA, US Navy), so their official photographs are public
domain by statute rather than by permission.

That matters because this directory is public: a face dataset scraped from
ordinary web images would be neither licensed for redistribution nor consented
to by the people in it. `MANIFEST.json` records the Commons page and credit for
all 18 files so the provenance can be checked.

## Why the probes are a fair test

The probe photograph of each person is a **different image** from the two used
to enrol them, so this measures recognition rather than recall of a stored
picture. The three `unknown_*` probes are people the system has never seen;
without them the accuracy figure would be close to meaningless, since a system
that accepts everybody scores perfectly on genuine probes alone.

Two selection rules in `build_dataset.py` exist because the first attempts at
this dataset were quietly invalid:

- **Crops of an already-used photograph are rejected.** Commons holds
  `X.jpg` next to `X (cropped).jpg`. Enrolling the original and probing with
  the crop gave a distance of 0.037 — that measures image retrieval, not face
  recognition.
- **The detected face must fill at least 3% of the frame.** A Grace Hopper
  probe was a three-person White House photograph in which the only detectable
  face was Ronald Reagan's, at 0.75% of frame. The system correctly rejected
  him as unenrolled, and the test scored that as a recognition failure.

## Honest caveats

**This is a small set.** Eight probes cannot distinguish 100% from 90%; the
confidence interval on 8/8 is wide. It demonstrates the system works end to end
on real, varied photographs of multiple people — it is not a benchmark.

**Sally Ride's probe is the closest call.** It matched at 0.994 against a
threshold of 1.012 — inside the gate by under 2%. Her folder also holds only
two usable photographs rather than three, because the detector rejected one
candidate, and averaging two photographs is measurably weaker than averaging
three. She is the probe most likely to flip if the dataset is rebuilt.

**For a statistically meaningful number**, see [`ml/README.md`](../ml/README.md)
and `python -m ml.aggregation`: identities held out of training entirely, with
0.44% EER at three enrolment photographs. This directory is the human-readable
demonstration; that one is the measurement.
