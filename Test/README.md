# Test set

A face dataset committed to the repository so the evaluation can be reproduced
and inspected rather than taken on trust.

**15 people enrolled from 41 public-domain photographs, then 32 probes
identified against them.**

| metric | value |
|---|---:|
| Accuracy | **87.5%** (28/32) |
| 95% confidence interval | 71.9% – 95.0% |
| Precision | 1.00 |
| Recall | 0.82 |
| F1 | 0.90 |
| False accept rate | **0%** |
| False reject rate | 18.2% |

Shipped defaults: YuNet + SFace, averaged enrolment embeddings, threshold
1.012. Per-probe distances in [results/results.md](results/results.md).

```bash
python Test/build_dataset.py    # fetch the images (needs internet, once)
python Test/run_test.py         # enrol, identify, write results
```

## Layout

```
Test/
  enrolled/<Person Name>/*.jpg   up to 3 photographs each, used to enrol
  probe/<Person Name>_N.jpg      DIFFERENT photographs, used to query
  probe/unknown_*.jpg            10 people who were never enrolled
  MANIFEST.json                  source URL, licence and credit per file
  results/                       results.json and results.md
```

## What the result says

**No stranger was ever accepted.** All ten impostors were rejected, and every
name the system did produce was correct — precision 1.00. In a face
recognition system that is the error direction that matters: admitting the
wrong person is worse than failing to admit the right one.

**It rejected four genuine faces out of 22.** John Glenn (both probes), Peggy
Whitson and Victor Glover came back Unknown. The system is conservative at this
threshold, and that is the trade it was tuned to make.

### Why those four failed, and why the threshold was not changed

| probe | distance | |
|---|---:|---|
| Peggy Whitson 1 | 1.023 | below the closest impostor |
| Victor Glover 2 | 1.059 | below the closest impostor |
| John Glenn 2 | 1.063 | below the closest impostor |
| John Glenn 1 | **1.242** | **farther than every real stranger** |

The closest impostor sits at 1.089. Loosening the gate to ~1.08 would recover
three of the four failures without admitting a single stranger *on this set*.

**That change has not been made, deliberately.** The threshold is fitted on LFW
validation identities that appear in neither training nor this test set. Moving
it to whatever happens to suit these 32 probes would make the number here
meaningless — it would measure the tuning, not the system.

The fourth failure could not be fixed by any threshold: John Glenn's first
probe is farther from his enrolled photographs than several unrelated people
are. Genuine and impostor distances **overlap** in this population, which is
the real limit — his enrolment photographs are from the Mercury era and the
probe is decades later.

## Licensing

Every image is **public domain**, verified against the Wikimedia Commons
licence field at download time; the fetcher rejects anything else, including
CC-BY. The subjects are US federal employees (NASA, US Navy), so their official
photographs are public domain by statute rather than by permission.

That matters because this directory is public: a face dataset scraped from
ordinary web images would be neither licensed for redistribution nor consented
to by the people in it. `MANIFEST.json` records the Commons page and credit for
all 74 files.

## Selection rules, and why they are frozen

Probe photographs are **different images** from the ones used to enrol, so this
measures recognition rather than recall of a stored picture. The ten
`unknown_*` probes are people the system has never seen; without them the
accuracy figure would be close to meaningless, since a system that accepts
everybody scores perfectly on genuine probes alone.

Two rules in `build_dataset.py` exist because earlier versions of this dataset
were quietly invalid:

- **Crops of an already-used photograph are rejected.** Commons holds `X.jpg`
  beside `X (cropped).jpg`; enrolling the original and probing with the crop
  scored a distance of 0.037, which measures image retrieval rather than face
  recognition.
- **The detected face must fill at least 3% of the frame.** One probe was a
  three-person White House photograph whose only detectable face was Ronald
  Reagan's, at 0.75% of frame. The system correctly rejected him as unenrolled
  and the test scored it as a recognition failure.

**These rules are now frozen.** They were both chosen while diagnosing a
specific defect, but they were chosen *after looking at which probes failed* —
and an earlier, smaller version of this set reported 100% partly because of
that. Adjusting selection rules in response to a score is how a test set stops
measuring anything. Whatever this dataset yields is now reported as-is,
including the four failures above.

## Honest caveats

**32 probes is still small.** The 95% confidence interval on 87.5% runs from
71.9% to 95.0%. This demonstrates the system works end to end on real, varied
photographs of fifteen different people; it is not a benchmark.

**Historical photographs are a hard case.** Several subjects are photographed
decades apart — the failures cluster there. A deployment enrolling people from
recent photographs should expect to do better than this set suggests.

**For a statistically meaningful number**, see [`ml/README.md`](../ml/README.md)
and `python -m ml.aggregation`: hundreds of identities held out of training
entirely, reporting 0.44% EER at three enrolment photographs. This directory is
the human-readable demonstration; that one is the measurement.
