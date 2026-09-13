# Open-source projects worth incorporating

A survey of what exists, what it would buy this system, and what it costs.
Every accuracy figure below was measured on **this repository's own LFW
protocol** — 462 held-out identities, 6,000 balanced pairs, no identity shared
with training (see [`ml/README.md`](../ml/README.md)) — not copied from a
project's own README.

---

## Summary

| project | licence | replaces | verdict |
|---|---|---|---|
| [OpenCV Zoo — YuNet + SFace](https://github.com/opencv/opencv_zoo) | MIT / Apache 2.0 | detector + embedder | **Adopted** — implemented as an opt-in engine |
| [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) | Apache 2.0 | — (new capability) | **Recommended** — closes a real security hole |
| [InsightFace](https://github.com/deepinsight/insightface) | MIT code, **non-commercial models** | embedder | Conditional — licence blocks commercial use |
| [DeepFace](https://github.com/serengil/deepface) | MIT (wrapper) | whole pipeline | Useful for comparison, not as a dependency |
| [facenet-pytorch](https://github.com/timesler/facenet-pytorch) | MIT | embedder | **Avoid** — unmaintained |
| [FAISS](https://github.com/facebookresearch/faiss) / [hnswlib](https://github.com/nmslib/hnswlib) | MIT / Apache 2.0 | matcher | **Not needed** — measured, see below |

---

## 1. OpenCV Zoo: YuNet + SFace — adopted

**What it is.** Two ONNX models maintained by the OpenCV project: YuNet
(detection + 5 landmarks, MIT) and SFace (128-d recognition, Apache 2.0).

**Why it stands out: no new dependencies.** Both are consumed through
`cv2.FaceDetectorYN` and `cv2.FaceRecognizerSF`, which are already part of the
`opencv-python` this project depends on. Confirmed present in the installed
OpenCV 5.0.0. The only addition is a 37 MB weights download.

**Measured on our LFW split:**

| metric | dlib (current) | YuNet + SFace |
|---|---:|---:|
| AUC | **0.9954** | 0.9902 |
| EER | 2.67% | **1.83%** |
| accuracy | 97.50% | **98.50%** |
| TAR @ FAR=1% | 95.6% | **97.6%** |
| TAR @ FAR=0.1% | 90.3% | **96.8%** |
| d′ | 4.08 | **4.43** |
| throughput | ~14 img/s (12 processes) | **~90 img/s (single thread)** |
| detector misses | — | **0 / 2223** |

dlib keeps a marginally higher AUC. That is a ranking measure integrated over
every threshold including absurd ones, and it is dominated by tail behaviour.
At every operating point a deployment would actually pick, SFace wins — most
decisively at FAR = 0.1%, where it accepts 96.8% of genuine faces against
dlib's 90.3%. If false accepts are the thing you care about, that 6.5-point
gap is the headline.

**Alignment is not optional.** A first port that fed SFace plain centre crops
scored 94.58% — *worse* than dlib. Running the intended pipeline
(YuNet landmarks → `alignCrop` similarity transform → `feature`) took it to
98.50%. Nearly four points of accuracy live in that one step, which is also
why the same trick is listed as future work for the dlib path.

**Secondary benefits.** It removes the dlib dependency entirely for anyone who
opts in — no CMake, no Visual Studio Build Tools, and no exposure to the
`pkg_resources` breakage that currently stops this project running on
setuptools ≥ 81. Startup drops from ~4 s to 0.16 s.

**Status: implemented.** [`src/opencv_engine.py`](../src/opencv_engine.py),
opt-in and off by default:

```python
FaceRecognitionSystem(engine="opencv")   # YuNet + SFace
FaceRecognitionSystem()                  # dlib, unchanged
```

Two safeguards came with it, because swapping embedding models is more
dangerous than it looks:

- **Engines cannot be mixed.** SFace and dlib vectors occupy different spaces,
  so comparing them yields confident nonsense rather than an obvious error.
  The database now records which engine wrote each person, and the system
  refuses to open a database written by a different one.
- **The threshold travels with the engine.** dlib's 0.60 is meaningless to
  SFace, whose fitted operating point is ~1.17 on L2-normalised vectors.

---

## 2. Silent-Face-Anti-Spoofing — recommended

**The gap it closes: a printed photograph currently unlocks this system.**
There is no liveness check anywhere in the pipeline, so holding a phone
displaying someone's face to the camera authenticates as that person. For
anything resembling access control this is the most serious remaining
weakness — more than a point of accuracy.

**What it is.** MiniVision's MiniFASNet, Apache 2.0, explicitly free for
commercial use. A pruned MobileFaceNet (0.081 GFLOPs) using Fourier-spectrum
auxiliary supervision to tell a real face from a photo or screen replay.

**Integration sketch.** A `LivenessChecker` alongside the detector, gating
`identify` before matching: a face that fails liveness returns "spoof
suspected" rather than a name. The web UI already has a warning state
(amber result card) that fits this without redesign.

**Caveat.** Published accuracy comes from the authors' own data; anti-spoofing
generalises poorly across cameras and lighting. Treat it as raising the cost
of an attack, not eliminating it, and measure it on your own capture setup
before relying on it.

---

## 3. InsightFace — strongest models, blocking licence

**What it is.** The reference ArcFace implementation, 29.7k stars, actively
developed (a liveness addon shipped 2026-09-09). The `buffalo_l` pack —
SCRFD detection, 5-point alignment, ArcFace R100 — is the strongest
open face pipeline available, around 99.8% on LFW against dlib's 99.38%.

**Why it is not adopted.** The code is MIT, but **the pretrained models are
released for non-commercial research only**, and `buffalo_l` specifically
directs you to `recognition-oss-pack@insightface.ai` for licensing. For an
academic assignment that is fine; for anything shipped it is a blocker, and
the restriction is on exactly the part that provides the value.

**If the licence works for you**, this is the biggest single accuracy jump
available and slots in as a third engine behind the same interface, at the
cost of an `onnxruntime` dependency and ~350 MB of weights.

---

## 4. DeepFace — useful reference, poor dependency

MIT-licensed wrapper over VGG-Face, FaceNet, ArcFace, Dlib, SFace,
GhostFaceNet and Buffalo_L behind one API. Genuinely useful for *evaluating*
alternatives quickly.

Not recommended as a dependency here: it pulls TensorFlow, and the wrapper's
MIT licence does not extend to the models it downloads — each carries its own
terms, which is easy to miss. Our `ml/benchmark.py` already provides the
comparison harness that DeepFace would be used for.

---

## 5. facenet-pytorch — avoid

MTCNN + InceptionResnetV1 on VGGFace2, MIT, ~5.1k stars, and a common
suggestion for exactly this use case. **It is no longer maintained** — no PyPI
release in over twelve months, and dependency scanners flag it as inactive. Its
accuracy also sits below SFace while requiring a full PyTorch runtime. There is
no reason to take it over the OpenCV path.

---

## 6. Vector search (FAISS / hnswlib) — measured, not needed

The matcher scans every enrolled embedding in a Python loop, which looks like
an obvious target for a vector index. Measured against 9,164 real embeddings:

| enrolled people | current matcher | vectorised numpy |
|---:|---:|---:|
| 100 | 0.68 ms | 0.07 ms |
| 2,000 | 8.8 ms | 1.9 ms |
| 9,000 | 21.6 ms | 7.1 ms |

At 9,000 people matching costs 21 ms, against roughly 200 ms to embed the
query face with dlib. **Search is not the bottleneck and will not be until
~100k identities.** Adding FAISS would mean a native dependency, an index to
keep in sync with the JSON store, and approximate results — to optimise 10% of
a request.

Worth doing instead, for free: replace the per-person Python loop with one
stacked `np.linalg.norm` call, a 3–10× win with no new dependency. The larger
cost in `/identify` is re-decoding the entire JSON database from base64 on
every request, which a cached, stacked matrix would remove.

---

## Recommended order

1. **Evaluate `engine="opencv"` on your own photographs** — implemented,
   opt-in, better at every operating point, and it removes the dlib install
   pain. Re-enrolment is required; the guard enforces it.
2. **Add liveness** — the only change on this list that closes a security
   hole rather than moving a metric.
3. **Vectorise the matcher and cache the decoded database** — an afternoon,
   no new dependencies.
4. **InsightFace** only if its model licence fits your deployment.

---

## Sources

- [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo) — [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT), [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) (Apache 2.0)
- [OpenCV DNN face detection & recognition tutorial](https://docs.opencv.org/4.13.0/d0/dd4/tutorial_dnn_face.html)
- [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) — [licence](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/LICENSE)
- [deepinsight/insightface](https://github.com/deepinsight/insightface)
- [serengil/deepface](https://github.com/serengil/deepface) — [licence](https://github.com/serengil/deepface/blob/master/LICENSE)
- [timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch) — [maintenance status](https://snyk.io/advisor/python/facenet-pytorch)
