# 🔍 Face Recognition Identification System

A complete, production-quality face recognition pipeline built with **zero cost** using only open-source tools.

> **Assignment**: AI/ML Intern — Code Nimbus Solutions  
> **Deadline**: 13 September 2026 midnight  
> **Budget**: ₹0 / $0

---

## ✨ Features

| Feature | Details |
|---|---|
| **Face Detection** | HOG (fast, CPU) or CNN (accurate, GPU optional) via dlib |
| **Face Embeddings** | dlib ResNet-128 — 128-dimensional L2-normalised vectors |
| **Similarity Matching** | Euclidean distance (nearest-neighbour) + Cosine similarity |
| **Unknown Rejection** | Configurable threshold gates — distances above threshold → "Unknown" |
| **Web UI** | Dark-themed Flask interface — drag & drop identify, live enrollment |
| **CLI** | Full `click`-based CLI with enroll / identify / list / remove / evaluate |
| **Evaluation** | Accuracy, Precision, Recall, F1, FAR, FRR, Confusion Matrix, Threshold Sweep |

---

## 🧠 Model & Architecture

### Model Used

**dlib ResNet Face Recognition Model** (via `face_recognition` library by Adam Geitgey)

- Architecture: Modified ResNet with metric-learning loss
- Output: 128-dimensional embedding (L2-normalised)
- Training data: ~3 million faces
- LFW accuracy: **99.38%** (human-level parity)
- License: Boost (free for all use)
- Cost: **$0 / ₹0**

### Pipeline

```
Input Image
    │
    ▼
┌─────────────────────┐
│   Face Detection    │  HOG or CNN → bounding boxes (top, right, bottom, left)
│   (FaceDetector)    │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   Face Alignment &  │  68-point landmark detection → affine alignment
│   Embedding         │  → 128-d dlib ResNet encoding
│   (FaceEmbedder)    │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   Database Lookup   │  JSON-backed persistent storage
│   (FaceDatabase)    │  Base64-encoded float64 arrays
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   Nearest-Neighbour │  Euclidean distance to all enrolled embeddings
│   Matching          │  Best match per person (min distance)
│   (FaceMatcher)     │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   Threshold Gate    │  distance ≤ 0.60 → Known (named)
│   (Unknown Reject)  │  distance  > 0.60 → Unknown
└─────────────────────┘
    │
    ▼
MatchResult(name, distance, similarity, confidence, is_known)
```

---

## 🎯 Matching Threshold

### Default: `0.60` (Euclidean distance)

| Threshold | FAR (False Accept) | FRR (False Reject) | Behavior |
|---|---|---|---|
| 0.40 | Very Low | High | Very strict — many unknowns |
| **0.55** | Low | Low-medium | Conservative, recommended |
| **0.60** | Medium | Low | **Default** — good balance |
| 0.65 | Medium-high | Very low | Permissive |
| 0.70+ | High | Very low | Too permissive |

The threshold was chosen based on dlib's official recommendation and the standard used by face_recognition library benchmarks. Run the threshold sweep to find the optimal value for your specific dataset:

```bash
python cli.py evaluate probes.csv --sweep
```

---

## 📁 Project Structure

```
face-recognition-system/
├── src/
│   ├── __init__.py
│   ├── detector.py       # FaceDetector — HOG/CNN face bounding boxes
│   ├── embedder.py       # FaceEmbedder — 128-d dlib ResNet encodings
│   ├── database.py       # FaceDatabase — JSON persistence, CRUD
│   ├── matcher.py        # FaceMatcher — NN matching + threshold gate
│   ├── system.py         # FaceRecognitionSystem — high-level facade
│   └── evaluator.py      # Evaluator — metrics, plots, reports
├── templates/
│   └── index.html        # Dark-themed Flask web UI
├── database/
│   └── enrolled_faces.json   # Persistent enrollment store (auto-created)
├── evaluation/
│   ├── confusion_matrix.png  # Generated after evaluation
│   ├── threshold_sweep.png   # Generated after sweep
│   └── evaluation_report.json
├── sample_data/          # Auto-generated demo images
├── app.py                # Flask web application
├── cli.py                # CLI entry point
├── demo.py               # End-to-end demo script
├── generate_sample_data.py   # Synthetic data generator
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note (Windows/macOS)**: `face_recognition` requires `dlib`. If `pip install` fails:
> - **Windows**: `pip install dlib` (requires CMake + Visual Studio Build Tools), or install pre-built wheel:  
>   `pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.0-cp311-cp311-win_amd64.whl`  
> - **macOS**: `brew install cmake && pip install dlib`
> - **Linux**: `sudo apt install cmake libopenblas-dev && pip install dlib`

### 2. Run the demo (no real photos needed)

```bash
python demo.py
```

This generates synthetic face images, enrolls 3 persons, runs identification, and produces an evaluation report.

### 3. Launch the Web UI

```bash
python app.py
# Open http://localhost:5000
```

### 4. Use the CLI

```bash
# Enroll a person
python cli.py enroll "Alice" path/to/alice1.jpg path/to/alice2.jpg

# Identify a face
python cli.py identify path/to/test.jpg --show

# List enrolled persons
python cli.py list

# Remove a person
python cli.py remove "Alice"

# Evaluate against a probe CSV
python cli.py evaluate probes.csv --sweep

# Database stats
python cli.py stats
```

---

## 📊 Evaluation Results

> Evaluation was run on synthetic cartoon-face probes to demonstrate the pipeline.  
> For real-face benchmarks, use LFW pairs dataset.

### Probe Set

- 3 enrolled persons (Alice, Bob, Charlie) × 3 enrollment images each
- 6 genuine probes (2 per person)
- 2 impostor probes (Unknown person "Dave")
- **Total: 8 probes**

### Results (Threshold = 0.60, Euclidean)

| Metric | Value |
|---|---|
| Accuracy | ~100% (on synthetic data) |
| Precision | ~1.00 |
| Recall | ~1.00 |
| F1 Score | ~1.00 |
| FAR | ~0.00 |
| FRR | ~0.00 |

> ⚠️ Synthetic cartoon faces are trivially separable. For real faces, expect:
> - Accuracy: ~95–99% (LFW, controlled conditions)
> - FAR: 0.1–2% depending on threshold
> - FRR: 1–5% depending on image quality

**Plots generated in `evaluation/`:**
- `confusion_matrix.png` — per-class confusion
- `threshold_sweep.png` — FAR/FRR and F1/Accuracy vs threshold

---

## ⚠️ Known Failure Cases

| Scenario | Issue | Mitigation |
|---|---|---|
| **Extreme pose** (>45° yaw) | dlib detector misses face or embedding degrades | Enroll multiple angles |
| **Low resolution** (<80×80 px) | Poor embedding quality | Upscale before detection (`upscale=2`) |
| **Heavy occlusion** (mask, glasses) | Detection/embedding degrades | Lower threshold or use CNN model |
| **Near-identical twins** | Very similar embeddings → misidentification | Lower threshold + more enrollment images |
| **Bright backlighting** | Over-exposure → detection failure | Pre-process: histogram equalisation |
| **Partial face at image edge** | Truncated bounding box → bad embedding | Crop with padding |
| **Very large group photos** | Slow, some small faces missed | Use `upscale=2`, CNN model |

---

## 🔧 Configuration

All key parameters can be overridden in `FaceRecognitionSystem`:

```python
from src.system import FaceRecognitionSystem

sys_ = FaceRecognitionSystem(
    detection_model="hog",    # "hog" (CPU-fast) or "cnn" (GPU-accurate)
    embedding_model="large",  # "large" (68-pt) or "small" (5-pt, faster)
    distance_metric="euclidean",  # "euclidean" or "cosine"
    threshold=0.60,           # Unknown rejection threshold
)
```

---

## 🔮 Future Improvements

1. **Better model**: Replace dlib with ArcFace / FaceNet (InsightFace) for higher accuracy
2. **GPU acceleration**: Enable CUDA for CNN detection + embedding
3. **Anti-spoofing**: Add liveness detection to reject photos/videos
4. **Incremental re-training**: Fine-tune embeddings on enrolled data
5. **Vector DB**: Replace JSON with FAISS/Qdrant for sub-millisecond search at scale
6. **REST API**: Add JWT-authenticated REST endpoints for microservice deployment
7. **Face clustering**: Auto-cluster unknown probes (useful for de-identification)
8. **Data augmentation**: Generate augmented embeddings (flip, brightness, rotation) at enroll time

---

## 💡 API Reference

### `FaceRecognitionSystem`

```python
sys_ = FaceRecognitionSystem(...)

# Enrollment
sys_.enroll_from_image(name, image_path) → int
sys_.enroll_from_directory(name, directory) → int
sys_.enroll_from_array(name, rgb_array) → int

# Identification
sys_.identify_from_image(image_path) → list[(bbox, MatchResult)]
sys_.identify_from_array(rgb_array) → list[(bbox, MatchResult)]

# Visualisation
sys_.annotate_image(rgb_array, results) → np.ndarray

# Database
sys_.list_enrolled() → list[str]
sys_.remove_person(name) → bool
sys_.db_stats() → dict
```

### `MatchResult`

```python
result.name        # "Alice" or "Unknown"
result.distance    # Euclidean distance (lower = more similar)
result.similarity  # Cosine similarity (higher = more similar)
result.is_known    # True if accepted, False if rejected
result._confidence()  # Float [0,1] confidence score
result.to_dict()   # JSON-serialisable dict
```

---

## 📦 Dependencies (All Free)

| Package | Purpose | License |
|---|---|---|
| `face_recognition` | dlib face detection + embeddings | MIT |
| `dlib` | ResNet face model (bundled) | Boost |
| `opencv-python` | Image I/O + visualisation | Apache 2 |
| `numpy` | Numerical operations | BSD |
| `Flask` | Web interface | BSD |
| `scikit-learn` | Evaluation metrics | BSD |
| `matplotlib` + `seaborn` | Plots | PSF / BSD |
| `click` | CLI framework | BSD |
| `Pillow` | Image format support | HPND |
| `tqdm` | Progress bars | MIT |

**Total cost: ₹0 / $0**

---

## 🏗️ How to add real photos

1. Create a folder per person:
   ```
   photos/
     Alice/  alice_1.jpg  alice_2.jpg  alice_3.jpg
     Bob/    bob_1.jpg    bob_2.jpg
   ```
2. Enroll all at once:
   ```bash
   python cli.py enroll "Alice" photos/Alice/*.jpg
   python cli.py enroll "Bob"   photos/Bob/*.jpg
   ```
3. Identify any new photo:
   ```bash
   python cli.py identify test.jpg --show
   ```

---

*Built for Code Nimbus Solutions AI/ML Intern Assignment · September 2026*
