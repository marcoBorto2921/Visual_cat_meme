# CatPose

Real-time webcam pose classifier — imitate a cat and see its photo appear on screen.

Pick a set of cat photos with fun, recognizable poses. For each one, strike the pose in front of your webcam. The system learns your specific landmark patterns and classifies your pose in real time.

## How it works

1. **Pick the photos** — drop one image per pose into `assets/cats/`. The filename (without extension) becomes the class label (e.g. `tongue_cat.jpg`, `grumpy_cat.jpg`).
2. **Collect training data** — for each cat, hold the pose in front of the webcam and press SPACE to capture landmark frames.
3. **Train the classifier** — an SVM learns exactly how your body and face landmarks move when you imitate each cat.
4. **Play** — MediaPipe detects your landmarks in real time → the classifier predicts which cat you're imitating → that cat's photo appears.

The system detects **body landmarks** (arms, shoulders, hips) and **face landmarks** (mouth openness, tongue out, head orientation) in parallel. Both are combined into a single feature vector before classification — this makes poses like "tongue out" distinguishable from a normal face, which body pose alone cannot do.

**No CLIP, no generic model.** The system learns *your* specific poses — it's personal and much more accurate.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

### 1. Add cat photos

```
assets/cats/tongue_cat.jpg
assets/cats/grumpy_cat.jpg
assets/cats/surprised_cat.jpg
```

Pick poses that are visually distinct from each other: arms raised, arms crossed, hand on face, tongue out, head tilted, etc.

### 2. Collect samples

```bash
python scripts/collect_samples.py
```

On first run, MediaPipe models are downloaded automatically (~30 MB each). For each cat:
- The photo appears on the right panel
- Mirror your webcam feed is on the left with the skeleton overlay
- Hold the pose and press **SPACE** to capture 30 landmark frames
- Press **N** to skip to the next cat

> For "tongue out" poses: face the webcam directly, make sure your face is well-lit and not backlit. MediaPipe's face model struggles with extreme angles or strong backlighting.

### 3. Train the classifier

```bash
python scripts/train_classifier.py
```

Prints accuracy and a full classification report. If accuracy is below 70%, collect more samples or make your poses more exaggerated.

### 4. Run

```bash
python main.py
```

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit |
| `d` | Toggle debug overlay (top-3 predictions with confidence scores) |
| `r` | Reset smoothing window |
| `s` | Save screenshot to `screenshots/` |

## Configuration

All parameters live in `configs/config.yaml`: camera index, confidence threshold, samples per pose, classifier model (svm/mlp), and more.

The `face:` block controls face landmark detection:
- `face.enabled: true/false` — enable or disable face features entirely. Disable if your webcam doesn't frame your face well or if you only want body-based classification.
- `face.model_path: null` — downloads the FaceLandmarker model automatically.
- `face.visibility_threshold` — landmarks below this threshold are ignored.

## Project structure

```
assets/cats/        ← your cat photos (one per class)
data/samples.csv    ← captured landmark frames (gitignored)
models/             ← classifier.pkl, label_encoder.pkl (gitignored)
src/pose/           ← MediaPipe PoseLandmarker wrapper
src/face/           ← MediaPipe FaceLandmarker wrapper + face feature extraction
src/classifier/     ← feature engineering, training, inference
src/display/        ← OpenCV dual-panel renderer
scripts/            ← collect_samples.py, train_classifier.py
```
