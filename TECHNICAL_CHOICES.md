# Technical Choices

## Pose Detection — MediaPipe Tasks API

MediaPipe 0.10+ replaced the legacy `solutions` API with the Tasks API (`PoseLandmarker`). The Tasks API is the only supported path on Windows with mediapipe 0.10.33. We use `RunningMode.IMAGE` (per-frame, no async callbacks) for simplicity and predictable latency.

The detector returns all 33 normalized landmarks (x, y, z, visibility). Filtering by visibility is handled downstream in `features.py`, so the detector stays decoupled from the feature engineering strategy.

## Classification — SVM on Landmark Features (not CLIP)

Instead of zero-shot retrieval via CLIP, we train a supervised classifier directly on the user's own pose landmarks. This has several advantages:

- **Personalized**: the model learns how *your specific body* moves when imitating each cat pose — not a generic text-image similarity.
- **Fast**: SVM inference on a 104-dim vector is microseconds on CPU.
- **No large model downloads**: no `torch`, no `transformers`, no multi-GB weights. scikit-learn only.
- **Accurate with few samples**: SVM with RBF kernel is robust with 20–50 samples per class.

## Feature Engineering — Normalized Landmark Vector (104-dim)

The full feature vector is the concatenation of pose features (99-dim) and face features (5-dim).

**Pose features (99-dim)**: Raw landmark coordinates depend on where the user stands relative to the camera. We normalize them:

1. **Center** on the hip midpoint (stable reference point).
2. **Scale** by the shoulder-to-shoulder distance (invariant to camera distance).
3. **Zero-pad** invisible landmarks (visibility below threshold).

Result: a 99-dimensional float32 vector (33 landmarks × x, y, z) representing the *shape* of the pose, not its absolute position.

**Face features (5-dim)**: See "Face Feature Augmentation" below.

## Classifier Choice — SVM with RBF Kernel (default)

`SVC(kernel="rbf", probability=True)` from scikit-learn:

- Excellent generalization with small datasets (20–50 samples per class).
- `predict_proba` gives a confidence score used for the confidence threshold and debug overlay.
- No critical hyperparameters to tune — `C=10, gamma="scale"` works well out of the box.

MLP (`MLPClassifier`) is available as an alternative via `classifier.model: "mlp"` in config, suitable when collecting >100 samples per class.

## Pose Smoothing

A sliding window (default: 7 frames) takes the mode of recent predicted labels to suppress flickering from noisy landmark estimates. The cat photo changes only when the stable label changes.

## Confidence Threshold

If the classifier's max probability is below `confidence_threshold` (default: 0.4), the right panel shows "?" instead of a cat photo — the user is not holding a recognizable pose.

## No GPU Required

SVM inference: < 1 ms on CPU. MediaPipe landmark detection: ~10–20 ms on CPU. Total pipeline easily achieves > 15 FPS on any modern laptop.

## Decision: Face Feature Augmentation

**Choice**: MediaPipe FaceLandmarker (478 landmarks + ARKit blendshapes) running in parallel with PoseLandmarker.

**Rationale**: PoseLandmarker treats the face as a single point (landmark 0 = nose). It cannot distinguish open mouth, tongue out, or head orientation. FaceLandmarker adds 5 scalar features that make these poses separable by the SVM:

| Index | Feature | Source | Description |
|-------|---------|--------|-------------|
| 0 | `jaw_open` | blendshape | jawOpen score [0,1] — how wide the jaw is dropped |
| 1 | `tongue_out` | blendshape | tongueOut score [0,1] — direct tongue detection |
| 2 | `head_yaw` | landmarks | horizontal head rotation, normalized to [-1, 1] |
| 3 | `head_pitch` | landmarks | vertical tilt, normalized to [-1, 1] |
| 4 | `head_roll` | landmarks | lateral tilt, normalized angle in [-0.5, 0.5] |

Using blendshape scores for `jaw_open` and `tongue_out` rather than geometric approximations (lip distances) is significantly more reliable. The FaceLandmarker model is internally trained to detect these states; hand-crafted geometry is sensitive to face angle, lighting, and individual anatomy.

**Alternatives considered**: MediaPipe Holistic (deprecated in 0.10+, not available with Tasks API). CLIP text-image similarity (no control over feature granularity, requires torch, ~10ms per frame overhead).

## Decision: Feature Vector Dimensionality (99 → 104)

**Choice**: Direct concatenation of pose features (99-dim) and face features (5-dim).

**Rationale**: All 5 face features are already in a comparable scale to the pose vector (~0–1 range). Direct concatenation works well with SVM RBF because `StandardScaler` is applied before fitting — it scales all 104 dimensions uniformly, so no manual re-weighting is needed.

**Alternatives considered**: Learnable feature fusion (requires MLP, incompatible with the default SVM). PCA dimensionality reduction (unnecessary overhead at 104-dim with small datasets).
