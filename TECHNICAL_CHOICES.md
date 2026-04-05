# Technical Choices

## Pose Detection — MediaPipe Tasks API

MediaPipe 0.10+ replaced the legacy `solutions` API with the Tasks API (`PoseLandmarker`). The Tasks API is the only supported path on Windows with mediapipe 0.10.33. We use `RunningMode.IMAGE` (per-frame, no async callbacks) for simplicity and predictable latency.

The detector returns all 33 normalized landmarks (x, y, z, visibility). Filtering by visibility is handled downstream in `features.py`, so the detector stays decoupled from the feature engineering strategy.

## Classification — SVM on Landmark Features (not CLIP)

Instead of zero-shot retrieval via CLIP, we train a supervised classifier directly on the user's own pose landmarks. This has several advantages:

- **Personalized**: the model learns how *your specific body* moves when imitating each cat pose — not a generic text-image similarity.
- **Fast**: SVM inference on a 99-dim vector is microseconds on CPU, vs. ~10ms for a CLIP text embedding.
- **No large model downloads**: no `torch`, no `transformers`, no multi-GB weights. scikit-learn only.
- **Accurate with few samples**: SVM with RBF kernel is robust with 20–50 samples per class.

## Feature Engineering — Normalized Landmark Vector

Raw landmark coordinates depend on where the user stands relative to the camera (translation and scale). We normalize them:

1. **Center** on the hip midpoint (stable reference point).
2. **Scale** by the shoulder-to-shoulder distance (invariant to camera distance).
3. **Zero-pad** invisible landmarks (visibility below threshold).

Result: a 99-dimensional float32 vector (33 landmarks × x, y, z) that represents the *shape* of the pose, not its absolute position.

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
