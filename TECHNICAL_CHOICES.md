# Technical Choices

## Pose Detection: MediaPipe Pose

- Runs on CPU with no CUDA requirement
- Single pip install, 33 high-quality landmarks with visibility scores
- Smooth landmark tracking across frames built-in
- Alternative considered: OpenPose (requires CUDA), MMPose (heavier setup)

## Classifier: Rule-Based (Phase 1)

- Zero training data required — instant feedback loop
- Interpretable: each pose maps to 1-2 geometric rules on normalized coordinates
- Majority-vote smoothing over last N frames eliminates flicker
- Upgrade path: scikit-learn MLP/SVM on collected samples via `scripts/collect_pose_data.py`

## Meme Backend: CATAAS (default)

- Zero API key required — works out of the box for demos
- Accepts text overlay queries for pose-relevant tags
- Reddit and Giphy backends available as drop-in replacements via config

## Display: OpenCV

- Already a MediaPipe dependency — no extra install
- Dual-panel layout: webcam skeleton (left) + meme (right)
- Debug bar toggle shows active rule and FPS without cluttering main view
- Target FPS: ≥20 on CPU

## Caching: TTL In-Memory Dict

- Prevents API hammering on every frame (30s default TTL)
- Per-pose-label keying: different meme per pose class
- `r` key forces cache invalidation for manual refresh
- `cache.clear()` on backend switch avoids stale images from old backend

## Config: Single YAML File

- All magic numbers and settings in `configs/config.yaml`
- No hardcoded values in source files
- Pose-to-tag mapping editable without touching code
