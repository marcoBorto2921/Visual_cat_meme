# Technical Choices

## Pose Detection — MediaPipe Tasks API

MediaPipe 0.10+ replaced the legacy `solutions` API with the Tasks API (`PoseLandmarker`). The Tasks API is the only supported path on Windows with mediapipe 0.10.33. We use `RunningMode.IMAGE` (per-frame, no async callbacks) for simplicity and predictable latency.

## Image Retrieval — CLIP ViT-B/32 via HuggingFace Transformers

Zero-shot retrieval: no labelled data, no training required. A natural-language description of the detected pose is encoded with CLIP text encoder. Cat photos are pre-indexed with the CLIP image encoder. At runtime, cosine similarity between text and image embeddings selects the best match.

CLIP ViT-B/32 produces 512-dimensional embeddings — fast to compute and compare even on CPU.

## No GPU required

Text embedding on CPU: ~5-10 ms. Image indexing is done once offline. Real-time loop performance is not bottlenecked by CLIP.

## Embedding caching

Text embeddings are cached in `CLIPRetriever` and recomputed only when the pose label changes. This keeps the main loop fast even on slow hardware.

## Pose smoothing

A sliding window (default: 5 frames) takes the mode of recent pose labels to suppress flickering from noisy landmark estimates.

## Index persistence

The CLIP image index (`clip_index/embeddings.npy` + `clip_index/filenames.json`) is built once and loaded from disk at startup. Pressing `i` at runtime triggers a rebuild without restarting.
