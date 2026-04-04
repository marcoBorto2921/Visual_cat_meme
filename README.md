# CatPose CLIP

Real-time webcam pose detection → CLIP-based zero-shot cat photo retrieval.

Strike a pose, get the matching cat meme.

## How it works

1. **MediaPipe Pose** detects your body landmarks via webcam
2. Geometric rules classify the pose (arms up, thinking, slouching, etc.)
3. A text description is encoded with **CLIP** (ViT-B/32)
4. The most similar cat photo from `assets/cats/` is shown (cosine similarity)

No training — pure zero-shot retrieval.

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

1. Drop JPG/PNG cat photos into `assets/cats/`
2. Build the CLIP index:
   ```bash
   python scripts/build_index.py
   ```
3. Run:
   ```bash
   python main.py
   ```

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit |
| `d` | Toggle debug overlay (top-3 matches with scores) |
| `r` | Force re-retrieval (ignore cache) |
| `i` | Re-index `assets/cats/` at runtime |
| `s` | Save screenshot to `screenshots/` |

## Configuration

All parameters in `configs/config.yaml` — camera index, similarity threshold, pose descriptions, CLIP model, etc.
