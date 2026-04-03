# CLAUDE.md — CatPose Meme Machine

> This file is the single source of truth for Claude Code.
> Read it entirely before doing anything. Act as a senior ML engineer throughout.

---

## Project Overview

| Field | Details |
|-------|---------|
| **Task** | Real-time pose detection via webcam → cat meme retrieval matching the detected pose |
| **Metric** | Qualitative (correct pose label + relevant meme displayed in real-time) |
| **Data** | Live webcam feed (no dataset — inference only at runtime) |
| **Target** | 7 pose classes: `arms_up`, `arms_wide`, `thinking`, `slouching`, `crossed_arms`, `hands_on_hips`, `neutral` |
| **Platform** | Local machine (no competition) |
| **URL** | N/A |
| **Deadline** | No deadline |
| **GPU Required** | No — MediaPipe runs efficiently on CPU |
| **External Data** | Cat memes fetched live from internet APIs |

### Pose Classes
- `arms_up` — both wrists raised above the head
- `arms_wide` — both arms extended horizontally outward
- `thinking` — one hand near the chin/cheek
- `slouching` — shoulders significantly lower than nose level, forward lean
- `crossed_arms` — wrists near the opposite elbows
- `hands_on_hips` — wrists near the hip landmarks
- `neutral` — no specific pose detected

---

## Repository Structure

```
catpose-meme-machine/
├── .claude/
│   ├── CLAUDE.md          ← this file (never committed to GitHub)
│   └── settings.json      ← {"dangerouslySkipPermissions": true}
├── .venv/                 ← virtual environment (never committed)
├── src/
│   ├── pose/
│   │   ├── detector.py    ← MediaPipe wrapper, returns 33 landmarks
│   │   └── classifier.py  ← rule-based + optional ML classifier
│   ├── meme/
│   │   ├── fetcher.py     ← abstract base + multiple backend implementations
│   │   ├── cataas.py      ← CATAAS API backend (no key needed)
│   │   ├── reddit.py      ← Reddit public JSON API backend
│   │   └── giphy.py       ← Giphy API backend (requires free key)
│   ├── display/
│   │   └── renderer.py    ← OpenCV rendering: webcam feed + pose label + meme overlay
│   └── utils/
│       ├── cache.py       ← meme image cache (TTL-based, avoids hammering APIs)
│       └── logger.py      ← structured logging
├── configs/
│   └── config.yaml        ← ALL settings: camera index, pose thresholds, meme backend, display params
├── scripts/
│   └── collect_pose_data.py  ← optional: collect labeled pose samples for ML classifier training
├── notebooks/
│   └── 01_pose_analysis.py   ← visualize landmark distributions per pose class
├── main.py                ← entry point: runs the full pipeline
├── requirements.txt       ← all dependencies pinned
├── README.md
├── TECHNICAL_CHOICES.md
├── Makefile
└── .gitignore
```

---

## Environment Setup

### ALWAYS do this first
```bash
cd catpose-meme-machine

# Create venv if it doesn't exist
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate
# Activate (Windows)
# .venv\Scripts\activate

# Verify correct interpreter
which python  # must point to .venv/bin/python

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ Every pip install and every python command must run inside the venv.
> Never install packages globally.

---

## Technical Strategy

### Pose Detection
Use **MediaPipe Pose** (mediapipe >= 0.10). It exposes 33 body landmarks, each with `x`, `y`, `z` (normalized) and `visibility`. Only use landmarks with `visibility > 0.5` in the classifier.

Key landmark indices (memorize these):
| Index | Name |
|-------|------|
| 0 | NOSE |
| 11 | LEFT_SHOULDER |
| 12 | RIGHT_SHOULDER |
| 13 | LEFT_ELBOW |
| 14 | RIGHT_ELBOW |
| 15 | LEFT_WRIST |
| 16 | RIGHT_WRIST |
| 23 | LEFT_HIP |
| 24 | RIGHT_HIP |
| 25 | LEFT_KNEE |
| 26 | RIGHT_KNEE |

### Pose Classification — Two Approaches

**Phase 1 (Baseline — implemented first): Rule-based classifier**
Geometric rules on normalized landmark coordinates. Fast, zero training data needed, interpretable.

Example rules:
- `arms_up`: `wrist_y < shoulder_y - 0.15` for both wrists (y is inverted, 0=top)
- `arms_wide`: `|wrist_x - shoulder_x| > 0.25` for both wrists and `|wrist_y - shoulder_y| < 0.1`
- `thinking`: one wrist within `0.15` distance of the chin/cheek area
- `slouching`: `shoulder_y > nose_y + 0.35` (shoulders far below nose in frame)
- `crossed_arms`: left wrist x > right shoulder x AND right wrist x < left shoulder x
- `hands_on_hips`: both wrists within `0.1` of their same-side hip landmark
- `neutral`: fallback if no rule matches

**Phase 2 (Optional ML upgrade): scikit-learn classifier**
Collect ~50-100 samples per class with `scripts/collect_pose_data.py`, then train an MLP or SVM on the 33×3 flattened landmark vector. Serialize with joblib. Switch via `config.yaml: classifier: type: ml`.

### Meme Retrieval

Three backends, selectable in `config.yaml: meme: backend`:

| Backend | Key needed | Notes |
|---------|-----------|-------|
| `cataas` | No | https://cataas.com/cat/{tag} — default, always works |
| `reddit` | No | r/catmemes public JSON — rate limited, add delay |
| `giphy` | Yes (free) | Best tag matching, requires `GIPHY_API_KEY` env var |

Pose → tag mapping (in `configs/config.yaml`):
```yaml
pose_to_tags:
  arms_up: ["happy cat", "excited cat", "jumping cat"]
  arms_wide: ["big cat", "dramatic cat", "surprised cat"]
  thinking: ["thinking cat", "serious cat", "smart cat"]
  slouching: ["sleepy cat", "lazy cat", "tired cat"]
  crossed_arms: ["grumpy cat", "mad cat", "annoyed cat"]
  hands_on_hips: ["sassy cat", "boss cat", "confident cat"]
  neutral: ["cat", "cute cat", "cat meme"]
```

### Caching Strategy
Cache fetched meme images in memory (dict: `pose_label → PIL.Image`). Refresh TTL: 30 seconds (configurable). This prevents hammering the API on every frame while keeping memes fresh enough.

### Display Layout (OpenCV)
Split the window into two panels side by side:
- **Left panel**: webcam feed with MediaPipe skeleton overlay + pose label text
- **Right panel**: current cat meme (resized to fit panel, maintaining aspect ratio)
- **Bottom bar**: pose confidence / active rule name (debug mode, toggleable with `d` key)

Target FPS: ≥ 20fps. If below threshold, reduce meme panel resolution or skip meme fetch frames.

### Key Technical Choices
- MediaPipe over OpenPose/MMPose: runs on CPU, no CUDA needed, single pip install
- CATAAS as default backend: zero config, works immediately, good for demo
- Rule-based classifier first: instant feedback loop, no data collection required
- TTL cache for memes: avoids rate limits, keeps real-time feel
- OpenCV for display: already a dependency of MediaPipe, no extra install

---

## Workflow — Follow This Order

1. **Setup** — create venv, install requirements
2. **Smoke test camera** — verify webcam opens with OpenCV (`cv2.VideoCapture(0)`)
3. **Pose detection** — implement `src/pose/detector.py`, display landmarks on webcam feed
4. **Rule classifier** — implement `src/pose/classifier.py` with all 7 rules
5. **Meme fetcher** — implement CATAAS backend first (`src/meme/cataas.py`)
6. **Renderer** — implement `src/display/renderer.py` with dual-panel layout
7. **Wire up** — connect everything in `main.py`
8. **Run** — `python main.py` — verify end-to-end
9. **Tune** — adjust thresholds in `config.yaml` until all poses trigger correctly
10. **Optional** — add Reddit/Giphy backends, then ML classifier upgrade

---

## Code Quality Standards

Act as a senior ML engineer. Every file must follow these standards:

- **Type hints** on all functions and class methods
- **Docstrings** on all classes and non-trivial functions (Google style)
- **PEP 8** — enforced via ruff
- **No hardcoded values** — everything in `configs/config.yaml`
- **Modular code** — one responsibility per file, no 500-line monoliths
- **Requirements** — pin every dependency with exact version in `requirements.txt`
- **Graceful error handling** — if meme API fails, display a fallback placeholder image (never crash)
- **No global state** — pass config explicitly, use dependency injection

---

## Config Convention

All settings live in `configs/config.yaml`. `main.py` loads it with PyYAML and passes the config dict down.

Example structure:
```yaml
camera:
  index: 0
  width: 640
  height: 480
  fps: 30

pose:
  classifier: rule_based   # or: ml
  ml_model_path: models/pose_classifier.joblib
  visibility_threshold: 0.5
  smoothing_window: 5       # frames to smooth pose label over

meme:
  backend: cataas           # or: reddit, giphy
  cache_ttl_seconds: 30
  request_timeout_seconds: 3
  giphy_api_key: ""         # override with GIPHY_API_KEY env var

display:
  window_title: "CatPose Meme Machine"
  meme_panel_width: 400
  debug_mode: false
  font_scale: 1.0

pose_to_tags:
  arms_up: ["happy cat", "excited cat", "jumping cat"]
  arms_wide: ["big cat", "dramatic cat", "surprised cat"]
  thinking: ["thinking cat", "serious cat", "smart cat"]
  slouching: ["sleepy cat", "lazy cat", "tired cat"]
  crossed_arms: ["grumpy cat", "mad cat", "annoyed cat"]
  hands_on_hips: ["sassy cat", "boss cat", "confident cat"]
  neutral: ["cat", "cute cat", "cat meme"]
```

---

## Git Conventions

- Branch: `main`
- Commits: `feat: ...` | `fix: ...` | `exp: ...` | `docs: ...`
- Never commit: `.venv/`, `.claude/`, API keys, `models/`
- Tag stable versions: `git tag v1.0`

### .gitignore must include:
```
.venv/
.claude/
*.pyc
__pycache__/
.env
models/
*.joblib
```

---

## Runtime Controls (keyboard shortcuts in OpenCV window)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `d` | Toggle debug overlay |
| `r` | Force refresh meme (ignores cache TTL) |
| `s` | Save screenshot to `screenshots/` |
| `1/2/3` | Switch meme backend (cataas/reddit/giphy) at runtime |

---

## Files to Create

Claude Code must create ALL of the following before considering setup complete:

- [ ] `requirements.txt` — pinned deps: mediapipe, opencv-python, requests, Pillow, PyYAML, numpy, ruff
- [ ] `configs/config.yaml` — full config as shown above
- [ ] `src/__init__.py`
- [ ] `src/pose/__init__.py`
- [ ] `src/pose/detector.py` — MediaPipe wrapper class
- [ ] `src/pose/classifier.py` — RuleBasedClassifier + optional MLClassifier
- [ ] `src/meme/__init__.py`
- [ ] `src/meme/fetcher.py` — abstract base class `MemeFetcher`
- [ ] `src/meme/cataas.py` — CATAAS implementation
- [ ] `src/meme/reddit.py` — Reddit implementation
- [ ] `src/meme/giphy.py` — Giphy implementation
- [ ] `src/display/__init__.py`
- [ ] `src/display/renderer.py` — OpenCV dual-panel renderer
- [ ] `src/utils/__init__.py`
- [ ] `src/utils/cache.py` — TTL cache class
- [ ] `src/utils/logger.py` — logging setup
- [ ] `main.py` — entry point, wires everything together
- [ ] `scripts/collect_pose_data.py` — pose data collection for ML upgrade
- [ ] `notebooks/01_pose_analysis.py` — landmark visualization
- [ ] `README.md`
- [ ] `TECHNICAL_CHOICES.md`
- [ ] `Makefile`
- [ ] `.gitignore`
- [ ] `screenshots/.gitkeep`
- [ ] `models/.gitkeep`

---

## Code Review Checklist

After writing all files, Claude Code must verify:

- [ ] `python main.py` runs without errors
- [ ] Webcam opens correctly (test with `cv2.VideoCapture(config['camera']['index'])`)
- [ ] All 7 pose classes trigger correctly during manual testing
- [ ] Meme fetches successfully from CATAAS backend
- [ ] App does NOT crash if meme API returns an error (fallback image shown)
- [ ] All config values are read from `configs/config.yaml`, none hardcoded
- [ ] All functions have type hints and docstrings
- [ ] No import errors (all packages in requirements.txt)
- [ ] Keyboard shortcuts `q`, `d`, `r` work correctly
- [ ] FPS stays ≥ 15fps on CPU (measure with `cv2.getTickFrequency()`)

---

## Future Extensions (do NOT implement now, just keep in mind)

- **ML classifier upgrade**: train SVM/MLP on collected pose samples (Phase 2)
- **Sequence poses**: detect motion patterns over time (e.g., waving = transition arms_wide → arms_up)
- **Multi-person**: extend to multiple people in frame
- **Web UI**: replace OpenCV window with a Flask/FastAPI + WebSocket frontend
- **Sound effects**: play a cat sound when pose changes
