# CatPose Meme Machine

Real-time webcam pose detection → matching cat meme display.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit |
| `d` | Toggle debug overlay |
| `r` | Force refresh meme |
| `s` | Save screenshot |
| `1/2/3` | Switch backend (cataas/reddit/giphy) |

## Pose Classes

| Pose | Description |
|------|-------------|
| `arms_up` | Both wrists raised above head |
| `arms_wide` | Both arms extended horizontally |
| `thinking` | One hand near chin/cheek |
| `slouching` | Shoulders significantly below nose |
| `crossed_arms` | Wrists near opposite elbows |
| `hands_on_hips` | Wrists near hip landmarks |
| `neutral` | No specific pose detected |

## Config

Edit `configs/config.yaml` to adjust camera index, pose thresholds, meme backend, and display options.

## Meme Backends

| Backend | Key needed | Config value |
|---------|-----------|-------------|
| CATAAS | No | `backend: cataas` |
| Reddit | No | `backend: reddit` |
| Giphy | Yes (free) | `backend: giphy` + `GIPHY_API_KEY` env var |

## Optional: Collect Pose Data (ML Upgrade)

```bash
python scripts/collect_pose_data.py --pose arms_up --samples 100
python notebooks/01_pose_analysis.py --input data/poses.csv
```
