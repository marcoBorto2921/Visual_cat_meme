# CLAUDE.md — CatPose CLIP

> Questo file è la single source of truth per Claude Code.
> Leggilo interamente prima di fare qualsiasi cosa. Agisci come senior ML engineer.
> Dopo ogni modifica rilevante: `git add -A && git commit -m "feat: <descrizione breve>"`.
> Alla fine del setup: `git push`.

---

## Project Overview

| Field | Details |
|-------|---------|
| **Task** | Real-time pose detection via webcam → image retrieval: mostra la foto di gatto più "simile" alla posa rilevata, usando embedding CLIP |
| **Metric** | Qualitativa — cosine similarity tra embedding posa e embedding foto gatto |
| **Data** | Live webcam feed + cartella locale `assets/cats/` con foto di gatti scelte dall'utente |
| **Target** | Nessun label esplicito — retrieval puro via nearest neighbor nello spazio CLIP |
| **Platform** | Local machine (nessuna competition) |
| **URL** | N/A |
| **Deadline** | Nessuna |
| **GPU Required** | No — CLIP (ViT-B/32) gira su CPU per l'embedding index; l'inferenza real-time è leggera |
| **External Data** | N/A |

---

## Idea Centrale

L'utente inserisce nella cartella `assets/cats/` le foto di gatti che preferisce (meme, foto, qualsiasi cosa).
A runtime, il sistema:
1. Rileva i landmark corporei dell'utente via **MediaPipe Pose**
2. Genera una **descrizione testuale** della posa rilevata (es. `"a person with arms raised above the head"`)
3. Calcola l'**embedding CLIP** di quella descrizione
4. Trova la **foto di gatto più vicina** nel CLIP embedding space (cosine similarity)
5. Mostra la foto sul pannello destro della finestra OpenCV

Non c'è training — è **zero-shot retrieval**. L'utente può aggiungere/rimuovere foto da `assets/cats/` e il sistema si aggiorna automaticamente al prossimo avvio (o premendo `i` per re-indicizzare a runtime).

---

## Repository Structure

```
catpose-clip/
├── .claude/
│   ├── CLAUDE.md          ← questo file (mai committato)
│   └── settings.json      ← {"dangerouslySkipPermissions": true}
├── .venv/                 ← virtual environment (mai committato)
├── assets/
│   └── cats/              ← l'utente inserisce qui le sue foto di gatti (JPG/PNG)
│       └── .gitkeep       ← cartella tracciata ma vuota nel repo
├── src/
│   ├── pose/
│   │   ├── __init__.py
│   │   ├── detector.py    ← MediaPipe wrapper, ritorna 33 landmark
│   │   └── describer.py   ← converte landmark → stringa testuale per CLIP
│   ├── clip_index/
│   │   ├── __init__.py
│   │   ├── indexer.py     ← carica foto da assets/cats/, calcola embedding, salva index
│   │   └── retriever.py   ← dato un query embedding, ritorna la foto più simile
│   ├── display/
│   │   ├── __init__.py
│   │   └── renderer.py    ← OpenCV dual-panel: webcam sx, gatto dx
│   └── utils/
│       ├── __init__.py
│       └── logger.py      ← structured logging
├── configs/
│   └── config.yaml        ← tutti i parametri configurabili
├── scripts/
│   └── build_index.py     ← script standalone per (ri)costruire l'index CLIP
├── main.py                ← entry point
├── requirements.txt
├── README.md
├── TECHNICAL_CHOICES.md
├── Makefile
└── .gitignore
```

---

## Environment Setup

### ALWAYS do this first
```bash
cd catpose-clip

python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate
# Activate (Windows)
# .venv\Scripts\activate

# Verifica che python punti al venv
which python  # deve essere .venv/bin/python

pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ Ogni pip install e ogni python command devono girare dentro il venv. Mai installare globalmente.

---

## Technical Strategy

### Pose Detection
**MediaPipe Pose** (mediapipe >= 0.10). 33 landmark, x/y/z normalizzati + visibility. Usa solo landmark con `visibility > 0.5`.

Landmark chiave:
| Index | Nome |
|-------|------|
| 0 | NOSE |
| 11 | LEFT_SHOULDER |
| 12 | RIGHT_SHOULDER |
| 15 | LEFT_WRIST |
| 16 | RIGHT_WRIST |
| 23 | LEFT_HIP |
| 24 | RIGHT_HIP |

### Pose → Testo (describer.py)

Converti i landmark in una stringa leggibile da CLIP. Usa regole geometriche semplici per scegliere la descrizione più accurata. Esempi:

```python
POSE_DESCRIPTIONS = {
    "arms_up":      "a person with both arms raised above the head, excited",
    "arms_wide":    "a person with arms stretched wide open to the sides",
    "thinking":     "a person touching their face with one hand, thinking",
    "slouching":    "a person slouching forward with drooping shoulders, tired",
    "crossed_arms": "a person with arms crossed on the chest, grumpy",
    "hands_on_hips":"a person with hands on hips, confident and sassy",
    "neutral":      "a person standing normally, calm and relaxed",
}
```

La regola di classificazione geometrica rimane semplice (come nel progetto originale) — serve solo per scegliere quale stringa mandare a CLIP, non deve essere perfetta perché CLIP gestisce la vaghezza semantica.

### CLIP Embedding Index

Usa **openai/clip-vit-base-patch32** via `transformers` (HuggingFace). Non serve PyTorch GPU.

**Build index** (fatto una volta, poi cached):
1. Carica tutte le immagini da `assets/cats/`
2. Per ognuna, calcola l'embedding CLIP immagine → vettore float32 di dim 512
3. Normalizza L2
4. Salva in `clip_index/embeddings.npy` e `clip_index/filenames.json`

**Retrieval** (real-time, ogni frame o ogni N frame):
1. Prendi la descrizione testuale della posa corrente
2. Calcola embedding CLIP del testo → vettore float32 dim 512, normalizzato L2
3. Cosine similarity = dot product (perché già normalizzati)
4. Ritorna il filename con similarity più alta

**Performance**: l'embedding del testo su CPU è ~5-10ms. Nessun impatto sull'FPS.

### Display Layout (OpenCV)
- **Pannello sinistro**: feed webcam con skeleton MediaPipe + label posa corrente + similarity score
- **Pannello destro**: foto del gatto più simile, resizata mantenendo aspect ratio
- **Debug overlay** (tasto `d`): mostra top-3 match con similarity score

### Caching
- Il testo embedding viene ricalcolato solo quando cambia la posa (non ogni frame)
- La foto risultante viene mantenuta finché la posa non cambia o non si preme `r`
- L'index viene costruito una volta e caricato da file ad ogni avvio

---

## Workflow — Segui Questo Ordine

1. **Setup** — crea venv, installa requirements
2. **Crea struttura** — tutte le cartelle e file elencati in "Files to Create"
3. **Implementa detector.py** — MediaPipe wrapper
4. **Implementa describer.py** — regole geometriche → stringa testuale
5. **Implementa indexer.py** — carica foto, calcola embedding CLIP, salva index
6. **Implementa retriever.py** — cosine similarity, ritorna path foto
7. **Implementa renderer.py** — dual-panel OpenCV
8. **Implementa main.py** — wiring completo
9. **Implementa scripts/build_index.py** — script standalone per build index
10. **Smoke test** — con 2-3 foto di placeholder in `assets/cats/`, verifica che il pipeline giri
11. **Commit e push** — `git add -A && git commit -m "feat: initial working pipeline" && git push`

> ⚠️ La cartella `assets/cats/` deve essere creata e tracciata con `.gitkeep` ma VUOTA — l'utente la riempirà con le sue foto. Non aggiungere foto placeholder nel repo.

---

## Config (configs/config.yaml)

```yaml
camera:
  index: 0
  width: 640
  height: 480
  fps: 30

pose:
  visibility_threshold: 0.5
  smoothing_window: 5        # frame di smoothing per evitare flickering del label

clip:
  model_name: "openai/clip-vit-base-patch32"
  index_dir: "clip_index"    # dove salvare embeddings.npy e filenames.json
  cats_dir: "assets/cats"    # dove l'utente mette le sue foto
  top_k: 3                   # quante foto mostrare in debug mode

display:
  window_title: "CatPose CLIP"
  cat_panel_width: 480
  debug_mode: false
  font_scale: 1.0
  similarity_threshold: 0.15  # sotto questa soglia mostra "no match" invece di una foto

pose_descriptions:
  arms_up:       "a person with both arms raised above the head, excited"
  arms_wide:     "a person with arms stretched wide open to the sides"
  thinking:      "a person touching their face with one hand, thinking"
  slouching:     "a person slouching forward with drooping shoulders, tired"
  crossed_arms:  "a person with arms crossed on the chest, grumpy"
  hands_on_hips: "a person with hands on hips, confident and sassy"
  neutral:       "a person standing normally, calm and relaxed"
```

---

## Keyboard Shortcuts (OpenCV window)

| Tasto | Azione |
|-------|--------|
| `q` | Quit |
| `d` | Toggle debug overlay (mostra top-3 match con score) |
| `r` | Forza re-retrieval (ignora cache) |
| `i` | Re-indicizza `assets/cats/` a runtime (utile dopo aver aggiunto foto) |
| `s` | Salva screenshot in `screenshots/` |

---

## Code Quality Standards

- **Type hints** su tutte le funzioni e metodi
- **Docstring Google-style** su tutte le classi e funzioni non-banali
- **PEP 8** — enforced via ruff
- **Zero valori hardcoded** — tutto in `configs/config.yaml`
- **Un file, una responsabilità** — no monoliti da 500 righe
- **Requirements** — ogni dipendenza pinnata con versione esatta
- **Graceful error handling** — se `assets/cats/` è vuota, mostra un messaggio chiaro e continua senza crash
- **Nessuno stato globale** — passa config esplicitamente

---

## Git Conventions

- Branch: `main`
- Commit dopo ogni step significativo: `feat: ...` | `fix: ...` | `docs: ...`
- Push finale dopo il setup completo
- Mai committare: `.venv/`, `.claude/`, `clip_index/embeddings.npy`, `clip_index/filenames.json`, foto in `assets/cats/`

### .gitignore deve includere:
```
.venv/
.claude/
__pycache__/
*.pyc
.env
clip_index/embeddings.npy
clip_index/filenames.json
assets/cats/*
!assets/cats/.gitkeep
screenshots/
```

---

## Files to Create

Claude Code deve creare TUTTI i seguenti file prima di considerare il setup completo:

- [ ] `requirements.txt` — mediapipe, opencv-python, transformers, torch (cpu), Pillow, PyYAML, numpy, ruff
- [ ] `configs/config.yaml` — come sopra
- [ ] `assets/cats/.gitkeep` — cartella vuota tracciata
- [ ] `clip_index/.gitkeep` — cartella vuota per l'index
- [ ] `screenshots/.gitkeep`
- [ ] `src/__init__.py`
- [ ] `src/pose/__init__.py`
- [ ] `src/pose/detector.py` — MediaPipe wrapper
- [ ] `src/pose/describer.py` — landmark → stringa testuale
- [ ] `src/clip_index/__init__.py`
- [ ] `src/clip_index/indexer.py` — build e salva embedding index
- [ ] `src/clip_index/retriever.py` — cosine similarity retrieval
- [ ] `src/display/__init__.py`
- [ ] `src/display/renderer.py` — OpenCV dual-panel
- [ ] `src/utils/__init__.py`
- [ ] `src/utils/logger.py` — logging setup
- [ ] `scripts/build_index.py` — standalone script per (ri)costruire index
- [ ] `main.py` — entry point, wiring completo
- [ ] `README.md`
- [ ] `TECHNICAL_CHOICES.md`
- [ ] `Makefile`
- [ ] `.gitignore`

---

## Code Review Checklist

Dopo aver scritto tutti i file, Claude Code deve verificare:

- [ ] `python scripts/build_index.py` gira senza errori (anche con cartella vuota — deve stampare un warning chiaro)
- [ ] `python main.py` si avvia e apre la webcam
- [ ] Con almeno 1 foto in `assets/cats/`, il retrieval funziona e mostra la foto
- [ ] Cambiando posa, la foto cambia
- [ ] Se `assets/cats/` è vuota, il programma non crasha — mostra un messaggio e il solo feed webcam
- [ ] Tutti i valori letti da `configs/config.yaml`, nessuno hardcoded
- [ ] Tutte le funzioni hanno type hints e docstring
- [ ] Nessun import error (tutti i package in requirements.txt)
- [ ] I tasti `q`, `d`, `r`, `i`, `s` funzionano
- [ ] FPS ≥ 15 su CPU (misura con `cv2.getTickFrequency()`)
- [ ] `git log` mostra commit per ogni step significativo
- [ ] `git push` completato con successo

---

## Future Extensions (non implementare ora)

- **ML classifier personalizzato**: raccogliere pose sample con `scripts/collect_pose_data.py` e addestrare SVM/MLP per migliorare la classificazione geometrica
- **Top-K display**: mostrare le top-3 foto più simili in una griglia invece di una sola
- **Similarity heatmap**: visualizzare quanto ogni foto si avvicina alla posa corrente
- **Aggiornamento index live**: watch sulla cartella `assets/cats/` con watchdog per re-indicizzare automaticamente quando si aggiungono foto
