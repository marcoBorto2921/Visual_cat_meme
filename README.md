# CatPose

Real-time webcam pose classification → mostra la foto del gatto che stai imitando.

Scegli un set di foto di gatti con pose divertenti, imita le pose davanti alla webcam, e il sistema impara a riconoscerti in tempo reale.

## Come funziona

1. **Scegli le foto** — metti in `assets/cats/` una foto per ogni posa che vuoi imitare. Il nome del file è il label della classe (es. `tongue_cat.jpg`, `grumpy_cat.jpg`).
2. **Raccogli il training data** — per ogni gatto, imiti la posa davanti alla webcam e premi SPAZIO per campionare i tuoi landmark corporei.
3. **Addestra il classificatore** — un SVM impara esattamente come si muovono i tuoi landmark quando imiti ciascun gatto.
4. **Gioca** — in real-time, MediaPipe rileva i tuoi landmark → il classificatore predice quale gatto stai imitando → compare la foto.

**Nessun CLIP, nessun modello generico.** Il sistema impara le *tue* pose specifiche — è personale e molto più preciso.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Utilizzo

### 1. Prepara le foto dei gatti

```
assets/cats/tongue_cat.jpg
assets/cats/grumpy_cat.jpg
assets/cats/surprised_cat.jpg
```

Usa pose ben distinguibili tra loro: braccia alzate, braccia incrociate, mano sul viso, ecc.

### 2. Raccogli i campioni

```bash
python scripts/collect_samples.py
```

Per ogni gatto:
- La foto appare sul pannello destro
- Imita la posa davanti alla webcam
- Premi **SPAZIO** per campionare 30 frame di landmark
- Premi **N** per passare al gatto successivo

### 3. Addestra il classificatore

```bash
python scripts/train_classifier.py
```

Stampa accuracy e classification report. Se l'accuracy è bassa (< 70%), raccogli più campioni.

### 4. Avvia

```bash
python main.py
```

## Keyboard shortcuts

| Tasto | Azione |
|-------|--------|
| `q` | Quit |
| `d` | Toggle debug overlay (top-3 predizioni con confidence) |
| `r` | Reset smoothing window |
| `s` | Salva screenshot in `screenshots/` |

## Configurazione

Tutti i parametri in `configs/config.yaml`: indice webcam, soglia confidence, numero sample per posa, modello (svm/mlp), ecc.

## Struttura

```
assets/cats/        ← le tue foto di gatti (una per classe)
data/samples.csv    ← landmark campionati (gitignored)
models/             ← classifier.pkl, label_encoder.pkl (gitignored)
src/pose/           ← MediaPipe wrapper
src/classifier/     ← feature extraction, training, inference
src/display/        ← OpenCV dual-panel renderer
scripts/            ← collect_samples.py, train_classifier.py
```
