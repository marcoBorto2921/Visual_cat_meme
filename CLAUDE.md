# CLAUDE.md — CatPose Classifier

> Questo file è la single source of truth per Claude Code.
> Leggilo interamente prima di fare qualsiasi cosa. Agisci come senior ML engineer.
> Dopo ogni modifica rilevante: `git add -A && git commit -m "feat: <descrizione breve>"`.
> Alla fine di ogni sessione: `git push`. Non chiedere conferme — esegui direttamente.

---

## Project Overview

| Field | Details |
|-------|---------|
| **Task** | Real-time pose classification via webcam → mostra la foto di gatto che l'utente sta imitando |
| **Metric** | Qualitativa — accuracy su validation set + confidence score a runtime |
| **Data** | Live webcam feed + cartella `assets/cats/` con foto di gatti scelte dall'utente (una per classe) |
| **Target** | Classificazione multiclasse: ogni classe = una foto di gatto specifica |
| **Platform** | Local machine (nessuna competition) |
| **URL** | N/A |
| **Deadline** | Nessuna |
| **GPU Required** | No — SVM/MLP su landmark 2D, tutto su CPU |
| **External Data** | N/A |

---

## Idea Centrale

L'utente sceglie un set di foto di gatti — ognuna con una posa riconoscibile e divertente (lingua fuori, zampe alzate, occhi sgranati, ecc.). Per ogni gatto, l'utente imita la posa davanti alla webcam e il sistema raccoglie i suoi landmark corporei come training data.

A runtime, MediaPipe rileva i landmark → un classificatore addestrato predice quale gatto stai imitando → compare la foto di quel gatto.

**Nessun CLIP, nessun retrieval generico.** Il modello impara esattamente *come si muovono i tuoi landmark* quando imiti ciascun gatto. È personale, preciso, e molto più divertente.

---

## Flusso Completo

```
1. Prepara le foto → assets/cats/<label>.jpg  (es. tongue_cat.jpg, grumpy_cat.jpg)
2. Raccogli i campioni → python scripts/collect_samples.py
   - mostra ogni foto a schermo
   - l'utente fa la posa
   - premi SPAZIO per campionare N frame di landmark
   - ripeti per ogni gatto
3. Addestra il classificatore → python scripts/train_classifier.py
   - legge data/samples.csv
   - addestra SVM (default) o MLP
   - salva in models/classifier.pkl + models/label_encoder.pkl
   - stampa accuracy su validation set
4. Esegui il programma → python main.py
   - MediaPipe rileva landmark in real-time
   - il classificatore predice il label
   - compare la foto del gatto corrispondente
```

---

## Repository Structure

```
catpose-classifier/
├── .claude/
│   ├── CLAUDE.md              ← questo file (mai committato)
│   └── settings.json          ← {"dangerouslySkipPermissions": true}
├── .venv/                     ← virtual environment (mai committato)
├── assets/
│   └── cats/                  ← l'utente mette qui le sue foto di gatti
│       └── .gitkeep           ← cartella tracciata ma vuota nel repo
├── data/
│   ├── samples.csv            ← landmark campionati (gitignored)
│   └── .gitkeep
├── models/
│   ├── classifier.pkl         ← modello addestrato (gitignored)
│   ├── label_encoder.pkl      ← encoder label (gitignored)
│   └── .gitkeep
├── screenshots/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── pose/
│   │   ├── __init__.py
│   │   └── detector.py        ← MediaPipe wrapper, ritorna 33 landmark
│   ├── classifier/
│   │   ├── __init__.py
│   │   ├── features.py        ← landmark → feature vector normalizzato
│   │   ├── trainer.py         ← addestra SVM/MLP, salva modello
│   │   └── predictor.py       ← carica modello, predice label + confidence
│   └── display/
│       ├── __init__.py
│       └── renderer.py        ← OpenCV dual-panel: webcam sx, gatto dx
├── scripts/
│   ├── collect_samples.py     ← UI per raccogliere training data
│   └── train_classifier.py   ← entry point training
├── configs/
│   └── config.yaml            ← tutti i parametri configurabili
├── main.py                    ← entry point real-time
├── requirements.txt
├── README.md
├── TECHNICAL_CHOICES.md
├── Makefile
└── .gitignore
```

---

## Environment Setup

### SEMPRE fare questo prima
```bash
cd catpose-classifier

python -m venv .venv

# Attiva (Linux/macOS)
source .venv/bin/activate
# Attiva (Windows)
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
**MediaPipe Pose Tasks API** (mediapipe >= 0.10). 33 landmark, x/y/z normalizzati + visibility. Usa `RunningMode.IMAGE` per frame-by-frame processing sincrono.

Landmark chiave usati per le feature:
| Index | Nome |
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

### Feature Engineering (features.py)

I landmark grezzi (x, y, z, visibility) non sono direttamente comparabili tra frame diversi perché dipendono dalla posizione dell'utente nello spazio. Vanno normalizzati:

1. **Filtra** solo landmark con `visibility > threshold`
2. **Centra** rispetto al centro dei fianchi (punto di riferimento stabile)
3. **Scala** dividendo per la distanza shoulder-to-shoulder (invariante alla distanza dalla camera)
4. **Concatena** x, y, z dei landmark filtrati → vettore float32

Il vettore risultante è invariante a traslazione e scala — dipende solo dalla *forma* della posa.

**Dimensione feature vector**: 33 landmark × 3 coordinate = 99 feature (padding con 0 per landmark non visibili).

### Classifier (trainer.py)

**Default: SVM con kernel RBF** via scikit-learn.
- Robusto con pochi sample (20-50 per classe)
- Nessun iperparametro critico da tunare
- `predict_proba` disponibile via `probability=True` → confidence score

**Alternativa: MLP** (MLPClassifier) se l'utente ha >100 sample per classe.
- Selezionabile via config: `classifier.model: "svm"` o `"mlp"`

**Training split**: 80% train / 20% validation, stratificato per classe.
Stampa classification report completo (precision, recall, F1 per classe).

### Smoothing a runtime

Sliding window di N frame (default: 7). La predizione stabile è la moda della window. Cambia foto solo quando il label stabile cambia — evita flickering.

### Confidence threshold

Se la confidence massima è sotto `classifier.confidence_threshold` (default: 0.4), mostra "?" invece di una foto — l'utente non sta imitando nessun gatto riconoscibile.

### Display Layout (renderer.py)
- **Pannello sinistro**: feed webcam con skeleton MediaPipe + label predetto + confidence score + FPS
- **Pannello destro**: foto del gatto predetto, resizata mantenendo aspect ratio
- **Debug overlay** (tasto `d`): mostra top-3 predizioni con confidence

---

## Script: collect_samples.py

UI interattiva per raccogliere training data. Flusso:

1. Legge tutte le foto da `assets/cats/` — ogni filename (senza estensione) è il label
2. Per ogni foto, in ordine:
   - Mostra la foto del gatto sul pannello destro in grande
   - Mostra il feed webcam sul pannello sinistro con skeleton
   - Stampa a schermo: `"Imita questo gatto! Premi SPAZIO per campionare, N per skippare"`
   - Quando l'utente preme SPAZIO: campiona `data_collection.samples_per_pose` frame consecutivi
   - Ogni frame: estrai landmark → feature vector → scrivi riga in `data/samples.csv`
   - Mostra contatore: "Campionati X/Y frame"
3. Alla fine: stampa riepilogo (quanti sample per classe) e suggerisce di lanciare train

> ⚠️ Se `assets/cats/` è vuota, stampa istruzioni chiare e termina senza crash.
> ⚠️ Se un frame non ha landmark visibili, skippalo silenziosamente (non contare come sample).

Formato `data/samples.csv`:
```
label,f0,f1,...,f98
tongue_cat,0.12,-0.34,...
grumpy_cat,-0.05,0.78,...
```

---

## Script: train_classifier.py

1. Legge `data/samples.csv`
2. Separa feature (f0..f98) da label
3. Encode label con `LabelEncoder`
4. Split stratificato 80/20
5. Addestra il modello scelto in config (`svm` o `mlp`)
6. Stampa classification report su validation set
7. Salva `models/classifier.pkl` e `models/label_encoder.pkl`
8. Se validation accuracy < 0.7, stampa warning: "Accuracy bassa — raccogli più campioni o controlla le pose"

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
  model_path: null  # null = scarica automaticamente MediaPipe

classifier:
  model: "svm"             # "svm" o "mlp"
  confidence_threshold: 0.4
  smoothing_window: 7      # frame per il majority vote
  svm:
    C: 10.0
    kernel: "rbf"
    gamma: "scale"
    probability: true
  mlp:
    hidden_layer_sizes: [128, 64]
    max_iter: 1000
    random_state: 42

data_collection:
  samples_per_pose: 30     # frame campionati per ogni posa
  cats_dir: "assets/cats"
  output_file: "data/samples.csv"

display:
  window_title: "CatPose"
  cat_panel_width: 480
  debug_mode: false
  font_scale: 1.0

paths:
  classifier: "models/classifier.pkl"
  label_encoder: "models/label_encoder.pkl"
  screenshots: "screenshots"
```

---

## Keyboard Shortcuts (OpenCV window)

| Tasto | Azione |
|-------|--------|
| `q` | Quit |
| `d` | Toggle debug overlay (top-3 predizioni con confidence) |
| `r` | Reset smoothing window (utile se la predizione è bloccata) |
| `s` | Salva screenshot in `screenshots/` |

---

## Workflow — Segui Questo Ordine

1. **Setup** — crea venv, installa requirements
2. **Crea struttura** — tutte le cartelle e file elencati in "Files to Create"
3. **Implementa `src/pose/detector.py`** — MediaPipe Tasks API wrapper
4. **Implementa `src/classifier/features.py`** — normalizzazione landmark → feature vector
5. **Implementa `src/classifier/trainer.py`** — train SVM/MLP, salva modello
6. **Implementa `src/classifier/predictor.py`** — carica modello, predice label + confidence
7. **Implementa `src/display/renderer.py`** — dual-panel OpenCV
8. **Implementa `scripts/collect_samples.py`** — UI raccolta dati
9. **Implementa `scripts/train_classifier.py`** — entry point training
10. **Implementa `main.py`** — wiring completo
11. **Smoke test** — verifica che collect_samples.py e main.py si avviano senza errori
12. **Scrivi README.md e TECHNICAL_CHOICES.md**
13. **Commit e push** — `git add -A && git commit -m "feat: initial working pipeline" && git push`

> ⚠️ La cartella `assets/cats/` deve essere tracciata con `.gitkeep` ma VUOTA nel repo.
> ⚠️ `data/samples.csv` e `models/*.pkl` sono gitignored — non committarli mai.

---

## Files to Create

Claude Code deve creare TUTTI i seguenti file prima di considerare il setup completo:

- [ ] `requirements.txt` — mediapipe, opencv-python, scikit-learn, Pillow, PyYAML, numpy, ruff
- [ ] `configs/config.yaml` — come sopra
- [ ] `assets/cats/.gitkeep`
- [ ] `data/.gitkeep`
- [ ] `models/.gitkeep`
- [ ] `screenshots/.gitkeep`
- [ ] `src/__init__.py`
- [ ] `src/pose/__init__.py`
- [ ] `src/pose/detector.py`
- [ ] `src/classifier/__init__.py`
- [ ] `src/classifier/features.py`
- [ ] `src/classifier/trainer.py`
- [ ] `src/classifier/predictor.py`
- [ ] `src/display/__init__.py`
- [ ] `src/display/renderer.py`
- [ ] `scripts/collect_samples.py`
- [ ] `scripts/train_classifier.py`
- [ ] `main.py`
- [ ] `README.md`
- [ ] `TECHNICAL_CHOICES.md`
- [ ] `Makefile`
- [ ] `.gitignore`

---

## Code Quality Standards

- **Type hints** su tutte le funzioni e metodi
- **Docstring Google-style** su tutte le classi e funzioni non-banali
- **PEP 8** — enforced via ruff
- **Zero valori hardcoded** — tutto in `configs/config.yaml`
- **Un file, una responsabilità** — no monoliti da 500 righe
- **Requirements** — ogni dipendenza pinnata con versione esatta
- **Graceful error handling**:
  - `assets/cats/` vuota → stampa istruzioni, termina con exit code 1
  - `models/classifier.pkl` assente → stampa istruzioni per il training, termina con exit code 1
  - frame senza landmark → skippa silenziosamente, non crashare
- **Nessuno stato globale** — passa config esplicitamente

---

## Code Review Checklist

Dopo aver scritto tutti i file, Claude Code deve verificare:

- [ ] `python scripts/collect_samples.py` si avvia e mostra UI corretta
- [ ] Con almeno 2 classi e 10 sample ciascuna, `python scripts/train_classifier.py` completa senza errori
- [ ] `python main.py` apre la webcam e classifica in real-time
- [ ] Se `assets/cats/` è vuota, il programma stampa istruzioni chiare e termina senza crash
- [ ] Se `models/classifier.pkl` è assente, `main.py` stampa istruzioni e termina senza crash
- [ ] Tutti i valori letti da `configs/config.yaml`, nessuno hardcoded
- [ ] Tutte le funzioni hanno type hints e docstring
- [ ] Nessun import error (tutti i package in requirements.txt)
- [ ] I tasti `q`, `d`, `r`, `s` funzionano in `main.py`
- [ ] FPS ≥ 15 su CPU
- [ ] `data/samples.csv` e `models/*.pkl` sono in `.gitignore`
- [ ] `git log` mostra commit per ogni step significativo
- [ ] `git push` completato con successo

---

## Response Style

- No preamble, no recap, no conferme verbali
- Esegui direttamente — non chiedere "sei sicuro?" o "vuoi che proceda?"
- Commit dopo ogni file o gruppo di file correlati
- Push alla fine del setup completo
- Se qualcosa è ambiguo, fai una scelta ragionevole e documentala in un commento

---

## Note per il README

Il README deve essere descrittivo e personale, non solo professionale. Deve spiegare:
- **Cosa fa** il progetto e perché è divertente
- **Come funziona** la logica (raccolta dati → training → real-time)
- **Come si usa** passo per passo, con esempi concreti di pose da imitare
- **Cosa succede** a runtime (cosa vedi sullo schermo)
- Le scelte tecniche principali in linguaggio accessibile
- Setup e istruzioni che funzionano davvero

---

## Future Extensions (non implementare ora)

- **collect_samples.py con augmentation**: flip orizzontale automatico dei landmark per raddoppiare i sample
- **Confidence heatmap**: visualizzare quanto ogni gatto è "vicino" alla posa corrente
- **Hot-reload**: aggiungere nuovi gatti senza riavviare (watchdog su `assets/cats/`)
- **Export pose**: salvare le pose catturate come GIF o video clip
- **Leaderboard**: tracciare quante volte riesci a imitare correttamente ogni gatto in una sessione
