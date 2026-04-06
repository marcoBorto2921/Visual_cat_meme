"""Interactive UI to collect training samples for the pose classifier.

Usage:
    python scripts/collect_samples.py

For each cat photo in assets/cats/:
  - Show the cat on the right panel
  - Show live webcam feed with skeleton on the left
  - Press SPACE to capture N landmark frames as training samples
  - Press N to skip to the next cat
  - Press Q to quit early
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.features import extract_features, FEATURE_DIM
from src.face.detector import FaceDetector
from src.pose.detector import PoseDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_config(path: str = "configs/config.yaml") -> dict:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_cat_images(cats_dir: Path) -> list[tuple[str, Path]]:
    """Load all cat images from the given directory.

    Args:
        cats_dir: Directory containing cat images.

    Returns:
        List of (label, path) tuples sorted by filename.

    Raises:
        SystemExit: If directory is empty or missing.
    """
    if not cats_dir.exists():
        print(
            f"\nERRORE: La cartella '{cats_dir}' non esiste.\n"
            "Crea la cartella e aggiungi le foto dei gatti prima di procedere.\n"
            "Esempio: assets/cats/tongue_cat.jpg, assets/cats/grumpy_cat.jpg\n"
        )
        sys.exit(1)

    images = [
        (p.stem, p)
        for p in sorted(cats_dir.iterdir())
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        print(
            f"\nERRORE: Nessuna immagine trovata in '{cats_dir}'.\n"
            "Aggiungi delle foto di gatti (es. tongue_cat.jpg, grumpy_cat.jpg)\n"
            "e rilancia lo script.\n"
        )
        sys.exit(1)

    return images


def render_collection_ui(
    frame_bgr: np.ndarray,
    cat_bgr: np.ndarray,
    label: str,
    collected: int,
    target: int,
    cat_panel_width: int,
    font_scale: float,
    sampling: bool,
) -> np.ndarray:
    """Compose the dual-panel UI frame for sample collection.

    Args:
        frame_bgr: Webcam frame (BGR).
        cat_bgr: Cat image resized to panel (BGR).
        label: Current cat label.
        collected: Frames collected so far for this cat.
        target: Total frames to collect.
        cat_panel_width: Width of the right panel.
        font_scale: Text scale.
        sampling: Whether sampling is currently active.

    Returns:
        Composed BGR image.
    """
    left = frame_bgr.copy()
    h, w = left.shape[:2]

    status_color = (0, 0, 255) if sampling else (0, 255, 0)
    status_text = f"CAMPIONANDO... {collected}/{target}" if sampling else "Premi SPAZIO per campionare"
    cv2.putText(
        left, status_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, status_color, 2,
    )
    cv2.putText(
        left, f"Gatto: {label}", (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 0), 2,
    )
    cv2.putText(
        left, "N=salta  Q=esci", (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55 * font_scale, (180, 180, 180), 1,
    )

    # Right panel: cat image with label overlay
    right = cat_bgr.copy()
    cv2.putText(
        right, label, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2,
    )
    cv2.putText(
        right, f"Imita questa posa!", (10, right.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * font_scale, (200, 200, 200), 1,
    )

    return np.hstack([left, right])


def resize_cat_image(path: Path, panel_width: int, panel_height: int) -> np.ndarray:
    """Load and resize a cat image to fill the panel while preserving aspect ratio.

    Args:
        path: Path to the cat image.
        panel_width: Target panel width.
        panel_height: Target panel height.

    Returns:
        BGR panel with the cat image centered.
    """
    from PIL import Image

    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    try:
        pil_img = Image.open(path).convert("RGB")
        orig_w, orig_h = pil_img.size
        scale = min(panel_width / orig_w, panel_height / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        x_off = (panel_width - new_w) // 2
        y_off = (panel_height - new_h) // 2
        panel[y_off : y_off + new_h, x_off : x_off + new_w] = img_bgr
    except Exception as exc:
        logger.warning("Cannot load '%s': %s", path, exc)
    return panel


def main() -> None:
    """Run the interactive sample collection UI."""
    config = load_config()
    cam_cfg = config["camera"]
    pose_cfg = config["pose"]
    dc_cfg = config["data_collection"]
    disp_cfg = config["display"]

    face_cfg = config.get("face", {"enabled": False})
    cats_dir = Path(dc_cfg["cats_dir"])
    output_file = Path(dc_cfg["output_file"])
    samples_per_pose: int = dc_cfg["samples_per_pose"]
    cat_panel_width: int = disp_cfg["cat_panel_width"]
    font_scale: float = disp_cfg["font_scale"]
    window_title: str = disp_cfg["window_title"] + " — Raccolta Dati"

    cat_images = load_cat_images(cats_dir)
    logger.info("Trovati %d gatti: %s", len(cat_images), [lbl for lbl, _ in cat_images])

    detector = PoseDetector(visibility_threshold=pose_cfg["visibility_threshold"])
    face_detector: FaceDetector | None = None
    if face_cfg.get("enabled", False):
        face_detector = FaceDetector(model_path=face_cfg.get("model_path"))

    cap = cv2.VideoCapture(cam_cfg["index"])
    if not cap.isOpened():
        logger.error("Impossibile aprire la webcam (indice %d)", cam_cfg["index"])
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    cap.set(cv2.CAP_PROP_FPS, cam_cfg["fps"])

    # Prepare output CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_file.exists()
    csv_file = open(output_file, "a", newline="")
    feature_cols = [f"f{i}" for i in range(FEATURE_DIM)]
    writer = csv.DictWriter(csv_file, fieldnames=["label"] + feature_cols)
    if write_header:
        writer.writeheader()

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    counts: dict[str, int] = {}

    try:
        for label, cat_path in cat_images:
            # Pre-load cat image at webcam frame height
            ret, probe = cap.read()
            panel_h = probe.shape[0] if ret else cam_cfg["height"]
            cat_panel = resize_cat_image(cat_path, cat_panel_width, panel_h)

            collected = 0
            sampling = False
            skip = False

            print(f"\n[{label}] Imita il gatto e premi SPAZIO per campionare.")

            while not skip:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)

                if sampling:
                    landmarks = detector.detect(frame)
                    face_result = face_detector.detect(frame) if face_detector else None
                    if landmarks:
                        features = extract_features(
                            landmarks, pose_cfg["visibility_threshold"], face_result
                        )
                        row = {"label": label}
                        row.update({f"f{i}": float(features[i]) for i in range(FEATURE_DIM)})
                        writer.writerow(row)
                        collected += 1
                    # If no landmarks, skip frame silently

                    if collected >= samples_per_pose:
                        sampling = False
                        skip = True
                        counts[label] = counts.get(label, 0) + collected
                        print(f"  [{label}] Campionati {collected} frame. Prossimo gatto...")

                composed = render_collection_ui(
                    frame, cat_panel, label, collected, samples_per_pose,
                    cat_panel_width, font_scale, sampling,
                )
                cv2.imshow(window_title, composed)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" ") and not sampling:
                    sampling = True
                    collected = 0
                    print(f"  [{label}] Campionamento avviato...")
                elif key == ord("n") or key == ord("N"):
                    counts[label] = counts.get(label, 0) + collected
                    skip = True
                    print(f"  [{label}] Skippato ({collected} frame raccolti).")
                elif key == ord("q") or key == ord("Q"):
                    counts[label] = counts.get(label, 0) + collected
                    print("\nUscita anticipata.")
                    _print_summary(counts, output_file)
                    return

    finally:
        csv_file.close()
        cap.release()
        detector.close()
        if face_detector:
            face_detector.close()
        cv2.destroyAllWindows()

    _print_summary(counts, output_file)


def _print_summary(counts: dict[str, int], output_file: Path) -> None:
    """Print collection summary and next step suggestion.

    Args:
        counts: Dict of label → samples collected.
        output_file: Path to the CSV file.
    """
    print("\n" + "=" * 50)
    print("RIEPILOGO CAMPIONAMENTO")
    print("=" * 50)
    for label, n in counts.items():
        print(f"  {label}: {n} campioni")
    total = sum(counts.values())
    print(f"\nTotale: {total} campioni salvati in '{output_file}'")
    print("\nProssimo step: python scripts/train_classifier.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
