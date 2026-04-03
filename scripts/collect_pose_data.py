"""Collect labeled pose samples for optional ML classifier training.

Usage:
    python scripts/collect_pose_data.py --pose arms_up --output data/poses.csv
"""

import argparse
import csv
from pathlib import Path

import cv2
import yaml

from src.pose.detector import PoseDetector


def main() -> None:
    """Run the pose data collection loop."""
    parser = argparse.ArgumentParser(description="Collect labeled pose samples")
    parser.add_argument("--pose", required=True, help="Pose label to record")
    parser.add_argument("--output", default="data/poses.csv", help="Output CSV path")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    args = parser.parse_args()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    detector = PoseDetector(visibility_threshold=config["pose"]["visibility_threshold"])

    cap = cv2.VideoCapture(config["camera"]["index"])
    collected = 0

    print(f"Collecting {args.samples} samples for pose '{args.pose}'. Press SPACE to capture, Q to quit.")

    with open(args.output, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        while collected < args.samples:
            ret, frame = cap.read()
            if not ret:
                continue
            annotated, landmarks = detector.process(frame)
            cv2.imshow("Collect Poses", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                row = [args.pose]
                for lm in landmarks:
                    if lm:
                        row.extend([lm.x, lm.y, lm.z, lm.visibility])
                    else:
                        row.extend([0.0, 0.0, 0.0, 0.0])
                writer.writerow(row)
                collected += 1
                print(f"  Captured {collected}/{args.samples}")
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print(f"Saved {collected} samples to {args.output}")


if __name__ == "__main__":
    main()
