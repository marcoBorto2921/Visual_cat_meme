"""Visualize landmark distributions per pose class from collected data.

Run: python notebooks/01_pose_analysis.py --input data/poses.csv
"""

import argparse
import csv
from collections import defaultdict

import numpy as np

LANDMARK_NAMES = {
    0: "NOSE",
    11: "L_SHOULDER",
    12: "R_SHOULDER",
    13: "L_ELBOW",
    14: "R_ELBOW",
    15: "L_WRIST",
    16: "R_WRIST",
    23: "L_HIP",
    24: "R_HIP",
}

KEY_INDICES = sorted(LANDMARK_NAMES.keys())


def main() -> None:
    """Print landmark statistics per pose class."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/poses.csv")
    args = parser.parse_args()

    pose_data: dict[str, list[list[float]]] = defaultdict(list)

    with open(args.input) as f:
        reader = csv.reader(f)
        for row in reader:
            label = row[0]
            values = list(map(float, row[1:]))
            # Each landmark is 4 floats: x, y, z, visibility
            key_vals = []
            for idx in KEY_INDICES:
                base = idx * 4
                key_vals.extend(values[base:base + 4])
            pose_data[label].append(key_vals)

    print("=== Landmark Statistics per Pose ===\n")
    for pose, samples in sorted(pose_data.items()):
        arr = np.array(samples)
        print(f"Pose: {pose} ({len(samples)} samples)")
        for i, idx in enumerate(KEY_INDICES):
            base = i * 4
            x_mean = arr[:, base].mean()
            y_mean = arr[:, base + 1].mean()
            vis_mean = arr[:, base + 3].mean()
            print(f"  {LANDMARK_NAMES[idx]:12s} x={x_mean:.3f} y={y_mean:.3f} vis={vis_mean:.3f}")
        print()


if __name__ == "__main__":
    main()
