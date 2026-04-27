"""
Naive baseline and oracle-aided evaluation for the accident detection competition.

Standalone version of oracle.ipynb. Produces naive predictions (video midpoint, frame center)
and an oracle-aided median baseline, then prints temporal, spatial, and classification metrics.

USAGE
-----
    python oracle.py [OPTIONS]

    See --help for all arguments.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dataset.accident_dataset import default_dataset_path, resolve_dataset_path
from metrics import LABELS_PATH, print_temporal_accuracy, print_spatial_accuracy

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Naive baseline and oracle-aided evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=default_dataset_path(REPO_ROOT),
        help="Path to dataset/ (default: ../../dataset)",
    )
    args = parser.parse_args()

    dataset_path = resolve_dataset_path(args.dataset_path)
    true_df = pd.read_csv(dataset_path / LABELS_PATH)

    # ---- Naive baseline ----
    naive = true_df.copy()
    # half of the video duration
    naive["accident_time"] = true_df["duration"] / 2
    # center of the video
    naive["center_x"] = 0.5
    naive["center_y"] = 0.5

    print_temporal_accuracy(
        predictions=naive,
        dataset_path=dataset_path,
    )
    print_spatial_accuracy(
        predictions=naive,
        dataset_path=dataset_path,
    )

    print("Classification task:")
    print(sum(true_df["type"] == "single") / len(true_df))

    # ---- Oracle aided ----
    naive_oracle = true_df.copy()
    naive_oracle["accident_time"] = np.median(true_df["accident_time"])

    print("Median - ", end="")
    print_temporal_accuracy(
        predictions=naive_oracle,
        dataset_path=dataset_path,
    )


if __name__ == "__main__":
    main()
