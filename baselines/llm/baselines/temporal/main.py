#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
LLM_BASELINES_DIR = SCRIPT_DIR.parents[1]
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(LLM_BASELINES_DIR))

from dataset.accident_dataset import default_dataset_path, resolve_dataset_path
from reasoning.utils import get_every_nth_frame
from reasoning.qwen import QwenVLReasoner
from reasoning.molmo import MolmoReasoner


def run_temporal_reasoning(model, name, df):
    results = []
    os.makedirs(f'results/{name}_temporal_parts', exist_ok=True)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        with open(f"progress_{name}.log", "a") as f:
            print(f"Processed row {i+1}/{len(df)}", file=f)

        data = row.to_dict()

        fps = row['no_frames'] / row['duration']
        imgs = get_every_nth_frame(row['video_path'], n=round(fps * 0.5))
        frame_id, temporal_raw = model.accident_temporal_reasoning(imgs)

        pred_ts = None if frame_id is None else frame_id / fps

        data['temporal'] = {
            'raw': temporal_raw,
            'pred_frame': frame_id,
            'pred_ts': pred_ts,
        }

        torch.save(data, f"results/{name}_temporal_parts/{i}.pkl")
        results.append(data)

    return results


# ---- CLI ----
def parse_range(r):
    """
    Parse START:END into slice indices.
    """
    if r is None:
        return None

    try:
        start, end = r.split(":")
        return int(start), int(end)
    except Exception:
        raise argparse.ArgumentTypeError(
            "Range must be in format START:END (e.g. 50:52)"
        )


def main():
    parser = argparse.ArgumentParser(description="Run temporal accident reasoning")

    parser.add_argument(
        "--model",
        choices=["molmo", "qwen"],
        required=True,
        help="Model to use"
    )
    parser.add_argument(
        "--range",
        type=parse_range,
        default=None,
        help="Optional iloc range START:END (e.g. 50:52)"
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=default_dataset_path(PROJECT_ROOT),
        help="Path to dataset/ or dataset/real_videos (default: ../../dataset/real_videos from repo root)",
    )

    args = parser.parse_args()

    # load data
    dataset_path = resolve_dataset_path(args.dataset_path)
    df = pd.read_csv(dataset_path / "labels.csv")
    df["video_path"] = (dataset_path / "videos").as_posix() + "/" + df["path"]

    # subset if requested
    if args.range is not None:
        start, end = args.range
        df = df.iloc[start:end]

    # model selection
    if args.model == "molmo":
        model = MolmoReasoner()
        name = "molmo"
    else:
        model = QwenVLReasoner()
        name = "qwen"

    run_temporal_reasoning(model=model, name=name, df=df)


if __name__ == "__main__":
    main()
