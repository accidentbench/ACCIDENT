"""
Full optical flow baseline: compute motion scores, temporal prediction, and evaluation.

Standalone version of optical_flow.ipynb (Steps 1-2 + Results), intended for running the full
pipeline outside of a Jupyter session. Visualization and submission CSV export are left to the notebook.

USAGE
-----
    python optical_flow.py [OPTIONS]

    # Quick test on 5 videos using 4 CPU cores:
    python optical_flow.py --take 5 --n-jobs 4

    # Recompute even if output already exists:
    python optical_flow.py --overwrite

    See --help for all arguments.
"""

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import ruptures as rpt
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dataset.accident_dataset import default_dataset_path, resolve_dataset_path
from metrics import print_temporal_accuracy

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


# ---------------------------------------------------------------------------
# Step 1: Computing Optical Flow Scores
# ---------------------------------------------------------------------------

# Algorithm parameters:
TARGET_FPS = 5  # Target frames per second for processing. Lower FPS = faster computation but less temporal resolution


def compute_optical_flow_score_on_frame(
    current_frame: cv2.typing.MatLike,
    previous_frame: cv2.typing.MatLike,
    motion_threshold: float = 0.0,
) -> float:
    """
    Computes the degree of motion between the current and previous frames.

    Args:
        current_frame: Current frame in grayscale.
        previous_frame: Previous frame in grayscale.
        motion_threshold: Threshold to filter out low-motion noise.

    Returns:
        A single value representing the degree of motion in the frame.
    """
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev=previous_frame,
        next=current_frame,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    # Compute magnitude and angle of flow vectors
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    magnitude = np.clip(magnitude, 0, 1e19)

    # Filter out small magnitudes (noise) using a threshold
    magnitude = magnitude[magnitude > motion_threshold]

    # Return the mean magnitude
    return magnitude.mean() if magnitude.size > 0 else 0.0


def compute_optical_flow_scores_on_video(
    video_path: Path, dataset_path: Path, target_fps: float
) -> dict:
    """
    Process a single video file to compute optical flow at the target FPS.

    Args:
        video_path: Absolute path to the video file.
        dataset_path: Dataset root used to compute the relative path stored in the result.
        target_fps: Target frames per second for sampling.

    Returns:
        Dictionary with keys 'path', 'scores', and 'frames'.
        Note: len(frames) == len(scores) + 1.
    """
    assert video_path.exists(), f"Video not found: {video_path}"

    cap = cv2.VideoCapture(str(video_path))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    # will use every `frame_skip`-th frame of the video
    frame_skip = int(original_fps / min(original_fps, target_fps))

    previous_frame = None
    scores = []
    frames = []
    current_frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame_idx % frame_skip == 0:
            current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(current_frame_idx)
            if previous_frame is not None:
                scores.append(
                    compute_optical_flow_score_on_frame(current_frame, previous_frame)
                )
            previous_frame = current_frame

        current_frame_idx += 1

    cap.release()
    return {
        "path": str(video_path.relative_to(dataset_path).as_posix()),
        "scores": np.array(scores),
        "frames": np.array(frames),
    }


# ---------------------------------------------------------------------------
# Step 2: Temporal Detection - When Did the Accident Happen?
# ---------------------------------------------------------------------------


def find_temporal_change(info: dict, metadata: pd.Series) -> dict:
    """
    Find the accident time using change-point detection on optical flow scores.

    Args:
        info: Dict with 'scores' and 'frames' from Step 1.
        metadata: Video metadata row with 'no_frames' and 'duration'.

    Returns:
        Dict with 'accident_time'.
    """
    # normalize the scores
    scores = info["scores"] / np.linalg.norm(info["scores"])

    # search for the biggest changes in the optical flow and get its frame
    changes = rpt.KernelCPD(kernel="rbf", min_size=1 if len(scores) < 8 else 2).fit_predict(scores, n_bkps=3)
    assert changes and len(changes) >= 1
    # take the middle frame/frame fraction as the optical flow is a change between two frames
    frame = (info["frames"][changes[0]] + info["frames"][changes[0] + 1]) / 2
    fps = metadata["no_frames"] / metadata["duration"]
    return {"accident_time": float(frame / fps)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Full optical flow baseline: compute motion scores, temporal prediction, evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=default_dataset_path(REPO_ROOT),
        help="Path to dataset/ (default: ../../dataset)",
    )
    parser.add_argument(
        "--optical-flow-path",
        type=Path,
        default=SCRIPT_DIR / "optical_flow.pkl",
        help="Path for the optical flow pickle file (default: ./optical_flow.pkl)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=5.0,
        help="Target FPS for frame sampling (default: 5)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel workers, -1 = all cores (default: -1)",
    )
    parser.add_argument(
        "--take",
        type=int,
        default=None,
        help="Process only the first N videos (default: all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if the optical flow pickle already exists",
    )
    args = parser.parse_args()

    dataset_path = resolve_dataset_path(args.dataset_path)
    optical_flow_path: Path = args.optical_flow_path

    video_paths = list((dataset_path / "real_videos").iterdir())
    metadata_df = pd.read_csv(dataset_path / "metadata-real.csv", index_col="path")

    if args.take is not None:
        video_paths = video_paths[: args.take]

    # ---- Step 1: Compute optical flow scores ----
    if optical_flow_path.exists() and not args.overwrite:
        print(f"Optical flow scores already computed and saved at {optical_flow_path}.")
        with open(optical_flow_path, "rb") as f:
            optical_flow = pickle.load(f)
    else:
        print(f"Step 1: Processing {len(video_paths)} videos from {dataset_path}")
        print(f"Target FPS: {args.target_fps} | Workers: {args.n_jobs}")

        optical_flow = Parallel(n_jobs=args.n_jobs)(
            delayed(compute_optical_flow_scores_on_video)(path, dataset_path, args.target_fps)
            for path in tqdm(video_paths)
        )

        with open(optical_flow_path, "wb") as f:
            pickle.dump(optical_flow, f)

        print(f"Saved {len(optical_flow)} results to {optical_flow_path}")

    # ---- Step 2: Temporal Detection ----
    print("Step 2: Temporal detection")

    take = args.take
    results: dict[str, dict] = {}
    for info in optical_flow[: take or len(optical_flow)]:
        metadata = metadata_df.loc[info["path"]]
        results[info["path"]] = find_temporal_change(info, metadata)

    # Put default values for center_x, center_y, and type
    for path, item in results.items():
        item["center_x"] = 0.5
        item["center_y"] = 0.5
        item["type"] = "single"

    # ---- Results ----
    results_df = pd.DataFrame([{"path": str(path), **i} for path, i in results.items()])
    results_df.to_csv(SCRIPT_DIR / "output_optical_flow.csv", index=False)

    print_temporal_accuracy(results_df, dataset_path)

    # ---- Ensemble (if bbox dynamics results are available) ----
    bbox_dynamics_path = SCRIPT_DIR / "output_bbox_dynamics.csv"
    if bbox_dynamics_path.exists():
        print("Ensemble (optical flow + bbox dynamics):")
        bbox_dynamics = pd.read_csv(bbox_dynamics_path)
        ensemble = bbox_dynamics.copy()
        of_times = results_df.set_index("path").loc[bbox_dynamics["path"]]["accident_time"].values
        ensemble["accident_time"] = (of_times + bbox_dynamics["accident_time"].values) / 2
        print_temporal_accuracy(predictions=ensemble, dataset_path=dataset_path)
    else:
        print(f"Skipping ensemble: {bbox_dynamics_path} not found.")


if __name__ == "__main__":
    main()
