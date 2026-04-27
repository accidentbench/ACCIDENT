"""
Full bounding-box dynamics baseline: YOLO detection, temporal & spatial prediction, and evaluation.

Standalone version of bbox-dynamics.ipynb (Steps 1-3 + Results), intended for running the full
pipeline outside of a Jupyter session. Visualization and submission CSV export are left to the notebook.

USAGE
-----
    python bbox_dynamics.py [OPTIONS]

    # Quick test on 5 videos:
    python bbox_dynamics.py --take 5

    # Recompute detections even if JSON files already exist:
    python bbox_dynamics.py --overwrite

    See --help for all arguments.
"""

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Generator, cast

import cv2
import numpy as np
import pandas as pd
import ruptures as rpt
import torch
from tqdm import tqdm
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dataset.accident_dataset import default_dataset_path, resolve_dataset_path
from metrics import print_temporal_accuracy, print_spatial_accuracy

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


# ---------------------------------------------------------------------------
# Step 1: Object Detection with YOLO v11
# ---------------------------------------------------------------------------


class Tracker:
    def __init__(
        self,
        model_path: str,
        image_resolution: int,
        batch_size: int,
        confidence_threshold: float,
        cuda_device_id: int,
    ):
        self.batch_size = batch_size
        self.image_resolution = image_resolution
        self.confidence_threshold = confidence_threshold

        self.model = YOLO(model_path)
        device = torch.device(f"cuda:{cuda_device_id}" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        print(f"Using YOLO model: {model_path} at {device.type} device.")

    def track(self, batches: Generator[list, None, None]) -> dict:
        """Inference on images.

        Args:
            batches: Iterable of batched images.
        Returns:
            List of processed predictions.
        """
        bboxes = []
        frames_indices = []
        class_ids, track_ids, confidences = [], [], []
        try:
            for batch_index, batch in enumerate(batches):
                results = self.model.track(
                    batch,
                    imgsz=self.image_resolution,  # must be a multiple of stride 32
                    verbose=False,
                    tracker="bytetrack.yaml",
                    persist=True,
                    conf=self.confidence_threshold,
                )

                for i, x in enumerate(results):
                    if x.boxes is not None and x.boxes.is_track:
                        frames_indices.append(batch_index * self.batch_size + i)
                        bboxes.append(x.boxes.xyxy.cpu().numpy().tolist())
                        class_ids.append(x.boxes.cls.cpu().numpy().astype(int).tolist())
                        track_ids.append(x.boxes.id.cpu().numpy().tolist())
                        confidences.append(x.boxes.conf.cpu().numpy().tolist())
        except Exception as e:
            print(f"ERROR: {e}")
        finally:
            assert (
                hasattr(self.model, "predictor")
                and self.model.predictor is not None
                and hasattr(self.model.predictor, "trackers")
                and self.model.predictor.trackers is not None
            )

            for tracker in self.model.predictor.trackers:
                tracker.reset()

        return {
            "frames": frames_indices,
            "bboxes": bboxes,
            "class_ids": class_ids,
            "track_ids": track_ids,
            "confidences": confidences,
        }

    def load_batched(self, cap: cv2.VideoCapture) -> Generator[list, None, None]:
        """Load video frames in batches.

        Args:
            cap: Opened cv2.VideoCapture object.
        Yields:
            Batches of video frames.
        """
        while True:
            batch: list = []
            for _ in range(self.batch_size):
                ret, frame = cap.read()
                if not ret:
                    return
                batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            yield batch

    def process_video_file(self, video_path: Path, dataset_path: Path) -> dict:
        """Track objects in a video file.

        Args:
            video_path: Path to a video file.
            dataset_path: Dataset root used to compute the relative path stored in the result.
        Returns:
            Dict with list of detections and other relevant data.
        """
        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))

            detections = self.track(self.load_batched(cap))
            detections = {
                **detections,
                "path": str(video_path.relative_to(dataset_path).as_posix()),
            }
        finally:
            if cap:
                cap.release()

        return detections


# ---------------------------------------------------------------------------
# Step 2: Temporal Detection - When Did the Accident Happen?
# ---------------------------------------------------------------------------


def find_bbox_size_change(detections: dict) -> tuple[int, np.ndarray]:
    """
    Find biggest change in bbox sizes in video.

    Returns:
        int: A frame with biggest change in bbox sizes
    """
    bbox_sizes = []
    for bboxes in detections["bboxes"]:
        if len(bboxes) == 0:
            if len(bbox_sizes) > 0:
                # If no bboxes detected in current frame, use the last known bbox size
                # (assuming objects are still present but not detected)
                bbox_sizes.append(bbox_sizes[-1])
            else:
                # No bboxes detected (yet), default to 0
                bbox_sizes.append(0)
        else:
            bbox_pixel_size = sum([abs(x2 - x1) * abs(y2 - y1) for x1, y1, x2, y2 in bboxes])
            bbox_sizes.append(bbox_pixel_size)

    if np.sum(bbox_sizes) == 0:
        # No bboxes detected in video, return the middle frame
        return len(detections["frames"]) // 2, bbox_sizes
    bbox_sizes = np.array(bbox_sizes)
    bbox_sizes = bbox_sizes / np.sum(bbox_sizes)

    changes = rpt.KernelCPD(kernel="rbf").fit_predict(bbox_sizes, n_bkps=3)
    assert changes and len(changes) >= 1
    return ((detections["frames"][changes[0]] + detections["frames"][changes[0] + 1]) / 2), bbox_sizes


# ---------------------------------------------------------------------------
# Step 3: Spatial Detection - Where Did the Accident Happen?
# ---------------------------------------------------------------------------


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Full bounding-box dynamics baseline: detection, temporal & spatial prediction, evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=default_dataset_path(REPO_ROOT),
        help="Path to dataset/ (default: ../../dataset)",
    )
    parser.add_argument(
        "--detections-dir",
        type=Path,
        default=SCRIPT_DIR / "inference-yolo11x",
        help="Directory for detection JSON files (default: ./inference-yolo11x)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./yolo11x.pt",
        help="Path to YOLO model weights (default: ./yolo11x.pt)",
    )
    parser.add_argument(
        "--image-resolution",
        type=int,
        default=1280,
        help="YOLO input resolution, must be a multiple of 32 (default: 1280)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of frames per batch (default: 8)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.15,
        help="Minimum detection confidence threshold (default: 0.15)",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="GPU device ID (default: 0)",
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
        help="Recompute even if detection JSON already exists",
    )
    args = parser.parse_args()

    dataset_path = resolve_dataset_path(args.dataset_path)
    detections_dir: Path = args.detections_dir
    detections_dir.mkdir(parents=True, exist_ok=True)

    video_paths = list((dataset_path / "real_videos").iterdir())
    metadata_df = pd.read_csv(dataset_path / "metadata-real.csv", index_col="path")

    if args.take is not None:
        video_paths = video_paths[: args.take]

    # ---- Step 1: YOLO detection ----
    print(f"Step 1: Processing {len(video_paths)} videos from {dataset_path}")

    tracker = Tracker(
        model_path=args.model_path,
        image_resolution=args.image_resolution,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence,
        cuda_device_id=args.cuda_device,
    )

    for filename in tqdm(video_paths):
        filename = cast(Path, filename)
        predictions_path = detections_dir / filename.with_suffix(".json").name

        # Skip if predictions already exist
        if predictions_path.exists() and not args.overwrite:
            print(f"Skipping {predictions_path}. Predictions already exists.")
            continue

        # Process video
        predictions = tracker.process_video_file(filename, dataset_path)

        # Save predictions
        with open(predictions_path, "w") as f:
            json.dump(predictions, f)

    # ---- Step 2: Temporal Detection ----
    print("Step 2: Temporal detection")

    results: dict[str, dict] = {}
    for video in tqdm(video_paths):
        video = cast(Path, video)
        with open(detections_dir / video.with_suffix(".json").name, "r") as f:
            info = json.load(f)

        # Get video metadata to calculate FPS and convert frame number to time
        metadata = metadata_df.loc[info["path"]]
        fps = metadata["no_frames"] / metadata["duration"]

        # Find frame with biggest change in bbox sizes
        frame, _ = find_bbox_size_change(info)
        results[info["path"]] = {"frame": frame, "accident_time": frame / fps}

    # ---- Step 3: Spatial Detection ----
    print("Step 3: Spatial detection")

    for video in tqdm(video_paths):
        with open(detections_dir / video.with_suffix(".json").name, "r") as f:
            info = json.load(f)

        result = results[info["path"]]
        accident_frame = int(result["frame"])
        metadata = metadata_df.loc[info["path"]]

        bboxes = next(bboxes for frame, bboxes in zip(info["frames"], info["bboxes"]) if frame == accident_frame)
        if len(bboxes) == 0:  # no detections in the accident frame, use frame center
            result["center_x"] = 0.5
            result["center_y"] = 0.5
            continue

        bbox_centers = [midpoint((x1, y1), (x2, y2)) for (x1, y1, x2, y2) in bboxes]
        if len(bbox_centers) == 1:  # only one detection, use its center
            result["center_x"] = bbox_centers[0][0] / metadata["width"]
            result["center_y"] = bbox_centers[0][1] / metadata["height"]
            continue

        (c1, c2) = min(
            combinations(bbox_centers, 2),
            key=lambda pair: euclidean_distance(pair[0], pair[1]),
        )

        # prediction = midpoint of those two centers
        prediction_center = midpoint(c1, c2)
        result["center_x"] = prediction_center[0] / metadata["width"]
        result["center_y"] = prediction_center[1] / metadata["height"]

    # Put the default type:
    for result in results.values():
        result["type"] = "single"

    # ---- Results ----
    results_df = pd.DataFrame(
        [{"path": str(path), **i, "frame": None} for path, i in results.items()]
    ).drop(columns=["frame"])

    print_temporal_accuracy(results_df, dataset_path)
    print_spatial_accuracy(results_df, dataset_path)

    # ---- Ensemble (if optical flow results are available) ----
    optical_flow_csv_path = SCRIPT_DIR / "output_optical_flow.csv"
    if optical_flow_csv_path.exists():
        print("Ensemble (optical flow + bbox dynamics):")
        optical_flow_df = pd.read_csv(optical_flow_csv_path)
        ensemble = results_df.copy()
        of_times = optical_flow_df.set_index("path").loc[results_df["path"]]["accident_time"].values
        ensemble["accident_time"] = (of_times + results_df["accident_time"].values) / 2
        print_temporal_accuracy(predictions=ensemble, dataset_path=dataset_path)
    else:
        print(f"Skipping ensemble: {optical_flow_csv_path} not found.")


if __name__ == "__main__":
    main()
