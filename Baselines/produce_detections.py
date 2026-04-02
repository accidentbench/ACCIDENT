import gc
import logging
import json
import os
import os.path as osp
from pathlib import Path
from tqdm import tqdm
import time
import traceback
from functools import wraps
from multiprocessing import Process, Queue
from typing import List, Tuple
import pandas as pd

import cv2
import numpy as np
import torch
from ultralytics import YOLO


YOLO_MODEL_PATH="./resources/yolo11x.pt"
RESOLUTION_CHANGE = 750

YOLO_IMAGE_RESOLUTION=1280
# YOLO_IMAGE_RESOLUTION=640

USE_SMALLER = YOLO_IMAGE_RESOLUTION == 640

YOLO_BATCH_SIZE=8
NUM_CPU_JOBS=16
CONF_THR = 0.15

DEVICE_ID = 0

TRACKING_TIMEOUT_SECS=600

RECOGNIZED_CLASSES = [
    # "person", "bicycle", "car", "motorcycle"
]


def timeit(func):
    """Run time measuring decorator."""

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function "{func.__name__}" Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


def init_yolo_model(model_path: str, device_id: int = DEVICE_ID) -> YOLO:
    """Creates new instance of YOLO model on a specific GPU or CPU."""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    model = YOLO(model_path)
    print(f"Moving YOLO model: {model_path} to {device.type} device.")
    model.to(device)
    return model


def yolo_track(model: YOLO, frames: List[np.ndarray]) -> List[dict]:
    """Detects objects in the video frames and tracks them.

    Frames are processed as a whole batch.

    :param model: Instance of an YOLO model.
    :param frames: List of images/video frames.
    :return: List of predictions.
    """
    results = model.track(
        frames,
        imgsz=YOLO_IMAGE_RESOLUTION,  # must be a multiple of stride 32
        verbose=False,
        tracker="bytetrack.yaml",
        persist=True,
        conf=CONF_THR,
    )

    if not results:
        return []

    id2label = results[0].names

    preds = []
    for i, x in enumerate(results):
        if not x.boxes.is_track:
            preds.append({
                "frame_number": i,
            })
            # print("Bbox is_track is false! Skipping prediction.")
            continue
        class_ids = x.boxes.cls.cpu().numpy().astype(int).tolist()
        preds.append(
            {
                # "full_image_path": x.path,
                # "image_path": os.path.relpath(x.path, dataset_dir),
                "frame_number": i,
                "bboxes": (x.boxes.xyxy.cpu().numpy()).tolist(),
                "class_ids": class_ids,
                "labels": [id2label[x] for x in class_ids],
                "track_ids": x.boxes.id.numpy().tolist(),
                "confs": x.boxes.conf.cpu().numpy().tolist(),
                "speed": x.speed,
            }
        )
    return preds


def read_batch_resized_frames(capture: cv2.VideoCapture, batch_size: int):
    """Read batch of frames from capture and optionally resizes them."""
    frames = []
    for frame_id in range(0, batch_size):
        # get image from video
        ret, image = capture.read()
        if image is None:
            continue

        # if new_size is not None:
        #     image = resize_image(image, new_size=new_size)
        # process i-th frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)
    return frames


def resize_image(image: np.ndarray, new_size: tuple) -> np.ndarray:
    """Resize image, keep aspect ratio."""
    resized_frame = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized_frame


def get_resize_scale(width: int, height: int, max_size: int) -> Tuple[float, Tuple]:
    """Get scaling factor and new dimensions.

    No new dimension is larger than max_size.
    """
    scale = min(max_size / height, max_size / width)
    if scale > 1:
        return 1.0, (width, height)
    new_size = (int(width * scale), int(height * scale))
    return scale, new_size


class GeoRegTracker:
    def __init__(
            self,
            model_path: str,
            batch_frames: int = 50,
    ):
        self.model_path = model_path
        self.batch_frames = batch_frames

        self.model = init_yolo_model(self.model_path)

        self.track_history = None
        self.fps, self.end_frame, self.image_size = None, None, None

        # self.scale, self.new_size = 1.0, None  # Frames within YOLO resolution are not resized.

    @timeit
    def process_video(
            self,
            video_path: str,
    ) -> list:
        """Track objects in a local video file.

        Loads local video file and inferences on the video frames.
        This is done in batches of frames depending on the size of memory,
        must be set manually in .env.

        Returns a list of predictions that compose of class info, object bbox,
        map info, and track id.

        :param video_path: Path to a video file.
        :return: List of processed predictions.
        """
        capture = None
        try:
            capture = self.open_video_capture(video_path)

            preds = self.track_parallel(
                capture=capture,
                batch_size=self.batch_frames,
            )

        except Exception as e:
            print(traceback.format_exc())
            raise e
        finally:
            # Clean up
            # Reset trackers safely
            if getattr(self.model, "predictor", None) and getattr(self.model.predictor, "trackers", None):
                for tracker in self.model.predictor.trackers:
                    tracker.reset()
            if capture:
                capture.release()


        return preds

    def open_video_capture(self, video_path: str) -> cv2.VideoCapture:
        """Creates video capture and loads video parameters.

        :param video_path: Path to the video file.
        :return: Opened video capture.
        """
        capture = cv2.VideoCapture(video_path)
        # Get video info
        self.fps = capture.get(cv2.CAP_PROP_FPS)
        print(f"FPS: {self.fps}")
        self.end_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # self.image_size = (
        #     int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        #     int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        # )
        # max_size = int(YOLO_IMAGE_RESOLUTION)
        # # Resize only if larger than YOLO input resolution
        # if any([size > max_size for size in self.image_size]):
        #     self.scale, self.new_size = get_resize_scale(*self.image_size, max_size)
        # print(f"Resize: {self.image_size} >> {self.new_size}")
        return capture

    def get_video_info(self) -> tuple:
        """Return stats about the processed video."""
        return self.fps, self.end_frame

    def track_parallel(
            self,
            capture: cv2.VideoCapture,
            batch_size: int,
    ) -> list:
        """Inference on images. Runs postprocessing in a parallel process.

        Loads images in parallel in batches, runs yolo inference and puts predictions into
        working queue, where the object smoothening and homography proceeds.
        The resulting predictions are harvested from the output queue in
        the end and returned.

        :param capture: Opened video capture.
        :param batch_size: Image batch size.
        :return: List of processed predictions.
        """
        tracked_predictions = []

        for batch_offset in range(self.end_frame // batch_size + 1):
            # print(
            #     f"Inference batch: {batch_offset + 1}, {self.end_frame // batch_size + 1}"
            # )
            images = read_batch_resized_frames(capture, batch_size)

            try:
                yolo_predictions = yolo_track(self.model, images)
                for prediction in yolo_predictions:
                    prediction["frame_number"] = batch_offset * batch_size + prediction["frame_number"]
            except Exception as e:
                print(f"ERROR: {e}")
                yolo_predictions = []
            tracked_predictions.extend(yolo_predictions)
            del images
            gc.collect()

        return tracked_predictions


if __name__ == "__main__":
    # dataset_dir = "./datasets"
    dataset_dir = "/media/marek/disk/datasets/sifco-accident/"
    metadata_path = osp.join(dataset_dir, "labels.csv")
    metadata = pd.read_csv(metadata_path)

    output_dir = osp.join(dataset_dir, "inference-yolo11x")
    os.makedirs(output_dir, exist_ok=True)

    inference_worker = GeoRegTracker(
        model_path=YOLO_MODEL_PATH,
        batch_frames=YOLO_BATCH_SIZE,
    )

    for i, row in tqdm(metadata.iloc[:600].iterrows(), total=len(metadata.iloc[:600])):
        height, width = row["height"], row["width"]
        max_size = max(height, width)
        if USE_SMALLER and max_size > RESOLUTION_CHANGE:
            continue

        path = row["path"]
        video_path = osp.join(dataset_dir, path)

        output_sub_dir = osp.join(output_dir, osp.dirname(path))
        # output_video_path = osp.join(output_sub_dir, f"{Path(path).stem}.mp4" )
        predictions_path = osp.join(output_sub_dir, Path(path).stem) +  ".json"

        os.makedirs(output_sub_dir, exist_ok=True)

        if osp.exists(predictions_path):
            print(f"Skipping {path}. Predictions already exists.")
            continue

        predictions = inference_worker.process_video(video_path)

        with open(predictions_path, "w") as f:
            json.dump(predictions, f)


