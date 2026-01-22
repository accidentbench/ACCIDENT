import pickle
import os
import os.path as osp
import json
import yaml

import cv2
import numpy as np

from pathlib import Path
from typing import Any


def load_pickle(file_path: str) -> dict:
    """
    Load a pickle file and return its contents.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        dict: Contents of the pickle file.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_frame_annotations(
    annotations: dict, video: str, frame: int, frame_image: np.ndarray
):
    """
    Extract annotations for a specific video frame and return bounding boxes, centers, chips, and track IDs.

    Args:
        annotations (dict): Dictionary containing video annotations.
        video (str): Video identifier.
        frame (int): Frame index.
        frame_image (np.ndarray): Image data of the current frame.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Array of bounding boxes.
            - np.ndarray: Array of bounding box centers.
            - list[np.ndarray]: List of cropped chips from the frame.
            - list[int]: List of track IDs.
    """
    bboxes, bbox_centers, chips, track_ids = [], [], [], []
    for ann_type in ["challenge_object"]:
        try:
            for i in range(len(annotations[video][frame][ann_type])):
                x1, y1, x2, y2 = annotations[video][frame][ann_type][i]["bbox"]
                track_ids.append(annotations[video][frame][ann_type][i]["track_id"])
                bboxes.append([x1, y1, x2, y2])
                bbox_centers.append([x1 + (abs(x2 - x1) / 2), y1 + (abs(y2 - y1) / 2)])
                chips.append(frame_image[int(y1) : int(y2), int(x1) : int(x2)])
        except Exception as e:
            print(f"KeyError: {video}_{frame}")
    bboxes, bbox_centers = np.array(bboxes), np.array(bbox_centers)
    return bboxes, bbox_centers, chips, track_ids


def load_video_frames(video_path: str) -> list:
    """
    Process a video and return a list of its frame data.

    Args:
        video_path (str): Path to the video file.

    Returns:
        list: A list of tuples containing frame identifiers and frame images.
    """
    video_name = Path(video_path).stem
    video_data = []
    video_stream = cv2.VideoCapture(video_path)
    if not video_stream.isOpened():
        raise ValueError(f"Video {video_name} could not be opened.")

    frame = 0
    while video_stream.isOpened():
        ret, frame_image = video_stream.read()
        if not ret:
            break
        video_data.append((f"{video_name}_{frame}", frame_image))
        frame += 1

    video_stream.release()
    return video_data

def load_json(path: str) -> Any:
    """Load a json file from path."""
    with open(path, "r") as fp:
        return json.load(fp)


def save_json(path: str, content: dict) -> None:
    """Save a json serializable dict to a file. Create dirs in the path if not already present."""
    dir_path = osp.dirname(path)
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    with open(path, "w") as fp:
        json.dump(content, fp)


def load_yaml(path: str) -> Any:
    """Load a yaml file from path."""
    with open(path, "r") as fp:
        return yaml.safe_load(fp)


def save_yaml(path: str, content: dict):
    """Save a yaml serializable dict to a file. Create dirs in the path if not already present."""
    dir_path = osp.dirname(path)
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    with open(path, "w") as fp:
        yaml.safe_dump(content, fp)

