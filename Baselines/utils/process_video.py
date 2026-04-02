import os
import warnings
from itertools import zip_longest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import ruptures as rpt
from PIL import Image

from .driver_state import (baseline_driver_state_changed,
                           bbox_size_drive_state_changed,
                           calculate_bbox_size_total,
                           optical_flow_driver_state_changed)
from .ioutils import load_frame_annotations, load_video_frames

warnings.filterwarnings("ignore")


def baseline_get_hazard(
    frame_image: np.ndarray,
    bbox_centers: np.ndarray,
    track_ids: list(),
    chips: list(),
    captioned_tracks: dict,
) -> list():
    """
    Identify the most probable hazard from the frame based on distance to the image center.

    Args:
        frame_image (np.ndarray): The frame image as a NumPy array.
        bbox_centers (np.ndarray): Array of bounding box centers.
        track_ids (list[int]): List of track IDs corresponding to bounding boxes.
        chips (list[np.ndarray]): List of cropped regions (chips) from the frame.
        captioned_tracks (dict): Dictionary mapping track IDs to captions.

    Returns:
        list[tuple]: A list containing a tuple of hazard track and its caption.
    """
    image_center = [frame_image.shape[1] / 2, frame_image.shape[0] / 2]
    if bbox_centers.shape[0] == 0:
        return [
            (" ", " "),
        ]
    potential_hazard_dists = np.linalg.norm(bbox_centers - image_center, axis=1)
    probable_hazard = np.argmin(potential_hazard_dists)
    hazard_track = track_ids[probable_hazard]

    if hazard_track not in captioned_tracks:
        hazard_chip = cv2.cvtColor(chips[probable_hazard], cv2.COLOR_BGR2RGB)
        hazard_chip = Image.fromarray(hazard_chip)
        # Generate caption
        # hazard_caption = ci.interrogate(hazard_chip)
        hazard_caption = " "

        hazard_caption = hazard_caption.replace(",", " ")

        captioned_tracks[hazard_track] = hazard_caption
    else:
        hazard_caption = captioned_tracks[hazard_track]

    return [
        (hazard_track, hazard_caption),
    ]


def get_hazards(
    frame_image: np.ndarray,
    bbox_centers: np.ndarray,
    track_ids: list(),
    chips: list(),
    captioned_tracks: dict,
    num_tracks: int = 100,
    run_captions: bool = False,
) -> list():
    """
    Identify potential hazards in the frame and generate captions if enabled.

    Args:
        frame_image (np.ndarray): The frame image as a NumPy array.
        bbox_centers (np.ndarray): Array of bounding box centers.
        track_ids (list[int]): List of track IDs corresponding to bounding boxes.
        chips (list[np.ndarray]): List of cropped regions (chips) from the frame.
        captioned_tracks (dict): Dictionary mapping track IDs to captions.
        num_tracks (int, optional): Maximum number of tracks to process. Defaults to 100.
        run_captions (bool, optional): Whether to generate captions for hazards. Defaults to False.

    Returns:
        list[tuple]: A list of tuples, each containing a hazard track and its caption.
    """
    hazards = []
    image_center = [frame_image.shape[1] / 2, frame_image.shape[0] / 2]
    if bbox_centers.shape[0] == 0:
        return [
            (" ", " "),
        ]
    potential_hazard_dists = np.linalg.norm(bbox_centers - image_center, axis=1)

    bbox_count = bbox_centers.shape[0]
    num_tracks = min(num_tracks, bbox_count)
    probable_hazard_indexes = np.argsort(potential_hazard_dists)[:num_tracks]

    for probable_hazard_idx in probable_hazard_indexes:
        hazard_track = track_ids[probable_hazard_idx]

        if hazard_track not in captioned_tracks:
            if run_captions:
                hazard_chip = cv2.cvtColor(
                    chips[probable_hazard_idx], cv2.COLOR_BGR2RGB
                )
                hazard_chip = Image.fromarray(hazard_chip)
                # Generate caption
                hazard_caption = ci.interrogate(hazard_chip)
            else:
                hazard_caption = " "

            hazard_caption = hazard_caption.replace(",", " ")

            captioned_tracks[hazard_track] = hazard_caption
        else:
            hazard_caption = captioned_tracks[hazard_track]

        hazards.append((hazard_track, hazard_caption))

    return hazards


def process_video(
    video_path: str,
    annotations: dict,
    run_tracks: bool = False,
    run_captions: bool = False,
):
    """
    Process a video to detect hazards and track driver state changes.

    Args:
        video_path (str): Path to the video file.
        annotations (dict): Dictionary containing annotations for the video.
        run_tracks (bool, optional): Whether to process tracks for hazards. Defaults to False.
        run_captions (bool, optional): Whether to generate captions for hazards. Defaults to False.

    Returns:
        list[dict]: A list of dictionaries containing results for each frame in the video.
    """
    video_name = Path(video_path).stem
    video_data = load_video_frames(video_path)
    bbox_sizes, bbox_centers_all = [], []
    captioned_tracks = {}
    video_results = []

    for id_frame, frame_image in video_data:
        frame_number = int(id_frame.split("_")[-1])
        bboxes, bbox_centers, chips, track_ids = load_frame_annotations(
            annotations, video_name, frame_number, frame_image
        )

        bbox_pixel_size = calculate_bbox_size_total(bboxes)
        bbox_sizes.append(bbox_pixel_size)
        bbox_centers_all.append(bbox_centers)

        hazard_tracks, hazard_captions = [], []
        if run_tracks:
            hazards = get_hazards(
                frame_image,
                bbox_centers,
                track_ids,
                chips,
                captioned_tracks,
                run_captions=run_captions,
            )
            for hazard_track, hazard_caption in hazards:
                hazard_tracks.append(hazard_track)
                hazard_captions.append(hazard_caption)

        video_results.append(
            {
                "id": id_frame,
                "bbox_size": bbox_pixel_size,
                "hazard_tracks": hazard_tracks,
                "hazard_captions": hazard_captions,
            }
        )

    driver_state_changed = bbox_size_drive_state_changed(bbox_sizes)
    # driver_state_changed = baseline_driver_state_changed(bbox_centers_all)
    # optical_flow = calculate_optical_flow(video_data)
    # driver_state_changed = optical_flow_driver_state_changed(optical_flow)
    assert len(video_results) == len(driver_state_changed)
    for video_result, driver_state in zip(video_results, driver_state_changed):
        video_result["driver_state"] = driver_state

    return video_results



def process_detections(
    detections: list[dict],
    allowed_classes: list[str],
):
    """
    Process a video to detect hazards and track driver state changes.

    Returns:
        list[dict]: A list of dictionaries containing results for each frame in the video.
    """
    bbox_sizes, bbox_centers_all = [], []

    for frame_detections in detections:
        frame_number = int(frame_detections["frame_number"])
        if not "bboxes" in frame_detections:
            if len(bbox_sizes) > 0:
                bbox_sizes.append(bbox_sizes[-1])
            else:
                bbox_sizes.append(0)
            continue

        cls_labels = frame_detections["labels"]
        if len(allowed_classes) > 0:
            filtered_indexes = [i for i, label in enumerate(cls_labels) if label in allowed_classes]
        else:
            filtered_indexes = list(range(len(cls_labels)))

        bboxes = frame_detections["bboxes"]
        bbox_centers = [[x1 + (abs(x2 - x1) / 2), y1 + (abs(y2 - y1) / 2)] for (x1, y1, x2, y2) in frame_detections["bboxes"]]
        track_ids = frame_detections["track_ids"]

        bboxes = np.array(bboxes)[filtered_indexes]
        bbox_centers = np.array(bbox_centers)[filtered_indexes]
        track_ids = np.array(track_ids)[filtered_indexes]

        bbox_pixel_size = calculate_bbox_size_total(bboxes)
        bbox_sizes.append(bbox_pixel_size)
        bbox_centers_all.append(bbox_centers)

    driver_state_changed = bbox_size_drive_state_changed(bbox_sizes)
    # driver_state_changed = baseline_driver_state_changed(bbox_centers_all)
    # optical_flow = calculate_optical_flow(video_data)
    # driver_state_changed = optical_flow_driver_state_changed(optical_flow)

    detected_hazard_frame = next((i for i, val in enumerate(driver_state_changed) if val == 1), None)

    results = {
        "bbox_sizes_frame": detected_hazard_frame
    }
    return results


def video_results_to_submission_format(video_results: list()) -> dict:
    """
    Convert video processing results into the required submission format.

    Args:
        video_results (list[dict]): List of dictionaries containing video processing results.

    Returns:
        dict: A dictionary formatted for submission.
    """
    submission_results = {}
    for results in video_results:
        id_frame = results["id"]
        driver_state = results["driver_state"]
        hazard_tracks = results["hazard_tracks"]
        hazard_captions = results["hazard_captions"]
        submission_results[id_frame] = {
            "Driver_State_Changed": driver_state,
        }

        for i, track, caption in zip_longest(
            range(23), hazard_tracks, hazard_captions, fillvalue=" "
        ):
            if i is None:
                print("Should not happen")
                continue
            track_col = f"Hazard_Track_{i}"
            caption_col = f"Hazard_Name_{i}"
            submission_results[id_frame][track_col] = track
            submission_results[id_frame][caption_col] = caption
    return submission_results
