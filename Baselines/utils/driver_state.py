import os

import cv2
import numpy as np
import pandas as pd
import ruptures as rpt  # our package
from sklearn.linear_model import LinearRegression


def determine_drive_state_changed(timeseries: list, n_bkps: int = 3) -> list:
    """
    Detects changes in driver state based on a given timeseries using kernel change point detection.

    This function normalizes the input timeseries, applies the Kernel Change Point Detection (CPD)
    algorithm to identify potential change points, and returns a list indicating whether a driver
    state change occurred at each time step.

    Parameters:
    ----------
    timeseries : list
        A list of numerical values representing the input timeseries data.
    n_bkps : int, optional
        The number of breakpoints to detect in the timeseries (default is 3).

    Returns:
    -------
    list
        A list of boolean values of the same length as `timeseries`. Each element is `True` if a
        driver state change is detected at that time step, otherwise `False`.

    Raises:
    ------
    AssertionError
        If the length of the output list `driver_state_changed` does not match the input `timeseries`.

    Notes:
    -----
    - This function relies on the `ruptures` library for detecting change points.
    - The timeseries is normalized using its L2 norm before processing.

    Example:
    -------
    >>> determine_drive_state_changed([1, 2, 3, 8, 9, 10], n_bkps=2)
    [False, False, True, True, True, True]
    """
    timeseries = np.array(timeseries)
    timeseries = timeseries / np.linalg.norm(timeseries)
    algo = rpt.KernelCPD(kernel="rbf").fit(timeseries)

    # Detect breakpoints
    breakpoints = algo.predict(n_bkps=n_bkps)
    driver_state_changed = [
        any(i >= bp for bp in breakpoints) for i in range(len(timeseries))
    ]
    assert len(driver_state_changed) == len(timeseries)
    return driver_state_changed


def calculate_bbox_size_total(bboxes: np.ndarray):
    """
    Calculate the total size of all bounding boxes.

    Args:
        bboxes (np.ndarray[float]): An array of bounding boxes where each bbox is represented as [x1, y1, x2, y2].

    Returns:
        float: Total size (area) of all bounding boxes.
    """
    size = 0
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        size += abs(x2 - x1) * abs(y2 - y1)
    return size


def calculate_optical_flow(
    video_data: list,
    motion_threshold: float = None,
):
    """
    Computes the degree of motion between the current and previous frames.

    Args:
        video_data (list): List of tuples containing frame IDs and frame images.
        motion_threshold (float, optional): Threshold to filter out low-motion noise. Defaults to None.

    Returns:
        list[float]: A list of motion scores for each frame.
    """
    optical_flow = []
    previous_frame = None
    for i, (id_frame, frame_image) in enumerate(video_data):
        frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
        if previous_frame is None:
            previous_frame = frame_image

        flow = cv2.calcOpticalFlowFarneback(
            prev=previous_frame,
            next=frame_image,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

        if motion_threshold is not None:
            significant_motion = magnitude[magnitude > motion_threshold]
        else:
            significant_motion = magnitude

        motion_score = significant_motion.mean() if significant_motion.size > 0 else 0.0
        optical_flow.append(motion_score)
        previous_frame = frame_image

    optical_flow.insert(0, optical_flow[0])  # Copy to the first frame
    return optical_flow


def optical_flow_driver_state_changed(optical_flow: list()):
    """
    Determines whether the driver's state has changed based on optical flow metrics.

    Args:
        optical_flow (list[float]): List of optical flow values for each frame.

    Returns:
        list[bool]: A list indicating whether the driver's state has changed for each frame.
    """
    return bbox_size_drive_state_changed(optical_flow)


def bbox_size_drive_state_changed(bbox_sizes: list):
    """
    Detects changes in driver state based on bounding box sizes.

    Args:
        bbox_sizes (list): List of bounding box sizes for each frame.

    Returns:
        list: A list indicating whether the driver's state has changed for each frame.
    """
    n_bkps = 3
    bbox_sizes = np.array(bbox_sizes)
    bbox_sizes = bbox_sizes / np.sum(bbox_sizes)

    algo = rpt.KernelCPD(kernel="rbf").fit(bbox_sizes)
    breakpoints = algo.predict(n_bkps=n_bkps)

    driver_state_changed = [i >= breakpoints[0] for i in range(bbox_sizes.shape[0])]
    assert len(driver_state_changed) == len(bbox_sizes)
    return driver_state_changed


def baseline_driver_state_changed(bbox_centers_all: list) -> np.ndarray:
    """
    Predicts driver state change using the median distances between bounding box centers.

    Args:
        bbox_centers_all (list): List of bounding box center coordinates for each frame.

    Returns:
        np.ndarray: Array indicating whether the driver's state has changed for each frame.
    """
    previous_bbox_centers = []
    median_dists = []

    driver_state_changed = [False for _ in range(len(bbox_centers_all))]
    driver_state_changed = np.array(driver_state_changed)

    for i, bbox_centers in enumerate(bbox_centers_all):
        if len(bbox_centers) == 0 or len(previous_bbox_centers) == 0:
            if len(bbox_centers) != 0:
                previous_bbox_centers.append(bbox_centers)
            continue

        dists = []
        for bbox_center in bbox_centers:
            potential_dists = np.linalg.norm(
                previous_bbox_centers - bbox_center, axis=1
            )
            min_dist = np.sort(potential_dists)[0]
            dists.append(min_dist)

        median_dist = np.median(dists)
        median_dists.append(median_dist)
        if len(median_dists) == 1:
            continue

        x = np.array(range(len(median_dists))).reshape(-1, 1)
        y = np.array(median_dists)
        speed_model = LinearRegression().fit(x, y)

        if speed_model.coef_[0] < 0:  # If slowing down, driver's state likely changed
            driver_state_changed[i:] = True
            break
    return driver_state_changed
