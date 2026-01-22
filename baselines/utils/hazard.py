import os
from itertools import zip_longest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def select_most_common(array, k):
    """
    Selects the top `k` most common elements from an array,
    prioritizing elements with higher counts and earlier occurrence.

    Args:
        array (np.ndarray): Input array containing elements.
        k (int): Number of top elements to return.

    Returns:
        list: A list of the `k` most common elements in the array.
    """
    unique, counts = np.unique(array, return_counts=True)
    first_indices = np.array([np.where(array == u)[0][0] for u in unique])
    sorted_indices = np.lexsort(
        (-counts, first_indices)
    )  # Negative counts for descending order
    top_k = unique[sorted_indices[:k]]
    return top_k.tolist()


def get_area(bbox):
    """
    Calculates the area of a bounding box.

    Args:
        bbox (tuple): A tuple containing the coordinates of the bounding box
                      in the format (x1, y1, x2, y2).

    Returns:
        float: The area of the bounding box.

    Raises:
        ValueError: If the bounding box dimensions are invalid (e.g., x2 <= x1 or y2 <= y1).
    """
    x1, y1, x2, y2 = bbox

    # Ensure valid bounding box
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid bounding box dimensions.")
    width = x2 - x1
    height = y2 - y1
    return width * height


class Hazard:
    """
    Represents a detected hazard in a video, including its associated track, frames, and captions.

    Attributes:
        video (str): The name of the video containing the hazard.
        track (int): The track ID associated with the hazard.
        frames (dict): A dictionary mapping frame IDs to their bounding box and additional data.
        caption_list (list): A list of captions generated for the hazard.
        caption_list_words (list): A list of words extracted from the captions.
    """

    def __init__(self, video, track, frames):
        """
        Initializes a Hazard object.

        Args:
            video (str): The name of the video containing the hazard.
            track (int): The track ID associated with the hazard.
            frames (dict): A dictionary mapping frame IDs to their bounding box and additional data.
        """
        self.video = video
        self.track = track
        self.frames = frames
        self.caption_list = []
        self.caption_list_words = []

    def __repr__(self):
        """
        Returns a string representation of the Hazard object.

        Returns:
            str: A string describing the hazard's video and track ID.
        """
        return f"Hazard {self.video} {self.track}"

    @property
    def caption(self):
        """
        Generates a combined caption for the hazard by concatenating all individual captions.

        Returns:
            str: A concatenated string of captions, or a single space if no captions exist.
        """
        if len(self.caption_list) == 0:
            return " "
        else:
            return " ".join(self.caption_list)

    @property
    def dangerous(self):
        """
        Determines whether the hazard is classified as dangerous based on its detected class.

        Returns:
            bool: True if the hazard is classified as dangerous, False otherwise.
        """
        return self.get_cifar_classes()[0] not in [
            "pickup_truck",
            "bus",
            "tank",
            "motorcycle",
            "cloud",
        ]
    @property
    def trajectory_check(self):
        """
        Determines whether the trajectory's bounding box area is sufficiently large.
    
        This method computes the bounding box area from the positions of objects in the 
        frames and evaluates if it exceeds a predefined threshold (2500). The bounding box 
        is defined as the smallest rectangle that encompasses all the positions.
    
        Returns:
            bool: True if the bounding box area is greater than 2500, indicating that 
            the area is large enough, otherwise False.
        """
        # Get the positions from the frames
        positions = [data['position'] for data in self.frames.values()]
        
        # Check if the bounding box area is large enough (greater than 10% of the image size)
        x_coords, y_coords = zip(*positions)
        bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        bbox_big_enough = bbox_area > 2500
        
        return bbox_big_enough

    def visualize(self, frame_idx=None, folder_path="./dataset/coool-benchmark/"):
        """
        Visualizes the hazard in a video frame by highlighting its bounding box and caption.

        Args:
            frame_idx (int, optional): The index of the frame to visualize. Defaults to the frame
                                       with the largest bounding box area.
            folder_path (str): The path to the folder containing the video files.

        Returns:
            np.ndarray: An image with the bounding box and caption overlaid.
        """
        if frame_idx is not None:
            frame_id = list(self.frames.keys())[frame_idx]
        else:
            frame_id = pd.DataFrame(self.frames).T["area"].idxmax()  # Largest by area

        cap = cv2.VideoCapture(f"{folder_path}/{self.video}.mp4")
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the target frame
            if frame_count == frame_id:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Draw bounding box
                x1, y1, x2, y2 = (
                    np.array(self.frames[frame_id]["bbox"]).round().astype(int)
                )
                image = cv2.rectangle(
                    image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2
                )

                # Add track_id text in the upper-left corner
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                font_thickness = 2
                text_size = cv2.getTextSize(
                    self.caption, font, font_scale, font_thickness
                )[0]

                # Calculate text position (upper-left corner)
                text_x = 10  # Fixed x-coordinate
                text_y = text_size[1] + 10  # Add a small offset from the top

                # Draw text background for better visibility (optional)
                text_bg_x2 = text_x + text_size[0]
                text_bg_y2 = text_y + 5  # Add a small padding below the text
                image = cv2.rectangle(
                    image,
                    (text_x - 5, text_y - text_size[1] - 5),
                    (text_bg_x2 + 5, text_bg_y2),
                    color=(0, 255, 0),
                    thickness=-1,
                )

                # Draw the text on the image
                image = cv2.putText(
                    image,
                    self.caption,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness=font_thickness,
                )
                break
            frame_count += 1
        cap.release()
        return image

    def get_cifar_classes(self):
        """
        Analyzes the hazard frames to determine the most likely CIFAR classes and their probabilities.

        Returns:
            tuple: The most probable CIFAR class and its corresponding probability.
        """
        df = pd.DataFrame(self.frames).T
        df_extended = pd.DataFrame(
            {
                "probs": np.concatenate(df["probs10"].values),
                "class": np.concatenate(df["class10"].values),
                "area": np.concatenate([np.repeat(v, 10) for v in df["area"]]),
            }
        )
        select = (
            df_extended.groupby("class")
            .apply(lambda g: (g["probs"] * g["area"]).mean())
            .sort_values()
        )
        return select.idxmax(), select.max()
