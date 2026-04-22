import re
import glob
import json
import logging
import os
import os.path as osp
import shutil
import subprocess
from typing import List, Tuple, Any, Dict

import gzip
import cv2
import yaml

logger = logging.getLogger("synthesizer")


def load_json(path: str) -> Any:
    """Load a json file from path."""
    with open(path, "r") as fp:
        return json.load(fp)


def save_json(path: str, content: dict, use_gzip: bool = True) -> None:
    """Save a json serializable dict to a file. Create dirs in the path if not already present."""
    dir_path = osp.dirname(path)
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    if use_gzip:
        with gzip.open(path, 'wt', encoding="ascii") as zipfile:
            json.dump(content, zipfile)
    else:
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


def read_txt(path: str) -> List[str]:
    """Read a txt file from path as a list of lines."""
    with open(path, "r") as fp:
        return fp.readlines()


def save_txt(path: str, content: str) -> None:
    """Save a str content to a txt file."""
    with open(path, "w") as fp:
        fp.write(content)


def remove_source(source: str, files_only: bool = True) -> None:
    """Remove all or only files in the source path."""
    if files_only:
        for file_name in os.listdir(source):
            if osp.isdir(osp.join(source, file_name)):
                shutil.rmtree(osp.join(source, file_name))
                continue

            os.remove(osp.join(source, file_name))
    else:
        shutil.rmtree(source)


def generate_mp4_cv2(
    image_dir: str,
    image_type: str,
    timestamp: str,
    framerate: float,
    image_format: str,
    output_path: str = None,
):
    """Generate an MP4 video from a sequence of saved images using OpenCV.

    Args:
        image_dir:
        image_type:
        timestamp:
        framerate:
        image_format:
        output_path:

    Returns:

    """
    if framerate < 1:
        logger.debug("Framerate is less than 1. Skipping video generation.")
        return

    image_pattern = os.path.join(
        image_dir, f"{image_type}_{timestamp}_*.{image_format}"
    )
    output_video = output_path or os.path.join(
        image_dir, f"{image_type}_{timestamp}.mp4"
    )

    image_files = sorted(glob.glob(image_pattern))
    image_files = [osp.basename(image_file) for image_file in image_files]
    if not image_files:
        logger.warning(f"No images found matching pattern {image_pattern}")
        return

    first_image_path = os.path.join(image_dir, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video, fourcc, framerate, (width, height))

    try:
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            frame = cv2.imread(image_path)
            if frame is None:
                logger.warning(f"Skipping unreadable image: {image_path}")
                continue
            video_writer.write(frame)
    except Exception as e:
        logger.error(f"Error during video creation: {e}")
    finally:
        video_writer.release()


def generate_mp4_ffmpeg(
    image_dir: str,
    image_type: str,
    timestamp: str,
    framerate: float,
    image_format: str,
    output_path: str = None,
) -> None | str:
    """Generate MP4 videos (H.264 encoded) from saved images using ffmpeg.

    If successful, return video path, otherwise return None.
    """
    if framerate < 1:
        logger.debug("Framerate smaller than 1. Skipping video generation.")
        return None
    output_path = output_path or os.path.join(
        image_dir, f"{image_type}_{timestamp}.mp4"
    )
    if osp.exists(output_path):
        logger.debug(f"Output file already exists: {output_path}")
        return None

    # Check start number
    pattern = osp.join(image_dir, f"{image_type}_{timestamp}_*.{image_format}")
    image_files = sorted(glob.glob(pattern))
    if not image_files:
        logger.warning(f"No image files found with pattern: {pattern}")
        return None
    # Extract start number from filename
    match = re.search(rf"{image_type}_{timestamp}_(\d+)\.{image_format}", image_files[0])
    if not match:
        logger.warning(f"Could not extract frame number from file: {image_files[0]}")
        return None
    start_number = int(match.group(1))

    input_pattern = osp.join(
        image_dir, f"{image_type}_{timestamp}_%06d.{image_format}"
    )

    cmd = [
        "ffmpeg",
        "-start_number", str(start_number),
        "-framerate", str(framerate),
        "-i", input_pattern,
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-y",  # Overwrite output
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Video written to {output_path}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"ffmpeg failed for {image_type}: {e}")
        return None

    return output_path
