import glob
import os
import os.path as osp
import shutil
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from joblib import Parallel, delayed

from .ioutils import (generate_mp4_cv2, generate_mp4_ffmpeg, load_json, load_yaml, read_txt,
                      save_json, save_txt, save_yaml)


IMAGE_FORMATS = ["png", "jpg"]


def xyxy2xywh(bbox: np.ndarray):
    """Transform 2d bbox format."""
    x, y, width, height = (
        int(bbox[0, 0]),
        int(bbox[0, 1]),
        int(bbox[1, 0] - bbox[0, 0]),
        int(bbox[1, 1] - bbox[0, 1]),
    )
    return x, y, width, height


def normalize_bbox(bbox_xywh: tuple, display_size: tuple) -> tuple:
    """(image_center_x, image_center_y, width, height), normalized to <0, 1>."""
    display_x, display_y = display_size
    x, y, width, height = bbox_xywh

    x = (display_x - x) / display_x
    y = (display_y - y) / display_y
    width /= display_x
    height /= display_y

    return x, y, width, height


def denormalize_bbox(normalized_bbox_xywh: tuple, display_size: tuple):
    """Normalized (center_x, center_y, width, height) to (image_center_x, image_center_y, width, height)."""
    display_x, display_y = display_size
    _x, _y, _width, _height = normalized_bbox_xywh

    x = display_x - (_x * display_x)
    y = display_y - (_y * display_y)
    width = _width * display_x
    height = _height * display_y

    return int(x), int(y), int(width), int(height)


def convert(obj):
    """Convert numpy object to python object."""
    if isinstance(obj, np.ndarray) or isinstance(obj, np.matrix):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def to_serializable(d):
    """Convert numpy objects to python objects, recursively."""
    if isinstance(d, dict):
        return {k: to_serializable(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [to_serializable(i) for i in d]
    else:
        return convert(d)


class CarlaAnnotator(ABC):
    """Create annotations in different styles (YOLO, COCO)."""

    def __init__(
        self,
        display_size: tuple,
        exp_dir: str,
        config_path: str,
        save_segmentation: bool = True,
    ):
        self.display_size = display_size
        self.exp_dir = exp_dir
        self.save_segmentation = save_segmentation
        self.segmentation_tags = load_segmentation_tags(config_path)

        self.annotation_history = {}
        self.collision_history = {}
        self.sensor_history = {}

    @abstractmethod
    def add_to_annotations(
        self,
        file_name: str,
        tagged_segmented_bboxes: List[Dict[str, Any]],
    ):
        """Create annotations for a new image."""
        pass

    def export_annotations(self) -> None:
        """Save annotations to a file."""
        output_path = osp.join(self.exp_dir, "train_annotations.json")
        save_json(output_path, self.annotation_history)
        time.sleep(1)

    def add_collision_data(self, collision_data: List[Dict[str, Any]], timestamp: str):
        """Save collision bounding box and colliding objects ids."""
        self.collision_history[timestamp] = to_serializable(collision_data)

    def add_sensor_data(self, sensor_data: Dict[str, Any], timestamp: str):
        """Save camera sensor data (location, rotation)."""
        self.sensor_history[timestamp] = to_serializable(sensor_data)

    @abstractmethod
    def generate_video_version(self, image_format: str = "png"):
        """Generate video from saved annotations."""


class UltralyticsAnnotator(CarlaAnnotator):
    """https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format"""

    def __init__(
        self,
        display_size: tuple,
        exp_dir: str,
        config_path: str,
        save_segmentation: bool = True,
    ):
        super().__init__(display_size, exp_dir, config_path, save_segmentation)
        self.setup_exp_dir()

    def setup_exp_dir(self):
        """Prepare file structure and saves class mapping."""
        annotations_header_path = osp.join(self.exp_dir, "data.yaml")
        if not osp.exists(annotations_header_path):
            annotations_header = {
                "names": {tag: name for tag, name in self.segmentation_tags.items()}
            }
            save_yaml(annotations_header_path, annotations_header)

        for data_type_dir in ["images", "labels"]:
            for subdir_name in ["train", "val"]:
                os.makedirs(
                    osp.join(self.exp_dir, data_type_dir, subdir_name), exist_ok=True
                )

    def add_to_annotations(
        self,
        file_name: str,
        tagged_segmented_bboxes: List[Dict[str, Any]],
    ):
        """Move image to correct folder and creates corresponding annotation file."""
        image_annotation = []
        for ann in tagged_segmented_bboxes:
            tag, bbox = ann["tag"], ann["2d_bbox"]

            bbox_xywh = xyxy2xywh(bbox)
            normalized_bbox = normalize_bbox(bbox_xywh, self.display_size)
            bbox_annotation_list = [
                str(tag),
                *[str(value) for value in normalized_bbox],
            ]
            image_annotation.append(" ".join(bbox_annotation_list))

        label_target_path = osp.join(
            self.exp_dir, "labels", "train", f"{Path(file_name).stem}.txt"
        )
        save_txt(label_target_path, "\n".join(image_annotation))

        _, timestamp, iteration = file_name.split("_")
        iteration = int(iteration.split(".")[0])
        if timestamp not in self.annotation_history:
            self.annotation_history[timestamp] = []

        tagged_segmented_bboxes = deepcopy(tagged_segmented_bboxes)
        for ann in tagged_segmented_bboxes:
            ann.pop("3d_bbox_proj")
            ann.pop("contour")

        self.annotation_history[timestamp].append(
            {
                "iteration": iteration,
                "file_name": file_name,
                "objects": to_serializable(tagged_segmented_bboxes),
            }
        )

    def export_annotations(self):
        """Save annotations to a file."""
        assert len(self.annotation_history) == 1, "Should not happen"
        timestamp, annotations = self.annotation_history.popitem()

        output_path = osp.join(self.exp_dir, "labels", "train", f"{timestamp}.json")
        train_annotations = {
            "base": annotations,
            "collision": self.collision_history[timestamp],
            "sensor": self.sensor_history[timestamp],
        }
        save_json(output_path, train_annotations)
        time.sleep(1)
        # Clean cache for the next simulation
        self.annotation_history = {}
        self.collision_history = {}
        self.sensor_history = {}

    def generate_video_version(self, image_format: str = "png"):
        """Generate video from saved frame annotations."""
        transfer_scenario_to_video_ultralytics(
            scenario_dir=osp.dirname(self.exp_dir),
            image_format=image_format,
        )


class COCOAnnotator(CarlaAnnotator):
    """https://cocodataset.org/#format-data"""

    def __init__(
        self,
        display_size: tuple,
        exp_dir: str,
        config_path: str,
        annotation_template_path: str,
        save_segmentation: bool = True,
    ):
        super().__init__(display_size, exp_dir, config_path, save_segmentation)
        self.annotations = self.setup_annotation_schema(
            annotation_template_path, self.segmentation_tags
        )
        self.last_image_id = None
        self.last_annotation_id = None

    @staticmethod
    def setup_annotation_schema(
        annotation_template_path: str,
        segmentation_tags: dict,
    ) -> dict:
        """Load empty COCO-style annotation schema from a .json file."""
        annotations = load_json(annotation_template_path)

        current_datetime = datetime.now()
        info_created_timestamp = current_datetime.strftime("%Y/%m/%d")
        annotations["info"]["date_created"] = info_created_timestamp

        COCOAnnotator.update_annotation_categories(annotations, segmentation_tags)

        return annotations

    @staticmethod
    def update_annotation_categories(
        annotations: dict,
        segmentation_tags: dict,
    ):
        """Add categorical information to a dict."""
        categories = []
        for tag, name in segmentation_tags.items():
            categories.append({"id": tag, "name": name})
        annotations["categories"] = categories

    def add_to_annotations(
        self,
        file_name: str,
        tagged_segmented_bboxes: list,
    ):
        """Create new annotations for a single image."""
        if self.last_image_id is None:
            self.last_image_id = 0

        image_entry = {
            "id": self.last_image_id,
            "width": self.display_size[0],
            "height": self.display_size[1],
            "file_name": file_name,
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.annotations["images"].append(image_entry)

        bbox_annotations = self.get_bbox_annotations(
            self.last_image_id, tagged_segmented_bboxes
        )
        self.annotations["annotations"].extend(bbox_annotations)
        self.last_image_id += 1

    def get_bbox_annotations(
        self, image_id: int, tagged_segmented_bboxes: list
    ) -> list:
        """Retrieve bbox information and transfer it to COCO format."""
        if self.last_annotation_id is None:
            self.last_annotation_id = 0

        bbox_annotations = []
        for tag, bbox, contour in tagged_segmented_bboxes:
            x, y, width, height = xyxy2xywh(bbox)
            area = float(width * height)
            if not self.save_segmentation:
                contour = []

            bbox_annotation = {
                "id": self.last_annotation_id,
                "category_id": tag,
                "iscrowd": 0,
                "segmentation": contour,
                "area": area,
                "image_id": image_id,
                "bbox": [x, y, width, height],
            }
            bbox_annotations.append(bbox_annotation)
            self.last_annotation_id += 1
        return bbox_annotations

    def export_annotations(self):
        """Save internal annotations set to a file."""
        output_path = osp.join(self.exp_dir, "annotations.json")
        save_json(output_path, self.annotations)
        time.sleep(1)

    def export_collision_data(self, collision_data: list):
        raise NotImplementedError

    def generate_video_version(self, image_format: str = "png"):
        raise NotImplementedError


def load_segmentation_tags(config_path: str) -> Dict[str, str]:
    """Load Carla tag/class_name mapping from config."""
    config = load_json(config_path)
    segmentation_tags = {
        description["value"]: name for name, description in config.items()
    }
    return segmentation_tags


def get_scenario_groups(
    file_names: List[str], image_type: Tuple[str] = ("rgb", "display")
) -> Set[Tuple[str, str]]:
    """Group frame files to (type, timestamp) groups to differentiate them for videos."""
    scenario_groups = set()
    for file_name in file_names:
        file_name_fragments = osp.basename(file_name).split("_")
        _image_type, timestamp, frame = file_name_fragments
        if _image_type not in image_type:
            continue
        scenario_groups.add((_image_type, timestamp))
    return scenario_groups


def read_ultralytics_annotation_file(
    annotation_file: str,
    display_size: Tuple[int, int],
):
    """Read annotation file in Ultralytics format."""
    annotations = []
    annotation_lines = read_txt(annotation_file)
    for line in annotation_lines:
        line_data = line.strip().split(" ")
        normalized_bbox_xywh = tuple(float(value) for value in line_data[1:])
        bbox_xywh = denormalize_bbox(normalized_bbox_xywh, display_size=display_size)
        annotations.append({"class_id": int(line_data[0]), "bbox": bbox_xywh})
    return annotations


def aggregate_ultralytics_annotations(
    annotation_dir: str,
    display_size: Tuple[int, int],
) -> Dict[str, Dict[str, List[dict]]]:
    """Read multiple annotation files in Ultralytics format."""
    aggregated_annotations = defaultdict(dict)

    annotation_files = glob.glob(osp.join(annotation_dir, "*.txt"))
    annotation_names = [
        osp.basename(annotation_file) for annotation_file in annotation_files
    ]
    annotation_groups = get_scenario_groups(annotation_names)

    for image_type, timestamp in annotation_groups:
        group_name = f"{image_type}_{timestamp}"
        grouped_annotation_files = sorted(
            glob.glob(osp.join(annotation_dir, f"{group_name}_*.txt"))
        )

        for annotation_file in grouped_annotation_files:
            frame = osp.basename(annotation_file).split("_")[-1].split(".")[0]
            aggregated_annotations[group_name][frame] = (
                read_ultralytics_annotation_file(annotation_file, display_size)
            )

    return aggregated_annotations


def get_video_dir(
    scenario_dir: str,
) -> str:
    """Helper function to get video directory."""
    return osp.join(scenario_dir + "_video")


def transfer_scenario_to_video_ultralytics(
    scenario_dir: str,
    image_format: str = "png",
    frame_rate: float = None,
):
    """Produce video from a saved scenario annotations in Ultralytics format."""
    video_dir = get_video_dir(scenario_dir)
    os.makedirs(video_dir, exist_ok=True)

    exp_names = os.listdir(scenario_dir)
    for exp_subdir in exp_names:
        exp_dir = osp.join(scenario_dir, exp_subdir)
        if not osp.isdir(exp_dir):
            print(f"Experiment dir: {exp_dir} does not exist. Skipping")
            continue
        video_exp_dir = osp.join(video_dir, exp_subdir)
        process_experiment_to_video(
            scenario_dir=scenario_dir,
            exp_dir=exp_dir,
            video_exp_dir=video_exp_dir,
            frame_rate=frame_rate,
            image_format=image_format,
        )


def process_experiment_to_video(
    scenario_dir: str,
    exp_dir: str,
    video_exp_dir: str,
    frame_rate: float,
    image_format: str,
    remove_no_collisions: bool = True,
    remove_original_sim_data: bool = False,
):
    """Process scenario to video."""
    annotation_dir = osp.join(exp_dir, "labels", "train")
    image_dir = osp.join(exp_dir, "images", "train")

    video_image_dir = osp.join(video_exp_dir, "images", "train")
    video_annotation_dir = osp.join(video_exp_dir, "labels", "train")
    os.makedirs(video_image_dir, exist_ok=True)
    os.makedirs(video_annotation_dir, exist_ok=True)

    config = load_yaml(osp.join(exp_dir, f"{osp.basename(scenario_dir)}.yaml"))
    if frame_rate is None:
        frame_rate = (
                config["template"]["simulation_fps"]
                / config["template"]["frames_per_image"]
        )
    image_names = glob.glob(osp.join(image_dir, f"*.{image_format}"))
    image_groups = get_scenario_groups(image_names)
    results = Parallel(n_jobs=-1)(
        delayed(generate_mp4_ffmpeg)(
            image_dir,
            image_type=image_type,
            timestamp=timestamp,
            framerate=frame_rate,
            image_format=image_format,
            output_path=osp.join(video_image_dir, f"{image_type}_{timestamp}.mp4"),
        )
        for image_type, timestamp in image_groups
    )

    for annotations in glob.glob(osp.join(annotation_dir, "*.json")):
        shutil.copy(
            annotations,
            osp.join(video_annotation_dir, osp.basename(annotations)),
        )

    for config in glob.glob(osp.join(exp_dir, "*.yaml")):
        shutil.copy(
            config,
            osp.join(video_exp_dir, osp.basename(config)),
        )
    for config in glob.glob(osp.join(exp_dir, "*.json")):
        shutil.copy(
            config,
            osp.join(video_exp_dir, osp.basename(config)),
        )

    if remove_no_collisions:
        print("Removing collisions")
        remove_no_collision_sims(video_exp_dir=video_exp_dir)
    if remove_original_sim_data:
        print("removing original simulation data")
        remove_simulated_images(image_dir=image_dir)


def remove_no_collision_sims(
    video_exp_dir: str,
):
    """Remove experiments without collision data."""
    video_image_dir = osp.join(video_exp_dir, "images", "train")
    video_annotation_dir = osp.join(video_exp_dir, "labels", "train")

    for annotation_path in glob.glob(osp.join(video_annotation_dir, "*.json")):
        annotation_data = load_json(annotation_path)
        collision_data = annotation_data["collision"]
        if len(collision_data) > 0:
            print(f"Collision data found for: {annotation_path}")
            continue

        file_timestamp = Path(annotation_path).stem

        annotation_video_files = []
        for video_path in glob.glob(osp.join(video_image_dir, "*.mp4")):
            if file_timestamp in video_path:
                annotation_video_files.append(video_path)

        if len(annotation_video_files) > 2:
            print(f"More than 2 videos found for: {annotation_path}. Should not happen. Skipping")
            continue

        for video_path in annotation_video_files:
            print(f"Removing video with no collision: {video_path}")
            os.remove(video_path)
        print(f"Removing annotation with no collision: {annotation_path}")
        os.remove(annotation_path)


def remove_simulated_images(
    image_dir: str,
):
    """Removed saved frames from simulation."""
    for path in os.listdir(image_dir):
        if any(path.endswith(suffix) for suffix in IMAGE_FORMATS):
            os.remove(osp.join(image_dir, path))
