from typing import Any, Dict, List, Set, Tuple

import cv2
import numpy as np

import carla

from .carlautils import rotation_to_dict, vector3d_to_dict


class BoundingBoxProcessor:
    """Generates 2D bboxes of specific carla objects based on the camera input and world status."""

    DETECTED_STATIC_CATEGORIES = {
        # 12: carla.CityObjectLabel.Pedestrians,
        # 13: carla.CityObjectLabel.Rider,
        14: carla.CityObjectLabel.Car,
        15: carla.CityObjectLabel.Truck,
        16: carla.CityObjectLabel.Bus,
        # 17: carla.CityObjectLabel.Train,
        18: carla.CityObjectLabel.Motorcycle,
        19: carla.CityObjectLabel.Bicycle,
        # 29: carla.CityObjectLabel.Van
    }

    RIDER_TAG = 13
    MOTORCYCLE_TAG = 18
    BICYCLE_TAG = 19
    DYNAMIC_TAG = 21  # For manually spawned props

    MIN_PIXELS_CHECK = {
        "lowered_tags": [12, RIDER_TAG, MOTORCYCLE_TAG, BICYCLE_TAG, DYNAMIC_TAG],
        "lowered_pixel_ratio": 0.35,
    }

    def __init__(
        self,
        seg_sensor: carla.Sensor,
        display_size: tuple,
        world: carla.World = None,
    ):
        self.seg_sensor = seg_sensor
        self.world = world
        self.display_size = display_size

    def get_segmented_2d_bboxes(
        self,
        spawned_actors: List[carla.Actor],
        segmented_image: np.ndarray,
        max_distance: int = 100,
        min_bbox_pixels: int = 128,
    ) -> List[Dict[str, Any]]:
        """Get tags, bboxes and contours of detected objects in the simulation scene.

        Has id, tag, 2d_bbox, 3d_bbox, and contour keys.
        """
        tagged_bboxes = self.get_all_3d_tagged_bboxes(spawned_actors, max_distance)

        for tagged_bbox in tagged_bboxes:
            tagged_bbox["2d_bbox"] = transform_bbox_to_2d(
                tagged_bbox["3d_bbox_proj"], self.display_size
            )
        # Filter 2d bboxes not present in the screen
        tagged_bboxes = [
            ann
            for ann in tagged_bboxes
            if (ann["2d_bbox"] >= 0).all()
            and (ann["2d_bbox"][:, 0] <= self.display_size[0]).all()
            and (ann["2d_bbox"][:, 1] <= self.display_size[1]).all()
        ]
        tagged_segmented_bboxes = self.filter_bboxes_by_segmentation(
            tagged_bboxes, segmented_image, min_bbox_pixels=min_bbox_pixels
        )
        return tagged_segmented_bboxes

    def get_all_3d_tagged_bboxes(
        self, spawned_actors: List[carla.Actor], max_distance: int
    ) -> List[Dict[str, Any]]:
        """Get tags and 3D bboxes of objects in the simulation scene."""
        tagged_bboxes = ClientSideBoundingBoxes.get_bounding_boxes(
            spawned_actors, self.seg_sensor, max_distance
        )

        world_bboxes = ClientSideBoundingBoxes.get_bounding_boxes_from_world(
            self.world, self.seg_sensor, self.DETECTED_STATIC_CATEGORIES, max_distance
        )
        tagged_bboxes.extend(world_bboxes)
        return tagged_bboxes

    def filter_bboxes_by_segmentation(
        self,
        tagged_bboxes: List[Dict[str, Any]],
        segmented_image: np.ndarray,
        min_bbox_pixels: int = 128,
    ) -> List[Dict[str, Any]]:
        """Filter only visible bboxes. Separate bboxes by pixels."""
        possible_tags = set([ann["tag"] for ann in tagged_bboxes])
        # Different pixel color, but single bbox instance
        if self.BICYCLE_TAG in possible_tags or self.MOTORCYCLE_TAG in possible_tags:
            possible_tags.add(self.RIDER_TAG)

        def _bbox_2d_size(bbox: np.ndarray) -> float:
            return np.prod(bbox[1, :] - bbox[0, :])

        # tag -> pixel mask
        pixel_masks = {
            tag: (segmented_image[:, :, 0] == tag).astype(np.uint8)
            for tag in possible_tags
        }

        tagged_bboxes = sorted(
            tagged_bboxes, key=lambda ann: _bbox_2d_size(ann["2d_bbox"]), reverse=True
        )

        filtered_tagged_contoured_bboxes = []
        used_unique_segmentation_ids = set()
        for ann in tagged_bboxes:
            tag, bbox = ann["tag"], ann["2d_bbox"]
            pixel_mask = pixel_masks.get(tag)
            if pixel_mask is None:
                continue
            # bbox pixels & pixel mask -> pixel mask
            binary_mask = get_instance_binary_mask(bbox, pixel_mask)

            # Solve overlapping pixel masks based on pixel size.
            (
                binary_mask,
                used_unique_segmentation_ids,
            ) = self.update_mask_by_pixel_segmentation_ids(
                binary_mask=binary_mask,
                segmented_image=segmented_image,
                used_ids=used_unique_segmentation_ids,
            )

            # Merge biker and bike masks
            if tag in [self.BICYCLE_TAG, self.MOTORCYCLE_TAG]:
                binary_mask = self._add_rider_mask(binary_mask, bbox, pixel_masks)

            _min_pixel_size = (
                min_bbox_pixels * self.MIN_PIXELS_CHECK["lowered_pixel_ratio"]
                if tag in self.MIN_PIXELS_CHECK["lowered_tags"]
                else min_bbox_pixels
            )
            # Filter by pixel size left
            if binary_mask.sum() > _min_pixel_size:
                contour = get_segmentation_contours(binary_mask)
                if not contour:
                    continue
                ann["contour"] = contour

                new_bbox = get_minimal_bbox(binary_mask)
                if _bbox_is_smaller_or_same_size(new_bbox, bbox):
                    ann.update({"2d_bbox": new_bbox})
                filtered_tagged_contoured_bboxes.append(ann)

        return filtered_tagged_contoured_bboxes

    def update_mask_by_pixel_segmentation_ids(
        self, binary_mask: np.ndarray, segmented_image: np.ndarray, used_ids: Set[int]
    ) -> tuple:
        """Separate pixels of overlapping bboxes based on the bbox size and pixel ids."""
        masked_pixels = segmented_image[binary_mask == 1]
        unique_segmentation_ids, counts = np.unique(
            masked_pixels[:, 1].astype(np.uint32) * 1000
            + masked_pixels[:, 2].astype(np.uint32),
            # Pixel id is divided into GB channels
            # Does not have to be precise
            return_counts=True,
        )

        unique_segmentation_ids = unique_segmentation_ids[np.argsort(counts)[::-1]]

        id_found = False
        for unique_segmentation_id in unique_segmentation_ids:
            if unique_segmentation_id not in used_ids:
                used_ids.add(unique_segmentation_id)
                G_pixel_id = unique_segmentation_id // 1000
                B_pixel_id = unique_segmentation_id % 1000

                mask = np.bitwise_and(
                    segmented_image[:, :, 1] == G_pixel_id,
                    segmented_image[:, :, 2] == B_pixel_id,
                )
                binary_mask = np.bitwise_and(mask, binary_mask)
                id_found = True
                break

        if not id_found:
            binary_mask = np.zeros(binary_mask.shape, dtype=np.uint8)

        return binary_mask, used_ids

    def _add_rider_mask(
        self, binary_mask: np.ndarray, bbox: np.ndarray, pixel_masks: dict
    ) -> np.ndarray:
        """Add rider pixel mask to bike/motorbike."""
        pixel_mask_rider = pixel_masks.get(self.RIDER_TAG)
        if pixel_mask_rider is not None:
            binary_mask_rider = get_instance_binary_mask(bbox, pixel_mask_rider)
            binary_mask = np.bitwise_or(binary_mask, binary_mask_rider)
        return binary_mask


def transform_bbox_to_2d(bbox_3d: np.ndarray, display_size: tuple) -> np.ndarray:
    """3D to 2D projection"""
    min_x = np.amin(bbox_3d[:, 0])
    min_y = np.amin(bbox_3d[:, 1])
    max_x = np.amax(bbox_3d[:, 0])
    max_y = np.amax(bbox_3d[:, 1])
    # bbox_2d = np.array([[int(min_x), int(min_y)], [int(max_x), int(max_y)]])

    bbox_2d = np.array(
        [
            [int(max(min_x, 0)), int(max(min_y, 0))],
            [
                int(min(max_x, display_size[0])),
                int(min(max_y, display_size[1])),
            ],
        ],
        dtype=np.int32,
    )
    return bbox_2d


def _bbox_is_smaller_or_same_size(new_bbox, old_bbox) -> bool:
    """Compare sizes"""
    return (
        new_bbox[0, 0] >= old_bbox[0, 0]
        or new_bbox[0, 1] >= old_bbox[0, 1]
        or new_bbox[1, 0] <= old_bbox[1, 0]
        or new_bbox[1, 1] <= old_bbox[1, 1]
    )


def get_instance_binary_mask(bbox: np.ndarray, pixel_mask: np.ndarray) -> np.ndarray:
    """Crop pixel mask by bbox."""
    bbox_mask = np.zeros(shape=pixel_mask.shape, dtype=np.uint8)
    bbox_mask[bbox[0, 1] : bbox[1, 1], bbox[0, 0] : bbox[1, 0]] = 1

    binary_mask = np.bitwise_and(bbox_mask, pixel_mask)

    return binary_mask


def get_segmentation_contours(binary_mask: np.ndarray) -> list:
    """Retrieve contour from a binary mask."""
    contours, hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    segmentation = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:  # Straight line
            segmentation.append(contour)
    return segmentation


def get_minimal_bbox(binary_mask: np.ndarray) -> np.ndarray:
    """Return left, top, right, bottom coordinates of binary mask."""
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return np.array([[xmin, ymin], [xmax, ymax]], dtype=np.int16)


# ==============================================================================
# -- ClientSideBoundingBoxes ---------------------------------------------------
# ==============================================================================


class ClientSideBoundingBoxes(object):
    # Copyright (c) 2019 Aptiv
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_bounding_boxes(
        actors: List[carla.Actor], camera: carla.Sensor, max_distance: int
    ) -> List[Dict[str, Any]]:
        """
        Retrieve 3D bboxes of spawned objects.
        """
        actors = filter_actors_by_location(actors, camera, max_distance)

        # TODO Props have no semantic tags, but color matches Dynamic Tag
        bounding_boxes = [
            {  # If missing, assume prop - Dynamic tag
                "id": actor.id,
                "tag": (
                    actor.semantic_tags[0]
                    if actor.semantic_tags
                    else BoundingBoxProcessor.DYNAMIC_TAG
                ),
                "3d_bbox_proj": ClientSideBoundingBoxes.get_bounding_box(actor, camera),
                # "3d_bbox_sim": ClientSideBoundingBoxes.get_3d_bbox_sim(
                #     actor.bounding_box
                # ),
                "location": vector3d_to_dict(actor.bounding_box.location),
                "extent": vector3d_to_dict(actor.bounding_box.extent),
                "rotation": rotation_to_dict(actor.bounding_box.rotation),
            }
            for actor in actors
        ]

        # Filter objects behind camera
        bounding_boxes = [
            ann for ann in bounding_boxes if all(ann["3d_bbox_proj"][:, 2] >= 0)
        ]

        return bounding_boxes

    @staticmethod
    def get_bounding_boxes_from_world(
        world: carla.World,
        camera: carla.Sensor,
        detected_static_categories: dict,
        max_distance: int,
    ) -> List[Dict[str, Any]]:
        """Retrieve bboxes of static world objects."""
        bounding_boxes = []
        for tag, carla_object_type in detected_static_categories.items():
            world_objects: List[carla.EnvironmentObject] = (
                world.get_environment_objects(carla_object_type)
            )
            world_objects = filter_actors_by_location(
                world_objects, camera, max_distance
            )

            static_objects = [
                {
                    "id": world_object.id,
                    "tag": tag,
                    "3d_bbox_proj": ClientSideBoundingBoxes.get_bounding_box_static(
                        world_object.bounding_box, camera
                    ),
                    # "3d_bbox_sim": ClientSideBoundingBoxes.get_3d_bbox_sim(
                    #     world_object.bounding_box
                    # ),
                    "location": vector3d_to_dict(world_object.bounding_box.location),
                    "extent": vector3d_to_dict(world_object.bounding_box.extent),
                    "rotation": rotation_to_dict(world_object.bounding_box.rotation),
                }
                for world_object in world_objects
            ]
            bounding_boxes.extend(static_objects)
        # Filter objects behind camera
        bounding_boxes = [
            ann for ann in bounding_boxes if all(ann["3d_bbox_proj"][:, 2] >= 0)
        ]
        return bounding_boxes

    @staticmethod
    def get_circumferential_bbox(
        bboxes_2d: List,
        display_size: Tuple[int, int],
    ) -> np.ndarray:
        """Get the maximal circumferential bounding box of multiple bboxes."""
        all_points = np.vstack(bboxes_2d)  # shape (N, 2)
        width, height = display_size
        x_min = int(np.clip(np.min(all_points[:, 0]), 0, width - 1))
        y_min = int(np.clip(np.min(all_points[:, 1]), 0, height - 1))
        x_max = int(np.clip(np.max(all_points[:, 0]), 0, width - 1))
        y_max = int(np.clip(np.max(all_points[:, 1]), 0, height - 1))
        return np.array([[x_min, y_min], [x_max, y_max]], dtype=int)

    @staticmethod
    def get_bounding_box(vehicle: carla.Actor, camera: carla.Sensor) -> np.ndarray:
        """Return 3D bounding box for a vehicle based on camera view."""
        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle.bounding_box)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(
            bb_cords, vehicle, camera
        )[:3, :]
        cords_y_minus_z_x = np.concatenate(
            [cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]]
        )
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate(
            [bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1
        )
        return camera_bbox

    @staticmethod
    def get_bounding_box_static(
        static_carla_object: carla.BoundingBox, camera: carla.Sensor
    ) -> np.ndarray:
        """
        Get 3d of static world object based on the camera view.
        """
        bb_cords = ClientSideBoundingBoxes._create_bb_points(static_carla_object)
        cords_x_y_z = ClientSideBoundingBoxes._static_to_sensor(
            bb_cords, static_carla_object, camera
        )[:3, :]
        cords_y_minus_z_x = np.concatenate(
            [cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]]
        )
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate(
            [bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1
        )
        return camera_bbox

    @staticmethod
    def _create_bb_points(bounding_box: carla.BoundingBox) -> np.ndarray:
        """
        Return 3D bounding box for a vehicle.
        """
        cords = np.zeros((8, 4))
        extent = bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def get_3d_bbox_sim(bounding_box: carla.BoundingBox) -> list:
        """Get list with 3D bounding box of an object in the simulation coordinates."""
        location = np.array(
            [bounding_box.location.x, bounding_box.location.y, bounding_box.location.z]
        )
        cords = ClientSideBoundingBoxes._create_bb_points(bounding_box)
        cords = cords[:, :3] + location
        return cords.tolist()

    @staticmethod
    def _vehicle_to_sensor(
        cords: np.ndarray, vehicle: carla.Actor, sensor: carla.Sensor
    ) -> np.ndarray:
        """Transform coordinates of a vehicle bounding box to sensor."""
        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _static_to_sensor(
        cords: np.ndarray, static_bbox: carla.BoundingBox, sensor: carla.Sensor
    ) -> np.ndarray:
        """
        Transform object's world coordinates to camera view.
        """
        bb_transform = carla.Transform(static_bbox.location, static_bbox.rotation)
        bb_world_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        world_cord = np.dot(bb_world_matrix, np.transpose(cords))
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords: np.ndarray, vehicle: carla.Actor) -> np.ndarray:
        """
        Transform coordinates of a vehicle bounding box to world.
        """
        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(
            vehicle.get_transform()
        )
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords: np.ndarray, sensor: carla.Sensor) -> np.ndarray:
        """
        Project world coordinates to sensor.
        """
        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform: carla.Transform) -> np.ndarray:
        """
        Create transformation matrix from carla transform object.
        """
        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix


def filter_actors_by_location(
    actors: List[carla.Actor], camera: carla.Sensor, max_distance: int
) -> List[carla.Actor]:
    """Remove bboxes behind sensor and beyond max distance."""
    filtered_actors = []
    sensor_transform = camera.get_transform()
    for actor in actors:
        if isinstance(actor, carla.EnvironmentObject):
            transform_location = actor.bounding_box.location
        else:
            transform_location = actor.get_transform().location
        distance = transform_location.distance(sensor_transform.location)

        if distance > max_distance:
            continue

        # Filter vehicles not in front
        forward_vec = sensor_transform.get_forward_vector()
        ray = transform_location - sensor_transform.location

        if forward_vec.dot(ray) > 1:
            filtered_actors.append(actor)

    return filtered_actors
