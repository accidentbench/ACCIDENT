import datetime
import logging
import os
import os.path as osp
import random
import time
import traceback
from typing import Any, Dict, List, Tuple, Union

import numpy as np

import carla
from scipy.optimize import Bounds

from .actors import (add_controller_to_pedestrian, spawn_pedestrian,
                     spawn_vehicle)
from .bbox_segmentation import BoundingBoxProcessor
from .carla_annotator import CarlaAnnotator, UltralyticsAnnotator, load_segmentation_tags
from .carlautils import rotation_to_dict, set_spector_view, vector3d_to_dict, get_ordered_spawn_points
from .collisions import CollisionsEvaluator
from .gui_window import CarlaGUI, should_quit
from .hooks import HookRegistry
from .log import setup_logging
from .sensors import (convert_raw_sensor_data, process_lidar_data, save_point_cloud_to_ply,
                      setup_lidar, setup_sensor)
from .synchronous_mode import CarlaSynchronizer

START_RECORDING_FRAME = int(os.environ["START_RECORDING_FRAME"])
IMAGE_FORMAT = os.environ["IMAGE_FORMAT"]
CONFIG_PATH = os.environ["CLASS_MAPPING_PATH"]

setup_logging("./logging.yaml")
logger = logging.getLogger("synthesizer")


class CarlaSynthesizer:

    def __init__(
        self,
        client: carla.Client,
        display_size: Tuple[int, int],
        exp_output_dir: str,
        camera_fov: int = 90,
        max_distance: int = None,
        use_lidar: bool = False,
        simulation_fps: int = 20,
        frames_per_image: int = 120,
        draw_data: bool = True,
        **kwargs,
    ):
        """Controls the flow of CARLA simulation to generate data based on the predefined scenario.

        First, the mode is changed to synchronous and the vehicles, pedestrians, and sensors are spawned.
        Simulation starts, hooks are called, data from sensors are retrieved, processed (mainly bounding boxes),
        plotted and saved to annotations. Finally, the simulation is stopped after specified time, cleared, and the mode
        is set back to asynchronous mode.

        Args:
            client: CARLA client to communicate with CARLA API.
            display_size: Image resolution in pixels.
            exp_output_dir: Experiment output directory to store generated data.
            camera_fov: Field of view of the camera in degrees.
            max_distance: Maximum distance from camera to recognize bounding boxes of actors.
            use_lidar: Generate also Lidar data or not.
            simulation_fps: Simulated fps.
            frames_per_image: [1-inf] Number of frames between captured images.
            draw_data: Plot the annotations to the image and capture it.
            **kwargs:
        """
        self.client = client
        self.display_size = display_size
        self.camera_fov = camera_fov
        self.exp_output_dir = exp_output_dir
        self.use_lidar = use_lidar
        self.simulation_fps = simulation_fps
        self.frames_per_image = frames_per_image
        self.draw_data = draw_data
        self.max_distance = max_distance
        self.collisions_ego_only = kwargs.get("collisions_ego_only", False)

        self.timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%s")

        self.world: carla.World = self.client.get_world()

        self.map: carla.Map = self.world.get_map()
        self.blueprint_library: carla.BlueprintLibrary = (
            self.world.get_blueprint_library()
        )
        self.traffic_manager: carla.TrafficManager = self.client.get_trafficmanager(
            8000
        )
        self.synchronizer = CarlaSynchronizer(
            self.world, self.traffic_manager, fps=self.simulation_fps
        )

        self.spawned_vehicles = []
        self.spawned_pedestrians = []
        self.spawned_props = []

        self.sensors = {}

        self.gui = None

        self.camera_transform = None
        self.calibration = None

        self.hook_registry = HookRegistry()

        self.extra_kws = kwargs
        self.change_line_percentage = kwargs.get("change_line_percentage", 0.0)

    def setup(
        self,
        camera_transform: carla.Transform,
        weather: carla.WeatherParameters,
        number_of_vehicles: int,
        number_of_pedestrians: int,
        omit_spawn_points: List[int] = None,
        omit_bikes: bool = False,
        static_camera: bool = False,
    ) -> None:
        """Setup carla environment, sync mode, actors, sensors, etc.

        Args:
            camera_transform: Location of camera, lidar, etc.
            weather: Weather preset.
            number_of_vehicles: Number of vehicles to try to spawn.
            number_of_pedestrians: Number of pedestrians to try to spawn.
            omit_spawn_points: Excluded spawn points to make them unoccupied.
            omit_bikes: Exclude 2-wheel vehicles.
            static_camera: Use static camera, else attach it to an ego vehicle.

        """
        self.camera_transform = camera_transform
        self.synchronizer.enable_sync_mode()

        if self.draw_data:
            segmentation_tags = load_segmentation_tags(CONFIG_PATH)
            self.gui = CarlaGUI(self.display_size, segmentation_tags)

        self.world.set_weather(weather)

        main_vehicle = None
        if not static_camera:
            main_vehicle = self.setup_main_actor()
        self.setup_sensors(camera_transform, main_vehicle)
        self.synchronizer.create_sensor_queues(self.sensors)

        self.setup_vehicles(number_of_vehicles, omit_spawn_points, omit_bikes)

        self.setup_pedestrians(number_of_pedestrians)

        change_auto_lane = self.change_line_percentage > 0
        for actor in self.spawned_vehicles:
            actor.set_autopilot(True, self.traffic_manager.get_port())
            self.traffic_manager.update_vehicle_lights(actor, True)
            self.traffic_manager.auto_lane_change(actor, change_auto_lane)

    def setup_main_actor(
        self, car_bp_id: str = "vehicle.lincoln.mkz_2020"
    ) -> carla.Actor:
        """Select and creates ego actor to which the sensors are attached.

        Args:
            car_bp_id: Optional blueprint id for ego actor.

        Returns:
            Ego actor.

        """
        start_pose = random.choice(self.map.get_spawn_points())
        main_bp = self.blueprint_library.find(car_bp_id)

        main_bp.set_attribute("role_name", "hero")

        vehicle = self.world.spawn_actor(main_bp, start_pose)
        self.spawned_vehicles.append(vehicle)

        return vehicle

    def setup_vehicles(self, number_of_vehicles: int, omit_spawn_points: List[int], omit_bikes: bool) -> None:
        """Generate vehicles at random map spawn points. Won't be generated if in collision.

        Args:
            number_of_vehicles: Number of vehicles to try to spawn.
            omit_spawn_points: Excluded spawn points.
            omit_bikes: Use 2-wheel blueprints also or not.

        """
        spawn_points = get_ordered_spawn_points(self.world)
        if omit_spawn_points is not None:
            spawn_points = [sp for i, sp in enumerate(spawn_points) if i not in omit_spawn_points]

        number_of_vehicles = min(number_of_vehicles, len(spawn_points))
        logger.info(f"Spawning {number_of_vehicles} vehicles")
        spawn_points = random.sample(spawn_points, number_of_vehicles)
        for spawn_point in spawn_points:
            vehicle = spawn_vehicle(
                self.world,
                self.traffic_manager,
                transform=spawn_point,
                omit_bikes=omit_bikes,
                change_line_percentage=self.change_line_percentage,
            )
            if vehicle is not None:
                self.spawned_vehicles.append(vehicle)

    def setup_pedestrians(self, number_of_pedestrians: int = 20) -> None:
        """Try to spawn pedestrians and assign AI controllers to them.

        Args:
            number_of_pedestrians: Number of pedestrians to try to spawn.

        """
        for _ in range(number_of_pedestrians):
            pedestrian = spawn_pedestrian(self.world)
            if pedestrian is not None:
                add_controller_to_pedestrian(self.world, pedestrian)
                self.spawned_pedestrians.append(pedestrian)

    def setup_sensors(
        self, sensor_transform: carla.Transform, vehicle: carla.Actor = None
    ) -> None:
        """Create RGB, segmentation (for annotations), and optionally lidar sensors.

        Args:
            sensor_transform: Location of the sensors.
            vehicle: Optional, if not None attach sensors to the vehicle.

        """
        camera_rgb_blueprint = self.blueprint_library.find("sensor.camera.rgb")
        self.sensors["camera_rgb"] = setup_sensor(
            self.world,
            camera_rgb_blueprint,
            sensor_transform,
            vehicle,
            self.display_size,
            self.camera_fov,
        )

        camera_instance_seg_blueprint = self.blueprint_library.find(
            "sensor.camera.instance_segmentation"
        )
        self.sensors["camera_instance_seg"] = setup_sensor(
            self.world,
            camera_instance_seg_blueprint,
            sensor_transform,
            vehicle,
            self.display_size,
            self.camera_fov,
        )

        if self.use_lidar:
            self.sensors["lidar_sensor"] = setup_lidar(
                self.world,
                sensor_transform,
                vehicle,
                self.camera_fov,
                sensor_range=self.max_distance,
                rotation_frequency=self.simulation_fps,
            )
        set_spector_view(self.world, target_transform=sensor_transform)

    def run(
        self,
        runtime_secs: int,
        annotator: CarlaAnnotator,
        min_bbox_pixels: int = 256,
    ) -> None:
        """Run synchronized simulation, capture data from sensors, process them and produce a synthetic dataset.

        Clears the simulation afterward.

        Args:
            runtime_secs: How long will the simulation run in real life (depends on HW).
            annotator: Used to save data in specific format.
            min_bbox_pixels: Minimum pixel size of bounding boxes to be recognized/included in annotations.

        """
        if isinstance(annotator, UltralyticsAnnotator):
            self.exp_output_dir = osp.join(self.exp_output_dir, "images", "train")
        if self.use_lidar:
            os.makedirs(osp.join(self.exp_output_dir, "lidar"), exist_ok=True)

        bbox_processor = BoundingBoxProcessor(
            self.sensors["camera_instance_seg"],
            self.display_size,
            self.world,
        )

        collision_evaluator = CollisionsEvaluator(
            self.sensors["camera_instance_seg"], self.display_size, expanding_mode=True, ego_only=self.collisions_ego_only
        )

        iteration = 0
        try:
            start_time = time.time()
            while time.time() - start_time < runtime_secs:
                # Advance the simulation and wait for the data.
                logger.debug(
                    f"Iteration: {iteration:05d}, Time ({time.time() - start_time:.02f}:"
                    f"{runtime_secs:.02f}) secs"
                )
                iteration += 1
                _ = self.call_hook_registry(iteration)
                data, event_data = self.synchronizer.tick(timeout=5.0)

                collision_events = [
                    event for _, event in event_data.items() if event is not None
                ]
                collision_evaluator.add_events(collision_events, self.traffic_manager)
                collision_evaluator.evaluate_collision_event(iteration)

                if (
                    iteration % self.frames_per_image != 0
                    or iteration < START_RECORDING_FRAME
                ):
                    logger.debug("Skipping current frame.")
                    continue

                snapshot, image_rgb, image_instance_seg = (
                    data["snapshot"],
                    data["camera_rgb"],
                    data["camera_instance_seg"],
                )
                lidar_data = data.get("lidar", None)
                if lidar_data is not None:
                    self.process_lidar_measurements(lidar_data, iteration)
                tagged_segmented_bboxes = bbox_processor.get_segmented_2d_bboxes(
                    [
                        *self.spawned_vehicles,
                        *self.spawned_pedestrians,
                        *self.spawned_props,
                    ],
                    convert_raw_sensor_data(image_instance_seg),
                    max_distance=self.max_distance,
                    min_bbox_pixels=min_bbox_pixels,
                )
                if len(tagged_segmented_bboxes) == 0:
                    continue

                rgb_image_name = f"rgb_{self.timestamp}_{iteration:06d}.{IMAGE_FORMAT}"
                image_rgb.save_to_disk(osp.join(self.exp_output_dir, rgb_image_name))
                annotator.add_to_annotations(
                    file_name=rgb_image_name,
                    tagged_segmented_bboxes=tagged_segmented_bboxes,
                )
                if self.draw_data:
                    self.process_draw_data(
                        iteration,
                        snapshot,
                        image_rgb,
                        image_instance_seg,
                        tagged_segmented_bboxes,
                        lidar_data,
                        collision_evaluator.get_collision_bbox(),
                    )
                    if should_quit():
                        break

        except Exception as e:
            logger.warning(
                f"Iteration: {iteration}, Error: {e}, {traceback.format_exc()}"
            )
        annotator.add_collision_data(
            collision_evaluator.collision_history, timestamp=self.timestamp
        )
        annotator.add_sensor_data(
            {
                "location": vector3d_to_dict(self.camera_transform.location),
                "rotation": rotation_to_dict(self.camera_transform.rotation),
            },
            timestamp=self.timestamp,
        )
        annotator.export_annotations()
        time.sleep(5)
        self.clean_synthesizer()
        self.synchronizer.tick(-1)
        self.synchronizer.disable_sync_mode()
        self.client.reload_world()
        logger.info("Synthesizer is finished.")

    def clean_synthesizer(self) -> None:
        """Removes actors and sensors in CARLA simulation and quits the pygame gui."""
        logger.info("Destroying actors and sensors.")
        for actor in (
            self.spawned_vehicles + self.spawned_pedestrians + self.spawned_props
        ):
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception as e:
                logger.warning(f"Destroying actor: {e}")

        for sensor in self.sensors.values():
            try:
                sensor.destroy()
            except Exception as e:
                logger.warning(f"Destroying sensor: {e}")

        if self.gui is not None:
            self.gui.quit()

    def process_lidar_measurements(self, lidar_data: carla.SensorData, iteration: int) -> None:
        """Process raw lidar sensor data and save them as a point cloud to ply file.

        Args:
            lidar_data: Lidar sensor data.
            iteration: current simulation frame.

        """
        lidar_data, pcd = process_lidar_data(
            lidar_data,
            self.sensors["lidar_sensor"],
            self.sensors["camera_rgb"],
        )
        save_point_cloud_to_ply(
            osp.join(
                self.exp_output_dir,
                "lidar",
                f"lidar_{self.timestamp}_{iteration:06d}.ply",
            ),
            pcd,
        )

    def process_draw_data(
        self,
        iteration: int,
        snapshot: carla.WorldSnapshot,
        image_rgb: carla.SensorData,
        image_instance_seg: carla.SensorData,
        tagged_segmented_bboxes: List[Dict[str, Any]] = None,
        lidar_data: np.ndarray = None,
        collision_bbox: np.ndarray = None,
    ) -> None:
        """Draw segmentation annotations (optionally lidar) on top of the RGB image, save them and display them in GUI.

        Args:
            iteration: Current simulation frame.
            snapshot: Simulation snapshot - has extra metadata from the current frame.
            image_rgb: Captured RGB image.
            image_instance_seg: Captured instance segmentation image.
            tagged_segmented_bboxes: Processed classified bounding boxes, with contours.
            lidar_data: Optional lidar data.
            collision_bbox: Optional bboxes encompassing collided actors.

        """
        image_instance_seg.convert(carla.ColorConverter.CityScapesPalette)
        rgb_image = convert_raw_sensor_data(image_rgb)
        segmented_image = convert_raw_sensor_data(image_instance_seg)
        output_path = osp.join(
            self.exp_output_dir,
            f"display_{self.timestamp}_{iteration:06}.{IMAGE_FORMAT}",
        )
        self.gui.run_draw(
            rgb_image,
            segmented_image,
            snapshot,
            tagged_segmented_bboxes,
            lidar_points=lidar_data,
            collision_bbox=collision_bbox,
        )
        self.gui.save_display(output_path)

    def register_simulation_hook(self, hook_definition: Dict[str, Any]) -> None:
        """Wrapper function that registers arbitrary hooks.

        These hooks are useful to define custom actor behaviour, such as spawn a specific actor in a specific location
        and give them custom controls. Hooks will be activated on the defined frame number and removed.

        Args:
            hook_definition: Different hooks have different definitions.

        """
        self.hook_registry.register_carla_hook(
            hook_definition,
            self.world,
            self.traffic_manager,
            self.spawned_vehicles,
            self.spawned_pedestrians,
            self.spawned_props,
            self.sensors,
            self.synchronizer,
        )

    def call_hook_registry(self, iteration: int) -> List[Any]:
        """Invoke registered hooks based on the frame number.

        Args:
            iteration: Current simulation frame.

        Returns:
            List of hooks outputs.

        """
        return self.hook_registry.invoke_hooks_by_frame(iteration)
