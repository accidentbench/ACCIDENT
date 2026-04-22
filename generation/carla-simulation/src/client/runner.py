import os
import os.path as osp
import json
import logging
import multiprocessing
import subprocess
import time
import psutil
import carla


from typing import List, Tuple, Dict, Any
from multiprocessing import Queue

from scenario import ScenarioTemplate
from core.carla_annotator import (CarlaAnnotator, COCOAnnotator,
                                  UltralyticsAnnotator, get_video_dir, process_experiment_to_video)
from core.log import setup_logging
from core.synthesizer import CarlaSynthesizer
from utils.error_handling import timeout_with_retry


setup_logging("./logging.yaml")
logger = logging.getLogger("runner")

USE_DOCKER = os.environ["USE_DOCKER"].lower() == "true"
CLASS_MAPPING_PATH = os.environ["CLASS_MAPPING_PATH"]
COCO_ANNOTATION_TEMPLATE_PATH = os.environ["COCO_ANNOTATION_TEMPLATE_PATH"]
START_CARLA_TIME_SEC = float(os.environ["START_RECORDING_FRAME"])
CARLA_HOST_NAME = os.environ["CARLA_HOST_NAME"]
CARLA_PORT = int(os.environ["CARLA_PORT"])
IMAGE_FORMAT = os.environ["IMAGE_FORMAT"]

PRODUCE_VIDEOS = os.environ["PRODUCE_VIDEOS"].lower() == "true"
REMOVE_NO_COLLISIONS = os.environ["REMOVE_NO_COLLISIONS"].lower() == "true"
REMOVE_ORIGINAL_SIM_DATA = os.environ["REMOVE_ORIGINAL_SIM_DATA"].lower() == "true"


class CarlaScenarioRunner:

    def __init__(
        self,
        scenario_variants: List[ScenarioTemplate],
        display_size: Tuple[int, int],
        exp_output_dir: str,
        map_name: str = None,
        annotation_format: str = "ultralytics",
    ):
        """Runs a list of scenarios.

        Initializes CARLA container and cleans/restarts it after each run is finished.

        Args:
            scenario_variants: List of scenario templates. Should contain all specific information about the run configuration.
            display_size: Size of produced images / simulation screenshots.
            exp_output_dir: Output directory to store experiment results.
            map_name: CARLA map name. If None, default map is used.
            annotation_format: Used to select the annotator type.
        """
        self.map_name = map_name

        assert len(scenario_variants) > 0, "No scenario!"
        self.scenario_variants = scenario_variants
        self.display_size = display_size
        self.exp_output_dir = exp_output_dir

        self.annotator = self.setup_annotator(annotation_format)
        self.carla_process = None

    def setup_annotator(self, annotation_format: str) -> CarlaAnnotator:
        """Create an Annotator instance to produce annotations in a specific format.

        Args:
            annotation_format: coco or ultralytics
        Returns:
            Annotator instance.
        """
        if annotation_format == "coco":
            annotator = COCOAnnotator(
                display_size=self.display_size,
                exp_dir=self.exp_output_dir,
                annotation_template_path=COCO_ANNOTATION_TEMPLATE_PATH,
                config_path=CLASS_MAPPING_PATH,
                save_segmentation=False,
            )
        elif annotation_format == "ultralytics":
            annotator = UltralyticsAnnotator(
                display_size=self.display_size,
                exp_dir=self.exp_output_dir,
                config_path=CLASS_MAPPING_PATH,
                save_segmentation=False,
            )
        else:
            raise NotImplementedError(
                f"Annotation style: {annotation_format} not implemented."
            )

        return annotator

    def run_scenario_variants(self) -> None:
        """Prepare environment and run scenarios sequentially."""
        logger.info("Setup carla")
        time.sleep(START_CARLA_TIME_SEC)  # Wait for carla, else client hangs

        for i, scenario_variant in enumerate(self.scenario_variants):
            scenario_description = json.dumps(
                scenario_variant, ensure_ascii=False, indent=4, default=lambda obj: str(obj)
            )
            logger.info(f"Currently running:\n{scenario_description}")
            try:
                self.setup_carla()
                self.run_single_scenario(scenario_variant)
                logger.info(f"Finished {i + 1}. run.")
                if PRODUCE_VIDEOS:
                    logger.info("Creating video version")
                    self.transfer_sim_variant_to_video(
                        scenario_variant=scenario_variant,
                    )

            except Exception as e:
                logger.warning(f"Exception occurred {i + 1}. run.\nError: {e}")
            finally:
                self.clean_carla()

    def setup_carla(self) -> None:
        """Pre-simulation environment setup."""
        if USE_DOCKER:
            pass

    def clean_carla(self) -> None:
        """After-simulation environment cleanup."""
        if USE_DOCKER:
            restart_carla_docker_service("carla-simulator")
            time.sleep(START_CARLA_TIME_SEC)

    def run_single_scenario(
        self,
        scenario: ScenarioTemplate,
    ) -> None:
        """Run scenario in a separate process.

        It is practical to run the simulation of the scenario in a separate process to terminate the simulation
        in the case of getting stuck.
        In case of COCO annotations, it is required to retrieve the annotator up to
        the last scenario run, so the data can be appended to the same *.json file.

        Args:
            scenario: Scenario template instance.
        """
        queue = Queue()
        process = multiprocessing.Process(
            target=run_synthesizer,
            args=(self.annotator, scenario, self.exp_output_dir, queue, self.map_name),
        )
        process.start()

        _annotator = None
        try:
            logger.debug("Waiting for queue")
            _annotator = queue.get(
                block=True, timeout=max(scenario["runtime_secs"] + 120, 120)
            )
        except Exception as e:
            logger.warning(f"Could not retrieve annotator! Error: {e}")
        finally:
            while not queue.empty():
                queue.get()
        if _annotator:
            self.annotator = _annotator

        process.join(15)
        if process.is_alive():
            process.kill()

    def transfer_sim_variant_to_video(
        self,
        scenario_variant: ScenarioTemplate,
    ):
        scenario_dir, exp_subdir = osp.split(self.exp_output_dir)

        video_dir = get_video_dir(str(scenario_dir))
        os.makedirs(video_dir, exist_ok=True)

        video_exp_dir = osp.join(video_dir, exp_subdir)

        process_experiment_to_video(
            scenario_dir=scenario_dir,
            exp_dir=self.exp_output_dir,
            video_exp_dir=video_exp_dir,
            frame_rate=scenario_variant.get("simulation_fps", 20),
            image_format=IMAGE_FORMAT,
            remove_no_collisions=REMOVE_NO_COLLISIONS,
            remove_original_sim_data=REMOVE_ORIGINAL_SIM_DATA,
        )


def run_synthesizer(
    annotator: CarlaAnnotator,
    scenario: ScenarioTemplate,
    exp_output_dir: str,
    queue: Queue,
    map_name: str = None,
) -> None:
    """Wrapper method to be run in a separate process.

    Args:
        annotator: Annotator instance.
        scenario: Definition of the scenario to simulate.
        exp_output_dir: Experiment output directory.
        queue: Queue to retrieve the annotator instance from the separate process.
        map_name: CARLA map name. If None, default map is used.
    """
    client = setup_client(CARLA_HOST_NAME, CARLA_PORT)
    if map_name is not None:
        client.load_world(map_name)
        logger.debug(f"Map set to {map_name}")
    synthesizer = CarlaSynthesizer(
        client,
        display_size=scenario["display_size"],
        exp_output_dir=exp_output_dir,
        camera_fov=scenario.get("camera_fov", 90),
        max_distance=scenario.get("max_distance", 140),
        use_lidar=scenario.get("use_lidar", False),
        simulation_fps=scenario.get("simulation_fps", 20),
        frames_per_image=scenario.get("frames_per_image", 120),
        draw_data=scenario.get("draw_data", True),
        change_line_percentage=scenario.get("change_line_percentage", 0.1),
        collisions_ego_only=scenario.get("collisions_ego_only", False),
    )
    synthesizer.setup(
        camera_transform=scenario["transform"],
        weather=scenario["weather"],
        number_of_vehicles=scenario.get("number_of_vehicles", 50),
        number_of_pedestrians=scenario.get("number_of_pedestrians", 20),
        omit_spawn_points=scenario.get("omit_spawn_points", None),
        omit_bikes=scenario.get("omit_bikes", False),
        static_camera=scenario.get("static_camera", False),
    )

    hooks = scenario.get("hooks", [])
    for hook in hooks:
        synthesizer.register_simulation_hook(hook)

    synthesizer.run(
        runtime_secs=scenario["runtime_secs"],
        annotator=annotator,
        min_bbox_pixels=scenario.get("min_pixels", 256),
    )

    queue.put(annotator)


@timeout_with_retry(timeout_secs=15)
def setup_client(
    host_name: str = "127.0.0.1", port: int = 2000, timeout: float = 10
) -> carla.Client:
    """Create Carla client to communicate with the simulator.

    Args:
        host_name: Hostname of the CARLA simulator.
        port: Port of the CARLA simulator.
        timeout: Timeout for a response.

    Returns:
        CARLA client instance to communicate with the simulator.
    """
    logger.debug(f"Connecting to {host_name}:{port}")
    client = carla.Client(host_name, port)
    assert client is not None
    client.set_timeout(timeout)
    return client


def is_carla_running_local() -> bool:
    """Determine if specific CARLA processes are running."""
    return any(
        [
            p.name() in ["CarlaUE4.sh", "CarlaUE4-Linux-Shipping"]
            for p in psutil.process_iter()
        ]
    )


def start_carla_local(carla_executable_path: str) -> subprocess.Popen:
    """Run CARLA executable."""
    command = [
        f"{carla_executable_path}",
        "-quality=epic",
    ]
    carla_process = subprocess.Popen(command, shell=True)
    return carla_process


def restart_carla_docker_service(service_name: str) -> int:
    """Call command to restart specific container."""
    logger.info(f"Restarting {service_name}")
    command = [f"docker restart {service_name}"]
    return subprocess.call(command, shell=True)


def stop_carla_docker_service(service_name: str) -> int:
    """Call command to stop specific container."""
    logger.info(f"Stopping {service_name}")
    command = [f"docker stop {service_name}"]
    return subprocess.call(command, shell=True)


def kill_processes(process_names: List[str]) -> None:
    """Kill the named processes."""
    for process in psutil.process_iter():
        if process.name() in process_names:
            try:
                process.kill()
                process.wait(3)
            except Exception as e:
                logger.warning(f"Error while killing: {process.name()}: {e}")
