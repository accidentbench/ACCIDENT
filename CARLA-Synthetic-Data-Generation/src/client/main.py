import argparse
import glob
import logging
import os
import os.path as osp
import traceback
from typing import List

from runner import CarlaScenarioRunner, stop_carla_docker_service
from scenario import ScenarioMaker, ScenarioTemplate

logger = logging.getLogger("runner")

SCENARIO_DIR = os.environ["SCENARIO_DIR"]
CARLA_HOST = os.environ["CARLA_HOST_NAME"]
DRAW_DATA = os.environ.get("DRAW_DATA", False) == "True"
USE_LIDAR = os.environ.get("USE_LIDAR", False) == "True"


def parse_arguments():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser("Run Carla Scenario")
    parser.add_argument(
        "--scenario_files",
        type=str,
        help="Names of `carla_scenario.yaml` files.",
        default=None,
    )
    parser.add_argument(
        "--annotation_format",
        type=str,
        help="Type of generated annotations.",
        choices=["coco", "ultralytics"],
        default="ultralytics",
    )
    return parser.parse_args()


def get_scenario_paths(scenario_dir: str, files_or_patterns: List[str] = None) -> List[str]:
    """Load scenario paths.

    Args:
        scenario_dir: Path to a scenario directory, where to search for scenario files.
        files_or_patterns: List of scenario file names or patterns to be used in glob.glob().
            If None, then the whole scenario dir is searched for *.yaml files.
    Returns:
        List of scenario paths.
    """
    if files_or_patterns is None:
        files_or_patterns = sorted(glob.glob(osp.join(scenario_dir, "*.yaml")))

    scenario_paths = []
    for pattern in files_or_patterns:
        scenario_paths.extend(sorted(glob.glob(osp.join(scenario_dir, pattern))))
    return scenario_paths


def run_single_scenario_file(scenario_path: str, annotation_format: str) -> None:
    """Run a single scenario config.
    From the config, multiple runs can be generated based on different sensor definitions and weathers. Runs are
    generated in a grid manner. The runs are sequentially run with the CarlaScenarioRunner.

    Args:
        scenario_path: Path to a scenario config file.
        annotation_format: Format in which to generate annotations.
    """
    scenario_template = ScenarioTemplate(
        scenario_path, draw_data=DRAW_DATA, use_lidar=USE_LIDAR
    )
    scenario_template["annotation_format"] = args.annotation_format

    display_size = scenario_template["display_size"]
    map_name = scenario_template["map_name"]

    scenario_variants = ScenarioMaker(
        scenario_template=scenario_template,
    ).create_grid()

    exp_output_dir = scenario_template.exp_output_dir

    runner = CarlaScenarioRunner(
        scenario_variants,
        display_size=display_size,
        exp_output_dir=exp_output_dir,
        map_name=map_name,
        annotation_format=annotation_format,
    )
    runner.run_scenario_variants()


def run_multiple_scenario_files(scenario_paths: List[str], annotation_format: str) -> None:
    """Run multiple scenarios.

    Args:
        scenario_paths: Paths to scenario config files.
        annotation_format: Format in which to generate annotations.
    """
    for scenario_path in scenario_paths:
        try:
            logger.info(f"Running scenario at: {scenario_path}")
            run_single_scenario_file(scenario_path, annotation_format)
            logger.info("Finished scenario.")
        except Exception as e:
            logger.error(f"Exception occurred. Error: {e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    args = parse_arguments()
    if args.scenario_files is not None:
        scenario_files = args.scenario_files.split(",")
    elif "SELECTED_SCENARIO_CONFIGS" in os.environ.keys():
        scenario_files = os.environ.get("SELECTED_SCENARIO_CONFIGS").split(",")
    else:
        scenario_files = None
        logger.info("Scenario path is None, select all scenarios.")

    annotation_format = "ultralytics"
    if args.annotation_format is not None:
        annotation_format = args.annotation_format
    elif "ANNOTATION_FORMAT" in os.environ.keys():
        annotation_format = os.environ.get("ANNOTATION_FORMAT")
    else:
        raise RuntimeError("Annotation format is not set.")

    scenario_paths = get_scenario_paths(SCENARIO_DIR, scenario_files)
    logger.info(f"Loaded scenario paths:\n{scenario_paths}")
    run_multiple_scenario_files(scenario_paths, annotation_format)

    stop_carla_docker_service("carla-simulator")
