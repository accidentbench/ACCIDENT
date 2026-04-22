import os
import os.path as osp
import random
import shutil
from typing import Dict, List, Any, Union
from pathlib import Path

import carla

from core.carlautils import (create_carla_weather,
                             create_transform_from_coordinates,
                             randomize_transform)
from core.ioutils import load_yaml


OUTPUT_DIR = os.environ["OUTPUT_DIR"]


def add_output_path(config: Dict[str, Any], exp_fmt: str) -> None:
    """Create experiment output directory path and add it to the config.

    Makes the output directory if it doesn't already exist.

    Args:
        config: Scenario configuration.
        exp_fmt: Format of the output directory (keys in the config to get their values).
    """

    _exp_name_vals = []
    for attr in exp_fmt.split("-"):
        assert attr in config, f"Unknown attribute {attr} in configuration file."
        _exp_name_vals.append(str(config[attr]))
    exp_name = "-".join(_exp_name_vals)

    output_path = osp.join(OUTPUT_DIR, exp_name)

    if os.path.isdir(output_path):
        existing_exps = [x for x in os.listdir(output_path) if x.startswith("exp")]
        last_exp = max([int(x[3:]) for x in existing_exps] or [0])
        exp_number = f"exp{last_exp + 1:02d}"
    else:
        exp_number = "exp01"

    output_path = osp.join(output_path, exp_number)
    config["output_path"] = output_path
    os.makedirs(output_path, exist_ok=True)


class ScenarioTemplate:

    def __init__(self, scenario_path: str, **kwargs):
        """Stores a specific run configuration.

        Args:
            scenario_path: Path to the scenario file (should be *.yaml).
            **kwargs: Additional arguments to update the scenario configuration.
        """
        self.scenario_path = scenario_path
        self.scenario = ScenarioTemplate.load_scenario(self.scenario_path)
        self.template = self.scenario["template"]
        self.template.update(kwargs)
        self.template["scenario_name"] = Path(self.scenario_path).stem
        add_output_path(self.template, exp_fmt="scenario_name")
        self.exp_output_dir = self.template["output_path"]

        self.sensors = self.scenario.get("sensors", None)
        assert self.sensors is not None, "No sensors defined. Should be added to the scenario config."
        self.weathers = self.scenario.get("weathers", None)
        assert self.weathers is not None, "No weathers defined. Should be added to the scenario config."

        shutil.copy(self.scenario_path, self.exp_output_dir)

    @staticmethod
    def load_scenario(scenario_path: str) -> Dict[str, Any]:
        """Load config file parameters.

        Args:
            scenario_path (str): A path to the config file.

        Returns:
            tuple: A template dict, a list of sensor coordinates and
                a list of weathers.
        """
        scenario = load_yaml(scenario_path)
        return scenario

    def __getitem__(self, item: str) -> Any:
        return self.template[item]

    def __setitem__(self, key: str, value: Any) -> None:
        self.template[key] = value

    def get_template(self) -> Dict[str, Any]:
        """Return scenario template."""
        return self.template

    def get_sensor_definitions(self) -> List[Dict[str, Any]]:
        """Return possible sensor coordinates."""
        return self.sensors

    def get_weathers(self) -> List[Union[str, Dict[str, Any]]]:
        """Return possible weathers"""
        return self.weathers


class ScenarioMaker:

    def __init__(
        self,
        scenario_template: ScenarioTemplate,
    ):
        """Produces grid of scenarios from a single configuration file or ScenarioTemplate.

        Args:
            scenario_template: The scenario template to use as a base.
        """
        self.scenario_template = scenario_template

        self.sensors = self.scenario_template.get_sensor_definitions()
        self.weathers = self.scenario_template.get_weathers()
        self.weathers = [create_carla_weather(weather) for weather in self.weathers]

    def create_grid(
        self,
        weathers: List[carla.WeatherParameters] = None,
        sensors: List[Dict[str, Any]] = None,
    ) -> List[ScenarioTemplate]:
        """Create a grid of scenarios with different weathers, sensor positions, etc.

        Args:
            weathers: List of weather presets.
            sensors: List of senser definitions. Sensor position is moved randomly based on the definition.

        Returns:
            List of scenario templates with grid weather and sensor options.
        """
        if weathers is None:
            weathers = self.weathers
        if sensors is None:
            sensors = self.sensors

        scenarios = []
        for weather in weathers:
            for sensor in sensors:
                scenario = self.scenario_template.get_template().copy()
                scenario["weather"] = weather
                sensor_transform = create_transform_from_coordinates(sensor)
                location_scaling = sensor.get("location_scaling", {})
                rotation_scaling = sensor.get("rotation_scaling", {})
                sensor_transform = randomize_transform(sensor_transform, location_scaling, rotation_scaling)
                scenario["transform"] = sensor_transform
                scenario["static_camera"] = sensor["static_camera"]

                if isinstance(scenario["camera_fov"], list):
                    scenario["camera_fov"] = random.choice(scenario["camera_fov"])
                scenarios.append(scenario)

        return scenarios
