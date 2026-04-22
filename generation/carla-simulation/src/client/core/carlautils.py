import random
import re
from typing import List, Union, Dict, Any

import carla


def randomize_transform(
    transform: carla.Transform,
    location_scaling: Dict[str, float],
    rotation_scaling: Dict[str, float],
) -> carla.Transform:
    """Randomize coordinates (location and rotation) based on the scaling with uniform distribution.

    Empty scaling (equals to zere) results in no randomization.
    """
    location = transform.location
    location.x = location.x + (random.random() - 0.5) * location_scaling.get("x", 0)
    location.y = location.y + (random.random() - 0.5) * location_scaling.get("y", 0)
    location.z = location.z + (random.random() - 0.5) * location_scaling.get("z", 0)

    rotation = transform.rotation
    rotation.pitch = rotation.pitch + (random.random() - 0.5) * rotation_scaling.get("pitch", 0)
    rotation.yaw = rotation.yaw + (random.random() - 0.5) * rotation_scaling.get("yaw", 0)
    rotation.roll = rotation.roll + (random.random() - 0.5) * rotation_scaling.get("roll", 0)
    return transform


def create_transform_from_coordinates(coordinates: Dict[str, Dict[str, float]]) -> carla.Transform:
    """Create carla transform from dict with location and rotation."""
    transform = create_carla_transform(
        **coordinates.get("location", {}), **coordinates.get("rotation", {})
    )
    return transform


def create_carla_transform(
    x: float = 0,
    y: float = 0,
    z: float = 0,
    pitch: float = 0,
    yaw: float = 0,
    roll: float = 0,
    **kwargs,
) -> carla.Transform:
    """Creates carla transform. Default values are zeros."""
    return carla.Transform(carla.Location(x, y, z), carla.Rotation(pitch, yaw, roll))


def vector3d_to_dict(vector: carla.Vector3D) -> Dict[str, float]:
    """Convertor from carla vector to serializable dict."""
    return {"x": vector.x, "y": vector.y, "z": vector.z}


def rotation_to_dict(rotation: carla.Rotation) -> Dict[str, float]:
    """Convertor from carla rotation to serializable dict."""
    return {"pitch": rotation.pitch, "yaw": rotation.yaw, "roll": rotation.roll}


def find_weather_presets() -> Dict[str, carla.WeatherParameters]:
    """Provides a dict with available weather presets."""
    rgx = re.compile(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")
    name = lambda x: "".join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match("[A-Z].+", x)]
    return {name(x): getattr(carla.WeatherParameters, x) for x in presets}


def create_carla_weather(weather_present: Union[str, Dict[str, Any]]) -> carla.WeatherParameters:
    """Create weather parameters based on a preset name or dict with kwarg parameters."""
    weather_presents = find_weather_presets()
    if isinstance(weather_present, str):
        return weather_presents[weather_present]
    elif isinstance(weather_present, dict):
        return carla.WeatherParameters(**weather_present)
    else:
        raise NotImplementedError


def _ids_to_blueprint(
    world: carla.World, blueprint_ids: List[str]
) -> Union[carla.ActorBlueprint, None]:
    """Select a random carla blueprint based on list of names/ids."""
    blueprint = None
    if blueprint_ids:
        blueprint_id = random.choice(blueprint_ids)
        blueprint = world.get_blueprint_library().find(blueprint_id)
    return blueprint


def set_spector_view(world: carla.World, target_transform: carla.Transform) -> None:
    """Sets a spector view in the world, usually to the position of camera.

    Useful when monitoring the simulation from display.
    """
    spectator = world.get_spectator()
    spectator.set_transform(target_transform)


def get_ordered_spawn_points(world: carla.World) -> List[carla.Transform]:
    current_map = world.get_map()
    spawn_points = current_map.get_spawn_points()
    spawn_points = sorted(spawn_points, key=lambda p: p.location.x * 10000 + p.location.y)
    return spawn_points
