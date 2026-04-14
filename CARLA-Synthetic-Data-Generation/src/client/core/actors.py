import logging
import random
from typing import Union

import carla

from .log import setup_logging

setup_logging("./logging.yaml")
logger = logging.getLogger("synthesizer")


def spawn_vehicle(
    world: carla.World,
    traffic_manager: carla.TrafficManager,
    blueprint: carla.ActorBlueprint = None,
    transform: carla.Transform = None,
    omit_bikes: bool = False,
    change_line_percentage: float = 0.0,
) -> Union[carla.Vehicle, None]:
    """Try to spawn an actor in the world.

    Args:
        world: Running CARLA world reference
        traffic_manager: Reference to traffic manager
        blueprint: Vehicle blueprint id. If None, then random a BP is selected.
        transform: Spawn location of the vehicle in the world.
        omit_bikes: When selecting random BPs, 2-wheeled vehicles will be omitted.
        change_line_percentage: [0-1] percentage change of randomly changing lines, when moving.

    Returns:
        Spawned vehicle reference or None, if not spawned.
    """
    if blueprint is None:
        blueprint_library = world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter("*vehicle*")
        if omit_bikes:
            vehicle_blueprints = [
                bp
                for bp in vehicle_blueprints
                if bp.get_attribute("number_of_wheels").as_int() == 4
            ]
        blueprint = random.choice(vehicle_blueprints)

    if transform is None:
        logger.debug(f"No transformation provided for {blueprint.id}. Selecting random transform...")
        world_map = world.get_map()
        spawn_points = world_map.get_spawn_points()
        transform = random.choice(spawn_points)

    vehicle = world.try_spawn_actor(blueprint, transform)

    if vehicle:
        traffic_manager.random_left_lanechange_percentage(
            vehicle, change_line_percentage
        )
        traffic_manager.random_right_lanechange_percentage(
            vehicle, change_line_percentage
        )
        return vehicle


def apply_controls_to_vehicle(
    vehicle: carla.Vehicle,
    throttle: float = 0.0,
    steer: float = 0.0,
    brake: float = 0.0,
    hand_brake: bool = False,
    reverse: bool = False,
) -> None:
    """Wrapper function to apply manual controls to a vehicle.

    Args:
        vehicle: Vehicle reference.
        throttle: [0-1] value of throttle strength.
        steer: [-1 - 1] value of steer strength.
        brake: [0-1] value of brake strength.
        hand_brake: Use hand brake or not.
        reverse: Use reverse or not.

    """
    vehicle.apply_control(
        carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=hand_brake,
            reverse=reverse,
        )
    )


def spawn_pedestrian(
    world: carla.World,
    blueprint: carla.ActorBlueprint = None,
    spawn_point: carla.Transform = None,
) -> Union[carla.Actor, None]:
    """Try to spawn a pedestrian.

    Args:
        world: Running CARLA world reference
        blueprint: Pedestrian blueprint id. If None, then random a BP is selected.
        spawn_point: Spawn location of the pedestrian in the world.

    Returns:
        Pedestrian reference or None, if not spawned.
    """
    blueprint_library = world.get_blueprint_library()
    if blueprint is None:
        pedestrian_blueprints = blueprint_library.filter("*walker.pedestrian*")
        blueprint = random.choice(pedestrian_blueprints)
    if blueprint.has_attribute("is_invincible"):
        blueprint.set_attribute("is_invincible", "false")
    if spawn_point is None:
        location = world.get_random_location_from_navigation()
        if location is None:
            return None
        spawn_point = carla.Transform(location, carla.Rotation())

    pedestrian = world.try_spawn_actor(blueprint, spawn_point)
    return pedestrian


def add_controller_to_pedestrian(
    world: carla.World,
    pedestrian: carla.Walker,
    controller_id: str = None,
    start_location: carla.Location = None,
    target_location: carla.Location = None,
    speed: float = 1.4,
) -> Union[carla.WalkerAIController, None]:
    """Create a controller and assign it to a pedestrian.
    When `controller.walker` is used, then the pedestrian will move in straight line if the start and target
    locations are specified.

    Args:
        world: Running CARLA world reference.
        pedestrian: Spawned pedestrian reference.
        controller_id: Controller BP ID.
        start_location: Location where the pedestrian is spawned. It is used to calculate the direction for the manual
            controls.
        target_location: Location where the pedestrian is supposed to walk to.
        speed: Pedestrian's speed. If high enough, the pedestrian will run instead of walking.

    Returns:
        AI controller reference or None, if manual control was selected.

    """
    controller = None
    if target_location is None:
        target_location = world.get_random_location_from_navigation()
    try:
        if controller_id is None or controller_id == "controller.ai.walker":
            controller_bp = world.get_blueprint_library().find("controller.ai.walker")
            controller = world.spawn_actor(
                controller_bp, pedestrian.get_transform(), pedestrian
            )
            controller.start()
            controller.go_to_location(target_location)
            controller.set_max_speed(speed)
        elif controller_id == "controller.walker":
            move_vector = carla.Vector3D(target_location) - carla.Vector3D(
                start_location
            )
            controller = carla.WalkerControl(move_vector, speed=speed)
            pedestrian.apply_control(controller)
            return None
        else:
            raise ValueError(f"Unknown controller id: {controller_id}")
    except Exception as e:
        logger.warning(f"Exception during pedestrian creation: {e}")
    return controller
