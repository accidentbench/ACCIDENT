import logging
import random
import uuid
from functools import partial
from typing import Callable, List, Dict, Any

import carla

from .actors import (add_controller_to_pedestrian, spawn_pedestrian,
                     spawn_vehicle, apply_controls_to_vehicle)
from .carlautils import (_ids_to_blueprint, create_carla_transform, create_carla_transform, get_ordered_spawn_points,
                         create_transform_from_coordinates,
                         randomize_transform)
from .log import setup_logging
from .sensors import create_collision_sensor
from .synchronous_mode import CarlaSynchronizer


setup_logging("./logging.yaml")
logger = logging.getLogger("synthesizer")


class HookRegistry:

    def __init__(self):
        """Hook registry to register and call custom actions in the simulation."""
        self._hooks = {}

    def add_hook(self, init_frame: int, func: Callable, repeat: bool = False) -> None:
        """Register a new hook function.

        Args:
            init_frame: Frame when the hook is called.
            func: the called function
            repeat: if the hook should be repeated. If so, then the init_number is also a period.

        """
        hook_name = str(uuid.uuid4())
        self._hooks[hook_name] = {
            "init_frame": init_frame,
            "func": func,
            "repeat": repeat,
        }
        logger.info(f"Hook '{hook_name}' registered")

    def remove_hook(self, hook_name: str) -> None:
        """Remove a hook based on name from the registry."""
        if hook_name in self._hooks:
            logger.info(f"Hook '{hook_name}' removed")
            self._hooks.pop(hook_name)

    def invoke_hooks_by_frame(
        self,
        iteration: int,
    ) -> List[Any]:
        """Invoke all hooks based on the init frame number.

        Args:
            iteration: Current simulation iteration.

        Returns:
            List of hook function call outputs.

        """
        results = []
        for hook_name, hook_data in list(self._hooks.items()):  # Can be removed while iterating over
            if hook_data["init_frame"] != 0 and iteration % hook_data["init_frame"] != 0:
                continue
            logger.info(f"Invoking hook '{hook_name}', iteration '{iteration}'")
            func = hook_data["func"]
            result = func()
            results.append(result)
            if not hook_data["repeat"]:
                self.remove_hook(hook_name)
        return results

    def get_hooks(
        self,
    ):
        """Return names of all registered hooks."""
        return self._hooks.keys()


    def register_carla_hook(
        self,
        hook_definition: Dict[str, Any],
        world: carla.World,
        traffic_manager: carla.TrafficManager,
        spawned_vehicles: List[carla.Vehicle],
        spawned_pedestrians: List[carla.Actor],
        spawned_props: List[carla.Actor],
        sensors: Dict[str, carla.Sensor] = None,
        synchronizer: CarlaSynchronizer = None,
    ) -> None:
        """Hook 'factory' to create different hook functions and save the hook to the registry.

        Args:
            hook_definition: Definition of the hook function (function name, init frame, fn kwargs etc.).
            world: Running CARLA world reference.
            traffic_manager: CARLA traffic manager reference.
            spawned_vehicles: List with all spawned vehicles.
            spawned_pedestrians: List with all spawned pedestrians.
            spawned_props: List with all spawned props.
            sensors: Optional dict with all spawned sensors.
            synchronizer: Optional synchronizer reference used to add new sensors.

        """
        frame = hook_definition["frame"]
        name = hook_definition["name"]
        kwargs = hook_definition.get("kwargs", {})
        repeat = hook_definition.get("repeat", False)

        if name == "spawn_vehicle_hook":
            self.add_hook(
                frame,
                func=partial(
                    spawn_vehicle_hook,
                    world,
                    traffic_manager,
                    spawned_vehicles,
                    sensors,
                    synchronizer,
                    **kwargs,
                ),
                repeat=repeat,
            )
        elif name == "spawn_pedestrian_hook":
            self.add_hook(
                frame,
                func=partial(
                    spawn_pedestrian_hook,
                    world,
                    spawned_pedestrians,
                    **kwargs,
                ),
                repeat=repeat,
            )
        elif name == "destroy_pedestrians_hook":
            self.add_hook(
                frame,
                func=partial(destroy_actors_hook, spawned_pedestrians),
                repeat=repeat,
            )
        elif name == "spawn_prop_hook":
            self.add_hook(
                frame,
                func=partial(spawn_prop_hook, world, spawned_props, **kwargs),
                repeat=repeat,
            )
        elif name == "destroy_props_hook":
            self.add_hook(
                frame,
                func=partial(destroy_actors_hook, spawned_props),
                repeat=repeat,
            )
        elif name == "set_speed_pedestrian_hook":
            self.add_hook(
                frame,
                func=partial(set_speed_pedestrian_hook, spawned_pedestrians, **kwargs),
                repeat=repeat,
            )


def spawn_vehicle_hook(
    world: carla.World,
    traffic_manager: carla.TrafficManager,
    spawned_vehicles: List[carla.Vehicle],
    sensors: Dict[str, carla.Sensor],
    synchronizer: CarlaSynchronizer,
    **kwargs,
) -> None:
    """Try to spawn a vehicle with custom behaviour.

    Args:
        world: Running CARLA world reference.
        traffic_manager: CARLA traffic manager reference.
        spawned_vehicles: List with all spawned vehicles to save new reference.
        sensors: List with all spawned sensors to save new reference.
        synchronizer: Synchronizer reference used to add new sensors.
        **kwargs: parameters of the vehicle behaviour

    """
    control = kwargs["control"]
    coordinates = kwargs["coordinates"]
    blueprint = _ids_to_blueprint(world, kwargs.get("blueprint_ids", []))
    spawn_points = get_ordered_spawn_points(world)

    spawn_point_index = coordinates.get("spawn_point", None)
    if spawn_point_index is None:
        spawn_point = create_transform_from_coordinates(coordinates)
    else:
        spawn_point = spawn_points[int(spawn_point_index)]

    location_scaling = coordinates.get("location_scaling", {})
    rotation_scaling = coordinates.get("rotation_scaling", {})
    spawn_point = randomize_transform(
        spawn_point, location_scaling=location_scaling, rotation_scaling=rotation_scaling,
    )

    vehicle = spawn_vehicle(
        world,
        traffic_manager,
        blueprint=blueprint,
        transform=spawn_point,
        change_line_percentage=0,
    )
    if vehicle is not None:
        # Set parameters of TM vehicle control, we don't want lane changes
        # traffic_manager.update_vehicle_lights(vehicle, True)
        autopilot = control.get("autopilot", None)
        if autopilot is not None:
            vehicle.set_autopilot(True, traffic_manager.get_port())
            traffic_manager.auto_lane_change(vehicle, True)
            path = autopilot.get("path", None)
            if path is not None:
                path = [spawn_points[sp].location for sp in path]
                traffic_manager.set_path(vehicle, path)
            traffic_manager.ignore_vehicles_percentage(actor=vehicle, perc=autopilot.get("ignore_vehicles", 0))
            traffic_manager.ignore_signs_percentage(actor=vehicle, perc=autopilot.get("ignore_signs", 0))
            traffic_manager.ignore_lights_percentage(actor=vehicle, perc=autopilot.get("ignore_lights", 0))
            traffic_manager.ignore_walkers_percentage(actor=vehicle, perc=autopilot.get("ignore_walkers", 0))
            # traffic_manager.global_percentage_speed_difference(actor=vehicle, percentage=autopilot.get("global_percentage_speed_difference", 30))
            # traffic_manager.vehicle_percentage_speed_difference(actor=vehicle, percentage=autopilot.get("vehicle_percentage_speed_difference", 30))
            if "desired_speed" in autopilot:
                traffic_manager.set_desired_speed(actor=vehicle, speed=autopilot.get("desired_speed"))

        else:
            apply_controls_to_vehicle(vehicle, **control.get("vehicle_control", {}))

        velocity = control.get("velocity", {})
        if len(velocity) > 0:
            velocity_vector = carla.Vector3D(x=velocity.get("x", 0), y=velocity.get("y", 0), z=velocity.get("z", 0))
            vehicle.set_target_velocity(velocity_vector)
            logger.debug(f"Changed {vehicle} velocity to {velocity_vector}")

        add_collision_sensor = kwargs.get("add_collision_sensor", False)
        if add_collision_sensor:
            collision_sensor = create_collision_sensor(world, vehicle)
            sensors["collision_sensor"] = collision_sensor
            synchronizer.add_sensor_queue(
                "collision_sensor", collision_sensor, event_based=True
            )

        spawned_vehicles.append(vehicle)
    else:
        logger.warning(f"Vehicle not spawned.")


def spawn_pedestrian_hook(
    world: carla.World, spawned_pedestrians: List[carla.Actor], **kwargs
) -> None:
    """Try to spawn a pedestrian with custom behaviour.

    Args:
        world: Running CARLA world reference.
        spawned_pedestrians: List with all spawned pedestrians to save new reference.
        **kwargs: Parameters for pedestrian behaviour.

    """
    blueprint = _ids_to_blueprint(world, kwargs.get("blueprint_ids", []))
    coordinates = kwargs["coordinates"]

    location_scaling = coordinates.get("location_scaling", {})
    rotation_scaling = coordinates.get("rotation_scaling", {})
    spawn_point = create_transform_from_coordinates(coordinates)
    spawn_point = randomize_transform(spawn_point, location_scaling=location_scaling, rotation_scaling=rotation_scaling)

    pedestrian = spawn_pedestrian(world, blueprint=blueprint, spawn_point=spawn_point)
    if pedestrian is not None:
        controller_id = kwargs.get("controller_id", None)
        target_location = kwargs.get("target_location", None)
        if target_location is not None:
            target_location = create_carla_transform(**target_location).location
        speed = kwargs.get("speed", 0.4)
        _ = add_controller_to_pedestrian(
            world,
            pedestrian,
            controller_id,
            start_location=spawn_point.location,
            target_location=target_location,
            speed=speed,
        )

        spawned_pedestrians.append(pedestrian)
    else:
        logger.warning(f"Pedestrian not spawned.")


def spawn_prop_hook(world: carla.World, spawned_props: List[carla.Actor], **kwargs) -> None:
    """Try to spawn a prop with custom location.

    Args:
        world: Running CARLA world reference.
        spawned_props: List with all spawned props to save new reference.
        **kwargs: Parameters of the spawned prop.

    """
    blueprint = _ids_to_blueprint(world, kwargs.get("blueprint_ids", []))
    if blueprint is None:
        blueprint_library = world.get_blueprint_library()
        props_blueprints = blueprint_library.filter("*statix.prop*")
        blueprint = random.choice(props_blueprints)

    coordinates = kwargs["coordinates"]
    location_scaling = coordinates.get("location_scaling", {})
    rotation_scaling = coordinates.get("rotation_scaling", {})
    spawn_point = create_transform_from_coordinates(coordinates)
    spawn_point = randomize_transform(spawn_point, location_scaling=location_scaling, rotation_scaling=rotation_scaling)

    prop = world.try_spawn_actor(blueprint, spawn_point)
    if prop is not None:
        prop.set_enable_gravity(True)
        spawned_props.append(prop)
    else:
        logger.warning(f"Prop not spawned.")


def destroy_actors_hook(spawned_actors: List[carla.Actor]) -> None:
    """Destroy all spawned actors, update the input list."""
    for actor in spawned_actors:
        try:
            if actor.is_alive:
                actor.destroy()
        except Exception as e:
            logger.warning(f"Error during actor destruction: {e}")
    spawned_actors[:] = []


def set_speed_pedestrian_hook(
    spawned_pedestrians: List[carla.Actor], speed: float, **kwargs
) -> None:
    """Update speed of pedestrians."""
    for pedestrian in spawned_pedestrians:
        pedestrian.apply_control(carla.WalkerControl(speed=speed))
