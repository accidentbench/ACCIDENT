#!/usr/bin/env python

import queue
from typing import Dict, Tuple, Union, Callable


import carla


def enable_synchronous_mode(
    world: carla.World,
    fixed_delta_seconds: float,
    traffic_manager: carla.TrafficManager = None,
) -> Tuple[int, carla.WorldSettings]:
    """Enable synchronous mode in the simulation.

    Args:
        world: World reference to running CARLA.
        fixed_delta_seconds: Time between simulation ticks or frames in seconds.
        traffic_manager: Traffic manager reference.
    Returns:
        Current frame number and the original world settings, which should be asynchronous.
    """
    _settings = world.get_settings()
    settings = carla.WorldSettings(
        synchronous_mode=True,
        fixed_delta_seconds=fixed_delta_seconds,
    )
    if traffic_manager is not None:
        # Recommended by docs, but vehicles do not move
        # traffic_manager.set_synchronous_mode(True)
        pass

    frame = world.apply_settings(settings)
    return frame, _settings


def disable_synchronous_mode(
    world: carla.World,
    original_settings: carla.WorldSettings,
    traffic_manager: carla.TrafficManager = None,
):
    """Disable synchronous mode in the simulation.

    Args:
        world: World reference to running CARLA.
        original_settings: Original world settings, which should be asynchronous.
        traffic_manager: Traffic manager reference.

    """
    if traffic_manager is not None:
        # Recommended by docs, but vehicles do not move
        # traffic_manager.set_synchronous_mode(False)
        pass

    world.apply_settings(original_settings)


class CarlaSynchronizer:

    def __init__(
        self,
        world: carla.World,
        traffic_manager: carla.TrafficManager = None,
        fps: int = 20,
    ):
        """Context manager to synchronize output from different sensors.
        Inspired by the CARLA synchronization Python example.

        Args:
            world: Running CARLA world reference.
            traffic_manager: Traffic manager reference to change synchronization mode.
            fps: Frames per second of the simulation.
        """
        self.world = world
        self.traffic_manager = traffic_manager
        self.frame = None
        self.delta_seconds = 1.0 / fps
        self._queues = {}
        self._event_queues = {}
        self._settings = None

        self._make_queue(self.world.on_tick, "snapshot", self._queues)

    def enable_sync_mode(self) -> None:
        """Enable synchronous mode. Saves original mode settings and the current frame number."""
        self.frame, self._settings = enable_synchronous_mode(
            self.world, self.delta_seconds, self.traffic_manager
        )

    def disable_sync_mode(self) -> None:
        """Disable synchronous mode. Applies original mode settings."""
        disable_synchronous_mode(self.world, self._settings, self.traffic_manager)

    @staticmethod
    def _make_queue(register_event: Callable, sensor_name: str, queues: Dict[str, queue.Queue]):
        """Put the sensor's listen function output to a dedicated queue and save the queue to the dict.

        Args:
            register_event: Sensor's listen function.
            sensor_name: Name of the sensor.
            queues: Dictionary with existing queues.

        """
        q = queue.Queue(maxsize=100)
        register_event(q.put)
        queues[sensor_name] = q

    def create_sensor_queues(self, sensors: Dict[str, carla.Sensor]) -> None:
        """Create queues used to synchronize output from different sensors and saves the active queues.

        Args:
            sensors: List of instantiated CARLA sensors.
        """
        for name, sensor in sensors.items():
            self._make_queue(sensor.listen, name, self._queues)

    def add_sensor_queue(
        self, name: str, sensor: carla.Sensor, event_based: bool = False
    ) -> None:
        """Add a dedicated queue gather output data from the sensor listening function.

        Args:
            name: Unique name of the sensor.
            sensor: Sensor instance.
            event_based: If the sensor outputs data the whole time of the simulation or not.

        """
        if name in self._queues or name in self._event_queues:
            raise RuntimeError(f"Sensor {name} already exists")
        if event_based:
            self._make_queue(sensor.listen, name, self._event_queues)
        else:
            self._make_queue(sensor.listen, name, self._queues)

    def tick(self, timeout: float) -> Tuple[dict, dict]:
        """Moves the simulation to the next frame and gathers the generated sensory output data.

        Args:
            timeout: How long to wait for the sensory data before moving on.

        Returns:
            Dicts with sensor name to generated data in the current step.

        """
        self.frame = self.world.tick()
        if timeout <= 0:
            return {}, {}
        data = {
            name: self._retrieve_data(q, timeout) for name, q in self._queues.items()
        }
        assert all(x.frame == self.frame for x in data.values())
        event_data = {
            name: self._retrieve_event_data(q) for name, q in self._event_queues.items()
        }
        return data, event_data

    def _retrieve_data(
        self, sensor_queue: queue.Queue, timeout: float
    ) -> carla.SensorData:
        """Retrieve sensor data from the sensor's queue.

        Args:
            sensor_queue: queue to retrieve sensor data from.
            timeout: How long to wait before giving up.

        Returns:
            Data from the sensor in the current frame/step.

        """
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

    def _retrieve_event_data(
        self, sensor_queue: queue.Queue
    ) -> Union[carla.SensorData, None]:
        """Retrieve sensor data from the sensor's queue.

        Some sensors can be event-based, such as collision sensor, so they do not have to be present
        in each frame.

        Args:
            sensor_queue: queue to retrieve sensor data from.

        Returns:
            Data from the sensor in the current frame/step.

        """
        while True:
            if sensor_queue.empty():
                return None
            data = sensor_queue.get_nowait()
            if data.frame == self.frame:
                return data
