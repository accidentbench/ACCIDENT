from typing import Any, Dict, List, Tuple, Union

import numpy as np

import carla

from .actors import apply_controls_to_vehicle
from .bbox_segmentation import ClientSideBoundingBoxes, transform_bbox_to_2d


class CollisionsEvaluator:

    def __init__(
        self,
        sensor: carla.Sensor,
        display_size: Tuple[int, int],
        expanding_mode: bool = False,
        ego_only: bool = False,
    ):
        """Evaluates collisions and creates a circumferential bounding box projected to the camera view.

        Args:
            sensor: collision sensor
            display_size: Captured resolution.
            expanding_mode: If the collision should expand with the moving actors or not
        """
        self.sensor = sensor
        self.display_size = display_size
        self.expanding_mode = expanding_mode
        self.ego_only = ego_only

        self.collision_history: List[Dict[str, Any]] = []
        self.actor = None
        self.other_actors = set()

    def add_events(
        self,
        events: List[carla.CollisionEvent],
        traffic_manager: carla.TrafficManager,
    ):
        """Save collision actors to the history."""
        for event in events:
            actor, other_actor = event.actor, event.other_actor
            if self.actor is None:
                self.actor = actor
                self.actor.set_autopilot(False, traffic_manager.get_port())
                apply_controls_to_vehicle(self.actor, throttle=0.0, brake=1.0, hand_brake=True)
                print(self.actor, "set to break")

            if not self.ego_only:
                self.other_actors.add(other_actor)
            try:
                other_actor.set_autopilot(False, traffic_manager.get_port())
                apply_controls_to_vehicle(other_actor, throttle=0.0, brake=1.0, hand_brake=True)
                print(other_actor, "set to break")
            except Exception as e:
                print("Error while stopping other actor:", e)

    def evaluate_collision_event(
        self,
        iteration: int,
    ) -> None:
        """Evaluate and expand collision location/bbox after it occurs.

        Based on colliding actors movement and their bboxes after the collision.
        """
        if self.actor is None:
            return None
        tagged_bboxes_3d = ClientSideBoundingBoxes.get_bounding_boxes(
            [self.actor, *self.other_actors], self.sensor, max_distance=150
        )
        bboxes_2d = [
            transform_bbox_to_2d(ann["3d_bbox_proj"], self.display_size)
            for ann in tagged_bboxes_3d
        ]

        if self.expanding_mode and len(self.collision_history) > 0:
            bboxes_2d.append(self.collision_history[-1]["collision_bbox"])

        if len(bboxes_2d) == 0:
            return None

        collision_bbox = ClientSideBoundingBoxes.get_circumferential_bbox(
            bboxes_2d, self.display_size
        )
        self.collision_history.append(
            {
                "iteration": iteration,
                "collision_bbox": collision_bbox,
                "ids": [self.actor.id, *set(actor.id for actor in self.other_actors)],
            }
        )

    def get_collision_bbox(self) -> Union[np.ndarray, None]:
        """Get the last collision bbox from the collision history."""
        if len(self.collision_history) == 0:
            return None
        return self.collision_history[-1]["collision_bbox"]
