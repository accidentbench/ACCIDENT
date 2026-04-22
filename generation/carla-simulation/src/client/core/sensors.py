import matplotlib.cm as cm
import numpy as np
import open3d as o3d

import carla

from typing import Tuple


VIRIDIS = np.array(cm.get_cmap("plasma").colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


def setup_sensor(
    world: carla.World,
    sensor_blueprint: carla.ActorBlueprint,
    sensor_transform: carla.Transform,
    vehicle: carla.Actor = None,
    display_size: tuple = None,
    sensor_fov: float = None,
) -> carla.Sensor:
    """Update sensor blueprint and spawn the sensor. Add calibration data based on the view.

    Args:
        world: Running CARLA world reference.
        sensor_blueprint: Sensor blueprint.
        sensor_transform: Sensor spawn position.
        vehicle: Optional vehicle instance to attach the sensor to.
        display_size: Captured image resolution.
        sensor_fov: Field of view of the sensor.

    Returns:
        Sensor instance.

    """
    if display_size is not None:
        sensor_blueprint.set_attribute("image_size_x", f"{display_size[0]}")
        sensor_blueprint.set_attribute("image_size_y", f"{display_size[1]}")
    if sensor_fov is not None:
        sensor_blueprint.set_attribute("fov", f"{sensor_fov}")

    sensor = world.spawn_actor(sensor_blueprint, sensor_transform, attach_to=vehicle)

    calibration = get_calibration(
        *display_size, sensor_blueprint.get_attribute("fov").as_float()
    )
    sensor.calibration = calibration
    return sensor


def create_collision_sensor(
    world: carla.World,
    vehicle: carla.Actor,
) -> carla.Sensor:
    """Create a collision sensor and attach it an actor.

    Args:
        world: Running CARLA world reference.
        vehicle: Vehicle instance to attach the collision sensor to.

    Returns:
        Collision sensor instance.

    """
    bp_library = world.get_blueprint_library()
    collision_sensor = world.spawn_actor(
        bp_library.find("sensor.other.collision"), carla.Transform(), attach_to=vehicle
    )
    return collision_sensor


def get_calibration(image_width: int, image_height: int, fov: float) -> np.ndarray:
    """Returns sensor calibration / projection matrix based on the display size and fov.

    Args:
        image_width: Camera image width.
        image_height: Camera image height.
        fov: Camera field of view.

    Returns:
        Camera calibration / projection matrix.

    """
    focal = image_width / (2.0 * np.tan(fov * np.pi / 360.0))
    calibration = np.identity(3)
    calibration[0, 0] = calibration[1, 1] = focal
    calibration[0, 2] = image_width / 2.0
    calibration[1, 2] = image_height / 2.0
    return calibration


def convert_raw_sensor_data(sensor_image: carla.SensorData) -> np.ndarray:
    """Converts sensor image to numpy array.

    Args:
        sensor_image: Sensor data from a camera sensor.

    Returns:
        [W, H, RGB] values

    """
    array = np.frombuffer(sensor_image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (sensor_image.height, sensor_image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


def setup_lidar(
    world: carla.World,
    sensor_transform: carla.Transform,
    vehicle: carla.Actor = None,
    sensor_fov: float = None,
    no_noise: bool = True,
    sensor_range: float = 250,
    rotation_frequency: int = 20,
) -> carla.Sensor:
    """Create Lidar sensor instance.

    Args:
        world: Running CARLA world reference.
        sensor_transform: Spawn position of the sensor.
        vehicle: Optional vehicle instance to attach the sensor to.
        sensor_fov: Lidar field of view.
        no_noise: Introduce noise to Lidar or not.
        sensor_range: Maximum recognized distance from the Lidar.
        rotation_frequency: How many times will the lidar make a full rotation in a second.

    Returns:
        Lidar sensor instance.

    """
    blueprint_library = world.get_blueprint_library()
    lidar_sensor_blueprint = blueprint_library.find("sensor.lidar.ray_cast")
    if sensor_fov is not None:
        lidar_sensor_blueprint.set_attribute("horizontal_fov", f"{sensor_fov}")

    if no_noise:
        lidar_sensor_blueprint.set_attribute("dropoff_general_rate", "0.0")
        lidar_sensor_blueprint.set_attribute("dropoff_intensity_limit", "1.0")
        lidar_sensor_blueprint.set_attribute("dropoff_zero_intensity", "0.0")
    lidar_sensor_blueprint.set_attribute("upper_fov", f"{30}")
    lidar_sensor_blueprint.set_attribute("lower_fov", f"{-30}")
    lidar_sensor_blueprint.set_attribute("channels", f"{64}")
    lidar_sensor_blueprint.set_attribute("range", f"{sensor_range}")
    lidar_sensor_blueprint.set_attribute("rotation_frequency", f"{rotation_frequency}")
    lidar_sensor_blueprint.set_attribute("points_per_second", str(200000))
    sensor = world.spawn_actor(
        lidar_sensor_blueprint, sensor_transform, attach_to=vehicle
    )
    return sensor


def process_lidar_data(lidar_data: carla.SensorData, lidar: carla.Sensor, camera: carla.Sensor) -> Tuple[np.ndarray, o3d.geometry.PointCloud]:
    """Project lidar points to the camera view.

    See CARLA Python lidar_to_camera.py example.

    Args:
        lidar_data: Lidar sensor data.
        lidar: Lidar sensor instance.
        camera: Camera sensor instance in the same position as lidar to get the projection matrix.

    Returns:
        Projected lidar points to the camera view.

    """
    p_cloud_size = len(lidar_data)
    p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype("f4")))
    p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))

    # Lidar intensity array of shape (p_cloud_size,) but, for now, let's
    # focus on the 3D points.
    intensity = np.array(p_cloud[:, 3])

    # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
    local_lidar_points = np.array(p_cloud[:, :3]).T

    # Add an extra 1.0 at the end of each 3d point so it becomes of
    # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
    local_lidar_points = np.r_[
        local_lidar_points, [np.ones(local_lidar_points.shape[1])]
    ]

    # This (4, 4) matrix transforms the points from lidar space to world space.
    lidar_2_world = lidar.get_transform().get_matrix()

    # Transform the points from lidar space to world space.
    world_points = np.dot(lidar_2_world, local_lidar_points)

    # This (4, 4) matrix transforms the points from world to sensor coordinates.
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    # Transform the points from world space to camera space.
    sensor_points = np.dot(world_2_camera, world_points)

    # New we must change from UE4's coordinate system to an "standard"
    # camera coordinate system (the same used by OpenCV):

    # ^ z                       . z
    # |                        /
    # |              to:      +-------> x
    # | . x                   |
    # |/                      |
    # +-------> y             v y

    # This can be achieved by multiplying by the following matrix:
    # [[ 0,  1,  0 ],
    #  [ 0,  0, -1 ],
    #  [ 1,  0,  0 ]]

    # Or, in this case, is the same as swapping:
    # (x, y ,z) -> (y, -z, x)
    point_in_camera_coords = np.array(
        [sensor_points[1], sensor_points[2] * -1, sensor_points[0]]
    )

    K = camera.calibration
    # Finally we can use our K matrix to do the actual 3D -> 2D.
    points_2d = np.dot(K, point_in_camera_coords)

    # Remember to normalize the x, y values by the 3rd value.
    points_2d = np.array(
        [
            points_2d[0, :] / points_2d[2, :],
            points_2d[1, :] / points_2d[2, :],
            points_2d[2, :],
        ]
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_in_camera_coords.T)
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2]),
    ]
    pcd.colors = o3d.utility.Vector3dVector(int_color)

    return points_2d[:2, :].T, pcd


def save_point_cloud_to_ply(output_path: str, pcd: o3d.geometry.PointCloud):
    o3d.io.write_point_cloud(output_path, pcd)
