import numpy as np

from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.system.sensord.sensors.asius_imu import transform_asius_imu


def test_asius_imu_device_frame_rotation():
  # locationd converts the published Android sensor convention into the
  # camera-aligned device frame with this matrix.
  device_from_sensor = np.array([
    [0.0, 0.0, -1.0],
    [0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0],
  ])
  sensor_transform = np.column_stack([
    transform_asius_imu([1.0, 0.0, 0.0]),
    transform_asius_imu([0.0, 1.0, 0.0]),
    transform_asius_imu([0.0, 0.0, 1.0]),
  ])

  device_transform = device_from_sensor @ sensor_transform @ device_from_sensor.T
  expected = rot_from_euler([0.0, np.radians(-120.0), 0.0])

  np.testing.assert_allclose(device_transform, expected, atol=1e-12)
  np.testing.assert_allclose(sensor_transform.T @ sensor_transform, np.eye(3), atol=1e-12)
  np.testing.assert_allclose(np.linalg.det(sensor_transform), 1.0, atol=1e-12)
