import math


def transform_asius_imu(v: list[float]) -> list[float]:
  """Rotate comma's LSM6DS3 output convention into the Asius sensor frame.

  The Asius v1 PCB is flipped relative to the road camera, whose optical axis
  is 60 degrees from the PCB plane. locationd's sensor-to-device conversion
  conjugates this rotation, producing the required -120 degree device-frame
  pitch correction.
  """
  x, y, z = v
  sin_120 = math.sqrt(3.0) / 2.0
  return [
    -0.5 * x + sin_120 * z,
    y,
    -sin_120 * x - 0.5 * z,
  ]
