from openpilot.common.transformations.camera import DEVICE_CAMERAS


def test_v1_camera_configs_match_mici():
  for sensor in ("ar0231", "ox03c10", "os04c10"):
    assert DEVICE_CAMERAS[("v1", sensor)] is DEVICE_CAMERAS[("mici", sensor)]
