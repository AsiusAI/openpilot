from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.test import OpenpilotTestCase


class TestCamera(OpenpilotTestCase):
  def test_v1_camera_configs_match_mici(self):
    for sensor in ("ar0231", "ox03c10", "os04c10"):
      with self.subTest(sensor=sensor):
        assert DEVICE_CAMERAS[("v1", sensor)] is DEVICE_CAMERAS[("mici", sensor)]
