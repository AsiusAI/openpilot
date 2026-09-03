from openpilot.selfdrive.selfdrived.selfdrived import get_camera_packets


def test_no_dcam_omits_driver_camera():
  assert get_camera_packets(no_dcam=True) == ["narrowRoadCameraState", "wideRoadCameraState"]


def test_driver_camera_enabled_by_default():
  assert get_camera_packets(no_dcam=False) == ["narrowRoadCameraState", "cabinCameraState", "wideRoadCameraState"]
