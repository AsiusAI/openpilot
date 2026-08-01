import unittest
from unittest import mock

from openpilot.system.hardware.openpilot_service import service_command


class TestOpenpilotService(unittest.TestCase):
  def test_systemd_service(self):
    with mock.patch("shutil.which", side_effect=lambda name: "/usr/bin/systemctl" if name == "systemctl" else None):
      assert service_command("stop") == ["systemctl", "stop", "comma.service"]
      assert service_command("start") == ["systemctl", "start", "comma.service"]

  def test_runit_service(self):
    with mock.patch("shutil.which", side_effect=lambda name: "/usr/bin/sv" if name == "sv" else None):
      assert service_command("stop") == ["sv", "down", "openpilot"]
      assert service_command("start") == ["sv", "up", "openpilot"]
