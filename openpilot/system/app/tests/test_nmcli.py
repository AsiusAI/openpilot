import subprocess
import unittest
from unittest.mock import patch

from openpilot.system.app import methods as athenad


class TestNmcli(unittest.TestCase):
  @patch("openpilot.system.app.methods.subprocess.check_output", return_value="connected\n")
  def test_nmcli_uses_noninteractive_sudo(self, check_output):
    self.assertEqual(athenad._nmcli(["device", "wifi", "connect", "test"]), "connected\n")
    check_output.assert_called_once_with(
      ["sudo", "-n", "nmcli", "device", "wifi", "connect", "test"],
      stderr=subprocess.STDOUT,
      encoding="utf-8",
    )

  @patch("openpilot.system.app.methods.subprocess.check_output")
  def test_nmcli_redacts_password_but_preserves_error(self, check_output):
    password = "do-not-log-this"
    check_output.side_effect = subprocess.CalledProcessError(10, ["nmcli"], output=f"connection failed for {password}")

    with self.assertRaises(Exception) as context:
      athenad._nmcli(["device", "wifi", "connect", "test", "password", password], sensitive=True)

    self.assertNotIn(password, str(context.exception))
    self.assertTrue("connection failed for <redacted>" in str(context.exception))
