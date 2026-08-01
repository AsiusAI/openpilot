import subprocess

import pytest

from openpilot.system.athena import asius_athenad as athenad


def test_nmcli_uses_noninteractive_sudo(mocker):
  check_output = mocker.patch("openpilot.system.athena.asius_athenad.subprocess.check_output", return_value="connected\n")

  assert athenad._nmcli(["device", "wifi", "connect", "test"]) == "connected\n"
  check_output.assert_called_once_with(
    ["sudo", "-n", "nmcli", "device", "wifi", "connect", "test"],
    stderr=subprocess.STDOUT,
    encoding="utf-8",
  )


def test_nmcli_redacts_password_but_preserves_error(mocker):
  password = "do-not-log-this"
  error = subprocess.CalledProcessError(10, ["nmcli"], output=f"connection failed for {password}")
  mocker.patch("openpilot.system.athena.asius_athenad.subprocess.check_output", side_effect=error)

  with pytest.raises(Exception) as exc_info:
    athenad._nmcli(["device", "wifi", "connect", "test", "password", password], sensitive=True)

  assert password not in str(exc_info.value)
  assert "connection failed for <redacted>" in str(exc_info.value)
