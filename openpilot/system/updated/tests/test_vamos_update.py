from pathlib import Path

import pytest

from openpilot.system.updated import updated


def test_prepare_vamos_update_stages_without_activation(monkeypatch, tmp_path: Path) -> None:
  commands: list[list[str]] = []
  consistency: list[bool] = []
  alerts: list[bool] = []

  monkeypatch.setattr(updated.HARDWARE, "get_os_version", lambda: "17.2")
  monkeypatch.setattr(updated, "OVERLAY_MERGED", str(tmp_path))
  monkeypatch.setattr(updated, "set_consistent_flag", consistency.append)
  monkeypatch.setattr(updated, "set_offroad_alert", lambda name, enabled: alerts.append(enabled))

  def fake_run(command: list[str], cwd: str | None = None) -> str:
    if command[0] == "bash":
      assert cwd == str(tmp_path)
      return "18.1"
    commands.append(command)
    return ""

  monkeypatch.setattr(updated, "run", fake_run)
  monkeypatch.setattr(updated, "run_vamos_update", lambda command: commands.append(command) or "")

  assert updated.prepare_vamos_update()
  assert consistency == [False]
  assert alerts == [True]
  assert commands == [[
    "sudo", "/usr/bin/vamos-update", "install",
    str(tmp_path / "openpilot/system/hardware/asius/vamos.json"),
    "--defer-activation",
  ]]


def test_prepare_vamos_update_skips_matching_version(monkeypatch, tmp_path: Path) -> None:
  monkeypatch.setattr(updated.HARDWARE, "get_os_version", lambda: "18.1")
  monkeypatch.setattr(updated, "OVERLAY_MERGED", str(tmp_path))
  monkeypatch.setattr(updated, "run", lambda command, cwd=None: "18.1")

  assert not updated.prepare_vamos_update()


def test_activate_vamos_update_clears_alert_on_failure(monkeypatch) -> None:
  alerts: list[bool] = []
  monkeypatch.setattr(updated, "set_offroad_alert", lambda name, enabled: alerts.append(enabled))
  monkeypatch.setattr(updated, "run", lambda command, cwd=None: (_ for _ in ()).throw(RuntimeError("EFI failure")))

  with pytest.raises(RuntimeError, match="EFI failure"):
    updated.activate_vamos_update()

  assert alerts == [False]


@pytest.mark.parametrize(
  ("vamos", "update_available", "user_request", "expected"),
  [
    (True, False, updated.UserRequest.NONE, True),
    (True, False, updated.UserRequest.CHECK, True),
    (True, False, updated.UserRequest.FETCH, False),
    (True, True, updated.UserRequest.NONE, False),
    (False, False, updated.UserRequest.NONE, False),
  ],
)
def test_should_skip_noop_vamos_fetch(monkeypatch, vamos: bool, update_available: bool,
                                      user_request: int, expected: bool) -> None:
  monkeypatch.setattr(updated, "VAMOS", vamos)

  assert updated.should_skip_noop_vamos_fetch(update_available, user_request) is expected


def test_vamos_stdout_progress_fallback(monkeypatch) -> None:
  progress: list[int] = []

  class FakeParams:
    def put(self, key: str, value: int, block: bool = False) -> None:
      assert key == "UpdaterProgress"
      assert block
      progress.append(value)

  class FakeProcess:
    stdout = iter([
      "vamos-update: writing system to /dev/rootfs_b\n",
      "vamos-update: system: 50%\n",
      "vamos-update: verifying system from disk\n",
      "vamos-update: writing esp to /dev/esp_b\n",
      "vamos-update: esp: 100%\n",
      "vamos-update: verifying esp from disk\n",
    ])

    @staticmethod
    def wait() -> int:
      return 0

  monkeypatch.setattr(updated, "Params", FakeParams)
  monkeypatch.setattr(updated.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

  output = updated.run_vamos_update(["vamos-update", "install", "vamos.json"])

  assert "verifying system" in output
  assert progress[-1] == 98
  assert {22, 45, 90, 95} <= set(progress)
