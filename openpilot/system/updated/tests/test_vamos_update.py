from pathlib import Path
import tempfile
import json

from openpilot.common.test import OpenpilotTestCase
from openpilot.system.updated import updated, vamos_update


def tmp_path():
  with tempfile.TemporaryDirectory() as directory:
    yield Path(directory)


class TestVamosUpdate(OpenpilotTestCase):
  @staticmethod
  def write_manifest(tmp_path: Path, version: str) -> None:
    manifest = tmp_path / "openpilot/system/hardware/v1/vamos.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"version": version}))

  def test_prepare_vamos_update_stages_without_activation(self, monkeypatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []
    consistency: list[bool] = []
    alerts: list[bool] = []

    monkeypatch.setattr(vamos_update, "set_offroad_alert", lambda name, enabled: alerts.append(enabled))
    self.write_manifest(tmp_path, "18.1")
    monkeypatch.setattr(vamos_update, "run_vamos_update", lambda command: commands.append(command) or "")

    assert vamos_update.prepare_vamos_update(str(tmp_path), "17.2", consistency.append)
    assert consistency == [False]
    assert alerts == [True]
    assert commands == [[
      "sudo", "/usr/bin/vamos-update", "install",
      str(tmp_path / "openpilot/system/hardware/v1/vamos.json"),
      "--defer-activation",
    ]]

  def test_prepare_vamos_update_skips_matching_version(self, monkeypatch, tmp_path: Path) -> None:
    self.write_manifest(tmp_path, "18.1")

    assert not vamos_update.prepare_vamos_update(str(tmp_path), "18.1", lambda consistent: None)

  def test_activate_vamos_update_clears_alert_on_failure(self, monkeypatch) -> None:
    alerts: list[bool] = []
    monkeypatch.setattr(vamos_update, "set_offroad_alert", lambda name, enabled: alerts.append(enabled))
    monkeypatch.setattr(vamos_update.subprocess, "check_output", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("EFI failure")))

    with self.assertRaisesRegex(RuntimeError, "EFI failure"):
      vamos_update.activate_vamos_update()

    assert alerts == [False]

  def test_should_skip_noop_vamos_fetch(self, monkeypatch) -> None:
    monkeypatch.setattr(vamos_update, "vamos_update_supported", lambda: True)
    cases = [
      (False, updated.UserRequest.NONE, True),
      (False, updated.UserRequest.CHECK, True),
      (False, updated.UserRequest.FETCH, False),
      (True, updated.UserRequest.NONE, False),
    ]
    for update_available, user_request, expected in cases:
      with self.subTest(update_available=update_available, user_request=user_request):
        assert vamos_update.should_skip_noop_vamos_fetch(update_available, user_request, updated.UserRequest.FETCH) is expected

    monkeypatch.setattr(vamos_update, "vamos_update_supported", lambda: False)
    assert not vamos_update.should_skip_noop_vamos_fetch(False, updated.UserRequest.NONE, updated.UserRequest.FETCH)

  def test_vamos_stdout_progress_fallback(self, monkeypatch) -> None:
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

    monkeypatch.setattr(vamos_update, "Params", FakeParams)
    monkeypatch.setattr(vamos_update.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    output = vamos_update.run_vamos_update(["vamos-update", "install", "vamos.json"])

    assert "verifying system" in output
    assert progress[-1] == 98
    assert {22, 45, 90, 95} <= set(progress)
