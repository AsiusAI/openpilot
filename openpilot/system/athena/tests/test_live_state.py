import json

from openpilot.system.athena import asius_athenad as athenad


class FakeParams:
  def __init__(self, values):
    self.values = values

  def get(self, key):
    return self.values.get(key)


def test_live_state_includes_operational_services():
  assert {"deviceState", "liveCalibration", "managerState", "onroadEvents", "selfdriveState"} <= set(athenad.LIVE_STATE_SERVICES)


def test_software_state_includes_vamos_progress(tmp_path, monkeypatch):
  state_path = tmp_path / "state.json"
  state_path.write_text(json.dumps({
    "state": "writing",
    "phase": "verifying",
    "progress": 73,
    "image": "system",
  }))
  monkeypatch.setattr(athenad, "VAMOS_UPDATE_STATE_FILE", state_path)

  state = athenad._software_update_state(FakeParams({
    "UpdaterState": "downloading...",
    "UpdateAvailable": b"0",
  }))

  assert state["UpdaterState"] == "downloading..."
  assert state["VamosUpdate"] == {
    "state": "writing",
    "phase": "verifying",
    "progress": 73,
    "image": "system",
  }


def test_invalid_vamos_state_is_ignored(tmp_path, monkeypatch):
  state_path = tmp_path / "state.json"
  state_path.write_text("{")
  monkeypatch.setattr(athenad, "VAMOS_UPDATE_STATE_FILE", state_path)

  assert "VamosUpdate" not in athenad._software_update_state(FakeParams({}))
