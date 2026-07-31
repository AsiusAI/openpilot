from itertools import combinations
import json

import pytest

from openpilot.system.athena import ble_pairing, websocketd


APP_KEY = "11111111111111111111111111111111111111111111"


@pytest.fixture
def pairing_params(tmp_path, monkeypatch):
  monkeypatch.setattr(websocketd, "PARAMS_DIR", tmp_path)
  monkeypatch.setattr(websocketd, "wall_time", lambda: 1_000)
  monkeypatch.setattr(ble_pairing, "wall_time", lambda: 1_000)
  websocketd.enable_pairing_mode()
  return tmp_path


def test_pairing_request_generates_repeatable_six_color_challenge(pairing_params):
  state = ble_pairing.create_ble_pairing(APP_KEY, "request-1", "phone")
  assert state["status"] == "pending"
  assert len(state["colors"]) == 6
  assert set(state["colors"]) == set(ble_pairing.PAIRING_COLORS)
  for camera_colors in (state["colors"][:3], state["colors"][3:]):
    for left, right in combinations(camera_colors, 2):
      assert ble_pairing.pairing_color_distance_squared(left, right) >= ble_pairing.MIN_PAIRING_COLOR_DISTANCE_SQUARED
  assert state["expiresAt"] == 1_060
  assert ble_pairing.create_ble_pairing(APP_KEY, "request-1", "phone") == state


def test_only_one_pairing_request_can_await_approval(pairing_params):
  ble_pairing.create_ble_pairing(APP_KEY, "request-1")
  with pytest.raises(RuntimeError, match="another app"):
    ble_pairing.create_ble_pairing("21111111111111111111111111111111111111111111", "request-2")


def test_pairing_request_requires_physical_pair_mode(tmp_path, monkeypatch):
  monkeypatch.setattr(websocketd, "PARAMS_DIR", tmp_path)
  with pytest.raises(PermissionError, match="not active"):
    ble_pairing.create_ble_pairing(APP_KEY, "request-1")


def test_approval_authorizes_peer_and_closes_pair_mode(pairing_params, monkeypatch):
  authorized = []
  monkeypatch.setattr(ble_pairing, "authorize_peer", lambda public_key, label=None: authorized.append((public_key, label)) or {"aclEpoch": "4"})
  ble_pairing.create_ble_pairing(APP_KEY, "request-1", "phone")

  approved = ble_pairing.approve_ble_pairing()
  assert approved is not None
  assert approved["status"] == "approved"
  assert approved["aclEpoch"] == 4
  assert approved["expiresAt"] == 1_010
  assert authorized == [(APP_KEY, "phone")]
  assert not websocketd.pairing_mode_active()

  assert ble_pairing.consume_approved_ble_pairing() == approved
  assert ble_pairing.get_ble_pairing() is None


def test_expired_pairing_is_removed(pairing_params):
  path = pairing_params / ble_pairing.ATHENA_BLE_PAIRING_PARAM
  path.write_text(json.dumps({"status": "pending", "expiresAt": 999}))
  assert ble_pairing.get_ble_pairing(now=1_000) is None
  assert not path.exists()
