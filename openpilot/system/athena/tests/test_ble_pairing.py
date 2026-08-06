from pathlib import Path
import tempfile

from openpilot.common.test import OpenpilotTestCase
from openpilot.system.athena import ble_pairing, websocketd


APP_KEY = "11111111111111111111111111111111111111111111"
SECOND_APP_KEY = "21111111111111111111111111111111111111111111"


def tmp_path():
  with tempfile.TemporaryDirectory() as directory:
    yield Path(directory)


def pairing_params(tmp_path, monkeypatch):
  monkeypatch.setattr(websocketd, "PARAMS_DIR", tmp_path)
  monkeypatch.setattr(websocketd, "wall_time", lambda: 1_000)
  websocketd.enable_pairing_mode()
  return tmp_path


class TestBlePairing(OpenpilotTestCase):
  def test_first_request_is_authorized_and_closes_pairing_mode(self, pairing_params, monkeypatch):
    authorized = []
    monkeypatch.setattr(ble_pairing, "authorize_peer", lambda public_key, label=None: authorized.append((public_key, label)) or {"aclEpoch": "4"})

    peer = ble_pairing.authorize_ble_peer(APP_KEY, "request-1", "phone")

    assert peer == {"publicKey": APP_KEY, "requestId": "request-1", "aclEpoch": 4}
    assert authorized == [(APP_KEY, "phone")]
    assert not websocketd.pairing_mode_active()

    with self.assertRaisesRegex(PermissionError, "not active"):
      ble_pairing.authorize_ble_peer(SECOND_APP_KEY, "request-2")

  def test_pairing_request_requires_physical_pair_mode(self, tmp_path, monkeypatch):
    monkeypatch.setattr(websocketd, "PARAMS_DIR", tmp_path)
    with self.assertRaisesRegex(PermissionError, "not active"):
      ble_pairing.authorize_ble_peer(APP_KEY, "request-1")

  def test_invalid_request_does_not_close_pairing_mode(self, pairing_params):
    with self.assertRaisesRegex(ValueError, "invalid app public key"):
      ble_pairing.authorize_ble_peer("invalid", "request-1")
    assert websocketd.pairing_mode_active()
