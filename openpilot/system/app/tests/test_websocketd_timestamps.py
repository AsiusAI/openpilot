import json

from openpilot.system.app import websocketd
from openpilot.system.app.tests.test_websocketd import (
  RECIPIENT_PRIVATE_KEY,
  RECIPIENT_PUBLIC_KEY,
  SENDER_PRIVATE_KEY,
  SENDER_PUBLIC_KEY,
  V4_PAYLOAD_FIXTURE,
  private_key,
)


def test_decrypt_payload_can_skip_wall_clock_validation(monkeypatch):
  monkeypatch.setattr(websocketd, "identity_private_key", lambda: private_key(RECIPIENT_PRIVATE_KEY))
  monkeypatch.setattr(websocketd, "wall_time", lambda: 1_800_000_000)

  payload = json.dumps(V4_PAYLOAD_FIXTURE)

  assert websocketd.decrypt_payload(payload, SENDER_PUBLIC_KEY, RECIPIENT_PUBLIC_KEY) is None
  assert websocketd.decrypt_payload(payload, SENDER_PUBLIC_KEY, RECIPIENT_PUBLIC_KEY, validate_timestamp=False) == "noble fixture"


def test_encrypt_payload_can_use_peer_timestamp(monkeypatch):
  monkeypatch.setattr(websocketd, "identity_private_key", lambda: private_key(SENDER_PRIVATE_KEY))
  monkeypatch.setattr(websocketd, "wall_time", lambda: 1_800_000_000)
  monkeypatch.setattr(websocketd.os, "urandom", lambda n: bytes(range(n)))

  payload = websocketd.encrypt_payload("noble fixture", SENDER_PUBLIC_KEY, RECIPIENT_PUBLIC_KEY, timestamp=1_700_000_000)

  assert json.loads(payload)["ts"] == 1_700_000_000
