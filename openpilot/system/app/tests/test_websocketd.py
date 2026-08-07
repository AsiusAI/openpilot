import json

from cryptography.hazmat.primitives.asymmetric import ed25519

from openpilot.system.app import websocketd
from openpilot.system.athena.identity import bytes_to_identity, identity_to_bytes, is_dongle_id


APP_KEY = "D6xksRG9VaWxAesrqRjb9NePxwhrBLi72SSJyJqahPtw"
SENDER_PRIVATE_KEY = "14wBqpZM9xaSheZzJSMawUKKwhdpChKbZ5eu5ky4Vigw"
SENDER_PUBLIC_KEY = "9C6hybhQ6Aycep9jaUnP6uL9ZYvDjUp1aSkFWPUFJtpj"
RECIPIENT_PRIVATE_KEY = "3ELeRTTg5W5hAYaEFznzFV1jknNFkjHqS8ytwvQEQP1Z"
RECIPIENT_PUBLIC_KEY = "GcQfK48DV9BzDuDeCyV2sShbAAY4vqmK8JSj1NBrwoVZ"
V4_PAYLOAD_FIXTURE = {
  "v": 4,
  "alg": "Ed25519-X25519-HKDF-SHA256-A256GCM",
  "from": SENDER_PUBLIC_KEY,
  "to": RECIPIENT_PUBLIC_KEY,
  "iv": "AAECAwQFBgcICQoL",
  "ts": 1_700_000_000,
  "ciphertext": "JVFAJZzZ7j9NQAtgpBQawF1w4X5us2-YMTt0KPE",
  "sig": "ta34BMs8rf54d2nMHeL30cyekmVA2rHGO8YRpV7WJSnvls0aPBwbvCuZMqHu28Bj8DYgkCXO9vdw8A5xw8GhBg",
}


def private_key(identity: str) -> ed25519.Ed25519PrivateKey:
  return ed25519.Ed25519PrivateKey.from_private_bytes(identity_to_bytes(identity))


def test_ed25519_base58_keys_are_fixed_width():
  key = bytes_to_identity(b"\x00" * 31 + b"\x01")

  assert len(key) == 44
  assert identity_to_bytes(key) == b"\x00" * 31 + b"\x01"
  assert not is_dongle_id(key[1:])


def test_authorized_peer_metadata(tmp_path, monkeypatch):
  monkeypatch.setattr(websocketd, "PARAMS_DIR", tmp_path)
  monkeypatch.setattr(websocketd, "wall_time", lambda: 1_234)

  peer = websocketd.authorize_peer(APP_KEY, label="Karel phone")

  assert peer["publicKey"] == APP_KEY
  assert peer["label"] == "Karel phone"
  assert peer["createdAt"] == 1_234
  assert websocketd.load_authorized_peers()[APP_KEY]["label"] == "Karel phone"


def test_payload_timestamp_valid_rejects_old_messages(monkeypatch):
  monkeypatch.setattr(websocketd, "wall_time", lambda: 1_000)

  assert websocketd.payload_timestamp_valid(1_000)
  assert websocketd.payload_timestamp_valid(941)
  assert not websocketd.payload_timestamp_valid(939)
  assert not websocketd.payload_timestamp_valid("1000")


def test_pairing_mode_window(tmp_path, monkeypatch):
  monkeypatch.setattr(websocketd, "PARAMS_DIR", tmp_path)
  now = 1_000
  monkeypatch.setattr(websocketd, "wall_time", lambda: now)

  assert not websocketd.pairing_mode_active()
  assert websocketd.enable_pairing_mode() == 1_180
  assert websocketd.pairing_mode_active()

  now = 1_181
  assert not websocketd.pairing_mode_active()


def test_pairing_url_uses_pair_route(tmp_path, monkeypatch):
  monkeypatch.setattr(websocketd, "PARAMS_DIR", tmp_path)
  monkeypatch.setattr(websocketd, "pairing_token", lambda recipient: f"token-for-{recipient}")

  assert websocketd.pairing_url(APP_KEY) == f"https://app.asius.ai/pair#token=token-for-{APP_KEY}"
  assert websocketd.pairing_mode_active()


def test_encrypt_payload_matches_web_v4_fixture(monkeypatch):
  monkeypatch.setattr(websocketd, "identity_private_key", lambda: private_key(SENDER_PRIVATE_KEY))
  monkeypatch.setattr(websocketd, "wall_time", lambda: 1_700_000_000)
  monkeypatch.setattr(websocketd.os, "urandom", lambda n: bytes(range(n)))

  payload = websocketd.encrypt_payload("noble fixture", SENDER_PUBLIC_KEY, RECIPIENT_PUBLIC_KEY)

  assert json.loads(payload) == V4_PAYLOAD_FIXTURE


def test_decrypt_payload_matches_web_v4_fixture(monkeypatch):
  monkeypatch.setattr(websocketd, "identity_private_key", lambda: private_key(RECIPIENT_PRIVATE_KEY))
  monkeypatch.setattr(websocketd, "wall_time", lambda: 1_700_000_000)

  payload = json.dumps(V4_PAYLOAD_FIXTURE)

  assert websocketd.decrypt_payload(payload, SENDER_PUBLIC_KEY, RECIPIENT_PUBLIC_KEY) == "noble fixture"


def test_decrypt_payload_rejects_v3():
  v3_payload = {
    "v": 3,
    "alg": "Ed25519-X25519-A256GCM",
    "from": SENDER_PUBLIC_KEY,
    "to": RECIPIENT_PUBLIC_KEY,
    "iv": "AAECAwQFBgcICQoL",
    "ts": 1_700_000_000,
    "ciphertext": "e7gbSbGKsKidcZ1WRkoe7cJ4WTYJDoCJ8cNv1iw",
    "sig": "K0TPQkudIrpI1R0iE_HIvuUK_YWNbBoAZHSJ5D96l4e94qOq4GQTAOuItr5LywoXqW4BMvvTQhNIBHn2lI-kBQ",
  }

  assert websocketd.decrypt_payload(json.dumps(v3_payload), SENDER_PUBLIC_KEY, RECIPIENT_PUBLIC_KEY) is None
