import base64
import json
from types import SimpleNamespace

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from openpilot.system.loggerd.data_api import (
  ENCRYPTION_MAGIC,
  access_document,
  b64url,
  can_wrap_for,
  canonical_json,
  encrypt_object,
  initial_state,
  public_identity,
  recover_state,
  update_state,
  x25519_private,
  x25519_public,
)
from openpilot.system.loggerd.data_uploader import DataUploader
from openpilot.system.loggerd.uploader import Uploader
from openpilot.system.app.identity import bytes_to_identity


def decode64(value: str) -> bytes:
  return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def test_access_document_and_object_crypto() -> None:
  device_key = ed25519.Ed25519PrivateKey.generate()
  reader_key = ed25519.Ed25519PrivateKey.generate()
  owner = public_identity(device_key)
  reader = public_identity(reader_key)
  state = initial_state([reader])

  document = access_document(device_key, state, [reader])
  signature = decode64(document.pop("signature"))
  device_key.public_key().verify(signature, canonical_json(document).encode())
  signed_document = document | {"signature": b64url(signature)}
  recovered = recover_state(device_key, signed_document, [reader])
  assert recovered["keys"][0]["key"] == state["keys"][0]["key"]

  grant = document["keys"][0]["grants"][1]
  shared = x25519_private(reader_key).exchange(x25519_public(owner))
  wrap_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=f"asius-data-access-v1:{owner}:{grant['keyId']}".encode(),
                  info=f"{owner}:{reader}".encode()).derive(shared)
  aad = {**{key: grant[key] for key in ("v", "alg", "from", "to", "keyId", "iv")}, "owner": owner}
  folder_key = AESGCM(wrap_key).decrypt(decode64(grant["iv"]), decode64(grant["ciphertext"]), canonical_json(aad).encode())

  path = "routes/2026-08-10--12-00-00--0/qlog.zst"
  plaintext = b"route bytes"
  encrypted = encrypt_object(owner, path, plaintext, state)
  assert encrypted.startswith(ENCRYPTION_MAGIC)
  offset = len(ENCRYPTION_MAGIC)
  header_length = int.from_bytes(encrypted[offset:offset + 4], "big")
  header = json.loads(encrypted[offset + 4:offset + 4 + header_length])
  object_key = HKDF(
    algorithm=hashes.SHA256(), length=32, salt=f"asius-data-object-v1:{owner}:{header['keyId']}".encode(), info=path.encode()
  ).derive(folder_key)
  assert AESGCM(object_key).decrypt(decode64(header["iv"]), encrypted[offset + 4 + header_length:], canonical_json(header).encode()) == plaintext


def test_revocation_rotates_active_key() -> None:
  first = public_identity(ed25519.Ed25519PrivateKey.generate())
  second = public_identity(ed25519.Ed25519PrivateKey.generate())
  state = initial_state([first, second])
  active = state["activeKey"]

  added = update_state(state, [first, second, public_identity(ed25519.Ed25519PrivateKey.generate())])
  assert added["activeKey"] == active
  revoked = update_state(added, [second])
  assert revoked["activeKey"] != active
  assert len(revoked["keys"]) == 2


def test_rejects_low_order_x25519_recipient() -> None:
  device_key = ed25519.Ed25519PrivateKey.generate()
  low_order_identity = bytes_to_identity(b"\x01" + b"\x00" * 31)
  assert not can_wrap_for(device_key, low_order_identity)


def test_data_uploader_reuses_default_uploader_flow() -> None:
  assert issubclass(DataUploader, Uploader)
  assert DataUploader.step is Uploader.step
  assert DataUploader.upload is Uploader.upload


def test_sharing_adds_only_the_asius_data_recipient(monkeypatch) -> None:
  device_key = ed25519.Ed25519PrivateKey.generate()
  app_reader = public_identity(ed25519.Ed25519PrivateKey.generate())
  asius_reader = public_identity(ed25519.Ed25519PrivateKey.generate())
  published = {}

  class FakeParams:
    state = None

    def get_bool(self, key):
      assert key == "ShareDrivingData"
      return True

    def get(self, key):
      assert key == "DataUploadState"
      return self.state

    def put(self, key, value, block=False):
      assert key == "DataUploadState" and block
      self.state = value

  class Client:
    def get_config(self):
      return {"retentionPublicKey": asius_reader}

    def put_access(self, document):
      published.update(document)

  monkeypatch.setattr("openpilot.system.loggerd.data_uploader.load_authorized_peers", lambda: {app_reader})
  uploader = DataUploader.__new__(DataUploader)
  uploader.private_key = device_key
  uploader.owner = public_identity(device_key)
  uploader.params = FakeParams()
  uploader.client = Client()

  state = uploader.sync_access()
  assert state["readers"] == sorted([app_reader, asius_reader])
  recipients = {grant["to"] for grant in published["keys"][0]["grants"]}
  assert recipients == {uploader.owner, app_reader, asius_reader}


def test_custom_upload_step_only_adds_compression_encryption_and_transport(tmp_path) -> None:
  owner_key = ed25519.Ed25519PrivateKey.generate()
  owner = public_identity(owner_key)
  state = initial_state([])
  source = tmp_path / "qlog"
  source.write_bytes(b"uncompressed log bytes" * 100)
  captured = {}

  class Client:
    def upload(self, path, encrypted, *, plaintext_length, route_start_time):
      captured.update(path=path, encrypted=encrypted, plaintext_length=plaintext_length, route_start_time=route_start_time)
      return SimpleNamespace(status_code=204)

  uploader = DataUploader.__new__(DataUploader)
  uploader.owner = owner
  uploader.client = Client()
  uploader.sync_access = lambda: state

  response = uploader.do_upload("2026-08-10--12-00-00--0/qlog.zst", str(source))
  assert response.status_code == 204
  assert captured["path"] == "routes/2026-08-10--12-00-00--0/qlog.zst"
  assert captured["plaintext_length"] < source.stat().st_size
  assert captured["route_start_time"] > 0
  assert captured["encrypted"].startswith(ENCRYPTION_MAGIC)
  assert b"uncompressed log bytes" not in captured["encrypted"]
