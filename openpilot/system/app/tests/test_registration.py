import tempfile
from pathlib import Path
from unittest import mock

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.system.app import identity
from openpilot.system.app.identity import (
  DONGLE_ID_LEN,
  dongle_id_from_public_key,
  public_key_from_dongle_id,
)
from openpilot.system.app.registration import (
  register,
  UNREGISTERED_DONGLE_ID,
)


class TestRegistration(OpenpilotTestCase):

  def setup_method(self):
    self.params = Params()

    persist_dir = Path(self.enterContext(tempfile.TemporaryDirectory())) / "persist" / "comma"
    persist_dir.mkdir(parents=True, exist_ok=True)

    self.priv_key = persist_dir / "id_ed25519"
    self.pub_key = persist_dir / "id_ed25519.pub"
    self.enterContext(mock.patch.object(identity, "PRIVATE_KEY_PATH", self.priv_key))
    self.enterContext(mock.patch.object(identity, "PUBLIC_KEY_PATH", self.pub_key))

  def _generate_keys(self) -> str:
    key = ed25519.Ed25519PrivateKey.generate()
    self.priv_key.write_bytes(key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()))
    self.pub_key.write_bytes(key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo))
    return self.pub_key.read_text()

  def test_public_key_roundtrip(self):
    public_key = self._generate_keys()
    dongle = dongle_id_from_public_key(public_key)

    assert len(dongle) == DONGLE_ID_LEN
    assert public_key_from_dongle_id(dongle) == public_key

  def test_valid_cache(self):
    public_key = self._generate_keys()
    dongle = dongle_id_from_public_key(public_key)

    self.params.put("DongleId", dongle, block=True)
    assert register() == dongle

  def test_creates_missing_ed25519_keys(self):
    dongle = register()

    assert self.priv_key.exists()
    assert self.pub_key.exists()
    assert dongle == dongle_id_from_public_key(self.pub_key.read_text())
    assert self.params.get("DongleId") == dongle

  def test_missing_cache(self):
    public_key = self._generate_keys()
    dongle = dongle_id_from_public_key(public_key)

    assert register() == dongle
    assert register() == dongle
    assert self.params.get("DongleId") == dongle

  def test_invalid_cache_is_replaced(self):
    public_key = self._generate_keys()
    self.params.put("DongleId", "0000000000000000", block=True)

    dongle = register()
    assert dongle == dongle_id_from_public_key(public_key)

  def test_key_create_failure(self, mocker):
    mocker.patch("openpilot.system.app.identity.ed25519.Ed25519PrivateKey.generate", side_effect=OSError("no write"))

    dongle = register()
    assert dongle == UNREGISTERED_DONGLE_ID
    assert self.params.get("DongleId") == dongle
