import tempfile
import json
from pathlib import Path
from unittest import mock

from Crypto.PublicKey import RSA
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

from openpilot.common import api
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.system.athena.identity import (
  DONGLE_ID_LEN,
  dongle_id_from_public_key,
  public_key_from_dongle_id,
)
from openpilot.system.athena.registration import (
  register,
  UNREGISTERED_DONGLE_ID,
)
from openpilot.system.athena import registration
from openpilot.system.athena.tests.helpers import MockResponse
from openpilot.common.hardware.hw import Paths


class TestRegistration(OpenpilotTestCase):

  def setup_method(self):
    self.params = Params()

    persist_root = Path(self.enterContext(tempfile.TemporaryDirectory())) / "persist"
    self.enterContext(mock.patch.object(Paths, "persist_root", staticmethod(lambda: str(persist_root))))

    persist_dir = Path(Paths.persist_root()) / "comma"
    persist_dir.mkdir(parents=True, exist_ok=True)

    self.priv_key = persist_dir / "id_ed25519"
    self.pub_key = persist_dir / "id_ed25519.pub"
    self.enterContext(mock.patch.object(api, "ASIUS", True))
    self.enterContext(mock.patch.object(registration, "ASIUS", True))

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
    mocker.patch("openpilot.system.athena.registration.ed25519.Ed25519PrivateKey.generate", side_effect=OSError("no write"))

    dongle = register()
    assert dongle == UNREGISTERED_DONGLE_ID
    assert self.params.get("DongleId") == dongle


class TestCommaRegistration(OpenpilotTestCase):

  def setup_method(self):
    self.params = Params()

    persist_root = Path(self.enterContext(tempfile.TemporaryDirectory())) / "persist"
    self.enterContext(mock.patch.object(Paths, "persist_root", staticmethod(lambda: str(persist_root))))
    self.enterContext(mock.patch.object(api, "ASIUS", False))
    self.enterContext(mock.patch.object(registration, "ASIUS", False))

    persist_dir = Path(Paths.persist_root()) / "comma"
    persist_dir.mkdir(parents=True, exist_ok=True)
    self.priv_key = persist_dir / "id_rsa"
    self.pub_key = persist_dir / "id_rsa.pub"
    self.dongle_id = persist_dir / "dongle_id"

  def _generate_keys(self):
    key = RSA.generate(2048)
    self.priv_key.write_bytes(key.export_key())
    self.pub_key.write_bytes(key.publickey().export_key())

  def test_valid_cache(self, mocker):
    self._generate_keys()
    dongle = "DONGLE_ID_123"
    request = mocker.patch.object(registration, "api_get", autospec=True)

    for persist, params in [(True, True), (True, False), (False, True)]:
      self.params.put("DongleId", dongle if params else "", block=True)
      self.dongle_id.write_text(dongle if persist else "")
      assert register() == dongle
      assert not request.called

  def test_no_keys(self, mocker):
    request = mocker.patch.object(registration, "api_get", autospec=True)
    assert register() == UNREGISTERED_DONGLE_ID
    assert not request.called

  def test_missing_cache(self, mocker):
    self._generate_keys()
    request = mocker.patch.object(registration, "api_get", autospec=True)
    request.return_value = MockResponse(json.dumps({'dongle_id': "DONGLE_ID_123"}), 200)

    assert register() == "DONGLE_ID_123"
    assert register() == "DONGLE_ID_123"
    assert request.call_count == 1

  def test_unregistered(self, mocker):
    self._generate_keys()
    request = mocker.patch.object(registration, "api_get", autospec=True)
    request.return_value = MockResponse(None, 402)

    assert register() == UNREGISTERED_DONGLE_ID
    assert request.call_count == 1
