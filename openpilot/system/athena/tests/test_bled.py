import asyncio

from openpilot.common.test import OpenpilotTestCase
from openpilot.system.athena import bled


APP_KEY = "11111111111111111111111111111111111111111111"
DEVICE_KEY = "21111111111111111111111111111111111111111111"


def make_engine():
  engine = object.__new__(bled.BlePeerEngine)
  engine.dongle_id = DEVICE_KEY
  engine.active_peers = {}
  engine.peer_clocks = {}
  return engine


async def _true():
  return True


class TestBled(OpenpilotTestCase):
  def test_advertisement_uses_product_name(self):
    assert bled.Advertisement().LocalName == bled.DEVICE_NAME

  def test_advertisement_refresh_reregisters_after_missing_registration(self, monkeypatch):
    async def run():
      calls = []

      async def checked_call(_bus, message):
        calls.append(message.member)
        if message.member == "UnregisterAdvertisement":
          raise RuntimeError("already removed")

      monkeypatch.setattr(bled, "checked_call", checked_call)
      bus = bled.MessageBus.__new__(bled.MessageBus)
      await bled.refresh_advertisement(bus, "/org/bluez/hci0")
      assert calls == ["UnregisterAdvertisement", "RegisterAdvertisement"]

    asyncio.run(run())

  def test_pairing_approval_waits_for_active_peer(self, monkeypatch):
    async def run():
      approved = {
        "status": "approved",
        "publicKey": APP_KEY,
        "requestId": "request-1",
        "aclEpoch": 4,
      }
      cleared = []
      engine = make_engine()

      monkeypatch.setattr(bled, "get_ble_pairing", lambda: approved)
      monkeypatch.setattr(bled, "clear_ble_pairing", lambda: cleared.append(True))

      async def send_body(*_args, **_kwargs):
        raise AssertionError("inactive peer must not receive approval")

      engine.send_body = send_body
      assert not await engine.send_pairing_approval()
      assert cleared == []

    asyncio.run(run())

  def test_pairing_approval_is_retained_until_delivered(self, monkeypatch):
    async def run():
      approved = {
        "status": "approved",
        "publicKey": APP_KEY,
        "requestId": "request-1",
        "aclEpoch": 4,
      }
      cleared = []
      sent = []
      engine = make_engine()
      engine.active_peers[APP_KEY] = 1.0

      monkeypatch.setattr(bled, "get_ble_pairing", lambda: approved)
      monkeypatch.setattr(bled, "clear_ble_pairing", lambda: cleared.append(True))

      async def send_body(recipient, body, initial=False):
        sent.append((recipient, body, initial))
        return len(sent) > 1

      engine.send_body = send_body
      assert not await engine.send_pairing_approval()
      assert cleared == []
      assert await engine.send_pairing_approval()
      assert cleared == [True]
      assert sent[-1][1]["type"] == "pair-response"
      assert sent[-1][1]["aclEpoch"] == 4

    asyncio.run(run())

  def test_authorized_pairing_retry_delivers_and_clears_approval(self, monkeypatch):
    async def run():
      approved = {
        "status": "approved",
        "publicKey": APP_KEY,
        "requestId": "request-1",
        "aclEpoch": 4,
      }
      cleared = []
      sent = []
      engine = make_engine()

      monkeypatch.setattr(bled, "unpack_peer_message", lambda _payload, _recipient, validate_timestamp=True: (
        APP_KEY,
        {"type": "ble-pair-request", "publicKey": APP_KEY, "requestId": "request-1"},
        False,
      ))
      monkeypatch.setattr(bled, "load_authorized_peers", lambda: {APP_KEY: {"aclEpoch": 4}})
      monkeypatch.setattr(bled, "get_ble_pairing", lambda: approved)
      monkeypatch.setattr(bled, "clear_ble_pairing", lambda: cleared.append(True))

      async def send_body(recipient, body, initial=False):
        sent.append((recipient, body, initial))
        return True

      engine.send_body = send_body
      await engine.handle_encrypted(b"authenticated-envelope")
      assert sent[-1][1]["type"] == "pair-response"
      assert cleared == [True]

    asyncio.run(run())

  def test_authenticated_pairing_error_preserves_request_id(self, monkeypatch):
    async def run():
      engine = make_engine()
      errors = []
      monkeypatch.setattr(bled, "unpack_peer_message", lambda _payload, _recipient, validate_timestamp=True: (
        APP_KEY,
        {"type": "ble-pair-request", "requestId": "request-2"},
        False,
      ))

      async def send_error(recipient, message, request_id=None):
        errors.append((recipient, message, request_id))

      engine.send_error = send_error
      await engine.send_authenticated_error(b"authenticated-envelope", PermissionError("pairing mode is not active"))
      assert errors == [(APP_KEY, "pairing mode is not active", "request-2")]

    asyncio.run(run())

  def test_unauthenticated_payload_gets_no_error(self, monkeypatch):
    async def run():
      engine = make_engine()
      errors = []
      monkeypatch.setattr(bled, "unpack_peer_message", lambda _payload, _recipient, validate_timestamp=True: (APP_KEY, None, True))

      async def send_error(recipient, message, request_id=None):
        errors.append((recipient, message, request_id))

      engine.send_error = send_error
      await engine.send_authenticated_error(b"invalid-envelope", ValueError("authentication failed"))
      assert errors == []

    asyncio.run(run())

  def test_ble_message_ignores_wall_clock(self, monkeypatch):
    async def run():
      engine = make_engine()
      handled = []
      monkeypatch.setattr(bled.time, "monotonic", lambda: 100.0)
      monkeypatch.setattr(bled, "peer_message_timestamp", lambda _payload: 1_700_000_000)
      monkeypatch.setattr(bled, "unpack_peer_message", lambda _payload, _recipient, validate_timestamp=True: (
        APP_KEY,
        {"type": "ble-pair-request", "publicKey": APP_KEY, "requestId": "request-1"},
        False,
      ) if not validate_timestamp else (_ for _ in ()).throw(AssertionError("Bluetooth must skip wall-time validation")))
      monkeypatch.setattr(bled, "load_authorized_peers", dict)

      async def handle_pair_request(sender, body):
        handled.append((sender, body))

      engine.handle_pair_request = handle_pair_request
      await engine.handle_encrypted(b"authenticated-envelope")

      assert handled[0][0] == APP_KEY
      assert engine.peer_clocks[APP_KEY] == (1_700_000_000, 100.0)

    asyncio.run(run())

  def test_ble_response_uses_peer_timestamp(self, monkeypatch):
    async def run():
      engine = make_engine()
      engine.peer_clocks[APP_KEY] = (1_700_000_000, 100.0)
      sent = []
      engine.tx = type("Tx", (), {"send_text": lambda _self, text, initial=False: sent.append((text, initial)) or _true()})()
      monkeypatch.setattr(bled.time, "monotonic", lambda: 105.0)
      monkeypatch.setattr(bled, "pack_peer_message", lambda sender, recipient, body, timestamp=None: (
        sender, recipient, body, timestamp
      ))

      assert await engine.send_body(APP_KEY, {"type": "ble-pair-challenge"})
      assert sent == [((DEVICE_KEY, APP_KEY, {"type": "ble-pair-challenge"}, 1_700_000_005), False)]

    asyncio.run(run())

  def test_ble_pairing_expiry_uses_peer_time(self, monkeypatch):
    async def run():
      engine = make_engine()
      engine.peer_clocks[APP_KEY] = (1_700_000_000, 100.0)
      sent = []
      monkeypatch.setattr(bled.time, "monotonic", lambda: 105.0)
      monkeypatch.setattr(bled, "create_ble_pairing", lambda *_args: {
        "colors": ["red", "green", "blue", "amber", "turquoise", "violet"],
        "expiresAt": 1_300,
      })

      async def send_body(recipient, body, initial=False):
        sent.append((recipient, body, initial))
        return True

      engine.send_body = send_body
      await engine.handle_pair_request(APP_KEY, {
        "publicKey": APP_KEY,
        "requestId": "request-1",
      })

      assert sent[0][1]["expiresAt"] == 1_700_000_005 + bled.PAIRING_CHALLENGE_SECONDS

    asyncio.run(run())
