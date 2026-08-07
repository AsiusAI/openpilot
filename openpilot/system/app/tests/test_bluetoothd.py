import asyncio

from openpilot.common.test import OpenpilotTestCase
from openpilot.system.app import bluetoothd as bled


APP_KEY = "11111111111111111111111111111111111111111111"
DEVICE_KEY = "21111111111111111111111111111111111111111111"


def make_engine():
  engine = object.__new__(bled.BlePeerEngine)
  engine.dongle_id = DEVICE_KEY
  engine.active_peers = {}
  engine.peer_clocks = {}
  return engine


def memory_params(monkeypatch):
  values = {}

  class MemoryParams:
    def get(self, key):
      return values.get(key)

    def put(self, key, value, block=False):
      values[key] = value

    def remove(self, key):
      values.pop(key, None)

  monkeypatch.setattr(bled, "Params", MemoryParams)
  return values


async def _true():
  return True


class TestBled(OpenpilotTestCase):
  def test_frame_round_trip(self):
    for length in (0, 1, 14, 15, 174, 175, 4096):
      with self.subTest(length=length):
        payload = bytes(index % 251 for index in range(length))
        frames = bled.encode_frames(payload, frame_bytes=20, message_id=42)
        assembler = bled.FrameAssembler()
        result = None
        for frame in frames:
          result = assembler.feed("peer", frame)
        assert result == payload
        assert all(len(frame) <= 20 for frame in frames)

  def test_frame_assemblers_are_isolated_by_peer(self):
    first = bled.encode_frames(b"a" * 30, frame_bytes=20, message_id=1)
    second = bled.encode_frames(b"b" * 30, frame_bytes=20, message_id=2)
    assembler = bled.FrameAssembler()

    assert assembler.feed("first", first[0]) is None
    assert assembler.feed("second", second[0]) is None
    assert assembler.feed("first", first[1]) is None
    assert assembler.feed("second", second[1]) is None
    assert assembler.feed("first", first[2]) == b"a" * 30
    assert assembler.feed("second", second[2]) == b"b" * 30

  def test_invalid_frame_sequence_discards_message(self):
    frames = bled.encode_frames(b"message long enough for frames", frame_bytes=14, message_id=7)
    assembler = bled.FrameAssembler()
    assert assembler.feed("peer", frames[0]) is None
    with self.assertRaisesRegex(bled.FrameError, "sequence"):
      assembler.feed("peer", frames[2])
    with self.assertRaisesRegex(bled.FrameError, "no start"):
      assembler.feed("peer", frames[1])

  def test_expired_frame_message_is_discarded(self):
    frames = bled.encode_frames(b"two frames", frame_bytes=12, message_id=9)
    assembler = bled.FrameAssembler(timeout_seconds=1)
    assert assembler.feed("peer", frames[0], now=5) is None
    with self.assertRaisesRegex(bled.FrameError, "no start"):
      assembler.feed("peer", frames[1], now=7)

  def test_rejects_invalid_frame_header(self):
    assembler = bled.FrameAssembler()
    with self.assertRaisesRegex(bled.FrameError, "shorter"):
      assembler.feed("peer", b"\x01")
    with self.assertRaisesRegex(bled.FrameError, "version"):
      assembler.feed("peer", bled.FRAME_HEADER.pack(bled.PROTOCOL_VERSION + 1, bled.FRAME_START | bled.FRAME_END, 1, 0))

  def test_pairing_mode_window(self, memory_params, monkeypatch):
    now = 1_000
    monkeypatch.setattr(bled.time, "time", lambda: now)

    assert not bled.pairing_mode_active()
    assert bled.enable_pairing_mode() == 1_300
    assert bled.pairing_mode_active()

    now = 1_301
    assert not bled.pairing_mode_active()

  def test_pairing_requires_physical_mode(self, monkeypatch):
    async def run():
      engine = make_engine()
      monkeypatch.setattr(bled, "pairing_mode_active", lambda: False)
      with self.assertRaisesRegex(PermissionError, "not active"):
        await engine.handle_pair_request(APP_KEY, {"publicKey": APP_KEY, "requestId": "request-1"})

    asyncio.run(run())

  def test_invalid_pairing_request_does_not_close_pairing_mode(self, monkeypatch):
    async def run():
      engine = make_engine()
      closed = []
      monkeypatch.setattr(bled, "pairing_mode_active", lambda: True)
      monkeypatch.setattr(bled, "disable_pairing_mode", lambda: closed.append(True))
      with self.assertRaisesRegex(ValueError, "invalid app public key"):
        await engine.handle_pair_request("invalid", {"publicKey": "invalid", "requestId": "request-1"})
      assert closed == []

    asyncio.run(run())

  def test_advertisement_only_identifies_product_in_pairing_mode(self):
    advertisement = bled.Advertisement()
    assert advertisement.LocalName == ""
    assert advertisement.ServiceUUIDs == []
    assert not advertisement.Discoverable

    advertisement.pairing = True
    assert advertisement.LocalName == bled.DEVICE_NAME
    assert advertisement.ServiceUUIDs == [bled.BLE_SERVICE_UUID]
    assert advertisement.Discoverable

  def test_gatt_requires_an_encrypted_link(self):
    assert "encrypt-write" in bled.RxCharacteristic(lambda _device, _frame: None).Flags
    assert "encrypt-read" in bled.TxCharacteristic().Flags

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

  def test_authorized_pairing_retry_delivers_response(self, monkeypatch):
    async def run():
      sent = []
      engine = make_engine()

      monkeypatch.setattr(bled, "unpack_peer_message", lambda _payload, _recipient, validate_timestamp=True: (
        APP_KEY,
        {"type": "ble-pair-request", "publicKey": APP_KEY, "requestId": "request-1"},
        False,
      ))
      monkeypatch.setattr(bled, "load_authorized_peers", lambda: {APP_KEY: {}})

      async def send_body(recipient, body, initial=False):
        sent.append((recipient, body, initial))
        return True

      engine.send_body = send_body
      await engine.handle_encrypted(b"authenticated-envelope")
      assert sent[-1][1] == {
        "type": "pair-response",
        "publicKey": DEVICE_KEY,
        "device-type": bled.DEVICE_TYPE,
      }

    asyncio.run(run())

  def test_ble_ping_echoes_identifier(self, monkeypatch):
    async def run():
      engine = make_engine()
      sent = []
      monkeypatch.setattr(bled, "unpack_peer_message", lambda *_args, **_kwargs: (
        APP_KEY,
        {"type": "ble-ping", "id": "ping-1"},
        False,
      ))
      monkeypatch.setattr(bled, "load_authorized_peers", lambda: {APP_KEY: {}})

      async def send_body(recipient, body, initial=False):
        sent.append((recipient, body, initial))
        return True

      engine.send_body = send_body
      await engine.handle_encrypted(b"authenticated-envelope")

      assert sent == [(APP_KEY, {"type": "ble-pong", "id": "ping-1"}, False)]

    asyncio.run(run())

  def test_terminal_event_routes_to_terminal_manager(self, monkeypatch):
    async def run():
      engine = make_engine()
      handled = []
      engine.terminal_manager = type("Terminal", (), {"handle": lambda _self, peer, payload: handled.append((peer, payload))})()
      monkeypatch.setattr(bled, "unpack_peer_message", lambda *_args, **_kwargs: (
        APP_KEY,
        {"type": "event", "name": "terminal", "payload": {"action": "open", "sessionId": "terminal-1"}},
        False,
      ))
      monkeypatch.setattr(bled, "load_authorized_peers", lambda: {APP_KEY: {}})

      await engine.handle_encrypted(b"authenticated-envelope")

      assert handled == [(APP_KEY, {"action": "open", "sessionId": "terminal-1"})]

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

      assert await engine.send_body(APP_KEY, {"type": "ble-session"})
      assert sent == [((DEVICE_KEY, APP_KEY, {"type": "ble-session"}, 1_700_000_005), False)]

    asyncio.run(run())

  def test_pairing_request_immediately_authorizes_and_responds(self, monkeypatch):
    async def run():
      engine = make_engine()
      sent = []
      monkeypatch.setattr(bled.time, "monotonic", lambda: 105.0)
      authorized = []
      monkeypatch.setattr(bled, "pairing_mode_active", lambda: True)
      monkeypatch.setattr(bled, "authorize_peer", lambda key, label=None: authorized.append((key, label)))
      monkeypatch.setattr(bled, "disable_pairing_mode", lambda: None)

      async def send_body(recipient, body, initial=False):
        sent.append((recipient, body, initial))
        return True

      engine.send_body = send_body
      await engine.handle_pair_request(APP_KEY, {
        "publicKey": APP_KEY,
        "requestId": "request-1",
      })

      assert engine.active_peers[APP_KEY] == 105.0
      assert authorized == [(APP_KEY, None)]
      assert sent == [(APP_KEY, {
        "type": "pair-response",
        "publicKey": DEVICE_KEY,
        "device-type": bled.DEVICE_TYPE,
      }, False)]

    asyncio.run(run())
