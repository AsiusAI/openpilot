#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import secrets
import signal
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

from dbus_fast import BusType, DBusError, Message, MessageType, PropertyAccess, Variant
from dbus_fast.annotations import DBusBool, DBusBytes, DBusDict, DBusObjectPath, DBusSignature, DBusStr, DBusUInt32
from dbus_fast.aio import MessageBus
from dbus_fast.service import ServiceInterface, dbus_method, dbus_property

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
import openpilot.cereal.messaging as messaging
from openpilot.system.app import methods
from openpilot.system.app.device_name import get_device_name
from openpilot.system.app.identity import is_dongle_id
from openpilot.system.app.terminal import TerminalManager
from openpilot.system.app.websocketd import (
  authorize_peer,
  load_authorized_peers,
  pack_peer_message,
  peer_message_timestamp,
  unpack_peer_message,
)

BLUEZ_SERVICE = "org.bluez"
DBUS_PROPERTIES = "org.freedesktop.DBus.Properties"
DBUS_OBJECT_MANAGER = "org.freedesktop.DBus.ObjectManager"
GATT_MANAGER = "org.bluez.GattManager1"
GATT_SERVICE = "org.bluez.GattService1"
GATT_CHARACTERISTIC = "org.bluez.GattCharacteristic1"
ADVERTISING_MANAGER = "org.bluez.LEAdvertisingManager1"
ADVERTISEMENT = "org.bluez.LEAdvertisement1"
ADAPTER = "org.bluez.Adapter1"
DEVICE = "org.bluez.Device1"
AGENT_MANAGER = "org.bluez.AgentManager1"
AGENT = "org.bluez.Agent1"

APPLICATION_PATH = "/ai/asius/ble"
SERVICE_PATH = f"{APPLICATION_PATH}/service0"
RX_PATH = f"{SERVICE_PATH}/rx"
TX_PATH = f"{SERVICE_PATH}/tx"
ADVERTISEMENT_PATH = "/ai/asius/advertisement0"
AGENT_PATH = "/ai/asius/agent0"

ACTIVE_PEER_SECONDS = 45.
PAIRING_MODE_SECONDS = 300
LIVE_STATE_INTERVAL_SECONDS = 1.
TERMINAL_OUTPUT_QUEUE_SIZE = 64
NOTIFICATION_FRAME_DELAY_SECONDS = 0.004
REGISTER_RETRY_SECONDS = 3.
DEVICE_TYPE = "asius-v1"
APP_PAIRING_UNTIL_PARAM = "AppPairingUntil"

BLE_SERVICE_UUID = "84a48ccf-5c26-56f7-91b8-5c39abd40cb9"
BLE_RX_UUID = "756e901e-4d8e-53ca-a196-41927498a27d"
BLE_TX_UUID = "6871393b-21dc-5830-b5b1-0debe5fd29c8"
PROTOCOL_VERSION = 1
FRAME_START = 1
FRAME_END = 2
FRAME_HEADER = struct.Struct(">BBHH")
INITIAL_FRAME_BYTES = 20
MAX_FRAME_BYTES = 180
MAX_MESSAGE_BYTES = 256 * 1024
ASSEMBLY_TIMEOUT_SECONDS = 10.

DBusStrList = Annotated[list[str], DBusSignature("as")]


class FrameError(ValueError):
  pass


def encode_frames(payload: bytes, frame_bytes: int = MAX_FRAME_BYTES, message_id: int | None = None) -> list[bytes]:
  if frame_bytes <= FRAME_HEADER.size:
    raise ValueError("frame size is too small")
  if len(payload) > MAX_MESSAGE_BYTES:
    raise ValueError("message is too large")

  message_id = secrets.randbelow(2**16) if message_id is None else message_id
  if not 0 <= message_id < 2**16:
    raise ValueError("message ID is out of range")

  chunk_bytes = frame_bytes - FRAME_HEADER.size
  chunks = [payload[offset:offset + chunk_bytes] for offset in range(0, len(payload), chunk_bytes)] or [b""]
  if len(chunks) >= 2**16:
    raise ValueError("message requires too many frames")

  frames = []
  for sequence, chunk in enumerate(chunks):
    flags = (FRAME_START if sequence == 0 else 0) | (FRAME_END if sequence == len(chunks) - 1 else 0)
    frames.append(FRAME_HEADER.pack(PROTOCOL_VERSION, flags, message_id, sequence) + chunk)
  return frames


@dataclass
class PartialMessage:
  message_id: int
  next_sequence: int
  payload: bytearray
  updated_at: float


class FrameAssembler:
  def __init__(self, max_message_bytes: int = MAX_MESSAGE_BYTES, timeout_seconds: float = ASSEMBLY_TIMEOUT_SECONDS):
    self.max_message_bytes = max_message_bytes
    self.timeout_seconds = timeout_seconds
    self.partial: dict[str, PartialMessage] = {}

  def feed(self, peer: str, frame: bytes, now: float | None = None) -> bytes | None:
    now = time.monotonic() if now is None else now
    self.partial = {key: value for key, value in self.partial.items() if now - value.updated_at <= self.timeout_seconds}

    if len(frame) < FRAME_HEADER.size:
      raise FrameError("frame is shorter than its header")

    version, flags, message_id, sequence = FRAME_HEADER.unpack_from(frame)
    if version != PROTOCOL_VERSION:
      raise FrameError("unsupported frame version")
    if flags & ~(FRAME_START | FRAME_END):
      raise FrameError("unsupported frame flags")

    if flags & FRAME_START:
      if sequence != 0:
        raise FrameError("first frame sequence must be zero")
      partial = self.partial[peer] = PartialMessage(message_id, 0, bytearray(), now)
    elif (partial := self.partial.get(peer)) is None:
      raise FrameError("continuation frame has no start")

    if partial.message_id != message_id:
      self.partial.pop(peer, None)
      raise FrameError("message ID changed during assembly")
    if partial.next_sequence != sequence:
      self.partial.pop(peer, None)
      raise FrameError("frame sequence is not contiguous")

    partial.payload.extend(frame[FRAME_HEADER.size:])
    partial.next_sequence += 1
    partial.updated_at = now
    if len(partial.payload) > self.max_message_bytes:
      self.partial.pop(peer, None)
      raise FrameError("message is too large")

    if not flags & FRAME_END:
      return None

    self.partial.pop(peer, None)
    return bytes(partial.payload)


def variant_value(value: Any) -> Any:
  return value.value if isinstance(value, Variant) else value


def enable_pairing_mode(duration_seconds: int = PAIRING_MODE_SECONDS) -> int:
  pairing_until = int(time.time()) + duration_seconds  # noqa: TID251
  Params().put(APP_PAIRING_UNTIL_PARAM, pairing_until, block=True)
  return pairing_until


def disable_pairing_mode() -> None:
  Params().remove(APP_PAIRING_UNTIL_PARAM)


def pairing_mode_active() -> bool:
  pairing_until = Params().get(APP_PAIRING_UNTIL_PARAM)
  return isinstance(pairing_until, int) and pairing_until >= int(time.time())  # noqa: TID251


class GattService(ServiceInterface):
  def __init__(self):
    super().__init__(GATT_SERVICE)

  @dbus_property(access=PropertyAccess.READ)
  def UUID(self) -> DBusStr:
    return BLE_SERVICE_UUID

  @dbus_property(access=PropertyAccess.READ)
  def Primary(self) -> DBusBool:
    return True


class GattCharacteristic(ServiceInterface):
  def __init__(self, uuid: str, flags: list[str]):
    super().__init__(GATT_CHARACTERISTIC)
    self.uuid = uuid
    self.flags = flags

  @dbus_property(access=PropertyAccess.READ)
  def UUID(self) -> DBusStr:
    return self.uuid

  @dbus_property(access=PropertyAccess.READ)
  def Service(self) -> DBusObjectPath:
    return SERVICE_PATH

  @dbus_property(access=PropertyAccess.READ)
  def Flags(self) -> DBusStrList:
    return self.flags


class RxCharacteristic(GattCharacteristic):
  def __init__(self, on_frame: Callable[[str, bytes], None]):
    super().__init__(BLE_RX_UUID, ["write", "encrypt-write"])
    self.on_frame = on_frame

  @dbus_method()
  def WriteValue(self, value: DBusBytes, options: DBusDict):
    device = str(variant_value(options.get("device", Variant("o", "/unknown"))))
    self.on_frame(device, bytes(value))


class TxCharacteristic(GattCharacteristic):
  def __init__(self):
    super().__init__(BLE_TX_UUID, ["notify", "encrypt-read"])
    self.notifying = False
    self.value = b""
    self.send_lock = asyncio.Lock()
    self.on_stop: Callable[[], None] | None = None

  @dbus_property(access=PropertyAccess.READ)
  def Value(self) -> DBusBytes:
    return self.value

  @dbus_property(access=PropertyAccess.READ)
  def Notifying(self) -> DBusBool:
    return self.notifying

  @dbus_method()
  def StartNotify(self):
    if not self.notifying:
      self.notifying = True
      self.emit_properties_changed({"Notifying": True})

  @dbus_method()
  def StopNotify(self):
    if self.notifying:
      self.notifying = False
      self.emit_properties_changed({"Notifying": False})
      if self.on_stop is not None:
        self.on_stop()

  async def send_text(self, text: str, initial: bool = False) -> bool:
    if not self.notifying:
      return False

    frame_bytes = INITIAL_FRAME_BYTES if initial else MAX_FRAME_BYTES
    async with self.send_lock:
      for frame in encode_frames(text.encode(), frame_bytes=frame_bytes):
        if not self.notifying:
          return False
        self.value = frame
        self.emit_properties_changed({"Value": self.value})
        await asyncio.sleep(NOTIFICATION_FRAME_DELAY_SECONDS)
    return True


class Advertisement(ServiceInterface):
  def __init__(self):
    super().__init__(ADVERTISEMENT)
    self.pairing = False

  @dbus_property(access=PropertyAccess.READ)
  def Type(self) -> DBusStr:
    return "peripheral"

  @dbus_property(access=PropertyAccess.READ)
  def ServiceUUIDs(self) -> DBusStrList:
    return [BLE_SERVICE_UUID] if self.pairing else []

  @dbus_property(access=PropertyAccess.READ)
  def LocalName(self) -> DBusStr:
    return get_device_name() if self.pairing else ""

  @dbus_property(access=PropertyAccess.READ)
  def Discoverable(self) -> DBusBool:
    return self.pairing

  @dbus_method()
  def Release(self):
    cloudlog.warning("asius.bluetooth.advertisement_released")


class PairingAgent(ServiceInterface):
  def __init__(self):
    super().__init__(AGENT)

  @staticmethod
  def require_pairing_mode() -> None:
    if not pairing_mode_active():
      raise DBusError("org.bluez.Error.Rejected", "physical pairing mode is not active")

  @dbus_method()
  def Release(self):
    pass

  @dbus_method()
  def RequestConfirmation(self, device: DBusObjectPath, passkey: DBusUInt32):
    self.require_pairing_mode()
    cloudlog.event("asius.bluetooth.os_pairing", device=device, passkey=passkey)

  @dbus_method()
  def RequestAuthorization(self, device: DBusObjectPath):
    self.require_pairing_mode()

  @dbus_method()
  def AuthorizeService(self, device: DBusObjectPath, uuid: DBusStr):
    if uuid.lower() != BLE_SERVICE_UUID.lower():
      raise DBusError("org.bluez.Error.Rejected", "service is not authorized")

  @dbus_method()
  def Cancel(self):
    pass


class BlePeerEngine:
  def __init__(self, tx: TxCharacteristic):
    self.tx = tx
    self.tx.on_stop = self.clear_active_peers
    self.assembler = FrameAssembler()
    self.incoming: asyncio.Queue[bytes] = asyncio.Queue()
    self.active_peers: dict[str, float] = {}
    self.peer_clocks: dict[str, tuple[int, float]] = {}
    self.params = Params()
    self.dongle_id = self.params.get("DongleId")
    self.sm = messaging.SubMaster(methods.LIVE_STATE_SERVICES)
    self.loop: asyncio.AbstractEventLoop | None = None
    self.terminal_output: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue(maxsize=TERMINAL_OUTPUT_QUEUE_SIZE)
    self.terminal_manager = TerminalManager(self.queue_terminal_output)

  def queue_terminal_output(self, peer: str, body: dict[str, Any]) -> None:
    if self.loop is None:
      return

    def enqueue() -> None:
      try:
        self.terminal_output.put_nowait((peer, body))
      except asyncio.QueueFull:
        cloudlog.event("asius.bluetooth.terminal_output_full", peer=peer, error=True)

    self.loop.call_soon_threadsafe(enqueue)

  def receive_frame(self, device: str, frame: bytes) -> None:
    try:
      payload = self.assembler.feed(device, frame)
      if payload is not None:
        self.incoming.put_nowait(payload)
    except FrameError as error:
      cloudlog.event("asius.bluetooth.invalid_frame", device=device, error=str(error))

  def clear_active_peers(self) -> None:
    self.active_peers.clear()

  def peer_timestamp(self, recipient: str) -> int | None:
    if recipient not in self.peer_clocks:
      return None
    peer_time, observed_at = self.peer_clocks[recipient]
    return int(peer_time + time.monotonic() - observed_at)

  async def send_body(self, recipient: str, body: dict[str, Any], initial: bool = False) -> bool:
    if self.dongle_id is None:
      return False
    return await self.tx.send_text(pack_peer_message(
      self.dongle_id, recipient, body, timestamp=self.peer_timestamp(recipient),
    ), initial=initial)

  def unpack_body(self, payload: bytes) -> tuple[str, dict | None, bool] | None:
    message = unpack_peer_message(payload.decode(), self.dongle_id, validate_timestamp=False)
    if message is None:
      return None

    sender, body, _ = message
    if body is None:
      return message

    timestamp = peer_message_timestamp(payload.decode())
    if timestamp is not None:
      self.peer_clocks[sender] = (timestamp, time.monotonic())
    return message

  async def send_error(self, recipient: str, message: str, request_id: Any = None) -> None:
    await self.send_body(recipient, {"type": "ble-error", "requestId": request_id, "error": message})

  async def send_authenticated_error(self, payload: bytes, error: Exception) -> None:
    if self.dongle_id is None:
      return
    try:
      message = self.unpack_body(payload)
      if message is None:
        return
      sender, body, _ = message
      if body is not None:
        await self.send_error(sender, str(error), body.get("requestId"))
    except Exception:
      pass

  async def handle_hello(self) -> None:
    response = {
      "type": "hello",
      "v": 1,
      "publicKey": self.dongle_id,
      "deviceType": DEVICE_TYPE,
      "name": get_device_name(),
      "maxFrameBytes": MAX_FRAME_BYTES,
    }
    await self.tx.send_text(json.dumps(response, separators=(",", ":")), initial=True)

  async def handle_pair_request(self, sender: str, body: dict[str, Any]) -> None:
    request_id = body.get("requestId")
    if body.get("publicKey") != sender:
      raise ValueError("pair request sender mismatch")
    if not isinstance(request_id, str):
      raise ValueError("pair request ID is missing")
    label = body.get("label") if isinstance(body.get("label"), str) else None

    if not pairing_mode_active():
      raise PermissionError("pairing mode is not active")
    if not is_dongle_id(sender):
      raise ValueError("invalid app public key")
    if not request_id or len(request_id) > 64:
      raise ValueError("invalid pairing request ID")
    if label is not None and len(label) > 80:
      raise ValueError("pairing label is too long")

    authorize_peer(sender, label)
    disable_pairing_mode()
    self.active_peers[sender] = time.monotonic()
    await self.send_body(sender, {
      "type": "pair-response",
      "publicKey": self.dongle_id,
      "device-type": DEVICE_TYPE,
      "name": get_device_name(),
    })
    cloudlog.event("asius.bluetooth.paired", sender=sender, request_id=request_id)

  async def handle_rpc(self, sender: str, body: dict[str, Any]) -> None:
    response = await asyncio.to_thread(lambda: json.loads(methods.handle(body, methods.dispatcher)))
    await self.send_body(sender, response)

  async def handle_encrypted(self, payload: bytes) -> None:
    if self.dongle_id is None:
      raise RuntimeError("device identity is unavailable")

    message = self.unpack_body(payload)
    if message is None:
      raise ValueError("expected an Athena peer envelope")
    sender, body, decrypt_failed = message
    if body is None:
      if decrypt_failed:
        raise ValueError("peer envelope failed authentication")
      return

    authorized = sender in load_authorized_peers()
    if not authorized:
      if body.get("type") != "ble-pair-request":
        raise PermissionError("app is not authorized")
      await self.handle_pair_request(sender, body)
      return

    self.active_peers[sender] = time.monotonic()
    message_type = body.get("type")
    if message_type == "ble-pair-request":
      await self.send_body(sender, {
        "type": "pair-response",
        "publicKey": self.dongle_id,
        "device-type": DEVICE_TYPE,
        "name": get_device_name(),
      })
    elif message_type == "ble-session":
      await self.send_body(sender, {"type": "ble-session", "ready": True})
    elif message_type == "ble-ping":
      await self.send_body(sender, {"type": "ble-pong", "id": body.get("id")})
    elif body.get("method"):
      await self.handle_rpc(sender, body)
    elif message_type == "event":
      if body.get("name") == "terminal":
        self.terminal_manager.handle(sender, body.get("payload"))
      else:
        cloudlog.event("asius.bluetooth.event", sender=sender, name=body.get("name"), payload=body.get("payload"))

  async def send_terminal_output(self, stop: asyncio.Event) -> None:
    while not stop.is_set():
      try:
        peer, body = await asyncio.wait_for(self.terminal_output.get(), timeout=0.5)
        await self.send_body(peer, body)
      except TimeoutError:
        pass
      except Exception:
        cloudlog.exception("asius.bluetooth.terminal_output_failed")

  async def process_messages(self, stop: asyncio.Event) -> None:
    while not stop.is_set():
      payload = b""
      try:
        payload = await asyncio.wait_for(self.incoming.get(), timeout=0.5)
        message = json.loads(payload)
        if isinstance(message, dict) and message.get("type") == "hello":
          await self.handle_hello()
        else:
          await self.handle_encrypted(payload)
      except TimeoutError:
        pass
      except PermissionError as error:
        cloudlog.event("asius.bluetooth.unauthorized", error=str(error))
        await self.send_authenticated_error(payload, error)
      except Exception as error:
        cloudlog.exception("asius.bluetooth.message_failed")
        await self.send_authenticated_error(payload, error)

  async def live_state(self, stop: asyncio.Event) -> None:
    while not stop.is_set():
      try:
        self.sm.update(0)
        now = time.monotonic()
        self.active_peers = {peer: seen for peer, seen in self.active_peers.items() if now - seen < ACTIVE_PEER_SECONDS}
        if self.tx.notifying and self.active_peers:
          snapshot = methods._live_state_snapshot(self.sm, self.params)
          for peer in list(self.active_peers):
            if peer in load_authorized_peers():
              await self.send_body(peer, {"type": "event", "name": "liveState", "payload": snapshot})
      except Exception:
        cloudlog.exception("asius.bluetooth.live_state_failed")
      await asyncio.sleep(LIVE_STATE_INTERVAL_SECONDS)

  async def run(self, stop: asyncio.Event) -> None:
    self.loop = asyncio.get_running_loop()
    tasks = [
      asyncio.create_task(self.process_messages(stop)),
      asyncio.create_task(self.live_state(stop)),
      asyncio.create_task(self.send_terminal_output(stop)),
    ]
    try:
      await stop.wait()
    finally:
      for task in tasks:
        task.cancel()
      await asyncio.gather(*tasks, return_exceptions=True)


def method_call(path: str, interface: str, member: str, signature: str = "", body: list[Any] | None = None) -> Message:
  return Message(
    destination=BLUEZ_SERVICE,
    path=path,
    interface=interface,
    member=member,
    signature=signature,
    body=body or [],
  )


async def checked_call(bus: MessageBus, message: Message) -> Message:
  reply = await bus.call(message)
  if reply.message_type == MessageType.ERROR:
    raise DBusError._from_message(reply)
  return reply


async def find_adapter(bus: MessageBus) -> str:
  reply = await checked_call(bus, method_call("/", DBUS_OBJECT_MANAGER, "GetManagedObjects"))
  for path, interfaces in reply.body[0].items():
    if GATT_MANAGER in interfaces and ADVERTISING_MANAGER in interfaces:
      return path
  raise RuntimeError("no Bluetooth LE peripheral adapter found")


async def set_adapter_property(bus: MessageBus, adapter: str, name: str, value: Variant) -> None:
  await checked_call(bus, method_call(adapter, DBUS_PROPERTIES, "Set", "ssv", [ADAPTER, name, value]))


async def register_bluez(bus: MessageBus, adapter: str, advertisement: Advertisement) -> None:
  await set_adapter_property(bus, adapter, "Powered", Variant("b", True))
  await set_adapter_property(bus, adapter, "Alias", Variant("s", get_device_name()))
  advertisement.pairing = pairing_mode_active()
  await set_adapter_property(bus, adapter, "Pairable", Variant("b", advertisement.pairing))
  await checked_call(bus, method_call("/org/bluez", AGENT_MANAGER, "RegisterAgent", "os", [AGENT_PATH, "NoInputNoOutput"]))
  await checked_call(bus, method_call("/org/bluez", AGENT_MANAGER, "RequestDefaultAgent", "o", [AGENT_PATH]))
  await checked_call(bus, method_call(adapter, GATT_MANAGER, "RegisterApplication", "oa{sv}", [APPLICATION_PATH, {}]))
  await checked_call(bus, method_call(adapter, ADVERTISING_MANAGER, "RegisterAdvertisement", "oa{sv}", [ADVERTISEMENT_PATH, {}]))


async def unregister_bluez(bus: MessageBus, adapter: str) -> None:
  for interface, member, path in (
    (ADVERTISING_MANAGER, "UnregisterAdvertisement", ADVERTISEMENT_PATH),
    (GATT_MANAGER, "UnregisterApplication", APPLICATION_PATH),
  ):
    try:
      await checked_call(bus, method_call(adapter, interface, member, "o", [path]))
    except Exception:
      pass
  try:
    await checked_call(bus, method_call("/org/bluez", AGENT_MANAGER, "UnregisterAgent", "o", [AGENT_PATH]))
  except Exception:
    pass


async def connected_device_count(bus: MessageBus, adapter: str) -> int:
  reply = await checked_call(bus, method_call("/", DBUS_OBJECT_MANAGER, "GetManagedObjects"))
  return sum(
    path.startswith(f"{adapter}/dev_") and DEVICE in interfaces and bool(variant_value(interfaces[DEVICE].get("Connected", False)))
    for path, interfaces in reply.body[0].items()
  )


async def refresh_advertisement(bus: MessageBus, adapter: str) -> None:
  try:
    await checked_call(bus, method_call(adapter, ADVERTISING_MANAGER, "UnregisterAdvertisement", "o", [ADVERTISEMENT_PATH]))
  except Exception:
    pass
  await checked_call(bus, method_call(adapter, ADVERTISING_MANAGER, "RegisterAdvertisement", "oa{sv}", [ADVERTISEMENT_PATH, {}]))


async def keep_advertising(bus: MessageBus, adapter: str, advertisement: Advertisement, stop: asyncio.Event) -> None:
  connected = -1
  pairing: bool | None = None
  while not stop.is_set():
    try:
      current = await connected_device_count(bus, adapter)
      current_pairing = pairing_mode_active()
      if current != connected or current_pairing != pairing:
        advertisement.pairing = current_pairing
        await set_adapter_property(bus, adapter, "Pairable", Variant("b", current_pairing))
        await refresh_advertisement(bus, adapter)
        connected = current
        pairing = current_pairing
        cloudlog.event("asius.bluetooth.advertisement_refreshed", connected=current, pairing=current_pairing)
    except Exception:
      cloudlog.exception("asius.bluetooth.advertisement_refresh_failed")
    try:
      await asyncio.wait_for(stop.wait(), timeout=0.25)
    except TimeoutError:
      pass


async def run_bluez(stop: asyncio.Event) -> None:
  while not stop.is_set():
    bus: MessageBus | None = None
    adapter: str | None = None
    engine_task: asyncio.Task | None = None
    advertising_task: asyncio.Task | None = None
    try:
      bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
      service = GattService()
      tx = TxCharacteristic()
      engine = BlePeerEngine(tx)
      rx = RxCharacteristic(engine.receive_frame)
      advertisement = Advertisement()

      bus.export(SERVICE_PATH, service)
      bus.export(RX_PATH, rx)
      bus.export(TX_PATH, tx)
      bus.export(ADVERTISEMENT_PATH, advertisement)
      bus.export(AGENT_PATH, PairingAgent())

      adapter = await find_adapter(bus)
      await register_bluez(bus, adapter, advertisement)
      cloudlog.event("asius.bluetooth.ready", adapter=adapter, service_uuid=BLE_SERVICE_UUID)

      engine_task = asyncio.create_task(engine.run(stop))
      advertising_task = asyncio.create_task(keep_advertising(bus, adapter, advertisement, stop))
      await stop.wait()
    except Exception:
      cloudlog.exception("asius.bluetooth.service_failed")
      try:
        await asyncio.wait_for(stop.wait(), timeout=REGISTER_RETRY_SECONDS)
      except TimeoutError:
        pass
    finally:
      if advertising_task is not None:
        advertising_task.cancel()
        await asyncio.gather(advertising_task, return_exceptions=True)
      if engine_task is not None:
        engine_task.cancel()
        await asyncio.gather(engine_task, return_exceptions=True)
      if bus is not None and adapter is not None:
        await unregister_bluez(bus, adapter)
      if bus is not None:
        bus.disconnect()


def main() -> None:
  async def async_main() -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
      loop.add_signal_handler(sig, stop.set)
    await run_bluez(stop)

  asyncio.run(async_main())


if __name__ == "__main__":
  main()
