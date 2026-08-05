#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import signal
import time
from collections.abc import Callable
from typing import Annotated, Any

from dbus_fast import BusType, DBusError, Message, MessageType, PropertyAccess, Variant
from dbus_fast.annotations import DBusBool, DBusBytes, DBusDict, DBusObjectPath, DBusSignature, DBusStr
from dbus_fast.aio import MessageBus
from dbus_fast.service import ServiceInterface, dbus_method, dbus_property

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.system.athena import asius_athenad as athenad
from openpilot.system.athena.ble_pairing import PAIRING_CHALLENGE_SECONDS, clear_ble_pairing, create_ble_pairing, get_ble_pairing
from openpilot.system.athena.ble_protocol import (
  BLE_RX_UUID,
  BLE_SERVICE_UUID,
  BLE_TX_UUID,
  INITIAL_FRAME_BYTES,
  MAX_FRAME_BYTES,
  FrameAssembler,
  FrameError,
  encode_frames,
)
from openpilot.system.athena.websocketd import (
  load_authorized_peers,
  pack_peer_message,
  peer_message_timestamp,
  unpack_peer_message,
)

try:
  import openpilot.cereal.messaging as messaging
except ModuleNotFoundError:
  import cereal.messaging as messaging


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

APPLICATION_PATH = "/ai/asius/ble"
SERVICE_PATH = f"{APPLICATION_PATH}/service0"
RX_PATH = f"{SERVICE_PATH}/rx"
TX_PATH = f"{SERVICE_PATH}/tx"
ADVERTISEMENT_PATH = "/ai/asius/advertisement0"

ACTIVE_PEER_SECONDS = 45.
LIVE_STATE_INTERVAL_SECONDS = 1.
NOTIFICATION_FRAME_DELAY_SECONDS = 0.004
REGISTER_RETRY_SECONDS = 3.
DEVICE_TYPE = "asius-v1"
DEVICE_NAME = "Asius v1"

DBusStrList = Annotated[list[str], DBusSignature("as")]


def variant_value(value: Any) -> Any:
  return value.value if isinstance(value, Variant) else value


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
    super().__init__(BLE_RX_UUID, ["write"])
    self.on_frame = on_frame

  @dbus_method()
  def WriteValue(self, value: DBusBytes, options: DBusDict):
    device = str(variant_value(options.get("device", Variant("o", "/unknown"))))
    self.on_frame(device, bytes(value))


class TxCharacteristic(GattCharacteristic):
  def __init__(self):
    super().__init__(BLE_TX_UUID, ["notify"])
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

  @dbus_property(access=PropertyAccess.READ)
  def Type(self) -> DBusStr:
    return "peripheral"

  @dbus_property(access=PropertyAccess.READ)
  def ServiceUUIDs(self) -> DBusStrList:
    return [BLE_SERVICE_UUID]

  @dbus_property(access=PropertyAccess.READ)
  def LocalName(self) -> DBusStr:
    return DEVICE_NAME

  @dbus_property(access=PropertyAccess.READ)
  def Discoverable(self) -> DBusBool:
    return True

  @dbus_method()
  def Release(self):
    cloudlog.warning("asius.bluetooth.advertisement_released")


class BlePeerEngine:
  def __init__(self, tx: TxCharacteristic):
    self.tx = tx
    self.tx.on_stop = self.clear_active_peers
    self.assembler = FrameAssembler()
    self.incoming: asyncio.Queue[tuple[str, bytes]] = asyncio.Queue()
    self.active_peers: dict[str, float] = {}
    self.peer_clocks: dict[str, tuple[int, float]] = {}
    self.params = Params()
    self.dongle_id = self.params.get("DongleId")
    self.sm = messaging.SubMaster(athenad.LIVE_STATE_SERVICES)

  def receive_frame(self, device: str, frame: bytes) -> None:
    try:
      payload = self.assembler.feed(device, frame)
      if payload is not None:
        self.incoming.put_nowait((device, payload))
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

  async def handle_hello(self, message: dict[str, Any]) -> None:
    response = {
      "type": "hello",
      "v": 1,
      "publicKey": self.dongle_id,
      "deviceType": DEVICE_TYPE,
      "name": DEVICE_NAME,
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

    state = create_ble_pairing(sender, request_id, label)
    self.active_peers[sender] = time.monotonic()
    expires_at = state["expiresAt"]
    if (peer_timestamp := self.peer_timestamp(sender)) is not None:
      expires_at = peer_timestamp + PAIRING_CHALLENGE_SECONDS
    await self.send_body(sender, {
      "type": "ble-pair-challenge",
      "requestId": request_id,
      "colors": state["colors"],
      "expiresAt": expires_at,
    })
    cloudlog.event("asius.bluetooth.pairing_challenge", sender=sender, request_id=request_id)

  async def handle_call(self, sender: str, body: dict[str, Any]) -> None:
    request_id = body.get("id")
    request: dict[str, Any] = {
      "jsonrpc": "2.0",
      "id": request_id,
      "method": body.get("method"),
    }
    if "params" in body:
      request["params"] = body["params"]

    if hasattr(athenad, "handle"):
      response = await asyncio.to_thread(lambda: json.loads(athenad.handle(request, athenad.dispatcher)))
    else:
      response = await asyncio.to_thread(lambda: json.loads(athenad.JSONRPCResponseManager.handle(json.dumps(request), athenad.dispatcher).json))
    await self.send_body(sender, {
      "type": "athena-response",
      "id": request_id,
      "result": response.get("result"),
      "error": response.get("error"),
    })

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
      sent = await self.send_body(sender, {
        "type": "pair-response",
        "publicKey": self.dongle_id,
        "device-type": DEVICE_TYPE,
        "aclEpoch": int(load_authorized_peers()[sender].get("aclEpoch", 0)),
      })
      approved = get_ble_pairing()
      if sent and approved is not None and approved.get("status") == "approved" and approved.get("publicKey") == sender:
        clear_ble_pairing()
        cloudlog.event("asius.bluetooth.paired", sender=sender, request_id=approved["requestId"])
    elif message_type in ("ble-session", "ble-ping"):
      await self.send_body(sender, {"type": "ble-session", "ready": True})
    elif message_type == "athena-call" or body.get("method"):
      await self.handle_call(sender, body)
    elif message_type == "athena-response":
      athenad.log_recv_queue.put_nowait(json.dumps({
        "jsonrpc": "2.0",
        "id": body.get("id"),
        "result": body.get("result"),
        "error": body.get("error"),
      }))
    elif message_type == "event":
      cloudlog.event("asius.bluetooth.event", sender=sender, name=body.get("name"), payload=body.get("payload"))

  async def process_messages(self, stop: asyncio.Event) -> None:
    while not stop.is_set():
      payload = b""
      try:
        _, payload = await asyncio.wait_for(self.incoming.get(), timeout=0.5)
        message = json.loads(payload)
        if isinstance(message, dict) and message.get("type") == "hello":
          await self.handle_hello(message)
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

  async def send_pairing_approval(self) -> bool:
    approved = get_ble_pairing()
    if approved is None or approved.get("status") != "approved":
      return False

    sender = approved["publicKey"]
    if sender not in self.active_peers:
      return False
    sent = await self.send_body(sender, {
      "type": "pair-response",
      "publicKey": self.dongle_id,
      "device-type": DEVICE_TYPE,
      "aclEpoch": approved["aclEpoch"],
    })
    if sent:
      clear_ble_pairing()
      cloudlog.event("asius.bluetooth.paired", sender=sender, request_id=approved["requestId"])
    return sent

  async def pairing_approvals(self, stop: asyncio.Event) -> None:
    while not stop.is_set():
      await self.send_pairing_approval()
      await asyncio.sleep(0.1)

  async def live_state(self, stop: asyncio.Event) -> None:
    while not stop.is_set():
      try:
        self.sm.update(0)
        now = time.monotonic()
        self.active_peers = {peer: seen for peer, seen in self.active_peers.items() if now - seen < ACTIVE_PEER_SECONDS}
        if self.tx.notifying and self.active_peers:
          snapshot = athenad._live_state_snapshot(self.sm, self.params)
          for peer in list(self.active_peers):
            if peer in load_authorized_peers():
              await self.send_body(peer, {"type": "event", "name": "liveState", "payload": snapshot})
      except Exception:
        cloudlog.exception("asius.bluetooth.live_state_failed")
      await asyncio.sleep(LIVE_STATE_INTERVAL_SECONDS)

  async def run(self, stop: asyncio.Event) -> None:
    tasks = [
      asyncio.create_task(self.process_messages(stop)),
      asyncio.create_task(self.pairing_approvals(stop)),
      asyncio.create_task(self.live_state(stop)),
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


async def register_bluez(bus: MessageBus, adapter: str) -> None:
  await set_adapter_property(bus, adapter, "Powered", Variant("b", True))
  await set_adapter_property(bus, adapter, "Alias", Variant("s", DEVICE_NAME))
  await set_adapter_property(bus, adapter, "Pairable", Variant("b", False))
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


async def keep_advertising(bus: MessageBus, adapter: str, stop: asyncio.Event) -> None:
  connected = -1
  while not stop.is_set():
    try:
      current = await connected_device_count(bus, adapter)
      if current != connected:
        await refresh_advertisement(bus, adapter)
        connected = current
        cloudlog.event("asius.bluetooth.advertisement_refreshed", connected=current)
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

      bus.export(SERVICE_PATH, service)
      bus.export(RX_PATH, rx)
      bus.export(TX_PATH, tx)
      bus.export(ADVERTISEMENT_PATH, Advertisement())

      adapter = await find_adapter(bus)
      await register_bluez(bus, adapter)
      cloudlog.event("asius.bluetooth.ready", adapter=adapter, service_uuid=BLE_SERVICE_UUID)

      engine_task = asyncio.create_task(engine.run(stop))
      advertising_task = asyncio.create_task(keep_advertising(bus, adapter, stop))
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
