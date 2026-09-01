#!/usr/bin/env python3

"""Read or change HCA (Lane Assist) adaptation channel 6 on VW PQ ZFLS racks.

This implements the vendor-specific KWP2000-over-TP2.0 sequence used by
1K0909144M SW 3201. It intentionally supports only this verified rack.
"""

import argparse
import atexit
import time
from dataclasses import dataclass

from opendbc.car.structs import CarParams
from panda import Panda


EPS_LOGICAL_ADDRESS = 0x09  # Address 44 in VCDS
HCA_CHANNEL = 0x06
SHORT_ADAPTATION_ID = b"\x01\x03"
SUPPORTED_PART_NUMBER = "1K0909144M"
SUPPORTED_SOFTWARE = "3201"
MIN_WRITE_VOLTAGE_MV = 12000
WRITE_CONFIRMATION = "ENGINE OFF, IGNITION ON"


class DiagnosticError(RuntimeError):
  pass


class DiagnosticTimeoutError(TimeoutError):
  pass


class NegativeResponseError(DiagnosticError):
  NRC = {
    0x10: "general reject",
    0x11: "service not supported",
    0x12: "sub-function not supported / invalid format",
    0x21: "busy; repeat request",
    0x22: "conditions not correct / request sequence error",
    0x31: "request out of range",
    0x33: "security access denied",
    0x35: "invalid key",
    0x36: "attempt limit exceeded",
    0x37: "required delay not expired",
    0x78: "response pending",
  }

  def __init__(self, response: bytes):
    self.response = response
    service = response[1] if len(response) > 1 else -1
    code = response[2] if len(response) > 2 else -1
    super().__init__(f"negative response to 0x{service:02X}: 0x{code:02X} ({self.NRC.get(code, 'unknown')})")


@dataclass(frozen=True)
class EpsIdentity:
  part_number: str
  software: str
  workshop_code: bytes
  component: str
  raw: bytes


class Tp20Transport:
  """Minimal VW TP2.0 transport for one diagnostic channel."""

  BROADCAST_ADDRESS = 0x200

  def __init__(self, panda: Panda, module: int, bus: int, timeout: float = 1.0, debug: bool = False):
    self.panda = panda
    self.bus = bus
    self.timeout = timeout
    self.debug = debug
    self.pending: list[tuple[int, bytes]] = []
    self.tx_sequence = 0
    self.rx_sequence = 0
    self.packet_delay = 0.01
    self.rx_address = 0
    self.tx_address = 0
    self._open_channel(module)

  def _print(self, direction: str, address: int, data: bytes) -> None:
    if self.debug:
      print(f"{direction} 0x{address:03X}: {data.hex(' ')}")

  def _recv_can(self, address: int | None = None) -> bytes:
    wanted = self.rx_address if address is None else address
    deadline = time.monotonic() + self.timeout
    while time.monotonic() < deadline:
      for index, (msg_address, data) in enumerate(self.pending):
        if msg_address == wanted:
          self.pending.pop(index)
          return data

      for msg_address, data, bus in self.panda.can_recv():
        if bus != self.bus:
          continue
        self._print("RX", msg_address, data)
        self.pending.append((msg_address, bytes(data)))
    raise DiagnosticTimeoutError(f"timeout waiting for CAN address 0x{wanted:03X}")

  def _send_can(self, data: bytes, address: int | None = None) -> None:
    target = self.tx_address if address is None else address
    self._print("TX", target, data)
    self.panda.can_send(target, data, self.bus, timeout=max(10, int(self.timeout * 1000)))
    time.sleep(self.packet_delay)

  def _open_channel(self, module: int) -> None:
    # Request receive address 0x300; the ECU supplies its transmit address.
    self._send_can(bytes([module, 0xC0, 0x00, 0x10, 0x00, 0x03, 0x01]), self.BROADCAST_ADDRESS)
    setup = self._recv_can(self.BROADCAST_ADDRESS + module)
    if len(setup) != 7 or setup[1] != 0xD0:
      raise DiagnosticError(f"unexpected TP2.0 setup response: {setup.hex(' ')}")

    self.rx_address = int.from_bytes(setup[2:4], "little")
    self.tx_address = int.from_bytes(setup[4:6], "little")
    if self.rx_address != 0x300:
      raise DiagnosticError(f"ECU accepted unexpected receive address 0x{self.rx_address:03X}")

    self._send_can(b"\xa0\x0f\x8a\xff\x0a\xff")
    timing = self._recv_can()
    if len(timing) != 6 or timing[0] != 0xA1:
      raise DiagnosticError(f"unexpected TP2.0 timing response: {timing.hex(' ')}")

  def send(self, data: bytes) -> None:
    if len(data) > 0xFF:
      raise ValueError("TP2.0 payload is limited to 255 bytes")

    payload = len(data).to_bytes(2, "big") + data
    while payload:
      final = len(payload) <= 7
      frame = bytes([(0x10 if final else 0x20) | self.tx_sequence]) + payload[:7]
      self._send_can(frame)
      if final:
        expected_ack = bytes([0xB0 | ((self.tx_sequence + 1) & 0xF)])
        actual_ack = self._recv_can()
        if actual_ack != expected_ack:
          raise DiagnosticError(f"unexpected TP2.0 ACK: {actual_ack.hex(' ')}")
      self.tx_sequence = (self.tx_sequence + 1) & 0xF
      payload = payload[7:]

  def recv(self) -> bytes:
    payload = bytearray()
    while True:
      frame = self._recv_can()
      if not frame:
        continue
      frame_type = frame[0] >> 4
      self.rx_sequence = frame[0] & 0xF
      payload.extend(frame[1:])
      if frame_type == 0x1:
        self._send_can(bytes([0xB0 | ((self.rx_sequence + 1) & 0xF)]))
        break
      if frame_type != 0x2:
        raise DiagnosticError(f"unexpected TP2.0 frame: {frame.hex(' ')}")

    if len(payload) < 2:
      raise DiagnosticError("short TP2.0 response")
    length = int.from_bytes(payload[:2], "big")
    response = bytes(payload[2 : 2 + length])
    if len(response) != length:
      raise DiagnosticError(f"truncated TP2.0 response: expected {length}, got {len(response)}")
    return response


class KwpClient:
  def __init__(self, transport: Tp20Transport, debug: bool = False):
    self.transport = transport
    self.debug = debug

  def request(self, request: bytes, pending_timeout: float = 10.0) -> bytes:
    if self.debug:
      print(f"KWP TX: {request.hex(' ')}")
    self.transport.send(request)

    deadline = time.monotonic() + pending_timeout
    while True:
      try:
        response = self.transport.recv()
      except DiagnosticTimeoutError:
        if time.monotonic() < deadline:
          continue
        raise
      if self.debug:
        print(f"KWP RX: {response.hex(' ')}")
      if response == bytes([0x7F, request[0], 0x78]) and time.monotonic() < deadline:
        continue
      if response[:1] == b"\x7f":
        raise NegativeResponseError(response)
      expected_sid = (request[0] + 0x40) & 0xFF
      if not response or response[0] != expected_sid:
        raise DiagnosticError(f"unexpected response to {request.hex(' ')}: {response.hex(' ')}")
      return response


def parse_identity(response: bytes) -> EpsIdentity:
  if not response.startswith(b"\x5a\x9b"):
    raise DiagnosticError(f"unexpected ECU identification response: {response.hex(' ')}")
  payload = response[2:]
  if len(payload) < 26:
    raise DiagnosticError(f"short ECU identification payload ({len(payload)} bytes)")
  return EpsIdentity(
    part_number=payload[:12].decode("latin-1").strip(" \x00"),
    software=payload[12:16].decode("latin-1").strip(" \x00"),
    workshop_code=payload[20:26],
    component=payload[26:].decode("latin-1").strip(" \x00"),
    raw=payload,
  )


def validate_identity(identity: EpsIdentity) -> None:
  if identity.part_number != SUPPORTED_PART_NUMBER or identity.software != SUPPORTED_SOFTWARE:
    raise DiagnosticError(
      f"refusing unsupported EPS {identity.part_number!r} SW {identity.software!r}; expected {SUPPORTED_PART_NUMBER} SW {SUPPORTED_SOFTWARE}"
    )


def workshop_code_for_writing(current: bytes) -> bytes:
  """Match Carista's fallback when an ECU reports an all-zero workshop code."""
  if len(current) != 6:
    raise ValueError("workshop code must be exactly six bytes")
  return bytes.fromhex("0181C8003039") if current == bytes(6) else current


def expect_prefix(response: bytes, prefix: bytes, label: str) -> None:
  if not response.startswith(prefix):
    raise DiagnosticError(f"unexpected {label} response: {response.hex(' ')}")


def parse_channel_value(response: bytes, channel: int) -> int:
  prefix = b"\x71\xba" + SHORT_ADAPTATION_ID
  expect_prefix(response, prefix, "adaptation read")
  data = response[len(prefix) :]

  # Exact SW 3201 response constructed by firmware function 0x0002f510 and
  # confirmed live for Channel 6:
  # 82 03 <value-hi> <value-lo> 04 25 <metadata-hi> <metadata-lo> FF
  # The response does not echo the selected channel.
  if len(data) != 9 or data[0:2] != b"\x82\x03" or data[4:6] != b"\x04\x25" or data[8] != 0xFF:
    raise DiagnosticError(f"unrecognized Channel {channel} payload: {data.hex(' ')}")
  return int.from_bytes(data[2:4], "big")


class ShortAdaptation:
  def __init__(self, kwp: KwpClient, channel: int):
    self.kwp = kwp
    self.channel = channel
    self.started = False

  def start_and_read(self) -> int:
    start = self.kwp.request(b"\x31\xb8" + SHORT_ADAPTATION_ID)
    expect_prefix(start, b"\x71\xb8" + SHORT_ADAPTATION_ID, "adaptation start")
    self.started = True

    pre_read = self.kwp.request(b"\x31\xba" + SHORT_ADAPTATION_ID)
    if pre_read != b"\x71\xba" + SHORT_ADAPTATION_ID + b"\x81":
      raise DiagnosticError(f"unexpected adaptation pre-read response: {pre_read.hex(' ')}")

    select = self.kwp.request(b"\x31\xb9" + SHORT_ADAPTATION_ID + bytes([self.channel]))
    expect_prefix(select, b"\x71\xb9" + SHORT_ADAPTATION_ID, "channel select")
    return self.read()

  def read(self) -> int:
    response = self.kwp.request(b"\x31\xba" + SHORT_ADAPTATION_ID)
    return parse_channel_value(response, self.channel)

  def write_temporary(self, value: int) -> None:
    request = b"\x31\xb9" + SHORT_ADAPTATION_ID + value.to_bytes(2, "big")
    response = self.kwp.request(request)
    expect_prefix(response, b"\x71\xb9" + SHORT_ADAPTATION_ID, "temporary adaptation write")

  def write_permanent(self, value: int, workshop_code: bytes) -> bytes:
    request = b"\x31\xbb" + SHORT_ADAPTATION_ID + value.to_bytes(2, "big") + workshop_code
    response = self.kwp.request(request, pending_timeout=20.0)
    expect_prefix(response, b"\x71\xbb" + SHORT_ADAPTATION_ID, "permanent adaptation write")
    return request

  def stop(self) -> None:
    if not self.started:
      return
    response = self.kwp.request(b"\x32\xb8" + SHORT_ADAPTATION_ID)
    expect_prefix(response, b"\x72\xb8" + SHORT_ADAPTATION_ID, "adaptation stop")
    self.started = False


def main() -> None:
  description = "Show or change HCA/Lane Assist adaptation Channel 6 on the verified Volkswagen PQ 1K0909144M SW 3201 steering rack."
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument("action", choices=("show", "enable", "disable"))
  parser.add_argument("--bus", type=int, default=1, help="panda CAN bus (default: 1)")
  parser.add_argument("--debug", action="store_true", help="print raw CAN and KWP traffic")
  args = parser.parse_args()

  panda = Panda()
  health = panda.health()
  voltage_mv = int(health["voltage"])
  print(f"Vehicle voltage: {voltage_mv / 1000:.2f} V")
  panda.can_clear(0xFFFF)
  panda.set_safety_mode(CarParams.SafetyModel.allOutput)
  atexit.register(panda.set_safety_mode, CarParams.SafetyModel.noOutput)

  transport = Tp20Transport(panda, EPS_LOGICAL_ADDRESS, args.bus, debug=args.debug)
  kwp = KwpClient(transport, debug=args.debug)
  session = kwp.request(b"\x10\x89")
  expect_prefix(session, b"\x50\x89", "diagnostic session")

  identity = parse_identity(kwp.request(b"\x1a\x9b"))
  print(f"EPS: {identity.part_number} SW {identity.software} ({identity.component})")
  print(f"Current workshop code: {identity.workshop_code.hex(' ')}")
  validate_identity(identity)

  adaptation = ShortAdaptation(kwp, HCA_CHANNEL)
  try:
    current = adaptation.start_and_read()
    print(f"Lane Assist Channel {HCA_CHANNEL}: {current} ({'ENABLED' if current == 1 else 'DISABLED' if current == 0 else 'UNKNOWN'})")

    if args.action == "show":
      return

    target = 1 if args.action == "enable" else 0
    if current == target:
      print("Requested value is already active; nothing to write.")
      return
    if current not in (0, 1):
      raise DiagnosticError(f"refusing to change unexpected Channel 6 value {current}")
    if voltage_mv < MIN_WRITE_VOLTAGE_MV:
      raise DiagnosticError(f"refusing persistent write below {MIN_WRITE_VOLTAGE_MV / 1000:.1f} V")

    workshop_code = workshop_code_for_writing(identity.workshop_code)
    temporary_request = b"\x31\xb9" + SHORT_ADAPTATION_ID + target.to_bytes(2, "big")
    permanent_request = b"\x31\xbb" + SHORT_ADAPTATION_ID + target.to_bytes(2, "big") + workshop_code
    print("\nNo write has been sent yet.")
    print(f"Temporary request: {temporary_request.hex(' ')}")
    print(f"Permanent request: {permanent_request.hex(' ')}")
    print("Before continuing: engine OFF, ignition ON, transmission in Park, steering untouched.")
    confirmation = input(f"Type exactly {WRITE_CONFIRMATION!r} to continue: ")
    if confirmation != WRITE_CONFIRMATION:
      print("Confirmation did not match; no write sent.")
      return

    adaptation.write_temporary(target)
    temporary_value = adaptation.read()
    if temporary_value != target:
      raise DiagnosticError(f"temporary verification failed: requested {target}, read {temporary_value}; permanent write NOT sent")
    sent = adaptation.write_permanent(target, workshop_code)
    if sent != permanent_request:
      raise AssertionError("permanent request changed unexpectedly")
    print("Permanent write accepted by EPS.")
    print("Turn ignition fully off, wait 30 seconds, then turn ignition on and run this tool with 'show'.")
  finally:
    adaptation.stop()


if __name__ == "__main__":
  main()
