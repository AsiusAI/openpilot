#!/usr/bin/env python3

"""Read/configure the verified VW PQ Golf EPS, ABS, and engine.

This implements the vendor-specific KWP2000-over-TP2.0 sequence used by the
verified 1K0909144M SW 3201 rack, 1K0907379BJ SW 0121 ABS, and
03C906016AJ SW 9458 engine ECU.
"""

import argparse
import atexit
import time
from dataclasses import dataclass

from opendbc.car.structs import CarParams
from panda import Panda


EPS_LOGICAL_ADDRESS = 0x09  # Address 44 in VCDS
ABS_LOGICAL_ADDRESS = 0x03  # Address 03 in VCDS
ENGINE_LOGICAL_ADDRESS = 0x01  # Address 01 in VCDS
HCA_CHANNEL = 0x06
SHORT_ADAPTATION_ID = b"\x01\x03"
SUPPORTED_PART_NUMBER = "1K0909144M"
SUPPORTED_SOFTWARE = "3201"
SUPPORTED_ABS_PART_NUMBER = "1K0907379BJ"
SUPPORTED_ABS_SOFTWARE = "0121"
SUPPORTED_ENGINE_PART_NUMBER = "03C906016AJ"
SUPPORTED_ENGINE_SOFTWARE = "9458"
ABS_STOCK_CODING = bytes.fromhex("143B400D112800FB281402E7881F0040350000")
ABS_ACC_CODING = bytes.fromhex("143B400D112800FB281402E7881F0040150000")
ENGINE_DISABLE_CRUISE_CODE = 16167
ENGINE_STOCK_CRUISE_CODE = 11463
ENGINE_ACC_CODE = 13377
MIN_WRITE_VOLTAGE_MV = 12000
MAX_WRITE_VOLTAGE_MV = 13000
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


@dataclass(frozen=True)
class LongCoding:
  value: bytes
  checksum: int
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

  def _recv_tp20(self) -> bytes:
    while True:
      frame = self._recv_can()
      # A3 is a TP2.0 keepalive/request-more control frame. It can arrive
      # between an application frame and its ACK/response, especially after
      # waiting for operator confirmation, and carries no application data.
      if frame == b"\xa3":
        continue
      if frame == b"\xa8":
        raise DiagnosticError("TP2.0 channel disconnected by ECU")
      return frame

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
    timing = self._recv_tp20()
    if len(timing) != 6 or timing[0] != 0xA1:
      raise DiagnosticError(f"unexpected TP2.0 timing response: {timing.hex(' ')}")

  def close(self) -> None:
    if self.tx_address:
      try:
        self._send_can(b"\xa8")
      except Exception:
        pass
      self.tx_address = 0

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
        actual_ack = self._recv_tp20()
        if actual_ack != expected_ack:
          raise DiagnosticError(f"unexpected TP2.0 ACK: {actual_ack.hex(' ')}")
      self.tx_sequence = (self.tx_sequence + 1) & 0xF
      payload = payload[7:]

  def recv(self) -> bytes:
    payload = bytearray()
    while True:
      frame = self._recv_tp20()
      if not frame:
        continue
      frame_type = frame[0] >> 4
      self.rx_sequence = frame[0] & 0xF
      payload.extend(frame[1:])
      if frame_type in (0x0, 0x1):
        self._send_can(bytes([0xB0 | ((self.rx_sequence + 1) & 0xF)]))
      if frame_type in (0x1, 0x3):
        break
      if frame_type not in (0x0, 0x2):
        raise DiagnosticError(f"unexpected TP2.0 frame: {frame.hex(' ')}")

    if len(payload) < 2:
      raise DiagnosticError("short TP2.0 response")
    # TP2.0 reserves the length MSB as an application type flag. Several VW
    # controllers, including the Golf's MK60EC1 ABS, set it on diagnostic
    # responses. It is not part of the 15-bit payload length.
    length = int.from_bytes(payload[:2], "big") & 0x7FFF
    response = bytes(payload[2 : 2 + length])
    if len(response) != length:
      raise DiagnosticError(f"truncated TP2.0 response: expected {length}, got {len(response)}")
    return response


class KwpClient:
  def __init__(self, transport: Tp20Transport, debug: bool = False):
    self.transport = transport
    self.debug = debug

  def request(self, request: bytes, pending_timeout: float = 10.0, busy_retries: int = 0) -> bytes:
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
      if response == bytes([0x7F, request[0], 0x21]) and busy_retries > 0:
        busy_retries -= 1
        time.sleep(0.1)
        self.transport.send(request)
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
    software=(bytes([payload[12] & 0x7F]) + payload[13:16]).decode("latin-1").strip(" \x00"),
    workshop_code=payload[20:26],
    component=payload[26:].decode("latin-1").strip(" \x00"),
    raw=payload,
  )


def validate_identity(identity: EpsIdentity) -> None:
  if identity.part_number != SUPPORTED_PART_NUMBER or identity.software != SUPPORTED_SOFTWARE:
    raise DiagnosticError(
      f"refusing unsupported EPS {identity.part_number!r} SW {identity.software!r}; expected {SUPPORTED_PART_NUMBER} SW {SUPPORTED_SOFTWARE}"
    )


def validate_abs_identity(identity: EpsIdentity) -> None:
  if identity.part_number != SUPPORTED_ABS_PART_NUMBER or identity.software != SUPPORTED_ABS_SOFTWARE:
    raise DiagnosticError(
      f"unsupported ABS {identity.part_number!r} SW {identity.software!r}; expected {SUPPORTED_ABS_PART_NUMBER} SW {SUPPORTED_ABS_SOFTWARE}"
    )


def validate_engine_identity(identity: EpsIdentity) -> None:
  if identity.part_number != SUPPORTED_ENGINE_PART_NUMBER or identity.software != SUPPORTED_ENGINE_SOFTWARE:
    raise DiagnosticError(
      f"refusing unsupported engine {identity.part_number!r} SW {identity.software!r}; expected {SUPPORTED_ENGINE_PART_NUMBER} SW {SUPPORTED_ENGINE_SOFTWARE}"
    )


def parse_long_coding(response: bytes) -> LongCoding:
  """Parse the VW KWP ReadECUIdentification(0x9A) long-coding record."""
  if not response.startswith(b"\x5a\x9a"):
    raise DiagnosticError(f"unexpected long-coding response: {response.hex(' ')}")
  data = response[2:]
  if len(data) < 12 or data[10] != 0x10:
    raise DiagnosticError(f"unrecognized long-coding header: {data.hex(' ')}")

  record_length = data[11]
  # The record length covers the coding bytes plus one trailing checksum.
  if record_length < 2 or len(data) < 12 + record_length:
    raise DiagnosticError(f"short long-coding record: {data.hex(' ')}")
  coding = bytes(data[12 : 11 + record_length])
  checksum = data[11 + record_length]
  return LongCoding(value=coding, checksum=checksum, raw=data)


def read_did(kwp: KwpClient, identifier: int) -> bytes:
  did = identifier.to_bytes(2, "big")
  response = kwp.request(b"\x22" + did)
  expect_prefix(response, b"\x62" + did, f"DID 0x{identifier:04X}")
  return response[3:]


def write_did(kwp: KwpClient, identifier: int, value: bytes) -> bytes:
  """Write one KWP2000 common identifier and require its echoed identifier."""
  did = identifier.to_bytes(2, "big")
  request = b"\x2e" + did + value
  response = kwp.request(request, pending_timeout=20.0)
  expect_prefix(response, b"\x6e" + did, f"DID 0x{identifier:04X} write")
  return request


def code2_request(code: int) -> bytes:
  """Build VW KWP2000 Access Authorization Code 2 (Coding II)."""
  if not 0 <= code <= 0xFFFF:
    raise ValueError("Code 2 must fit in 16 bits")
  return b"\x27\x02" + code.to_bytes(2, "big")


def apply_code2(kwp: KwpClient, code: int) -> bytes:
  request = code2_request(code)
  response = kwp.request(request, pending_timeout=20.0)
  expect_prefix(response, b"\x67\x02", f"Code 2 {code}")
  return request


def validate_abs_coding_transition(current: bytes, target: bytes) -> None:
  if {current, target} != {ABS_STOCK_CODING, ABS_ACC_CODING}:
    raise DiagnosticError("refusing ABS coding outside the exact saved stock/ACC pair")
  differences = [(index, old, new) for index, (old, new) in enumerate(zip(current, target, strict=True)) if old != new]
  if differences != [(16, 0x35 if current == ABS_STOCK_CODING else 0x15, 0x15 if target == ABS_ACC_CODING else 0x35)]:
    raise DiagnosticError(f"refusing unexpected ABS coding delta: {differences}")


def confirm_persistent_write(action: str) -> None:
  confirmation = f"{WRITE_CONFIRMATION}: {action}"
  print("\nNo write has been sent yet.")
  print("Before continuing: engine OFF, ignition ON, transmission in Park, controls untouched.")
  entered = input(f"Type exactly {confirmation!r} to continue: ")
  if entered != confirmation:
    raise DiagnosticError("confirmation did not match; no write sent")


def workshop_code_for_writing(current: bytes) -> bytes:
  """Match Carista's fallback when an ECU reports an all-zero workshop code."""
  if len(current) != 6:
    raise ValueError("workshop code must be exactly six bytes")
  return bytes.fromhex("0181C8003039") if current == bytes(6) else current


def require_safe_write_voltage(panda: Panda) -> int:
  voltage_mv = int(panda.health()["voltage"])
  if voltage_mv < MIN_WRITE_VOLTAGE_MV:
    raise DiagnosticError(f"refusing persistent write below {MIN_WRITE_VOLTAGE_MV / 1000:.1f} V")
  if voltage_mv > MAX_WRITE_VOLTAGE_MV:
    raise DiagnosticError(f"refusing persistent write above {MAX_WRITE_VOLTAGE_MV / 1000:.1f} V; engine may still be running")
  return voltage_mv


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
  description = "Read/configure the verified Golf EPS, MK60EC1 ABS, and MED17.5.5 engine."
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument(
    "action",
    choices=(
      "show",
      "enable",
      "disable",
      "abs-info",
      "engine-info",
      "abs-acc-enable",
      "abs-stock-restore",
      "engine-acc-enable",
      "engine-stock-restore",
    ),
  )
  parser.add_argument("--bus", type=int, default=1, help="panda CAN bus (default: 1)")
  parser.add_argument("--debug", action="store_true", help="print raw CAN and KWP traffic")
  parser.add_argument(
    "--experimental-long-write",
    action="store_true",
    help="allow a guarded engine/ABS longitudinal configuration write after showing the complete request",
  )
  args = parser.parse_args()

  panda = Panda()
  health = panda.health()
  voltage_mv = int(health["voltage"])
  print(f"Vehicle voltage: {voltage_mv / 1000:.2f} V")
  panda.can_clear(0xFFFF)
  panda.set_safety_mode(CarParams.SafetyModel.allOutput)
  atexit.register(panda.set_safety_mode, CarParams.SafetyModel.noOutput)

  module = {
    "abs-info": ABS_LOGICAL_ADDRESS,
    "abs-acc-enable": ABS_LOGICAL_ADDRESS,
    "abs-stock-restore": ABS_LOGICAL_ADDRESS,
    "engine-info": ENGINE_LOGICAL_ADDRESS,
    "engine-acc-enable": ENGINE_LOGICAL_ADDRESS,
    "engine-stock-restore": ENGINE_LOGICAL_ADDRESS,
  }.get(args.action, EPS_LOGICAL_ADDRESS)
  transport = Tp20Transport(panda, module, args.bus, debug=args.debug)
  atexit.register(transport.close)
  kwp = KwpClient(transport, debug=args.debug)
  session = kwp.request(b"\x10\x89", busy_retries=5)
  expect_prefix(session, b"\x50\x89", "diagnostic session")

  if args.action in ("abs-info", "abs-acc-enable", "abs-stock-restore"):
    part_number = read_did(kwp, 0xF187).decode("latin-1").strip(" \x00")
    software = read_did(kwp, 0xF189).decode("latin-1").strip(" \x00")
    component = read_did(kwp, 0xF197).decode("latin-1").strip(" \x00")
    print(f"ABS: {part_number} SW {software} ({component})")
    if part_number != SUPPORTED_ABS_PART_NUMBER or software != SUPPORTED_ABS_SOFTWARE:
      raise DiagnosticError(f"unsupported ABS {part_number!r} SW {software!r}; expected {SUPPORTED_ABS_PART_NUMBER} SW {SUPPORTED_ABS_SOFTWARE}")
    coding = read_did(kwp, 0x0600)
    print(f"Long coding ({len(coding)} bytes): {coding.hex().upper()}")
    if len(coding) <= 16:
      raise DiagnosticError(f"long coding is too short to contain Byte 16: {len(coding)} bytes")
    acc_not_installed = bool(coding[16] & (1 << 5))
    print(f"Byte 16: {coding[16]:02X}; Bit 5: {int(acc_not_installed)}")
    print(f"ACC coding state: {'NOT INSTALLED' if acc_not_installed else 'INSTALLED'}")
    if args.action == "abs-info":
      print("Read-only: no adaptation or coding write was sent.")
      return

    identity = parse_identity(kwp.request(b"\x1a\x9b"))
    validate_abs_identity(identity)
    target = ABS_ACC_CODING if args.action == "abs-acc-enable" else ABS_STOCK_CODING
    target_label = "enable ABS ACC brake-torque interface" if args.action == "abs-acc-enable" else "restore stock ABS coding"
    if coding == target:
      print(f"Requested state is already active; nothing to write ({target.hex().upper()}).")
      return
    validate_abs_coding_transition(coding, target)

    workshop_code = workshop_code_for_writing(identity.workshop_code)
    fingerprint_request = b"\x2e\xf1\x98" + workshop_code
    coding_request = b"\x2e\x06\x00" + target
    print(f"Planned change: Byte 16 {coding[16]:02X} -> {target[16]:02X}; every other byte unchanged.")
    print(f"Rollback coding: {ABS_STOCK_CODING.hex().upper()}")
    print(f"Workshop fingerprint request: {fingerprint_request.hex(' ')}")
    print(f"Coding request:               {coding_request.hex(' ')}")
    if not args.experimental_long_write:
      print("Preview only: no write sent. The --experimental-long-write gate is required in addition to typed confirmation.")
      return
    confirm_persistent_write(target_label)
    print(f"Pre-write voltage: {require_safe_write_voltage(panda) / 1000:.2f} V")
    sent_fingerprint = write_did(kwp, 0xF198, workshop_code)
    if sent_fingerprint != fingerprint_request:
      raise AssertionError("workshop fingerprint request changed unexpectedly")
    print(f"Pre-coding voltage: {require_safe_write_voltage(panda) / 1000:.2f} V")
    sent_coding = write_did(kwp, 0x0600, target)
    if sent_coding != coding_request:
      raise AssertionError("ABS coding request changed unexpectedly")
    readback = read_did(kwp, 0x0600)
    if readback != target:
      raise DiagnosticError(f"CRITICAL: ABS accepted the write but read back {readback.hex().upper()}, expected {target.hex().upper()}")
    print(f"ABS write accepted and verified: {readback.hex().upper()}")
    print("Turn ignition fully off, wait at least 30 seconds, then turn ignition on and run 'abs-info'.")
    return

  identity = parse_identity(kwp.request(b"\x1a\x9b"))
  if args.action in ("engine-info", "engine-acc-enable", "engine-stock-restore"):
    print(f"Engine: {identity.part_number} SW {identity.software} ({identity.component})")
    print(f"Current workshop code: {identity.workshop_code.hex(' ')}")
    coding_type = identity.raw[16]
    if coding_type == 0x10:
      coding = parse_long_coding(kwp.request(b"\x1a\x9a"))
      print(f"Long coding ({len(coding.value)} bytes): {coding.value.hex().upper()}")
      print(f"Coding record checksum: {coding.checksum:02X}")
    elif coding_type == 0x03:
      print(f"Short coding: {identity.raw[17:20].hex().upper()}")
    else:
      print(f"Unknown coding record type: {coding_type:02X}")
    if args.action == "engine-info":
      print("Read-only: no adaptation or coding write was sent.")
      return

    validate_engine_identity(identity)
    target_code = ENGINE_ACC_CODE if args.action == "engine-acc-enable" else ENGINE_STOCK_CRUISE_CODE
    target_label = "enable engine ACC without Follow-to-Stop" if args.action == "engine-acc-enable" else "restore stock engine cruise"
    disable_request = code2_request(ENGINE_DISABLE_CRUISE_CODE)
    target_request = code2_request(target_code)
    print(f"Sequence: disable cruise/ACC with Code 2 {ENGINE_DISABLE_CRUISE_CODE}, then apply Code 2 {target_code}.")
    print(f"Disable request: {disable_request.hex(' ')}")
    print(f"Target request:  {target_request.hex(' ')}")
    print(f"Rollback sequence: {ENGINE_DISABLE_CRUISE_CODE} then {ENGINE_STOCK_CRUISE_CODE}.")
    print("Code 30903 is deliberately not offered: it selects Follow-to-Stop/Front Assist and is inappropriate for this radarless H31 test.")
    if not args.experimental_long_write:
      print("Preview only: no write sent. The --experimental-long-write gate is required in addition to typed confirmation.")
      return
    confirm_persistent_write(target_label)
    print(f"Pre-write voltage: {require_safe_write_voltage(panda) / 1000:.2f} V")
    sent_disable = apply_code2(kwp, ENGINE_DISABLE_CRUISE_CODE)
    if sent_disable != disable_request:
      raise AssertionError("engine disable request changed unexpectedly")
    try:
      print(f"Pre-target voltage: {require_safe_write_voltage(panda) / 1000:.2f} V")
      sent_target = apply_code2(kwp, target_code)
      if sent_target != target_request:
        raise AssertionError("engine target request changed unexpectedly")
    except Exception:
      if target_code != ENGINE_STOCK_CRUISE_CODE:
        print("ACC activation failed after disabling cruise; attempting immediate stock-cruise rollback.")
        apply_code2(kwp, ENGINE_STOCK_CRUISE_CODE)
      raise
    print("Engine Code 2 sequence accepted.")
    print("Turn ignition fully off for at least 15 seconds, then turn ignition on with the engine still off and run 'engine-info'.")
    return

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

    print(f"Pre-write voltage: {require_safe_write_voltage(panda) / 1000:.2f} V")
    adaptation.write_temporary(target)
    temporary_value = adaptation.read()
    if temporary_value != target:
      raise DiagnosticError(f"temporary verification failed: requested {target}, read {temporary_value}; permanent write NOT sent")
    print(f"Pre-commit voltage: {require_safe_write_voltage(panda) / 1000:.2f} V")
    sent = adaptation.write_permanent(target, workshop_code)
    if sent != permanent_request:
      raise AssertionError("permanent request changed unexpectedly")
    print("Permanent write accepted by EPS.")
    print("Turn ignition fully off, wait 30 seconds, then turn ignition on and run this tool with 'show'.")
  finally:
    adaptation.stop()


if __name__ == "__main__":
  main()
