import sys
from types import ModuleType, SimpleNamespace


# Keep these protocol/parser tests independent of connected hardware and the
# recursively checked-out panda/opendbc submodules.
if "panda" not in sys.modules:
  panda = ModuleType("panda")
  panda.Panda = object
  sys.modules["panda"] = panda
if "opendbc.car.structs" not in sys.modules:
  opendbc = ModuleType("opendbc")
  car = ModuleType("opendbc.car")
  structs = ModuleType("opendbc.car.structs")
  structs.CarParams = SimpleNamespace(SafetyModel=SimpleNamespace(allOutput=0, noOutput=0))
  sys.modules.update({"opendbc": opendbc, "opendbc.car": car, "opendbc.car.structs": structs})

from tools.scripts.car.vw_pq_config import Tp20Transport, parse_identity, parse_long_coding


def transport_with_frames(frames):
  transport = Tp20Transport.__new__(Tp20Transport)
  transport.rx_sequence = 0
  transport._recv_tp20 = lambda: frames.pop(0)
  transport.sent = []
  transport._send_can = transport.sent.append
  return transport


def test_tp20_masks_application_flag_from_length():
  response = bytes.fromhex("62 F1 87 31 4B 30 39 30 37 33 37 39 42 4A")
  payload = (0x8000 | len(response)).to_bytes(2, "big") + response
  frames = [bytes([0x20 + i]) + payload[i * 7 : (i + 1) * 7] for i in range(len(payload) // 7)]
  frames.append(bytes([0x10 + len(frames)]) + payload[len(frames) * 7 :])
  transport = transport_with_frames(frames)

  assert transport.recv() == response
  assert transport.sent == [bytes([0xB0 | ((transport.rx_sequence + 1) & 0xF)])]


def test_tp20_handles_ack_block_and_unacknowledged_last_frame():
  response = bytes.fromhex("5A 9A 00 01 02 03 04 05 06 07")
  payload = len(response).to_bytes(2, "big") + response
  transport = transport_with_frames(
    [
      b"\x00" + payload[:7],  # more data, ACK required
      b"\x31" + payload[7:],  # final data, no ACK required
    ]
  )

  assert transport.recv() == response
  assert transport.sent == [b"\xb1"]


def test_parse_long_coding_record():
  expected = bytes.fromhex("143B600D092B00FF281006E7901C0041150C")
  body = bytes(10) + b"\x10" + bytes([len(expected) + 1]) + expected + b"\xa5"
  result = parse_long_coding(b"\x5a\x9a" + body)

  assert result.value == expected
  assert result.checksum == 0xA5


def test_parse_identity_masks_long_coding_flag_from_software():
  payload = b"1K0907379BJ " + b"\xb0" + b"121" + b"\x10\x00\x00\x00" + bytes(6) + b"ESP MK60EC1"
  identity = parse_identity(b"\x5a\x9b" + payload)

  assert identity.part_number == "1K0907379BJ"
  assert identity.software == "0121"
  assert identity.component == "ESP MK60EC1"
