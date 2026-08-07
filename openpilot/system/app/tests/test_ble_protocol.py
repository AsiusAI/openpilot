from openpilot.common.test import OpenpilotTestCase
from openpilot.system.app.ble_protocol import (
  FRAME_END,
  FRAME_HEADER,
  FRAME_START,
  PROTOCOL_VERSION,
  FrameAssembler,
  FrameError,
  encode_frames,
)


class TestBleProtocol(OpenpilotTestCase):
  def test_frame_round_trip(self):
    for length in (0, 1, 14, 15, 174, 175, 4096):
      with self.subTest(length=length):
        payload = bytes(index % 251 for index in range(length))
        frames = encode_frames(payload, frame_bytes=20, message_id=42)
        assembler = FrameAssembler()

        result = None
        for frame in frames:
          result = assembler.feed("peer", frame)
        assert result == payload
        assert all(len(frame) <= 20 for frame in frames)

  def test_assemblers_are_isolated_by_peer(self):
    first = encode_frames(b"a" * 30, frame_bytes=20, message_id=1)
    second = encode_frames(b"b" * 30, frame_bytes=20, message_id=2)
    assembler = FrameAssembler()

    assert assembler.feed("first", first[0]) is None
    assert assembler.feed("second", second[0]) is None
    assert assembler.feed("first", first[1]) is None
    assert assembler.feed("second", second[1]) is None
    assert assembler.feed("first", first[2]) == b"a" * 30
    assert assembler.feed("second", second[2]) == b"b" * 30

  def test_invalid_sequence_discards_message(self):
    frames = encode_frames(b"message long enough for frames", frame_bytes=14, message_id=7)
    assembler = FrameAssembler()
    assert assembler.feed("peer", frames[0]) is None
    with self.assertRaisesRegex(FrameError, "sequence"):
      assembler.feed("peer", frames[2])
    with self.assertRaisesRegex(FrameError, "no start"):
      assembler.feed("peer", frames[1])

  def test_expired_message_is_discarded(self):
    frames = encode_frames(b"two frames", frame_bytes=12, message_id=9)
    assembler = FrameAssembler(timeout_seconds=1)
    assert assembler.feed("peer", frames[0], now=5) is None
    with self.assertRaisesRegex(FrameError, "no start"):
      assembler.feed("peer", frames[1], now=7)

  def test_rejects_invalid_header(self):
    assembler = FrameAssembler()
    with self.assertRaisesRegex(FrameError, "shorter"):
      assembler.feed("peer", b"\x01")
    with self.assertRaisesRegex(FrameError, "version"):
      assembler.feed("peer", FRAME_HEADER.pack(PROTOCOL_VERSION + 1, FRAME_START | FRAME_END, 1, 0))
