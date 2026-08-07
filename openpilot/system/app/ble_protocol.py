from __future__ import annotations

import secrets
import struct
import time
from dataclasses import dataclass


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
    self._partial: dict[str, PartialMessage] = {}

  def discard(self, peer: str) -> None:
    self._partial.pop(peer, None)

  def expire(self, now: float | None = None) -> None:
    now = time.monotonic() if now is None else now
    expired = [peer for peer, partial in self._partial.items() if now - partial.updated_at > self.timeout_seconds]
    for peer in expired:
      self.discard(peer)

  def feed(self, peer: str, frame: bytes, now: float | None = None) -> bytes | None:
    now = time.monotonic() if now is None else now
    self.expire(now)

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
      partial = PartialMessage(message_id, 0, bytearray(), now)
      self._partial[peer] = partial
    else:
      partial = self._partial.get(peer)
      if partial is None:
        raise FrameError("continuation frame has no start")

    if partial.message_id != message_id:
      self.discard(peer)
      raise FrameError("message ID changed during assembly")
    if partial.next_sequence != sequence:
      self.discard(peer)
      raise FrameError("frame sequence is not contiguous")

    partial.payload.extend(frame[FRAME_HEADER.size:])
    partial.next_sequence += 1
    partial.updated_at = now
    if len(partial.payload) > self.max_message_bytes:
      self.discard(peer)
      raise FrameError("message is too large")

    if not flags & FRAME_END:
      return None

    self.discard(peer)
    return bytes(partial.payload)
