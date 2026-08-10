from __future__ import annotations

import base64
import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


COMMON_PSSH_SYSTEM_ID = bytes.fromhex("1077efecc0b24d02ace33c1e52e2fb4b")
CONTAINER_TYPES = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}
VIDEO_SAMPLE_TYPES = {b"avc1", b"avc3", b"hvc1", b"hev1"}
AUDIO_SAMPLE_TYPES = {b"mp4a"}


def b64url(value: bytes) -> str:
  return base64.urlsafe_b64encode(value).decode().rstrip("=")


@dataclass(frozen=True)
class Box:
  start: int
  size: int
  type: bytes
  header_size: int

  @property
  def payload_start(self) -> int:
    return self.start + self.header_size

  @property
  def end(self) -> int:
    return self.start + self.size


@dataclass(frozen=True)
class Track:
  track_id: int
  sample_type: bytes
  codec: str
  nal_length_size: int | None
  nal_header_size: int | None
  timescale: int


@dataclass(frozen=True)
class Run:
  box: Box
  data_offset: int
  sample_sizes: tuple[int, ...]
  sample_durations: tuple[int, ...]


@dataclass(frozen=True)
class AuxiliaryInfo:
  entries: tuple[bytes, ...]
  subsamples: bool

  def boxes(self, offset: int) -> tuple[bytes, bytes, bytes]:
    sizes = bytes(len(entry) for entry in self.entries)
    saiz = make_box(b"saiz", b"\0\0\0\0" + b"\0" + struct.pack(">I", len(self.entries)) + sizes)
    saio = make_box(b"saio", b"\0\0\0\0" + struct.pack(">II", 1, offset))
    flags = 2 if self.subsamples else 0
    senc = make_box(b"senc", flags.to_bytes(4, "big") + struct.pack(">I", len(self.entries)) + b"".join(self.entries))
    return saiz, saio, senc

  @property
  def added_size(self) -> int:
    return sum(map(len, self.boxes(0)))


def read_boxes(data: bytes | bytearray, start: int, end: int) -> list[Box]:
  boxes: list[Box] = []
  cursor = start
  while cursor < end:
    if end - cursor < 8:
      raise ValueError("truncated MP4 box header")
    size = struct.unpack_from(">I", data, cursor)[0]
    header_size = 8
    if size == 1:
      if end - cursor < 16:
        raise ValueError("truncated extended MP4 box header")
      size = struct.unpack_from(">Q", data, cursor + 8)[0]
      header_size = 16
    elif size == 0:
      size = end - cursor
    if size < header_size or cursor + size > end:
      raise ValueError("invalid MP4 box size")
    boxes.append(Box(cursor, size, bytes(data[cursor + 4:cursor + 8]), header_size))
    cursor += size
  if cursor != end:
    raise ValueError("unaligned MP4 boxes")
  return boxes


def make_box(box_type: bytes, payload: bytes) -> bytes:
  size = len(payload) + 8
  if len(box_type) != 4 or size >= 2**32:
    raise ValueError("unsupported MP4 box")
  return struct.pack(">I4s", size, box_type) + payload


def child(data: bytes | bytearray, parent: Box, box_type: bytes) -> Box:
  return next(box for box in read_boxes(data, parent.payload_start, parent.end) if box.type == box_type)


def descendant(data: bytes | bytearray, parent: Box, *types: bytes) -> Box:
  current = parent
  for box_type in types:
    current = child(data, current, box_type)
  return current


def sample_entries(data: bytes | bytearray, stsd: Box) -> list[Box]:
  if stsd.size < stsd.header_size + 8:
    raise ValueError("invalid stsd box")
  entries = read_boxes(data, stsd.payload_start + 8, stsd.end)
  if len(entries) != struct.unpack_from(">I", data, stsd.payload_start + 4)[0]:
    raise ValueError("invalid stsd entry count")
  return entries


def sample_entry_children(data: bytes | bytearray, entry: Box) -> list[Box]:
  if entry.type in VIDEO_SAMPLE_TYPES:
    children_start = entry.start + 86
  elif entry.type in AUDIO_SAMPLE_TYPES:
    children_start = entry.start + 36
  else:
    return []
  if children_start > entry.end:
    raise ValueError("invalid sample entry")
  return read_boxes(data, children_start, entry.end)


def reverse_bits32(value: int) -> int:
  return int(f"{value:032b}"[::-1], 2)


def hevc_codec(config: bytes, sample_type: bytes) -> str:
  if len(config) < 23 or config[0] != 1:
    raise ValueError("invalid hvcC configuration")
  profile_space = (config[1] >> 6) & 3
  profile_prefix = "" if profile_space == 0 else "ABC"[profile_space - 1]
  profile_idc = config[1] & 0x1f
  compatibility = reverse_bits32(int.from_bytes(config[2:6], "big"))
  tier = "H" if config[1] & 0x20 else "L"
  constraints = config[6:12].hex().upper().rstrip("0")
  if len(constraints) % 2:
    constraints += "0"
  suffix = f".{constraints}" if constraints else ""
  return f"{sample_type.decode()}.{profile_prefix}{profile_idc}.{compatibility:X}.{tier}{config[12]}{suffix}"


def parse_track(data: bytes | bytearray, trak: Box) -> Track | None:
  tkhd = child(data, trak, b"tkhd")
  version = data[tkhd.payload_start]
  track_id_offset = tkhd.payload_start + (20 if version == 1 else 12)
  track_id = struct.unpack_from(">I", data, track_id_offset)[0]
  mdhd = descendant(data, trak, b"mdia", b"mdhd")
  timescale_offset = mdhd.payload_start + (20 if data[mdhd.payload_start] == 1 else 12)
  timescale = struct.unpack_from(">I", data, timescale_offset)[0]
  if timescale <= 0:
    raise ValueError("invalid media timescale")
  stsd = descendant(data, trak, b"mdia", b"minf", b"stbl", b"stsd")
  entries = sample_entries(data, stsd)
  if len(entries) != 1:
    raise ValueError("CENC packaging requires one sample description per track")
  entry = entries[0]
  children = {box.type: box for box in sample_entry_children(data, entry)}
  if entry.type in {b"hvc1", b"hev1"}:
    hvcc = children.get(b"hvcC")
    if hvcc is None:
      raise ValueError("HEVC sample entry is missing hvcC")
    config = bytes(data[hvcc.payload_start:hvcc.end])
    return Track(track_id, entry.type, hevc_codec(config, entry.type), (config[21] & 3) + 1, 2, timescale)
  if entry.type in {b"avc1", b"avc3"}:
    avcc = children.get(b"avcC")
    if avcc is None or avcc.size < avcc.header_size + 5:
      raise ValueError("AVC sample entry is missing avcC")
    config = bytes(data[avcc.payload_start:avcc.end])
    codec = f"{entry.type.decode()}.{config[1]:02X}{config[2]:02X}{config[3]:02X}"
    return Track(track_id, entry.type, codec, (config[4] & 3) + 1, 1, timescale)
  if entry.type == b"mp4a":
    return Track(track_id, entry.type, "mp4a.40.2", None, None, timescale)
  return None


def protection_boxes(original_type: bytes, kid: bytes) -> bytes:
  frma = make_box(b"frma", original_type)
  schm = make_box(b"schm", b"\0\0\0\0cenc\0\x01\0\0")
  tenc = make_box(b"tenc", b"\0\0\0\0\0\0\x01\x10" + kid)
  return make_box(b"sinf", frma + schm + make_box(b"schi", tenc))


def transform_sample_entry(data: bytes | bytearray, entry: Box, kid: bytes) -> bytes:
  raw = bytes(data[entry.start:entry.end])
  if entry.type not in VIDEO_SAMPLE_TYPES | AUDIO_SAMPLE_TYPES:
    return raw
  encrypted_type = b"encv" if entry.type in VIDEO_SAMPLE_TYPES else b"enca"
  return make_box(encrypted_type, raw[8:] + protection_boxes(entry.type, kid))


def transform_moov_box(data: bytes | bytearray, box: Box, kid: bytes) -> bytes:
  if box.type == b"stsd":
    prefix = bytes(data[box.payload_start:box.payload_start + 8])
    return make_box(box.type, prefix + b"".join(transform_sample_entry(data, entry, kid) for entry in sample_entries(data, box)))
  if box.type not in CONTAINER_TYPES:
    return bytes(data[box.start:box.end])
  payload = b"".join(transform_moov_box(data, nested, kid) for nested in read_boxes(data, box.payload_start, box.end))
  if box.type == b"moov":
    pssh_payload = b"\x01\0\0\0" + COMMON_PSSH_SYSTEM_ID + struct.pack(">I", 1) + kid + b"\0\0\0\0"
    payload += make_box(b"pssh", pssh_payload)
  return make_box(box.type, payload)


def fullbox_flags(data: bytes | bytearray, box: Box) -> int:
  return int.from_bytes(data[box.payload_start + 1:box.payload_start + 4], "big")


def parse_tfhd(data: bytes | bytearray, box: Box, moof_start: int) -> tuple[int, int, int | None, int | None]:
  flags = fullbox_flags(data, box)
  cursor = box.payload_start + 4
  track_id = struct.unpack_from(">I", data, cursor)[0]
  cursor += 4
  if flags & 0x000001:
    base_data_offset = struct.unpack_from(">Q", data, cursor)[0]
    cursor += 8
  elif flags & 0x020000:
    base_data_offset = moof_start
  else:
    raise ValueError("CENC packaging requires default-base-is-moof fragments")
  if flags & 0x000002:
    cursor += 4
  default_sample_duration = None
  if flags & 0x000008:
    default_sample_duration = struct.unpack_from(">I", data, cursor)[0]
    cursor += 4
  default_sample_size = None
  if flags & 0x000010:
    default_sample_size = struct.unpack_from(">I", data, cursor)[0]
    cursor += 4
  return track_id, base_data_offset, default_sample_duration, default_sample_size


def parse_trun(data: bytes | bytearray, box: Box, default_sample_duration: int | None, default_sample_size: int | None,
               previous_end: int | None, base_data_offset: int) -> Run:
  flags = fullbox_flags(data, box)
  cursor = box.payload_start + 4
  sample_count = struct.unpack_from(">I", data, cursor)[0]
  cursor += 4
  if flags & 0x000001:
    data_offset = base_data_offset + struct.unpack_from(">i", data, cursor)[0]
    cursor += 4
  elif previous_end is not None:
    data_offset = previous_end
  else:
    raise ValueError("first trun must declare a data offset")
  if flags & 0x000004:
    cursor += 4
  sample_sizes: list[int] = []
  sample_durations: list[int] = []
  for _ in range(sample_count):
    if flags & 0x000100:
      sample_duration = struct.unpack_from(">I", data, cursor)[0]
      cursor += 4
    elif default_sample_duration is not None:
      sample_duration = default_sample_duration
    else:
      raise ValueError("fragment does not declare sample durations")
    sample_durations.append(sample_duration)
    if flags & 0x000200:
      sample_size = struct.unpack_from(">I", data, cursor)[0]
      cursor += 4
    elif default_sample_size is not None:
      sample_size = default_sample_size
    else:
      raise ValueError("fragment does not declare sample sizes")
    sample_sizes.append(sample_size)
    if flags & 0x000400:
      cursor += 4
    if flags & 0x000800:
      cursor += 4
  if cursor != box.end:
    raise ValueError("invalid trun box")
  return Run(box, data_offset, tuple(sample_sizes), tuple(sample_durations))


def encrypt_sample(data: bytearray, offset: int, size: int, track: Track, key: bytes, iv: bytes) -> bytes:
  if offset < 0 or offset + size > len(data):
    raise ValueError("sample is outside the MP4 file")
  encryptor = Cipher(algorithms.AES(key), modes.CTR(iv)).encryptor()
  if track.nal_length_size is None or track.nal_header_size is None:
    data[offset:offset + size] = encryptor.update(bytes(data[offset:offset + size])) + encryptor.finalize()
    return iv

  cursor = offset
  end = offset + size
  subsamples: list[tuple[int, int]] = []
  while cursor < end:
    length_end = cursor + track.nal_length_size
    if length_end > end:
      raise ValueError("truncated NAL length")
    nal_size = int.from_bytes(data[cursor:length_end], "big")
    if nal_size < track.nal_header_size or length_end + nal_size > end:
      raise ValueError("invalid NAL size")
    clear_size = track.nal_length_size + track.nal_header_size
    encrypted_size = nal_size - track.nal_header_size
    encrypted_start = cursor + clear_size
    data[encrypted_start:encrypted_start + encrypted_size] = encryptor.update(bytes(data[encrypted_start:encrypted_start + encrypted_size]))
    subsamples.append((clear_size, encrypted_size))
    cursor = length_end + nal_size
  encryptor.finalize()
  if len(subsamples) > 0xffff:
    raise ValueError("too many CENC subsamples")
  entry = bytearray(iv + struct.pack(">H", len(subsamples)))
  for clear_size, encrypted_size in subsamples:
    entry += struct.pack(">HI", clear_size, encrypted_size)
  return bytes(entry)


def fragment_auxiliary_info(data: bytearray, moof: Box, tracks: dict[int, Track], key: bytes) -> dict[int, AuxiliaryInfo]:
  result: dict[int, AuxiliaryInfo] = {}
  for traf in (box for box in read_boxes(data, moof.payload_start, moof.end) if box.type == b"traf"):
    tfhd = child(data, traf, b"tfhd")
    track_id, base_data_offset, default_sample_duration, default_sample_size = parse_tfhd(data, tfhd, moof.start)
    track = tracks.get(track_id)
    if track is None:
      raise ValueError(f"unsupported track {track_id}")
    runs: list[Run] = []
    previous_end = None
    for trun in (box for box in read_boxes(data, traf.payload_start, traf.end) if box.type == b"trun"):
      run = parse_trun(data, trun, default_sample_duration, default_sample_size, previous_end, base_data_offset)
      runs.append(run)
      previous_end = run.data_offset + sum(run.sample_sizes)

    entries: list[bytes] = []
    for run in runs:
      sample_offset = run.data_offset
      for sample_size in run.sample_sizes:
        iv = os.urandom(16)
        entry = encrypt_sample(data, sample_offset, sample_size, track, key, iv)
        if len(entry) > 255:
          raise ValueError("CENC auxiliary sample information is too large")
        entries.append(entry)
        sample_offset += sample_size
    result[traf.start] = AuxiliaryInfo(tuple(entries), track.nal_length_size is not None)
  return result


def patch_trun_data_offset(data: bytes | bytearray, box: Box, delta: int) -> bytes:
  raw = bytearray(data[box.start:box.end])
  if fullbox_flags(data, box) & 0x000001:
    offset_position = box.header_size + 8
    previous = struct.unpack_from(">i", raw, offset_position)[0]
    struct.pack_into(">i", raw, offset_position, previous + delta)
  return bytes(raw)


def transform_moof(data: bytes | bytearray, moof: Box, auxiliary: dict[int, AuxiliaryInfo]) -> bytes:
  added_size = sum(info.added_size for info in auxiliary.values())
  payload = bytearray()
  for nested in read_boxes(data, moof.payload_start, moof.end):
    if nested.type != b"traf":
      payload += data[nested.start:nested.end]
      continue
    info = auxiliary[nested.start]
    traf_payload = bytearray()
    for traf_child in read_boxes(data, nested.payload_start, nested.end):
      if traf_child.type == b"trun":
        traf_payload += patch_trun_data_offset(data, traf_child, added_size)
      else:
        traf_payload += data[traf_child.start:traf_child.end]

    traf_start_in_moof = 8 + len(payload)
    empty_saiz, empty_saio, senc = info.boxes(0)
    senc_start_in_moof = traf_start_in_moof + 8 + len(traf_payload) + len(empty_saiz) + len(empty_saio)
    saiz, saio, senc = info.boxes(senc_start_in_moof + 16)
    payload += make_box(b"traf", bytes(traf_payload) + saiz + saio + senc)
  transformed = make_box(b"moof", bytes(payload))
  if len(transformed) != moof.size + added_size:
    raise ValueError("unexpected CENC moof size")
  return transformed


def fragment_timing(data: bytes | bytearray, moof: Box, track: Track) -> tuple[float, float]:
  for traf in (box for box in read_boxes(data, moof.payload_start, moof.end) if box.type == b"traf"):
    tfhd = child(data, traf, b"tfhd")
    track_id, base_data_offset, default_sample_duration, default_sample_size = parse_tfhd(data, tfhd, moof.start)
    if track_id != track.track_id:
      continue
    tfdt = child(data, traf, b"tfdt")
    decode_time = struct.unpack_from(">Q" if data[tfdt.payload_start] == 1 else ">I", data, tfdt.payload_start + 4)[0]
    duration = 0
    previous_end = None
    for trun in (box for box in read_boxes(data, traf.payload_start, traf.end) if box.type == b"trun"):
      run = parse_trun(data, trun, default_sample_duration, default_sample_size, previous_end, base_data_offset)
      duration += sum(run.sample_durations)
      previous_end = run.data_offset + sum(run.sample_sizes)
    return decode_time / track.timescale, duration / track.timescale
  raise ValueError("media fragment is missing the primary video track")


def derive_cenc_key(folder_key: bytes, owner: str, key_id: str, path: str) -> tuple[bytes, bytes]:
  if len(folder_key) != 32:
    raise ValueError("folder key must be 256 bits")
  key = HKDF(
    algorithm=hashes.SHA256(),
    length=16,
    salt=f"asius-data-cenc-v1:{owner}:{key_id}".encode(),
    info=path.encode(),
  ).derive(folder_key)
  kid = hashlib.sha256(f"asius-data-cenc-kid-v1:{owner}:{key_id}:{path}".encode()).digest()[:16]
  return key, kid


def package_cenc_mp4(source: str | Path, destination: str | Path, key: bytes, kid: bytes) -> dict[str, Any]:
  if len(key) != 16 or len(kid) != 16:
    raise ValueError("CENC key and KID must be 128 bits")
  source_path = Path(source)
  clear = bytearray(source_path.read_bytes())
  top_level = read_boxes(clear, 0, len(clear))
  moov = next((box for box in top_level if box.type == b"moov"), None)
  first_moof = next((box for box in top_level if box.type == b"moof"), None)
  if moov is None or first_moof is None or moov.start > first_moof.start:
    raise ValueError("input must be a fast-start fragmented MP4")

  tracks = {
    track.track_id: track
    for trak in (box for box in read_boxes(clear, moov.payload_start, moov.end) if box.type == b"trak")
    if (track := parse_track(clear, trak)) is not None
  }
  if not tracks or not any(track.sample_type in VIDEO_SAMPLE_TYPES for track in tracks.values()):
    raise ValueError("MP4 has no supported video track")

  auxiliary = {
    box.start: fragment_auxiliary_info(clear, box, tracks, key)
    for box in top_level
    if box.type == b"moof"
  }
  output = bytearray()
  for box in top_level:
    if box.type == b"moov":
      output += transform_moov_box(clear, box, kid)
    elif box.type == b"moof":
      output += transform_moof(clear, box, auxiliary[box.start])
    elif box.type != b"mfra":
      output += clear[box.start:box.end]

  destination_path = Path(destination)
  destination_path.write_bytes(output)
  output_boxes = read_boxes(output, 0, len(output))
  moof_starts = [box.start for box in output_boxes if box.type == b"moof"]
  if not moof_starts:
    raise ValueError("packaged MP4 has no media fragments")
  ranges = [(0, moof_starts[0]), *[(start, moof_starts[index + 1] if index + 1 < len(moof_starts) else len(output)) for index, start in enumerate(moof_starts)]]
  fragments = [
    {
      "kind": "init" if index == 0 else "media",
      "offset": start,
      "length": end - start,
      "sha256": b64url(hashlib.sha256(output[start:end]).digest()),
    }
    for index, (start, end) in enumerate(ranges)
  ]
  primary = next(track for track in tracks.values() if track.sample_type in VIDEO_SAMPLE_TYPES)
  timings = [fragment_timing(clear, box, primary) for box in top_level if box.type == b"moof"]
  for fragment, (start_time, duration) in zip(fragments[1:], timings, strict=True):
    fragment["timeMs"] = round(start_time * 1000)
    fragment["durationMs"] = max(1, round(duration * 1000))
  return {
    "v": 1,
    "alg": "CENC-AES-CTR",
    "kid": b64url(kid),
    "codec": primary.codec,
    "contentType": f'video/mp4; codecs="{primary.codec}"',
    "plaintextLength": len(clear),
    "encryptedLength": len(output),
    "checksumSha256": b64url(hashlib.sha256(output).digest()),
    "fragments": fragments,
  }
