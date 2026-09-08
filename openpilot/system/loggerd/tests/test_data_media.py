import base64
import hashlib
import json
import shutil
import subprocess

import pytest

from openpilot.system.loggerd.data_media import derive_cenc_key, package_cenc_mp4


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def frame_hash(path, key: bytes | None = None) -> list[str]:
  command = [FFMPEG, "-hide_banner", "-loglevel", "error"]
  if key is not None:
    command += ["-decryption_key", key.hex()]
  command += ["-i", str(path), "-map", "0:v:0", "-f", "framehash", "-hash", "sha256", "-"]
  output = subprocess.check_output(command, text=True)
  return [line.rsplit(",", 1)[-1].strip() for line in output.splitlines() if line and not line.startswith("#")]


@pytest.mark.skipif(FFMPEG is None or FFPROBE is None, reason="FFmpeg is required")
def test_cenc_fmp4_round_trip(tmp_path) -> None:
  clear = tmp_path / "clear.mp4"
  encrypted = tmp_path / "encrypted.mp4"
  frame = bytes([16]) * (64 * 64) + bytes([128]) * (64 * 64 // 2)
  subprocess.run(
    [
      FFMPEG,
      "-hide_banner", "-loglevel", "error", "-y",
      "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", "64x64", "-r", "20", "-i", "pipe:0",
      "-c:v", "libx264", "-preset", "ultrafast", "-g", "20", "-bf", "0",
      "-movflags", "+frag_keyframe+empty_moov+default_base_moof+skip_sidx", str(clear),
    ],
    input=frame * 40,
    check=True,
  )
  key = bytes.fromhex("00112233445566778899aabbccddeeff")
  kid = bytes.fromhex("ffeeddccbbaa99887766554433221100")
  manifest = package_cenc_mp4(clear, encrypted, key, kid)

  assert manifest["alg"] == "CENC-AES-CTR"
  assert manifest["codec"].startswith("avc1.")
  assert manifest["kid"] == base64.urlsafe_b64encode(kid).decode().rstrip("=")
  assert manifest["encryptedLength"] == encrypted.stat().st_size
  assert manifest["fragments"][0]["kind"] == "init"
  assert all(fragment["kind"] == "media" for fragment in manifest["fragments"][1:])
  assert all(isinstance(fragment["timeMs"], int) for fragment in manifest["fragments"][1:])
  assert all(isinstance(fragment["durationMs"], int) for fragment in manifest["fragments"][1:])

  payload = encrypted.read_bytes()
  cursor = 0
  for fragment in manifest["fragments"]:
    assert fragment["offset"] == cursor
    chunk = payload[cursor:cursor + fragment["length"]]
    assert base64.urlsafe_b64encode(hashlib.sha256(chunk).digest()).decode().rstrip("=") == fragment["sha256"]
    cursor += fragment["length"]
  assert cursor == len(payload)

  probe = subprocess.check_output(
    [FFPROBE, "-v", "error", "-decryption_key", key.hex(), "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "json", str(encrypted)],
    text=True,
  )
  assert json.loads(probe)["streams"][0]["nb_read_packets"] == "40"
  assert frame_hash(clear) == frame_hash(encrypted, key)


def test_cenc_key_is_scoped_to_owner_epoch_and_path() -> None:
  folder_key = bytes(range(32))
  first = derive_cenc_key(folder_key, "owner", "epoch", "routes/route--0/fcamera.mp4")
  assert first == derive_cenc_key(folder_key, "owner", "epoch", "routes/route--0/fcamera.mp4")
  assert first != derive_cenc_key(folder_key, "owner", "epoch", "routes/route--1/fcamera.mp4")
  assert first != derive_cenc_key(folder_key, "owner", "next", "routes/route--0/fcamera.mp4")
