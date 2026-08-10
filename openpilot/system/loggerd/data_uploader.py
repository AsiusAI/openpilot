#!/usr/bin/env python3
from __future__ import annotations

import base64
import os
import random
import re
import tempfile
import threading
import time
from collections.abc import Iterator

import requests

import openpilot.cereal.messaging as messaging
from openpilot.common.hardware.hw import Paths
from openpilot.common.params import Params
from openpilot.common.realtime import set_core_affinity
from openpilot.common.swaglog import cloudlog
from openpilot.common.utils import get_upload_stream
from openpilot.system.app.identity import get_device_private_key, is_dongle_id
from openpilot.system.app.websocketd import load_authorized_peers
from openpilot.system.loggerd.data_api import (
  DataApiClient,
  access_document,
  b64url,
  can_wrap_for,
  canonical_json,
  encrypt_object,
  initial_state,
  public_identity,
  recover_state,
  update_state,
)
from openpilot.system.loggerd.data_media import derive_cenc_key, package_cenc_mp4
from openpilot.system.loggerd.uploader import NetworkType, Uploader, allow_sleep, clear_locks, force_wifi

UPLOAD_ATTR_NAME = "user.asius_upload"
UPLOAD_ATTR_VALUE = b"1"
STATE_PARAM = "DataUploadState"
MAX_FILE_BYTES = 128 * 1024 * 1024
ROUTE_FILES = {"qlog", "qlog.zst", "rlog", "rlog.zst", "qcamera.mp4", "fcamera.mp4", "ecamera.mp4", "dcamera.mp4"}
CENC_FILES = {"qcamera.mp4", "fcamera.mp4", "ecamera.mp4", "dcamera.mp4"}
ROUTE_SEGMENT_RE = re.compile(r"^.+--\d+$")


class DataUploader(Uploader):
  upload_attr_name = UPLOAD_ATTR_NAME
  upload_attr_value = UPLOAD_ATTR_VALUE
  max_file_size = MAX_FILE_BYTES
  upload_all_files = True

  def __init__(self, dongle_id: str, root: str, params: Params | None = None, client: DataApiClient | None = None):
    self.private_key = get_device_private_key()
    self.owner = public_identity(self.private_key)
    if dongle_id != self.owner:
      raise ValueError("DongleId does not match the device identity")

    super().__init__(dongle_id, root)
    self.immediate_folders = []
    self.immediate_priority = {"qlog": 0, "qlog.zst": 0, "qcamera.mp4": 1, "fcamera.mp4": 2, "ecamera.mp4": 3, "dcamera.mp4": 4}
    self.params = params or self.params
    self.client = client or DataApiClient(self.params.get("DataApiHost", return_default=True), self.private_key)

  def list_upload_files(self, metered: bool) -> Iterator[tuple[str, str, str]]:
    for name, key, filename in super().list_upload_files(metered):
      folder = key.split("/", 1)[0]
      if name in ROUTE_FILES and ROUTE_SEGMENT_RE.fullmatch(folder):
        yield name, key, filename

  def sync_access(self) -> dict:
    readers = []
    for key in sorted(load_authorized_peers()):
      if is_dongle_id(key) and can_wrap_for(self.private_key, key):
        readers.append(key)
      else:
        cloudlog.event("data_upload_invalid_recipient", recipient=key)
    if self.params.get_bool("ShareDrivingData"):
      asius_reader = self.client.get_config().get("retentionPublicKey")
      if not isinstance(asius_reader, str) or not is_dongle_id(asius_reader) or not can_wrap_for(self.private_key, asius_reader):
        raise ValueError("Data API returned an invalid retention public key")
      readers.append(asius_reader)
    readers = sorted(set(readers))
    state = self.params.get(STATE_PARAM)
    state = initial_state(readers) if not isinstance(state, dict) else update_state(state, readers)
    if state.get("publishedVersion") == state["version"]:
      return state

    try:
      self.client.put_access(access_document(self.private_key, state, readers))
    except requests.HTTPError as error:
      if error.response is None or error.response.status_code != 409:
        raise
      state = recover_state(self.private_key, self.client.get_access(), readers)
      self.client.put_access(access_document(self.private_key, state, readers))

    state["publishedVersion"] = state["version"]
    self.params.put(STATE_PARAM, state, block=True)
    return state

  def do_upload(self, key: str, fn: str) -> requests.Response:
    state = self.sync_access()
    first_component = key.split("/", 1)[0]
    if not ROUTE_SEGMENT_RE.fullmatch(first_component):
      raise ValueError("data uploader only accepts route segments")
    object_path = f"routes/{key}"
    segment = int(first_component.rsplit("--", 1)[1])
    route_start_time = int((os.path.getctime(os.path.dirname(fn)) - segment * 60) * 1000)

    if os.path.basename(object_path) in CENC_FILES:
      active = next(entry for entry in state["keys"] if entry["id"] == state["activeKey"])
      folder_key = base64.urlsafe_b64decode(active["key"] + "=" * (-len(active["key"]) % 4))
      content_key, kid = derive_cenc_key(folder_key, self.owner, active["id"], object_path)
      temporary_name = ""
      try:
        with tempfile.NamedTemporaryFile(prefix=".asius-cenc-", suffix=".mp4", dir=os.path.dirname(fn), delete=False) as temporary:
          temporary_name = temporary.name
        media = package_cenc_mp4(fn, temporary_name, content_key, kid)
        unsigned_media = media | {"owner": self.owner, "path": object_path, "keyId": active["id"]}
        signed_media = unsigned_media | {"signature": b64url(self.private_key.sign(canonical_json(unsigned_media).encode()))}
        response = self.client.upload_file(
          object_path,
          temporary_name,
          content_type="video/mp4",
          plaintext_length=os.path.getsize(fn),
          route_start_time=route_start_time,
          checksum=media["checksumSha256"],
          media=signed_media,
        )
        cloudlog.event("data_upload_cenc", key=key, path=object_path, encrypted_size=media["encryptedLength"], fragments=len(media["fragments"]))
        return response
      finally:
        if temporary_name:
          try:
            os.unlink(temporary_name)
          except FileNotFoundError:
            pass

    stream, _ = get_upload_stream(fn, key.endswith(".zst") and not fn.endswith(".zst"))
    try:
      plaintext = stream.read()
      encrypted = encrypt_object(self.owner, object_path, plaintext, state)
    finally:
      stream.close()

    response = self.client.upload(
      object_path,
      encrypted,
      plaintext_length=len(plaintext),
      route_start_time=route_start_time,
    )
    cloudlog.event("data_upload_encrypted", key=key, path=object_path, encrypted_size=len(encrypted))
    return response


def main(exit_event: threading.Event | None = None) -> None:
  exit_event = exit_event or threading.Event()
  try:
    set_core_affinity([0, 1, 2, 3])
  except Exception:
    cloudlog.exception("failed to set core affinity")

  clear_locks(Paths.log_root())
  params = Params()
  dongle_id = params.get("DongleId")
  if dongle_id is None:
    raise RuntimeError("data uploader cannot start without DongleId")

  sm = messaging.SubMaster(["deviceState"])
  uploader = DataUploader(dongle_id, Paths.log_root(), params=params)
  backoff = 0.1
  while not exit_event.is_set():
    sm.update(0)
    offroad = params.get_bool("IsOffroad")
    network_type = sm["deviceState"].networkType if not force_wifi else NetworkType.wifi
    if network_type == NetworkType.none:
      if allow_sleep:
        time.sleep(60 if offroad else 5)
      continue

    try:
      uploader.sync_access()
    except Exception:
      cloudlog.exception("data access sync failed")
      if allow_sleep:
        time.sleep(backoff + random.uniform(0, backoff))
      backoff = min(backoff * 2, 120)
      continue

    success = uploader.step(sm["deviceState"].networkType.raw, sm["deviceState"].networkMetered)
    if success is None:
      backoff = 60 if offroad else 5
    elif success:
      backoff = 0.1
    else:
      cloudlog.info("data upload backoff %r", backoff)
      backoff = min(backoff * 2, 120)
    if allow_sleep:
      time.sleep(backoff + random.uniform(0, backoff))


if __name__ == "__main__":
  main()
