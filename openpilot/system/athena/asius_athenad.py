#!/usr/bin/env python3
"""Asius-specific Athena RPC and live device integration."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from functools import partial, total_ordering
from pathlib import Path
from queue import Queue
from typing import Any, Protocol, cast
from collections.abc import Callable

import requests
from requests.adapters import HTTPAdapter, DEFAULT_POOLBLOCK
from websocket import ABNF, WebSocket, WebSocketTimeoutException

import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from openpilot.cereal.services import SERVICE_LIST
from openpilot.common.api import get_key_pair
from openpilot.common.utils import CallbackReader, get_upload_stream
from openpilot.common.params import Params
from openpilot.common.hardware import HARDWARE
from openpilot.common.swaglog import cloudlog
from openpilot.common.version import get_build_metadata
from openpilot.common.hardware.hw import Paths
from openpilot.system.athena.rpc import Dispatcher, handle
from openpilot.system.athena.terminal import TerminalManager
from openpilot.system.athena.websocketd import (
  authorize_peer,
  bump_acl_epoch,
  get_acl_epoch,
  load_authorized_peers,
  pack_peer_message,
  pairing_mode_active,
  pairing_url,
  save_authorized_peers,
  unpack_peer_message,
  verify_pair_token,
)


class ParamsReader(Protocol):
  def get(self, key: str) -> Any: ...


def main(exit_event: threading.Event | None = None) -> None:
  from openpilot.system.athena.websocketd import main as websocket_main
  websocket_main(exit_event)


ATHENA_HOST = Params().get("AthenaHost", return_default=True)

RECONNECT_TIMEOUT_S = 70

RETRY_DELAY = 10  # seconds
MAX_RETRY_COUNT = 30  # Try for at most 5 minutes if upload fails immediately
MAX_AGE = 31 * 24 * 3600  # seconds
WS_FRAME_SIZE = 4096
DEVICE_STATE_UPDATE_INTERVAL = 1.0  # in seconds
DEFAULT_UPLOAD_PRIORITY = 99  # higher number = lower priority
LIVE_STATE_INTERVAL_S = 1.0
VAMOS_UPDATE_STATE_FILE = Path("/data/vamos-update/state.json")
SAVE_PARAMS_BLOCKED_KEYS = {
  "AccessToken",
  "ApiCache_Device",
  "AthenadAuthorizedKeys",
  "AthenadAuthorizedKeysEpoch",
  "AthenadPairingUntil",
  "AthenadUploadQueue",
  "DoUninstall",
  "DongleId",
  "GithubSshKeys",
  "GithubUsername",
  "HardwareSerial",
  "LastAthenaPingTime",
  "SecOCKey",
  "AthenadPid",
}
LIVE_STATE_SERVICES = [
  "deviceState",
  "liveCalibration",
  "managerState",
  "onroadEvents",
  "selfdriveState",
]
LIVE_STATE_PARAM_KEYS = [
  "DongleId",
  "HardwareSerial",
  "LastAthenaPingTime",
  "OpenpilotEnabledToggle",
  "ExperimentalMode",
  "ExperimentalModeConfirmed",
  "DisengageOnAccelerator",
  "LongitudinalPersonality",
  "IsLdwEnabled",
  "RecordFront",
  "RecordAudio",
  "IsMetric",
  "SshEnabled",
  "AdbEnabled",
  "JoystickDebugMode",
  "LongitudinalManeuverMode",
  "LateralManeuverMode",
  "AlphaLongitudinalEnabled",
  "ShowDebugInfo",
  "GsmMetered",
  "GsmRoaming",
  "GsmApn",
  "UpdaterState",
  "UpdaterProgress",
  "UpdateAvailable",
  "UpdateFailedCount",
  "UpdaterFetchAvailable",
  "UpdaterCurrentDescription",
  "UpdaterNewDescription",
  "UpdaterTargetBranch",
  "UpdaterAvailableBranches",
  "UpdaterLastFetchTime",
  "LastUpdateTime",
  "LastUpdateException",
]

# https://bytesolutions.com/dscp-tos-cos-precedence-conversion-chart,
# https://en.wikipedia.org/wiki/Differentiated_services
UPLOAD_TOS = 0x20  # CS1, low priority background traffic

NetworkType = log.DeviceState.NetworkType

UploadFileDict = dict[str, str | int | float | bool]
UploadItemDict = dict[str, str | bool | int | float | dict[str, str]]

UploadFilesToUrlResponse = dict[str, int | list[UploadItemDict] | list[str]]


class UploadTOSAdapter(HTTPAdapter):
  def init_poolmanager(self, connections, maxsize, block=DEFAULT_POOLBLOCK, **pool_kwargs):
    pool_kwargs["socket_options"] = [(socket.IPPROTO_IP, socket.IP_TOS, UPLOAD_TOS)]
    super().init_poolmanager(connections, maxsize, block, **pool_kwargs)


UPLOAD_SESS = requests.Session()
UPLOAD_SESS.mount("http://", UploadTOSAdapter())
UPLOAD_SESS.mount("https://", UploadTOSAdapter())


@dataclass
class UploadFile:
  fn: str
  url: str
  headers: dict[str, str]
  allow_cellular: bool
  priority: int = DEFAULT_UPLOAD_PRIORITY

  @classmethod
  def from_dict(cls, d: dict) -> UploadFile:
    return cls(d.get("fn", ""), d.get("url", ""), d.get("headers", {}), d.get("allow_cellular", False), d.get("priority", DEFAULT_UPLOAD_PRIORITY))


@dataclass
@total_ordering
class UploadItem:
  path: str
  url: str
  headers: dict[str, str]
  created_at: int
  id: str | None
  retry_count: int = 0
  current: bool = False
  progress: float = 0
  allow_cellular: bool = False
  priority: int = DEFAULT_UPLOAD_PRIORITY

  @classmethod
  def from_dict(cls, d: dict) -> UploadItem:
    return cls(d["path"], d["url"], d["headers"], d["created_at"], d["id"], d["retry_count"], d["current"],
               d["progress"], d["allow_cellular"], d["priority"])

  def __lt__(self, other):
    if not isinstance(other, UploadItem):
      return NotImplemented
    return self.priority < other.priority

  def __eq__(self, other):
    if not isinstance(other, UploadItem):
      return NotImplemented
    return self.priority == other.priority


dispatcher = Dispatcher()
dispatcher["echo"] = lambda s: s
send_queue: Queue[str] = queue.Queue()
upload_queue: Queue[UploadItem] = queue.PriorityQueue()
cancelled_uploads: set[str] = set()

cur_upload_items: dict[int, UploadItem | None] = {}


def strip_zst_extension(fn: str) -> str:
  if fn.endswith('.zst'):
    return fn[:-4]
  return fn


class AbortTransferException(Exception):
  pass


class UploadQueueCache:

  @staticmethod
  def initialize(upload_queue: Queue[UploadItem]) -> None:
    try:
      upload_queue_json = Params().get("AthenadUploadQueue")
      if upload_queue_json is not None:
        for item in upload_queue_json:
          upload_queue.put(UploadItem.from_dict(item))
    except Exception:
      cloudlog.exception("athena.UploadQueueCache.initialize.exception")

  @staticmethod
  def cache(upload_queue: Queue[UploadItem]) -> None:
    try:
      queue: list[UploadItem | None] = list(upload_queue.queue)
      items = [asdict(i) for i in queue if i is not None and (i.id not in cancelled_uploads)]
      Params().put("AthenadUploadQueue", items, block=True)
    except Exception:
      cloudlog.exception("athena.UploadQueueCache.cache.exception")


def handle_long_poll(ws: WebSocket, exit_event: threading.Event | None) -> None:
  end_event = threading.Event()

  threads = [
    threading.Thread(target=ws_manage, args=(ws, end_event), name='ws_manage'),
    threading.Thread(target=ws_recv, args=(ws, end_event), name='ws_recv'),
    threading.Thread(target=ws_send, args=(ws, end_event), name='ws_send'),
    threading.Thread(target=upload_handler, args=(end_event,), name='upload_handler'),
    threading.Thread(target=upload_handler, args=(end_event,), name='upload_handler2'),
    threading.Thread(target=upload_handler, args=(end_event,), name='upload_handler3'),
    threading.Thread(target=upload_handler, args=(end_event,), name='upload_handler4'),
    threading.Thread(target=live_state_handler, args=(end_event,), name='live_state_handler'),
  ]

  for thread in threads:
    thread.start()
  try:
    while not end_event.wait(0.1):
      if exit_event is not None and exit_event.is_set():
        end_event.set()
  except (KeyboardInterrupt, SystemExit):
    end_event.set()
    raise
  finally:
    for thread in threads:
      cloudlog.debug(f"athena.joining {thread.name}")
      thread.join()


def retry_upload(tid: int, end_event: threading.Event, increase_count: bool = True) -> None:
  item = cur_upload_items[tid]
  if item is not None and item.retry_count < MAX_RETRY_COUNT:
    new_retry_count = item.retry_count + 1 if increase_count else item.retry_count

    item = replace(
      item,
      retry_count=new_retry_count,
      progress=0,
      current=False
    )
    upload_queue.put_nowait(item)
    UploadQueueCache.cache(upload_queue)

    cur_upload_items[tid] = None

    for _ in range(RETRY_DELAY):
      time.sleep(1)
      if end_event.is_set():
        break


def cb(sm, item, tid, end_event: threading.Event, sz: int, cur: int) -> None:
  # Abort transfer if connection changed to metered after starting upload
  # or if athenad is shutting down to re-connect the websocket
  if not item.allow_cellular:
    if (time.monotonic() - sm.recv_time['deviceState']) > DEVICE_STATE_UPDATE_INTERVAL:
      sm.update(0)
      if sm['deviceState'].networkMetered:
        raise AbortTransferException

  if end_event.is_set():
    raise AbortTransferException

  cur_upload_items[tid] = replace(item, progress=cur / sz if sz else 1)


def upload_handler(end_event: threading.Event) -> None:
  sm = messaging.SubMaster(['deviceState'])
  tid = threading.get_ident()

  while not end_event.is_set():
    cur_upload_items[tid] = None

    try:
      cur_upload_items[tid] = item = replace(upload_queue.get(timeout=1), current=True)

      if item.id in cancelled_uploads:
        cancelled_uploads.remove(item.id)
        continue

      # Remove item if too old
      age = datetime.now() - datetime.fromtimestamp(item.created_at / 1000)
      if age.total_seconds() > MAX_AGE:
        cloudlog.event("athena.upload_handler.expired", item=item, error=True)
        continue

      # Check if uploading over metered connection is allowed
      sm.update(0)
      metered = sm['deviceState'].networkMetered
      network_type = sm['deviceState'].networkType.raw
      if metered and (not item.allow_cellular):
        retry_upload(tid, end_event, False)
        continue

      try:
        fn = item.path
        try:
          sz = os.path.getsize(fn)
        except OSError:
          sz = -1

        cloudlog.event("athena.upload_handler.upload_start", fn=fn, sz=sz, network_type=network_type, metered=metered, retry_count=item.retry_count)

        with _do_upload(item, partial(cb, sm, item, tid, end_event)) as response:
          if response.status_code not in (200, 201, 401, 403, 412):
            cloudlog.event("athena.upload_handler.retry", status_code=response.status_code, fn=fn, sz=sz, network_type=network_type, metered=metered)
            retry_upload(tid, end_event)
          else:
            cloudlog.event("athena.upload_handler.success", fn=fn, sz=sz, network_type=network_type, metered=metered)

        UploadQueueCache.cache(upload_queue)
      except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.SSLError):
        cloudlog.event("athena.upload_handler.timeout", fn=fn, sz=sz, network_type=network_type, metered=metered)
        retry_upload(tid, end_event)
      except AbortTransferException:
        cloudlog.event("athena.upload_handler.abort", fn=fn, sz=sz, network_type=network_type, metered=metered)
        retry_upload(tid, end_event, False)

    except queue.Empty:
      pass
    except Exception:
      cloudlog.exception("athena.upload_handler.exception")


def _do_upload(upload_item: UploadItem, callback: Callable | None = None) -> requests.Response:
  path = upload_item.path
  compress = False

  # If file does not exist, but does exist without the .zst extension we will compress on the fly
  if not os.path.exists(path) and os.path.exists(strip_zst_extension(path)):
    path = strip_zst_extension(path)
    compress = True

  stream = None
  try:
    stream, content_length = get_upload_stream(path, compress)
    response = UPLOAD_SESS.put(upload_item.url,
                               data=CallbackReader(stream, callback, content_length) if callback else stream,
                               headers={**upload_item.headers, 'Content-Length': str(content_length)},
                               timeout=30)
    return response
  finally:
    if stream:
      stream.close()


# security: user should be able to request any message from their car
@dispatcher.add_method
def getMessage(service: str, timeout: int = 1000) -> dict:
  if service is None or service not in SERVICE_LIST:
    raise Exception("invalid service")

  socket = messaging.sub_sock(service, timeout=timeout)
  try:
    ret = messaging.recv_one(socket)

    if ret is None:
      raise TimeoutError

    # this is because capnp._DynamicStructReader doesn't have typing information
    return cast(dict, ret.to_dict())
  finally:
    del socket


@dispatcher.add_method
def getVersion() -> dict[str, str]:
  build_metadata = get_build_metadata()
  return {
    "version": build_metadata.openpilot.version,
    "remote": build_metadata.openpilot.git_normalized_origin,
    "branch": build_metadata.channel,
    "commit": build_metadata.openpilot.git_commit,
  }


def scan_dir(path: str, prefix: str) -> list[str]:
  files = []
  # only walk directories that match the prefix
  # (glob and friends traverse entire dir tree)
  with os.scandir(path) as i:
    for e in i:
      rel_path = os.path.relpath(e.path, Paths.log_root())
      if e.is_dir(follow_symlinks=False):
        # add trailing slash
        rel_path = os.path.join(rel_path, '')
        # if prefix is a partial dir name, current dir will start with prefix
        # if prefix is a partial file name, prefix with start with dir name
        if rel_path.startswith(prefix) or prefix.startswith(rel_path):
          files.extend(scan_dir(e.path, prefix))
      else:
        if rel_path.startswith(prefix):
          files.append(rel_path)
  return files

@dispatcher.add_method
def listDataDirectory(prefix='') -> list[str]:
  return scan_dir(Paths.log_root(), prefix)


@dispatcher.add_method
def getAllParams() -> dict[str, str | bool | int | float | None]:
  from openpilot.common.params_pyx import ParamKeyType
  import datetime

  params = Params()
  result: dict[str, str | bool | int | float | None] = {}

  for key in [k.decode('utf-8') for k in params.all_keys()]:
    if params.get_type(key) == ParamKeyType.BYTES:
      continue
    value = params.get(key)
    if isinstance(value, datetime.datetime):
      value = value.timestamp()
    result[key] = value

  return result


@dispatcher.add_method
def saveParams(params_to_update: dict[str, str | bool | int | float | dict | list | None]) -> dict[str, str]:
  params = Params()
  results = {}

  for key, value in params_to_update.items():
    try:
      if key in SAVE_PARAMS_BLOCKED_KEYS:
        results[key] = "error: blocked"
        continue
      if value is None:
        params.remove(key)
        results[key] = "ok: removed"
      else:
        if not isinstance(value, (str, bool, int, float)):
          value = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        params.put(key, value)
        results[key] = "ok"
    except Exception as e:
      results[key] = f"error: {e}"

  return results


@dispatcher.add_method
def uploadFileToUrl(fn: str, url: str, headers: dict[str, str]) -> UploadFilesToUrlResponse:
  # this is because mypy doesn't understand that the decorator doesn't change the return type
  response: UploadFilesToUrlResponse = uploadFilesToUrls([{
    "fn": fn,
    "url": url,
    "headers": headers,
  }])
  return response


@dispatcher.add_method
def uploadFilesToUrls(files_data: list[UploadFileDict]) -> UploadFilesToUrlResponse:
  files = map(UploadFile.from_dict, files_data)

  items: list[UploadItemDict] = []
  failed: list[str] = []
  for file in files:
    if len(file.fn) == 0 or file.fn[0] == '/' or '..' in file.fn or len(file.url) == 0:
      failed.append(file.fn)
      continue

    path = os.path.join(Paths.log_root(), file.fn)
    if not os.path.exists(path) and not os.path.exists(strip_zst_extension(path)):
      failed.append(file.fn)
      continue

    # Skip item if already in queue
    url = file.url.split('?')[0]
    if any(url == item['url'].split('?')[0] for item in listUploadQueue()):
      continue

    item = UploadItem(
      path=path,
      url=file.url,
      headers=file.headers,
      created_at=int(time.time() * 1000),  # noqa: TID251
      id=None,
      allow_cellular=file.allow_cellular,
      priority=file.priority,
    )
    upload_id = hashlib.sha1(str(item).encode()).hexdigest()
    item = replace(item, id=upload_id)
    upload_queue.put_nowait(item)
    items.append(asdict(item))

  UploadQueueCache.cache(upload_queue)

  resp: UploadFilesToUrlResponse = {"enqueued": len(items), "items": items}
  if failed:
    cloudlog.event("athena.uploadFilesToUrls.failed", failed=failed, error=True)
    resp["failed"] = failed

  return resp


@dispatcher.add_method
def listUploadQueue() -> list[UploadItemDict]:
  items = list(upload_queue.queue) + list(cur_upload_items.values())
  return [asdict(i) for i in items if (i is not None) and (i.id not in cancelled_uploads)]


@dispatcher.add_method
def cancelUpload(upload_id: str | list[str]) -> dict[str, int | str]:
  if not isinstance(upload_id, list):
    upload_id = [upload_id]

  uploading_ids = {item.id for item in list(upload_queue.queue)}
  cancelled_ids = uploading_ids.intersection(upload_id)
  if len(cancelled_ids) == 0:
    return {"success": 0, "error": "not found"}

  cancelled_uploads.update(cancelled_ids)
  return {"success": 1}

@dispatcher.add_method
def setRouteViewed(route: str) -> dict[str, int | str]:
  # maintain a list of the last 10 routes viewed in connect
  params = Params()

  r = params.get("AthenadRecentlyViewedRoutes")
  routes = [] if r is None else r.split(",")
  routes.append(route)

  # remove duplicates
  routes = list(dict.fromkeys(routes))

  params.put("AthenadRecentlyViewedRoutes", ",".join(routes[-10:]), block=True)
  return {"success": 1}


@dispatcher.add_method
def getPublicKey() -> str | None:
  _, _, public_key = get_key_pair()
  return public_key


@dispatcher.add_method
def getAuthorizedPeers() -> dict[str, Any]:
  return {
    "aclEpoch": get_acl_epoch(),
    "peers": [
      {
        "publicKey": public_key,
        "aclEpoch": peer.get("aclEpoch"),
        "label": peer.get("label"),
        "createdAt": peer.get("createdAt"),
      }
      for public_key, peer in load_authorized_peers().items()
    ],
  }


@dispatcher.add_method
def removeAuthorizedPeer(publicKey: str) -> dict[str, Any]:
  peers = load_authorized_peers()
  removed = peers.pop(publicKey, None) is not None
  if removed:
    save_authorized_peers(peers)
    bump_acl_epoch()

  return {"removed": removed, "aclEpoch": get_acl_epoch()}


@dispatcher.add_method
def getPairingUrl() -> str:
  params = Params()
  dongle_id = params.get("DongleId") or ""
  return pairing_url(dongle_id)


@dispatcher.add_method
def getSshAuthorizedKeys() -> str:
  return cast(str, Params().get("GithubSshKeys") or "")


@dispatcher.add_method
def getGithubUsername() -> str:
  return cast(str, Params().get("GithubUsername") or "")


@dispatcher.add_method
def setGithubUsername(username: str) -> dict[str, str]:
  params = Params()
  username = username.strip()
  if not username:
    params.remove("GithubUsername")
    params.remove("GithubSshKeys")
    return {"GithubUsername": "ok: removed", "GithubSshKeys": "ok: removed"}

  response = requests.get(f"https://github.com/{username}.keys", timeout=15)
  response.raise_for_status()
  keys = response.text.strip()
  if not keys:
    raise Exception(f"No SSH keys found for user '{username}'")

  params.put("GithubUsername", username, block=True)
  params.put("GithubSshKeys", keys, block=True)
  return {"GithubUsername": "ok", "GithubSshKeys": "ok"}


@dispatcher.add_method
def getSimInfo():
  return HARDWARE.get_sim_info()


@dispatcher.add_method
def getNetworkType():
  return HARDWARE.get_network_type()


@dispatcher.add_method
def getNetworkMetered() -> bool:
  network_type = HARDWARE.get_network_type()
  return HARDWARE.get_network_metered(network_type)


@dispatcher.add_method
def getNetworks():
  return HARDWARE.get_networks()


TAILSCALE_COMMAND = ["sudo", "-n", "tailscale"]
TAILSCALE_SOCKET = Path("/var/run/tailscale/tailscaled.sock")
TAILSCALE_ENABLED = Path("/data/tailscale/enabled")


@dispatcher.add_method
def getTailscaleState() -> dict[str, Any]:
  if not TAILSCALE_SOCKET.exists():
    return {"running": False, "connected": False, "user": None}
  try:
    raw = subprocess.check_output([*TAILSCALE_COMMAND, "status", "--json"], stderr=subprocess.STDOUT, encoding="utf-8", timeout=5)
    status = json.loads(raw)
  except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
    return {"running": False, "connected": False, "user": None, "error": str(e)}

  connected = status.get("BackendState") == "Running"
  self_state = status.get("Self", {})
  user_id = str(self_state.get("UserID", ""))
  user = status.get("User", {}).get(user_id, {}).get("LoginName") or status.get("CurrentTailnet", {}).get("Name") if connected else None

  return {
    "running": True,
    "connected": connected,
    "user": user,
    "backendState": status.get("BackendState", "Unknown"),
    "authUrl": status.get("AuthURL") or None,
    "ips": status.get("TailscaleIPs", []),
    "hostname": self_state.get("HostName") or None,
    "dnsName": self_state.get("DNSName") or None,
    "online": bool(self_state.get("Online")),
    "relay": self_state.get("Relay") or None,
    "keyExpiry": self_state.get("KeyExpiry") or None,
  }


@dispatcher.add_method
def configureTailscale(disconnect: bool = False) -> str | None:
  if disconnect:
    subprocess.run([*TAILSCALE_COMMAND, "logout"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    subprocess.run(["sudo", "-n", "rm", "-f", str(TAILSCALE_ENABLED)], check=True)
    subprocess.run(["sudo", "-n", "sv", "down", "tailscaled"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["sudo", "-n", "sv", "down", "tailscaled/log"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return None

  subprocess.run(["sudo", "-n", "install", "-d", "-m", "700", str(TAILSCALE_ENABLED.parent)], check=True)
  subprocess.run(["sudo", "-n", "touch", str(TAILSCALE_ENABLED)], check=True)
  subprocess.run(["sudo", "-n", "rm", "-f", "/var/service/tailscaled/down"], check=True)
  subprocess.run(["sudo", "-n", "rm", "-f", "/var/service/tailscaled/log/down"], check=True)
  subprocess.run(["sudo", "-n", "sv", "up", "tailscaled/log"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  subprocess.run(["sudo", "-n", "sv", "up", "tailscaled"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
  for _ in range(50):
    if TAILSCALE_SOCKET.exists():
      break
    time.sleep(0.1)

  state = getTailscaleState()
  if state["connected"]:
    return None
  if state.get("authUrl"):
    return cast(str, state["authUrl"])

  subprocess.run(
    [*TAILSCALE_COMMAND, "login", "--accept-dns=false", "--ssh=false", "--timeout=1s"],
    check=False,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    timeout=3,
  )
  state = getTailscaleState()
  if state.get("authUrl"):
    return cast(str, state["authUrl"])
  raise RuntimeError("Tailscale did not provide a login URL")


def _local_ips() -> list[dict[str, str]]:
  try:
    output = subprocess.check_output(["ip", "-j", "addr", "show"], encoding="utf-8")
    interfaces = json.loads(output)
    return [
      {"interface": iface.get("ifname", ""), "address": addr.get("local", "")}
      for iface in interfaces
      for addr in iface.get("addr_info", [])
      if addr.get("family") == "inet" and addr.get("local")
    ]
  except Exception:
    cloudlog.exception("athena.local_ips.exception")
    return []


def _nmcli(args: list[str], sensitive: bool = False) -> str:
  try:
    return subprocess.check_output(["sudo", "-n", "nmcli", *args], stderr=subprocess.STDOUT, encoding="utf-8")
  except subprocess.CalledProcessError as e:
    safe_args = args.copy()
    secrets = []
    if sensitive:
      for index in range(1, len(safe_args)):
        if safe_args[index - 1] == "password":
          secrets.append(safe_args[index])
          safe_args[index] = "<redacted>"
    output = e.output if not sensitive or secrets else ""
    for secret in secrets:
      output = output.replace(secret, "<redacted>")
    raise Exception(f"nmcli failed: {' '.join(safe_args)} {output}".strip()) from e


def _nmcli_fields(line: str) -> list[str]:
  return line.rstrip("\n").split(":")


def _tethering_ssid() -> str:
  dongle_id = Params().get("DongleId")
  return "weedle" + (f"-{dongle_id[:4]}" if dongle_id else "")


def _saved_wifi_connections() -> set[str]:
  saved: set[str] = set()
  output = _nmcli(["-t", "--escape", "no", "-f", "NAME,TYPE", "connection", "show"])
  for line in output.splitlines():
    fields = _nmcli_fields(line)
    if len(fields) >= 2 and fields[1] == "802-11-wireless":
      saved.add(fields[0].removeprefix("openpilot connection "))
  return saved


def _wifi_connection_name(ssid: str) -> str:
  output = _nmcli(["-t", "--escape", "no", "-f", "NAME,TYPE", "connection", "show"])
  for line in output.splitlines():
    fields = _nmcli_fields(line)
    if len(fields) >= 2 and fields[1] == "802-11-wireless":
      name = fields[0]
      if name == ssid or name.removeprefix("openpilot connection ") == ssid:
        return name
  return f"openpilot connection {ssid}"


def _active_wifi_connection() -> str:
  output = _nmcli(["-t", "--escape", "no", "-f", "NAME,TYPE,DEVICE", "connection", "show", "--active"])
  for line in output.splitlines():
    fields = _nmcli_fields(line)
    if len(fields) >= 3 and fields[1] == "802-11-wireless":
      return fields[0]
  return ""


def _nmcli_wifi_networks() -> list[dict[str, str | int | bool]]:
  saved = _saved_wifi_connections()
  networks_by_ssid: dict[str, dict[str, str | int | bool]] = {}
  output = _nmcli(["-t", "--escape", "no", "-f", "ACTIVE,SSID,SIGNAL,SECURITY", "device", "wifi", "list"])
  for line in output.splitlines():
    fields = _nmcli_fields(line)
    if len(fields) < 4:
      continue
    active, ssid, signal, security = fields[0], fields[1], fields[2], ":".join(fields[3:])
    if not ssid:
      continue

    network = {
      "ssid": ssid,
      "strength": int(signal or 0),
      "securityType": 0 if not security else 1,
      "saved": ssid in saved,
      "tethering": ssid == _tethering_ssid(),
      "connected": active == "yes",
      "connecting": False,
    }
    existing = networks_by_ssid.get(ssid)
    if existing is None or int(network["strength"]) > int(existing["strength"]):
      networks_by_ssid[ssid] = network
  return sorted(networks_by_ssid.values(), key=lambda n: (not bool(n["connected"]), not bool(n["saved"]), -int(n["strength"]), str(n["ssid"]).lower()))


def _nmcli_wifi_state() -> dict[str, str | int | None]:
  output = _nmcli(["-t", "--escape", "no", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device", "status"])
  for line in output.splitlines():
    fields = _nmcli_fields(line)
    if len(fields) >= 4 and fields[1] == "wifi":
      connection = fields[3].removeprefix("openpilot connection ") or None
      status = 2 if fields[2].startswith("connected") else 1 if fields[2] == "connecting" else 0
      return {"ssid": connection, "status": status}
  return {"ssid": None, "status": 0}


def _connection_metered() -> int:
  connection = _active_wifi_connection()
  if not connection:
    return 0
  value = _nmcli(["-t", "--escape", "no", "-g", "connection.metered", "connection", "show", connection]).strip().lower()
  return {"yes": 1, "no": 2}.get(value, 0)


def _tethering_password() -> str:
  try:
    return _nmcli(["-s", "-g", "802-11-wireless-security.psk", "connection", "show", "Hotspot"], sensitive=True).strip()
  except Exception:
    return "swagswagcomma"


def _read_vamos_update_state() -> dict[str, Any] | None:
  try:
    value = json.loads(VAMOS_UPDATE_STATE_FILE.read_text(encoding="utf-8"))
    return _json_safe(value) if isinstance(value, dict) else None
  except (FileNotFoundError, json.JSONDecodeError, OSError):
    return None


def _software_update_state(params: ParamsReader | None = None) -> dict[str, Any]:
  params = params or Params()
  keys = [
    "UpdaterCurrentDescription",
    "UpdaterCurrentReleaseNotes",
    "UpdaterNewDescription",
    "UpdaterNewReleaseNotes",
    "UpdaterState",
    "UpdaterProgress",
    "UpdaterTargetBranch",
    "UpdaterAvailableBranches",
    "UpdaterLastFetchTime",
    "LastUpdateTime",
    "LastUpdateException",
    "UpdateAvailable",
    "UpdaterFetchAvailable",
    "UpdateFailedCount",
  ]
  state: dict[str, Any] = {}
  for key in keys:
    value = params.get(key)
    if isinstance(value, bytes):
      value = value.decode("utf-8", "replace")
    state[key] = _json_safe(value)
  vamos_state = _read_vamos_update_state()
  if vamos_state is not None:
    state["VamosUpdate"] = vamos_state
  return state


def _signal_updated(signal_name: str) -> dict[str, int | str]:
  result = subprocess.run(["pkill", f"-{signal_name}", "-f", "openpilot.system.updated.updated"], check=False)
  return {"success": 1 if result.returncode in (0, 1) else 0, "returncode": result.returncode}


@dispatcher.add_method
def getNetworkState() -> dict:
  params = Params()
  local_ips = _local_ips()
  wifi_ip = next((ip["address"] for ip in local_ips if ip["interface"] == "wlan0"), "")
  wifi_state = _nmcli_wifi_state()
  return {
    "networkType": int(HARDWARE.get_network_type()),
    "networkMetered": HARDWARE.get_network_metered(HARDWARE.get_network_type()),
    "wifi": wifi_state,
    "networks": _nmcli_wifi_networks(),
    "localIp": wifi_ip,
    "localIps": local_ips,
    "currentNetworkMetered": _connection_metered(),
    "tetheringActive": wifi_state["ssid"] == "Hotspot" or wifi_state["ssid"] == _tethering_ssid(),
    "tetheringSsid": _tethering_ssid(),
    "tetheringPassword": _tethering_password(),
    "gsmRoaming": params.get_bool("GsmRoaming"),
    "gsmMetered": params.get_bool("GsmMetered"),
    "gsmApn": params.get("GsmApn") or "",
  }


@dispatcher.add_method
def refreshNetworks() -> dict:
  _nmcli(["device", "wifi", "rescan", "ifname", "wlan0"])
  return getNetworkState()


@dispatcher.add_method
def connectNetwork(ssid: str, password: str = "", hidden: bool = False) -> dict[str, int]:
  if ssid in _saved_wifi_connections() and not password and not hidden:
    _nmcli(["connection", "up", _wifi_connection_name(ssid)])
  else:
    args = ["device", "wifi", "connect", ssid, "ifname", "wlan0"]
    if password:
      args += ["password", password]
    if hidden:
      args += ["hidden", "yes"]
    _nmcli(args, sensitive=bool(password))
  return {"success": 1}


@dispatcher.add_method
def forgetNetwork(ssid: str) -> dict[str, int]:
  for name in (f"openpilot connection {ssid}", ssid):
    subprocess.run(["sudo", "-n", "nmcli", "connection", "delete", name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  return {"success": 1}


@dispatcher.add_method
def setTethering(enabled: bool) -> dict[str, int]:
  if enabled:
    _nmcli(["device", "wifi", "hotspot", "ifname", "wlan0", "ssid", _tethering_ssid(), "password", _tethering_password()], sensitive=True)
  else:
    subprocess.run(["sudo", "-n", "nmcli", "connection", "down", "Hotspot"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  return {"success": 1}


@dispatcher.add_method
def setTetheringPassword(password: str) -> dict[str, int | str]:
  if len(password) < 8:
    return {"success": 0, "error": "password must be at least 8 characters"}
  _nmcli(["connection", "modify", "Hotspot", "802-11-wireless-security.psk", password], sensitive=True)
  return {"success": 1}


@dispatcher.add_method
def setCurrentNetworkMetered(metered: int | str) -> dict[str, int]:
  connection = _active_wifi_connection()
  if not connection:
    return {"success": 0}
  value = {"default": "unknown", "metered": "yes", "unmetered": "no", 0: "unknown", 1: "yes", 2: "no"}.get(metered, "unknown")
  _nmcli(["connection", "modify", connection, "connection.metered", value])
  return {"success": 1}


@dispatcher.add_method
def getSoftwareUpdateState() -> dict[str, Any]:
  return _software_update_state()


@dispatcher.add_method
def checkSoftwareUpdate() -> dict[str, int | str]:
  return _signal_updated("SIGUSR1")


@dispatcher.add_method
def downloadSoftwareUpdate() -> dict[str, int | str]:
  return _signal_updated("SIGHUP")


@dispatcher.add_method
def setUpdateBranch(branch: str) -> dict[str, int | str]:
  branch = branch.strip()
  if not branch:
    return {"success": 0, "error": "branch is required"}
  Params().put("UpdaterTargetBranch", branch, block=True)
  return checkSoftwareUpdate()


@dispatcher.add_method
def installSoftwareUpdate() -> dict[str, int]:
  Params().put_bool("DoReboot", True, block=True)
  return {"success": 1}


@dispatcher.add_method
def startStream(sdp: str, enabled: bool = True) -> dict:
  from openpilot.system.athena.athenad import startStream as upstream_start_stream
  return upstream_start_stream(sdp, enabled)


def _json_safe(value: Any) -> Any:
  if isinstance(value, bytes):
    return base64.b64encode(value).decode("utf-8")
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  if isinstance(value, dict):
    return {str(k): _json_safe(v) for k, v in value.items()}
  if isinstance(value, list):
    return [_json_safe(v) for v in value]
  return str(value)


def _live_state_snapshot(sm: messaging.SubMaster, params: Params) -> dict[str, Any]:
  build_metadata = get_build_metadata()
  services: dict[str, Any] = {}
  for service in LIVE_STATE_SERVICES:
    try:
      if sm.recv_frame[service] > 0:
        services[service] = sm[service].to_dict()
    except Exception:
      cloudlog.exception("athena.live_state.service_failed service=%s", service)

  param_values = {}
  for key in LIVE_STATE_PARAM_KEYS:
    try:
      param_values[key] = _json_safe(params.get(key, return_default=True))
    except Exception:
      cloudlog.exception("athena.live_state.param_failed key=%s", key)

  return {
    "ts": time.time(),  # noqa: TID251
    "dongleId": params.get("DongleId"),
    "serial": params.get("HardwareSerial"),
    "version": {
      "version": build_metadata.openpilot.version,
      "remote": build_metadata.openpilot.git_normalized_origin,
      "branch": build_metadata.channel,
      "commit": build_metadata.openpilot.git_commit,
    },
    "params": param_values,
    "software": _software_update_state(params),
    "services": services,
    "authorizedPeers": list(load_authorized_peers().keys()),
  }


def send_peer_payload(to: str, body: dict) -> None:
  dongle_id = Params().get("DongleId")
  peer = load_authorized_peers().get(to)
  if dongle_id is None or peer is None:
    raise Exception("unknown Athena peer")
  recipient = peer.get("publicKey")
  if not isinstance(recipient, str):
    raise Exception("invalid Athena peer")

  send_queue.put_nowait(pack_peer_message(dongle_id, recipient, body))


terminal_manager = TerminalManager(send_peer_payload)


def broadcast_peer_event(name: str, payload: Any) -> None:
  for public_key in list(load_authorized_peers().keys()):
    try:
      send_peer_payload(public_key, {"type": "event", "name": name, "payload": payload})
    except Exception:
      cloudlog.exception("athena.websocket.broadcast_failed public_key=%s", public_key)


def live_state_handler(end_event: threading.Event) -> None:
  params = Params()
  sm = messaging.SubMaster(LIVE_STATE_SERVICES)

  while not end_event.is_set():
    try:
      sm.update(0)
      if load_authorized_peers():
        broadcast_peer_event("liveState", _live_state_snapshot(sm, params))
    except Exception:
      cloudlog.exception("athena.live_state_handler.exception")
    end_event.wait(LIVE_STATE_INTERVAL_S)


def handle_athena_call(sender: str, body: dict) -> None:
  request_id = body.get("id")
  request = {
    "jsonrpc": "2.0",
    "id": request_id,
    "method": body.get("method"),
  }
  if "params" in body:
    request["params"] = body["params"]

  response = json.loads(handle(request, dispatcher))
  send_peer_payload(sender, {
    "type": "athena-response",
    "id": request_id,
    "result": response.get("result"),
    "error": response.get("error"),
  })


def handle_peer_message(data: str) -> bool:
  try:
    dongle_id = Params().get("DongleId")
    if dongle_id is None:
      return True

    peer_message = unpack_peer_message(data, dongle_id)
    if peer_message is None:
      return False

    sender, body, decrypt_failed = peer_message
    if body is None:
      if decrypt_failed:
        cloudlog.event("athena.websocket.decrypt_failed", sender=sender, error=True)
      return True

    if body.get("type") == "pair-request":
      if body.get("publicKey") != sender:
        raise Exception("pair request sender mismatch")
      if not pairing_mode_active():
        raise Exception("pairing mode is not active")
      if not verify_pair_token(body.get("pairToken"), dongle_id):
        raise Exception("invalid pair token")
      authorize_peer(body["publicKey"], label=body.get("label") if isinstance(body.get("label"), str) else None)
      cloudlog.event("athena.websocket.paired", sender=sender)
      send_peer_payload(sender, {"type": "pair-response", "publicKey": dongle_id, "device-type": HARDWARE.get_device_type()})
      return True

    if sender not in load_authorized_peers():
      cloudlog.event("athena.websocket.unauthorized", sender=sender, error=True)
      return True

    if body.get("type") == "athena-call":
      handle_athena_call(sender, body)
    elif body.get("type") == "event":
      if body.get("name") == "terminal":
        terminal_manager.handle(sender, body.get("payload"))
      else:
        cloudlog.event("athena.websocket.event", sender=sender, name=body.get("name"), payload=body.get("payload"))
    elif body.get("method"):
      body["type"] = "athena-call"
      handle_athena_call(sender, body)
    return True
  except Exception:
    cloudlog.exception("athena.websocket.handle_peer_message_failed")
    return True


def ws_recv(ws: WebSocket, end_event: threading.Event) -> None:
  last_ping = int(time.monotonic() * 1e9)
  while not end_event.is_set():
    try:
      opcode, data = ws.recv_data(control_frame=True)
      if opcode in (ABNF.OPCODE_TEXT, ABNF.OPCODE_BINARY):
        if opcode == ABNF.OPCODE_TEXT:
          data = data.decode("utf-8")
        if isinstance(data, str):
          handle_peer_message(data)
      elif opcode == ABNF.OPCODE_PING:
        last_ping = int(time.monotonic() * 1e9)
        Params().put("LastAthenaPingTime", last_ping, block=True)
    except WebSocketTimeoutException:
      ns_since_last_ping = int(time.monotonic() * 1e9) - last_ping
      if ns_since_last_ping > RECONNECT_TIMEOUT_S * 1e9:
        cloudlog.exception("athenad.ws_recv.timeout")
        end_event.set()
    except Exception:
      cloudlog.exception("athenad.ws_recv.exception")
      end_event.set()


def ws_send(ws: WebSocket, end_event: threading.Event) -> None:
  while not end_event.is_set():
    try:
      data = send_queue.get(timeout=1)
      try:
        if json.loads(data).get("type") != "peer":
          continue
      except Exception:
        continue
      for i in range(0, len(data), WS_FRAME_SIZE):
        frame = data[i:i+WS_FRAME_SIZE]
        last = i + WS_FRAME_SIZE >= len(data)
        opcode = ABNF.OPCODE_TEXT if i == 0 else ABNF.OPCODE_CONT
        ws.send_frame(ABNF.create_frame(frame, opcode, last))
    except queue.Empty:
      pass
    except Exception:
      cloudlog.exception("athenad.ws_send.exception")
      end_event.set()


def ws_manage(ws: WebSocket, end_event: threading.Event) -> None:
  params = Params()
  onroad_prev = None
  sock = ws.sock

  while True:
    onroad = not params.get_bool("IsOffroad")
    if onroad != onroad_prev:
      onroad_prev = onroad

      if sock is not None:
        # While not sending data, onroad, we can expect to time out in 7 + (7 * 2) = 21s
        #                         offroad, we can expect to time out in 30 + (10 * 3) = 60s
        # FIXME: TCP_USER_TIMEOUT is effectively 2x for some reason (32s), so it's mostly unused
        if sys.platform == 'linux':
          sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT, 16000 if onroad else 0)
          sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 7 if onroad else 30)
        elif sys.platform == 'darwin':
          sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, 7 if onroad else 30)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 7 if onroad else 10)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 2 if onroad else 3)

    if end_event.wait(5):
      break
