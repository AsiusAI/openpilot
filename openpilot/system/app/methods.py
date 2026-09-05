#!/usr/bin/env python3
"""RPC methods shared by the Asius app transports."""

from __future__ import annotations

import base64
import json
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Protocol, cast

import requests
from websocket import ABNF, WebSocket, WebSocketTimeoutException

import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from openpilot.system.app.identity import get_device_public_key
from openpilot.common.params import Params
from openpilot.common.hardware import HARDWARE
from openpilot.common.swaglog import cloudlog
from openpilot.common.version import get_build_metadata
from openpilot.system.athena import athenad as upstream_athena
from openpilot.system.athena.rpc import Dispatcher, handle
from openpilot.system.app.device_name import get_device_name, set_device_name
from openpilot.system.app.terminal import TerminalManager
from openpilot.system.app.websocketd import (
  authorize_peer,
  load_authorized_peers,
  pack_peer_message,
  pairing_url,
  save_authorized_peers,
  unpack_peer_message,
  verify_pair_token,
)


class ParamsReader(Protocol):
  def get(self, key: str) -> Any: ...


RELAY_HOST = Params().get("WebsocketHost", return_default=True)

RECONNECT_TIMEOUT_S = 70
WS_FRAME_SIZE = 4096
LIVE_STATE_INTERVAL_S = 1.0
VAMOS_UPDATE_STATE_FILE = Path("/data/vamos-update/state.json")
VAMOS_WIFI_COMMAND = Path("/usr/bin/vamos-wifi")
SAVE_PARAMS_BLOCKED_KEYS = {
  "AccessToken",
  "ApiCache_Device",
  "AppAuthorizedKeys",
  "AppPairingUntil",
  "AthenadUploadQueue",
  "DoUninstall",
  "DongleId",
  "DataUploadState",
  "GithubSshKeys",
  "GithubUsername",
  "HardwareSerial",
  "LastAthenaPingTime",
  "SecOCKey",
  "AthenadPid",
  "BluetoothdPid",
  "WebsocketdPid",
}
LIVE_STATE_SERVICES = [
  "deviceState",
  "peripheralState",
  "liveCalibration",
  "managerState",
  "onroadEvents",
  "selfdriveState",
]
LIVE_STATE_PARAM_KEYS = [
  "DongleId",
  "DataApiHost",
  "DataUploadEnabled",
  "ShareDrivingData",
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

NetworkType = log.DeviceState.NetworkType

dispatcher = Dispatcher()
dispatcher["echo"] = lambda s: s
for method in (
  upstream_athena.getMessage,
  upstream_athena.getVersion,
  upstream_athena.listDataDirectory,
  upstream_athena.uploadFileToUrl,
  upstream_athena.uploadFilesToUrls,
  upstream_athena.listUploadQueue,
  upstream_athena.cancelUpload,
  upstream_athena.setRouteViewed,
  upstream_athena.getSshAuthorizedKeys,
  upstream_athena.getGithubUsername,
  upstream_athena.getSimInfo,
  upstream_athena.getNetworkType,
  upstream_athena.getNetworkMetered,
):
  dispatcher.add_method(method)

send_queue: Queue[str] = queue.Queue()
upload_queue = upstream_athena.upload_queue
cur_upload_items = upstream_athena.cur_upload_items
UploadQueueCache = upstream_athena.UploadQueueCache


def handle_long_poll(ws: WebSocket, exit_event: threading.Event | None) -> None:
  end_event = threading.Event()

  threads = [
    threading.Thread(target=upstream_athena.ws_manage, args=(ws, end_event), name='ws_manage'),
    threading.Thread(target=ws_recv, args=(ws, end_event), name='ws_recv'),
    threading.Thread(target=ws_send, args=(ws, end_event), name='ws_send'),
    threading.Thread(target=upstream_athena.upload_handler, args=(end_event,), name='upload_handler'),
    threading.Thread(target=upstream_athena.upload_handler, args=(end_event,), name='upload_handler2'),
    threading.Thread(target=upstream_athena.upload_handler, args=(end_event,), name='upload_handler3'),
    threading.Thread(target=upstream_athena.upload_handler, args=(end_event,), name='upload_handler4'),
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
def getPublicKey() -> str | None:
  return get_device_public_key()


@dispatcher.add_method
def getDeviceName() -> str:
  return get_device_name()


@dispatcher.add_method
def setDeviceName(name: str) -> dict[str, str]:
  return {"name": set_device_name(name)}


@dispatcher.add_method
def getAuthorizedPeers() -> dict[str, Any]:
  return {
    "peers": [
      {
        "publicKey": public_key,
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

  return {"removed": removed}


@dispatcher.add_method
def getPairingUrl() -> str:
  params = Params()
  dongle_id = params.get("DongleId") or ""
  return pairing_url(dongle_id)


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


def _nmcli(args: list[str], sensitive: bool = False, input_text: str | None = None) -> str:
  try:
    return subprocess.check_output(
      ["sudo", "-n", "nmcli", *args],
      input=input_text,
      stderr=subprocess.STDOUT,
      encoding="utf-8",
    )
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
  fields = []
  current = []
  escaped = False
  for char in line.rstrip("\n"):
    if escaped:
      current.append(char)
      escaped = False
    elif char == "\\":
      escaped = True
    elif char == ":":
      fields.append("".join(current))
      current = []
    else:
      current.append(char)
  if escaped:
    current.append("\\")
  fields.append("".join(current))
  return fields


def _tethering_ssid() -> str:
  dongle_id = Params().get("DongleId")
  return "weedle" + (f"-{dongle_id[:4]}" if dongle_id else "")


def _wifi_connections(saved_only: bool = False) -> dict[str, str]:
  """Return SSID -> NetworkManager connection name.

  NetworkManager persists a profile before it has successfully associated. A
  zero timestamp identifies those never-connected profiles; presenting them as
  saved makes the app retry bad credentials without asking for a password.
  """
  connections: dict[str, str] = {}
  output = _nmcli(["-t", "--escape", "yes", "-f", "NAME,TYPE,TIMESTAMP", "connection", "show"])
  for line in output.splitlines():
    fields = _nmcli_fields(line)
    if len(fields) >= 3 and fields[1] == "802-11-wireless" and (not saved_only or int(fields[2] or 0) > 0):
      name = fields[0]
      connections[name.removeprefix("openpilot connection ")] = name
  return connections


def _saved_wifi_connections() -> set[str]:
  return set(_wifi_connections(saved_only=True))


def _wifi_connection_name(ssid: str) -> str:
  return _wifi_connections().get(ssid, f"openpilot connection {ssid}")


def _delete_wifi_connections(ssid: str) -> None:
  names = {name for profile_ssid, name in _wifi_connections().items() if profile_ssid == ssid}
  for name in names:
    subprocess.run(["sudo", "-n", "nmcli", "connection", "delete", name], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _vamos_wifi(args: list[str], password: str = "") -> bool:
  if not VAMOS_WIFI_COMMAND.is_file():
    return False
  try:
    subprocess.check_output(
      ["sudo", "-n", str(VAMOS_WIFI_COMMAND), *args],
      input=f"{password}\n",
      stderr=subprocess.STDOUT,
      encoding="utf-8",
    )
  except subprocess.CalledProcessError as e:
    raise Exception(e.output.strip() or f"vamos-wifi {' '.join(args)} failed") from e
  return True


def _activate_new_wifi_connection(ssid: str, password: str, hidden: bool) -> None:
  if _vamos_wifi(["connect", ssid, *(["--hidden"] if hidden else [])], password):
    return

  name = f"openpilot connection {ssid}"
  _delete_wifi_connections(ssid)

  password_file = None
  try:
    if password:
      if "\n" in password or "\r" in password:
        raise Exception("Wi-Fi passwords cannot contain line breaks")
      password_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix="nmcli-passwd-", dir="/tmp", delete=False)
      password_file.write(f"802-11-wireless-security.psk:{password}\n")
      password_file.close()

    # Give NetworkManager only the user intent. It completes the connection
    # from the selected AP's capabilities, including its security mode.
    args = [*(["--ask"] if password_file is not None else []), "--wait", "45"]
    args += ["device", "wifi", "connect", ssid, "ifname", "wlan0", "name", name]
    if hidden:
      args += ["hidden", "yes"]
    _nmcli(args, sensitive=password_file is not None, input_text=f"{password}\n" if password_file is not None else None)
  except Exception as e:
    _delete_wifi_connections(ssid)
    raise Exception(f"Could not connect to {ssid}. Check the password and hotspot settings.") from e
  finally:
    if password_file is not None:
      Path(password_file.name).unlink(missing_ok=True)


_network_operation_lock = threading.Lock()
_network_operation: dict[str, str | float] = {"action": "", "ssid": "", "state": "idle", "error": "", "updatedAt": 0.0}


def _network_operation_snapshot() -> dict[str, str | float]:
  with _network_operation_lock:
    return _network_operation.copy()


def _start_network_operation(action: str, ssid: str, callback: Callable[[], None]) -> dict[str, int | str]:
  global _network_operation
  with _network_operation_lock:
    if _network_operation["state"] == "running":
      return {"success": 0, "error": f"network {_network_operation['action']} already in progress"}
    _network_operation = {"action": action, "ssid": ssid, "state": "running", "error": "", "updatedAt": time.time()}  # noqa: TID251

  def worker() -> None:
    global _network_operation
    # Allow the RPC response to leave over the current Wi-Fi connection before
    # NetworkManager tears it down. Bluetooth callers benefit from the same
    # asynchronous, observable operation state.
    time.sleep(0.5)
    try:
      callback()
      state, error = "succeeded", ""
    except Exception as e:
      cloudlog.exception("app.network_operation_failed action=%s ssid=%s", action, ssid)
      state, error = "failed", str(e)
    with _network_operation_lock:
      _network_operation = {"action": action, "ssid": ssid, "state": state, "error": error, "updatedAt": time.time()}  # noqa: TID251

  threading.Thread(target=worker, daemon=True, name=f"network-{action}").start()
  return {"success": 1}


def _active_wifi_connection() -> str:
  output = _nmcli(["-t", "--escape", "yes", "-f", "NAME,TYPE,DEVICE", "connection", "show", "--active"])
  for line in output.splitlines():
    fields = _nmcli_fields(line)
    if len(fields) >= 3 and fields[1] == "802-11-wireless":
      return fields[0]
  return ""


def _active_ethernet_interface() -> str:
  output = _nmcli(["-t", "--escape", "yes", "-f", "DEVICE,TYPE,STATE", "device", "status"])
  for line in output.splitlines():
    fields = _nmcli_fields(line)
    if len(fields) >= 3 and fields[1] == "ethernet" and fields[2].startswith("connected"):
      return fields[0]
  return ""


def _nmcli_wifi_networks() -> list[dict[str, str | int | bool]]:
  saved = _saved_wifi_connections()
  operation = _network_operation_snapshot()
  networks_by_ssid: dict[str, dict[str, str | int | bool]] = {}
  output = _nmcli(["-t", "--escape", "yes", "-f", "ACTIVE,SSID,SIGNAL,SECURITY", "device", "wifi", "list"])
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
      "connecting": operation["state"] == "running" and operation["action"] == "connect" and operation["ssid"] == ssid,
      "passwordRequired": bool(security) and (ssid not in saved or (operation["state"] == "failed" and operation["ssid"] == ssid)),
    }
    existing = networks_by_ssid.get(ssid)
    if existing is None or int(network["strength"]) > int(existing["strength"]):
      networks_by_ssid[ssid] = network
  return sorted(networks_by_ssid.values(), key=lambda n: (not bool(n["connected"]), not bool(n["saved"]), -int(n["strength"]), str(n["ssid"]).lower()))


def _nmcli_wifi_state() -> dict[str, str | int | None]:
  output = _nmcli(["-t", "--escape", "yes", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device", "status"])
  for line in output.splitlines():
    fields = _nmcli_fields(line)
    if len(fields) >= 4 and fields[1] == "wifi":
      status = 2 if fields[2].startswith("connected") else 1 if fields[2] == "connecting" else 0
      connection = (fields[3].removeprefix("openpilot connection ") or None) if status else None
      if isinstance(connection, str) and connection.lower() in ("--", "none"):
        connection = None
      return {"ssid": connection, "status": status}
  return {"ssid": None, "status": 0}


def _connection_metered() -> int:
  connection = _active_wifi_connection()
  if not connection:
    return 0
  value = _nmcli(["-t", "--escape", "yes", "-g", "connection.metered", "connection", "show", connection]).strip().lower()
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
  local_ips = [ip for ip in _local_ips() if ip["interface"] != "lo"]
  wifi_ip = next((ip["address"] for ip in local_ips if ip["interface"] == "wlan0"), "")
  wifi_state = _nmcli_wifi_state()
  ethernet_interface = _active_ethernet_interface()
  network_type = int(HARDWARE.get_network_type())
  if wifi_state["status"] != 2 and ethernet_interface:
    network_type = 6
  local_ip = wifi_ip or next((ip["address"] for ip in local_ips if ip["interface"] == ethernet_interface), "")
  return {
    "networkType": network_type,
    "networkMetered": HARDWARE.get_network_metered(HARDWARE.get_network_type()),
    "wifi": wifi_state,
    "networks": _nmcli_wifi_networks(),
    "operation": _network_operation_snapshot(),
    "localIp": local_ip,
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
  if not _vamos_wifi(["scan"]):
    _nmcli(["device", "wifi", "rescan", "ifname", "wlan0"])
  return getNetworkState()


@dispatcher.add_method
def connectNetwork(ssid: str, password: str = "", hidden: bool = False) -> dict[str, int | str]:
  if not ssid:
    return {"success": 0, "error": "SSID is required"}

  def connect() -> None:
    if ssid in _saved_wifi_connections() and not password and not hidden:
      _nmcli(["--wait", "45", "connection", "up", _wifi_connection_name(ssid)])
      return

    _activate_new_wifi_connection(ssid, password, hidden)

  return _start_network_operation("connect", ssid, connect)


@dispatcher.add_method
def disconnectNetwork(ssid: str = "") -> dict[str, int | str]:
  active_ssid = _nmcli_wifi_state()["ssid"] or ""
  if ssid and active_ssid != ssid:
    return {"success": 0, "error": f"{ssid} is not connected"}
  if not active_ssid:
    return {"success": 1}

  def disconnect() -> None:
    if not _vamos_wifi(["disconnect"]):
      _nmcli(["device", "disconnect", "wlan0"])

  return _start_network_operation("disconnect", active_ssid, disconnect)


@dispatcher.add_method
def forgetNetwork(ssid: str) -> dict[str, int | str]:
  if not ssid:
    return {"success": 0, "error": "SSID is required"}

  def forget() -> None:
    if not _vamos_wifi(["forget", ssid]):
      _delete_wifi_connections(ssid)

  return _start_network_operation("forget", ssid, forget)


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
        data = sm[service]
        services[service] = ([event.to_dict() for event in data]
                             if service == "onroadEvents" else data.to_dict())
    except Exception:
      cloudlog.exception("athena.live_state.service_failed service=%s", service)

  if "deviceState" in services:
    services["deviceState"]["uptime"] = int(time.monotonic())

  param_values = {}
  for key in LIVE_STATE_PARAM_KEYS:
    try:
      param_values[key] = _json_safe(params.get(key, return_default=True))
    except Exception:
      cloudlog.exception("athena.live_state.param_failed key=%s", key)

  return {
    "ts": time.time(),  # noqa: TID251
    "dongleId": params.get("DongleId"),
    "deviceName": get_device_name(),
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


def handle_rpc(sender: str, body: dict) -> None:
  send_peer_payload(sender, json.loads(handle(body, dispatcher)))


def handle_peer_message(data: str) -> bool:
  try:
    dongle_id = Params().get("DongleId")
    if dongle_id is None:
      return True

    # The relay authenticates each live connection and enforces increasing sequence
    # numbers. E2E signatures are still verified; the device wall clock is irrelevant.
    peer_message = unpack_peer_message(data, dongle_id, validate_timestamp=False)
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
      if not verify_pair_token(body.get("pairToken"), dongle_id):
        raise Exception("invalid pair token")
      authorize_peer(body["publicKey"], label=body.get("label") if isinstance(body.get("label"), str) else None)
      cloudlog.event("athena.websocket.paired", sender=sender)
      send_peer_payload(sender, {
        "type": "pair-response",
        "publicKey": dongle_id,
        "device-type": HARDWARE.get_device_type(),
        "name": get_device_name(),
      })
      return True

    if sender not in load_authorized_peers():
      cloudlog.event("athena.websocket.unauthorized", sender=sender, error=True)
      return True

    if body.get("type") == "event":
      if body.get("name") == "terminal":
        terminal_manager.handle(sender, body.get("payload"))
      else:
        cloudlog.event("athena.websocket.event", sender=sender, name=body.get("name"), payload=body.get("payload"))
    elif body.get("method"):
      handle_rpc(sender, body)
    return True
  except Exception:
    cloudlog.exception("athena.websocket.handle_peer_message_failed")
    return True


def ws_recv(ws: WebSocket, end_event: threading.Event) -> None:
  last_ping = int(time.monotonic() * 1e9)
  last_ping_sent = 0.0
  received_sequences: dict[tuple[str, str], int] = {}
  while not end_event.is_set():
    try:
      # Cloudflare's hibernating relay answers control pings without an object timer.
      now = time.monotonic()
      if now - last_ping_sent >= 25:
        ws.ping()
        last_ping_sent = now
      opcode, data = ws.recv_data(control_frame=True)
      if opcode in (ABNF.OPCODE_TEXT, ABNF.OPCODE_BINARY):
        if opcode == ABNF.OPCODE_TEXT:
          data = data.decode("utf-8")
        if isinstance(data, str):
          message = json.loads(data)
          if message.get("type") == "peer":
            session, sequence = message.get("relaySession"), message.get("sequence")
            key = (message.get("from"), session)
            if not isinstance(session, str) or type(sequence) is not int or sequence <= received_sequences.get(key, 0):
              continue
            if len(received_sequences) >= 1024 and key not in received_sequences:
              received_sequences.pop(next(iter(received_sequences)))
            received_sequences[key] = sequence
          handle_peer_message(data)
      elif opcode in (ABNF.OPCODE_PING, ABNF.OPCODE_PONG):
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
  sequence = 0
  while not end_event.is_set():
    try:
      data = send_queue.get(timeout=1)
      try:
        message = json.loads(data)
        if message.get("type") != "peer":
          continue
        sequence += 1
        message["sequence"] = sequence
        data = json.dumps(message)
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
