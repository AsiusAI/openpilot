from __future__ import annotations

from itertools import combinations, permutations
import json
import secrets
from typing import Any

from openpilot.system.athena.identity import is_dongle_id
from openpilot.system.athena.websocketd import (
  authorize_peer,
  disable_pairing_mode,
  pairing_mode_active,
  read_raw_param,
  remove_raw_param,
  wall_time,
  write_raw_param,
)


ATHENA_BLE_PAIRING_PARAM = "AthenadBlePairing"
PAIRING_CHALLENGE_SECONDS = 60
PAIRING_APPROVED_SECONDS = 10

PAIRING_COLORS = {
  "red": (255, 0, 0),
  "green": (0, 255, 0),
  "blue": (0, 0, 255),
  "amber": (255, 80, 0),
  "turquoise": (0, 255, 80),
  "violet": (150, 0, 255),
}
MIN_PAIRING_COLOR_DISTANCE_SQUARED = 80_000


def pairing_color_distance_squared(left: str, right: str) -> int:
  return sum((a - b) ** 2 for a, b in zip(PAIRING_COLORS[left], PAIRING_COLORS[right], strict=True))


PAIRING_COLOR_CODES = tuple(
  code for code in permutations(PAIRING_COLORS)
  if all(
    pairing_color_distance_squared(left, right) >= MIN_PAIRING_COLOR_DISTANCE_SQUARED
    for camera_colors in (code[:3], code[3:])
    for left, right in combinations(camera_colors, 2)
  )
)


def clear_ble_pairing() -> None:
  remove_raw_param(ATHENA_BLE_PAIRING_PARAM)


def get_ble_pairing(now: float | None = None) -> dict[str, Any] | None:
  now = wall_time() if now is None else now
  try:
    state = json.loads(read_raw_param(ATHENA_BLE_PAIRING_PARAM) or "null")
    if not isinstance(state, dict) or float(state.get("expiresAt", 0)) < now:
      clear_ble_pairing()
      return None
    if state.get("status") not in ("pending", "approved"):
      clear_ble_pairing()
      return None
    return state
  except Exception:
    clear_ble_pairing()
    return None


def create_ble_pairing(public_key: str, request_id: str, label: str | None = None,
                       now: float | None = None) -> dict[str, Any]:
  now = wall_time() if now is None else now
  if not pairing_mode_active():
    raise PermissionError("pairing mode is not active")
  if not is_dongle_id(public_key):
    raise ValueError("invalid app public key")
  if not request_id or len(request_id) > 64:
    raise ValueError("invalid pairing request ID")
  if label is not None and len(label) > 80:
    raise ValueError("pairing label is too long")

  current = get_ble_pairing(now)
  if current is not None:
    if current.get("publicKey") == public_key and current.get("requestId") == request_id:
      return current
    raise RuntimeError("another app is awaiting physical approval")

  random = secrets.SystemRandom()
  state: dict[str, Any] = {
    "status": "pending",
    "publicKey": public_key,
    "requestId": request_id,
    "colors": list(random.choice(PAIRING_COLOR_CODES)),
    "createdAt": int(now),
    "expiresAt": int(now) + PAIRING_CHALLENGE_SECONDS,
  }
  if label:
    state["label"] = label
  write_raw_param(ATHENA_BLE_PAIRING_PARAM, json.dumps(state))
  return state


def approve_ble_pairing(now: float | None = None) -> dict[str, Any] | None:
  now = wall_time() if now is None else now
  state = get_ble_pairing(now)
  if state is None or state.get("status") != "pending":
    return None

  peer = authorize_peer(state["publicKey"], label=state.get("label"))
  approved = {
    **state,
    "status": "approved",
    "aclEpoch": int(peer["aclEpoch"]),
    "approvedAt": int(now),
    "expiresAt": int(now) + PAIRING_APPROVED_SECONDS,
  }
  write_raw_param(ATHENA_BLE_PAIRING_PARAM, json.dumps(approved))
  disable_pairing_mode()
  return approved


def consume_approved_ble_pairing(now: float | None = None) -> dict[str, Any] | None:
  state = get_ble_pairing(now)
  if state is None or state.get("status") != "approved":
    return None
  clear_ble_pairing()
  return state
