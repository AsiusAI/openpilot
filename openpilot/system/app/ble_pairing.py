from __future__ import annotations

from typing import Any

from openpilot.system.app.identity import is_dongle_id
from openpilot.system.app.websocketd import authorize_peer, disable_pairing_mode, pairing_mode_active


def authorize_ble_peer(public_key: str, request_id: str, label: str | None = None) -> dict[str, Any]:
  if not pairing_mode_active():
    raise PermissionError("pairing mode is not active")
  if not is_dongle_id(public_key):
    raise ValueError("invalid app public key")
  if not request_id or len(request_id) > 64:
    raise ValueError("invalid pairing request ID")
  if label is not None and len(label) > 80:
    raise ValueError("pairing label is too long")

  authorize_peer(public_key, label=label)
  disable_pairing_mode()
  return {
    "publicKey": public_key,
    "requestId": request_id,
  }
