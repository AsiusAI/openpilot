from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DEVICE_NAME = "Asius v1"
DEVICE_NAME_PATH = Path("/data/asius/device-name")
MAX_DEVICE_NAME_LENGTH = 40


def get_device_name() -> str:
  try:
    name = DEVICE_NAME_PATH.read_text(encoding="utf-8").strip()
  except OSError:
    return DEFAULT_DEVICE_NAME
  return name if name and len(name) <= MAX_DEVICE_NAME_LENGTH and name.isprintable() else DEFAULT_DEVICE_NAME


def set_device_name(name: str) -> str:
  name = name.strip()
  if not name:
    raise ValueError("device name is required")
  if len(name) > MAX_DEVICE_NAME_LENGTH:
    raise ValueError(f"device name must be {MAX_DEVICE_NAME_LENGTH} characters or fewer")
  if not name.isprintable():
    raise ValueError("device name cannot contain control characters")

  DEVICE_NAME_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
  temporary = DEVICE_NAME_PATH.with_suffix(".tmp")
  temporary.write_text(f"{name}\n", encoding="utf-8")
  temporary.chmod(0o600)
  os.replace(temporary, DEVICE_NAME_PATH)
  return name
