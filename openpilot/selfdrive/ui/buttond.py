#!/usr/bin/env python3
import argparse
import glob
import os
import signal
import struct
import time
from dataclasses import dataclass
from enum import Enum, auto

EV_KEY = 0x01
KEY_PROG1 = 148
KEY_RELEASED = 0
KEY_PRESSED = 1
PAIR_HOLD_SECONDS = 3.0
RECONNECT_SECONDS = 5.0
RUNTIME_HZ = 20.0
INPUT_EVENT = struct.Struct("@llHHI")


class ButtonAction(Enum):
  PAIR = auto()


# TODO: Move pairing to the dedicated GPIO button once panda-v5 hardware is
# available. PWER is only a temporary input for Dragon bench hardware.
def find_power_key() -> str | None:
  for name_path in sorted(glob.glob("/sys/class/input/event*/device/name")):
    try:
      with open(name_path, encoding="utf-8") as f:
        if f.read().strip() == "pmic_pwrkey":
          event_name = name_path.split("/")[-3]
          return f"/dev/input/{event_name}"
    except OSError:
      continue
  return None


@dataclass
class PowerButton:
  event_device: str | None = None
  hold_seconds: float = PAIR_HOLD_SECONDS
  fd: int | None = None
  pressed_at: float | None = None
  pair_sent: bool = False
  next_retry: float = 0.0
  pending: bytes = b""

  def connect(self, now: float) -> bool:
    if self.fd is not None:
      return True
    if now < self.next_retry:
      return False

    path = self.event_device or find_power_key()
    if path is None:
      self.next_retry = now + RECONNECT_SECONDS
      return False

    try:
      self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
      self.event_device = path
      self.pending = b""
      return True
    except OSError:
      self.next_retry = now + RECONNECT_SECONDS
      return False

  def close(self) -> None:
    if self.fd is not None:
      os.close(self.fd)
      self.fd = None
    self.pressed_at = None
    self.pair_sent = False
    self.pending = b""

  def _long_hold(self, now: float) -> list[ButtonAction]:
    if self.pressed_at is not None and not self.pair_sent and now - self.pressed_at >= self.hold_seconds:
      self.pair_sent = True
      return [ButtonAction.PAIR]
    return []

  def poll(self, now: float | None = None) -> list[ButtonAction]:
    now = time.monotonic() if now is None else now
    actions = self._long_hold(now)
    if not self.connect(now):
      return actions
    fd = self.fd
    if fd is None:
      return actions

    try:
      while True:
        chunk = os.read(fd, INPUT_EVENT.size * 16)
        if not chunk:
          self.close()
          break
        self.pending += chunk
    except BlockingIOError:
      pass
    except OSError:
      self.close()
      return actions

    complete = len(self.pending) - (len(self.pending) % INPUT_EVENT.size)
    for offset in range(0, complete, INPUT_EVENT.size):
      _, _, event_type, code, value = INPUT_EVENT.unpack_from(self.pending, offset)
      if event_type != EV_KEY or code != KEY_PROG1:
        continue

      if value == KEY_PRESSED:
        self.pressed_at = now
        self.pair_sent = False
      elif value == KEY_RELEASED and self.pressed_at is not None:
        actions.extend(self._long_hold(now))
        self.pressed_at = None
        self.pair_sent = False

    self.pending = self.pending[complete:]
    actions.extend(self._long_hold(now))
    return actions


def main() -> None:
  from openpilot.common.realtime import Ratekeeper
  from openpilot.common.swaglog import cloudlog
  from openpilot.system.athena.ble_pairing import approve_ble_pairing
  from openpilot.system.athena.websocketd import enable_pairing_mode

  parser = argparse.ArgumentParser()
  parser.add_argument("--event-device")
  args = parser.parse_args()

  done = False

  def sigterm_handler(signum, frame) -> None:
    nonlocal done
    done = True

  signal.signal(signal.SIGINT, sigterm_handler)
  signal.signal(signal.SIGTERM, sigterm_handler)

  button = PowerButton(event_device=args.event_device)
  rk = Ratekeeper(RUNTIME_HZ)

  while not done:
    for action in button.poll():
      if action == ButtonAction.PAIR:
        approved = approve_ble_pairing()
        if approved is not None:
          cloudlog.event("asius.button.pairing_approved", public_key=approved["publicKey"], request_id=approved["requestId"])
        else:
          pairing_until = enable_pairing_mode()
          cloudlog.event("asius.button.pairing_mode", pairing_until=pairing_until)
    rk.keep_time()

  button.close()


if __name__ == "__main__":
  main()
