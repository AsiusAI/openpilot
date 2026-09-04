#!/usr/bin/env python3
import argparse
import os
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STARTUP_GRACE = 30.
RUNTIME_HZ = 4.
STARTED_AT = time.monotonic()
CAM_LED_ADDR = 0x64
CAM_LED_BUSES = (16, 18, 20)
CAM_LED_STATUS_CAMERAS = (2, 3)
CAM_LED_RETRY_INTERVAL = 5.
CAM_LED_SYSFS_ROOT = Path("/sys/class/leds")
CAM_LED_STARTUP_BRIGHTNESS = 26
CAM_LED_MIN_BRIGHTNESS_PERCENT = 5.
CAM_LED_MAX_BRIGHTNESS_PERCENT = 10.
CAM_LED_CHANNELS = (
  ("red", 0), ("green", 0), ("blue", 0),
  ("red", 1), ("green", 1), ("blue", 1),
  ("red", 2), ("green", 2), ("blue", 2),
)

IS31FL3199_SHUTDOWN = 0x00
IS31FL3199_CTRL1 = 0x01
IS31FL3199_CTRL2 = 0x02
IS31FL3199_CONFIG2 = 0x04
IS31FL3199_PWM_BASE = 0x07
IS31FL3199_UPDATE = 0x10
IS31FL3199_RESET = 0xff

IS31FL3199_ALL_CHANNELS_CTRL1 = 0x77
IS31FL3199_ALL_CHANNELS_CTRL2 = 0x07
IS31FL3199_CURRENT_40MA = 0x40

Ratekeeper: Any = None
log: Any = None
messaging: Any = None
def _pairing_mode_inactive() -> bool:
  return False


pairing_mode_active: Callable[[], bool] = _pairing_mode_inactive


def cloudlog_exception(message: str) -> None:
  try:
    from openpilot.common.swaglog import cloudlog
    cloudlog.exception(message)
  except Exception as e:
    print(f"{message}: {e}")


@dataclass(frozen=True)
class LedState:
  name: str
  red: int
  green: int
  blue: int


BLUE = LedState("blue", 0, 0, 180)
GREEN = LedState("green", 0, 180, 35)
YELLOW = LedState("yellow", 180, 130, 0)
BROWN = LedState("brown", 120, 75, 20)
RED = LedState("red", 180, 0, 0)
OFF = LedState("off", 0, 0, 0)

WARNING_ALERT_SOUNDS = {
  "warningSoft",
  "warningImmediate",
  "prompt",
  "promptRepeat",
  "promptDistracted",
  "preAlert",
}


def clamp(value: float, low: float, high: float) -> float:
  return max(low, min(high, value))


def interp(value: float, x0: float, x1: float, y0: float, y1: float) -> float:
  if x1 == x0:
    return y1
  return y0 + (clamp((value - x0) / (x1 - x0), 0., 1.) * (y1 - y0))


def max_brightness(state: LedState, brightness: int = 255) -> LedState:
  peak = max(state.red, state.green, state.blue)
  if peak <= 0:
    return OFF
  brightness = int(clamp(brightness, 0, 255))
  scale = brightness / peak
  return LedState(
    state.name,
    round(state.red * scale),
    round(state.green * scale),
    round(state.blue * scale),
  )


@dataclass
class CameraLedBoard:
  name: str
  camera_num: int
  bus_num: int
  bus: Any = None
  kernel_leds: bool = False
  initialized: bool = False
  next_retry: float = 0.

  @property
  def channel_paths(self) -> list[Path]:
    return [
      CAM_LED_SYSFS_ROOT / f"asius:cam{self.camera_num}:{color}:{package}"
      for color, package in CAM_LED_CHANNELS
    ]

  def kernel_leds_available(self) -> bool:
    return all((path / "brightness").exists() for path in self.channel_paths)

  def connect(self):
    if self.kernel_leds_available():
      if self.bus is not None:
        self.bus.close()
        self.bus = None
      self.kernel_leds = True
      return None
    if self.bus is None:
      from openpilot.common.i2c import SMBus
      self.bus = SMBus(self.bus_num)
    return self.bus

  def write(self, register: int, value: int) -> None:
    self.connect().write_byte_data(CAM_LED_ADDR, register, value)

  def close(self) -> None:
    if self.bus is not None:
      self.bus.close()
      self.bus = None
    self.kernel_leds = False
    self.initialized = False

  def init(self) -> None:
    self.connect()
    if self.kernel_leds:
      for path in self.channel_paths:
        trigger = path / "trigger"
        if trigger.exists():
          trigger.write_text("none")
    else:
      self.write(IS31FL3199_RESET, 0x00)
      time.sleep(0.001)
      self.write(IS31FL3199_CONFIG2, IS31FL3199_CURRENT_40MA)
    self.initialized = True

  def set_channels(self, channels: list[int]) -> None:
    if len(channels) != len(CAM_LED_CHANNELS):
      raise ValueError(f"expected {len(CAM_LED_CHANNELS)} LED channels, got {len(channels)}")
    if any(value < 0 or value > 255 for value in channels):
      raise ValueError("LED channel brightness must be between 0 and 255")
    if not self.initialized:
      self.init()

    if self.kernel_leds:
      for path, value in zip(self.channel_paths, channels, strict=True):
        (path / "brightness").write_text(str(value))
      return

    for channel, value in enumerate(channels):
      self.write(IS31FL3199_PWM_BASE + channel, value)

    if max(channels) > 0:
      self.write(IS31FL3199_CTRL1, IS31FL3199_ALL_CHANNELS_CTRL1)
      self.write(IS31FL3199_CTRL2, IS31FL3199_ALL_CHANNELS_CTRL2)
      self.write(IS31FL3199_UPDATE, 0x00)
      self.write(IS31FL3199_SHUTDOWN, 0x01)
    else:
      self.write(IS31FL3199_UPDATE, 0x00)
      self.write(IS31FL3199_SHUTDOWN, 0x00)

  def set(self, state: LedState) -> None:
    self.set_channels([
      state.red, state.green, state.blue,
      state.red, state.green, state.blue,
      state.red, state.green, state.blue,
    ])


class CameraLeds:
  def __init__(self) -> None:
    self.boards = []
    if os.getenv("NO_DCAM") != "1":
      self.boards.append(CameraLedBoard("driver", camera_num=1, bus_num=CAM_LED_BUSES[0]))
    self.boards += [
      CameraLedBoard("road", camera_num=2, bus_num=CAM_LED_BUSES[1]),
      CameraLedBoard("wide", camera_num=3, bus_num=CAM_LED_BUSES[2]),
    ]
    self.last_state: LedState | None = None
    self.last_channels: dict[int, list[int]] | None = None
    self.last_sent = 0.

  def set(self, state: LedState, force: bool = False, brightness: int = 255) -> None:
    state = max_brightness(state, brightness)
    now = time.monotonic()
    if not force and state == self.last_state and (now - self.last_sent) < 1.0:
      return

    any_success = False
    for board in self.boards:
      if now < board.next_retry:
        continue
      try:
        board.set(state if board.camera_num in CAM_LED_STATUS_CAMERAS else OFF)
        board.next_retry = 0.
        any_success = True
      except Exception:
        board.next_retry = now + CAM_LED_RETRY_INTERVAL
        cloudlog_exception(f"failed to set {board.name} camera LEDs")
        board.close()

    if any_success:
      self.last_state = state
      self.last_channels = None
      self.last_sent = now

  def set_channels(self, camera_channels: dict[int, list[int]], force: bool = False) -> None:
    now = time.monotonic()
    if not force and camera_channels == self.last_channels and (now - self.last_sent) < 1.0:
      return

    any_success = False
    for board in self.boards:
      channels = camera_channels.get(board.camera_num)
      if channels is None or now < board.next_retry:
        continue
      try:
        board.set_channels(channels)
        board.next_retry = 0.
        any_success = True
      except Exception:
        board.next_retry = now + CAM_LED_RETRY_INTERVAL
        cloudlog_exception(f"failed to set {board.name} camera LED channels")
        board.close()
    self.last_state = None
    if any_success:
      self.last_channels = camera_channels
      self.last_sent = now

  def clear(self) -> None:
    try:
      self.set(OFF, force=True)
    except Exception:
      cloudlog_exception("failed to clear camera LEDs")
    finally:
      self.last_state = None
      self.last_channels = None
      self.close()

  def close(self) -> None:
    for board in self.boards:
      board.close()


def manager_failed(sm) -> bool:
  if time.monotonic() - STARTED_AT < STARTUP_GRACE:
    return False
  if not sm.seen['managerState']:
    return False

  for process in sm['managerState'].processes:
    if process.name == "ledd":
      continue
    if process.shouldBeRunning and not process.running:
      return True
  return False


def process_should_run(sm, name: str) -> bool:
  if not sm.seen['managerState']:
    return True

  for process in sm['managerState'].processes:
    if process.name == name:
      return process.shouldBeRunning
  return True


def panda_disconnected(sm) -> bool:
  if not process_should_run(sm, "pandad"):
    return False

  if not sm.seen['pandaStates'] or not sm.alive['pandaStates']:
    return True

  panda_states = sm['pandaStates']
  if len(panda_states) == 0:
    return True

  for panda_state in panda_states:
    if panda_state.pandaType == log.PandaState.PandaType.unknown:
      return True
  return False


def panda_failed(sm) -> bool:
  if time.monotonic() - STARTED_AT < STARTUP_GRACE:
    return False
  if panda_disconnected(sm):
    return False

  for panda_state in sm['pandaStates']:
    if panda_state.faultStatus == log.PandaState.FaultStatus.faultPerm:
      return True
    if len(panda_state.faults) > 0 or panda_state.heartbeatLost:
      return True
  return False


def selfdrive_state_available(sm) -> bool:
  return sm.seen['selfdriveState'] and sm.alive['selfdriveState'] and sm.valid['selfdriveState']


def persistent_error(sm) -> bool:
  if (
    sm.seen['deviceState'] and sm.alive['deviceState'] and sm['deviceState'].started and
    not selfdrive_state_available(sm)
  ):
    return True
  if manager_failed(sm) or panda_failed(sm):
    return True
  if time.monotonic() - STARTED_AT < STARTUP_GRACE:
    return False
  return panda_disconnected(sm)


def engaged_warning(sm) -> bool:
  if not sm.seen['selfdriveState'] or not sm.alive['selfdriveState']:
    return False

  selfdrive_state = sm['selfdriveState']
  if not selfdrive_state.active:
    return False

  if sm.seen['driverMonitoringState'] and sm.alive['driverMonitoringState']:
    if sm['driverMonitoringState'].alertLevel != log.DriverMonitoringState.AlertLevel.none:
      return True

  return (
    selfdrive_state.state == log.SelfdriveState.OpenpilotState.softDisabling or
    selfdrive_state.alertSound.raw in WARNING_ALERT_SOUNDS
  )


def calibration_state(sm) -> LedState | None:
  if not sm.seen['extrinsicsCalibration'] or not sm.alive['extrinsicsCalibration']:
    return None
  if not sm.seen['deviceState'] or not sm.alive['deviceState'] or not sm['deviceState'].started:
    return None
  if sm['extrinsicsCalibration'].calStatus != log.ExtrinsicsCalibration.Status.calibrated:
    return BROWN
  return None


def blinking(state: LedState, now: float | None = None) -> LedState:
  now = time.monotonic() if now is None else now
  return state if int(now * 2) % 2 == 0 else OFF


def camera_led_brightness(sm) -> int:
  service = "wideRoadCameraState"
  if not sm.seen[service] or not sm.alive[service] or not sm.valid[service]:
    return CAM_LED_STARTUP_BRIGHTNESS

  light_sensor = clamp(100. - sm[service].exposureValPercent, 0., 100.)
  if light_sensor <= 8.:
    normalized_light = light_sensor / 903.3
  else:
    normalized_light = ((light_sensor + 16.) / 116.) ** 3.

  brightness_percent = interp(normalized_light, 0., 1., CAM_LED_MIN_BRIGHTNESS_PERCENT, CAM_LED_MAX_BRIGHTNESS_PERCENT)
  return round(255. * brightness_percent / 100.)


def pairing_led_channels(brightness: int = 255) -> dict[int, list[int]] | None:
  if not pairing_mode_active():
    return None

  values = (0, round(255 * brightness / 255.), round(80 * brightness / 255.)) if int(time.monotonic() * 2) % 2 == 0 else (0, 0, 0)
  channels = list(values) * 3

  return {
    1: [0] * len(CAM_LED_CHANNELS),
    2: channels,
    3: channels,
  }


def led_state(sm, now: float | None = None) -> LedState:
  if not sm.seen['deviceState'] or not sm.alive['deviceState'] or not sm['deviceState'].started:
    return BLUE

  if persistent_error(sm):
    return RED

  calibrating = calibration_state(sm)
  if calibrating is not None:
    return calibrating

  selfdrive_available = selfdrive_state_available(sm)
  if selfdrive_available:
    selfdrive_state = sm['selfdriveState']
    if selfdrive_state.active:
      return blinking(RED, now) if engaged_warning(sm) else GREEN
    if sm['deviceState'].started and (
      not selfdrive_state.engageable or
      selfdrive_state.state == log.SelfdriveState.OpenpilotState.preEnabled
    ):
      return YELLOW

  return BLUE


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--clear", action="store_true")
  args = parser.parse_args()

  led = CameraLeds()
  if args.clear:
    led.clear()
    return

  global Ratekeeper, log, messaging, pairing_mode_active
  from openpilot.common.realtime import Ratekeeper as OpenpilotRatekeeper
  from openpilot.system.app.bluetoothd import pairing_mode_active as app_pairing_mode_active
  try:
    from openpilot.cereal import log as cereal_log, messaging as cereal_messaging
  except ModuleNotFoundError:
    from cereal import log as cereal_log, messaging as cereal_messaging
  Ratekeeper = OpenpilotRatekeeper
  log = cereal_log
  messaging = cereal_messaging
  pairing_mode_active = app_pairing_mode_active

  done = False

  def sigterm_handler(signum, frame) -> None:
    nonlocal done
    done = True

  signal.signal(signal.SIGINT, sigterm_handler)
  signal.signal(signal.SIGTERM, sigterm_handler)

  sm = messaging.SubMaster([
    'deviceState',
    'driverMonitoringState',
    'extrinsicsCalibration',
    'managerState',
    'pandaStates',
    'selfdriveState',
    'wideRoadCameraState',
  ], ignore_avg_freq=['managerState'])
  rk = Ratekeeper(RUNTIME_HZ)

  while not done:
    sm.update(0)
    brightness = camera_led_brightness(sm)
    pairing_channels = pairing_led_channels(brightness)
    if pairing_channels is not None:
      led.set_channels(pairing_channels)
    else:
      led.set(led_state(sm), brightness=brightness)
    rk.keep_time()

  led.clear()


if __name__ == "__main__":
  main()
