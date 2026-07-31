#!/usr/bin/env python3
import argparse
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUNTIME_TIMEOUT = 3
STARTUP_GRACE = 30.
RUNTIME_HZ = 4.
ACCELERATION_DUE_TO_GRAVITY = 9.81
DEFAULT_MAX_LAT_ACCEL = 3.0
STARTED_AT = time.monotonic()
CAM_LED_ADDR = 0x64
CAM_LED_BUSES = (16, 18, 20)
CAM_LED_STATUS_CAMERAS = (2, 3)
CAM_LED_RETRY_INTERVAL = 5.
CAM_LED_SYSFS_ROOT = Path("/sys/class/leds")
CAM_LED_STARTUP_BRIGHTNESS = 204
CAM_LED_MIN_BRIGHTNESS_PERCENT = 30.
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

AudibleAlert: Any = None
Ratekeeper: Any = None
log: Any = None
messaging: Any = None
get_ble_pairing: Any = None


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


STARTING = LedState("starting", 0, 0, 180)
READY = LedState("ready", 0, 0, 180)
PAIRING = LedState("pairing", 0, 180, 180)
ENGAGED = LedState("engaged", 0, 180, 35)
YELLOW = LedState("yellow", 180, 130, 0)
CALIBRATING = LedState("calibrating", 120, 75, 20)
WARNING = LedState("warning", 180, 130, 0)
CRITICAL = LedState("critical", 180, 0, 0)
UPDATING = LedState("updating", 150, 0, 180)
OFF = LedState("off", 0, 0, 0)


def clamp(value: float, low: float, high: float) -> float:
  return max(low, min(high, value))


def interp(value: float, x0: float, x1: float, y0: float, y1: float) -> float:
  if x1 == x0:
    return y1
  return y0 + (clamp((value - x0) / (x1 - x0), 0., 1.) * (y1 - y0))


def blend(name: str, start: LedState, end: LedState, amount: float) -> LedState:
  amount = clamp(amount, 0., 1.)
  return LedState(
    name,
    int(start.red + (end.red - start.red) * amount),
    int(start.green + (end.green - start.green) * amount),
    int(start.blue + (end.blue - start.blue) * amount),
  )


def max_brightness(state: LedState, brightness: int = 255) -> LedState:
  peak = max(state.red, state.green, state.blue)
  if peak <= 0:
    return OFF
  brightness = int(clamp(brightness, 0, 255))
  scale = brightness / peak
  return LedState(
    state.name,
    int(state.red * scale),
    int(state.green * scale),
    int(state.blue * scale),
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
    self.boards = [
      CameraLedBoard("driver", camera_num=1, bus_num=CAM_LED_BUSES[0]),
      CameraLedBoard("road", camera_num=2, bus_num=CAM_LED_BUSES[1]),
      CameraLedBoard("wide", camera_num=3, bus_num=CAM_LED_BUSES[2]),
    ]
    self.last_state: LedState | None = None
    self.last_channels: dict[int, list[int]] | None = None
    self.last_sent = 0.

  def set(self, state: LedState, timeout: int = RUNTIME_TIMEOUT, force: bool = False,
          brightness: int = 255) -> None:
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


def driver_monitoring_alert_state(sm) -> LedState | None:
  if not sm.seen['driverMonitoringState'] or not sm.alive['driverMonitoringState']:
    return None

  dm_state = sm['driverMonitoringState']
  if dm_state.lockout or dm_state.alwaysOnLockout or dm_state.alertLevel == log.DriverMonitoringState.AlertLevel.three:
    return CRITICAL
  if dm_state.alertLevel in (log.DriverMonitoringState.AlertLevel.one, log.DriverMonitoringState.AlertLevel.two):
    return WARNING
  return None


def alert_state(sm) -> LedState | None:
  dm_alert = driver_monitoring_alert_state(sm)
  if dm_alert is not None:
    return dm_alert

  if not sm.seen['selfdriveState'] or not sm.alive['selfdriveState']:
    return None

  selfdrive_state = sm['selfdriveState']
  alert_sound = selfdrive_state.alertSound.raw
  has_alert = (
    selfdrive_state.alertStatus != log.SelfdriveState.AlertStatus.normal or
    selfdrive_state.alertSize != log.SelfdriveState.AlertSize.none or
    alert_sound != AudibleAlert.none or
    bool(selfdrive_state.alertText1) or
    bool(selfdrive_state.alertText2)
  )
  if not has_alert:
    return None

  if (
    selfdrive_state.state == log.SelfdriveState.OpenpilotState.softDisabling or
    selfdrive_state.alertStatus == log.SelfdriveState.AlertStatus.critical or
    selfdrive_state.alertSize == log.SelfdriveState.AlertSize.full or
    alert_sound == AudibleAlert.warningImmediate
  ):
    return CRITICAL
  return WARNING


def calibration_state(sm) -> LedState | None:
  if not sm.seen['liveCalibration'] or not sm.alive['liveCalibration']:
    return None
  if not sm.seen['deviceState'] or not sm.alive['deviceState'] or not sm['deviceState'].started:
    return None
  if sm['liveCalibration'].calStatus != log.LiveCalibrationData.Status.calibrated:
    return CALIBRATING
  return None


def steering_utilization(sm) -> float:
  if not sm.seen['carControl'] or not sm.alive['carControl'] or not sm['carControl'].latActive:
    return 0.
  if not sm.seen['controlsState'] or not sm.alive['controlsState']:
    return 0.

  controls_state = sm['controlsState']
  lac = getattr(controls_state.lateralControlState, controls_state.lateralControlState.which())
  util = 0.

  if controls_state.lateralControlState.which() == 'angleState':
    if sm.seen['carState'] and sm.alive['carState'] and sm.seen['liveParameters'] and sm.alive['liveParameters']:
      car_state = sm['carState']
      live_parameters = sm['liveParameters']
      actual_lateral_accel = controls_state.curvature * car_state.vEgo ** 2
      desired_lateral_accel = controls_state.desiredCurvature * car_state.vEgo ** 2
      accel_diff = desired_lateral_accel - actual_lateral_accel
      roll_compensation = live_parameters.roll * ACCELERATION_DUE_TO_GRAVITY * interp(car_state.vEgo, 5., 15., 0., 1.)
      lateral_acceleration = actual_lateral_accel - roll_compensation
      max_lateral_acceleration = DEFAULT_MAX_LAT_ACCEL
      if sm.seen['carParams'] and sm['carParams'].maxLateralAccel > 0.:
        max_lateral_acceleration = sm['carParams'].maxLateralAccel
      util = abs(clamp((lateral_acceleration + accel_diff) / max_lateral_acceleration, -1., 1.))
  elif sm.seen['carOutput'] and sm.alive['carOutput']:
    util = abs(clamp(sm['carOutput'].actuatorsOutput.torque, -1., 1.))

  if getattr(lac, "saturated", False):
    util = max(util, 0.95)
  return util


def engaged_state(sm) -> LedState:
  util = steering_utilization(sm)
  if util < 0.65:
    return ENGAGED
  if util < 0.85:
    return blend("engaged_yellow", ENGAGED, YELLOW, (util - 0.65) / 0.20)
  return blend("engaged_red", YELLOW, CRITICAL, (util - 0.85) / 0.15)


def camera_led_brightness(sm) -> int:
  for service in ("wideRoadCameraState", "roadCameraState"):
    if not sm.seen[service] or not sm.alive[service] or not sm.valid[service]:
      continue

    light_sensor = clamp(100. - sm[service].exposureValPercent, 0., 100.)
    if light_sensor <= 8.:
      normalized_light = light_sensor / 903.3
    else:
      normalized_light = ((light_sensor + 16.) / 116.) ** 3.

    brightness_percent = interp(normalized_light, 0., 1., CAM_LED_MIN_BRIGHTNESS_PERCENT, 100.)
    return round(255. * brightness_percent / 100.)

  return CAM_LED_STARTUP_BRIGHTNESS


def pairing_led_channels(brightness: int = 255) -> dict[int, list[int]] | None:
  from openpilot.system.athena.ble_pairing import PAIRING_COLORS

  pairing = get_ble_pairing()
  if pairing is None and not pairing_mode_active():
    return None

  colors = pairing.get("colors") if pairing is not None else None
  if not isinstance(colors, list) or len(colors) != 6 or any(color not in PAIRING_COLORS for color in colors):
    colors = ["turquoise"] * 6 if int(time.monotonic() * 2) % 2 == 0 else ["off"] * 6

  def code_channels(camera_colors: list[str]) -> list[int]:
    channels = []
    for color in camera_colors:
      values = PAIRING_COLORS[color] if color != "off" else (0, 0, 0)
      channels.extend(round(value * brightness / 255.) for value in values)
    return channels

  return {
    1: [0] * len(CAM_LED_CHANNELS),
    2: code_channels(colors[:3]),
    3: code_channels(colors[3:]),
  }


def updater_led_state(updater_state: str, now: float | None = None) -> LedState | None:
  if not any(phase in updater_state.lower() for phase in ("downloading", "finalizing", "installing")):
    return None
  now = time.monotonic() if now is None else now
  return UPDATING if int(now * 2) % 2 == 0 else OFF


def led_state(sm, updater_state: str = "") -> LedState:
  if manager_failed(sm) or panda_failed(sm):
    return CRITICAL

  update_state = updater_led_state(updater_state)
  if update_state is not None:
    return update_state

  if not sm.seen['deviceState']:
    return STARTING

  alert = alert_state(sm)
  if alert is not None:
    return alert

  calibrating = calibration_state(sm)
  if calibrating is not None:
    return calibrating

  if sm.seen['selfdriveState'] and sm.alive['selfdriveState']:
    selfdrive_state = sm['selfdriveState']
    if selfdrive_state.enabled:
      return engaged_state(sm)
    if sm['deviceState'].started and not selfdrive_state.engageable:
      return WARNING

  if panda_disconnected(sm):
    return STARTING if time.monotonic() - STARTED_AT < STARTUP_GRACE else CRITICAL

  return READY


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--clear", action="store_true")
  args = parser.parse_args()

  led = CameraLeds()
  if args.clear:
    led.clear()
    return

  global AudibleAlert, Ratekeeper, get_ble_pairing, log, messaging, pairing_mode_active
  from openpilot.common.params import Params
  from openpilot.common.realtime import Ratekeeper as OpenpilotRatekeeper
  from openpilot.system.athena.ble_pairing import get_ble_pairing as athena_get_ble_pairing
  from openpilot.system.athena.websocketd import pairing_mode_active as athena_pairing_mode_active
  try:
    from openpilot.cereal import log as cereal_log, messaging as cereal_messaging
    AudibleAlert = cereal_log.SelfdriveState.AudibleAlert
  except ModuleNotFoundError:
    from cereal import car, log as cereal_log, messaging as cereal_messaging
    AudibleAlert = car.CarControl.HUDControl.AudibleAlert
  Ratekeeper = OpenpilotRatekeeper
  log = cereal_log
  messaging = cereal_messaging
  get_ble_pairing = athena_get_ble_pairing
  pairing_mode_active = athena_pairing_mode_active
  params = Params()

  done = False

  def sigterm_handler(signum, frame) -> None:
    nonlocal done
    done = True

  signal.signal(signal.SIGINT, sigterm_handler)
  signal.signal(signal.SIGTERM, sigterm_handler)

  sm = messaging.SubMaster([
    'carControl',
    'carOutput',
    'carParams',
    'carState',
    'controlsState',
    'deviceState',
    'driverMonitoringState',
    'liveCalibration',
    'liveParameters',
    'managerState',
    'pandaStates',
    'roadCameraState',
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
      led.set(led_state(sm, params.get("UpdaterState") or ""), brightness=brightness)
    rk.keep_time()

  led.clear()


if __name__ == "__main__":
  main()
