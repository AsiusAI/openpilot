import os
import time
from abc import abstractmethod, ABC
from dataclasses import dataclass, fields

from openpilot.cereal import log
from openpilot.common.esim.base import LPABase

NetworkType = log.DeviceState.NetworkType
NetworkStrength = log.DeviceState.NetworkStrength

@dataclass
class ThermalZone:
  # a zone from /sys/class/thermal/thermal_zone*
  name: str             # a.k.a type
  scale: float = 1000.  # scale to get degrees in C
  label: str | None = None
  zone_number = -1

  def read(self) -> float:
    if self.zone_number < 0:
      for n in os.listdir("/sys/devices/virtual/thermal"):
        if not n.startswith("thermal_zone"):
          continue
        with open(os.path.join("/sys/devices/virtual/thermal", n, "type")) as f:
          if f.read().strip() == self.name:
            self.zone_number = int(n.removeprefix("thermal_zone"))
            break

    try:
      with open(f"/sys/devices/virtual/thermal/thermal_zone{self.zone_number}/temp") as f:
        return int(f.read()) / self.scale
    except FileNotFoundError:
      return 0


class HwmonThermalZone(ThermalZone):
  def __init__(self, name: str, hwmon_name: str, attribute: str = "temp1_input", scale: float = 1000.,
               poll_interval: float = 30., hwmon_root: str = "/sys/class/hwmon"):
    super().__init__(name, scale)
    self.hwmon_name = hwmon_name
    self.attribute = attribute
    self.poll_interval = poll_interval
    self.hwmon_root = hwmon_root
    self._path: str | None = None
    self._last_read: float | None = None
    self._temperature = 0.

  def _find_path(self) -> str | None:
    try:
      hwmon_devices = os.listdir(self.hwmon_root)
    except FileNotFoundError:
      return None

    for device in hwmon_devices:
      device_path = os.path.join(self.hwmon_root, device)
      try:
        with open(os.path.join(device_path, "name")) as f:
          if f.read().strip() == self.hwmon_name:
            return os.path.join(device_path, self.attribute)
      except OSError:
        continue
    return None

  def read(self) -> float:
    now = time.monotonic()
    if self._last_read is not None and now - self._last_read < self.poll_interval:
      return self._temperature
    self._last_read = now

    if self._path is None:
      self._path = self._find_path()
    if self._path is None:
      return self._temperature

    try:
      with open(self._path) as f:
        self._temperature = int(f.read()) / self.scale
    except FileNotFoundError:
      self._path = None
    except (OSError, ValueError):
      pass
    return self._temperature


@dataclass
class ThermalConfig:
  cpu: list[ThermalZone] | None = None
  gpu: list[ThermalZone] | None = None
  dsp: ThermalZone | None = None
  pmic: list[ThermalZone] | None = None
  memory: ThermalZone | None = None
  intake: ThermalZone | None = None
  exhaust: ThermalZone | None = None
  gnss: ThermalZone | None = None
  bottomSoc: ThermalZone | None = None
  thermal_zones: list[ThermalZone] | None = None

  def get_msg(self):
    ret = {}
    for f in fields(ThermalConfig):
      v = getattr(self, f.name)
      if v is not None:
        if f.name == "thermal_zones":
          zones = [(x, x.read()) for x in v]
          ret["thermalZones"] = [{"name": x.label or x.name, "temp": temp} for x, temp in zones if temp != 0]
        elif isinstance(v, list):
          ret[f.name + "TempC"] = [x.read() for x in v]
        else:
          ret[f.name + "TempC"] = v.read()
    return ret

class HardwareBase(ABC):
  @staticmethod
  def get_cmdline() -> dict[str, str]:
    with open('/proc/cmdline') as f:
      cmdline = f.read()
    return {kv[0]: kv[1] for kv in [s.split('=') for s in cmdline.split(' ')] if len(kv) == 2}

  @staticmethod
  def read_param_file(path, parser, default=0):
    try:
      with open(path) as f:
        return parser(f.read())
    except Exception:
      return default

  def booted(self) -> bool:
    return True

  def reboot(self, reason=None):
    print("REBOOT!")

  def uninstall(self):
    print("uninstall")

  def get_os_version(self):
    return None

  @abstractmethod
  def get_device_type(self):
    pass

  def get_imei(self) -> str:
    return ""

  def get_serial(self):
    return ""

  def get_network_info(self):
    return None

  def get_network_type(self):
    return NetworkType.none

  def get_sim_info(self):
    return {
      'sim_id': '',
      'mcc_mnc': None,
      'network_type': ["Unknown"],
      'sim_state': ["ABSENT"],
      'data_connected': False
    }

  def get_sim_lpa(self) -> LPABase:
    raise NotImplementedError("SIM LPA not available")

  def get_network_strength(self, network_type):
    return NetworkStrength.unknown

  def get_network_metered(self, network_type) -> bool:
    return network_type not in (NetworkType.none, NetworkType.wifi, NetworkType.ethernet)

  def get_current_power_draw(self):
    return 0

  def get_som_power_draw(self):
    return 0

  def shutdown(self):
    print("SHUTDOWN!")

  def get_thermal_config(self):
    return ThermalConfig()

  def set_display_power(self, on: bool):
    pass

  def set_screen_brightness(self, percentage):
    pass

  def get_screen_brightness(self):
    return 0

  def set_power_save(self, powersave_enabled):
    pass

  def get_gpu_usage_percent(self):
    return 0

  def get_modem_temperatures(self):
    return []

  def get_modem_state(self) -> dict:
    return {}

  def get_ufs_health(self) -> dict:
    return {}

  def initialize_hardware(self):
    pass

  def reset_internal_panda(self):
    pass

  def recover_internal_panda(self):
    pass

  def get_modem_data_usage(self):
    return -1, -1

  def set_ir_power(self, percent: int):
    pass
