import glob
import os
import subprocess
from functools import cached_property

from openpilot.common.gpio import get_irqs_for_action
from openpilot.common.hardware.base import HwmonThermalZone, ThermalConfig, ThermalZone
from openpilot.common.hardware.asius.ufs import UfsHealthReader
from openpilot.common.hardware.comma.hardware import HardwareComma
from openpilot.common.utils import sudo_write


def _affine_irq(val: int, action: str) -> None:
  irqs = get_irqs_for_action(action)
  if not irqs:
    print(f"No IRQs found for '{action}'")
    return

  for irq in irqs:
    sudo_write(str(val), f"/proc/irq/{irq}/smp_affinity_list")


def _sudo_write_if_exists(val: str, path: str) -> None:
  if os.path.exists(path):
    sudo_write(val, path)


def _set_gpu_power_save(powersave_enabled: bool) -> None:
  power_control = "/sys/devices/platform/soc@0/3d00000.gpu/power/control"
  if powersave_enabled:
    sudo_write("auto", power_control)
  else:
    sudo_write("on", power_control)
    sudo_write("userspace", "/sys/class/devfreq/3d00000.gpu/governor")
    sudo_write("812000000", "/sys/class/devfreq/3d00000.gpu/userspace/set_freq")


def _raise_thermal_limits() -> None:
  trip_overrides = {
    90000: 100000,
    95000: 105000,
    100000: 108000,
    105000: 109000,
  }

  for zone in glob.glob("/sys/class/thermal/thermal_zone*/"):
    try:
      zone_type = open(zone + "type").read().strip()
    except OSError:
      continue

    if not any(zone_type.startswith(prefix) for prefix in ("cpu", "aoss", "ddr", "video", "cpuss", "gpuss")):
      continue

    for i in range(4):
      temp_path = zone + f"trip_point_{i}_temp"
      type_path = zone + f"trip_point_{i}_type"
      try:
        temp = int(open(temp_path).read().strip())
        trip_type = open(type_path).read().strip()
      except (OSError, ValueError):
        continue

      if trip_type in ("passive", "hot") and temp in trip_overrides:
        sudo_write(str(trip_overrides[temp]), temp_path)


class HardwareAsius(HardwareComma):
  @cached_property
  def amplifier(self):
    return None

  def get_device_type(self):
    return "v0"

  def get_serial(self):
    with open("/sys/devices/soc0/serial_number") as serial_file:
      return serial_file.read().strip()

  @cached_property
  def ufs_health(self):
    return UfsHealthReader()

  def get_thermal_config(self):
    return ThermalConfig(cpu=[ThermalZone(f"cpu{i}-thermal") for i in range(8)],
                         gpu=[ThermalZone("gpuss0-thermal"), ThermalZone("gpuss1-thermal")],
                         dsp=ThermalZone("nspss0-thermal"),
                         memory=ThermalZone("ddr-thermal"),
                         thermal_zones=[ThermalZone("ufs-thermal", label="ufsBoard"),
                                        HwmonThermalZone("ufsCase", "ufs", poll_interval=30.)])

  def get_ufs_health(self) -> dict:
    return self.ufs_health.read()

  def set_power_save(self, powersave_enabled):
    _set_gpu_power_save(powersave_enabled)
    super().set_power_save(powersave_enabled)

  def initialize_hardware(self):
    subprocess.run("sudo chmod a+w /dev/kmsg", shell=True)
    for stats_path in glob.glob("/sys/kernel/debug/ufshcd/*/stats"):
      subprocess.run(["sudo", "chmod", "a+r", stats_path], check=False)
    _sudo_write_if_exists("f", "/proc/irq/default_smp_affinity")

    _affine_irq(1, "msm_vidc")
    _affine_irq(1, "i2c_geni")

    sudo_write("userspace", "/sys/class/devfreq/3d00000.gpu/governor")
    sudo_write("812000000", "/sys/class/devfreq/3d00000.gpu/userspace/set_freq")
    _raise_thermal_limits()

    _affine_irq(3, "spi_geni")
    try:
      pid = subprocess.check_output(["pgrep", "-f", "spi0"], encoding='utf8').strip()
      subprocess.call(["sudo", "chrt", "-f", "-p", "1", pid])
      subprocess.call(["sudo", "taskset", "-pc", "3", pid])
    except subprocess.CalledProcessError as e:
      print(str(e))
