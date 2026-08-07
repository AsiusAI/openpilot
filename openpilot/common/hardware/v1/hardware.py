import glob
import os
import subprocess
from functools import cached_property

from openpilot.common.gpio import get_irqs_for_action
from openpilot.common.hardware.base import ThermalConfig, ThermalZone
from openpilot.common.hardware.tici.hardware import Tici
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


class Asius(Tici):
  @cached_property
  def amplifier(self):
    return None

  def get_device_type(self):
    return "v1"

  def get_serial(self):
    with open("/sys/devices/soc0/serial_number") as serial_file:
      return serial_file.read().strip()

  def get_thermal_config(self):
    return ThermalConfig(cpu=[ThermalZone(f"cpu{i}-thermal") for i in range(8)],
                         gpu=[ThermalZone("gpuss0-thermal"), ThermalZone("gpuss1-thermal")],
                         dsp=ThermalZone("nspss0-thermal"),
                         memory=ThermalZone("ddr-thermal"))

  def set_power_save(self, powersave_enabled):
    _set_gpu_power_save(powersave_enabled)

    for i in range(4, 8):
      val = '0' if powersave_enabled else '1'
      sudo_write(val, f'/sys/devices/system/cpu/cpu{i}/online')

    for policy in ('0', '4'):
      if powersave_enabled and policy == '4':
        continue
      governor = 'ondemand' if powersave_enabled else 'performance'
      _sudo_write_if_exists(governor, f'/sys/devices/system/cpu/cpufreq/policy{policy}/scaling_governor')
      if not powersave_enabled:
        sudo_write('1689600', f'/sys/devices/system/cpu/cpufreq/policy{policy}/scaling_max_freq')

    _affine_irq(7, "kgsl-3d0")
    for action in ("a5", "cci", "cpas_camnoc", "cpas-cdm", "csid", "ife", "csid-lite", "ife-lite"):
      _affine_irq(6, action)

  def initialize_hardware(self):
    subprocess.run("sudo chmod a+w /dev/kmsg", shell=True)
    _sudo_write_if_exists("f", "/proc/irq/default_smp_affinity")

    _affine_irq(1, "msm_vidc")
    _affine_irq(1, "i2c_geni")
    _affine_irq(5, "fts_ts")
    _affine_irq(5, "msm_drm")

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
