from pathlib import Path

from openpilot.common.hardware.asius.ufs import UfsHealthReader
from openpilot.common.hardware.base import HwmonThermalZone, ThermalConfig, ThermalZone


def write_value(root: Path, relative_path: str, value: str) -> None:
  path = root / relative_path
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(value)


def test_ufs_health_reader(tmp_path: Path):
  controller = tmp_path / "bus/platform/drivers/ufshcd-qcom/1d84000.ufshc"
  write_value(controller, "string_descriptors/manufacturer_name", "YMTC\n")
  write_value(controller, "string_descriptors/product_name", "YMUS8A1TE2D1C1\n")
  write_value(controller, "string_descriptors/product_revision", "0301\n")
  write_value(controller, "device_descriptor/specification_version", "0x0310\n")
  write_value(controller, "health_descriptor/eol_info", "0x01\n")
  write_value(controller, "health_descriptor/life_time_estimation_a", "0x02\n")
  write_value(controller, "health_descriptor/life_time_estimation_b", "0x03\n")
  write_value(controller, "critical_health", "4\n")
  write_value(tmp_path, "kernel/debug/ufshcd/1d84000.ufshc/stats", """\
PHY Adapter Layer errors (except LINERESET): 1
Data Link Layer errors: 2
IS Fatal errors (CEFES, SBFES, HCFES, DFES): 3
Logical Unit Resets: 4
Host Resets: 5
""")

  now = [100.]
  reader = UfsHealthReader(tmp_path, refresh_interval=60., clock=lambda: now[0])
  assert reader.read() == {
    "present": True,
    "manufacturer": "YMTC",
    "product": "YMUS8A1TE2D1C1",
    "revision": "0301",
    "specificationVersion": 0x310,
    "eolInfo": 1,
    "lifeTimeEstimationA": 2,
    "lifeTimeEstimationB": 3,
    "criticalHealthCount": 4,
    "phyErrorCount": 1,
    "dataLinkErrorCount": 2,
    "fatalErrorCount": 3,
    "logicalUnitResetCount": 4,
    "hostResetCount": 5,
  }

  write_value(controller, "health_descriptor/eol_info", "0x02\n")
  assert reader.read()["eolInfo"] == 1
  now[0] += 61.
  assert reader.read()["eolInfo"] == 2


def test_ufs_health_reader_without_ufs(tmp_path: Path):
  assert UfsHealthReader(tmp_path).read() == {}


def test_hwmon_temperature_and_generic_thermal_zones(tmp_path: Path):
  hwmon = tmp_path / "hwmon7"
  write_value(hwmon, "name", "ufs\n")
  write_value(hwmon, "temp1_input", "41875\n")

  ufs_case = HwmonThermalZone("ufsCase", "ufs", poll_interval=0., hwmon_root=str(tmp_path))
  ufs_board = ThermalZone("unused", label="ufsBoard")
  ufs_board.read = lambda: 38.5  # type: ignore[method-assign]

  assert ufs_case.read() == 41.875
  assert ThermalConfig(thermal_zones=[ufs_board, ufs_case]).get_msg() == {
    "thermalZones": [
      {"name": "ufsBoard", "temp": 38.5},
      {"name": "ufsCase", "temp": 41.875},
    ],
  }
