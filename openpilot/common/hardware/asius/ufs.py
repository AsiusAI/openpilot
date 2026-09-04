import time
from collections.abc import Callable
from pathlib import Path


class UfsHealthReader:
  def __init__(self, sysfs_root: str | Path = "/sys", refresh_interval: float = 10 * 60,
               clock: Callable[[], float] = time.monotonic):
    self.sysfs_root = Path(sysfs_root)
    self.refresh_interval = refresh_interval
    self.clock = clock
    self._last_refresh: float | None = None
    self._health: dict = {}

  @staticmethod
  def _read_text(path: Path) -> str | None:
    try:
      return path.read_text().strip()
    except OSError:
      return None

  @classmethod
  def _read_int(cls, path: Path) -> int | None:
    value = cls._read_text(path)
    if value is None:
      return None
    try:
      return int(value, 0)
    except ValueError:
      return None

  def _find_controller(self) -> Path | None:
    drivers = self.sysfs_root / "bus/platform/drivers"
    for health_descriptor in drivers.glob("ufshcd-*/*/health_descriptor"):
      if health_descriptor.is_dir():
        return health_descriptor.parent.resolve()
    return None

  def _read_error_stats(self, controller: Path) -> dict[str, int]:
    stats_path = self.sysfs_root / "kernel/debug/ufshcd" / controller.name / "stats"
    stats = self._read_text(stats_path)
    if stats is None:
      return {}

    labels = {
      "PHY Adapter Layer errors (except LINERESET)": "phyErrorCount",
      "Data Link Layer errors": "dataLinkErrorCount",
      "Network Layer errors": "networkErrorCount",
      "Transport Layer errors": "transportErrorCount",
      "Generic DME errors": "dmeErrorCount",
      "Auto-hibernate errors": "autoHibern8ErrorCount",
      "IS Fatal errors (CEFES, SBFES, HCFES, DFES)": "fatalErrorCount",
      "DME Link Startup errors": "linkStartupErrorCount",
      "PM Resume errors": "resumeErrorCount",
      "PM Suspend errors": "suspendErrorCount",
      "Logical Unit Resets": "logicalUnitResetCount",
      "Host Resets": "hostResetCount",
      "SCSI command aborts": "scsiAbortCount",
    }
    result = {}
    for line in stats.splitlines():
      label, separator, value = line.rpartition(":")
      if not separator or label.strip() not in labels:
        continue
      try:
        result[labels[label.strip()]] = int(value.strip())
      except ValueError:
        continue
    return result

  def read(self) -> dict:
    now = self.clock()
    if self._last_refresh is not None and now - self._last_refresh < self.refresh_interval:
      return self._health.copy()
    self._last_refresh = now

    controller = self._find_controller()
    if controller is None:
      self._health = {}
      return {}

    health: dict = {"present": True}
    text_fields = {
      "manufacturer": "string_descriptors/manufacturer_name",
      "product": "string_descriptors/product_name",
      "revision": "string_descriptors/product_revision",
    }
    int_fields = {
      "specificationVersion": "device_descriptor/specification_version",
      "eolInfo": "health_descriptor/eol_info",
      "lifeTimeEstimationA": "health_descriptor/life_time_estimation_a",
      "lifeTimeEstimationB": "health_descriptor/life_time_estimation_b",
      "criticalHealthCount": "critical_health",
      "deviceLevelExceptionCount": "device_lvl_exception_count",
    }

    for name, relative_path in text_fields.items():
      if (value := self._read_text(controller / relative_path)) is not None:
        health[name] = value
    for name, relative_path in int_fields.items():
      if (value := self._read_int(controller / relative_path)) is not None:
        health[name] = value
    health.update(self._read_error_stats(controller))

    self._health = health
    return health.copy()
