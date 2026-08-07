#!/usr/bin/env python3

import time
from multiprocessing import Process

from openpilot.common.hardware import HARDWARE
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.common.version import get_build_metadata
from openpilot.system.manager.process import launcher


def manage(module: str, process_name: str, pid_param: str) -> None:
  params = Params()
  build_metadata = get_build_metadata()

  cloudlog.bind_global(dongle_id=params.get("DongleId"),
                       version=build_metadata.openpilot.version,
                       origin=build_metadata.openpilot.git_normalized_origin,
                       branch=build_metadata.channel,
                       commit=build_metadata.openpilot.git_commit,
                       dirty=build_metadata.openpilot.is_dirty,
                       device=HARDWARE.get_device_type())

  try:
    while True:
      cloudlog.info(f"starting {process_name}")
      process = Process(name=process_name, target=launcher, args=(module, process_name))
      process.start()
      process.join()
      cloudlog.event(f"{process_name} exited", exitcode=process.exitcode)
      time.sleep(5)
  except Exception:
    cloudlog.exception(f"manage_{process_name}.exception")
  finally:
    params.remove(pid_param)
