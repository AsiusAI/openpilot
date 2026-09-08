#!/usr/bin/env python3

import time
from multiprocessing import Process

from openpilot.common.params import Params
from openpilot.system.manager.process import launcher
from openpilot.common.swaglog import cloudlog
from openpilot.common.hardware import HARDWARE
from openpilot.common.version import get_build_metadata

ATHENA_MGR_PID_PARAM = "AthenadPid"


def manage(module: str, process_name: str, pid_param: str) -> None:
  params = Params()
  dongle_id = params.get("DongleId")
  build_metadata = get_build_metadata()

  cloudlog.bind_global(dongle_id=dongle_id,
                       version=build_metadata.openpilot.version,
                       origin=build_metadata.openpilot.git_normalized_origin,
                       branch=build_metadata.channel,
                       commit=build_metadata.openpilot.git_commit,
                       dirty=build_metadata.openpilot.is_dirty,
                       device=HARDWARE.get_device_type())

  try:
    while 1:
      cloudlog.info(f"starting {process_name}")
      proc = Process(name=process_name, target=launcher, args=(module, process_name))
      proc.start()
      proc.join()
      cloudlog.event(f"{process_name} exited", exitcode=proc.exitcode)
      time.sleep(5)
  except Exception:
    cloudlog.exception(f"manage_{process_name}.exception")
  finally:
    params.remove(pid_param)


def main():
  manage("openpilot.system.athena.athenad", "athenad", ATHENA_MGR_PID_PARAM)


if __name__ == '__main__':
  main()
