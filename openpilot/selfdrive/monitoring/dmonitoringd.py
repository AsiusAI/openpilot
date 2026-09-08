#!/usr/bin/env python3
import os

import openpilot.cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, DT_DMON, Ratekeeper
from openpilot.selfdrive.monitoring.policy import DriverMonitoring


NO_DCAM = os.getenv("NO_DCAM") == "1"


def get_no_dcam_state():
  dat = messaging.new_message('driverMonitoringState', valid=True)
  dat.driverMonitoringState.visionPolicyState.awarenessPercent = 100
  dat.driverMonitoringState.wheeltouchPolicyState.awarenessPercent = 100
  return dat


def no_dcam_thread():
  pm = messaging.PubMaster(['driverMonitoringState'])
  rk = Ratekeeper(1 / DT_DMON, print_delay_threshold=None)

  while True:
    pm.send('driverMonitoringState', get_no_dcam_state())
    rk.keep_time()


def dmonitoringd_thread():
  config_realtime_process([0, 1, 2, 3], 5)

  if NO_DCAM:
    no_dcam_thread()
    return

  params = Params()
  pm = messaging.PubMaster(['driverMonitoringState'])
  sm = messaging.SubMaster(['driverStateV2', 'extrinsicsCalibration', 'carState', 'selfdriveState', 'modelV2'], poll='driverStateV2')

  DM = DriverMonitoring(rhd_saved=params.get_bool("IsRhdDetected"), always_on=params.get_bool("AlwaysOnDM"))
  demo_mode=False

  # 20Hz <- dmonitoringmodeld
  while True:
    sm.update()
    if not sm.updated['driverStateV2']:
      # iterate when model has new output
      continue

    valid = sm.all_checks()
    if demo_mode and sm.valid['driverStateV2']:
      DM.run_step(sm, demo=True)
    elif valid:
      DM.run_step(sm, demo=demo_mode)

    # publish
    dat = DM.get_state_packet(valid=valid)
    pm.send('driverMonitoringState', dat)

    # load live always-on toggle
    if sm['driverStateV2'].frameId % 40 == 1:
      DM.always_on = params.get_bool("AlwaysOnDM")
      demo_mode = params.get_bool("IsDriverViewEnabled")

    # save rhd virtual toggle every 5 mins
    if (sm['driverStateV2'].frameId % 6000 == 0 and not demo_mode and
     DM.wheelpos_offsetter.filtered_stat.n > DM.settings._WHEELPOS_FILTER_MIN_COUNT and
     DM.wheel_on_right == (DM.wheelpos_offsetter.filtered_stat.M > DM.settings._WHEELPOS_THRESHOLD)):
      params.put_bool("IsRhdDetected", DM.wheel_on_right)

def main():
  dmonitoringd_thread()


if __name__ == '__main__':
  main()
