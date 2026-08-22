from openpilot.cereal import log
from openpilot.selfdrive.monitoring.dmonitoringd import get_no_dcam_state


def test_no_dcam_state_is_valid_and_neutral():
  msg = get_no_dcam_state()
  state = msg.driverMonitoringState

  assert msg.valid
  assert state.alertLevel == log.DriverMonitoringState.AlertLevel.none
  assert not state.lockout
  assert not state.alwaysOnLockout
  assert not state.noResponseForceDecel
  assert state.visionPolicyState.awarenessPercent == 100
  assert state.wheeltouchPolicyState.awarenessPercent == 100
