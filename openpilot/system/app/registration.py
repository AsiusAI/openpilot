#!/usr/bin/env python3
from openpilot.common.params import Params
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from openpilot.common.hardware import PC
from openpilot.common.swaglog import cloudlog
from openpilot.system.app.identity import get_or_create_device_identity


UNREGISTERED_DONGLE_ID = "UnregisteredDevice"

def register(show_spinner=False) -> str | None:
  params = Params()
  try:
    dongle_id = get_or_create_device_identity()
  except Exception:
    dongle_id = UNREGISTERED_DONGLE_ID
    cloudlog.exception("failed to create Ed25519 identity")

  if dongle_id:
    params.put("DongleId", dongle_id, block=True)
    set_offroad_alert("Offroad_UnregisteredHardware", (dongle_id == UNREGISTERED_DONGLE_ID) and not PC)
  return dongle_id


if __name__ == "__main__":
  print(register())
