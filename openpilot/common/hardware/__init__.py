import os
from typing import cast

from openpilot.common.hardware.base import HardwareBase
from openpilot.common.hardware.v1.hardware import Asius
from openpilot.common.hardware.comma.hardware import HardwareComma
from openpilot.common.hardware.pc.hardware import HardwarePc

AGNOS = os.path.isfile('/AGNOS')
COMMA_HARDWARE = AGNOS
ASIUS = os.path.isfile('/ASIUS')
ASIUS_HARDWARE = ASIUS
V1 = ASIUS_HARDWARE
PC = not (COMMA_HARDWARE or ASIUS_HARDWARE)

if ASIUS_HARDWARE:
  HARDWARE = cast(HardwareBase, Asius())
elif COMMA_HARDWARE:
  HARDWARE = cast(HardwareBase, HardwareComma())
else:
  HARDWARE = cast(HardwareBase, HardwarePc())
DEVICE_TYPE = HARDWARE.get_device_type()
