import os
from typing import cast

from openpilot.common.hardware.base import HardwareBase
from openpilot.common.hardware.asius.hardware import HardwareAsius
from openpilot.common.hardware.comma.hardware import HardwareComma
from openpilot.common.hardware.pc.hardware import HardwarePc

AGNOS = os.path.isfile('/AGNOS')
COMMA_HARDWARE = AGNOS
ASIUS_HARDWARE = os.path.isfile('/ASIUS')
PC = not (COMMA_HARDWARE or ASIUS_HARDWARE)

if ASIUS_HARDWARE:
  HARDWARE = cast(HardwareBase, HardwareAsius())
elif COMMA_HARDWARE:
  HARDWARE = cast(HardwareBase, HardwareComma())
else:
  HARDWARE = cast(HardwareBase, HardwarePc())
