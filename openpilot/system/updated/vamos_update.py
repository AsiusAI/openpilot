import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert

VAMOS_UPDATE = Path("/usr/bin/vamos-update")


def vamos_update_supported() -> bool:
  return VAMOS_UPDATE.is_file()


def run_vamos_update(cmd: list[str]) -> str:
  params = Params()
  output: list[str] = []
  progress = 0
  process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf8")
  assert process.stdout is not None

  for line in process.stdout:
    output.append(line)
    cloudlog.info(line.rstrip())
    match = re.search(r"vamos-update: (system|esp): (\d+)%", line)
    if match is not None:
      image, image_progress = match.group(1), int(match.group(2))
      estimated = int(image_progress * 0.45) if image == "system" else 90 + int(image_progress * 0.05)
      progress = max(progress, estimated)
    elif "vamos-update: verifying system from disk" in line:
      progress = max(progress, 45)
    elif "vamos-update: writing esp " in line:
      progress = max(progress, 90)
    elif "vamos-update: verifying esp from disk" in line:
      progress = max(progress, 95)
    params.put("UpdaterProgress", progress, block=True)

  returncode = process.wait()
  result = "".join(output)
  if returncode != 0:
    raise subprocess.CalledProcessError(returncode, cmd, output=result)
  params.put("UpdaterProgress", 98, block=True)
  return result


def prepare_vamos_update(overlay_merged: str, current_version: str,
                         set_consistent_flag: Callable[[bool], None]) -> bool:
  manifest_path = Path(overlay_merged) / "openpilot/system/hardware/v1/vamos.json"
  with manifest_path.open() as manifest_file:
    updated_version = str(json.load(manifest_file)["version"])

  cloudlog.info(f"vamOS version check: {current_version} vs {updated_version}")
  if current_version == updated_version:
    return False

  # Keep the openpilot overlay unbootable until both inactive-slot images have
  # been written and verified. Trial activation happens only after finalization.
  set_consistent_flag(False)
  cloudlog.info(f"Beginning background installation for vamOS {updated_version}")
  set_offroad_alert("Offroad_NeosUpdate", True)

  try:
    run_vamos_update(["sudo", "/usr/bin/vamos-update", "install", str(manifest_path), "--defer-activation"])
  except Exception:
    set_offroad_alert("Offroad_NeosUpdate", False)
    raise
  return True


def activate_vamos_update() -> None:
  try:
    subprocess.check_output(["sudo", "/usr/bin/vamos-update", "activate"], stderr=subprocess.STDOUT, encoding="utf8")
  finally:
    set_offroad_alert("Offroad_NeosUpdate", False)


def should_skip_noop_vamos_fetch(update_available: bool, user_request: int, fetch_request: int) -> bool:
  return vamos_update_supported() and not update_available and user_request != fetch_request
