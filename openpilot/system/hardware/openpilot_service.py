#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess


def service_command(action: str) -> list[str]:
  if shutil.which("systemctl") is not None:
    return ["systemctl", action, "comma.service"]
  if shutil.which("sv") is not None:
    return ["sv", "down" if action == "stop" else "up", "openpilot"]
  raise RuntimeError("no supported openpilot service manager found")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("action", choices=("start", "stop"))
  args = parser.parse_args()
  command = service_command(args.action)
  if os.geteuid() != 0:
    command.insert(0, "sudo")
  subprocess.run(command, check=True)


if __name__ == "__main__":
  main()
