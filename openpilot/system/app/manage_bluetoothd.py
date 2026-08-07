#!/usr/bin/env python3

from openpilot.system.athena.manage_athenad import manage


def main() -> None:
  manage("openpilot.system.app.bluetoothd", "bluetoothd", "BluetoothdPid")


if __name__ == "__main__":
  main()
