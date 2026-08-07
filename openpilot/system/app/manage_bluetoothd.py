#!/usr/bin/env python3

from openpilot.system.app.manager import manage


def main() -> None:
  manage("openpilot.system.app.bluetoothd", "bluetoothd", "BluetoothdPid")


if __name__ == "__main__":
  main()
