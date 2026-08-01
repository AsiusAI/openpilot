import os

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.ui.buttond import (
  ButtonAction,
  EV_KEY,
  INPUT_EVENT,
  KEY_PRESSED,
  KEY_PROG1,
  KEY_RELEASED,
  PowerButton,
)


def write_event(fd: int, value: int, code: int = KEY_PROG1) -> None:
  os.write(fd, INPUT_EVENT.pack(0, 0, EV_KEY, code, value))


class TestButtond(OpenpilotTestCase):
  def test_short_press_is_ignored(self):
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    try:
      button = PowerButton(fd=read_fd)
      write_event(write_fd, KEY_PRESSED)
      assert button.poll(now=10.0) == []
      write_event(write_fd, KEY_RELEASED)
      assert button.poll(now=11.0) == []
    finally:
      os.close(write_fd)
      button.close()

  def test_three_second_hold_pairs_once(self):
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    try:
      button = PowerButton(fd=read_fd)
      write_event(write_fd, KEY_PRESSED)
      assert button.poll(now=10.0) == []
      assert button.poll(now=12.99) == []
      assert button.poll(now=13.0) == [ButtonAction.PAIR]
      assert button.poll(now=14.0) == []
      write_event(write_fd, KEY_RELEASED)
      assert button.poll(now=14.1) == []
    finally:
      os.close(write_fd)
      button.close()

  def test_power_key_code_is_ignored(self):
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    try:
      button = PowerButton(fd=read_fd)
      write_event(write_fd, KEY_PRESSED, code=116)
      write_event(write_fd, KEY_RELEASED, code=116)
      assert button.poll(now=10.0) == []
    finally:
      os.close(write_fd)
      button.close()
