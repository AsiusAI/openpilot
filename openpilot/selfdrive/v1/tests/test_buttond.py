import os

from openpilot.selfdrive.v1.buttond import (
  EV_KEY,
  INPUT_EVENT,
  KEY_PRESSED,
  KEY_PROG1,
  KEY_RELEASED,
  ButtonAction,
  PandaButton,
)


def input_event(value: int) -> bytes:
  return INPUT_EVENT.pack(0, 0, EV_KEY, KEY_PROG1, value)


def test_pair_after_long_hold():
  read_fd, write_fd = os.pipe()
  try:
    button = PandaButton(event_device=f"/proc/self/fd/{read_fd}")
    os.write(write_fd, input_event(KEY_PRESSED))

    assert button.poll(now=1.0) == []
    assert button.poll(now=3.9) == []
    assert button.poll(now=4.0) == [ButtonAction.PAIR]
    assert button.poll(now=5.0) == []

    os.write(write_fd, input_event(KEY_RELEASED))
    assert button.poll(now=5.1) == []
  finally:
    button.close()
    os.close(read_fd)
    os.close(write_fd)


def test_short_press_does_not_pair():
  read_fd, write_fd = os.pipe()
  try:
    button = PandaButton(event_device=f"/proc/self/fd/{read_fd}")
    os.write(write_fd, input_event(KEY_PRESSED))
    assert button.poll(now=1.0) == []

    os.write(write_fd, input_event(KEY_RELEASED))
    assert button.poll(now=2.0) == []
    assert button.poll(now=5.0) == []
  finally:
    button.close()
    os.close(read_fd)
    os.close(write_fd)
