from pathlib import Path
import tempfile
from types import SimpleNamespace

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.ui import ledd
from openpilot.selfdrive.ui.ledd import CAM_LED_BUSES, CAM_LED_CHANNELS, READY, CameraLeds, LedState, max_brightness


def tmp_path():
  with tempfile.TemporaryDirectory() as directory:
    yield Path(directory)


class TestLedd(OpenpilotTestCase):
  def test_all_camera_led_buses_are_configured(self):
    leds = CameraLeds()
    assert CAM_LED_BUSES == (16, 18, 20)
    assert [(board.name, board.camera_num, board.bus_num) for board in leds.boards] == [
      ("driver", 1, 16),
      ("road", 2, 18),
      ("wide", 3, 20),
    ]

  def test_status_colors_scale_to_requested_brightness(self):
    assert max_brightness(LedState("test", 0, 90, 180)) == LedState("test", 0, 127, 255)
    assert max_brightness(LedState("test", 0, 90, 180), 204) == LedState("test", 0, 102, 204)

  def test_userspace_controls_every_kernel_led_channel(self, tmp_path, monkeypatch):
    monkeypatch.setattr(ledd, "CAM_LED_SYSFS_ROOT", tmp_path)
    camera_values = {}
    expected = {}
    for camera_num in (1, 2, 3):
      values = []
      for channel, (color, package) in enumerate(CAM_LED_CHANNELS):
        path = tmp_path / f"asius:cam{camera_num}:{color}:{package}"
        path.mkdir()
        (path / "brightness").write_text("0")
        (path / "trigger").write_text("[asius-boot] timer")
        value = (camera_num * 50 + channel * 7) % 256
        expected[path] = value
        values.append(value)
      camera_values[camera_num] = values

    leds = CameraLeds()
    leds.set_channels(camera_values)

    for path, value in expected.items():
      assert (path / "trigger").read_text() == "none"
      assert (path / "brightness").read_text() == str(value)

  def test_kernel_boot_blink_runs_until_first_openpilot_state(self, tmp_path, monkeypatch):
    monkeypatch.setattr(ledd, "CAM_LED_SYSFS_ROOT", tmp_path)
    channel_paths = []
    for camera_num in (1, 2, 3):
      for color, package in CAM_LED_CHANNELS:
        path = tmp_path / f"asius:cam{camera_num}:{color}:{package}"
        path.mkdir()
        (path / "brightness").write_text("0")
        (path / "trigger").write_text("[asius-boot]")
        channel_paths.append((path, color))

    leds = CameraLeds()
    assert all((path / "trigger").read_text() == "[asius-boot]" for path, _ in channel_paths)

    leds.set(READY, force=True)

    for path, color in channel_paths:
      assert (path / "trigger").read_text() == "none"
      expected = "255" if "cam1" not in path.name and color == "blue" else "0"
      assert (path / "brightness").read_text() == expected

  def test_status_policy_keeps_cam1_off_but_explicit_channels_can_enable_it(self, monkeypatch):
    leds = CameraLeds()
    states = {}
    for board in leds.boards:
      monkeypatch.setattr(board, "set", lambda state, camera_num=board.camera_num: states.update({camera_num: state}))

    leds.set(READY, force=True)
    assert states[1] == ledd.OFF
    assert states[2] == LedState("ready", 0, 0, 255)
    assert states[3] == LedState("ready", 0, 0, 255)

    channels = [255, 0, 0] * 3
    monkeypatch.setattr(leds.boards[0], "set_channels", lambda values: states.update({1: values}))
    leds.set_channels({1: channels}, force=True)
    assert states[1] == channels

  def test_missing_camera_boards_do_not_stop_ledd(self, monkeypatch):
    for missing_count in (1, 3):
      with self.subTest(missing_count=missing_count):
        leds = CameraLeds()
        updated = []

        def fail(_state):
          raise OSError("camera board missing")

        monkeypatch.setattr(ledd, "cloudlog_exception", lambda _message: None)
        for board in leds.boards[:missing_count]:
          monkeypatch.setattr(board, "set", fail)
        for board in leds.boards[missing_count:]:
          monkeypatch.setattr(board, "set", lambda _state, name=board.name, updates=updated: updates.append(name))

        leds.set(LedState("test", 0, 0, 180), force=True)
        assert updated == [board.name for board in leds.boards[missing_count:]]

  def test_pairing_feedback_exclusively_controls_all_camera_leds(self, monkeypatch):
    monkeypatch.setattr(ledd, "get_ble_pairing", lambda: None)
    monkeypatch.setattr(ledd, "pairing_mode_active", lambda: True)
    monkeypatch.setattr(ledd.time, "monotonic", lambda: 10.0)
    channels = ledd.pairing_led_channels()
    assert channels == {
      1: [0] * 9,
      2: [0, 255, 80] * 3,
      3: [0, 255, 80] * 3,
    }

    monkeypatch.setattr(ledd.time, "monotonic", lambda: 10.5)
    assert ledd.pairing_led_channels() == {1: [0] * 9, 2: [0] * 9, 3: [0] * 9}

  def test_pairing_color_code_uses_six_distinct_leds_on_cam2_and_cam3(self, monkeypatch):
    monkeypatch.setattr(ledd, "get_ble_pairing", lambda: {"colors": ["red", "green", "blue", "amber", "turquoise", "violet"]})
    monkeypatch.setattr(ledd, "pairing_mode_active", lambda: False)
    assert ledd.pairing_led_channels() == {
      1: [0] * 9,
      2: [255, 0, 0, 0, 255, 0, 0, 0, 255],
      3: [255, 80, 0, 0, 255, 80, 150, 0, 255],
    }

  def test_camera_led_brightness_matches_openpilot_screen_mapping(self):
    cases = [(100., 76), (50., 109), (0., 255)]
    for exposure_percent, expected in cases:
      with self.subTest(exposure_percent=exposure_percent):
        class SubMaster:
          seen = {"wideRoadCameraState": True, "roadCameraState": False}
          alive = {"wideRoadCameraState": True, "roadCameraState": False}
          valid = {"wideRoadCameraState": True, "roadCameraState": False}

          def __init__(self, exposure):
            self.exposure = exposure

          def __getitem__(self, name):
            assert name == "wideRoadCameraState"
            return SimpleNamespace(exposureValPercent=self.exposure)

        assert ledd.camera_led_brightness(SubMaster(exposure_percent)) == expected

  def test_camera_led_brightness_falls_back_to_road_then_startup(self):
    class SubMaster:
      seen = {"wideRoadCameraState": False, "roadCameraState": True}
      alive = {"wideRoadCameraState": False, "roadCameraState": True}
      valid = {"wideRoadCameraState": False, "roadCameraState": True}

      def __getitem__(self, name):
        assert name == "roadCameraState"
        return SimpleNamespace(exposureValPercent=0.)

    sm = SubMaster()
    assert ledd.camera_led_brightness(sm) == 255
    sm.seen["roadCameraState"] = False
    assert ledd.camera_led_brightness(sm) == ledd.CAM_LED_STARTUP_BRIGHTNESS

  def test_update_led_blinks_violet_for_active_update_phases(self):
    assert ledd.updater_led_state("idle", now=10.) is None
    assert ledd.updater_led_state("checking...", now=10.) is None
    assert ledd.updater_led_state("downloading...", now=10.) == ledd.UPDATING
    assert ledd.updater_led_state("finalizing update...", now=10.5) == ledd.OFF

  def test_disabled_pandad_is_not_disconnected(self):
    process = SimpleNamespace(name="pandad", shouldBeRunning=False)

    class SubMaster:
      seen = {"managerState": True, "pandaStates": False}
      alive = {"pandaStates": False}

      def __getitem__(self, name):
        assert name == "managerState"
        return SimpleNamespace(processes=[process])

    assert not ledd.panda_disconnected(SubMaster())

  def test_enabled_pandad_without_state_is_disconnected(self):
    process = SimpleNamespace(name="pandad", shouldBeRunning=True)

    class SubMaster:
      seen = {"managerState": True, "pandaStates": False}
      alive = {"pandaStates": False}

      def __getitem__(self, name):
        assert name == "managerState"
        return SimpleNamespace(processes=[process])

    assert ledd.panda_disconnected(SubMaster())
