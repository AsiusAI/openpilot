#!/usr/bin/env python3

import time
import unittest
import numpy as np

from openpilot.common.parameterized import parameterized
from openpilot.common.hardware import ASIUS_HARDWARE
from openpilot.common.test import OpenpilotTestCase
from openpilot.cereal.services import SERVICE_LIST
from openpilot.tools.lib.log_time_series import msgs_to_time_series
from openpilot.system.camerad.snapshot import get_snapshots
from openpilot.selfdrive.test.helpers import collect_logs, log_collector, processes_context

TEST_TIMESPAN = 10
CAMERAS = ('narrowRoadCameraState', 'cabinCameraState', 'wideRoadCameraState')
EXPOSURE_STABLE_COUNT = 3
EXPOSURE_RANGE = (0.10, 0.65)
MAX_TEST_TIME = 25
STARTUP_FRAME_IGNORE = 10
MAX_ISOLATED_FRAME_DROPS = 2


def _numpy_rgb2gray(im):
  return np.clip(im[:,:,2] * 0.114 + im[:,:,1] * 0.587 + im[:,:,0] * 0.299, 0, 255).astype(np.uint8)

def _exposure_stats(im):
  h, w = im.shape[:2]
  gray = _numpy_rgb2gray(im[h//10:9*h//10, w//10:9*w//10])
  return float(np.median(gray) / 255.), float(np.mean(gray) / 255.)

def _in_range(median, mean):
  lo, hi = EXPOSURE_RANGE
  return lo < median < hi and lo < mean < hi

def _exposure_stable(results):
  return all(
    len(v) >= EXPOSURE_STABLE_COUNT and all(_in_range(*s) for s in v[-EXPOSURE_STABLE_COUNT:])
    for v in results.values()
  )

def run_and_log(procs, services, duration):
  with processes_context(procs):
    return collect_logs(services, duration)

def _camera_session():
  """Single camerad session that collects logs and exposure data.
     Runs until exposure stabilizes (min TEST_TIMESPAN seconds for enough log data)."""
  with processes_context(["camerad"]), log_collector(CAMERAS) as (raw_logs, lock):
    exposure = {cam: [] for cam in CAMERAS}
    start = time.monotonic()
    while time.monotonic() - start < MAX_TEST_TIME:
      rpic, dpic = get_snapshots(frame="narrowRoadCameraState", front_frame="cabinCameraState")
      wpic, _ = get_snapshots(frame="wideRoadCameraState")
      for cam, img in zip(CAMERAS, [rpic, dpic, wpic], strict=True):
        exposure[cam].append(_exposure_stats(img))

      if time.monotonic() - start >= TEST_TIMESPAN and _exposure_stable(exposure):
        break

    elapsed = time.monotonic() - start

  with lock:
    ts = msgs_to_time_series(raw_logs)

  for cam in CAMERAS:
    expected_frames = SERVICE_LIST[cam].frequency * elapsed
    cnt = len(ts[cam]['t'])
    assert expected_frames*0.8 < cnt < expected_frames*1.2, f"unexpected frame count {cam}: {expected_frames=}, got {cnt}"

    timestamps = ts[cam]['timestampSof'][STARTUP_FRAME_IGNORE:] / 1e6
    dts = np.abs(np.diff(timestamps) - 1000/SERVICE_LIST[cam].frequency)
    bad_intervals = np.count_nonzero(dts >= 1.0)
    assert bad_intervals <= MAX_ISOLATED_FRAME_DROPS, \
      f"{cam} dts(ms) out of spec: {bad_intervals} bad intervals, " \
      f"max diff {dts.max()}, 99 percentile {np.percentile(dts, 99)}"
    assert dts.max() < 1000/SERVICE_LIST[cam].frequency + 1.0, f"{cam} dropped consecutive frames"

  return ts, exposure

class TestCamerad(OpenpilotTestCase):
  COMMA_HARDWARE_TEST = True

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.logs, cls.exposure_data = _camera_session()

  @parameterized.expand(CAMERAS, names=("cam",))
  def test_camera_exposure(self, cam):
    lo, hi = EXPOSURE_RANGE
    checks = self.exposure_data[cam]
    assert len(checks) >= EXPOSURE_STABLE_COUNT, f"{cam}: only got {len(checks)} samples"

    # check that exposure converges into the valid range
    passed = sum(_in_range(med, mean) for med, mean in checks)
    assert passed >= EXPOSURE_STABLE_COUNT, \
      f"{cam}: only {passed}/{len(checks)} checks in range. " + \
      " | ".join(f"#{i+1}: med={m:.4f} mean={u:.4f}" for i, (m, u) in enumerate(checks))

    # check that exposure is stable once converged (no regressions)
    in_range = False
    for i, (median, mean) in enumerate(checks):
      ok = _in_range(median, mean)
      if in_range and not ok:
        self.fail(f"{cam}: exposure regressed on sample {i+1} " +
                    f"(median={median:.4f}, mean={mean:.4f}, expected: ({lo}, {hi}))")
      in_range = ok

  def test_frame_skips(self):
    for c in CAMERAS:
      frame_diffs = np.diff(self.logs[c]['frameId'][STARTUP_FRAME_IGNORE:] if ASIUS_HARDWARE else self.logs[c]['frameId'])
      if ASIUS_HARDWARE:
        assert frame_diffs.min() == 1, f"{c} has duplicate or decreasing frame IDs"
        assert frame_diffs.max() <= 2, f"{c} dropped consecutive frames"
        dropped_frames = np.count_nonzero(frame_diffs != 1)
        assert dropped_frames <= MAX_ISOLATED_FRAME_DROPS, f"{c} dropped too many frames: {dropped_frames}"
      else:
        assert set(frame_diffs) == {1, }, f"{c} has frame skips"

  def test_frame_sync(self):
    if ASIUS_HARDWARE:
      self.skipTest("Asius cameras are not hardware-synchronized")

    SYNCED_CAMS = ('narrowRoadCameraState', 'wideRoadCameraState')
    n = range(len(self.logs['narrowRoadCameraState']['t'][:-10]))

    frame_ids = {i: [self.logs[cam]['frameId'][i] for cam in CAMERAS] for i in n}
    assert all(len(set(v)) == 1 for v in frame_ids.values()), "frame IDs not aligned"

    # road and wide cameras should be synced within 1.1ms
    synced_times = {i: [self.logs[cam]['timestampSof'][i] for cam in SYNCED_CAMS] for i in n}
    diffs = {i: (max(ts) - min(ts))/1e6 for i, ts in synced_times.items()}
    laggy_frames = {k: v for k, v in diffs.items() if v > 1.1}
    assert len(laggy_frames) == 0, f"Frames not synced properly: {laggy_frames=}"

    # cabin camera should be staggered ~25ms from road camera
    for i in n:
      offset_ms = abs(self.logs['cabinCameraState']['timestampSof'][i] - self.logs['narrowRoadCameraState']['timestampSof'][i]) / 1e6
      assert 20 < offset_ms < 30, f"cabin camera stagger out of range at frame {i}: {offset_ms:.1f}ms (expected ~25ms)"

  def test_sanity_checks(self):
    self._sanity_checks(self.logs)

  def _sanity_checks(self, ts):
    for c in CAMERAS:
      assert c in ts
      assert len(ts[c]['t']) > 20

      # should monotonically increase
      assert np.all(np.diff(ts[c]['frameId']) >= 1)
      assert 0 not in ts[c]['requestId']
      assert np.all(np.diff(ts[c]['requestId']) >= 1)

      # EOF > SOF
      assert np.all((ts[c]['timestampEof'] - ts[c]['timestampSof']) > 0)

      # logMonoTime > SOF
      assert np.all((ts[c]['t'] - ts[c]['timestampSof']/1e9) > 1e-7)

      # EOF timestamps can be reconstructed from SOF or supplied directly by the camera driver.
      assert np.all((ts[c]['t'] - ts[c]['timestampEof']/1e9) > -0.10)        # when EOF > logMonoTime, it should never be more than two frames

if __name__ == "__main__":
  unittest.main()
