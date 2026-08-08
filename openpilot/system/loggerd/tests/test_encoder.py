#!/usr/bin/env python3

import math
import os
import shutil
import subprocess
import time
import unittest
from pathlib import Path

import numpy as np
from tqdm import trange

from openpilot.common.test import OpenpilotTestCase
from openpilot.common.hardware import ASIUS_HARDWARE
from openpilot.common.params import Params
from openpilot.common.timeout import Timeout
from openpilot.system.manager.process_config import managed_processes
from openpilot.tools.lib.logreader import LogReader
from openpilot.common.hardware.hw import Paths

SEGMENT_LENGTH = 2
FULL_SIZE = 2507572
def hevc_size(w): return FULL_SIZE // 2 if w <= 1344 else FULL_SIZE
CAMERAS = [
  ("fcamera.hevc", 20, hevc_size, "narrowRoadEncodeIdx"),
  ("dcamera.hevc", 20, hevc_size, "cabinEncodeIdx"),
  ("ecamera.hevc", 20, hevc_size, "wideRoadEncodeIdx"),
]
if not ASIUS_HARDWARE:
  CAMERAS.append(("qcamera.ts", 20, lambda x: 130000, None))
CAMERAD_PROCESS = "camerad"
ENCODERD_PROCESS = "encoderd"
WARMUP_SEGMENTS = 1

# we check frame count, so we don't have to be too strict on size
FILE_SIZE_TOLERANCE = 0.9


class TestEncoder(OpenpilotTestCase):
  COMMA_HARDWARE_TEST = True

  def setup_method(self):
    self._clear_logs()
    os.environ["LOGGERD_TEST"] = "1"
    os.environ["LOGGERD_SEGMENT_LENGTH"] = str(SEGMENT_LENGTH)

  def teardown_method(self):
    self._clear_logs()

  def _clear_logs(self):
    if os.path.exists(Paths.log_root()):
      shutil.rmtree(Paths.log_root())

  def _get_latest_segment_path(self):
    last_route = sorted(Path(Paths.log_root()).iterdir())[-1]
    return os.path.join(Paths.log_root(), last_route)

  # TODO: this should run faster than real time
  def test_log_rotation(self):
    Params().put_bool("RecordFront", True, block=True)

    managed_processes['sensord'].start()
    managed_processes['loggerd'].start()
    managed_processes[ENCODERD_PROCESS].start()

    time.sleep(1.0)
    managed_processes[CAMERAD_PROCESS].start()

    num_segments = 3 + WARMUP_SEGMENTS

    # wait for loggerd to make the dir for first segment
    route_prefix_path = None
    with Timeout(int(SEGMENT_LENGTH*3)):
      while route_prefix_path is None:
        try:
          route_prefix_path = self._get_latest_segment_path().rsplit("--", 1)[0]
        except Exception:
          time.sleep(0.1)

    def check_seg(i):
      # check each camera file size
      for camera, fps, size_lambda, encode_idx_name in CAMERAS:
        file_path = f"{route_prefix_path}--{i}/{camera}"

        # check file exists
        assert os.path.exists(file_path), f"segment #{i}: '{file_path}' missing"

        # TODO: this ffprobe call is really slow
        # get width and check frame count
        cmd = f"ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets,width -of csv=p=0 {file_path}"
        expected_frames = fps * SEGMENT_LENGTH
        probe = subprocess.check_output(cmd, shell=True, encoding='utf8').split('\n')[0].strip().split(',')
        frame_width, frame_count = int(probe[0]), int(probe[1])
        min_frames = expected_frames - fps
        assert min_frames <= frame_count <= expected_frames, \
          f"segment #{i}: {camera} expected {min_frames}-{expected_frames} frames, got {frame_count}"

        # sanity check file size
        file_size = os.path.getsize(file_path)
        target_size = size_lambda(frame_width)
        assert math.isclose(file_size, target_size, rel_tol=FILE_SIZE_TOLERANCE), \
                        f"{file_path} size {file_size} isn't close to target size {target_size}"

        # Check encodeIdx
        if encode_idx_name is not None:
          rlog_path = f"{route_prefix_path}--{i}/rlog.zst"
          msgs = [m for m in LogReader(rlog_path) if m.which() == encode_idx_name]
          encode_msgs = [getattr(m, encode_idx_name) for m in msgs]

          valid = [m.valid for m in msgs]
          segment_idxs = [m.segmentId for m in encode_msgs]
          encode_idxs = [m.encodeId for m in encode_msgs]
          # Check frame count
          assert frame_count == len(segment_idxs)
          assert frame_count == len(encode_idxs)

          # Check for duplicates or skips
          assert 0 <= segment_idxs[0] < fps
          assert len(set(segment_idxs)) == len(segment_idxs)
          assert set(np.diff(segment_idxs)) == {1, }

          assert all(valid)

          assert len(set(encode_idxs)) == len(encode_idxs)
          assert set(np.diff(encode_idxs)) == {1, }
      shutil.rmtree(f"{route_prefix_path}--{i}")

    rotation_start = time.monotonic()
    try:
      for i in trange(num_segments):
        # poll for next segment
        with Timeout(int(SEGMENT_LENGTH*10), error_msg=f"timed out waiting for segment {i}"):
          while Path(f"{route_prefix_path}--{i+1}") not in Path(Paths.log_root()).iterdir():
            time.sleep(0.1)
        if i < WARMUP_SEGMENTS:
          shutil.rmtree(f"{route_prefix_path}--{i}")
        else:
          check_seg(i)
      elapsed = time.monotonic() - rotation_start
      assert elapsed < SEGMENT_LENGTH * (num_segments + 2), f"encoder rotation took {elapsed:.1f}s"
    finally:
      managed_processes['loggerd'].stop()
      managed_processes[ENCODERD_PROCESS].stop()
      managed_processes[CAMERAD_PROCESS].stop()
      managed_processes['sensord'].stop()


if __name__ == "__main__":
  unittest.main()
