#!/usr/bin/env python3
import ctypes
import os
from typing import cast

os.environ['GMMU'] = '0' # for usbgpu fast loading, noop for qcom
from tinygrad.device import Buffer, Device, TinyELF
from tinygrad.dtype import dtypes
from tinygrad.runtime.autogen import opencl as cl
from tinygrad.tensor import Tensor
import threading
import time
import numpy as np
import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.cereal.messaging import PubMaster, SubMaster
from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from opendbc.car.car_helpers import get_demo_car_params
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, should_stop, smooth_value, get_curvature_from_plan
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.compile_modeld_v1 import make_input_queues, WARP_INPUTS, POLICY_INPUTS
from openpilot.selfdrive.modeld.fill_model_msg import fill_model_msg, fill_driving_model_data, fill_pose_msg, PublishState
from openpilot.common.file_chunker import open_file_chunked
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.helpers import usbgpu_present, usbgpu_compiled, modeld_pkl_path, get_tg_input_devices, load_oob

PROCESS_NAME = "openpilot.selfdrive.modeld.modeld"
SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')

LAT_SMOOTH_SECONDS = 0.1
LONG_SMOOTH_SECONDS = 0.3
MIN_LAT_CONTROL_SPEED = 0.3
BIG_MODEL_TIMEOUT = 60

CL_MEM_EXT_HOST_PTR_QCOM, CL_MEM_HOST_WRITEBACK_QCOM, CL_MEM_ION_HOST_PTR_QCOM = 1 << 29, 0x40A5, 0x40A8

FAST_CL_WARP_SOURCE = r"""
static inline int clip_round(float value, int high) { return min(max((int)rint(value), 0), high); }
__kernel void modeld_warp_nv12(
    __global uchar *out_img, __global uchar *out_big,
    __global uchar *ring_img, __global uchar *ring_big,
    const __global uchar *frame, const __global uchar *big_frame,
    const __global float *transforms, const int slot) {
  const int global_id = get_global_id(0);
  if (global_id >= 2 * FRAME_ELEMS) return;
  const int is_big = global_id >= FRAME_ELEMS;
  const int gid = global_id - is_big * FRAME_ELEMS;
  const int pos = gid % FRAME_PIX, chan = gid / FRAME_PIX;
  const int y = pos / HALF_W, x = pos - y * HALF_W;
  const __global uchar *src = is_big ? big_frame : frame;
  const __global float *matrix = transforms + is_big * 9;
  __global uchar *out = is_big ? out_big : out_img, *ring = is_big ? ring_big : ring_img;
  uchar value;
  if (chan < 4) {
    const int dx = x * 2 + (chan >= 2), dy = y * 2 + ((chan == 1) || (chan == 3));
    const float scale = matrix[6] * dx + matrix[7] * dy + matrix[8];
    const int sx = clip_round((matrix[0] * dx + matrix[1] * dy + matrix[2]) / scale, CAM_W - 1);
    const int sy = clip_round((matrix[3] * dx + matrix[4] * dy + matrix[5]) / scale, CAM_H - 1);
    value = src[sy * STRIDE + sx];
  } else {
    const float scale = matrix[6] * (x * 2) + matrix[7] * (y * 2) + matrix[8];
    const int sx = clip_round((matrix[0] * x + matrix[1] * y + matrix[2] * .5f) / scale, CAM_W / 2 - 1);
    const int sy = clip_round((matrix[3] * x + matrix[4] * y + matrix[5] * .5f) / scale, CAM_H / 2 - 1);
    value = src[STRIDE * Y_HEIGHT + sy * STRIDE + sx * 2 + chan - 4];
  }
  ring[slot * FRAME_ELEMS + gid] = value;
  out[chan * FRAME_PIX + pos] = ring[((slot + 1) % RING_SLOTS) * FRAME_ELEMS + gid];
  out[(chan + 6) * FRAME_PIX + pos] = value;
}
"""


class _QcomExtHostPtr(ctypes.Structure):
  _fields_ = [('allocation_type', ctypes.c_uint32), ('host_cache_policy', ctypes.c_uint32)]


class _QcomIonHostPtr(ctypes.Structure):
  _fields_ = [('ext_host_ptr', _QcomExtHostPtr), ('ion_filedesc', ctypes.c_int), ('ion_hostptr', ctypes.c_void_p)]


class ImportedCLFrame:
  def __init__(self, fd: int, host_addr: int, size: int, device: str):
    from tinygrad.runtime.autogen import opencl as cl
    self.host_ptr = _QcomIonHostPtr(_QcomExtHostPtr(CL_MEM_ION_HOST_PTR_QCOM, CL_MEM_HOST_WRITEBACK_QCOM), fd, host_addr)
    dev, status = Device[device], ctypes.c_int32()
    cl_mem = cl.clCreateBuffer(dev.context, cl.CL_MEM_READ_ONLY | cl.CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
                               size, ctypes.cast(ctypes.pointer(self.host_ptr), ctypes.c_void_p), ctypes.byref(status))
    if status.value != 0:
      raise RuntimeError(f"failed to import VisionBuf fd into OpenCL: fd={fd} status={status.value}")
    self.tensor = Tensor.empty(size, dtype='uint8', device=device)
    cast(Buffer, self.tensor.uop.buffer).allocate(opaque=cl_mem)


class FastCLWarp:
  def __init__(self, device: str, cam_w: int, cam_h: int, stride: int, y_height: int,
               output_shape: tuple[int, ...], frame_skip: int):
    self.device = Device[device]
    self.outputs = [Tensor.empty(output_shape, dtype='uint8', device=device).realize() for _ in range(2)]
    self.transforms = Tensor.empty((2, 3, 3), dtype='float32', device=device).realize()
    self.transforms_np = np.empty((2, 3, 3), dtype=np.float32)
    self.ring_slots, self.write_slot = frame_skip + 1, 0
    half_h, half_w = output_shape[2:]
    frame_elems = 6 * half_h * half_w
    ring_shape = (self.ring_slots, 6, half_h, half_w)
    self.rings = [Tensor.zeros(ring_shape, dtype='uint8', device=device).contiguous().realize() for _ in range(2)]
    self.groups, self.local_size = ((2 * frame_elems + 255) // 256, 1, 1), (256, 1, 1)
    source, names = FAST_CL_WARP_SOURCE, ("FRAME_ELEMS", "FRAME_PIX", "RING_SLOTS", "HALF_W", "STRIDE", "Y_HEIGHT", "CAM_W", "CAM_H")
    for name, value in zip(names, (frame_elems, half_h * half_w, self.ring_slots, half_w, stride, y_height, cam_w, cam_h), strict=True):
      source = source.replace(name, str(value))
    signature = tuple((None, i, dtypes.float32 if i == 6 else dtypes.uint8, ()) for i in range(7))
    signature += ((None, 7, dtypes.int32, ()),)
    self.program = self.device.runtime(TinyELF(source.encode(), "modeld_warp_nv12", self.device.renderer.target, signature))

  def __call__(self, frame: Tensor, big_frame: Tensor, transforms: dict[str, np.ndarray]) -> tuple[Tensor, Tensor]:
    self.transforms_np[:] = transforms['img'], transforms['big_img']
    transforms_buffer = self.transforms._buffer()
    transforms_buffer.allocator._copyin(transforms_buffer._buf, memoryview(self.transforms_np).cast('B'))
    self.write_slot = (self.write_slot + 1) % self.ring_slots
    buffers = (*self.outputs, *self.rings, frame, big_frame, self.transforms)
    self.program(
      *(tensor._buffer()._buf for tensor in buffers),
      vals=(self.write_slot,), global_size=self.groups, local_size=self.local_size,
    )
    return self.outputs[0], self.outputs[1]


def get_action_from_model(model_output: dict[str, np.ndarray], prev_action: log.ModelDataV2.Action,
                          lat_action_t: float, long_action_t: float, v_ego: float) -> log.ModelDataV2.Action:
  if 'action' not in model_output:
    plan = model_output['plan'][0]
    desired_accel = get_accel_from_plan(plan[:,Plan.VELOCITY][:,0],
                                        plan[:,Plan.ACCELERATION][:,0],
                                        ModelConstants.T_IDXS,
                                        action_t=long_action_t)
    desired_curvature = get_curvature_from_plan(plan[:,Plan.T_FROM_CURRENT_EULER][:,2],
                                                plan[:,Plan.ORIENTATION_RATE][:,2],
                                                ModelConstants.T_IDXS,
                                                v_ego,
                                                lat_action_t)
  else:
    desired_accel = model_output['action'][0,1]
    desired_curvature = model_output['action'][0,0] / (max(1.0, v_ego))**2
  stop = should_stop(v_ego, desired_accel)
  desired_accel = smooth_value(desired_accel, prev_action.desiredAcceleration, LONG_SMOOTH_SECONDS)
  if v_ego > MIN_LAT_CONTROL_SPEED:
    desired_curvature = smooth_value(desired_curvature, prev_action.desiredCurvature, LAT_SMOOTH_SECONDS)
  else:
    desired_curvature = prev_action.desiredCurvature

  return log.ModelDataV2.Action(desiredCurvature=float(desired_curvature),
                                desiredAcceleration=float(desired_accel),
                                shouldStop=bool(stop))


class FrameMeta:
  frame_id: int = 0
  timestamp_sof: int = 0
  timestamp_eof: int = 0

  def __init__(self, vipc=None):
    if vipc is not None:
      self.frame_id, self.timestamp_sof, self.timestamp_eof = vipc.frame_id, vipc.timestamp_sof, vipc.timestamp_eof


class ModelState:
  prev_desire: np.ndarray  # for tracking the rising edge of the pulse

  def __init__(self, cam_w: int, cam_h: int, usbgpu: bool):
    input_devices = get_tg_input_devices(PROCESS_NAME, usbgpu)
    self.WARP_DEV, self.QUEUE_DEV = input_devices['WARP_DEV'], input_devices['QUEUE_DEV']
    jits = load_oob(open_file_chunked(modeld_pkl_path(usbgpu)))
    metadata = jits['metadata']
    self.input_shapes = metadata['input_shapes']
    self.vision_input_names = [k for k in self.input_shapes if 'img' in k]
    self.output_slices = metadata['output_slices']
    self.model_output: np.ndarray | None = None

    self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)

    self.frame_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
    self.input_queues, self.npy = make_input_queues(self.input_shapes, self.frame_skip, device=self.QUEUE_DEV)
    self.full_frames: dict[str, Tensor] = {}
    self._blob_cache: dict[tuple[str, int], Tensor] = {}
    self.parser = Parser(inplace=True)
    self.frame_buf_params = {k: get_nv12_info(cam_w, cam_h) for k in ('img', 'big_img')}
    stride, y_height, _, _ = self.frame_buf_params['img']
    self.frame_buf_used_size = stride * (y_height + cam_h // 2)
    self.run_policy = jits['run_policy']
    self.warp = jits[(cam_w,cam_h)]
    self.fast_cl_warp = jits.get('direct_images', False) and self.WARP_DEV == self.QUEUE_DEV and self.WARP_DEV.startswith("CL")
    if self.fast_cl_warp:
      self.cl_warp = FastCLWarp(self.WARP_DEV, cam_w, cam_h, stride, y_height, self.input_shapes['img'], self.frame_skip)
      self.cl_import_enabled, self.prewarped = True, None
      self.cl_imported_frames, self.cl_uploaded_frames = {}, {}
      warm_frames = {name: Tensor.zeros(self.input_shapes[name], dtype='uint8', device=self.WARP_DEV).realize() for name in self.vision_input_names}
      for _ in range(10):
        out, = self.run_policy(**{k: self.input_queues[k] for k in POLICY_INPUTS[2:] if k in self.input_queues}, **warm_frames)
        out.numpy()
      self.input_queues, self.npy = make_input_queues(self.input_shapes, self.frame_skip, device=self.QUEUE_DEV)

  def slice_outputs(self, model_outputs: np.ndarray, output_slices: dict[str, slice]) -> dict[str, np.ndarray]:
    parsed_model_outputs = {k: model_outputs[np.newaxis, v] for k,v in output_slices.items()}
    return parsed_model_outputs

  def preload_frames(self, bufs: dict[str, VisionBuf], transforms: dict[str, np.ndarray] | None = None) -> None:
    if not self.fast_cl_warp:
      return
    for key, buf in bufs.items():
      yuv_size = self.frame_buf_used_size
      frame = np.frombuffer(buf.data, dtype=np.uint8, count=yuv_size)
      cache_key = (key, int(buf.fd), frame.ctypes.data, yuv_size)
      if self.cl_import_enabled:
        try:
          if cache_key not in self.cl_imported_frames:
            self.cl_imported_frames[cache_key] = ImportedCLFrame(int(buf.fd), frame.ctypes.data, yuv_size, self.WARP_DEV)
          self.full_frames[key] = self.cl_imported_frames[cache_key].tensor
          continue
        except RuntimeError as e:
          cloudlog.warning("disabling CL dma-buf frame import: %s", e)
          self.cl_import_enabled = False
      if key not in self.cl_uploaded_frames:
        self.cl_uploaded_frames[key] = Tensor.empty(yuv_size, dtype='uint8', device=self.WARP_DEV).realize()
      frame_buffer = self.cl_uploaded_frames[key]._buffer()
      frame_buffer.allocator._copyin(frame_buffer._buf, buf.data[:yuv_size])
      self.full_frames[key] = self.cl_uploaded_frames[key]
    if transforms is not None:
      self.prewarped = self.cl_warp(self.full_frames['img'], self.full_frames['big_img'], transforms)
    status = cl.clFlush(Device[self.WARP_DEV].queue)
    if status != 0:
      raise RuntimeError(f"failed to flush preloaded OpenCL work: status={status}")

  def run(self, bufs: dict[str, VisionBuf], transforms: dict[str, np.ndarray],
          inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
    if self.fast_cl_warp and any(key not in self.full_frames for key in bufs):
      self.preload_frames(bufs)
    if not self.fast_cl_warp:
      for key in bufs.keys():
        yuv_size = self.frame_buf_params[key][3]
        ptr = np.frombuffer(bufs[key].data, dtype=np.uint8).ctypes.data
        # There is a ringbuffer of imgs, just cache tensors pointing to all of them
        cache_key = (key, ptr)
        if cache_key not in self._blob_cache:
          self._blob_cache[cache_key] = Tensor.from_blob(ptr, (yuv_size,), dtype='uint8', device=self.WARP_DEV)
        self.full_frames[key] = self._blob_cache[cache_key]

    # Model decides when action is completed, so desire input is just a pulse triggered on rising edge
    inputs['desire_pulse'][0] = 0
    self.npy['desire'][:] = np.where(inputs['desire_pulse'] - self.prev_desire > .99, inputs['desire_pulse'], 0)
    self.prev_desire[:] = inputs['desire_pulse']
    self.npy['traffic_convention'][:] = inputs['traffic_convention']
    self.npy['action_t'][:] = inputs['action_t']
    if self.fast_cl_warp:
      img, big_img = self.prewarped or self.cl_warp(self.full_frames['img'], self.full_frames['big_img'], transforms)
      self.prewarped = None
      outs, = self.run_policy(**{k: self.input_queues[k] for k in POLICY_INPUTS[2:] if k in self.input_queues},
                              img=img, big_img=big_img)
    else:
      self.npy['tfm'][:,:] = transforms['img'][:,:]
      self.npy['big_tfm'][:,:] = transforms['big_img'][:,:]
      warped = self.warp(**{k: self.input_queues[k] for k in WARP_INPUTS}, frame=self.full_frames['img'], big_frame=self.full_frames['big_img'])
      outs, = self.run_policy(**{k: self.input_queues[k] for k in POLICY_INPUTS if k in self.input_queues}, warped=warped)
    if self.fast_cl_warp:
      if self.model_output is None:
        self.model_output = np.empty(outs.shape, dtype=np.float32)
      status = cl.clEnqueueReadBuffer(Device[self.QUEUE_DEV].queue, outs._buffer()._buf, cl.CL_TRUE, 0,
                                      self.model_output.nbytes, ctypes.c_void_p(self.model_output.ctypes.data), 0, None, None)
      if status != 0:
        raise RuntimeError(f"failed to read model output from OpenCL: status={status}")
      Device[self.QUEUE_DEV].pending_copyin.clear()
      model_output = self.model_output[0]
    else:
      model_output = outs.numpy()[0]
    outputs_dict = self.parser.parse_outputs(self.slice_outputs(model_output, self.output_slices))
    self.npy['prev_feat'][:] = model_output[self.output_slices['hidden_state']]

    if SEND_RAW_PRED:
      outputs_dict['raw_pred'] = model_output.copy()
    return outputs_dict

  def warmup(self) -> None:
    dummy_frames = {k: np.zeros(self.frame_buf_params[k][3], dtype=np.uint8) for k in self.vision_input_names}
    eye = np.eye(3, dtype=np.float32)
    dims = {'desire_pulse': ModelConstants.DESIRE_LEN, 'traffic_convention': 2, 'action_t': 2}
    self.run(dummy_frames, dict.fromkeys(self.vision_input_names, eye), {k: np.zeros(v, dtype=np.float32) for k, v in dims.items()})
    self.input_queues, self.npy = make_input_queues(self.input_shapes, self.frame_skip, device=self.QUEUE_DEV)
    self.prev_desire[:] = 0
    self.full_frames.clear()
    self._blob_cache.clear()


def main(demo=False):
  cloudlog.warning("modeld init")

  USBGPU = usbgpu_present() and usbgpu_compiled()
  params = Params()
  params.put_bool("UsbGpuLoading", USBGPU)
  params.remove("UsbGpuActive")

  config_realtime_process([6, 7], 54)

  # visionipc clients
  while True:
    available_streams = VisionIpcClient.available_streams("camerad", block=False)
    if available_streams:
      use_extra_client = VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams and VisionStreamType.VISION_STREAM_ROAD in available_streams
      main_wide_camera = VisionStreamType.VISION_STREAM_ROAD not in available_streams
      break
    time.sleep(.1)

  vipc_client_main_stream = VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide_camera else VisionStreamType.VISION_STREAM_ROAD
  vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True)
  vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False)
  cloudlog.warning(f"vision stream set up, main_wide_camera: {main_wide_camera}, use_extra_client: {use_extra_client}")

  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while use_extra_client and not vipc_client_extra.connect(False):
    time.sleep(0.1)

  cloudlog.warning(f"connected main cam with buffer size: {vipc_client_main.buffer_len} ({vipc_client_main.width} x {vipc_client_main.height})")
  if use_extra_client:
    cloudlog.warning(f"connected extra cam with buffer size: {vipc_client_extra.buffer_len} ({vipc_client_extra.width} x {vipc_client_extra.height})")

  st = time.monotonic()
  cloudlog.warning("loading model")
  model = None
  if USBGPU:
    big_model = None
    def load_big():
      nonlocal big_model
      try:
        m = ModelState(vipc_client_main.width, vipc_client_main.height, True)
        m.warmup()
        big_model = m
      except Exception:
        cloudlog.exception("big model load failed")
    loader = threading.Thread(target=load_big, daemon=True)
    loader.start()
    loader.join(BIG_MODEL_TIMEOUT)
    model = big_model
    params.put_bool("UsbGpuActive", model is not None)

  small_model = ModelState(vipc_client_main.width, vipc_client_main.height, False) if model is None or USBGPU else None
  if model is None:
    model = small_model
  params.put_bool("UsbGpuLoading", False)
  cloudlog.warning(f"models loaded in {time.monotonic() - st:.1f}s, modeld starting")

  # messaging
  pm = PubMaster(["modelV2", "drivingModelData", "cameraOdometry"])
  sm = SubMaster(["deviceState", "carState", "roadCameraState", "liveCalibration", "driverMonitoringState", "carControl", "liveDelay"])

  publish_state = PublishState()
  params = Params()

  # setup filter to track dropped frames
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / ModelConstants.MODEL_RUN_FREQ)
  frame_id = 0
  last_vipc_frame_id = 0
  run_count = 0

  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  live_calib_seen = False
  buf_main, buf_extra = None, None
  meta_main = FrameMeta()
  meta_extra = FrameMeta()

  if demo:
    CP = get_demo_car_params()
  else:
    CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("modeld got CarParams: %s", CP.brand)

  # TODO this needs more thought, use .2s extra for now to estimate other delays
  # TODO Move smooth seconds to action function
  long_delay = CP.longitudinalActuatorDelay + LONG_SMOOTH_SECONDS
  prev_action = log.ModelDataV2.Action()

  DH = DesireHelper()

  while True:
    # Keep receiving frames until we are at least 1 frame ahead of previous extra frame
    while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
      buf_main = vipc_client_main.recv()
      meta_main = FrameMeta(vipc_client_main)
      if buf_main is None:
        break

    if buf_main is None:
      cloudlog.debug("vipc_client_main no frame")
      continue

    if use_extra_client:
      # Keep receiving extra frames until frame id matches main camera
      while True:
        buf_extra = vipc_client_extra.recv()
        meta_extra = FrameMeta(vipc_client_extra)
        if buf_extra is None or meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
          break

      if buf_extra is None:
        cloudlog.debug("vipc_client_extra no frame")
        continue

      if abs(meta_main.timestamp_sof - meta_extra.timestamp_sof) > 10000000:
        cloudlog.error(f"frames out of sync! main: {meta_main.frame_id} ({meta_main.timestamp_sof / 1e9:.5f}),\
                         extra: {meta_extra.frame_id} ({meta_extra.timestamp_sof / 1e9:.5f})")

    else:
      # Use single camera
      buf_extra = buf_main
      meta_extra = meta_main

    sm.update(0)
    desire = DH.desire
    is_rhd = sm["driverMonitoringState"].isRHD
    frame_id = sm["roadCameraState"].frameId
    v_ego = max(sm["carState"].vEgo, 0.)
    lat_delay = sm["liveDelay"].lateralDelay + LAT_SMOOTH_SECONDS
    if sm.updated["liveCalibration"] and sm.seen['roadCameraState'] and sm.seen['deviceState']:
      device_from_calib_euler = np.array(sm["liveCalibration"].rpyCalib, dtype=np.float32)
      dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['roadCameraState'].sensor))]
      model_transform_main = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics if main_wide_camera else dc.fcam.intrinsics, False).astype(np.float32)
      has_wide_camera = use_extra_client or main_wide_camera
      model_transform_extra = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics if has_wide_camera else dc.fcam.intrinsics, True).astype(np.float32)
      live_calib_seen = True

    traffic_convention = np.zeros(2)
    traffic_convention[int(is_rhd)] = 1

    vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    if desire >= 0 and desire < ModelConstants.DESIRE_LEN:
      vec_desire[desire] = 1

    # tracked dropped frames
    vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
    frames_dropped = frame_dropped_filter.update(min(vipc_dropped_frames, 10))
    if run_count < 10: # let frame drops warm up
      frame_dropped_filter.x = 0.
      frames_dropped = 0.
    run_count = run_count + 1

    frame_drop_ratio = frames_dropped / (1 + frames_dropped)

    bufs = {name: buf_extra if 'big' in name else buf_main for name in model.vision_input_names}
    transforms = {name: model_transform_extra if 'big' in name else model_transform_main for name in model.vision_input_names}
    model.preload_frames(bufs, transforms)
    frame_delay = DT_MDL # compensate for time passed since the frame was captured: current_time - timestamp_eof is 50ms on average
    action_delay = DT_MDL / 2 # middle of the interval between model output (current state) and next frame (expected state)
    lat_action_t = lat_delay + frame_delay + action_delay
    long_action_t = long_delay + frame_delay + action_delay
    inputs: dict[str, np.ndarray] = {
      'desire_pulse': vec_desire,
      'traffic_convention': traffic_convention,
      'action_t': np.array([lat_action_t, long_action_t], dtype=np.float32),
    }

    mt1 = time.perf_counter()
    try:
      model_output = model.run(bufs, transforms, inputs)
    except Exception:
      if not params.get_bool("UsbGpuActive"):
        raise
      # fallback to small model
      cloudlog.exception("big model failed, fall back to small")
      params.put_bool("UsbGpuActive", False)
      model = small_model
      run_count = 0
      model_output = None
    mt2 = time.perf_counter()
    model_execution_time = mt2 - mt1

    if model_output is not None:
      modelv2_send = messaging.new_message('modelV2')
      drivingdata_send = messaging.new_message('drivingModelData')
      posenet_send = messaging.new_message('cameraOdometry')

      action = get_action_from_model(model_output, prev_action, lat_action_t, long_action_t, v_ego)
      prev_action = action
      fill_model_msg(modelv2_send, model_output, action,
                     publish_state, meta_main.frame_id, meta_extra.frame_id, frame_id,
                     frame_drop_ratio, meta_main.timestamp_eof, model_execution_time, live_calib_seen)

      desire_state = modelv2_send.modelV2.meta.desireState
      l_lane_change_prob = desire_state[log.Desire.laneChangeLeft]
      r_lane_change_prob = desire_state[log.Desire.laneChangeRight]
      lane_change_prob = l_lane_change_prob + r_lane_change_prob
      DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob)
      modelv2_send.modelV2.meta.laneChangeState = DH.lane_change_state
      modelv2_send.modelV2.meta.laneChangeDirection = DH.lane_change_direction

      fill_driving_model_data(drivingdata_send, modelv2_send)
      fill_pose_msg(posenet_send, model_output, meta_main.frame_id, vipc_dropped_frames, meta_main.timestamp_eof, live_calib_seen)
      pm.send('modelV2', modelv2_send)
      pm.send('drivingModelData', drivingdata_send)
      pm.send('cameraOdometry', posenet_send)
    last_vipc_frame_id = meta_main.frame_id


if __name__ == "__main__":
  try:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='A boolean for demo mode.')
    args = parser.parse_args()
    main(demo=args.demo)
  except KeyboardInterrupt:
    cloudlog.warning("got SIGINT")
