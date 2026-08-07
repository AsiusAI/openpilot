import ctypes
from typing import cast

import numpy as np

from tinygrad.device import Buffer, Device, TinyELF
from tinygrad.dtype import dtypes
from tinygrad.runtime.autogen import opencl as cl
from tinygrad.tensor import Tensor


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
