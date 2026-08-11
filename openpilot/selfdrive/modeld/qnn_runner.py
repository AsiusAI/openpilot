import os
from pathlib import Path

import numpy as np
from openpilot.common.file_chunker import get_manifest_path, open_file_chunked


class NumpyFramePreprocessor:
  """Diagnostic nearest-neighbor NV12 warp for comparing without the GPU path."""
  def __init__(self, cam_w: int, cam_h: int, nv12_info: tuple[int, int, int, int], model_h: int, model_w: int):
    self.cam_w, self.cam_h = cam_w, cam_h
    self.stride, self.y_height, self.uv_height, _ = nv12_info
    self.model_h, self.model_w = model_h, model_w
    self.y_x = np.tile(np.arange(model_w * 2, dtype=np.float32), model_h * 2)
    self.y_y = np.repeat(np.arange(model_h * 2, dtype=np.float32), model_w * 2)
    self.uv_x = np.tile(np.arange(model_w, dtype=np.float32), model_h)
    self.uv_y = np.repeat(np.arange(model_h, dtype=np.float32), model_w)
    self.uv_scale = np.array([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [2.0, 2.0, 1.0]], dtype=np.float32)

  @staticmethod
  def sample(src: np.ndarray, matrix: np.ndarray, x: np.ndarray, y: np.ndarray,
             width: int, height: int, stride: int) -> np.ndarray:
    denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    src_x = (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denominator
    src_y = (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denominator
    # tinygrad's round() follows C round semantics (half away from zero).
    x_index = np.floor(src_x + 0.5).clip(0, width - 1).astype(np.int32)
    y_index = np.floor(src_y + 0.5).clip(0, height - 1).astype(np.int32)
    return src[y_index * stride + x_index]

  def prepare(self, frame: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    y = self.sample(frame, matrix, self.y_x, self.y_y, self.cam_w, self.cam_h, self.stride)
    uv_offset = self.stride * self.y_height
    uv = frame[uv_offset:uv_offset + self.stride * self.uv_height].reshape(self.uv_height, self.stride)
    uv_matrix = matrix * self.uv_scale
    uv_width, uv_height = self.cam_w // 2, self.cam_h // 2
    u = self.sample(uv[:uv_height, :self.cam_w:2].reshape(-1), uv_matrix,
                    self.uv_x, self.uv_y, uv_width, uv_height, uv_width)
    v = self.sample(uv[:uv_height, 1:self.cam_w:2].reshape(-1), uv_matrix,
                    self.uv_x, self.uv_y, uv_width, uv_height, uv_width)

    y = y.reshape(self.model_h * 2, self.model_w * 2)
    output = np.empty((6, self.model_h, self.model_w), dtype=np.uint8)
    output[0] = y[0::2, 0::2]
    output[1] = y[1::2, 0::2]
    output[2] = y[0::2, 1::2]
    output[3] = y[1::2, 1::2]
    output[4] = u.reshape(self.model_h, self.model_w)
    output[5] = v.reshape(self.model_h, self.model_w)
    return output

  def run(self, frames: dict[str, np.ndarray], transforms: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack((self.prepare(frames['img'], transforms['img']),
                     self.prepare(frames['big_img'], transforms['big_img'])))


class QnnModelRunner:
  def __init__(self, model_path: str, input_shapes: dict[str, tuple[int, ...]], frame_skip: int, use_qnn: bool):
    import onnxruntime as ort

    options = ort.SessionOptions()
    if use_qnn:
      if 'QNNExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('onnxruntime was built without the QNN execution provider')
      # The HTP executes each fused partition; ORT's CPU thread pool only sees
      # a few fixed-shape bookkeeping nodes, so one worker avoids idle threads.
      options.intra_op_num_threads = int(os.getenv('QNN_INTRA_OP_THREADS', '1'))
      options.log_severity_level = int(os.getenv('QNN_ORT_LOG_LEVEL', '3'))
      providers = [('QNNExecutionProvider', {
        'backend_path': os.getenv('QNN_BACKEND_PATH', '/usr/lib/libQnnHtp.so'),
        'soc_model': '35',
        'htp_arch': '68',
        'device_id': '0',
        'offload_graph_io_quantization': '0',
        'htp_performance_mode': os.getenv('QNN_PERFORMANCE_MODE', 'burst'),
      })]
      profile_path = os.getenv('QNN_PROFILE_PATH')
      if profile_path:
        providers[0][1].update(profiling_level='basic', profiling_file_path=profile_path)
    else:
      providers = ['CPUExecutionProvider']

    # Keep fork-specific models out of comma's shared LFS store. Like the
    # tinygrad artifacts, a model may be committed as <=45 MiB Git chunks.
    model_source: str | bytes = model_path
    if not Path(model_path).is_file():
      with open_file_chunked(model_path) as model_stream:
        model_source = model_stream.read()
    self.session = ort.InferenceSession(model_source, sess_options=options, providers=providers)
    if use_qnn and self.session.get_providers()[0] != 'QNNExecutionProvider':
      raise RuntimeError(f'QNN execution provider failed to load: {self.session.get_providers()}')
    self.frame_skip = frame_skip
    img_shape = input_shapes['img']
    n_frames = img_shape[1] // 6
    img_queue_shape = (frame_skip * (n_frames - 1) + 1, 6, img_shape[2], img_shape[3])
    self.img_q = np.zeros(img_queue_shape, dtype=np.uint8)
    self.big_img_q = np.zeros(img_queue_shape, dtype=np.uint8)
    self.feat_q = np.zeros((frame_skip * input_shapes['features_buffer'][1], 1, input_shapes['features_buffer'][2]), dtype=np.float32)
    self.desire_q = np.zeros((frame_skip * input_shapes['desire_pulse'][1], 1, input_shapes['desire_pulse'][2]), dtype=np.float32)

  @staticmethod
  def shift(buf: np.ndarray, value: np.ndarray) -> None:
    buf[:-len(value)] = buf[len(value):]
    buf[-len(value):] = value

  def reset(self) -> None:
    for buf in (self.img_q, self.big_img_q, self.feat_q, self.desire_q):
      buf.fill(0)

  def run(self, warped, npy: dict[str, np.ndarray]) -> np.ndarray:
    warped_np = warped if isinstance(warped, np.ndarray) else warped.numpy()
    self.shift(self.img_q, warped_np[0:1])
    self.shift(self.big_img_q, warped_np[1:2])
    self.shift(self.feat_q, npy['prev_feat'].reshape(1, 1, -1))
    self.shift(self.desire_q, npy['desire'].reshape(1, 1, -1))

    desire = self.desire_q.reshape(-1, self.frame_skip, *self.desire_q.shape[1:]).max(1).reshape(1, -1, self.desire_q.shape[-1])
    inputs = {
      'img': self.img_q[::self.frame_skip].reshape(1, -1, *self.img_q.shape[2:]).transpose(0, 2, 3, 1),
      'big_img': self.big_img_q[::self.frame_skip].reshape(1, -1, *self.big_img_q.shape[2:]).transpose(0, 2, 3, 1),
      'features_buffer': self.feat_q[::self.frame_skip].reshape(1, -1, self.feat_q.shape[-1]),
      'desire_pulse': desire,
      'traffic_convention': npy['traffic_convention'],
      'action_t': npy['action_t'],
    }
    return self.session.run(None, inputs)[0][0]


def default_qnn_model_path(models_dir: Path) -> str | None:
  configured = os.getenv('MODEL_ONNX_PATH')
  if configured:
    return configured
  path = models_dir / 'driving_supercombo_qnn.onnx'
  return str(path) if path.is_file() or Path(get_manifest_path(path)).is_file() else None
