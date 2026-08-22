#include "system/loggerd/encoder/venus_encoder.h"

#include <fcntl.h>
#include <linux/dma-buf.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>

#include "common/swaglog.h"
#include "common/util.h"

namespace {

constexpr unsigned int kInputBufferCount = 8;
constexpr unsigned int kCaptureBufferCount = 8;

bool ioctl_ok(int fd, unsigned long request, void *arg, const char *name) {
  int ret;
  do {
    ret = ioctl(fd, request, arg);
  } while (ret < 0 && errno == EINTR);

  if (ret < 0) {
    LOGE("Venus encoder %s failed: %d (%s)", name, errno, strerror(errno));
    return false;
  }
  return true;
}

bool sync_dma_buffer(int fd, uint64_t flags) {
  struct dma_buf_sync sync = {.flags = flags};
  int ret;
  do {
    ret = ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
  } while (ret < 0 && errno == EINTR);
  return ret == 0 || errno == ENOTTY;
}

std::vector<capnp::byte> extract_codec_header(const capnp::byte *data,
                                              size_t size, bool hevc) {
  for (size_t i = 0; i + 4 < size; ++i) {
    size_t start_code_size = 0;
    if (data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1) {
      start_code_size = 3;
    } else if (i + 4 < size && data[i] == 0 && data[i + 1] == 0 &&
               data[i + 2] == 0 && data[i + 3] == 1) {
      start_code_size = 4;
    }
    if (start_code_size == 0)
      continue;

    const uint8_t nal_type =
        hevc ? (data[i + start_code_size] >> 1) & 0x3f
             : data[i + start_code_size] & 0x1f;
    const bool is_video_slice =
        hevc ? nal_type <= 31 : (nal_type >= 1 && nal_type <= 5);
    if (is_video_slice && i > 0)
      return std::vector<capnp::byte>(data, data + i);
  }
  return {};
}

bool bitstream_is_keyframe(const capnp::byte *data, size_t size, bool hevc) {
  for (size_t i = 0; i + 4 < size; ++i) {
    size_t start_code_size = 0;
    if (data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1) {
      start_code_size = 3;
    } else if (i + 4 < size && data[i] == 0 && data[i + 1] == 0 &&
               data[i + 2] == 0 && data[i + 3] == 1) {
      start_code_size = 4;
    }
    if (start_code_size == 0)
      continue;

    const uint8_t nal_type =
        hevc ? (data[i + start_code_size] >> 1) & 0x3f
             : data[i + start_code_size] & 0x1f;
    if ((!hevc && nal_type == 5) ||
        (hevc && nal_type >= 19 && nal_type <= 21)) {
      return true;
    }
  }
  return false;
}

} // namespace

VenusEncoder::VenusEncoder(const EncoderInfo &encoder_info, int in_width,
                           int in_height, int in_stride, int in_uv_offset)
    : VideoEncoder(encoder_info, in_width, in_height),
      camera_stride(in_stride),
      camera_uv_offset(in_uv_offset) {
  valid = initialize();
  if (!valid) {
    cleanup();
    LOGW("Venus encoder initialization failed for %s; using software fallback",
         encoder_info.publish_name);
  }
}

VenusEncoder::~VenusEncoder() {
  encoder_close();
  cleanup();
}

std::string VenusEncoder::find_device() const {
  for (int index = 0; index < 64; ++index) {
    std::string path = util::string_format("/dev/video%d", index);
    int candidate =
        HANDLE_EINTR(open(path.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC));
    if (candidate < 0)
      continue;

    struct v4l2_capability capability = {};
    bool found = ioctl(candidate, VIDIOC_QUERYCAP, &capability) == 0 &&
                 strcmp(reinterpret_cast<const char *>(capability.driver),
                        "qcom-venus") == 0 &&
                 strstr(reinterpret_cast<const char *>(capability.card),
                        "encoder") != nullptr;
    close(candidate);
    if (found)
      return path;
  }
  return {};
}

bool VenusEncoder::set_control(uint32_t id, int32_t value, bool required) {
  struct v4l2_control control = {
      .id = id,
      .value = value,
  };
  if (ioctl(fd, VIDIOC_S_CTRL, &control) == 0)
    return true;

  if (required) {
    LOGE("Venus encoder control 0x%x=%d failed: %d (%s)", id, value, errno,
         strerror(errno));
  } else {
    LOGW("Venus encoder optional control 0x%x=%d unavailable: %d (%s)", id,
         value, errno, strerror(errno));
  }
  return !required;
}

bool VenusEncoder::initialize() {
  if (access("/dev/kvm", F_OK) != 0) {
    LOGE("Venus requires EL2; /dev/kvm is unavailable");
    return false;
  }

  const std::string path = find_device();
  if (path.empty()) {
    LOGE("Qualcomm Venus encoder device not found");
    return false;
  }

  fd = HANDLE_EINTR(open(path.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC));
  if (fd < 0) {
    LOGE("failed to open Venus encoder %s: %d (%s)", path.c_str(), errno,
         strerror(errno));
    return false;
  }

  struct v4l2_capability capability = {};
  if (!ioctl_ok(fd, VIDIOC_QUERYCAP, &capability, "VIDIOC_QUERYCAP"))
    return false;
  const uint32_t caps =
      capability.device_caps ? capability.device_caps : capability.capabilities;
  if (!(caps & V4L2_CAP_VIDEO_M2M_MPLANE) || !(caps & V4L2_CAP_STREAMING)) {
    LOGE("Venus encoder %s lacks multiplanar streaming capability",
         path.c_str());
    return false;
  }

  const EncoderSettings settings = encoder_info.get_settings(in_width);
  hevc =
      settings.encode_type == cereal::EncodeIndex::Type::FULL_H_E_V_C;

  struct v4l2_format input_format = {};
  input_format.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  input_format.fmt.pix_mp.width = in_width;
  input_format.fmt.pix_mp.height = in_height;
  input_format.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
  input_format.fmt.pix_mp.field = V4L2_FIELD_NONE;
  input_format.fmt.pix_mp.colorspace = V4L2_COLORSPACE_REC709;
  if (!ioctl_ok(fd, VIDIOC_S_FMT, &input_format, "VIDIOC_S_FMT input"))
    return false;

  struct v4l2_format capture_format = {};
  capture_format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  capture_format.fmt.pix_mp.width = out_width;
  capture_format.fmt.pix_mp.height = out_height;
  capture_format.fmt.pix_mp.pixelformat =
      hevc ? V4L2_PIX_FMT_HEVC : V4L2_PIX_FMT_H264;
  capture_format.fmt.pix_mp.field = V4L2_FIELD_NONE;
  if (!ioctl_ok(fd, VIDIOC_S_FMT, &capture_format, "VIDIOC_S_FMT capture"))
    return false;

  struct v4l2_selection crop = {};
  crop.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  crop.target = V4L2_SEL_TGT_CROP;
  crop.r.width = in_width;
  crop.r.height = in_height;
  if (!ioctl_ok(fd, VIDIOC_S_SELECTION, &crop, "VIDIOC_S_SELECTION crop"))
    return false;

  input_size = input_format.fmt.pix_mp.plane_fmt[0].sizeimage;
  input_stride = input_format.fmt.pix_mp.plane_fmt[0].bytesperline;
  input_uv_offset = input_stride * input_format.fmt.pix_mp.height;
  const unsigned int required_input_size =
      input_uv_offset + input_stride * ((in_height + 1) / 2);
  if (capture_format.fmt.pix_mp.num_planes != 1 ||
      input_format.fmt.pix_mp.num_planes != 1 ||
      capture_format.fmt.pix_mp.plane_fmt[0].sizeimage == 0 ||
      input_size < required_input_size) {
    LOGE("Venus encoder returned invalid buffer geometry");
    return false;
  }

  struct v4l2_streamparm stream_parameters = {};
  stream_parameters.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  stream_parameters.parm.output.timeperframe.numerator = 1;
  stream_parameters.parm.output.timeperframe.denominator = encoder_info.fps;
  if (!ioctl_ok(fd, VIDIOC_S_PARM, &stream_parameters, "VIDIOC_S_PARM"))
    return false;

  current_bitrate = settings.bitrate;
  if (!set_control(V4L2_CID_MPEG_VIDEO_BITRATE_MODE,
                   V4L2_MPEG_VIDEO_BITRATE_MODE_VBR) ||
      !set_control(V4L2_CID_MPEG_VIDEO_BITRATE, settings.bitrate) ||
      !set_control(V4L2_CID_MPEG_VIDEO_GOP_SIZE, settings.gop_size) ||
      !set_control(V4L2_CID_MPEG_VIDEO_B_FRAMES, settings.b_frames) ||
      !set_control(V4L2_CID_MPEG_VIDEO_HEADER_MODE,
                   V4L2_MPEG_VIDEO_HEADER_MODE_JOINED_WITH_1ST_FRAME)) {
    return false;
  }
  set_control(V4L2_CID_MPEG_VIDEO_BITRATE_PEAK, settings.bitrate * 3 / 2,
              false);
  if (hevc) {
    if (!set_control(V4L2_CID_MPEG_VIDEO_HEVC_PROFILE,
                     V4L2_MPEG_VIDEO_HEVC_PROFILE_MAIN) ||
        !set_control(V4L2_CID_MPEG_VIDEO_HEVC_LEVEL,
                     V4L2_MPEG_VIDEO_HEVC_LEVEL_3_1)) {
      return false;
    }
  } else {
    set_control(V4L2_CID_MPEG_VIDEO_H264_PROFILE,
                V4L2_MPEG_VIDEO_H264_PROFILE_CONSTRAINED_BASELINE, false);
    set_control(V4L2_CID_MPEG_VIDEO_H264_LEVEL, V4L2_MPEG_VIDEO_H264_LEVEL_3_2,
                false);
    if (!set_control(V4L2_CID_MPEG_VIDEO_H264_I_PERIOD, 1))
      return false;
  }

  struct v4l2_requestbuffers capture_request = {};
  capture_request.count = kCaptureBufferCount;
  capture_request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  capture_request.memory = V4L2_MEMORY_MMAP;
  if (!ioctl_ok(fd, VIDIOC_REQBUFS, &capture_request,
                "VIDIOC_REQBUFS capture") ||
      capture_request.count == 0) {
    return false;
  }

  capture_buffers.resize(capture_request.count);
  for (unsigned int i = 0; i < capture_buffers.size(); ++i) {
    struct v4l2_plane plane = {};
    struct v4l2_buffer buffer = {};
    buffer.index = i;
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.m.planes = &plane;
    buffer.length = 1;
    if (!ioctl_ok(fd, VIDIOC_QUERYBUF, &buffer, "VIDIOC_QUERYBUF capture"))
      return false;

    void *addr = mmap(nullptr, plane.length, PROT_READ | PROT_WRITE, MAP_SHARED,
                      fd, plane.m.mem_offset);
    if (addr == MAP_FAILED) {
      LOGE("Venus encoder capture mmap failed: %d (%s)", errno,
           strerror(errno));
      return false;
    }
    capture_buffers[i] = {.addr = addr, .len = plane.length};
    if (!queue_capture_buffer(i))
      return false;
  }

  struct v4l2_requestbuffers input_request = {};
  input_request.count = kInputBufferCount;
  input_request.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  input_request.memory = V4L2_MEMORY_DMABUF;
  if (!ioctl_ok(fd, VIDIOC_REQBUFS, &input_request, "VIDIOC_REQBUFS input") ||
      input_request.count == 0) {
    return false;
  }
  input_buffer_count = input_request.count;
  input_buffers.resize(input_buffer_count);
  for (unsigned int i = 0; i < input_buffer_count; ++i) {
    input_buffers[i].allocate(input_size);
    if (input_buffers[i].handle != -1) {
      LOGE("failed to allocate Venus DMA input buffer %u", i);
      return false;
    }
    input_buffers[i].init_yuv(in_width, in_height, input_stride,
                              input_uv_offset);
    free_input_buffers.push(i);
  }

  v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  if (!ioctl_ok(fd, VIDIOC_STREAMON, &type, "VIDIOC_STREAMON capture"))
    return false;
  capture_streaming = true;
  type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  if (!ioctl_ok(fd, VIDIOC_STREAMON, &type, "VIDIOC_STREAMON input"))
    return false;
  output_streaming = true;

  LOGW("using Venus hardware encoder %s for %s (%dx%d to %dx%d, camera stride/uv %u/%u, "
       "input stride/uv %u/%u, %d fps)",
       path.c_str(), encoder_info.publish_name, in_width, in_height, out_width, out_height,
       camera_stride, camera_uv_offset, input_stride, input_uv_offset,
       encoder_info.fps);
  return true;
}

bool VenusEncoder::queue_capture_buffer(unsigned int index) {
  struct v4l2_plane plane = {};
  plane.length = capture_buffers[index].len;
  struct v4l2_buffer buffer = {};
  buffer.index = index;
  buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  buffer.memory = V4L2_MEMORY_MMAP;
  buffer.m.planes = &plane;
  buffer.length = 1;
  return ioctl_ok(fd, VIDIOC_QBUF, &buffer, "VIDIOC_QBUF capture");
}

void VenusEncoder::cleanup() {
  if (fd < 0)
    return;

  if (output_streaming) {
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    ioctl(fd, VIDIOC_STREAMOFF, &type);
    output_streaming = false;
  }
  if (capture_streaming) {
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    ioctl(fd, VIDIOC_STREAMOFF, &type);
    capture_streaming = false;
  }

  for (auto &buffer : capture_buffers) {
    if (buffer.addr != nullptr && buffer.addr != MAP_FAILED)
      munmap(buffer.addr, buffer.len);
  }
  capture_buffers.clear();
  for (auto &buffer : input_buffers) {
    if (buffer.addr != nullptr && buffer.addr != MAP_FAILED)
      buffer.free();
  }
  input_buffers.clear();
  close(fd);
  fd = -1;
  valid = false;
}

void VenusEncoder::encoder_open() {
  if (!valid || is_open)
    return;
  stop_dequeue = false;
  counter = 0;
  ++segment_num;
  dequeue_thread = std::thread(&VenusEncoder::dequeue_handler, this);
  is_open = true;
}

bool VenusEncoder::copy_input(VisionBuf *source, VisionBuf &destination) const {
  if (!sync_dma_buffer(source->fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ)) {
    LOGE("failed to begin Venus DMA buffer synchronization");
    return false;
  }
  if (!sync_dma_buffer(destination.fd,
                       DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE)) {
    sync_dma_buffer(source->fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
    LOGE("failed to begin Venus DMA buffer synchronization");
    return false;
  }

  for (int row = 0; row < in_height; ++row) {
    memcpy(destination.y + row * input_stride,
           source->y + row * source->stride, in_width);
  }
  for (int row = 0; row < (in_height + 1) / 2; ++row) {
    memcpy(destination.uv + row * input_stride,
           source->uv + row * source->stride, in_width);
  }

  const bool destination_ok =
      sync_dma_buffer(destination.fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
  const bool source_ok =
      sync_dma_buffer(source->fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
  if (!destination_ok || !source_ok) {
    LOGE("failed to finish Venus DMA buffer synchronization");
    return false;
  }
  return true;
}

int VenusEncoder::encode_frame(VisionBuf *buf, VisionIpcBufExtra *extra) {
  const size_t required_camera_size =
      camera_uv_offset + camera_stride * ((in_height + 1) / 2);
  if (!valid || !is_open || buf->fd < 0 ||
      buf->width != static_cast<size_t>(in_width) ||
      buf->height != static_cast<size_t>(in_height) ||
      buf->stride != camera_stride || buf->uv_offset != camera_uv_offset ||
      buf->len < required_camera_size) {
    LOGE("invalid Venus input for %s: fd=%d %zux%zu stride=%zu uv=%zu len=%zu, "
         "expected %dx%d stride=%u uv=%u",
         encoder_info.publish_name, buf->fd, buf->width, buf->height,
         buf->stride, buf->uv_offset, buf->len, in_width, in_height,
         camera_stride, camera_uv_offset);
    return -1;
  }

  unsigned int index;
  if (!free_input_buffers.try_pop(index, 1000)) {
    LOGE("Venus encoder %s input queue timed out", encoder_info.publish_name);
    return -1;
  }

  VisionBuf &input = input_buffers[index];
  if (!copy_input(buf, input)) {
    free_input_buffers.push(index);
    return -1;
  }

  struct v4l2_plane plane = {};
  plane.bytesused = input_size;
  plane.length = input.len;
  plane.m.fd = input.fd;
  struct v4l2_buffer buffer = {};
  buffer.index = index;
  buffer.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  buffer.memory = V4L2_MEMORY_DMABUF;
  buffer.flags = V4L2_BUF_FLAG_TIMESTAMP_COPY;
  buffer.timestamp.tv_sec = extra->timestamp_eof / 1000000000ULL;
  buffer.timestamp.tv_usec = (extra->timestamp_eof / 1000ULL) % 1000000ULL;
  buffer.m.planes = &plane;
  buffer.length = 1;

  if (!ioctl_ok(fd, VIDIOC_QBUF, &buffer, "VIDIOC_QBUF input")) {
    free_input_buffers.push(index);
    return -1;
  }

  outstanding.fetch_add(1);
  extras.push(*extra);
  return counter++;
}

void VenusEncoder::dequeue_handler() {
  util::set_thread_name(
      ("dq-" + std::string(encoder_info.publish_name)).c_str());
  uint32_t index = 0;

  while (!stop_dequeue || outstanding.load() > 0) {
    struct pollfd poll_fd = {.fd = fd, .events = POLLIN | POLLOUT};
    int ret = poll(&poll_fd, 1, 100);
    if (ret < 0) {
      if (errno != EINTR)
        LOGE("Venus encoder poll failed: %d (%s)", errno, strerror(errno));
      continue;
    }
    if (ret == 0)
      continue;

    if (poll_fd.revents & POLLIN) {
      while (true) {
        struct v4l2_plane plane = {};
        struct v4l2_buffer buffer = {};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.m.planes = &plane;
        buffer.length = 1;
        if (ioctl(fd, VIDIOC_DQBUF, &buffer) < 0) {
          if (errno != EAGAIN && errno != EINTR) {
            LOGE("Venus encoder VIDIOC_DQBUF capture failed: %d (%s)", errno,
                 strerror(errno));
          }
          break;
        }

        VisionIpcBufExtra extra;
        if (buffer.index >= capture_buffers.size()) {
          LOGE("Venus encoder returned invalid capture index %u", buffer.index);
          outstanding = 0;
          stop_dequeue = true;
          break;
        }
        const bool has_metadata = extras.try_pop(extra, 100);
        const bool valid_buffer = plane.data_offset <= plane.bytesused &&
                                  plane.bytesused <= capture_buffers[buffer.index].len;
        if (!valid_buffer) {
          LOGE("Venus encoder returned an invalid capture buffer");
        } else if (!has_metadata) {
          LOGE("Venus encoder produced a frame without metadata");
        } else {
          auto *data = static_cast<capnp::byte *>(
                           capture_buffers[buffer.index].addr) +
                       plane.data_offset;
          const size_t data_size = plane.bytesused - plane.data_offset;
          if (codec_header.empty())
            codec_header = extract_codec_header(data, data_size, hevc);

          const unsigned int flags =
              buffer.flags | (bitstream_is_keyframe(data, data_size, hevc)
                                  ? V4L2_BUF_FLAG_KEYFRAME
                                  : 0);
          kj::ArrayPtr<capnp::byte> header = {};
          const bool data_has_header =
              !codec_header.empty() && data_size >= codec_header.size() &&
              memcmp(data, codec_header.data(), codec_header.size()) == 0;
          if ((flags & V4L2_BUF_FLAG_KEYFRAME) && !codec_header.empty())
            header = kj::arrayPtr(codec_header.data(), codec_header.size());
          size_t payload_size = data_size;
          if (data_has_header) {
            payload_size -= codec_header.size();
            memmove(data, data + codec_header.size(), payload_size);
          }
          publisher_publish(
              segment_num, index++, extra, flags, header,
              kj::arrayPtr(data, payload_size));
        }
        if (outstanding.load() > 0)
          outstanding.fetch_sub(1);
        if (!queue_capture_buffer(buffer.index)) {
          outstanding = 0;
          stop_dequeue = true;
        }
      }
    }

    if (poll_fd.revents & POLLOUT) {
      while (true) {
        struct v4l2_plane plane = {};
        struct v4l2_buffer buffer = {};
        buffer.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
        buffer.memory = V4L2_MEMORY_DMABUF;
        buffer.m.planes = &plane;
        buffer.length = 1;
        if (ioctl(fd, VIDIOC_DQBUF, &buffer) < 0) {
          if (errno != EAGAIN && errno != EINTR) {
            LOGE("Venus encoder VIDIOC_DQBUF input failed: %d (%s)", errno,
                 strerror(errno));
          }
          break;
        }
        if (buffer.index >= input_buffer_count) {
          LOGE("Venus encoder returned invalid input index %u", buffer.index);
          outstanding = 0;
          stop_dequeue = true;
          break;
        }
        free_input_buffers.push(buffer.index);
      }
    }

    if (poll_fd.revents & (POLLERR | POLLNVAL)) {
      LOGE("Venus encoder %s poll error: 0x%x", encoder_info.publish_name,
           poll_fd.revents);
      std::abort();
    }
  }
}

void VenusEncoder::encoder_close() {
  if (!is_open)
    return;

  for (int i = 0;
       i < 200 && (outstanding.load() > 0 ||
                   free_input_buffers.size() < input_buffer_count);
       ++i)
    util::sleep_for(10);
  if (outstanding.load() > 0 ||
      free_input_buffers.size() < input_buffer_count) {
    LOGE("Venus encoder %s close timed out with %u frames outstanding and "
         "%zu/%u input buffers free",
         encoder_info.publish_name, outstanding.load(),
         free_input_buffers.size(), input_buffer_count);
    outstanding = 0;
  }
  stop_dequeue = true;
  if (dequeue_thread.joinable())
    dequeue_thread.join();
  is_open = false;
}

void VenusEncoder::set_bitrate(int bitrate) {
  if (!valid || bitrate == current_bitrate)
    return;
  if (bitrate <= 0 || !set_control(V4L2_CID_MPEG_VIDEO_BITRATE, bitrate)) {
    LOGE("failed to update %s bitrate to %d", encoder_info.publish_name,
         bitrate);
    return;
  }
  set_control(V4L2_CID_MPEG_VIDEO_BITRATE_PEAK, bitrate * 3 / 2, false);
  current_bitrate = bitrate;
}

void VenusEncoder::request_keyframe() {
  // vamOS programs every H.264 I-frame as an IDR and livestream GOPs are only
  // five frames. Runtime sync-frame requests can crash Dragon's Venus firmware,
  // so let WebRTC wait at most one GOP for the next periodic IDR.
}
