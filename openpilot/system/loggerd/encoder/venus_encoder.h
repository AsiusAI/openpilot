#pragma once

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "common/queue.h"
#include "system/loggerd/encoder/encoder.h"

class VenusEncoder : public VideoEncoder {
public:
  VenusEncoder(const EncoderInfo &encoder_info, int in_width, int in_height,
               int in_stride, int in_uv_offset);
  ~VenusEncoder();

  int encode_frame(VisionBuf *buf, VisionIpcBufExtra *extra) override;
  void encoder_open() override;
  void encoder_close() override;
  void set_bitrate(int bitrate) override;
  void request_keyframe() override;

private:
  struct CaptureBuffer {
    void *addr = nullptr;
    size_t len = 0;
  };

  bool initialize();
  void cleanup();
  void dequeue_handler();
  bool queue_capture_buffer(unsigned int index);
  bool set_control(uint32_t id, int32_t value, bool required = true);
  std::string find_device() const;
  bool copy_input(VisionBuf *source, VisionBuf &destination) const;

  int fd = -1;
  bool valid = false;
  bool is_open = false;
  bool capture_streaming = false;
  bool output_streaming = false;
  int segment_num = -1;
  int counter = 0;
  int current_bitrate = -1;
  bool hevc = false;
  unsigned int input_buffer_count = 0;
  unsigned int input_size = 0;
  unsigned int input_stride = 0;
  unsigned int input_uv_offset = 0;
  const unsigned int camera_stride;
  const unsigned int camera_uv_offset;

  std::atomic<bool> stop_dequeue = false;
  std::atomic<unsigned int> outstanding = 0;
  std::thread dequeue_thread;
  SafeQueue<VisionIpcBufExtra> extras;
  SafeQueue<unsigned int> free_input_buffers;
  std::vector<CaptureBuffer> capture_buffers;
  std::vector<VisionBuf> input_buffers;
  std::vector<capnp::byte> codec_header;
};
