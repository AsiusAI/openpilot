#pragma once

#include <cassert>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#ifdef __ASIUS_HARDWARE__
struct i2c_random_wr_payload {
  uint32_t reg_addr;
  uint32_t reg_data;
};

#define CAM_ISP_PATTERN_BAYER_RGRGRG 0
#define CAM_ISP_PATTERN_BAYER_GRGRGR 1
#define CAM_ISP_PATTERN_BAYER_BGBGBG 2
#define CAM_ISP_PATTERN_BAYER_GBGBGB 3
#define CAM_FORMAT_MIPI_RAW_10 10
#define CAM_FORMAT_MIPI_RAW_12 12
#define CSI_RAW10 0x2b
#define CSI_RAW12 0x2c
#else
#include "media/cam_isp.h"
#include "media/cam_sensor.h"
#endif

#include "openpilot/cereal/gen/cpp/log.capnp.h"
#include "system/camerad/sensors/ox03c10_registers.h"
#include "system/camerad/sensors/os04c10_registers.h"

#define ANALOG_GAIN_MAX_CNT 55

class SensorInfo {
public:
  SensorInfo() = default;
  virtual std::vector<i2c_random_wr_payload> getExposureRegisters(int exposure_time, int new_exp_g, bool dc_gain_enabled) const { return {}; }
  virtual float getExposureScore(float desired_ev, int exp_t, int exp_g_idx, float exp_gain, int gain_idx) const {return 0; }
  virtual int getSlaveAddress(int port) const { assert(0); }

  cereal::FrameData::ImageSensor image_sensor = cereal::FrameData::ImageSensor::UNKNOWN;
  float pixel_size_mm;
  uint32_t frame_width, frame_height;
  uint32_t frame_stride;
  uint32_t frame_offset = 0;
  uint32_t extra_height = 0;
  int out_scale = 1;
  int registers_offset = -1;
  int stats_offset = -1;
  int hdr_offset = -1;

  int exposure_time_min;
  int exposure_time_max;

  float dc_gain_factor;
  int dc_gain_min_weight;
  int dc_gain_max_weight;
  float dc_gain_on_grey;
  float dc_gain_off_grey;

  float ev_scale = 1.0;
  float sensor_analog_gains[ANALOG_GAIN_MAX_CNT];
  int analog_gain_min_idx;
  int analog_gain_max_idx;
  int analog_gain_rec_idx;
  int analog_gain_cost_delta;
  float analog_gain_cost_low;
  float analog_gain_cost_high;
  float target_grey_factor;
  float min_ev;
  float max_ev;

  bool data_word;
  uint32_t probe_reg_addr;
  uint32_t probe_expected_data;
  std::vector<i2c_random_wr_payload> start_reg_array;
  std::vector<i2c_random_wr_payload> init_reg_array;

  uint32_t bits_per_pixel;
  uint32_t bayer_pattern;
  uint32_t mipi_format;
  uint32_t mclk_frequency;
  uint32_t frame_data_type;

  uint32_t readout_time_ns;  // used to recover EOF from SOF

  // ISP image processing params
  uint32_t black_level;
  std::vector<uint32_t> color_correct_matrix;  // 3x3
  std::vector<uint32_t> gamma_lut_rgb;         // gamma LUTs are length 64 * sizeof(uint32_t); same for r/g/b here
  void prepare_gamma_lut() {
    for (int i = 0; i < 64; i++) {
      gamma_lut_rgb[i] |= ((uint32_t)(gamma_lut_rgb[i+1] - gamma_lut_rgb[i]) << 10);
    }
    gamma_lut_rgb.pop_back();
  }
  std::vector<uint32_t> linearization_lut;     // length 36
  std::vector<uint32_t> linearization_pts;     // length 4
  std::vector<uint32_t> vignetting_lut;        // length 221

  const int num() const {
    return static_cast<int>(image_sensor);
  };
};

class OX03C10 : public SensorInfo {
public:
  OX03C10();
  std::vector<i2c_random_wr_payload> getExposureRegisters(int exposure_time, int new_exp_g, bool dc_gain_enabled) const override;
  float getExposureScore(float desired_ev, int exp_t, int exp_g_idx, float exp_gain, int gain_idx) const override;
  int getSlaveAddress(int port) const override;
};

class OS04C10 : public SensorInfo {
public:
  OS04C10();
  std::vector<i2c_random_wr_payload> getExposureRegisters(int exposure_time, int new_exp_g, bool dc_gain_enabled) const override;
  float getExposureScore(float desired_ev, int exp_t, int exp_g_idx, float exp_gain, int gain_idx) const override;
  int getSlaveAddress(int port) const override;
};
