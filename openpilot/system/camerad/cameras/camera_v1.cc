#include "system/camerad/cameras/camera_common.h"

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <linux/media.h>
#include <linux/media-bus-format.h>
#include <linux/v4l2-controls.h>
#include <linux/v4l2-subdev.h>
#include <linux/videodev2.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "common/params.h"
#include "common/swaglog.h"
#include "common/timing.h"
#include "system/camerad/cameras/hw.h"
#include "system/camerad/cameras/nv12_info.h"
#include "system/camerad/sensors/sensor.h"


ExitHandler do_exit;

struct OneCamRoute {
  int csiphy;
  int csid;
  int vfe;
  const char *sensor;
};

// Asius v1 camera routing:
//   Runtime camerad path: CSIPHY -> CSID PIX pad -> VFE PIX -> NV12 DMABUF.
//   Raw RDI probing lives in standalone bring-up tools, not in this runtime path.
// openpilot camera_num 0 is wide road, camera_num 1 is road,
// camera_num 2 is driver. For Asius v1:
//   CAM1 -> driver, CAM2 -> road, CAM3 -> wide road.
struct V1CamConfig {
  uint32_t csiphy_entity;
  uint32_t csid_entity;
  uint32_t vfe_pix_entity;
  int pix_video_dev;
  const char *sensor_name;
  int csiphy_subdev;
  int csid_subdev;
  int vfe_pix_subdev;
};

static int find_v4l_dev(const char *prefix, const char *name) {
  for (int i = 0; i < 64; i++) {
    auto path = util::string_format("/sys/class/video4linux/%s%d/name", prefix, i);
    auto dev_name = util::read_file(path);
    if (!dev_name.empty() && dev_name.find(name) == 0) return i;
  }
  return -1;
}

static uint32_t find_media_entity(int media_fd, const char *name) {
  struct media_entity_desc ent = {};
  for (ent.id = 0 | MEDIA_ENT_ID_FLAG_NEXT; ; ent.id |= MEDIA_ENT_ID_FLAG_NEXT) {
    if (ioctl(media_fd, MEDIA_IOC_ENUM_ENTITIES, &ent) < 0) break;
    if (strcmp(ent.name, name) == 0) return ent.id;
  }
  return 0;
}

static V1CamConfig resolve_cam_config(int media_fd, int cam_idx) {
  static const OneCamRoute routing[] = {
    {3, 1, 1, "os04c10 20-0036"},
    {2, 0, 0, "os04c10 18-0036"},
    {0, 2, 2, "os04c10 16-0036"},
  };
  const auto *r = &routing[cam_idx];
  V1CamConfig cfg = {};
  cfg.pix_video_dev = -1;
  cfg.csiphy_subdev = -1;
  cfg.csid_subdev = -1;
  cfg.vfe_pix_subdev = -1;
  cfg.sensor_name = r->sensor;
  cfg.csiphy_entity = find_media_entity(media_fd, util::string_format("msm_csiphy%d", r->csiphy).c_str());
  cfg.csid_entity = find_media_entity(media_fd, util::string_format("msm_csid%d", r->csid).c_str());
  cfg.vfe_pix_entity = find_media_entity(media_fd, util::string_format("msm_vfe%d_pix", r->vfe).c_str());
  cfg.pix_video_dev = find_v4l_dev("video", util::string_format("msm_vfe%d_video3", r->vfe).c_str());
  cfg.vfe_pix_subdev = find_v4l_dev("v4l-subdev", util::string_format("msm_vfe%d_pix", r->vfe).c_str());

  cfg.csiphy_subdev = find_v4l_dev("v4l-subdev", util::string_format("msm_csiphy%d", r->csiphy).c_str());
  cfg.csid_subdev = find_v4l_dev("v4l-subdev", util::string_format("msm_csid%d", r->csid).c_str());
  return cfg;
}

static V1CamConfig v1_cams[3];


struct OneSensorRegWrite {
  uint16_t addr;
  uint16_t data;
};

struct OneSensorWriteRegsCmd {
  uint64_t regs;
  uint32_t count;
  uint8_t data_width;
  uint8_t pad[3];
};

#define ONE_SENSOR_WRITE_REGS _IOW('S', 1, struct OneSensorWriteRegsCmd)

struct VfeRegWrite {
  uint32_t offset;
  uint32_t value;
};

struct VfeWriteRegsCmd {
  uint64_t regs;
  uint32_t count;
  uint32_t pad;
};

struct VfeDmiCmd {
  uint32_t dmi_cfg_offset;
  uint8_t ram_select;
  uint8_t pad[3];
  uint32_t count;
  uint64_t data;
};

#define VFE_IOC_MAGIC '#'
#define VFE_WRITE_REGS _IOW(VFE_IOC_MAGIC, 1, struct VfeWriteRegsCmd)
#define VFE_WRITE_DMI _IOW(VFE_IOC_MAGIC, 2, struct VfeDmiCmd)
#define VFE_REG_UPDATE _IO(VFE_IOC_MAGIC, 6)

static int v1_ioctl(int fd, unsigned long request, void *arg = nullptr) {
  int ret;
  int try_cnt = 0;
  do {
    ret = ioctl(fd, request, arg);
  } while (ret == -1 && errno == EINTR && try_cnt++ < 100);
  return ret;
}

// The mainline CAMSS graph links VFE_LINE_PIX from CSID source pad 4.
// The CSID-gen2 driver still programs that PIX/IPP path for sensor VC0.
static constexpr int ONE_PIX_CSID_SOURCE_PAD = 4;

static constexpr uint32_t ONE_SENSOR_DELAY_MS = 0xffffffffU;
static constexpr uint32_t OS04_RAW10_20FPS_VTS = 0x1275;

static const std::vector<i2c_random_wr_payload> &os04_default_init_regs() {
  static const std::vector<i2c_random_wr_payload> regs = {
    {0x0103, 0x01},
    {0x0301, 0x84},
    {0x0303, 0x01},
    {0x0305, 0x5b},
    {0x0306, 0x00},
    {0x0307, 0x17},
    {0x0323, 0x04},
    {0x0324, 0x01},
    {0x0325, 0x62},
    {0x3012, 0x06},
    {0x3013, 0x02},
    {0x3016, 0x32},
    {0x3021, 0x03},
    {0x3106, 0x25},
    {0x3107, 0xa1},
    {0x3500, 0x00},
    {0x3501, 0x04},
    {0x3502, 0x40},
    {0x3503, 0x88},
    {0x3508, 0x00},
    {0x3509, 0x80},
    {0x350a, 0x04},
    {0x350b, 0x00},
    {0x350c, 0x00},
    {0x350d, 0x80},
    {0x350e, 0x04},
    {0x350f, 0x00},
    {0x3510, 0x00},
    {0x3511, 0x01},
    {0x3512, 0x20},
    {0x3624, 0x02},
    {0x3625, 0x4c},
    {0x3660, 0x00},
    {0x3666, 0xa5},
    {0x3667, 0xa5},
    {0x366a, 0x64},
    {0x3673, 0x0d},
    {0x3672, 0x0d},
    {0x3671, 0x0d},
    {0x3670, 0x0d},
    {0x3685, 0x00},
    {0x3694, 0x0d},
    {0x3693, 0x0d},
    {0x3692, 0x0d},
    {0x3691, 0x0d},
    {0x3696, 0x4c},
    {0x3697, 0x4c},
    {0x3698, 0x40},
    {0x3699, 0x80},
    {0x369a, 0x18},
    {0x369b, 0x1f},
    {0x369c, 0x14},
    {0x369d, 0x80},
    {0x369e, 0x40},
    {0x369f, 0x21},
    {0x36a0, 0x12},
    {0x36a1, 0x5d},
    {0x36a2, 0x66},
    {0x370a, 0x00},
    {0x370e, 0x0c},
    {0x3710, 0x00},
    {0x3713, 0x00},
    {0x3725, 0x02},
    {0x372a, 0x03},
    {0x3738, 0xce},
    {0x3748, 0x00},
    {0x374a, 0x00},
    {0x374c, 0x00},
    {0x374e, 0x00},
    {0x3756, 0x00},
    {0x3757, 0x0e},
    {0x3767, 0x00},
    {0x3771, 0x00},
    {0x377b, 0x20},
    {0x377c, 0x00},
    {0x377d, 0x0c},
    {0x3781, 0x03},
    {0x3782, 0x00},
    {0x3789, 0x14},
    {0x3795, 0x02},
    {0x379c, 0x00},
    {0x379d, 0x00},
    {0x37b8, 0x04},
    {0x37ba, 0x03},
    {0x37bb, 0x00},
    {0x37bc, 0x04},
    {0x37be, 0x08},
    {0x37c4, 0x11},
    {0x37c5, 0x80},
    {0x37c6, 0x14},
    {0x37c7, 0x08},
    {0x37da, 0x11},
    {0x381f, 0x08},
    {0x3829, 0x03},
    {0x3881, 0x00},
    {0x3888, 0x04},
    {0x388b, 0x00},
    {0x3c80, 0x10},
    {0x3c86, 0x00},
    {0x3c8c, 0x20},
    {0x3c9f, 0x01},
    {0x3d85, 0x1b},
    {0x3d8c, 0x71},
    {0x3d8d, 0xe2},
    {0x3f00, 0x0b},
    {0x3f06, 0x04},
    {0x400a, 0x01},
    {0x400b, 0x50},
    {0x400e, 0x08},
    {0x4043, 0x7e},
    {0x4045, 0x7e},
    {0x4047, 0x7e},
    {0x4049, 0x7e},
    {0x4090, 0x14},
    {0x40b0, 0x00},
    {0x40b1, 0x00},
    {0x40b2, 0x00},
    {0x40b3, 0x00},
    {0x40b4, 0x00},
    {0x40b5, 0x00},
    {0x40b7, 0x00},
    {0x40b8, 0x00},
    {0x40b9, 0x00},
    {0x40ba, 0x00},
    {0x4301, 0x00},
    {0x4303, 0x00},
    {0x4502, 0x04},
    {0x4503, 0x00},
    {0x4504, 0x06},
    {0x4506, 0x00},
    {0x4507, 0x64},
    {0x4803, 0x10},
    {0x480c, 0x32},
    {0x480e, 0x00},
    {0x4813, 0x00},
    {0x4819, 0x70},
    {0x481f, 0x30},
    {0x4823, 0x3c},
    {0x4825, 0x32},
    {0x4833, 0x10},
    {0x484b, 0x07},
    {0x488b, 0x00},
    {0x4d00, 0x04},
    {0x4d01, 0xad},
    {0x4d02, 0xbc},
    {0x4d03, 0xa1},
    {0x4d04, 0x1f},
    {0x4d05, 0x4c},
    {0x4d0b, 0x01},
    {0x4e00, 0x2a},
    {0x4e0d, 0x00},
    {0x5001, 0x09},
    {0x5004, 0x00},
    {0x5080, 0x04},
    {0x5036, 0x00},
    {0x5180, 0x70},
    {0x5181, 0x10},
    {0x520a, 0x03},
    {0x520b, 0x06},
    {0x520c, 0x0c},
    {0x580b, 0x0f},
    {0x580d, 0x00},
    {0x580f, 0x00},
    {0x5820, 0x00},
    {0x5821, 0x00},
    {0x301c, 0xf0},
    {0x301e, 0xb4},
    {0x301f, 0xd0},
    {0x3022, 0x01},
    {0x3109, 0xe7},
    {0x3600, 0x00},
    {0x3610, 0x65},
    {0x3611, 0x85},
    {0x3613, 0x3a},
    {0x3615, 0x60},
    {0x3621, 0x90},
    {0x3620, 0x0c},
    {0x3629, 0x00},
    {0x3661, 0x04},
    {0x3664, 0x70},
    {0x3665, 0x00},
    {0x3681, 0xa6},
    {0x3682, 0x53},
    {0x3683, 0x2a},
    {0x3684, 0x15},
    {0x3700, 0x2a},
    {0x3701, 0x12},
    {0x3703, 0x28},
    {0x3704, 0x0e},
    {0x3706, 0x4a},
    {0x3709, 0x4a},
    {0x370b, 0xa2},
    {0x370c, 0x01},
    {0x370f, 0x04},
    {0x3714, 0x24},
    {0x3716, 0x24},
    {0x3719, 0x11},
    {0x371a, 0x1e},
    {0x3720, 0x00},
    {0x3724, 0x13},
    {0x373f, 0xb0},
    {0x3741, 0x4a},
    {0x3743, 0x4a},
    {0x3745, 0x4a},
    {0x3747, 0x4a},
    {0x3749, 0xa2},
    {0x374b, 0xa2},
    {0x374d, 0xa2},
    {0x374f, 0xa2},
    {0x3755, 0x10},
    {0x376c, 0x00},
    {0x378d, 0x30},
    {0x3790, 0x4a},
    {0x3791, 0xa2},
    {0x3798, 0xc0},
    {0x379e, 0x00},
    {0x379f, 0x04},
    {0x37a1, 0x01},
    {0x37a2, 0x1e},
    {0x37a8, 0x01},
    {0x37a9, 0x1e},
    {0x37ac, 0xa0},
    {0x37b9, 0x01},
    {0x37bd, 0x01},
    {0x37bf, 0x26},
    {0x37c0, 0x11},
    {0x37c2, 0x04},
    {0x37cd, 0x19},
    {0x37e0, 0x08},
    {0x37e6, 0x04},
    {0x37e5, 0x02},
    {0x37e1, 0x0c},
    {0x3737, 0x04},
    {0x37d8, 0x02},
    {0x37e2, 0x10},
    {0x3739, 0x10},
    {0x3662, 0x10},
    {0x37e4, 0x20},
    {0x37e3, 0x08},
    {0x37d9, 0x08},
    {0x4040, 0x00},
    {0x4041, 0x07},
    {0x4008, 0x02},
    {0x4009, 0x0d},
    {0x3800, 0x00},
    {0x3801, 0x00},
    {0x3802, 0x00},
    {0x3803, 0x00},
    {0x3804, 0x0a},
    {0x3805, 0x8f},
    {0x3806, 0x05},
    {0x3807, 0xff},
    {0x3808, 0x0a},
    {0x3809, 0x80},
    {0x380a, 0x05},
    {0x380b, 0xf0},
    {0x380c, 0x04},
    {0x380d, 0x2e},
    {0x380e, 0x12},
    {0x380f, 0x75},
    {0x3811, 0x09},
    {0x3813, 0x09},
    {0x3814, 0x01},
    {0x3815, 0x01},
    {0x3816, 0x01},
    {0x3817, 0x01},
    {0x3820, 0x88},
    {0x3821, 0x00},
    {0x3880, 0x25},
    {0x3882, 0x20},
    {0x3c91, 0x0b},
    {0x3c94, 0x45},
    {0x4000, 0xf3},
    {0x4001, 0x60},
    {0x4003, 0x40},
    {0x4300, 0xff},
    {0x4302, 0x0f},
    {0x4305, 0x83},
    {0x4505, 0x84},
    {0x4809, 0x1e},
    {0x480a, 0x04},
    {0x4837, 0x0a},
    {0x4c00, 0x08},
    {0x4c01, 0x00},
    {0x4c04, 0x00},
    {0x4c05, 0x00},
    {0x5000, 0xf9},
    {0x3624, 0x00},
    {0x3822, 0x14},
    {0x0100, 0x00},
  };
  return regs;
}

static void disable_csid_tpg(int csid_subdev, int cam_idx) {
  if (csid_subdev < 0) return;

  int fd = open(util::string_format("/dev/v4l-subdev%d", csid_subdev).c_str(), O_RDWR);
  if (fd < 0) return;

  struct v4l2_control ctrl = {};
  ctrl.id = V4L2_CID_TEST_PATTERN;
  ctrl.value = 0;
  if (ioctl(fd, VIDIOC_S_CTRL, &ctrl) != 0) {
    LOGE("cam %d: disabling CSID TPG failed: %d (%s)", cam_idx, errno, strerror(errno));
  } else {
    LOG("cam %d: disabled CSID TPG", cam_idx);
  }

  close(fd);
}

static bool write_sensor_regs(int sensor_fd, const std::vector<i2c_random_wr_payload> &reg_array,
                              const char *name, int cam_idx) {
  if (reg_array.empty()) return true;

  std::vector<OneSensorRegWrite> regs;
  regs.reserve(reg_array.size());
  size_t regs_written = 0;

  auto flush_regs = [&]() {
    if (regs.empty()) return true;

    OneSensorWriteRegsCmd cmd = {};
    cmd.regs = (uint64_t)(uintptr_t)regs.data();
    cmd.count = regs.size();
    cmd.data_width = 1;

    int ret = HANDLE_EINTR(ioctl(sensor_fd, ONE_SENSOR_WRITE_REGS, &cmd));
    if (ret != 0) {
      LOGE("cam %d: failed to write %s sensor regs (%zu): %d (%s)",
           cam_idx, name, regs.size(), errno, strerror(errno));
      return false;
    }

    regs_written += regs.size();
    regs.clear();
    return true;
  };

  for (const auto &r : reg_array) {
    if (r.reg_addr == ONE_SENSOR_DELAY_MS) {
      if (!flush_regs()) return false;

      const uint32_t delay_ms = std::min(r.reg_data, 10000U);
      LOG("cam %d: delaying %u ms during %s sensor regs", cam_idx, delay_ms, name);
      usleep(delay_ms * 1000U);
      continue;
    }

    regs.push_back({(uint16_t)r.reg_addr, (uint16_t)r.reg_data});
    if (r.reg_addr == 0x0103) {
      if (!flush_regs()) return false;
      usleep(5000);
    }
  }

  if (!flush_regs()) return false;

  if (strcmp(name, "exposure") != 0) {
    LOG("cam %d: wrote %zu %s sensor regs", cam_idx, regs_written, name);
  }
  return true;
}

static int v1_physical_cam_num(int camera_num);

struct Os04VfeWbRegs {
  bool valid = false;
  int blue = 0;
  int green = 0;
  int red = 0;
};

static Os04VfeWbRegs default_os04_vfe_wb_regs(int cam_idx) {
  Os04VfeWbRegs wb;

  switch (v1_physical_cam_num(cam_idx)) {
    case 1:
      wb.valid = true;
      wb.blue = 0x00cd;
      wb.green = 0x0080;
      wb.red = 0x00e1;
      break;
    case 2:
      // CamThink road module, tuned against the comma four OS04 output.
      wb.valid = true;
      wb.blue = 0x00b4;
      wb.green = 0x0080;
      wb.red = 0x00d2;
      break;
    case 3:
      // CamThink wide module, balanced for the comma four OS04 CCM below.
      wb.valid = true;
      wb.blue = 0x00bc;
      wb.green = 0x0080;
      wb.red = 0x00d1;
      break;
    default:
      break;
  }
  return wb;
}

static std::vector<VfeRegWrite> os04_vfe_wb_reg_writes(const Os04VfeWbRegs &wb) {
  if (!wb.valid) return {};
  return {
    {0x6fc, ((uint32_t)(wb.blue & 0xffff) << 16) | (uint32_t)(wb.green & 0xffff)},
    {0x700, (uint32_t)(wb.red & 0xffff)},
    {0x704, 0x00000000},
    {0x708, 0x00000000},
  };
}

static std::vector<VfeRegWrite> os04_vfe_demosaic_reg_writes(int cam_idx) {
  if (v1_physical_cam_num(cam_idx) != 3) return {};

  // Match comma four's OS04 IFE state. The Dragon kernel baseline writes the
  // first interpolation coefficient into all 16 directional slots, while the
  // comma pipeline leaves the remaining reset-state slots at zero.
  std::vector<VfeRegWrite> regs = {
    {0x6f8, 0x00000100},
    {0x71c, 0x00008000},
    {0x720, 0x08000066},
  };
  for (uint32_t offset = 0x724; offset <= 0x75c; offset += 4) {
    regs.push_back({offset, 0x00000000});
  }
  return regs;
}

static std::vector<VfeRegWrite> os04_vfe_ccm_reg_writes(int cam_idx) {
  static constexpr uint32_t identity_ccm[] = {
    0x00000080, 0x00000000, 0x00000000,
    0x00000000, 0x00000080, 0x00000000,
    0x00000000, 0x00000000, 0x00000080,
    0x00000000, 0x00000000, 0x00000000,
    0x00000000,
  };
  static constexpr uint32_t c4_os04_ccm[] = {
    0x000000c2, 0x00000fe0, 0x00000fde,
    0x00000fa7, 0x000000d9, 0x00001000,
    0x00000fca, 0x00000fef, 0x000000c7,
    0x00000000, 0x00000000, 0x00000000,
    0x00000000,
  };
  const uint32_t *ccm = v1_physical_cam_num(cam_idx) == 3 ? c4_os04_ccm : identity_ccm;
  std::vector<VfeRegWrite> regs;
  regs.reserve(std::size(identity_ccm));
  for (size_t i = 0; i < std::size(identity_ccm); i++) {
    regs.push_back({0x760 + (uint32_t)i * 4, ccm[i]});
  }
  return regs;
}

static std::vector<VfeRegWrite> os04_vfe_yuv_reg_writes(int cam_idx) {
  if (v1_physical_cam_num(cam_idx) != 3) return {};

  // Use comma four's OS04 RGB-to-YUV conversion unchanged. Module/lens color
  // response is compensated by white balance before this conversion.
  static constexpr uint32_t yuv[] = {
    0x00680208, 0x00000108, 0x00400000, 0x03ff0000,
    0x01c01ed8, 0x00001f68, 0x02000000, 0x03ff0000,
    0x1fb81e88, 0x000001c0, 0x02000000, 0x03ff0000,
  };
  std::vector<VfeRegWrite> regs;
  regs.reserve(std::size(yuv));
  for (size_t i = 0; i < std::size(yuv); i++) {
    regs.push_back({0xf30 + (uint32_t)i * 4, yuv[i]});
  }
  return regs;
}

static std::vector<VfeRegWrite> default_os04_vfe_tuning_regs(int cam_idx) {
  // Stream start resets CORE_CFG; restore the CamThink module's RGGB phase.
  std::vector<VfeRegWrite> regs = {{0x050, 0x00000000}};
  std::vector<VfeRegWrite> wb = os04_vfe_wb_reg_writes(default_os04_vfe_wb_regs(cam_idx));
  regs.insert(regs.end(), wb.begin(), wb.end());
  std::vector<VfeRegWrite> demosaic = os04_vfe_demosaic_reg_writes(cam_idx);
  regs.insert(regs.end(), demosaic.begin(), demosaic.end());
  std::vector<VfeRegWrite> ccm = os04_vfe_ccm_reg_writes(cam_idx);
  regs.insert(regs.end(), ccm.begin(), ccm.end());
  std::vector<VfeRegWrite> yuv = os04_vfe_yuv_reg_writes(cam_idx);
  regs.insert(regs.end(), yuv.begin(), yuv.end());
  return regs;
}

static bool apply_os04_20fps_timing(int sensor_fd, int cam_idx) {
  return write_sensor_regs(sensor_fd, {
    {0x380e, (OS04_RAW10_20FPS_VTS >> 8) & 0xff},
    {0x380f, OS04_RAW10_20FPS_VTS & 0xff},
  }, "20fps timing", cam_idx);
}

static int v1_physical_cam_num(int camera_num) {
  // openpilot camera_num 0/1/2 maps to physical CAM3/CAM2/CAM1.
  static const int physical[] = {3, 2, 1};
  return (camera_num >= 0 && camera_num < 3) ? physical[camera_num] : camera_num;
}

class OneCamera {
public:
  CameraConfig cc;
  std::unique_ptr<SensorInfo> sensor;
  bool enabled;

  int video_fd = -1;
  int sensor_fd = -1;
  bool streaming = false;

  int n_bufs = 4;
  std::unique_ptr<VisionBuf[]> vfe_buffers;

  uint32_t output_width = 0, output_height = 0;
  uint32_t stride = 0, y_height = 0, uv_height = 0, yuv_size = 0, uv_offset = 0;
  uint32_t vipc_stride = 0, vipc_y_height = 0, vipc_uv_height = 0, vipc_yuv_size = 0, vipc_uv_offset = 0;

  OneCamera(const CameraConfig &config) : cc(config), enabled(true) {}

  void camera_open(VisionIpcServer *v);
  void camera_close();
  void setup_media_links();
  void set_formats();
  bool write_vfe_regs(const std::vector<VfeRegWrite> &regs, const char *name);
  bool apply_vfe_tuning();
  bool apply_vfe_gamma();
  void queue_all_buffers();
  void stream_on();
  void stop_streaming();
  int dequeue_frame(uint64_t *timestamp);
  void queue_frame(int index);
  void set_exposure(int exposure_time, int gain_idx);

  VisionIpcServer *vipc_server = nullptr;
  VisionStreamType stream_type;
};


static void reset_all_media_links() {
  int media_fd = open("/dev/media0", O_RDWR);
  if (media_fd < 0) return;

  for (int i = 0; i < 20; i++) {
    std::string vpath = util::string_format("/dev/video%d", i);
    int vfd = open(vpath.c_str(), O_RDWR);
    if (vfd >= 0) {
      int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
      ioctl(vfd, VIDIOC_STREAMOFF, &type);
      type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      ioctl(vfd, VIDIOC_STREAMOFF, &type);
      close(vfd);
    }
  }

  std::vector<uint32_t> csid_ents, vfe_ents, csiphy_ents;
  struct media_entity_desc ent = {};
  for (ent.id = 0 | MEDIA_ENT_ID_FLAG_NEXT; ; ent.id |= MEDIA_ENT_ID_FLAG_NEXT) {
    if (ioctl(media_fd, MEDIA_IOC_ENUM_ENTITIES, &ent) < 0) break;
    if (strncmp(ent.name, "msm_csiphy", 10) == 0) csiphy_ents.push_back(ent.id);
    if (strncmp(ent.name, "msm_csid", 8) == 0) csid_ents.push_back(ent.id);
    if (strncmp(ent.name, "msm_vfe", 7) == 0) vfe_ents.push_back(ent.id);
  }
  for (uint32_t csiphy : csiphy_ents) {
    for (uint32_t csid : csid_ents) {
      struct media_link_desc link = {};
      link.source = {.entity = csiphy, .index = 1};
      link.sink = {.entity = csid, .index = 0};
      link.flags = 0;
      ioctl(media_fd, MEDIA_IOC_SETUP_LINK, &link);
    }
  }
  for (uint32_t csid : csid_ents) {
    for (uint32_t vfe : vfe_ents) {
      for (int pad = 1; pad <= 4; pad++) {
        struct media_link_desc link = {};
        link.source = {.entity = csid, .index = (uint16_t)pad};
        link.sink = {.entity = vfe, .index = 0};
        link.flags = 0;
        ioctl(media_fd, MEDIA_IOC_SETUP_LINK, &link);
      }
    }
  }
  close(media_fd);
  LOG("reset all media links");
}

void OneCamera::setup_media_links() {
  int media_fd = open("/dev/media0", O_RDWR);
  if (media_fd < 0) {
    LOGE("failed to open /dev/media0");
    return;
  }

  int cam_idx = cc.camera_num;
  auto &dcfg = v1_cams[cam_idx];

  struct media_link_desc link = {};

  disable_csid_tpg(dcfg.csid_subdev, cam_idx);

  // CSIPHY -> CSID (source pad 1 -> sink pad 0)
  link.source = {.entity = (uint32_t)dcfg.csiphy_entity, .index = 1};
  link.sink = {.entity = (uint32_t)dcfg.csid_entity, .index = 0};
  link.flags = MEDIA_LNK_FL_ENABLED;
  if (ioctl(media_fd, MEDIA_IOC_SETUP_LINK, &link) != 0)
    LOGE("cam %d: csiphy->csid link FAILED: %d (%s)", cam_idx, errno, strerror(errno));
  memset(&link, 0, sizeof(link));

  // Mainline CAMSS exposes the PIX path on the CSID source pad selected here.
  link.source = {.entity = (uint32_t)dcfg.csid_entity, .index = ONE_PIX_CSID_SOURCE_PAD};
  link.sink = {.entity = (uint32_t)dcfg.vfe_pix_entity, .index = 0};
  link.flags = MEDIA_LNK_FL_ENABLED;
  if (ioctl(media_fd, MEDIA_IOC_SETUP_LINK, &link) != 0)
    LOGE("cam %d: csid->vfe PIX link FAILED: %d (%s)", cam_idx, errno, strerror(errno));

  close(media_fd);
  LOG("cam %d: media links set up (VFE PIX mode)", cam_idx);
}

void OneCamera::set_formats() {
  int cam_idx = cc.camera_num;
  auto &dcfg = v1_cams[cam_idx];
  constexpr uint32_t media_bus_code = MEDIA_BUS_FMT_SBGGR10_1X10;
  // set format on CSIPHY subdev
  int csiphy_fd = open(util::string_format("/dev/v4l-subdev%d", dcfg.csiphy_subdev).c_str(), O_RDWR);
  if (csiphy_fd >= 0) {
    struct v4l2_subdev_format sfmt = {};
    sfmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    sfmt.pad = 0;
    sfmt.format.width = sensor->frame_width;
    sfmt.format.height = sensor->frame_height;
    sfmt.format.code = media_bus_code;
    ioctl(csiphy_fd, VIDIOC_SUBDEV_S_FMT, &sfmt);
    sfmt.pad = 1;
    ioctl(csiphy_fd, VIDIOC_SUBDEV_S_FMT, &sfmt);
    close(csiphy_fd);
  }

  // set format on CSID subdev
  int csid_fd = open(util::string_format("/dev/v4l-subdev%d", dcfg.csid_subdev).c_str(), O_RDWR);
  if (csid_fd >= 0) {
    struct v4l2_subdev_format sfmt = {};
    sfmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    sfmt.pad = 0;
    sfmt.format.width = sensor->frame_width;
    sfmt.format.height = sensor->frame_height;
    sfmt.format.code = media_bus_code;
    ioctl(csid_fd, VIDIOC_SUBDEV_S_FMT, &sfmt);

    // PIX source pad matching the selected virtual channel.
    sfmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    sfmt.pad = ONE_PIX_CSID_SOURCE_PAD;
    sfmt.format.width = sensor->frame_width;
    sfmt.format.height = sensor->frame_height;
    sfmt.format.code = media_bus_code;
    ioctl(csid_fd, VIDIOC_SUBDEV_S_FMT, &sfmt);
    close(csid_fd);
  }

  // set format on sensor subdev
  if (sensor_fd >= 0) {
    struct v4l2_subdev_format sfmt = {};
    sfmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    sfmt.pad = 0;
    sfmt.format.width = sensor->frame_width;
    sfmt.format.height = sensor->frame_height;
    sfmt.format.code = media_bus_code;
    ioctl(sensor_fd, VIDIOC_SUBDEV_S_FMT, &sfmt);
  }

  int vfe_pix_fd = open(util::string_format("/dev/v4l-subdev%d", dcfg.vfe_pix_subdev).c_str(), O_RDWR);
  if (vfe_pix_fd >= 0) {
    struct v4l2_subdev_format sfmt = {};
    sfmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    sfmt.pad = 0;
    sfmt.format.width = sensor->frame_width;
    sfmt.format.height = sensor->frame_height;
    sfmt.format.code = media_bus_code;
    ioctl(vfe_pix_fd, VIDIOC_SUBDEV_S_FMT, &sfmt);

    struct v4l2_subdev_selection sel = {};
    sel.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    sel.pad = 0;
    sel.target = V4L2_SEL_TGT_COMPOSE;
    sel.r.left = 0;
    sel.r.top = 0;
    sel.r.width = output_width;
    sel.r.height = output_height;
    if (ioctl(vfe_pix_fd, VIDIOC_SUBDEV_S_SELECTION, &sel) != 0) {
      LOGE("cam %d: VFE PIX S_SELECTION compose failed: %d (%s)", cam_idx, errno, strerror(errno));
    }

    sfmt.pad = 1;
    sfmt.format.width = output_width;
    sfmt.format.height = output_height;
    sfmt.format.code = MEDIA_BUS_FMT_YUYV8_1_5X8;
    if (ioctl(vfe_pix_fd, VIDIOC_SUBDEV_S_FMT, &sfmt) != 0) {
      LOGE("cam %d: VFE PIX source S_FMT failed: %d (%s)", cam_idx, errno, strerror(errno));
    } else {
      LOG("cam %d: VFE PIX source format %ux%u code=0x%x",
          cam_idx, sfmt.format.width, sfmt.format.height, sfmt.format.code);
    }
    close(vfe_pix_fd);
  }

  struct v4l2_format vfmt = {};
  vfmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  vfmt.fmt.pix_mp.width = output_width;
  vfmt.fmt.pix_mp.height = output_height;
  vfmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
  vfmt.fmt.pix_mp.num_planes = 1;
  vfmt.fmt.pix_mp.plane_fmt[0].bytesperline = stride;
  vfmt.fmt.pix_mp.plane_fmt[0].sizeimage = yuv_size;
  if (ioctl(video_fd, VIDIOC_S_FMT, &vfmt) == 0) {
    stride = vfmt.fmt.pix_mp.plane_fmt[0].bytesperline;
    y_height = output_height;
    uv_height = (output_height + 1) / 2;
    uv_offset = stride * y_height;
    yuv_size = vfmt.fmt.pix_mp.plane_fmt[0].sizeimage;
    LOG("cam %d: VFE PIX format set raw=%dx%d out=%dx%d stride=%u size=%u video_stride=%u sizeimage=%u",
        cam_idx, sensor->frame_width, sensor->frame_height, vfmt.fmt.pix_mp.width, vfmt.fmt.pix_mp.height,
        stride, yuv_size, vfmt.fmt.pix_mp.plane_fmt[0].bytesperline, vfmt.fmt.pix_mp.plane_fmt[0].sizeimage);
  } else {
    LOGE("cam %d: VFE PIX S_FMT failed: %d (%s)", cam_idx, errno, strerror(errno));
  }
}

bool OneCamera::write_vfe_regs(const std::vector<VfeRegWrite> &regs, const char *name) {
  if (video_fd < 0) return false;
  if (regs.empty()) return true;

  for (size_t offset = 0; offset < regs.size(); ) {
    const size_t count = std::min<size_t>(regs.size() - offset, 1024);
    VfeWriteRegsCmd cmd = {};
    cmd.regs = (uint64_t)(uintptr_t)(regs.data() + offset);
    cmd.count = count;
    if (v1_ioctl(video_fd, VFE_WRITE_REGS, &cmd) != 0) {
      LOGE("cam %d: failed to write %s VFE regs offset=%zu count=%zu: %d (%s)",
           cc.camera_num, name, offset, count, errno, strerror(errno));
      return false;
    }
    offset += count;
  }

  if (v1_ioctl(video_fd, VFE_REG_UPDATE) != 0) {
    LOGE("cam %d: failed to commit %s VFE regs: %d (%s)",
         cc.camera_num, name, errno, strerror(errno));
    return false;
  }

  LOG("cam %d: wrote %zu %s VFE regs", cc.camera_num, regs.size(), name);
  return true;
}

bool OneCamera::apply_vfe_tuning() {
  return write_vfe_regs(default_os04_vfe_tuning_regs(cc.camera_num), "OS04 tuning");
}

bool OneCamera::apply_vfe_gamma() {
  const uint8_t banks[] = {26, 28, 30};
  for (const uint8_t bank : banks) {
    VfeDmiCmd cmd = {};
    cmd.dmi_cfg_offset = 0xc24;
    cmd.ram_select = bank;
    cmd.count = sensor->gamma_lut_rgb.size();
    cmd.data = (uint64_t)(uintptr_t)sensor->gamma_lut_rgb.data();
    if (v1_ioctl(video_fd, VFE_WRITE_DMI, &cmd) != 0) {
      LOGE("cam %d: failed to write OS04 gamma DMI ram=%u: %d (%s)",
           cc.camera_num, bank, errno, strerror(errno));
      return false;
    }
  }

  if (v1_ioctl(video_fd, VFE_REG_UPDATE) != 0) {
    LOGE("cam %d: failed to commit OS04 gamma DMI: %d (%s)",
         cc.camera_num, errno, strerror(errno));
    return false;
  }

  LOG("cam %d: wrote standard OS04 gamma DMI", cc.camera_num);
  return true;
}

void OneCamera::camera_open(VisionIpcServer *v) {
  if (!enabled) return;

  vipc_server = v;
  stream_type = cc.stream_type;

  int cam_idx = cc.camera_num;
  auto &dcfg = v1_cams[cam_idx];
  sensor = std::make_unique<OS04C10>();
  LOG("cam %d: using OS04C10 RAW10 media path", cam_idx);
  sensor->bits_per_pixel = 10;
  sensor->mipi_format = CAM_FORMAT_MIPI_RAW_10;
  sensor->frame_data_type = CSI_RAW10;
  sensor->frame_stride = sensor->frame_width * 10 / 8;
  // This RAW10 mode uses HTS=1070, half the qcom2 mode's 2140. Scale EV by
  // half as much so target-grey calculations represent the same exposure.
  sensor->ev_scale = 75.0f;
  sensor->exposure_time_max = 4717;
  sensor->analog_gain_max_idx = 0x1e;
  sensor->max_ev = sensor->exposure_time_max * sensor->dc_gain_factor *
                   sensor->sensor_analog_gains[sensor->analog_gain_max_idx];

  if (dcfg.vfe_pix_entity == 0 || dcfg.pix_video_dev < 0 || dcfg.vfe_pix_subdev < 0) {
    LOGE("cam %d: required VFE PIX path is unavailable "
         "(entity=%u video=%d subdev=%d)",
         cam_idx, dcfg.vfe_pix_entity, dcfg.pix_video_dev, dcfg.vfe_pix_subdev);
    enabled = false;
    return;
  }

  const int output_scale = sensor->out_scale;
  output_width = std::max(2U, (sensor->frame_width / output_scale) & ~1U);
  output_height = std::max(2U, (sensor->frame_height / output_scale) & ~1U);
  auto [s, yh, uvh, sz] = get_nv12_info(output_width, output_height);
  vipc_stride = stride = s;
  vipc_y_height = y_height = yh;
  vipc_uv_height = uv_height = uvh;
  vipc_yuv_size = yuv_size = sz;
  vipc_uv_offset = uv_offset = stride * y_height;

  // open video device
  int dev = dcfg.pix_video_dev;
  std::string path = util::string_format("/dev/video%d", dev);
  video_fd = open(path.c_str(), O_RDWR);
  if (video_fd < 0) {
    LOGE("cam %d: failed to open %s: %d", cam_idx, path.c_str(), errno);
    enabled = false;
    return;
  }
  LOG("cam %d: opened %s (VFE PIX V4L2 DMABUF)", cam_idx, path.c_str());

  // find sensor subdev. The Dragon exposes enough CAMSS entities that the
  // third OS04 can land above v4l-subdev31.
  for (int i = 0; i < 96; i++) {
    const std::string name_path = util::string_format("/sys/class/video4linux/v4l-subdev%d/name", i);
    if (access(name_path.c_str(), R_OK) != 0) continue;
    std::string name = util::read_file(name_path);
    if (name.find(dcfg.sensor_name) == 0) {
      sensor_fd = open(util::string_format("/dev/v4l-subdev%d", i).c_str(), O_RDWR);
      break;
    }
  }
  if (sensor_fd < 0) {
    LOGE("cam %d: sensor subdev '%s' not found, disabling", cam_idx, dcfg.sensor_name);
    enabled = false;
    return;
  }

  if (!write_sensor_regs(sensor_fd, os04_default_init_regs(), "init file", cam_idx)) {
    enabled = false;
    return;
  }

  if (!apply_os04_20fps_timing(sensor_fd, cam_idx)) {
    enabled = false;
    return;
  }

  // Do not enable a CAMSS route until the sensor has responded. Leaving an
  // absent camera's media links enabled can interfere with working cameras.
  setup_media_links();
  set_formats();

  if (access("/dev/dma_heap/system", R_OK | W_OK) != 0) {
    LOGE("cam %d: required DMA heap is unavailable: %d (%s)", cam_idx, errno, strerror(errno));
    enabled = false;
    return;
  }

  // V4L2 buffers for normal VFE PIX NV12 frames.
  struct v4l2_requestbuffers req = {};
  req.count = n_bufs;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  req.memory = V4L2_MEMORY_DMABUF;
  int reqbufs_ret = ioctl(video_fd, VIDIOC_REQBUFS, &req);
  if (reqbufs_ret != 0) {
    LOGE("cam %d: REQBUFS DMABUF failed, disabling camera instead of using V4L2 MMAP CPU-copy path: %d (%s)",
         cam_idx, errno, strerror(errno));
    enabled = false;
    return;
  }
  n_bufs = req.count;

  // The Dragon VFE requires a wider native capture stride than the standard
  // Venus NV12 layout used by comma's VisionIPC consumers. Capture into a
  // private VFE ring, then publish normalized buffers below.
  vfe_buffers = std::make_unique<VisionBuf[]>(n_bufs);
  for (int i = 0; i < n_bufs; i++) {
    vfe_buffers[i].allocate(yuv_size);
  }

  v->create_buffers_with_sizes(stream_type, VIPC_BUFFER_COUNT,
                               output_width, output_height,
                               vipc_yuv_size, vipc_stride, vipc_uv_offset);

  LOG("cam %d: VIPC buffers created (%s, %ux%u, scale=%d, %u bytes, stride=%u; VFE stride=%u)",
      cam_idx, "VFE PIX V4L2 DMABUF NV12",
      output_width, output_height, output_scale, vipc_yuv_size, vipc_stride, stride);
}

void OneCamera::queue_all_buffers() {
  if (!enabled) return;

  for (int i = 0; i < n_bufs; i++) {
    queue_frame(i);
  }
}

void OneCamera::stream_on() {
  if (!enabled) return;

  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  if (ioctl(video_fd, VIDIOC_STREAMON, &type) != 0) {
    LOGE("cam %d: STREAMON failed: %d (%s)", cc.camera_num, errno, strerror(errno));
    enabled = false;
    return;
  }

  if (!write_sensor_regs(sensor_fd, sensor->start_reg_array, "start", cc.camera_num)) {
    ioctl(video_fd, VIDIOC_STREAMOFF, &type);
    enabled = false;
    return;
  }

  if (!apply_vfe_tuning()) {
    ioctl(video_fd, VIDIOC_STREAMOFF, &type);
    enabled = false;
    return;
  }

  if (!apply_vfe_gamma()) {
    ioctl(video_fd, VIDIOC_STREAMOFF, &type);
    enabled = false;
    return;
  }

  streaming = true;
  LOG("cam %d: VFE PIX V4L2 streaming started", cc.camera_num);
}

void OneCamera::stop_streaming() {
  if (streaming && sensor_fd >= 0) {
    write_sensor_regs(sensor_fd, {{0x100, 0}}, "stop", cc.camera_num);
  }

  if (video_fd >= 0 && streaming) {
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    ioctl(video_fd, VIDIOC_STREAMOFF, &type);
  }
  streaming = false;
}

void OneCamera::queue_frame(int index) {
  struct v4l2_buffer vbuf = {};
  struct v4l2_plane planes[1] = {};
  vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  vbuf.memory = V4L2_MEMORY_DMABUF;
  vbuf.index = index;
  vbuf.length = 1;
  vbuf.m.planes = planes;
  planes[0].m.fd = vfe_buffers[index].fd;
  planes[0].length = yuv_size;
  int ret = ioctl(video_fd, VIDIOC_QBUF, &vbuf);
  if (ret != 0) LOGE("cam %d: QBUF idx=%d failed: %d (%s)", cc.camera_num, index, errno, strerror(errno));
}

int OneCamera::dequeue_frame(uint64_t *timestamp) {
  struct pollfd pfd = {video_fd, POLLIN, 0};
  int ret = poll(&pfd, 1, 20);
  if (ret == 0) return -ETIMEDOUT;
  if (ret < 0) return -errno;
  if (!(pfd.revents & POLLIN)) return -EIO;

  struct v4l2_buffer dbuf = {};
  struct v4l2_plane planes[1] = {};
  dbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  dbuf.memory = V4L2_MEMORY_DMABUF;
  dbuf.length = 1;
  dbuf.m.planes = planes;
  if (ioctl(video_fd, VIDIOC_DQBUF, &dbuf) != 0) return -errno;

  *timestamp = (uint64_t)dbuf.timestamp.tv_sec * 1000000000ULL +
               (uint64_t)dbuf.timestamp.tv_usec * 1000ULL;
  return dbuf.index;
}

void OneCamera::set_exposure(int exposure_time, int gain_idx) {
  if (sensor_fd < 0) return;

  write_sensor_regs(sensor_fd, sensor->getExposureRegisters(exposure_time, gain_idx, false),
                    "exposure", cc.camera_num);
}

void OneCamera::camera_close() {
  if (video_fd >= 0) {
    stop_streaming();
    close(video_fd);
    video_fd = -1;
  }
  if (sensor_fd >= 0) {
    close(sensor_fd);
    sensor_fd = -1;
  }
  if (vfe_buffers != nullptr) {
    for (int i = 0; i < n_bufs; i++) {
      vfe_buffers[i].free();
    }
    vfe_buffers.reset();
  }
}

struct Os04AeSample {
  float grey_frac = 0.5f;
};

static constexpr int OS04_AE_HISTORY_SIZE = 4;
static constexpr int OS04_EXPOSURE_DELAY_FRAMES = 3;

class CameraState {
public:
  OneCamera camera;
  uint64_t last_frame_ns = 0;
  int exposure_time = 1600;
  int gain_idx = 8;
  float current_ev = 0;
  float os04_ev_history[OS04_AE_HISTORY_SIZE] = {};
  float best_ev_score = 0;
  int new_exp_g = 0;
  int new_exp_t = 0;

  Rect ae_xywh = {};
  float measured_grey_fraction = 0;
  float target_grey_fraction = 0.125;

  uint32_t frame_id = 0;
  std::unique_ptr<PubMaster> pm;
  float fl_pix = 0;

  CameraState(const CameraConfig &config) : camera(config) {}
  ~CameraState() { camera.camera_close(); }

  void init(VisionIpcServer *v);
  void process_pix_frame(int buf_idx, uint64_t timestamp);
  void update_exposure_score(float desired_ev, int exp_t, int exp_g_idx, float exp_gain);
  void set_camera_exposure(const Os04AeSample &ae_sample);
  void set_exposure_rect();

};

void CameraState::init(VisionIpcServer *v) {
  camera.camera_open(v);
  if (!camera.enabled) return;

  // The v1 starts the road sensor first, then wide almost one period later.
  // Number wide one frame ahead so same-numbered road/wide frames refer to the
  // same 20 Hz capture instant.
  if (camera.cc.stream_type == VISION_STREAM_WIDE_ROAD) {
    frame_id = 1;
  }

  fl_pix = camera.cc.focal_len / camera.sensor->pixel_size_mm;
  pm = std::make_unique<PubMaster>(std::vector{camera.cc.publish_name});

  exposure_time = std::clamp(600, camera.sensor->exposure_time_min, camera.sensor->exposure_time_max);
  gain_idx = camera.sensor->analog_gain_rec_idx;

  float gain = camera.sensor->sensor_analog_gains[gain_idx];
  current_ev = gain * exposure_time;
  std::fill(std::begin(os04_ev_history), std::end(os04_ev_history), gain * exposure_time);
  camera.set_exposure(exposure_time, gain_idx);

  set_exposure_rect();
}

void CameraState::set_exposure_rect() {
  // AE rectangle for NV12 frames
  const int width = camera.output_width ? camera.output_width : camera.sensor->frame_width;
  const int height = camera.output_height ? camera.output_height : camera.sensor->frame_height;
  ae_xywh = {
    (int)(width * 0.05f),
    (int)(height * 0.15f),
    (int)(width * 0.9f),
    (int)(height * 0.75f),
  };
}

static Os04AeSample calculate_os04_ae_sample_nv12(const uint8_t *base, int stride, Rect ae_xywh,
                                                   int x_skip, int y_skip) {
  Os04AeSample ret;
  if (base == nullptr || stride <= 0) return ret;

  int lum_med;
  uint32_t lum_binning[256] = {0};

  unsigned int lum_total = 0;

  for (int y = ae_xywh.y; y < ae_xywh.y + ae_xywh.h; y += y_skip) {
    for (int x = ae_xywh.x; x < ae_xywh.x + ae_xywh.w; x += x_skip) {
      uint8_t lum = base[(y * stride) + x];
      lum_binning[lum]++;
      lum_total += 1;
    }
  }
  if (lum_total == 0) return ret;

  unsigned int lum_cur = 0;
  for (lum_med = 255; lum_med >= 0; lum_med--) {
    lum_cur += lum_binning[lum_med];
    if (lum_cur >= lum_total / 2) break;
  }

  ret.grey_frac = lum_med / 256.0f;
  return ret;
}

void CameraState::update_exposure_score(float desired_ev, int exp_t, int exp_g_idx, float exp_gain) {
  float score = camera.sensor->getExposureScore(desired_ev, exp_t, exp_g_idx, exp_gain, gain_idx);
  if (score < best_ev_score) {
    new_exp_t = exp_t;
    new_exp_g = exp_g_idx;
    best_ev_score = score;
  }
}

void CameraState::set_camera_exposure(const Os04AeSample &ae_sample) {
  if (!camera.enabled) return;

  const float dt = 0.05;
  const float ts_grey = 10.0;
  const float ts_ev = 0.05;
  const float k_grey = (dt / ts_grey) / (1.0 + dt / ts_grey);
  const float k_ev = (dt / ts_ev) / (1.0 + dt / ts_ev);

  const auto &sens = camera.sensor;
  const int old_exp_t = exposure_time;
  const int old_gain_idx = gain_idx;

  const float cur_ev_ =
      os04_ev_history[(frame_id + OS04_AE_HISTORY_SIZE - OS04_EXPOSURE_DELAY_FRAMES) %
                      OS04_AE_HISTORY_SIZE];
  const float scaled_ev = cur_ev_ * sens->ev_scale;
  const float new_target_grey = std::clamp(
      0.4f - 0.3f * std::log2(1.0f + sens->target_grey_factor * scaled_ev) / std::log2(6000.0f),
      0.1f, 0.4f);
  float target_grey = (1.0f - k_grey) * target_grey_fraction + k_grey * new_target_grey;

  const float grey_frac = std::clamp(ae_sample.grey_frac, 1.0f / 256.0f, 1.0f);
  float desired_ev = std::clamp(cur_ev_ * target_grey / grey_frac, sens->min_ev, sens->max_ev);
  float history_ev = 0.0f;
  for (int i = 0; i < 3; i++) {
    history_ev += os04_ev_history[(frame_id + OS04_AE_HISTORY_SIZE - 1 - i) % OS04_AE_HISTORY_SIZE] / 3.0f;
  }
  desired_ev = (1.0f - k_ev) * history_ev + k_ev * desired_ev;

  best_ev_score = 1e6;
  new_exp_g = 0;
  new_exp_t = 0;

  constexpr int gain_step = 4;
  int min_g = std::max(gain_idx - gain_step, sens->analog_gain_min_idx);
  int max_g = std::min(gain_idx + gain_step, sens->analog_gain_max_idx);
  for (int g = min_g; g <= max_g; g++) {
    float gain = sens->sensor_analog_gains[g];
    int t = std::clamp((int)std::round(desired_ev / gain), sens->exposure_time_min, sens->exposure_time_max);
    update_exposure_score(desired_ev, t, g, gain);
  }

  measured_grey_fraction = grey_frac;
  target_grey_fraction = target_grey;
  gain_idx = new_exp_g;
  exposure_time = new_exp_t;

  const float new_ev = exposure_time * sens->sensor_analog_gains[gain_idx];
  current_ev = new_ev;
  os04_ev_history[frame_id % OS04_AE_HISTORY_SIZE] = new_ev;
  if (exposure_time != old_exp_t || gain_idx != old_gain_idx) {
    camera.set_exposure(exposure_time, gain_idx);
  }
}

void CameraState::process_pix_frame(int buf_idx, uint64_t timestamp) {
  frame_id++;
  uint64_t timestamp_eof = timestamp + camera.sensor->readout_time_ns;

  VisionBuf *capture = &camera.vfe_buffers[buf_idx];
  VisionBuf *vb = camera.vipc_server->get_buffer(camera.stream_type, buf_idx);
  if (capture != nullptr && vb != nullptr) {
    capture->sync(VISIONBUF_SYNC_FROM_DEVICE);
    const uint8_t *nv12 = (const uint8_t *)capture->addr;
    set_camera_exposure(calculate_os04_ae_sample_nv12(nv12, camera.stride, ae_xywh, 4, 4));

    uint8_t *published = (uint8_t *)vb->addr;
    for (uint32_t y = 0; y < camera.output_height; y++) {
      memcpy(published + y * camera.vipc_stride, nv12 + y * camera.stride, camera.output_width);
    }
    for (uint32_t y = 0; y < camera.output_height / 2; y++) {
      memcpy(published + camera.vipc_uv_offset + y * camera.vipc_stride,
             nv12 + camera.uv_offset + y * camera.stride, camera.output_width);
    }
    vb->sync(VISIONBUF_SYNC_TO_DEVICE);
  }

  VisionIpcBufExtra extra = {frame_id, timestamp, timestamp_eof};
  vb->set_frame_id(frame_id);
  camera.vipc_server->send(vb, &extra, false);

  MessageBuilder msg;
  auto framed = (msg.initEvent().*camera.cc.init_camera_state)();
  framed.setFrameId(frame_id);
  framed.setRequestId(frame_id);
  framed.setTimestampEof(timestamp_eof);
  framed.setTimestampSof(timestamp);
  framed.setIntegLines(exposure_time);
  framed.setGain(camera.sensor->sensor_analog_gains[gain_idx]);
  framed.setSensor(camera.sensor->image_sensor);
  framed.setMeasuredGreyFraction(measured_grey_fraction);
  framed.setTargetGreyFraction(target_grey_fraction);
  framed.setExposureValPercent(util::map_val(current_ev,
    camera.sensor->min_ev, camera.sensor->max_ev, 0.0f, 100.0f));
  pm->send(camera.cc.publish_name, msg);
}

void camerad_thread() {
  LOG("-- v1 camerad starting (VFE PIX DMABUF required; no runtime RDI/MMAP CPU fallback)");

  VisionIpcServer v("camerad");

  int media_fd = open("/dev/media0", O_RDWR);
  if (media_fd < 0) {
    LOGE("failed to open /dev/media0");
    return;
  }
  for (const auto &config : ALL_CAMERA_CONFIGS) {
    const int i = config.camera_num;
    v1_cams[i] = resolve_cam_config(media_fd, i);
    LOG("cam %d: csiphy=%u csid=%u vfe_pix=%u pix_dev=%d pix_subdev=%d",
        i, v1_cams[i].csiphy_entity, v1_cams[i].csid_entity,
        v1_cams[i].vfe_pix_entity, v1_cams[i].pix_video_dev, v1_cams[i].vfe_pix_subdev);
  }
  close(media_fd);

  reset_all_media_links();

  std::vector<std::unique_ptr<CameraState>> cams;
  for (const auto &config : ALL_CAMERA_CONFIGS) {
    auto cam = std::make_unique<CameraState>(config);
    cam->init(&v);
    cams.emplace_back(std::move(cam));
  }

  v.start_listener();

  for (auto &cam : cams) {
    cam->camera.queue_all_buffers();
  }
  constexpr int start_gap_us = 38000;
  auto stream_one = [&](const std::unique_ptr<CameraState> &cam) {
    cam->camera.stream_on();
    usleep(start_gap_us);
  };
  for (auto it = cams.rbegin(); it != cams.rend(); ++it) stream_one(*it);

  LOG("-- v1 camerad streaming");

  // Rebuild the complete media and VisionIPC graph when an expected camera
  // stops producing frames. VisionIPC streams cannot be recreated safely in
  // place, so returning lets manager restart camerad and re-probe every sensor.
  // While a flex is disconnected this repeats every thirty seconds; once it is
  // reconnected, the next camerad start restores the stream without a reboot.
  constexpr uint64_t camera_reconnect_timeout_ns = 30ULL * 1000 * 1000 * 1000;
  const uint64_t streaming_started_ns = nanos_since_boot();
  for (auto &cam : cams) cam->last_frame_ns = streaming_started_ns;

  while (!do_exit) {
    for (auto &cam : cams) {
      const uint64_t now = nanos_since_boot();
      if (!cam->camera.enabled || now - cam->last_frame_ns >= camera_reconnect_timeout_ns) {
        LOGE("cam %d: no frames for 30 seconds; restarting camerad to re-probe hardware",
             cam->camera.cc.camera_num);
        return;
      }

      uint64_t timestamp;
      int buf_idx = cam->camera.dequeue_frame(&timestamp);
      if (buf_idx < 0) {
        if (buf_idx != -ETIMEDOUT) {
          LOGW_100("cam %d: dequeue failed: %d (%s)", cam->camera.cc.camera_num, -buf_idx, strerror(-buf_idx));
          usleep(1000);
        }
        continue;
      }

      cam->last_frame_ns = nanos_since_boot();
      cam->process_pix_frame(buf_idx, timestamp);
      cam->camera.queue_frame(buf_idx);
    }
  }

  LOG("-- v1 camerad stopping");
  for (auto &cam : cams) {
    cam->camera.stop_streaming();
  }
}
