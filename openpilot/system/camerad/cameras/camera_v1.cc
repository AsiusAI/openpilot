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
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>

#include "common/params.h"
#include "common/swaglog.h"
#include "common/timing.h"
#include "system/camerad/cameras/hw.h"
#include "system/camerad/cameras/ife_v1.h"
#include "system/camerad/cameras/nv12_info.h"
#include "system/camerad/sensors/sensor.h"


ExitHandler do_exit;

const bool env_debug_frames = getenv("DEBUG_FRAMES") != nullptr;

static int getenv_int_clamped(const char *name, int default_value, int min_value, int max_value);

struct OneCamRoute {
  int csiphy;
  int csid;
  int rdi_vfe;
  int preferred_pix_vfe;
  const char *sensor;
};

// Asius v1 camera routing:
//   Runtime camerad path: CSIPHY -> CSID PIX pad -> VFE PIX -> NV12 DMABUF.
//   Raw RDI probing lives in standalone bring-up tools, not in this runtime path.
// openpilot camera_num 0 is wide road, camera_num 1 is road,
// camera_num 2 is driver. For Asius v1:
//   CAM1 -> driver, CAM2 -> road, CAM3 -> wide road.
struct OneCamConfig {
  uint32_t csiphy_entity;
  uint32_t csid_entity;
  uint32_t vfe_rdi_entity;
  uint32_t vfe_pix_entity;
  int rdi_video_dev;
  int pix_video_dev;
  const char *sensor_name;
  int csiphy_subdev;
  int csid_subdev;
  int vfe_rdi_subdev;
  int vfe_pix_subdev;
  int pix_vfe;
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

static OneCamConfig resolve_cam_config(int media_fd, int cam_idx) {
  static const OneCamRoute routing[] = {
    {3, 1, 1, 1, "os04c10 20-0036"},
    {2, 0, 0, 0, "os04c10 18-0036"},
    {0, 2, 2, 2, "os04c10 16-0036"},
  };
  const auto *r = &routing[cam_idx];
  OneCamConfig cfg = {};
  cfg.rdi_video_dev = -1;
  cfg.pix_video_dev = -1;
  cfg.csiphy_subdev = -1;
  cfg.csid_subdev = -1;
  cfg.vfe_rdi_subdev = -1;
  cfg.vfe_pix_subdev = -1;
  cfg.pix_vfe = -1;
  cfg.sensor_name = r->sensor;
  cfg.csiphy_entity = find_media_entity(media_fd, util::string_format("msm_csiphy%d", r->csiphy).c_str());
  cfg.csid_entity = find_media_entity(media_fd, util::string_format("msm_csid%d", r->csid).c_str());
  cfg.vfe_rdi_entity = find_media_entity(media_fd, util::string_format("msm_vfe%d_rdi0", r->rdi_vfe).c_str());
  cfg.rdi_video_dev = find_v4l_dev("video", util::string_format("msm_vfe%d_video0", r->rdi_vfe).c_str());

  int pix_vfe = r->preferred_pix_vfe;
  std::string pix_env_name = util::string_format("ASIUS_CAM%d_PIX_VFE", cam_idx);
  const char *pix_env = getenv(pix_env_name.c_str());
  if (pix_env != nullptr && pix_env[0] != '\0') pix_vfe = atoi(pix_env);
  auto resolve_pix = [&](int vfe) {
    if (vfe < 0) return false;
    cfg.vfe_pix_entity = find_media_entity(media_fd, util::string_format("msm_vfe%d_pix", vfe).c_str());
    cfg.pix_video_dev = find_v4l_dev("video", util::string_format("msm_vfe%d_video3", vfe).c_str());
    cfg.vfe_pix_subdev = find_v4l_dev("v4l-subdev", util::string_format("msm_vfe%d_pix", vfe).c_str());
    cfg.pix_vfe = vfe;
    return cfg.vfe_pix_entity != 0 && cfg.pix_video_dev >= 0 && cfg.vfe_pix_subdev >= 0;
  };
  if (!resolve_pix(pix_vfe) && pix_vfe != r->rdi_vfe) resolve_pix(r->rdi_vfe);

  cfg.csiphy_subdev = find_v4l_dev("v4l-subdev", util::string_format("msm_csiphy%d", r->csiphy).c_str());
  cfg.csid_subdev = find_v4l_dev("v4l-subdev", util::string_format("msm_csid%d", r->csid).c_str());
  cfg.vfe_rdi_subdev = find_v4l_dev("v4l-subdev", util::string_format("msm_vfe%d_rdi0", r->rdi_vfe).c_str());
  return cfg;
}

static OneCamConfig one_cams[3];


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

struct VfeDmiUpload {
  uint32_t dmi_cfg_offset;
  uint8_t ram_select;
  std::vector<uint32_t> data;
  std::string source;
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

struct VfeMapBufCmd {
  int32_t fd;
  uint32_t pad;
  uint64_t iova;
  uint64_t size;
};

struct VfeUnmapBufCmd {
  uint64_t iova;
};

struct VfeSetBufCmd {
  uint32_t wm_index;
  uint32_t stride;
  uint64_t iova;
  uint32_t frame_inc;
  uint32_t pad;
};

#define VFE_IOC_MAGIC '#'
#define VFE_WRITE_REGS _IOW(VFE_IOC_MAGIC, 1, struct VfeWriteRegsCmd)
#define VFE_WRITE_DMI _IOW(VFE_IOC_MAGIC, 2, struct VfeDmiCmd)
#define VFE_MAP_BUF _IOWR(VFE_IOC_MAGIC, 3, struct VfeMapBufCmd)
#define VFE_UNMAP_BUF _IOW(VFE_IOC_MAGIC, 4, struct VfeUnmapBufCmd)
#define VFE_SET_BUF _IOW(VFE_IOC_MAGIC, 5, struct VfeSetBufCmd)
#define VFE_REG_UPDATE _IO(VFE_IOC_MAGIC, 6)
#define VFE_START _IO(VFE_IOC_MAGIC, 7)
#define VFE_STOP _IO(VFE_IOC_MAGIC, 8)
#define VFE_WAIT_SOF _IO(VFE_IOC_MAGIC, 9)

static int one_ioctl(int fd, unsigned long request, void *arg = nullptr) {
  int ret;
  int try_cnt = 0;
  do {
    ret = ioctl(fd, request, arg);
  } while (ret == -1 && errno == EINTR && try_cnt++ < 100);
  return ret;
}

static std::unique_ptr<SensorInfo> make_one_sensor() {
  return std::make_unique<OS04C10>();
}

static uint32_t one_media_bus_code() {
  return getenv("ASIUS_OS04_RAW12") != nullptr ? MEDIA_BUS_FMT_SBGGR12_1X12 : MEDIA_BUS_FMT_SBGGR10_1X10;
}

static bool one_uses_csid_tpg() {
  return getenv("ASIUS_CSID_TPG") != nullptr;
}

static bool one_uses_vfe_pix() {
  return true;
}

static bool one_uses_pix_v4l2() {
  return getenv("ASIUS_CAM_PIX_IOCTL") == nullptr;
}

static bool one_has_dma_heap() {
  return access("/dev/dma_heap/system", R_OK | W_OK) == 0;
}

static bool one_ae_disabled() {
  return getenv("ASIUS_CAM_DISABLE_AE") != nullptr;
}

static int one_ae_interval() {
  // The AE controller's three-frame EV history assumes one update per frame.
  return getenv_int_clamped("ASIUS_CAM_AE_INTERVAL", 1, 1, 120);
}

static int one_pix_csid_source_pad() {
  const char *pad = getenv("ASIUS_CAM_PIX_CSID_SRC_PAD");
  if (pad != nullptr && pad[0] != '\0') return std::clamp(atoi(pad), 1, 4);

  // The mainline CAMSS graph links VFE_LINE_PIX from CSID source pad 4.
  // The CSID-gen2 driver still programs that PIX/IPP path for sensor VC0.
  return 4;
}

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

static int one_csid_tpg_mode() {
  const char *mode = getenv("ASIUS_CSID_TPG");
  if (mode == nullptr || mode[0] == '\0') return 1;
  return std::max(1, atoi(mode));
}

static void set_csid_tpg_ctrl(int csid_subdev, int mode, int cam_idx, const char *action) {
  if (csid_subdev < 0) return;

  int fd = open(util::string_format("/dev/v4l-subdev%d", csid_subdev).c_str(), O_RDWR);
  if (fd < 0) return;

  struct v4l2_control ctrl = {};
  ctrl.id = V4L2_CID_TEST_PATTERN;
  ctrl.value = mode;
  if (ioctl(fd, VIDIOC_S_CTRL, &ctrl) != 0) {
    LOGE("cam %d: %s CSID TPG mode %d failed: %d (%s)",
         cam_idx, action, mode, errno, strerror(errno));
  } else {
    LOG("cam %d: %s CSID TPG mode %d", cam_idx, action, mode);
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

static std::string trim_copy(const std::string &s) {
  size_t first = 0;
  while (first < s.size() && std::isspace((unsigned char)s[first])) first++;
  size_t last = s.size();
  while (last > first && std::isspace((unsigned char)s[last - 1])) last--;
  return s.substr(first, last - first);
}

static int one_physical_cam_num(int camera_num);
static const char *one_cam_env_value(const char *base_name, int camera_num);
static float getenv_cam_float_clamped(const char *base_name, int camera_num, float default_value, float min_value, float max_value);
static int getenv_cam_int_clamped(const char *base_name, int camera_num, int default_value, int min_value, int max_value);

static std::vector<i2c_random_wr_payload> parse_sensor_reg_spec(const std::string &spec, const char *source, int cam_idx) {
  std::vector<i2c_random_wr_payload> regs;

  size_t start = 0;
  while (start < spec.size()) {
    size_t end = spec.find_first_of(",;\n\r", start);
    std::string token = spec.substr(start, end == std::string::npos ? std::string::npos : end - start);
    size_t comment = token.find('#');
    if (comment != std::string::npos) token.resize(comment);
    token = trim_copy(token);
    if (token.empty()) {
      if (end == std::string::npos) break;
      start = end + 1;
      continue;
    }

    size_t sep = token.find('=');
    if (sep == std::string::npos) sep = token.find(':');
    if (sep == std::string::npos) {
      LOGE("cam %d: ignoring malformed %s token '%s'", cam_idx, source, token.c_str());
    } else {
      char *addr_end = nullptr;
      char *data_end = nullptr;
      std::string addr_s = trim_copy(token.substr(0, sep));
      std::string data_s = trim_copy(token.substr(sep + 1));
      std::string addr_l = addr_s;
      std::transform(addr_l.begin(), addr_l.end(), addr_l.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (addr_l == "delay" || addr_l == "delay_ms" || addr_l == "msleep" || addr_l == "sleep") {
        unsigned long delay_ms = strtoul(data_s.c_str(), &data_end, 0);
        if ((data_end && *data_end) || delay_ms > 10000) {
          LOGE("cam %d: ignoring invalid %s token '%s'", cam_idx, source, token.c_str());
        } else {
          regs.push_back({ONE_SENSOR_DELAY_MS, (uint32_t)delay_ms});
        }
        if (end == std::string::npos) break;
        start = end + 1;
        continue;
      }

      unsigned long addr = strtoul(addr_s.c_str(), &addr_end, 0);
      unsigned long data = strtoul(data_s.c_str(), &data_end, 0);
      if ((addr_end && *addr_end) || (data_end && *data_end) || addr > 0xffff || data > 0xff) {
        LOGE("cam %d: ignoring invalid %s token '%s'", cam_idx, source, token.c_str());
      } else {
        regs.push_back({(uint32_t)addr, (uint32_t)data});
      }
    }
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return regs;
}

static std::vector<i2c_random_wr_payload> parse_sensor_reg_overrides(const char *env_name, int cam_idx) {
  const char *env = getenv(env_name);
  if (env == nullptr || env[0] == '\0') return {};
  return parse_sensor_reg_spec(env, env_name, cam_idx);
}

static std::vector<i2c_random_wr_payload> parse_sensor_reg_path(const char *path, const char *source, int cam_idx) {
  if (path == nullptr || path[0] == '\0') return {};

  std::string spec = util::read_file(path);
  if (spec.empty()) {
    LOGE("cam %d: %s '%s' is empty or unreadable", cam_idx, source, path);
    return {};
  }

  auto regs = parse_sensor_reg_spec(spec, source, cam_idx);
  LOG("cam %d: loaded %zu regs from %s '%s'", cam_idx, regs.size(), source, path);
  return regs;
}

static bool write_sensor_reg_overrides(int sensor_fd, const char *env_name, const char *name, int cam_idx) {
  std::vector<i2c_random_wr_payload> regs = parse_sensor_reg_overrides(env_name, cam_idx);
  return write_sensor_regs(sensor_fd, regs, name, cam_idx);
}

static bool write_sensor_reg_path(int sensor_fd, const char *path, const char *source, const char *name, int cam_idx) {
  std::vector<i2c_random_wr_payload> regs = parse_sensor_reg_path(path, source, cam_idx);
  return !regs.empty() && write_sensor_regs(sensor_fd, regs, name, cam_idx);
}

static std::vector<VfeRegWrite> parse_vfe_reg_spec(const std::string &spec, const char *source, int cam_idx) {
  std::vector<VfeRegWrite> regs;

  size_t start = 0;
  while (start < spec.size()) {
    size_t end = spec.find_first_of(",;\n\r", start);
    std::string token = spec.substr(start, end == std::string::npos ? std::string::npos : end - start);
    size_t comment = token.find('#');
    if (comment != std::string::npos) token.resize(comment);
    token = trim_copy(token);
    if (token.empty()) {
      if (end == std::string::npos) break;
      start = end + 1;
      continue;
    }

    size_t sep = token.find('=');
    if (sep == std::string::npos) sep = token.find(':');
    if (sep == std::string::npos) {
      LOGE("cam %d: ignoring malformed %s token '%s'", cam_idx, source, token.c_str());
    } else {
      char *offset_end = nullptr;
      char *value_end = nullptr;
      std::string offset_s = trim_copy(token.substr(0, sep));
      std::string value_s = trim_copy(token.substr(sep + 1));
      unsigned long offset = strtoul(offset_s.c_str(), &offset_end, 0);
      unsigned long value = strtoul(value_s.c_str(), &value_end, 0);
      if ((offset_end && *offset_end) || (value_end && *value_end) || offset > 0x2ffc || value > 0xffffffffUL || (offset & 3)) {
        LOGE("cam %d: ignoring invalid %s token '%s'", cam_idx, source, token.c_str());
      } else {
        regs.push_back({(uint32_t)offset, (uint32_t)value});
      }
    }
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return regs;
}

static std::vector<VfeDmiUpload> parse_vfe_dmi_spec(const std::string &spec, const char *source, int cam_idx) {
  std::vector<VfeDmiUpload> uploads;

  size_t start = 0;
  while (start < spec.size()) {
    size_t end = spec.find_first_of(";\n\r", start);
    std::string token = spec.substr(start, end == std::string::npos ? std::string::npos : end - start);
    size_t comment = token.find('#');
    if (comment != std::string::npos) token.resize(comment);
    token = trim_copy(token);
    if (token.empty()) {
      if (end == std::string::npos) break;
      start = end + 1;
      continue;
    }

    size_t sep = token.find('=');
    if (sep == std::string::npos) sep = token.find(':');
    if (sep == std::string::npos) {
      LOGE("cam %d: ignoring malformed %s DMI token '%s'", cam_idx, source, token.c_str());
      if (end == std::string::npos) break;
      start = end + 1;
      continue;
    }

    std::string target_s = trim_copy(token.substr(0, sep));
    std::string values_s = trim_copy(token.substr(sep + 1));
    uint32_t dmi_cfg_offset = 0xc24;
    unsigned long ram_select = 0;
    char *cfg_end = nullptr;
    char *ram_end = nullptr;

    size_t target_sep = target_s.find(':');
    if (target_sep != std::string::npos) {
      std::string cfg_s = trim_copy(target_s.substr(0, target_sep));
      std::string ram_s = trim_copy(target_s.substr(target_sep + 1));
      unsigned long cfg = strtoul(cfg_s.c_str(), &cfg_end, 0);
      ram_select = strtoul(ram_s.c_str(), &ram_end, 0);
      if ((cfg_end && *cfg_end) || cfg > 0x2ffc || (cfg & 3)) {
        LOGE("cam %d: ignoring invalid %s DMI target '%s'", cam_idx, source, target_s.c_str());
        if (end == std::string::npos) break;
        start = end + 1;
        continue;
      }
      dmi_cfg_offset = (uint32_t)cfg;
    } else {
      ram_select = strtoul(target_s.c_str(), &ram_end, 0);
    }

    if ((ram_end && *ram_end) || ram_select > 0xff) {
      LOGE("cam %d: ignoring invalid %s DMI RAM '%s'", cam_idx, source, target_s.c_str());
      if (end == std::string::npos) break;
      start = end + 1;
      continue;
    }

    std::replace(values_s.begin(), values_s.end(), ',', ' ');
    std::vector<uint32_t> values;
    size_t value_start = 0;
    while (value_start < values_s.size()) {
      value_start = values_s.find_first_not_of(" \t", value_start);
      if (value_start == std::string::npos) break;
      size_t value_end = values_s.find_first_of(" \t", value_start);
      std::string value_s = values_s.substr(value_start, value_end == std::string::npos ? std::string::npos : value_end - value_start);
      char *value_parse_end = nullptr;
      unsigned long value = strtoul(value_s.c_str(), &value_parse_end, 0);
      if ((value_parse_end && *value_parse_end) || value > 0xffffffffUL) {
        LOGE("cam %d: ignoring invalid %s DMI value '%s'", cam_idx, source, value_s.c_str());
        values.clear();
        break;
      }
      values.push_back((uint32_t)value);
      if (value_end == std::string::npos) break;
      value_start = value_end + 1;
    }

    if (values.empty()) {
      LOGE("cam %d: ignoring empty %s DMI upload '%s'", cam_idx, source, token.c_str());
    } else {
      uploads.push_back({dmi_cfg_offset, (uint8_t)ram_select, std::move(values), source});
    }

    if (end == std::string::npos) break;
    start = end + 1;
  }
  return uploads;
}

struct Os04VfeWbRegs {
  bool valid = false;
  int blue = 0;
  int green = 0;
  int red = 0;
};

static Os04VfeWbRegs default_os04_vfe_wb_regs(int cam_idx) {
  Os04VfeWbRegs wb;

  switch (one_physical_cam_num(cam_idx)) {
    case 1:
    case 2:
    case 3:
      wb.valid = true;
      wb.blue = 0x00cd;
      wb.green = 0x0080;
      wb.red = 0x00e1;
      break;
    default:
      break;
  }
  if (wb.valid) {
    wb.blue = getenv_cam_int_clamped("ASIUS_CAM_WB_BLUE", cam_idx, wb.blue, 0x40, 0x400);
    wb.green = getenv_cam_int_clamped("ASIUS_CAM_WB_GREEN", cam_idx, wb.green, 0x40, 0x400);
    wb.red = getenv_cam_int_clamped("ASIUS_CAM_WB_RED", cam_idx, wb.red, 0x40, 0x400);
  }
  return wb;
}

static float default_os04_target_grey(int cam_idx) {
  switch (one_physical_cam_num(cam_idx)) {
    case 2:
      return 0.45f;
    case 3:
      return 0.38f;
    default:
      return 0.48f;
  }
}

static int default_os04_awb_range(int cam_idx) {
  switch (one_physical_cam_num(cam_idx)) {
    case 1:
    case 2:
    case 3:
      return 0x80;
    default:
      return 0x18;
  }
}

static float default_os04_gamma_k(int) {
  return 7.0f;
}

static float default_os04_ae_rgb_clip_response(int cam_idx) {
  switch (one_physical_cam_num(cam_idx)) {
    case 2:
    case 3:
      return 1.0f;
    default:
      return 0.25f;
  }
}

static float default_os04_ae_rgb_clip_min_ratio(int cam_idx) {
  switch (one_physical_cam_num(cam_idx)) {
    case 2:
    case 3:
      return 0.65f;
    default:
      return 0.85f;
  }
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

static std::vector<VfeRegWrite> os04_vfe_ccm_reg_writes() {
  // The CamThink module needs different WB from comma's OS04 module, but no
  // additional matrix after that correction.
  static constexpr uint32_t ccm[13] = {
    0x00000080, 0x00000000, 0x00000000,
    0x00000000, 0x00000080, 0x00000000,
    0x00000000, 0x00000000, 0x00000080,
    0x00000000, 0x00000000, 0x00000000,
    0x00000000,
  };
  static constexpr size_t ccm_count = sizeof(ccm) / sizeof(ccm[0]);

  std::vector<VfeRegWrite> regs;
  regs.reserve(ccm_count);
  for (size_t i = 0; i < ccm_count; i++) {
    regs.push_back({0x760 + (uint32_t)i * 4, ccm[i]});
  }
  return regs;
}

static std::vector<VfeRegWrite> os04_vfe_vignetting_config_reg_writes() {
  return {
    {0x6bc, 0x0b3c0000},
    {0x6c0, 0x00670067},
    {0x6c4, 0xd3b1300c},
    {0x6c8, 0x13b1300c},
    {0x6d8, 0xec4e4000},
    {0x6dc, 0x0100c003},
  };
}

static bool default_os04_vfe_rolloff_enabled(int cam_idx) {
  // The provisional OS04 rolloff table is not lens/module calibrated. It
  // reduced CAM3 noise in an earlier bench setup, but later captures showed
  // large green/magenta spatial drift. Keep rolloff opt-in until it is tuned
  // against real flat-field captures for the final lens stack.
  return false;
}

static std::vector<VfeRegWrite> default_os04_vfe_tuning_regs(int cam_idx) {
  if (getenv("ASIUS_CAM_DISABLE_DEFAULT_VFE_TUNING") != nullptr) return {};
  // Stream start resets CORE_CFG; restore the CamThink module's RGGB phase.
  std::vector<VfeRegWrite> regs = {{0x050, 0x00000000}};
  std::vector<VfeRegWrite> wb = os04_vfe_wb_reg_writes(default_os04_vfe_wb_regs(cam_idx));
  regs.insert(regs.end(), wb.begin(), wb.end());
  std::vector<VfeRegWrite> ccm = os04_vfe_ccm_reg_writes();
  regs.insert(regs.end(), ccm.begin(), ccm.end());
  return regs;
}

static std::vector<VfeRegWrite> parse_cam_vfe_reg_overrides(const char *env_name, int cam_idx) {
  std::vector<VfeRegWrite> regs;

  auto append_env = [&](const std::string &name) {
    const char *env = getenv(name.c_str());
    if (env == nullptr || env[0] == '\0') return;
    auto parsed = parse_vfe_reg_spec(env, name.c_str(), cam_idx);
    regs.insert(regs.end(), parsed.begin(), parsed.end());
  };

  const char *suffix = env_name;
  constexpr const char *prefix = "ASIUS_CAM_";
  constexpr size_t prefix_len = 10;
  if (strncmp(env_name, prefix, prefix_len) == 0) suffix = env_name + prefix_len;

  append_env(env_name);
  append_env(util::string_format("ASIUS_CAM%d_%s", cam_idx, suffix));
  append_env(util::string_format("ASIUS_PHYS_CAM%d_%s", one_physical_cam_num(cam_idx), suffix));

  return regs;
}

static std::vector<VfeRegWrite> parse_vfe_reg_path(const char *path, const char *source, int cam_idx) {
  if (path == nullptr || path[0] == '\0') return {};

  std::string spec = util::read_file(path);
  if (spec.empty()) {
    LOGE("cam %d: %s '%s' is empty or unreadable", cam_idx, source, path);
    return {};
  }

  auto regs = parse_vfe_reg_spec(spec, source, cam_idx);
  LOG("cam %d: loaded %zu VFE regs from %s '%s'", cam_idx, regs.size(), source, path);
  return regs;
}

static std::vector<VfeRegWrite> parse_cam_vfe_reg_paths(const char *path_env_name, int cam_idx) {
  std::vector<VfeRegWrite> regs;

  auto append_path_env = [&](const std::string &name) {
    const char *path = getenv(name.c_str());
    if (path == nullptr || path[0] == '\0') return;
    auto parsed = parse_vfe_reg_path(path, name.c_str(), cam_idx);
    regs.insert(regs.end(), parsed.begin(), parsed.end());
  };

  const char *suffix = path_env_name;
  constexpr const char *prefix = "ASIUS_CAM_";
  constexpr size_t prefix_len = 10;
  if (strncmp(path_env_name, prefix, prefix_len) == 0) suffix = path_env_name + prefix_len;

  append_path_env(path_env_name);
  append_path_env(util::string_format("ASIUS_CAM%d_%s", cam_idx, suffix));
  append_path_env(util::string_format("ASIUS_PHYS_CAM%d_%s", one_physical_cam_num(cam_idx), suffix));

  return regs;
}

static std::vector<VfeDmiUpload> parse_cam_vfe_dmi_overrides(const char *env_name, int cam_idx) {
  std::vector<VfeDmiUpload> uploads;

  auto append_env = [&](const std::string &name) {
    const char *env = getenv(name.c_str());
    if (env == nullptr || env[0] == '\0') return;
    auto parsed = parse_vfe_dmi_spec(env, name.c_str(), cam_idx);
    uploads.insert(uploads.end(), std::make_move_iterator(parsed.begin()), std::make_move_iterator(parsed.end()));
  };

  const char *suffix = env_name;
  constexpr const char *prefix = "ASIUS_CAM_";
  constexpr size_t prefix_len = 10;
  if (strncmp(env_name, prefix, prefix_len) == 0) suffix = env_name + prefix_len;

  append_env(env_name);
  append_env(util::string_format("ASIUS_CAM%d_%s", cam_idx, suffix));
  append_env(util::string_format("ASIUS_PHYS_CAM%d_%s", one_physical_cam_num(cam_idx), suffix));

  return uploads;
}

static std::vector<VfeDmiUpload> parse_vfe_dmi_path(const char *path, const char *source, int cam_idx) {
  if (path == nullptr || path[0] == '\0') return {};

  std::string spec = util::read_file(path);
  if (spec.empty()) {
    LOGE("cam %d: %s '%s' is empty or unreadable", cam_idx, source, path);
    return {};
  }

  auto uploads = parse_vfe_dmi_spec(spec, source, cam_idx);
  LOG("cam %d: loaded %zu VFE DMI uploads from %s '%s'", cam_idx, uploads.size(), source, path);
  return uploads;
}

static std::vector<VfeDmiUpload> parse_cam_vfe_dmi_paths(const char *path_env_name, int cam_idx) {
  std::vector<VfeDmiUpload> uploads;

  auto append_path_env = [&](const std::string &name) {
    const char *path = getenv(name.c_str());
    if (path == nullptr || path[0] == '\0') return;
    auto parsed = parse_vfe_dmi_path(path, name.c_str(), cam_idx);
    uploads.insert(uploads.end(), std::make_move_iterator(parsed.begin()), std::make_move_iterator(parsed.end()));
  };

  const char *suffix = path_env_name;
  constexpr const char *prefix = "ASIUS_CAM_";
  constexpr size_t prefix_len = 10;
  if (strncmp(path_env_name, prefix, prefix_len) == 0) suffix = path_env_name + prefix_len;

  append_path_env(path_env_name);
  append_path_env(util::string_format("ASIUS_CAM%d_%s", cam_idx, suffix));
  append_path_env(util::string_format("ASIUS_PHYS_CAM%d_%s", one_physical_cam_num(cam_idx), suffix));

  return uploads;
}

static bool apply_os04_20fps_timing(int sensor_fd, int cam_idx) {
  if (getenv("ASIUS_CAM_DISABLE_20FPS_TIMING") != nullptr) return true;
  return write_sensor_regs(sensor_fd, {
    {0x380e, (OS04_RAW10_20FPS_VTS >> 8) & 0xff},
    {0x380f, OS04_RAW10_20FPS_VTS & 0xff},
  }, "20fps timing", cam_idx);
}

static int one_output_scale(const SensorInfo &sensor, int cam_idx) {
  int scale = std::max(sensor.out_scale, 1);
  const char *scale_env = getenv("ASIUS_CAM_OUT_SCALE");
  if (scale_env != nullptr && scale_env[0] != '\0') {
    int env_scale = atoi(scale_env);
    if (env_scale < 1 || env_scale > 8) {
      LOGE("cam %d: ignoring invalid ASIUS_CAM_OUT_SCALE=%s", cam_idx, scale_env);
    } else {
      scale = env_scale;
    }
  }
  return scale;
}

static void apply_os04_frame_size_overrides(SensorInfo &sensor, int cam_idx) {
  const char *w_env = getenv("ASIUS_CAM_FRAME_WIDTH");
  const char *h_env = getenv("ASIUS_CAM_FRAME_HEIGHT");
  if (w_env == nullptr && h_env == nullptr) return;

  int width = w_env != nullptr ? atoi(w_env) : sensor.frame_width;
  int height = h_env != nullptr ? atoi(h_env) : sensor.frame_height;
  if (width <= 0 || height <= 0 || width > 8192 || height > 8192) {
    LOGE("cam %d: ignoring invalid ASIUS_CAM_FRAME_WIDTH/HEIGHT=%s/%s",
         cam_idx, w_env ? w_env : "", h_env ? h_env : "");
    return;
  }

  sensor.frame_width = width;
  sensor.frame_height = height;
  sensor.frame_stride = width * sensor.bits_per_pixel / 8;
  LOG("cam %d: overriding OS04 frame size to %dx%d", cam_idx, width, height);
}

static int getenv_int_clamped(const char *name, int default_value, int min_value, int max_value) {
  const char *env = getenv(name);
  if (env == nullptr || env[0] == '\0') return default_value;

  char *end = nullptr;
  long value = strtol(env, &end, 10);
  if (end == env || (end != nullptr && *end != '\0')) return default_value;
  return std::clamp((int)value, min_value, max_value);
}

static int one_physical_cam_num(int camera_num) {
  // openpilot camera_num 0/1/2 maps to physical CAM3/CAM2/CAM1.
  static const int physical[] = {3, 2, 1};
  return (camera_num >= 0 && camera_num < 3) ? physical[camera_num] : camera_num;
}

static const char *one_cam_env_value(const char *base_name, int camera_num) {
  const char *suffix = base_name;
  constexpr const char *prefix = "ASIUS_CAM_";
  constexpr size_t prefix_len = 10;
  if (strncmp(base_name, prefix, prefix_len) == 0) suffix = base_name + prefix_len;

  std::string physical_name = util::string_format("ASIUS_PHYS_CAM%d_%s", one_physical_cam_num(camera_num), suffix);
  const char *physical = getenv(physical_name.c_str());
  if (physical != nullptr && physical[0] != '\0') return physical;

  std::string indexed_name = util::string_format("ASIUS_CAM%d_%s", camera_num, suffix);
  const char *indexed = getenv(indexed_name.c_str());
  if (indexed != nullptr && indexed[0] != '\0') return indexed;

  const char *global = getenv(base_name);
  return (global != nullptr && global[0] != '\0') ? global : nullptr;
}

static float getenv_cam_float_clamped(const char *base_name, int camera_num, float default_value, float min_value, float max_value) {
  const char *env = one_cam_env_value(base_name, camera_num);
  if (env == nullptr) return default_value;

  char *end = nullptr;
  float value = strtof(env, &end);
  if (end == env || (end != nullptr && *end != '\0')) return default_value;
  return std::clamp(value, min_value, max_value);
}

static int getenv_cam_int_clamped(const char *base_name, int camera_num, int default_value, int min_value, int max_value) {
  const char *env = one_cam_env_value(base_name, camera_num);
  if (env == nullptr) return default_value;

  char *end = nullptr;
  long value = strtol(env, &end, 10);
  if (end == env || (end != nullptr && *end != '\0')) return default_value;
  return std::clamp((int)value, min_value, max_value);
}

static std::vector<uint32_t> build_os04_gamma_lut(float k) {
  std::vector<uint32_t> points;
  points.reserve(65);
  for (int i = 0; i < 65; i++) {
    const float fx = i / 64.0f;
    const float y = (k * fx) / (1.0f + (k - 1.0f) * fx);
    points.push_back((uint32_t)(std::clamp(y, 0.0f, 1.0f) * 1023.0f + 0.5f));
  }

  std::vector<uint32_t> lut;
  lut.reserve(64);
  for (int i = 0; i < 64; i++) {
    const uint32_t base = points[i];
    const uint32_t delta = points[i + 1] - points[i];
    lut.push_back(base | (delta << 10));
  }
  return lut;
}

class OneCamera {
public:
  CameraConfig cc;
  std::unique_ptr<SensorInfo> sensor;
  CameraBuf buf;
  bool enabled;

  int video_fd = -1;
  int sensor_fd = -1;
  bool use_pix = false;
  bool use_pix_v4l2 = false;
  bool use_v4l2_dmabuf = false;
  bool streaming = false;

  int n_bufs = 4;
  uint64_t pix_iovas[VIPC_BUFFER_COUNT] = {};
  uint64_t pix_map_sizes[VIPC_BUFFER_COUNT] = {};
  int pix_current_idx = 0;
  int pix_next_idx = 1;

  uint32_t output_width = 0, output_height = 0;
  uint32_t stride = 0, y_height = 0, uv_height = 0, yuv_size = 0, uv_offset = 0;

  OneCamera(const CameraConfig &config) : cc(config), enabled(config.enabled) {}

  void camera_open(VisionIpcServer *v);
  void camera_close();
  void setup_media_links();
  void set_formats();
  bool map_pix_buffers();
  bool configure_pix_isp();
  bool write_vfe_regs(const std::vector<VfeRegWrite> &regs, const char *name);
  bool write_vfe_dmis(const std::vector<VfeDmiUpload> &uploads, const char *name);
  bool apply_vfe_reg_overrides(const char *env_name, const char *path_env_name, const char *name);
  bool apply_vfe_vignetting_override();
  bool apply_vfe_gamma_override();
  bool apply_vfe_dmi_overrides(const char *env_name, const char *path_env_name, const char *name);
  bool set_pix_buffer(int index);
  bool use_custom_pix_ioctl() const { return use_pix && !use_pix_v4l2; }
  bool use_direct_vipc_buffers() const { return use_custom_pix_ioctl() || use_v4l2_dmabuf; }
  void queue_all_buffers();
  void stream_on();
  void start_streaming();
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
  auto &dcfg = one_cams[cam_idx];

  struct media_link_desc link = {};

  if (!one_uses_csid_tpg()) {
    set_csid_tpg_ctrl(dcfg.csid_subdev, 0, cam_idx, "disabled");

    // CSIPHY -> CSID (source pad 1 -> sink pad 0)
    link.source = {.entity = (uint32_t)dcfg.csiphy_entity, .index = 1};
    link.sink = {.entity = (uint32_t)dcfg.csid_entity, .index = 0};
    link.flags = MEDIA_LNK_FL_ENABLED;
    if (ioctl(media_fd, MEDIA_IOC_SETUP_LINK, &link) != 0)
      LOGE("cam %d: csiphy->csid link FAILED: %d (%s)", cam_idx, errno, strerror(errno));
    memset(&link, 0, sizeof(link));
  } else {
    LOG("cam %d: using CSID test pattern generator, skipping CSIPHY link", cam_idx);
  }

  // Mainline CAMSS exposes the PIX path on the CSID source pad selected here.
  link.source = {.entity = (uint32_t)dcfg.csid_entity, .index = (uint16_t)one_pix_csid_source_pad()};
  link.sink = {.entity = (uint32_t)dcfg.vfe_pix_entity, .index = 0};
  link.flags = MEDIA_LNK_FL_ENABLED;
  if (ioctl(media_fd, MEDIA_IOC_SETUP_LINK, &link) != 0)
    LOGE("cam %d: csid->vfe PIX link FAILED: %d (%s)", cam_idx, errno, strerror(errno));

  close(media_fd);
  LOG("cam %d: media links set up (VFE PIX mode)", cam_idx);
}

void OneCamera::set_formats() {
  int cam_idx = cc.camera_num;
  auto &dcfg = one_cams[cam_idx];
  const uint32_t media_bus_code = one_media_bus_code();
  const bool use_tpg = one_uses_csid_tpg();

  if (!use_tpg) {
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
  }

  // set format on CSID subdev
  int csid_fd = open(util::string_format("/dev/v4l-subdev%d", dcfg.csid_subdev).c_str(), O_RDWR);
  if (csid_fd >= 0) {
    struct v4l2_subdev_format sfmt = {};
    if (use_tpg) {
      set_csid_tpg_ctrl(dcfg.csid_subdev, one_csid_tpg_mode(), cam_idx, "set");
    } else {
      sfmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
      sfmt.pad = 0;
      sfmt.format.width = sensor->frame_width;
      sfmt.format.height = sensor->frame_height;
      sfmt.format.code = media_bus_code;
      ioctl(csid_fd, VIDIOC_SUBDEV_S_FMT, &sfmt);
    }

    // PIX source pad matching the selected virtual channel.
    sfmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    sfmt.pad = one_pix_csid_source_pad();
    sfmt.format.width = sensor->frame_width;
    sfmt.format.height = sensor->frame_height;
    sfmt.format.code = media_bus_code;
    ioctl(csid_fd, VIDIOC_SUBDEV_S_FMT, &sfmt);
    close(csid_fd);
  }

  // set format on sensor subdev
  if (sensor_fd >= 0 && !use_tpg) {
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

bool OneCamera::map_pix_buffers() {
  for (int i = 0; i < VIPC_BUFFER_COUNT; i++) {
    VisionBuf *vb = vipc_server->get_buffer(stream_type, i);
    if (vb == nullptr || vb->fd < 0) {
      LOGE("cam %d: missing VIPC buffer idx=%d fd=%d", cc.camera_num, i, vb ? vb->fd : -1);
      return false;
    }

    VfeMapBufCmd cmd = {};
    cmd.fd = vb->fd;
    if (one_ioctl(video_fd, VFE_MAP_BUF, &cmd) != 0) {
      LOGE("cam %d: VFE_MAP_BUF idx=%d fd=%d failed: %d (%s)",
           cc.camera_num, i, vb->fd, errno, strerror(errno));
      return false;
    }
    if (cmd.iova == 0 || cmd.size < yuv_size) {
      LOGE("cam %d: VFE_MAP_BUF idx=%d returned invalid iova=0x%llx size=%llu expected>=%u",
           cc.camera_num, i, (unsigned long long)cmd.iova, (unsigned long long)cmd.size, yuv_size);
      return false;
    }
    pix_iovas[i] = cmd.iova;
    pix_map_sizes[i] = cmd.size;
  }
  LOG("cam %d: mapped %d VIPC buffers into VFE", cc.camera_num, VIPC_BUFFER_COUNT);
  return true;
}

bool OneCamera::configure_pix_isp() {
  auto [regs, dmis] = build_initial_config_flat(cc, sensor.get(), output_width, output_height);

  for (size_t offset = 0; offset < regs.size(); ) {
    const size_t count = std::min<size_t>(regs.size() - offset, 1024);
    VfeWriteRegsCmd cmd = {};
    cmd.regs = (uint64_t)(uintptr_t)(regs.data() + offset);
    cmd.count = count;
    if (one_ioctl(video_fd, VFE_WRITE_REGS, &cmd) != 0) {
      LOGE("cam %d: VFE_WRITE_REGS offset=%zu count=%zu failed: %d (%s)",
           cc.camera_num, offset, count, errno, strerror(errno));
      return false;
    }
    offset += count;
  }

  for (const auto &dmi : dmis) {
    VfeDmiCmd cmd = {};
    cmd.dmi_cfg_offset = dmi.cfg_offset;
    cmd.ram_select = dmi.ram_select;
    cmd.count = dmi.count;
    cmd.data = (uint64_t)(uintptr_t)dmi.data;
    if (one_ioctl(video_fd, VFE_WRITE_DMI, &cmd) != 0) {
      LOGE("cam %d: VFE_WRITE_DMI ram=%u count=%u failed: %d (%s)",
           cc.camera_num, dmi.ram_select, dmi.count, errno, strerror(errno));
      return false;
    }
  }

  LOG("cam %d: programmed VFE ISP (%zu regs, %zu DMI uploads, %ux%u -> %ux%u)",
      cc.camera_num, regs.size(), dmis.size(), sensor->frame_width, sensor->frame_height,
      output_width, output_height);
  return true;
}

bool OneCamera::write_vfe_regs(const std::vector<VfeRegWrite> &regs, const char *name) {
  if (!use_pix || video_fd < 0) return true;
  if (regs.empty()) return true;

  for (size_t offset = 0; offset < regs.size(); ) {
    const size_t count = std::min<size_t>(regs.size() - offset, 1024);
    VfeWriteRegsCmd cmd = {};
    cmd.regs = (uint64_t)(uintptr_t)(regs.data() + offset);
    cmd.count = count;
    if (one_ioctl(video_fd, VFE_WRITE_REGS, &cmd) != 0) {
      LOGE("cam %d: failed to write %s VFE regs offset=%zu count=%zu: %d (%s)",
           cc.camera_num, name, offset, count, errno, strerror(errno));
      return false;
    }
    offset += count;
  }

  if (one_ioctl(video_fd, VFE_REG_UPDATE) != 0) {
    LOGE("cam %d: failed to commit %s VFE regs: %d (%s)",
         cc.camera_num, name, errno, strerror(errno));
    return false;
  }

  LOG("cam %d: wrote %zu %s VFE regs", cc.camera_num, regs.size(), name);
  return true;
}

bool OneCamera::write_vfe_dmis(const std::vector<VfeDmiUpload> &uploads, const char *name) {
  if (!use_pix || video_fd < 0) return true;
  if (uploads.empty()) return true;

  size_t values = 0;
  for (const auto &upload : uploads) {
    VfeDmiCmd cmd = {};
    cmd.dmi_cfg_offset = upload.dmi_cfg_offset;
    cmd.ram_select = upload.ram_select;
    cmd.count = (uint32_t)upload.data.size();
    cmd.data = (uint64_t)(uintptr_t)upload.data.data();
    if (one_ioctl(video_fd, VFE_WRITE_DMI, &cmd) != 0) {
      LOGE("cam %d: failed to write %s VFE DMI cfg=0x%x ram=%u count=%u from %s: %d (%s)",
           cc.camera_num, name, cmd.dmi_cfg_offset, cmd.ram_select, cmd.count,
           upload.source.c_str(), errno, strerror(errno));
      return false;
    }
    values += upload.data.size();
  }

  if (one_ioctl(video_fd, VFE_REG_UPDATE) != 0) {
    LOGE("cam %d: failed to commit %s VFE DMI uploads: %d (%s)",
         cc.camera_num, name, errno, strerror(errno));
    return false;
  }

  LOG("cam %d: wrote %zu %s VFE DMI uploads (%zu dwords)",
      cc.camera_num, uploads.size(), name, values);
  return true;
}

bool OneCamera::apply_vfe_reg_overrides(const char *env_name, const char *path_env_name, const char *name) {
  if (!use_pix || video_fd < 0) return true;

  std::vector<VfeRegWrite> regs;
  if (sensor && sensor->image_sensor == cereal::FrameData::ImageSensor::OS04C10) {
    regs = default_os04_vfe_tuning_regs(cc.camera_num);
  }
  auto env_regs = parse_cam_vfe_reg_overrides(env_name, cc.camera_num);
  regs.insert(regs.end(), env_regs.begin(), env_regs.end());
  auto path_regs = parse_cam_vfe_reg_paths(path_env_name, cc.camera_num);
  regs.insert(regs.end(), path_regs.begin(), path_regs.end());
  return write_vfe_regs(regs, name);
}

bool OneCamera::apply_vfe_dmi_overrides(const char *env_name, const char *path_env_name, const char *name) {
  if (!use_pix || video_fd < 0) return true;

  auto uploads = parse_cam_vfe_dmi_overrides(env_name, cc.camera_num);
  auto path_uploads = parse_cam_vfe_dmi_paths(path_env_name, cc.camera_num);
  uploads.insert(uploads.end(), std::make_move_iterator(path_uploads.begin()), std::make_move_iterator(path_uploads.end()));
  return write_vfe_dmis(uploads, name);
}

bool OneCamera::apply_vfe_vignetting_override() {
  if (!use_pix || video_fd < 0 || !sensor ||
      sensor->image_sensor != cereal::FrameData::ImageSensor::OS04C10) {
    return true;
  }

  if (one_cam_env_value("ASIUS_CAM_DISABLE_VIGNETTING_DMI", cc.camera_num) != nullptr) return true;
  if (!default_os04_vfe_rolloff_enabled(cc.camera_num) &&
      one_cam_env_value("ASIUS_CAM_ENABLE_VIGNETTING_DMI", cc.camera_num) == nullptr &&
      one_cam_env_value("ASIUS_CAM_ENABLE_ROLLOFF", cc.camera_num) == nullptr) {
    return true;
  }

  if (sensor->vignetting_lut.empty()) {
    LOGE("cam %d: OS04 vignetting DMI requested but LUT is empty", cc.camera_num);
    return false;
  }

  std::vector<VfeRegWrite> regs = os04_vfe_vignetting_config_reg_writes();
  for (size_t offset = 0; offset < regs.size(); ) {
    const size_t count = std::min<size_t>(regs.size() - offset, 1024);
    VfeWriteRegsCmd cmd = {};
    cmd.regs = (uint64_t)(uintptr_t)(regs.data() + offset);
    cmd.count = count;
    if (one_ioctl(video_fd, VFE_WRITE_REGS, &cmd) != 0) {
      LOGE("cam %d: failed to write OS04 vignetting config offset=%zu count=%zu: %d (%s)",
           cc.camera_num, offset, count, errno, strerror(errno));
      return false;
    }
    offset += count;
  }

  struct VignettingBank {
    uint8_t ram_select;
    const char *name;
  };
  const VignettingBank banks[] = {
    {14, "GRR"},
    {15, "GBB"},
  };
  for (const auto &bank : banks) {
    VfeDmiCmd cmd = {};
    cmd.dmi_cfg_offset = 0xc24;
    cmd.ram_select = bank.ram_select;
    cmd.count = (uint32_t)sensor->vignetting_lut.size();
    cmd.data = (uint64_t)(uintptr_t)sensor->vignetting_lut.data();
    if (one_ioctl(video_fd, VFE_WRITE_DMI, &cmd) != 0) {
      LOGE("cam %d: failed to write OS04 vignetting DMI ram=%u/%s count=%u: %d (%s)",
           cc.camera_num, bank.ram_select, bank.name, cmd.count, errno, strerror(errno));
      return false;
    }
  }

  if (one_ioctl(video_fd, VFE_REG_UPDATE) != 0) {
    LOGE("cam %d: failed to commit OS04 vignetting DMI: %d (%s)",
         cc.camera_num, errno, strerror(errno));
    return false;
  }

  LOG("cam %d: wrote OS04 vignetting config and DMI override count=%zu",
      cc.camera_num, sensor->vignetting_lut.size());
  return true;
}

bool OneCamera::apply_vfe_gamma_override() {
  if (!use_pix || video_fd < 0 || !sensor ||
      sensor->image_sensor != cereal::FrameData::ImageSensor::OS04C10) {
    return true;
  }

  if (getenv("ASIUS_CAM_DISABLE_GAMMA_OVERRIDE") != nullptr) return true;

  const float default_k = default_os04_gamma_k(cc.camera_num);
  const float k = getenv_cam_float_clamped("ASIUS_CAM_GAMMA_K", cc.camera_num, default_k, 1.0f, 40.0f);
  const float g_k = getenv_cam_float_clamped("ASIUS_CAM_GAMMA_G_K", cc.camera_num, k, 1.0f, 40.0f);
  const float b_k = getenv_cam_float_clamped("ASIUS_CAM_GAMMA_B_K", cc.camera_num, k, 1.0f, 40.0f);
  const float r_k = getenv_cam_float_clamped("ASIUS_CAM_GAMMA_R_K", cc.camera_num, k, 1.0f, 40.0f);
  std::vector<uint32_t> gamma_g = build_os04_gamma_lut(g_k);
  std::vector<uint32_t> gamma_b = build_os04_gamma_lut(b_k);
  std::vector<uint32_t> gamma_r = build_os04_gamma_lut(r_k);
  struct GammaBank {
    uint8_t ram_select;
    const std::vector<uint32_t> *gamma;
    float k;
  };
  const GammaBank banks[] = {
    {26, &gamma_g, g_k},
    {28, &gamma_b, b_k},
    {30, &gamma_r, r_k},
  };
  for (const auto &bank : banks) {
    VfeDmiCmd cmd = {};
    cmd.dmi_cfg_offset = 0xc24;
    cmd.ram_select = bank.ram_select;
    cmd.count = bank.gamma->size();
    cmd.data = (uint64_t)(uintptr_t)bank.gamma->data();
    if (one_ioctl(video_fd, VFE_WRITE_DMI, &cmd) != 0) {
      LOGE("cam %d: failed to write OS04 gamma DMI ram=%u k=%.2f: %d (%s)",
           cc.camera_num, bank.ram_select, bank.k, errno, strerror(errno));
      return false;
    }
  }

  if (one_ioctl(video_fd, VFE_REG_UPDATE) != 0) {
    LOGE("cam %d: failed to commit OS04 gamma DMI k=%.2f: %d (%s)",
         cc.camera_num, k, errno, strerror(errno));
    return false;
  }

  LOG("cam %d: wrote OS04 gamma DMI override g=%.2f b=%.2f r=%.2f",
      cc.camera_num, g_k, b_k, r_k);
  return true;
}

bool OneCamera::set_pix_buffer(int index) {
  if (index < 0 || index >= VIPC_BUFFER_COUNT || pix_iovas[index] == 0) return false;

  VfeSetBufCmd y = {};
  y.wm_index = 3;
  y.stride = stride;
  y.iova = pix_iovas[index];
  y.frame_inc = stride * y_height;
  if (one_ioctl(video_fd, VFE_SET_BUF, &y) != 0) {
    LOGE("cam %d: VFE_SET_BUF Y idx=%d failed: %d (%s)", cc.camera_num, index, errno, strerror(errno));
    return false;
  }

  VfeSetBufCmd uv = {};
  uv.wm_index = 4;
  uv.stride = stride;
  uv.iova = pix_iovas[index] + uv_offset;
  uv.frame_inc = stride * uv_height;
  if (one_ioctl(video_fd, VFE_SET_BUF, &uv) != 0) {
    LOGE("cam %d: VFE_SET_BUF UV idx=%d failed: %d (%s)", cc.camera_num, index, errno, strerror(errno));
    return false;
  }

  if (one_ioctl(video_fd, VFE_REG_UPDATE) != 0) {
    LOGE("cam %d: VFE_REG_UPDATE idx=%d failed: %d (%s)", cc.camera_num, index, errno, strerror(errno));
    return false;
  }

  return true;
}

void OneCamera::camera_open(VisionIpcServer *v) {
  if (!enabled) return;

  vipc_server = v;
  stream_type = cc.stream_type;

  int cam_idx = cc.camera_num;
  auto &dcfg = one_cams[cam_idx];
  sensor = make_one_sensor();
  if (!sensor) {
    LOGE("cam %d: failed to create sensor config", cam_idx);
    enabled = false;
    return;
  }
  if (getenv("ASIUS_OS04_RAW12") == nullptr) {
    LOG("cam %d: using OS04C10 RAW10 media path", cam_idx);
    sensor->bits_per_pixel = 10;
    sensor->mipi_format = CAM_FORMAT_MIPI_RAW_10;
    sensor->frame_data_type = CSI_RAW10;
    sensor->frame_stride = sensor->frame_width * 10 / 8;
  }
  apply_os04_frame_size_overrides(*sensor, cam_idx);

  use_pix = one_uses_vfe_pix() && dcfg.vfe_pix_entity != 0 &&
            dcfg.pix_video_dev >= 0 && dcfg.vfe_pix_subdev >= 0;
  use_pix_v4l2 = use_pix && one_uses_pix_v4l2();
  if (one_uses_vfe_pix() && !use_pix) {
    LOGE("cam %d: VFE PIX unavailable, disabling camera; OS04 CPU debayer is not available on this branch "
         "(pix_entity=%u pix_dev=%d pix_subdev=%d)",
         cam_idx, dcfg.vfe_pix_entity, dcfg.pix_video_dev, dcfg.vfe_pix_subdev);
    enabled = false;
    return;
  }

  const int output_scale = one_output_scale(*sensor, cam_idx);
  output_width = std::max(2U, (sensor->frame_width / output_scale) & ~1U);
  output_height = std::max(2U, (sensor->frame_height / output_scale) & ~1U);
  auto [s, yh, uvh, sz] = get_nv12_info(output_width, output_height);
  stride = s;
  y_height = yh;
  uv_height = uvh;
  yuv_size = sz;
  uv_offset = stride * y_height;

  // open video device
  int dev = dcfg.pix_video_dev;
  std::string path = util::string_format("/dev/video%d", dev);
  video_fd = open(path.c_str(), O_RDWR);
  if (video_fd < 0) {
    LOGE("cam %d: failed to open %s: %d", cam_idx, path.c_str(), errno);
    enabled = false;
    return;
  }
  LOG("cam %d: opened %s (%s mode)", cam_idx, path.c_str(),
      use_pix_v4l2 ? "VFE PIX V4L2" : "VFE PIX ioctl");

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

  if (one_uses_csid_tpg()) {
    LOG("cam %d: skipping sensor init for CSID TPG", cam_idx);
  } else {
    const char *init_reg_file = getenv("ASIUS_CAM_INIT_REG_FILE");
    bool init_ok = (init_reg_file != nullptr && init_reg_file[0] != '\0') ?
                   write_sensor_reg_path(sensor_fd, init_reg_file, "ASIUS_CAM_INIT_REG_FILE", "init file", cam_idx) :
                   write_sensor_regs(sensor_fd, os04_default_init_regs(), "init file", cam_idx);
    if (!init_ok) {
      enabled = false;
      return;
    }
  }

  if (!one_uses_csid_tpg() && !apply_os04_20fps_timing(sensor_fd, cam_idx)) {
    enabled = false;
    return;
  }

  if (!one_uses_csid_tpg() &&
      !write_sensor_reg_overrides(sensor_fd, "ASIUS_CAM_INIT_REG_OVERRIDES", "init overrides", cam_idx)) {
    enabled = false;
    return;
  }

  // Do not enable a CAMSS route until the sensor has responded. Leaving an
  // absent camera's media links enabled can interfere with working cameras.
  setup_media_links();
  set_formats();

  use_v4l2_dmabuf = use_pix && use_pix_v4l2 && one_has_dma_heap();
  if (use_pix && use_pix_v4l2) {
    if (!use_v4l2_dmabuf) {
      LOGE("cam %d: VFE PIX V4L2 requires DMABUF on this branch; refusing V4L2 MMAP CPU-copy path", cam_idx);
      enabled = false;
      return;
    }
    const int default_bufs = 4;
    n_bufs = getenv_int_clamped("ASIUS_CAM_V4L2_BUFFER_COUNT", default_bufs, 4, VIPC_BUFFER_COUNT);
  }

  if (use_custom_pix_ioctl()) {
    v->create_buffers_with_sizes(stream_type, VIPC_BUFFER_COUNT,
                                 output_width, output_height,
                                 yuv_size, stride, uv_offset);
    if (!map_pix_buffers()) {
      enabled = false;
      return;
    }
    if (!configure_pix_isp()) {
      enabled = false;
      return;
    }
    if (one_ioctl(video_fd, VFE_REG_UPDATE) != 0) {
      LOGE("cam %d: initial VFE_REG_UPDATE failed: %d (%s)", cc.camera_num, errno, strerror(errno));
      enabled = false;
      return;
    }
    LOG("cam %d: VIPC buffers created (VFE PIX ioctl NV12, %ux%u, scale=%d, %u bytes, stride=%u)",
        cam_idx, output_width, output_height, output_scale, yuv_size, stride);
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

  v->create_buffers_with_sizes(stream_type, VIPC_BUFFER_COUNT,
                               output_width, output_height,
                               yuv_size, stride, uv_offset);

  LOG("cam %d: VIPC buffers created (%s, %ux%u, scale=%d, %u bytes, stride=%u)",
      cam_idx, "VFE PIX V4L2 DMABUF NV12",
      output_width, output_height, output_scale, yuv_size, stride);
}

void OneCamera::queue_all_buffers() {
  if (!enabled) return;
  if (use_custom_pix_ioctl()) return;

  for (int i = 0; i < n_bufs; i++) {
    queue_frame(i);
  }
}

void OneCamera::stream_on() {
  if (!enabled) return;

  if (!one_uses_csid_tpg() &&
      !write_sensor_reg_overrides(sensor_fd, "ASIUS_CAM_PRESTART_REG_OVERRIDES", "prestart overrides", cc.camera_num)) {
    enabled = false;
    return;
  }

  if (use_custom_pix_ioctl()) {
    if (one_ioctl(video_fd, VFE_START) != 0) {
      LOGE("cam %d: VFE_START failed: %d (%s)", cc.camera_num, errno, strerror(errno));
      enabled = false;
      return;
    }

    pix_current_idx = 0;
    pix_next_idx = 1;
    if (!set_pix_buffer(pix_current_idx)) {
      one_ioctl(video_fd, VFE_STOP);
      enabled = false;
      return;
    }

    if (!one_uses_csid_tpg() &&
        !write_sensor_regs(sensor_fd, sensor->start_reg_array, "start", cc.camera_num)) {
      one_ioctl(video_fd, VFE_STOP);
      enabled = false;
      return;
    }

    if (!one_uses_csid_tpg() &&
        !write_sensor_reg_overrides(sensor_fd, "ASIUS_CAM_POSTSTART_REG_OVERRIDES", "poststart overrides", cc.camera_num)) {
      one_ioctl(video_fd, VFE_STOP);
      enabled = false;
      return;
    }

    if (!apply_vfe_vignetting_override()) {
      one_ioctl(video_fd, VFE_STOP);
      enabled = false;
      return;
    }

    if (!apply_vfe_reg_overrides("ASIUS_CAM_VFE_REG_OVERRIDES", "ASIUS_CAM_VFE_REG_FILE", "poststart overrides")) {
      one_ioctl(video_fd, VFE_STOP);
      enabled = false;
      return;
    }

    if (!apply_vfe_gamma_override()) {
      one_ioctl(video_fd, VFE_STOP);
      enabled = false;
      return;
    }

    if (!apply_vfe_dmi_overrides("ASIUS_CAM_VFE_DMI_OVERRIDES", "ASIUS_CAM_VFE_DMI_FILE", "poststart overrides")) {
      one_ioctl(video_fd, VFE_STOP);
      enabled = false;
      return;
    }

    streaming = true;
    LOG("cam %d: VFE PIX streaming started", cc.camera_num);
    return;
  }

  bool sensor_started = false;

  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  if (ioctl(video_fd, VIDIOC_STREAMON, &type) != 0) {
    LOGE("cam %d: STREAMON failed: %d (%s)", cc.camera_num, errno, strerror(errno));
    if (sensor_started) write_sensor_regs(sensor_fd, {{0x100, 0}}, "stop", cc.camera_num);
    enabled = false;
    return;
  }

  if (!sensor_started && !one_uses_csid_tpg() &&
      !write_sensor_regs(sensor_fd, sensor->start_reg_array, "start", cc.camera_num)) {
    ioctl(video_fd, VIDIOC_STREAMOFF, &type);
    enabled = false;
    return;
  }

  if (!one_uses_csid_tpg() &&
      !write_sensor_reg_overrides(sensor_fd, "ASIUS_CAM_POSTSTART_REG_OVERRIDES", "poststart overrides", cc.camera_num)) {
    ioctl(video_fd, VIDIOC_STREAMOFF, &type);
    enabled = false;
    return;
  }

  if (!apply_vfe_vignetting_override()) {
    ioctl(video_fd, VIDIOC_STREAMOFF, &type);
    enabled = false;
    return;
  }

  if (!apply_vfe_reg_overrides("ASIUS_CAM_VFE_REG_OVERRIDES", "ASIUS_CAM_VFE_REG_FILE", "poststart overrides")) {
    ioctl(video_fd, VIDIOC_STREAMOFF, &type);
    enabled = false;
    return;
  }

  if (!apply_vfe_gamma_override()) {
    ioctl(video_fd, VIDIOC_STREAMOFF, &type);
    enabled = false;
    return;
  }

  if (!apply_vfe_dmi_overrides("ASIUS_CAM_VFE_DMI_OVERRIDES", "ASIUS_CAM_VFE_DMI_FILE", "poststart overrides")) {
    ioctl(video_fd, VIDIOC_STREAMOFF, &type);
    enabled = false;
    return;
  }

  streaming = true;
  LOG("cam %d: VFE PIX V4L2 streaming started", cc.camera_num);
}

void OneCamera::start_streaming() {
  queue_all_buffers();
  stream_on();
}

void OneCamera::stop_streaming() {
  if (streaming && sensor_fd >= 0 && !one_uses_csid_tpg()) {
    write_sensor_regs(sensor_fd, {{0x100, 0}}, "stop", cc.camera_num);
  }

  if (video_fd >= 0 && streaming) {
    if (use_custom_pix_ioctl()) {
      one_ioctl(video_fd, VFE_STOP);
    } else {
      int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
      ioctl(video_fd, VIDIOC_STREAMOFF, &type);
    }
  }
  streaming = false;
}

void OneCamera::queue_frame(int index) {
  if (use_custom_pix_ioctl()) return;

  struct v4l2_buffer vbuf = {};
  struct v4l2_plane planes[1] = {};
  vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  vbuf.memory = V4L2_MEMORY_DMABUF;
  vbuf.index = index;
  vbuf.length = 1;
  vbuf.m.planes = planes;
  VisionBuf *vb = vipc_server->get_buffer(stream_type, index);
  planes[0].m.fd = vb->fd;
  planes[0].length = yuv_size;
  int ret = ioctl(video_fd, VIDIOC_QBUF, &vbuf);
  if (ret != 0) LOGE("cam %d: QBUF idx=%d failed: %d (%s)", cc.camera_num, index, errno, strerror(errno));
}

int OneCamera::dequeue_frame(uint64_t *timestamp) {
  if (use_custom_pix_ioctl()) {
    if (one_ioctl(video_fd, VFE_WAIT_SOF) != 0) return -errno;

    *timestamp = nanos_since_boot();
    int ready_idx = pix_current_idx;
    pix_current_idx = pix_next_idx;
    pix_next_idx = (pix_next_idx + 1) % VIPC_BUFFER_COUNT;
    if (!set_pix_buffer(pix_current_idx)) return -EIO;

    const uint32_t readout_us = (sensor->readout_time_ns + 999) / 1000;
    usleep(readout_us + 1000);
    return ready_idx;
  }

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

  if (one_uses_csid_tpg()) return;

  write_sensor_regs(sensor_fd, sensor->getExposureRegisters(exposure_time, gain_idx, false),
                    "exposure", cc.camera_num);
}

void OneCamera::camera_close() {
  if (video_fd >= 0) {
    stop_streaming();
    if (use_custom_pix_ioctl()) {
      for (int i = 0; i < VIPC_BUFFER_COUNT; i++) {
        if (pix_iovas[i] != 0) {
          VfeUnmapBufCmd cmd = {.iova = pix_iovas[i]};
          one_ioctl(video_fd, VFE_UNMAP_BUF, &cmd);
          pix_iovas[i] = 0;
          pix_map_sizes[i] = 0;
        }
      }
    }
    close(video_fd);
    video_fd = -1;
  }
  if (sensor_fd >= 0) {
    close(sensor_fd);
    sensor_fd = -1;
  }
}

struct Os04AeSample {
  float grey_frac = 0.5f;
  float rgb_clip_hi_frac = 0.0f;
};

static constexpr int OS04_AE_HISTORY_SIZE = 4;
static constexpr int OS04_EXPOSURE_DELAY_FRAMES = 3;
static constexpr float OS04_AE_MAX_EV_GROWTH_PER_FRAME = 0.10f;

class CameraState {
public:
  OneCamera camera;
  int exposure_time = 1600;
  bool dc_gain_enabled = false;
  int dc_gain_weight = 0;
  int gain_idx = 8;
  float analog_gain_frac = 0;
  float cur_ev[3] = {};
  float os04_ev_history[OS04_AE_HISTORY_SIZE] = {};
  float best_ev_score = 0;
  int new_exp_g = 0;
  int new_exp_t = 0;

  Rect ae_xywh = {};
  Rect awb_xywh = {};
  float measured_grey_fraction = 0;
  float target_grey_fraction = 0.125;
  bool awb_enabled = false;
  int awb_interval = 40;
  int awb_start_frame = 240;
  int awb_deadband = 2;
  int awb_response = 1;
  int awb_max_step = 2;
  int awb_y_min = 40;
  int awb_y_max = 235;
  int awb_chroma_limit = 24;
  int awb_min_samples = 64;
  int awb_blue = 0;
  int awb_green = 0;
  int awb_red = 0;
  int awb_blue_min = 0;
  int awb_blue_max = 0;
  int awb_red_min = 0;
  int awb_red_max = 0;

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
  void init_awb();
  void update_awb(const uint8_t *nv12);

  float get_gain_factor() const {
    return (1 + dc_gain_weight * (camera.sensor->dc_gain_factor-1) / camera.sensor->dc_gain_max_weight);
  }
};

void CameraState::init(VisionIpcServer *v) {
  camera.camera_open(v);
  if (!camera.enabled) return;

  // The One starts the road sensor first, then wide almost one period later.
  // Number wide one frame ahead so same-numbered road/wide frames refer to the
  // same 20 Hz capture instant.
  if (camera.cc.stream_type == VISION_STREAM_WIDE_ROAD) {
    frame_id = 1;
  }

  fl_pix = camera.cc.focal_len / camera.sensor->pixel_size_mm;
  pm = std::make_unique<PubMaster>(std::vector{camera.cc.publish_name});

  if (camera.sensor->image_sensor == cereal::FrameData::ImageSensor::OS04C10) {
    exposure_time = getenv_cam_int_clamped("ASIUS_CAM_START_EXPOSURE_LINES", camera.cc.camera_num, 600,
                                           camera.sensor->exposure_time_min,
                                           camera.sensor->exposure_time_max);
    exposure_time = std::clamp(exposure_time, camera.sensor->exposure_time_min,
                               camera.sensor->exposure_time_max);
    gain_idx = camera.sensor->analog_gain_rec_idx;
    target_grey_fraction = getenv_cam_float_clamped("ASIUS_CAM_TARGET_GREY", camera.cc.camera_num,
                                                    default_os04_target_grey(camera.cc.camera_num), 0.05f, 0.75f);
  }

  float gain = camera.sensor->sensor_analog_gains[gain_idx];
  cur_ev[0] = cur_ev[1] = cur_ev[2] = gain * exposure_time;
  std::fill(std::begin(os04_ev_history), std::end(os04_ev_history), gain * exposure_time);
  camera.set_exposure(exposure_time, gain_idx);

  set_exposure_rect();
  init_awb();
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
  awb_xywh = {
    0,
    0,
    width,
    height,
  };
}

static Os04AeSample calculate_os04_ae_sample_nv12(const uint8_t *base, int stride, int uv_offset,
                                                   Rect ae_xywh, int width, int height,
                                                   int x_skip, int y_skip) {
  Os04AeSample ret;
  if (base == nullptr || stride <= 0) return ret;

  int lum_med;
  uint32_t lum_binning[256] = {0};

  unsigned int lum_total = 0;
  unsigned int rgb_clip_total = 0;
  unsigned int rgb_clip_count = 0;
  const bool sample_rgb_clip = base != nullptr && stride > 0 && uv_offset > 0 && width > 1 && height > 1;
  const uint8_t *uv_plane = sample_rgb_clip ? base + uv_offset : nullptr;

  for (int y = ae_xywh.y; y < ae_xywh.y + ae_xywh.h; y += y_skip) {
    for (int x = ae_xywh.x; x < ae_xywh.x + ae_xywh.w; x += x_skip) {
      uint8_t lum = base[(y * stride) + x];
      lum_binning[lum]++;
      lum_total += 1;

      if (sample_rgb_clip) {
        const int x_uv = std::clamp(x & ~1, 0, width - 2);
        const int y_even = std::clamp(y & ~1, 0, height - 2);
        const uint8_t *uv = uv_plane + (y_even / 2) * stride + x_uv;
        // Match the video-range YUV->RGB conversion used by the snapshot path,
        // otherwise high Y values near 235 look unclipped to AE but clip in RGB.
        const float yf = 1.164f * ((float)lum - 16.0f);
        const float uf = (float)uv[0] - 128.0f;
        const float vf = (float)uv[1] - 128.0f;
        const float r = std::clamp(yf + 1.596f * vf, 0.0f, 255.0f);
        const float g = std::clamp(yf - 0.392f * uf - 0.813f * vf, 0.0f, 255.0f);
        const float b = std::clamp(yf + 2.017f * uf, 0.0f, 255.0f);
        const float rgb_luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        rgb_clip_count += (rgb_luma >= 250.0f || r >= 255.0f || g >= 255.0f || b >= 255.0f) ? 1 : 0;
        rgb_clip_total += 1;
      }
    }
  }
  if (lum_total == 0) return ret;

  unsigned int lum_cur = 0;
  for (lum_med = 255; lum_med >= 0; lum_med--) {
    lum_cur += lum_binning[lum_med];
    if (lum_cur >= lum_total / 2) break;
  }

  ret.grey_frac = lum_med / 256.0f;
  ret.rgb_clip_hi_frac = rgb_clip_total > 0 ? (float)rgb_clip_count / (float)rgb_clip_total : 0.0f;
  return ret;
}

struct Nv12ChromaMedian {
  bool valid = false;
  int u = 128;
  int v = 128;
  unsigned int samples = 0;
  unsigned int neutral_samples = 0;
};

static Nv12ChromaMedian calculate_chroma_median_nv12(const uint8_t *base, int stride, int uv_offset,
                                                     Rect xywh, int width, int height,
                                                     int x_skip, int y_skip, int y_min, int y_max,
                                                     int chroma_limit, unsigned int min_neutral_samples) {
  Nv12ChromaMedian ret;
  if (base == nullptr || stride <= 0 || uv_offset <= 0 || width < 2 || height < 2) return ret;

  x_skip = std::max(2, x_skip & ~1);
  y_skip = std::max(2, y_skip);
  int x0 = std::clamp(xywh.x, 0, width - 2) & ~1;
  int x1 = std::clamp(xywh.x + xywh.w, x0 + 2, width);
  int y0 = std::clamp(xywh.y, 0, height - 2);
  int y1 = std::clamp(xywh.y + xywh.h, y0 + 2, height);

  y_min = std::clamp(y_min, 0, 255);
  y_max = std::clamp(y_max, y_min, 255);
  chroma_limit = std::clamp(chroma_limit, 0, 255);

  uint32_t u_all_hist[256] = {};
  uint32_t v_all_hist[256] = {};
  uint32_t u_neutral_hist[256] = {};
  uint32_t v_neutral_hist[256] = {};
  unsigned int all_total = 0;
  unsigned int neutral_total = 0;
  const uint8_t *uv_plane = base + uv_offset;
  for (int y = y0; y < y1; y += y_skip) {
    const uint8_t *row = uv_plane + (y / 2) * stride;
    for (int x = x0; x + 1 < x1; x += x_skip) {
      const int y_even = y & ~1;
      const int avg_y = (base[y_even * stride + x] + base[y_even * stride + x + 1] +
                         base[(y_even + 1) * stride + x] + base[(y_even + 1) * stride + x + 1] + 2) / 4;
      if (avg_y < y_min || avg_y > y_max) continue;

      const int u = row[x];
      const int v = row[x + 1];
      u_all_hist[u]++;
      v_all_hist[v]++;
      all_total++;
      if (std::abs(u - 128) + std::abs(v - 128) <= chroma_limit) {
        u_neutral_hist[u]++;
        v_neutral_hist[v]++;
        neutral_total++;
      }
    }
  }
  if (all_total == 0) return ret;

  auto median_from_hist = [](const uint32_t hist[256], unsigned int total) {
    const unsigned int target = (total + 1) / 2;
    unsigned int cur = 0;
    for (int i = 0; i < 256; i++) {
      cur += hist[i];
      if (cur >= target) return i;
    }
    return 128;
  };

  const bool use_neutral = neutral_total >= min_neutral_samples;
  const uint32_t *u_hist = use_neutral ? u_neutral_hist : u_all_hist;
  const uint32_t *v_hist = use_neutral ? v_neutral_hist : v_all_hist;
  const unsigned int total = use_neutral ? neutral_total : all_total;
  ret.valid = true;
  ret.u = median_from_hist(u_hist, total);
  ret.v = median_from_hist(v_hist, total);
  ret.samples = total;
  ret.neutral_samples = neutral_total;
  return ret;
}

void CameraState::init_awb() {
  if (!camera.enabled || !camera.use_pix || !camera.sensor ||
      camera.sensor->image_sensor != cereal::FrameData::ImageSensor::OS04C10) {
    return;
  }
  if (one_cam_env_value("ASIUS_CAM_DISABLE_AWB", camera.cc.camera_num) != nullptr) return;

  const bool awb_explicit = one_cam_env_value("ASIUS_CAM_ENABLE_AWB", camera.cc.camera_num) != nullptr;
  const bool has_vfe_override =
      one_cam_env_value("ASIUS_CAM_VFE_REG_OVERRIDES", camera.cc.camera_num) != nullptr ||
      one_cam_env_value("ASIUS_CAM_VFE_REG_FILE", camera.cc.camera_num) != nullptr;
  if (!awb_explicit && has_vfe_override) return;
  if (!awb_explicit && getenv("ASIUS_CAM_DISABLE_DEFAULT_VFE_TUNING") != nullptr) return;

  const Os04VfeWbRegs wb = default_os04_vfe_wb_regs(camera.cc.camera_num);
  if (!wb.valid) return;

  awb_interval = getenv_cam_int_clamped("ASIUS_CAM_AWB_INTERVAL", camera.cc.camera_num, 20, 5, 240);
  awb_start_frame = getenv_cam_int_clamped("ASIUS_CAM_AWB_START_FRAME", camera.cc.camera_num, 40, 0, 2000);
  awb_deadband = getenv_cam_int_clamped("ASIUS_CAM_AWB_DEADBAND", camera.cc.camera_num, 1, 0, 16);
  awb_response = getenv_cam_int_clamped("ASIUS_CAM_AWB_RESPONSE", camera.cc.camera_num, 4, 1, 16);
  awb_max_step = getenv_cam_int_clamped("ASIUS_CAM_AWB_MAX_STEP", camera.cc.camera_num, 16, 1, 32);
  awb_y_min = getenv_cam_int_clamped("ASIUS_CAM_AWB_Y_MIN", camera.cc.camera_num, 40, 0, 255);
  awb_y_max = getenv_cam_int_clamped("ASIUS_CAM_AWB_Y_MAX", camera.cc.camera_num, 235, 0, 255);
  if (awb_y_max < awb_y_min) awb_y_max = awb_y_min;
  awb_chroma_limit = getenv_cam_int_clamped("ASIUS_CAM_AWB_CHROMA_LIMIT", camera.cc.camera_num, 24, 0, 255);
  awb_min_samples = getenv_cam_int_clamped("ASIUS_CAM_AWB_MIN_SAMPLES", camera.cc.camera_num, 64, 1, 100000);
  const int awb_range = getenv_cam_int_clamped("ASIUS_CAM_AWB_RANGE", camera.cc.camera_num,
                                               default_os04_awb_range(camera.cc.camera_num), 0, 0x100);

  awb_blue = wb.blue;
  awb_green = wb.green;
  awb_red = wb.red;
  awb_blue_min = std::clamp(awb_blue - awb_range, 0x40, 0x400);
  awb_blue_max = std::clamp(awb_blue + awb_range, 0x40, 0x400);
  awb_red_min = std::clamp(awb_red - awb_range, 0x40, 0x400);
  awb_red_max = std::clamp(awb_red + awb_range, 0x40, 0x400);
  awb_enabled = true;
  LOG("cam %d: OS04 AWB enabled start=%d interval=%d deadband=%d response=%d step=%d y=%d-%d chroma=%d min_samples=%d blue=0x%x red=0x%x range=0x%x",
      camera.cc.camera_num, awb_start_frame, awb_interval, awb_deadband, awb_response, awb_max_step,
      awb_y_min, awb_y_max, awb_chroma_limit, awb_min_samples, awb_blue, awb_red, awb_range);
}

void CameraState::update_awb(const uint8_t *nv12) {
  if (!awb_enabled) return;
  const bool log_awb = one_cam_env_value("ASIUS_CAM_LOG_AWB", camera.cc.camera_num) != nullptr;

  const Nv12ChromaMedian med = calculate_chroma_median_nv12(nv12, camera.stride, camera.uv_offset,
                                                            awb_xywh, camera.output_width,
                                                            camera.output_height, 16, 16, awb_y_min,
                                                            awb_y_max, awb_chroma_limit, awb_min_samples);
  if (!med.valid) {
    if (log_awb) {
      LOG("cam %d: OS04 AWB invalid sample y=%d-%d chroma=%d min_samples=%d blue=0x%x red=0x%x",
          camera.cc.camera_num, awb_y_min, awb_y_max, awb_chroma_limit, awb_min_samples, awb_blue, awb_red);
    }
    return;
  }

  auto step_from_median = [&](int median) {
    const int err = 128 - median;
    if (std::abs(err) <= awb_deadband) return 0;
    return std::clamp(err * awb_response, -awb_max_step, awb_max_step);
  };

  const int new_blue = std::clamp(awb_blue + step_from_median(med.u), awb_blue_min, awb_blue_max);
  const int new_red = std::clamp(awb_red + step_from_median(med.v), awb_red_min, awb_red_max);
  if (new_blue == awb_blue && new_red == awb_red) {
    if (log_awb) {
      LOG("cam %d: OS04 AWB stable U=%d V=%d samples=%u neutral=%u blue=0x%x red=0x%x",
          camera.cc.camera_num, med.u, med.v, med.samples, med.neutral_samples, awb_blue, awb_red);
    }
    return;
  }

  awb_blue = new_blue;
  awb_red = new_red;
  Os04VfeWbRegs wb;
  wb.valid = true;
  wb.blue = awb_blue;
  wb.green = awb_green;
  wb.red = awb_red;
  if (!camera.write_vfe_regs(os04_vfe_wb_reg_writes(wb), "AWB")) {
    awb_enabled = false;
    LOGE("cam %d: disabling OS04 AWB after VFE write failure", camera.cc.camera_num);
    return;
  }
  LOG("cam %d: OS04 AWB U=%d V=%d samples=%u neutral=%u blue=0x%x red=0x%x", camera.cc.camera_num,
      med.u, med.v, med.samples, med.neutral_samples, awb_blue, awb_red);
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

  const bool os04 = sens->image_sensor == cereal::FrameData::ImageSensor::OS04C10;
  const float cur_ev_ = os04 ?
      os04_ev_history[(frame_id + OS04_AE_HISTORY_SIZE - OS04_EXPOSURE_DELAY_FRAMES) %
                      OS04_AE_HISTORY_SIZE] :
      cur_ev[(frame_id - 1) % 3];
  const float commanded_ev = exposure_time * sens->sensor_analog_gains[gain_idx];
  float new_target_grey = os04 ?
                          getenv_cam_float_clamped("ASIUS_CAM_TARGET_GREY", camera.cc.camera_num,
                                                   default_os04_target_grey(camera.cc.camera_num), 0.05f, 0.75f) :
                          std::clamp(0.4f - 0.3f * (float)(log2(1.0 + sens->target_grey_factor*cur_ev_) / log2(6000.0)), 0.1f, 0.4f);
  float target_grey = (1.0f - k_grey) * target_grey_fraction + k_grey * new_target_grey;

  const float grey_frac = std::clamp(ae_sample.grey_frac, 1.0f / 256.0f, 1.0f);
  float desired_ev = std::clamp(cur_ev_ * target_grey / grey_frac, sens->min_ev, sens->max_ev);
  const float desired_ev_before_clip = desired_ev;
  if (os04 && ae_sample.rgb_clip_hi_frac > 0.0f &&
      one_cam_env_value("ASIUS_CAM_DISABLE_AE_RGB_CLIP_GUARD", camera.cc.camera_num) == nullptr) {
    const float clip_limit = getenv_cam_float_clamped("ASIUS_CAM_AE_RGB_CLIP_LIMIT", camera.cc.camera_num,
                                                      0.08f, 0.001f, 0.50f);
    if (ae_sample.rgb_clip_hi_frac > clip_limit) {
      const float response = getenv_cam_float_clamped("ASIUS_CAM_AE_RGB_CLIP_RESPONSE", camera.cc.camera_num,
                                                     default_os04_ae_rgb_clip_response(camera.cc.camera_num),
                                                     0.05f, 1.0f);
      const float min_ratio = getenv_cam_float_clamped("ASIUS_CAM_AE_RGB_CLIP_MIN_RATIO", camera.cc.camera_num,
                                                      default_os04_ae_rgb_clip_min_ratio(camera.cc.camera_num),
                                                      0.25f, 1.0f);
      const float clip_ratio = std::clamp(std::pow(clip_limit / ae_sample.rgb_clip_hi_frac, response),
                                          min_ratio, 1.0f);
      desired_ev = std::min(desired_ev, cur_ev_ * clip_ratio);
    } else {
      // Approach the clipping threshold continuously instead of alternating
      // between unrestricted gain and a hard cap as samples cross the limit.
      const float headroom = 1.0f - ae_sample.rgb_clip_hi_frac / clip_limit;
      desired_ev = std::min(desired_ev, cur_ev_ * (1.0f + OS04_AE_MAX_EV_GROWTH_PER_FRAME * headroom));
    }
  }

  if (os04) {
    desired_ev = (1.0f - k_ev) * commanded_ev + k_ev * desired_ev;
  } else {
    float k = (1.0f - k_ev) / 3.0f;
    desired_ev = (k * cur_ev[0]) + (k * cur_ev[1]) + (k * cur_ev[2]) + (k_ev * desired_ev);
  }

  best_ev_score = 1e6;
  new_exp_g = 0;
  new_exp_t = 0;

  const int gain_step = os04 ? getenv_cam_int_clamped("ASIUS_CAM_AE_GAIN_STEP", camera.cc.camera_num, 4, 1, 16) : 1;
  int min_g = std::max(gain_idx - gain_step, sens->analog_gain_min_idx);
  int max_g = std::min(gain_idx + gain_step, sens->analog_gain_max_idx);
  for (int g = min_g; g <= max_g; g++) {
    float gain = sens->sensor_analog_gains[g];
    int t = std::clamp((int)std::round(desired_ev / gain), sens->exposure_time_min, sens->exposure_time_max);
    update_exposure_score(desired_ev, t, g, gain);
  }

  measured_grey_fraction = grey_frac;
  target_grey_fraction = target_grey;
  analog_gain_frac = sens->sensor_analog_gains[new_exp_g];
  gain_idx = new_exp_g;
  exposure_time = new_exp_t;

  const float new_ev = exposure_time * analog_gain_frac;
  cur_ev[frame_id % 3] = new_ev;
  if (os04) os04_ev_history[frame_id % OS04_AE_HISTORY_SIZE] = new_ev;
  if (one_cam_env_value("ASIUS_CAM_LOG_AE", camera.cc.camera_num) != nullptr) {
    LOG("cam %d: OS04 AE grey=%.4f target=%.4f rgb_clip=%.4f cur_ev=%.2f desired_ev=%.2f unclipped_ev=%.2f exp %d->%d gain_idx %d->%d gain %.3f",
        camera.cc.camera_num, grey_frac, target_grey, ae_sample.rgb_clip_hi_frac,
        cur_ev_, desired_ev, desired_ev_before_clip,
        old_exp_t, exposure_time, old_gain_idx, gain_idx, analog_gain_frac);
  }
  if (exposure_time != old_exp_t || gain_idx != old_gain_idx) {
    camera.set_exposure(exposure_time, gain_idx);
  }
}

void CameraState::process_pix_frame(int buf_idx, uint64_t timestamp) {
  frame_id++;
  uint64_t timestamp_eof = timestamp + camera.sensor->readout_time_ns;

  VisionBuf *vb = camera.vipc_server->get_buffer(camera.stream_type, buf_idx);
  if (vb != nullptr) {
    const bool sample_ae = !one_ae_disabled() && frame_id % one_ae_interval() == 0;
    const bool sample_awb = awb_enabled && frame_id >= (uint32_t)awb_start_frame && frame_id % awb_interval == 0;
    if (sample_ae || sample_awb) {
      vb->sync(VISIONBUF_SYNC_FROM_DEVICE);
      const uint8_t *nv12 = (const uint8_t *)vb->addr;
      if (sample_ae) {
        set_camera_exposure(calculate_os04_ae_sample_nv12(nv12, camera.stride, camera.uv_offset,
                                                          ae_xywh, camera.output_width,
                                                          camera.output_height, 4, 4));
      }
      if (sample_awb) update_awb(nv12);
    }
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
  framed.setExposureValPercent(util::map_val(cur_ev[frame_id % 3],
    camera.sensor->min_ev, camera.sensor->max_ev, 0.0f, 100.0f));
  pm->send(camera.cc.publish_name, msg);
}

void camerad_thread() {
  LOG("-- One camerad starting (VFE PIX DMABUF required; no runtime RDI/MMAP CPU fallback)");

  VisionIpcServer v("camerad");

  int media_fd = open("/dev/media0", O_RDWR);
  if (media_fd < 0) {
    LOGE("failed to open /dev/media0");
    return;
  }
  for (int i = 0; i < 3; i++) {
    one_cams[i] = resolve_cam_config(media_fd, i);
    LOG("cam %d: csiphy=%u csid=%u vfe_rdi=%u rdi_dev=%d vfe_pix=%u pix_dev=%d pix_subdev=%d",
        i, one_cams[i].csiphy_entity, one_cams[i].csid_entity,
        one_cams[i].vfe_rdi_entity, one_cams[i].rdi_video_dev,
        one_cams[i].vfe_pix_entity, one_cams[i].pix_video_dev, one_cams[i].vfe_pix_subdev);
  }
  close(media_fd);

  reset_all_media_links();

  const char *single_cam_env = getenv("SINGLE_CAM");
  int single_cam = single_cam_env != nullptr ? atoi(single_cam_env[0] ? single_cam_env : "0") : -1;
  std::vector<std::unique_ptr<CameraState>> cams;
  for (const auto &config : ALL_CAMERA_CONFIGS) {
    if (config.camera_num > 2) continue;
    if (single_cam >= 0 && config.camera_num != single_cam) continue;
    if (getenv("DISABLE_WIDE_ROAD") != nullptr && config.stream_type == VISION_STREAM_WIDE_ROAD) continue;
    if (getenv("DISABLE_ROAD") != nullptr && config.stream_type == VISION_STREAM_ROAD) continue;
    if (getenv("DISABLE_DRIVER") != nullptr && config.stream_type == VISION_STREAM_DRIVER) continue;
    auto cam = std::make_unique<CameraState>(config);
    cam->init(&v);
    cams.emplace_back(std::move(cam));
  }

  v.start_listener();

  for (auto &cam : cams) {
    cam->camera.queue_all_buffers();
  }
  const bool forward_start = getenv("ASIUS_CAM_START_FORWARD") != nullptr;
  const int start_gap_us = getenv_int_clamped("ASIUS_CAM_START_GAP_US", 38000, 0, 1000000);
  auto stream_one = [&](const std::unique_ptr<CameraState> &cam) {
    cam->camera.stream_on();
    if (start_gap_us > 0) usleep(start_gap_us);
  };
  if (forward_start) {
    for (const auto &cam : cams) stream_one(cam);
  } else {
    for (auto it = cams.rbegin(); it != cams.rend(); ++it) stream_one(*it);
  }

  LOG("-- One camerad streaming");

  while (!do_exit) {
    for (auto &cam : cams) {
      if (!cam->camera.enabled) continue;

      uint64_t timestamp;
      int buf_idx = cam->camera.dequeue_frame(&timestamp);
      if (buf_idx < 0) {
        if (env_debug_frames) printf("cam %d: dequeue timeout\n", cam->camera.cc.camera_num);
        if (buf_idx != -ETIMEDOUT) {
          LOGW_100("cam %d: dequeue failed: %d (%s)", cam->camera.cc.camera_num, -buf_idx, strerror(-buf_idx));
          usleep(1000);
        }
        continue;
      }

      if (cam->camera.use_direct_vipc_buffers()) {
        cam->process_pix_frame(buf_idx, timestamp);
      } else {
        LOGE("cam %d: refusing non-DMABUF CPU-copy camera path on hardware VFE branch",
             cam->camera.cc.camera_num);
        continue;
      }

      if (env_debug_frames) {
        printf("cam %d frame %u buf %d ts %.2f ms exp %d gain %.3f (%s)\n",
               cam->camera.cc.camera_num, cam->frame_id, buf_idx, timestamp / 1e6,
               cam->exposure_time, cam->camera.sensor->sensor_analog_gains[cam->gain_idx],
               cam->camera.use_pix_v4l2 ? "VFE PIX V4L2" : "VFE PIX ioctl");
      }

      if (!cam->camera.use_custom_pix_ioctl()) cam->camera.queue_frame(buf_idx);
    }
  }

  LOG("-- One camerad stopping");
  for (auto &cam : cams) {
    cam->camera.stop_streaming();
  }
}
