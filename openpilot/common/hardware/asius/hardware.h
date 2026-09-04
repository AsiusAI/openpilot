#pragma once

#include <map>
#include <string>

#include "common/hardware/base.h"
#include "common/util.h"

class HardwareAsius : public HardwareNone {
public:
  static std::string get_name() { return "v0"; }

  static cereal::InitData::DeviceType get_device_type() {
    return cereal::InitData::DeviceType::V0;
  }

  static std::string get_serial() {
    return util::strip(util::read_file("/sys/devices/soc0/serial_number"));
  }

  static std::map<std::string, std::string> get_init_logs(bool = false) {
    std::map<std::string, std::string> logs = {
      {"/BUILD", util::read_file("/BUILD")},
      {"lsblk", util::check_output("lsblk -o NAME,SIZE,STATE,VENDOR,MODEL,REV,SERIAL")},
      {"ufs health", util::check_output(
        "grep -H . "
        "/sys/bus/platform/drivers/ufshcd-*/*/string_descriptors/manufacturer_name "
        "/sys/bus/platform/drivers/ufshcd-*/*/string_descriptors/product_name "
        "/sys/bus/platform/drivers/ufshcd-*/*/string_descriptors/product_revision "
        "/sys/bus/platform/drivers/ufshcd-*/*/device_descriptor/specification_version "
        "/sys/bus/platform/drivers/ufshcd-*/*/health_descriptor/eol_info "
        "/sys/bus/platform/drivers/ufshcd-*/*/health_descriptor/life_time_estimation_a "
        "/sys/bus/platform/drivers/ufshcd-*/*/health_descriptor/life_time_estimation_b "
        "/sys/bus/platform/drivers/ufshcd-*/*/critical_health "
        "/sys/bus/platform/drivers/ufshcd-*/*/power_info/gear "
        "/sys/bus/platform/drivers/ufshcd-*/*/power_info/lane "
        "/sys/bus/platform/drivers/ufshcd-*/*/power_info/mode "
        "/sys/bus/platform/drivers/ufshcd-*/*/power_info/rate "
        "/sys/bus/platform/drivers/ufshcd-*/*/power_info/link_state 2>/dev/null")},
      {"ufs errors", util::check_output("sudo cat /sys/kernel/debug/ufshcd/*/stats 2>/dev/null")},
    };

    const std::string cmdline = util::read_file("/proc/cmdline");
    constexpr char slot_key[] = "vamos.slot=";
    const size_t slot_start = cmdline.find(slot_key);
    if (slot_start != std::string::npos) {
      const size_t value_start = slot_start + sizeof(slot_key) - 1;
      const size_t value_end = cmdline.find(' ', value_start);
      logs["boot slot"] = cmdline.substr(value_start, value_end - value_start);
    }

    return logs;
  }
};
