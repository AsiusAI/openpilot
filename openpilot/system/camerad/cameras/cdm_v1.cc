#include "system/camerad/cameras/cdm_v1.h"

void collect_cont(std::vector<reg_write> &out, uint32_t base, const std::vector<uint32_t> &vals) {
  for (size_t i = 0; i < vals.size(); i++) {
    out.push_back({base + (uint32_t)(i * 4), vals[i]});
  }
}

void collect_random(std::vector<reg_write> &out, const std::vector<uint32_t> &vals) {
  for (size_t i = 0; i + 1 < vals.size(); i += 2) {
    out.push_back({vals[i], vals[i + 1]});
  }
}
