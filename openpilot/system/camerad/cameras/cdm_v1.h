#pragma once

#include "system/camerad/cameras/cdm.h"

struct reg_write {
  uint32_t offset;
  uint32_t value;
};

struct dmi_upload {
  uint32_t cfg_offset;
  uint8_t ram_select;
  const uint32_t *data;
  uint32_t count;
};

void collect_cont(std::vector<reg_write> &out, uint32_t base, const std::vector<uint32_t> &vals);
void collect_random(std::vector<reg_write> &out, const std::vector<uint32_t> &vals);
