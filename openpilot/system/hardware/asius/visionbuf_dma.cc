#include "msgq/visionipc/visionbuf.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {

int sync_dma_buffer(int fd, uint64_t flags) {
  struct dma_buf_sync sync = {.flags = flags};
  int ret;
  do {
    ret = ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
  } while (ret < 0 && errno == EINTR);
  return (ret == 0 || errno == ENOTTY) ? 0 : ret;
}

}  // namespace

void VisionBuf::allocate(size_t length) {
  const long page_size = sysconf(_SC_PAGESIZE);
  const size_t alignment = page_size > 0 ? static_cast<size_t>(page_size) : 4096;

  len = length;
  mmap_len = ((length + sizeof(uint64_t) + alignment - 1) / alignment) * alignment;

  int heap_fd = open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
  assert(heap_fd >= 0);

  struct dma_heap_allocation_data allocation = {};
  allocation.len = mmap_len;
  allocation.fd_flags = O_RDWR | O_CLOEXEC;
  int ret;
  do {
    ret = ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &allocation);
  } while (ret < 0 && errno == EINTR);
  close(heap_fd);
  assert(ret == 0);

  fd = allocation.fd;
  addr = mmap(nullptr, mmap_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  assert(addr != MAP_FAILED);
  memset(addr, 0, mmap_len);

  handle = -1;
  frame_id = reinterpret_cast<uint64_t *>(static_cast<uint8_t *>(addr) + len);
}

void VisionBuf::import() {
  assert(fd >= 0);
  addr = mmap(nullptr, mmap_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  assert(addr != MAP_FAILED);
  handle = -1;
  frame_id = reinterpret_cast<uint64_t *>(static_cast<uint8_t *>(addr) + len);
}

void VisionBuf::init_yuv(size_t init_width, size_t init_height,
                         size_t init_stride, size_t init_uv_offset) {
  width = init_width;
  height = init_height;
  stride = init_stride;
  uv_offset = init_uv_offset;
  y = static_cast<uint8_t *>(addr);
  uv = y + uv_offset;
}

int VisionBuf::sync(int dir) {
  assert(dir == VISIONBUF_SYNC_FROM_DEVICE || dir == VISIONBUF_SYNC_TO_DEVICE);
  const uint64_t access = dir == VISIONBUF_SYNC_FROM_DEVICE ? DMA_BUF_SYNC_READ : DMA_BUF_SYNC_WRITE;
  int ret = sync_dma_buffer(fd, DMA_BUF_SYNC_START | access);
  return ret == 0 ? sync_dma_buffer(fd, DMA_BUF_SYNC_END | access) : ret;
}

int VisionBuf::free() {
  int ret = munmap(addr, mmap_len);
  if (ret != 0) return ret;
  return close(fd);
}

uint64_t VisionBuf::get_frame_id() {
  return *frame_id;
}

void VisionBuf::set_frame_id(uint64_t id) {
  *frame_id = id;
}
