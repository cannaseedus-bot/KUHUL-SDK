#include "dds.h"
#include <cstddef>
#include <stdexcept>
#include <vector>

SCX_DDS_Header parse_dds_header(const uint8_t* data, size_t size) {
  if (size < sizeof(SCX_DDS_Header)) throw std::runtime_error("dds too small");
  SCX_DDS_Header h{};
  size_t off = 0;
  auto rd32 = [&](uint32_t& v) {
    v = data[off] | (data[off + 1] << 8) | (data[off + 2] << 16) | (data[off + 3] << 24);
    off += 4;
  };
  rd32(h.magic);
  h.version = data[off] | (data[off + 1] << 8); off += 2;
  h.bits = data[off] | (data[off + 1] << 8); off += 2;
  rd32(h.tensor_count);
  uint64_t off64 = 0;
  for (int i = 0; i < 8; ++i) off64 |= static_cast<uint64_t>(data[off + i]) << (8 * i);
  h.data_offset = off64;

  if (h.magic != SCX_DDS_MAGIC) throw std::runtime_error("dds bad magic");
  if (h.tensor_count == 0) throw std::runtime_error("dds empty");
  if (h.data_offset < sizeof(SCX_DDS_Header)) throw std::runtime_error("dds invalid data offset");
  return h;
}
