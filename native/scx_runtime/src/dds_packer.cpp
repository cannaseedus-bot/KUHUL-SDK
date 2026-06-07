#include "dds.h"

std::vector<uint8_t> pack_int4(const std::vector<float>& in) {
  std::vector<uint8_t> out((in.size() + 1) / 2);
  for (size_t i = 0; i < in.size(); i += 2) {
    uint8_t a = static_cast<uint8_t>(in[i] * 7.5f) & 0xF;
    uint8_t b = (i + 1 < in.size()) ? (static_cast<uint8_t>(in[i + 1] * 7.5f) & 0xF) : 0;
    out[i / 2] = a | (b << 4);
  }
  return out;
}
