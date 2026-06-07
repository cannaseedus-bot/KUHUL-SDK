#include "runtime_state.h"

uint8_t kv_delta_encode(float v, float prev) {
  float d = v - prev;
  int q = static_cast<int>(d * 8.0f) + 8; // map to [-8,7] -> [0,15]
  if (q < 0) q = 0;
  if (q > 15) q = 15;
  return static_cast<uint8_t>(q);
}

float kv_delta_decode(float prev, uint8_t nib) {
  float d = (static_cast<int>(nib & 0xF) - 8) / 8.0f;
  return prev + d;
}
