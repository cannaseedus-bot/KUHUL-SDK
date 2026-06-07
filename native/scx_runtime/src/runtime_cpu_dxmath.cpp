#include <DirectXMath.h>
#include "runtime.h"
#include "runtime_state.h"

using namespace DirectX;

static inline float unpack4(uint8_t v, int idx) {
  int nib = (v >> (idx * 4)) & 0xF;
  return (nib - 8) / 8.0f;
}

void layernorm_dx(float* x, float* out, int n) {
  XMVECTOR mean = XMVectorZero();
  for (int i = 0; i < n; i += 4) {
    mean = XMVectorAdd(mean, XMLoadFloat4(reinterpret_cast<XMFLOAT4*>(&x[i])));
  }
  mean = XMVectorScale(mean, 1.0f / n);

  XMVECTOR var = XMVectorZero();
  for (int i = 0; i < n; i += 4) {
    XMVECTOR v = XMLoadFloat4(reinterpret_cast<XMFLOAT4*>(&x[i]));
    XMVECTOR diff = XMVectorSubtract(v, mean);
    var = XMVectorAdd(var, XMVectorMultiply(diff, diff));
  }
  var = XMVectorScale(var, 1.0f / n);
  XMVECTOR inv_std = XMVectorReciprocalSqrt(var);

  for (int i = 0; i < n; i += 4) {
    XMVECTOR v = XMLoadFloat4(reinterpret_cast<XMFLOAT4*>(&x[i]));
    XMVECTOR norm = XMVectorMultiply(XMVectorSubtract(v, mean), inv_std);
    XMStoreFloat4(reinterpret_cast<XMFLOAT4*>(&out[i]), norm);
  }
}

void matmul_int4_dx(const uint8_t* A, const uint8_t* B, float* C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      XMVECTOR acc = XMVectorZero();
      for (int k = 0; k < K; k += 2) {
        float a0 = unpack4(A[(i * K + k) >> 1], 0);
        float a1 = unpack4(A[(i * K + k) >> 1], 1);
        float b0 = unpack4(B[(k * N + j) >> 1], 0);
        float b1 = unpack4(B[(k * N + j) >> 1], 1);
        XMVECTOR va = XMVectorSet(a0, a1, 0, 0);
        XMVECTOR vb = XMVectorSet(b0, b1, 0, 0);
        acc = XMVectorAdd(acc, XMVectorMultiply(va, vb));
      }
      float out[4];
      XMStoreFloat4(reinterpret_cast<XMFLOAT4*>(out), acc);
      C[i * N + j] = out[0] + out[1];
    }
  }
}
