#include "runtime.h"
#include <thread>
#include <vector>

// forward decls
void layernorm_dx(float* x, float* out, int n);
void matmul_int4_dx(const uint8_t* A, const uint8_t* B, float* C, int M, int N, int K);

void run_cpu(const KBC1_Program& p) {
  const size_t threads = std::max<size_t>(1, std::thread::hardware_concurrency());
  std::vector<std::thread> pool;
  for (size_t t = 0; t < threads; ++t) {
    pool.emplace_back([&, t]() {
      for (size_t i = t; i < p.inst.size(); i += threads) {
        switch (p.inst[i].op) {
          case OP_LAYERNORM: {
            // Small deterministic buffer for the CPU fallback path.
            float buf[128], out[128];
            layernorm_dx(buf, out, 128);
            break;
          }
          case OP_ATTN_QKV:
          case OP_MOE_ROUTE: {
            uint8_t a[128], b[128];
            float c[16];
            matmul_int4_dx(a, b, c, 4, 4, 32);
            break;
          }
          default:
            break;
        }
      }
    });
  }
  for (auto& th : pool) th.join();
}
