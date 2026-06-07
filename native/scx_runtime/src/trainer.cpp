#include "runtime.h"
#include "semantic_prepass.h"
#include "shard_loader.h"
#include <DirectXMath.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <functional>

using namespace DirectX;

struct Tensor {
  std::vector<float> data;
};

// Simple global weights for demo; size matches vocab=16.
static std::vector<float> g_weights(16, 0.0f);

static Tensor forward_pass(const std::string& prompt) {
  Tensor t;
  t.data.resize(g_weights.size());
  // Deterministic logits from prompt hash + weights.
  size_t h = std::hash<std::string>{}(prompt);
  for (size_t i = 0; i < t.data.size(); ++i) {
    // mix hash with weight
    uint32_t v = static_cast<uint32_t>((h >> (i % 16)) ^ (0x9e3779b9u * (i + 1)));
    float base = static_cast<int>(v & 0xFF) / 255.0f;
    t.data[i] = base + g_weights[i];
  }
  return t;
}

static float cross_entropy(const Tensor& logits, int target) {
  float max_logit = -1e30f;
  for (float v : logits.data) if (v > max_logit) max_logit = v;
  float sum = 0.0f;
  for (float v : logits.data) sum += std::exp(v - max_logit);
  float log_prob = logits.data[target] - max_logit - std::log(sum + 1e-9f);
  return -log_prob;
}

static Tensor softmax_grad(const Tensor& logits, int target) {
  Tensor g;
  g.data.resize(logits.data.size());
  float max_logit = -1e30f;
  for (float v : logits.data) if (v > max_logit) max_logit = v;
  float sum = 0.0f;
  for (float v : logits.data) sum += std::exp(v - max_logit);
  for (size_t i = 0; i < logits.data.size(); ++i) {
    float p = std::exp(logits.data[i] - max_logit) / (sum + 1e-9f);
    g.data[i] = p - (i == static_cast<size_t>(target) ? 1.0f : 0.0f);
  }
  return g;
}

static void sgd_update_dx(std::vector<float>& w, const Tensor& grad, float lr) {
  size_t n = w.size();
  const float* gptr = grad.data.data();
  float* wptr = w.data();
  XMVECTOR lr_v = XMVectorReplicate(lr);
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    XMVECTOR wv = XMLoadFloat4(reinterpret_cast<XMFLOAT4*>(wptr + i));
    XMVECTOR gv = XMLoadFloat4(reinterpret_cast<const XMFLOAT4*>(gptr + i));
    XMVECTOR nw = XMVectorSubtract(wv, XMVectorMultiply(lr_v, gv));
    XMStoreFloat4(reinterpret_cast<XMFLOAT4*>(wptr + i), nw);
  }
  for (; i < n; ++i) {
    wptr[i] -= lr * gptr[i];
  }
}

static std::string pack_int4_to_dds(const std::vector<float>& w) {
  // Simple DDS header + one tensor; int4 pack two values per byte.
  const uint32_t MAGIC = 0x53444453; // 'SDDS'
  const uint16_t VERSION = 1;
  const uint16_t BITS = 4;
  const uint32_t TENSOR_COUNT = 1;
  const uint64_t HEADER_SIZE = 4 + 2 + 2 + 4 + 8;
  const uint64_t TENSOR_REC_SIZE = 4 + 2 + 2 + 4 * 4 + 8 + 8;

  uint64_t data_offset = HEADER_SIZE + TENSOR_REC_SIZE;
  std::vector<uint8_t> payload((w.size() + 1) / 2, 0);
  for (size_t i = 0; i < w.size(); i += 2) {
    int a = static_cast<int>(std::round(w[i] * 8.0f)) + 8;
    int b = (i + 1 < w.size()) ? static_cast<int>(std::round(w[i + 1] * 8.0f)) + 8 : 0;
    if (a < 0) a = 0; if (a > 15) a = 15;
    if (b < 0) b = 0; if (b > 15) b = 15;
    payload[i >> 1] = static_cast<uint8_t>((a & 0xF) | ((b & 0xF) << 4));
  }

  uint64_t size_bytes = payload.size();
  uint64_t total = data_offset + size_bytes;
  std::vector<uint8_t> buf(static_cast<size_t>(total), 0);
  size_t off = 0;
  auto w32 = [&](uint32_t v){ buf[off++] = v & 0xFF; buf[off++] = (v>>8)&0xFF; buf[off++] = (v>>16)&0xFF; buf[off++] = (v>>24)&0xFF; };
  auto w16 = [&](uint16_t v){ buf[off++] = v & 0xFF; buf[off++] = (v>>8)&0xFF; };
  auto w64 = [&](uint64_t v){ for(int i=0;i<8;i++){ buf[off++] = static_cast<uint8_t>((v>>(8*i)) & 0xFF); } };

  w32(MAGIC); w16(VERSION); w16(BITS); w32(TENSOR_COUNT); w64(data_offset);

  // tensor record
  w32(0);      // id
  w16(BITS);   // dtype
  w16(1);      // dims
  w32(static_cast<uint32_t>(w.size())); w32(0); w32(0); w32(0); // shape[0]=size, rest 0
  w64(data_offset);
  w64(size_bytes);

  std::copy(payload.begin(), payload.end(), buf.begin() + data_offset);

  return std::string(reinterpret_cast<char*>(buf.data()), buf.size());
}

static void write_dds(const std::string& path, const std::vector<float>& w) {
  auto bin = pack_int4_to_dds(w);
  std::ofstream out(path, std::ios::binary);
  out.write(bin.data(), static_cast<std::streamsize>(bin.size()));
}

std::string run_train_step(const std::string& prompt, int target_id) {
  SemanticPrepassResult semantic_prepass = semantic_first_stop(prompt, "train_step");

  // Forward pass through the deterministic scaffold model.
  Tensor logits = forward_pass(prompt);

  // Loss
  float loss = cross_entropy(logits, target_id);

  // Backward
  Tensor grad = softmax_grad(logits, target_id);

  // Update (SGD)
  const float lr = 1e-3f;
  sgd_update_dx(g_weights, grad, lr);

  // Persist weights to DDS
  write_dds("C:\\public_html\\MX2LM\\codex\\AS-XCFE\\artifacts\\training\\trained.weights.dds", g_weights);

  std::ostringstream oss;
  oss << "{ \"ok\": true"
      << ", \"semantic_first_stop\": " << (semantic_prepass.ok ? "true" : "false")
      << ", \"semantic_report\": \"" << json_escape(semantic_prepass.report_path) << "\""
      << ", \"semantic_exit_code\": " << semantic_prepass.exit_code
      << ", \"loss\": " << loss
      << ", \"gpu\": false }";
  return oss.str();
}

// ---------------------------------------------------------------------------
// run_train_from_shard
//
// Loads a SCXQDDS shard (produced by shard-artifacts.js --apply), then slides
// a context window of length `context_len` over the INT8 token sequence.
// Each step treats token[i + context_len] as the next-token target and the
// djb2 hash of the preceding `context_len` bytes as the prompt key.
// SGD + DirectXMath weight update — same path as run_train_step.
// ---------------------------------------------------------------------------
std::string run_train_from_shard(const std::string& shard_path,
                                  int   context_len,
                                  float lr)
{
  const ShardData sd = load_sqdds_shard(shard_path);
  if (!sd.ok || sd.token_count == 0) {
    std::ostringstream oss;
    oss << "{ \"ok\": false, \"error\": \"shard load failed\", \"path\": \""
        << shard_path << "\" }";
    return oss.str();
  }

  if (context_len < 1) context_len = 1;
  const size_t n = sd.token_count;
  if (n <= static_cast<size_t>(context_len)) {
    std::ostringstream oss;
    oss << "{ \"ok\": false, \"error\": \"shard too small\","
        << " \"token_count\": " << n << " }";
    return oss.str();
  }

  float total_loss = 0.0f;
  size_t steps     = 0;

  for (size_t i = 0; i + static_cast<size_t>(context_len) < n; ++i) {
    // Build a deterministic prompt key from the context window bytes.
    uint32_t h = 5381u;
    for (int k = 0; k < context_len; ++k) {
      h = ((h << 5) + h) ^ static_cast<uint32_t>(sd.tokens[i + k]);
    }
    const std::string ctx_key = std::to_string(h);

    // Target is the next token (clamped to vocab size)
    const int target = static_cast<int>(sd.tokens[i + context_len])
                       % static_cast<int>(g_weights.size());

    Tensor logits = forward_pass(ctx_key);
    total_loss   += cross_entropy(logits, target);
    Tensor grad   = softmax_grad(logits, target);
    sgd_update_dx(g_weights, grad, lr);
    ++steps;
  }

  // Persist updated weights as SCXQDDS-compatible DDS
  write_dds("C:\\public_html\\MX2LM\\codex\\AS-XCFE\\artifacts\\training\\trained.weights.dds", g_weights);

  const float avg_loss = steps > 0 ? total_loss / static_cast<float>(steps) : 0.0f;

  std::ostringstream oss;
  oss << "{ \"ok\": true"
      << ", \"shard\": \"" << shard_path << "\""
      << ", \"token_count\": " << sd.token_count
      << ", \"steps\": " << steps
      << ", \"context_len\": " << context_len
      << ", \"avg_loss\": " << avg_loss
      << ", \"gpu\": false }";
  return oss.str();
}
