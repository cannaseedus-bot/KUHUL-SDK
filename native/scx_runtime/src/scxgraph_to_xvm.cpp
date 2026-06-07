#include "scxgraph.h"
#include "../../xvm-d3d12/src/xvm_core.h"
#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// ScxGraph → XVMState compiler
//
// Mapping:
//   node  → one XVM fiber (fiber index == node index)
//   edge  → one shared memory slot (slot index == edge index)
//   role  → fiber bytecode program
//   pos   → initial r0/r1 register values (quantized to 0-255)
//
// Shared memory layout:
//   slots [0 .. edges.size()-1]  = inter-fiber edge signals
//   slot  [edges.size()]         = global accumulator (output of sink nodes)
// ---------------------------------------------------------------------------

namespace scx {

static uint8_t quant(float v) {
  int i = static_cast<int>(v * 255.f + 0.5f);
  if (i < 0)   i = 0;
  if (i > 255) i = 255;
  return static_cast<uint8_t>(i);
}

// XVM opcodes (must match xvm_core.cpp)
static constexpr uint8_t kLoadConst   = 0x01;
static constexpr uint8_t kMov         = 0x02;
static constexpr uint8_t kAdd         = 0x03;
static constexpr uint8_t kMul         = 0x05;
static constexpr uint8_t kAtomicAdd   = 0x10;
static constexpr uint8_t kLoadShared  = 0x30;
static constexpr uint8_t kStoreShared = 0x31;
static constexpr uint8_t kReturn      = 0x3f;

static void emit(std::vector<uint8_t>& code, std::initializer_list<uint8_t> bytes) {
  for (auto b : bytes) code.push_back(b);
}

// Emit the bytecode program for one node.
// in_slot  = shared slot to read from  (0xFF = no incoming edge)
// out_slot = shared slot to write to   (0xFF = no outgoing edge)
// accum    = accumulator slot index
static void emit_node_program(
    std::vector<uint8_t>& code,
    const ScxNode& node,
    uint8_t in_slot,
    uint8_t out_slot,
    uint8_t accum)
{
  const uint8_t px = quant(node.pos[0]);
  const uint8_t py = quant(node.pos[1]);

  // Source: generates initial signal (no incoming edge expected)
  const bool is_source = (node.role == "vector"   || node.role == "intent"   ||
                          node.role == "sensory"  || node.role == "router"   ||
                          node.role == "query"    || node.role == "framework");
  // Mid: transforms and propagates signal
  const bool is_mid    = (node.role == "matrix"       || node.role == "dispatch"    ||
                          node.role == "associative"  || node.role == "planner"     ||
                          node.role == "fusion"       || node.role == "orchestrator" ||
                          node.role == "di-router");
  // Sink: micronaut-di, executor, builder, executive, runtime, result, or unknown

  if (is_source) {
    // Load spatial position, compute proxy dot-product, push to outgoing edge.
    emit(code, {kLoadConst, 0, px});                          // r0 = px
    emit(code, {kLoadConst, 1, py});                          // r1 = py
    emit(code, {kMul, 0, 1});                                 // r0 *= r1
    if (out_slot != 0xFF)
      emit(code, {kStoreShared, out_slot, 0});                // shared[out] = r0
    emit(code, {kReturn});

  } else if (is_mid) {
    // Read from incoming edge, accumulate with spatial value, propagate.
    if (in_slot != 0xFF)
      emit(code, {kLoadShared, 2, in_slot});                  // r2 = shared[in]
    emit(code, {kLoadConst, 0, px});                          // r0 = px
    if (in_slot != 0xFF)
      emit(code, {kAdd, 0, 2});                               // r0 += r2
    emit(code, {kLoadConst, 1, py});                          // r1 = py
    emit(code, {kMul, 0, 1});                                 // r0 *= r1
    if (out_slot != 0xFF)
      emit(code, {kStoreShared, out_slot, 0});                // shared[out] = r0
    emit(code, {kReturn});

  } else {
    // Sink node: drain incoming signal into the global accumulator.
    if (in_slot != 0xFF) {
      emit(code, {kLoadShared, 2, in_slot});                  // r2 = shared[in]
      emit(code, {kAtomicAdd, accum, 2});                     // shared[accum] += r2
      emit(code, {kMov, 0, 2});                               // r0 = r2  (readable result)
    } else {
      emit(code, {kLoadConst, 0, px});                        // r0 = px (no input)
    }
    emit(code, {kReturn});
  }
}

// ---------------------------------------------------------------------------

bool populate_xvm_from_graph(const ScxGraph& graph, XVMState& vm) {
  if (!graph.ok || graph.nodes.empty()) return false;

  vm.code.clear();
  vm.fibers.clear();
  vm.shared.clear();
  vm.tick = 0;

  const auto edge_count = static_cast<uint8_t>(
      graph.edges.size() < 254 ? graph.edges.size() : 254);
  const uint8_t accum_slot = edge_count;          // accumulator sits after edge slots

  vm.shared.assign(static_cast<size_t>(edge_count) + 1, 0u);

  // Edge lookup helpers
  auto in_slot_for = [&](const std::string& node_id) -> uint8_t {
    for (size_t i = 0; i < graph.edges.size() && i < 254; ++i)
      if (graph.edges[i].to == node_id) return static_cast<uint8_t>(i);
    return 0xFF;
  };
  auto out_slot_for = [&](const std::string& node_id) -> uint8_t {
    for (size_t i = 0; i < graph.edges.size() && i < 254; ++i)
      if (graph.edges[i].from == node_id) return static_cast<uint8_t>(i);
    return 0xFF;
  };

  for (size_t n = 0; n < graph.nodes.size(); ++n) {
    const ScxNode& node = graph.nodes[n];
    const uint32_t code_start = static_cast<uint32_t>(vm.code.size());

    emit_node_program(
        vm.code, node,
        in_slot_for(node.id),
        out_slot_for(node.id),
        accum_slot);

    XVMFiber fiber{};
    fiber.pc    = code_start;
    fiber.flags = 1;                                          // active
    fiber.r0    = quant(node.pos[0]);
    fiber.r1    = quant(node.pos[1]);
    fiber.r2    = 0;
    fiber.r3    = static_cast<uint32_t>(n);                  // node index
    vm.fibers.push_back(fiber);
  }

  return !vm.code.empty();
}

} // namespace scx
