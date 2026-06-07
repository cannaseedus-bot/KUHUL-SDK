#include "runtime.h"
#include "kbc1.h"
#include "kbc1_loader.h"
#include "gta1_loader.h"
#include "manifest_loader.h"
#include "semantic_prepass.h"
#include "scxgraph.h"
#include "shard_loader.h"
#include "../../xvm-d3d12/src/xvm_core.h"
#include <cstdlib>
#include <filesystem>
#include <string>
#include <sstream>
#include <thread>

// Declared in kbc1_compiler.cpp
KBC1_Program compile_minimal_16layer_moe();
KBC1_Program compile_from_manifest(const ManifestInfo& m);

// Declared in scxgraph_to_xvm.cpp
namespace scx { bool populate_xvm_from_graph(const scx::ScxGraph& graph, XVMState& vm); }

namespace {

std::filesystem::path find_upward(std::filesystem::path start, const std::filesystem::path& suffix) {
  for (int i = 0; i < 12; ++i) {
    auto candidate = start / suffix;
    if (std::filesystem::exists(candidate)) return candidate;
    if (!start.has_parent_path()) break;
    auto parent = start.parent_path();
    if (parent == start) break;
    start = parent;
  }
  return {};
}

std::filesystem::path discover_fpga_root() {
  if (const char* env = std::getenv("KUHUL_FPGA_ROOT")) {
    if (*env && std::filesystem::exists(env)) return std::filesystem::path(env);
  }
  auto cwd = std::filesystem::current_path();
  return find_upward(cwd, "native/dotnet/PowerShell-LLM/Models/native/fpga");
}

} // namespace

std::string run_inference(const std::string& prompt) {
  SemanticPrepassResult semantic_prepass = semantic_first_stop(prompt, "inference");
  const auto fpga_root = discover_fpga_root();
  const auto fpga_device = fpga_root / "ffp1" / "runtime" / "device.bin";
  const auto fpga_benchmark = fpga_root / "ffp1" / "runtime" / "benchmark.json";

  // ── Shard registry ─────────────────────────────────────────────────────────
  // Any model can read all available SCXQDDS shards by setting SHARD_ROOT to
  // the artifacts/training/ directory.  The registry is loaded once per call;
  // a production runtime would cache this globally.
  ShardRegistry shard_reg;
  const char* shard_root = std::getenv("SHARD_ROOT");
  if (shard_root) {
    shard_reg = load_shard_registry(shard_root);
  }

  // Optional: direct KBC1 from env
  KBC1_Program prog{};
  std::string kbc1Source = "compiled";
  size_t kbc1Bytes = 0;

  const char* kbc1_path = std::getenv("KBC1_PATH");
  if (kbc1_path) {
    if (load_kbc1(kbc1_path, prog)) {
      kbc1Source = "env";
      kbc1Bytes = prog.inst.size() * (2+2+4*4);
    } else {
      // fall back
      prog = {};
    }
  }

  // Optional: load manifest (lightweight, non-GPU path) if MANIFEST_PATH is set.
  ManifestInfo manifest{};
  const char* manifest_path = std::getenv("MANIFEST_PATH");
  if (manifest_path) {
    load_manifest(manifest_path, manifest);
  }

  // Optional: load GTA1 if GTA1_PATH is set.
  Gta1Info gta{};
  const char* gta_path = std::getenv("GTA1_PATH");
  bool gta_loaded = false;
  if (gta_path) {
    gta_loaded = load_gta1(gta_path, gta);
  }

  // If no env KBC1, use manifest-provided KBC1 bytes
  if (prog.inst.empty() && manifest.ok && !manifest.kbc1_bytes.empty()) {
    // parse manifest kbc1 bytes as same format (u16 op, u16 argc, u32 args[4])
    const auto& b = manifest.kbc1_bytes;
    size_t off = 0;
    while (off + 2 + 2 + 16 <= b.size()) {
      uint16_t op = *reinterpret_cast<const uint16_t*>(&b[off]); off += 2;
      uint16_t argc = *reinterpret_cast<const uint16_t*>(&b[off]); off += 2;
      uint32_t args[4];
      for (int i=0;i<4;i++) { args[i] = *reinterpret_cast<const uint32_t*>(&b[off]); off += 4; }
      KBC1_Inst inst{op, argc, {args[0], args[1], args[2], args[3]}};
      prog.inst.push_back(inst);
    }
    if (!prog.inst.empty()) {
      kbc1Source = "manifest";
      kbc1Bytes = b.size();
    }
  }

  // If still empty, compile from manifest graph or fallback minimal
  if (prog.inst.empty()) {
    if (manifest.ok) {
      prog = compile_from_manifest(manifest);
      kbc1Source = "compiled";
    } else {
      prog = compile_minimal_16layer_moe();
      kbc1Source = "compiled";
    }
    kbc1Bytes = prog.inst.size() * (2+2+4*4);
  }

  // XVM path: if the manifest has a parsed scxGraph, compile it into XVM fibers
  // and run them in parallel — one fiber per graph node.  This is the direct
  // manifest-OS execution path; KBC1/GPU is used as fallback when scxGraph is absent.
  bool xvm_ran = false;
  uint32_t xvm_fibers = 0;
  uint32_t xvm_ticks  = 0;
  if (manifest.ok && manifest.scx_graph.ok) {
    XVMState vm;
    if (scx::populate_xvm_from_graph(manifest.scx_graph, vm)) {
      const uint32_t threads = std::thread::hardware_concurrency();
      const uint64_t ticks   = static_cast<uint64_t>(manifest.scx_graph.nodes.size()) * 4;
      xvm_run_cpu_ticks_mt(vm, ticks, threads);
      xvm_fibers = vm.fiberCount();
      xvm_ticks  = static_cast<uint32_t>(vm.tick);
      xvm_ran    = true;
    }
  }

  // KBC1 / GPU path (runs when XVM is skipped, or in parallel for hybrid).
  const bool use_gpu = init_gpu_if_available();
  if (!xvm_ran) {
    if (use_gpu) {
      run_d3d12(prog, manifest.ok ? &manifest : nullptr);
    } else {
      run_cpu(prog);
    }
  }

  std::ostringstream oss;
  oss << "{ \"ok\": true"
      << ", \"semantic_first_stop\": " << (semantic_prepass.ok ? "true" : "false")
      << ", \"semantic_report\": \"" << json_escape(semantic_prepass.report_path) << "\""
      << ", \"semantic_input\": \"" << json_escape(semantic_prepass.input_path) << "\""
      << ", \"semantic_kernel_cli\": \"" << json_escape(semantic_prepass.semantic_kernel_cli) << "\""
      << ", \"semantic_exit_code\": " << semantic_prepass.exit_code
      << ", \"fpga\": " << (!fpga_root.empty() ? "true" : "false")
      << ", \"fpga_root\": \"" << json_escape(fpga_root.string()) << "\""
      << ", \"fpga_device\": " << (std::filesystem::exists(fpga_device) ? "true" : "false")
      << ", \"fpga_device_path\": \"" << json_escape(fpga_device.string()) << "\""
      << ", \"fpga_benchmark\": " << (std::filesystem::exists(fpga_benchmark) ? "true" : "false")
      << ", \"gpu\": " << (use_gpu ? "true" : "false")
      << ", \"xvm\": " << (xvm_ran ? "true" : "false");
  if (xvm_ran) {
    oss << ", \"xvm_fibers\": " << xvm_fibers
        << ", \"xvm_ticks\": "  << xvm_ticks
        << ", \"xvm_nodes\": "  << manifest.scx_graph.nodes.size()
        << ", \"xvm_edges\": "  << manifest.scx_graph.edges.size()
        << ", \"xvm_coord\": \"" << manifest.scx_graph.coord_frame << "\"";
  }
  oss << ", \"inst\": "         << prog.inst.size()
      << ", \"kbc1Source\": \"" << kbc1Source << "\""
      << ", \"kbc1Bytes\": "    << kbc1Bytes
      << ", \"gta\": "          << (gta_loaded ? "true" : "false");
  if (gta_loaded) {
    oss << ", \"coord_frame\": \"" << gta.coord_frame << "\""
        << ", \"nodes\": "  << gta.nodes.size()
        << ", \"edges\": "  << gta.edges.size()
        << ", \"tensors\": " << gta.tensors.size();
  }
  if (manifest.ok) {
    oss << ", \"manifest\": true"
        << ", \"manifest_coord\": \"" << manifest.coord_frame << "\""
        << ", \"manifest_nodes\": "   << manifest.nodes
        << ", \"manifest_edges\": "   << manifest.edges
        << ", \"manifest_tensors\": " << manifest.tensors.size();
  } else {
    oss << ", \"manifest\": false";
  }
  oss << ", \"prompt_hash\": " << std::hash<std::string>{}(prompt);
  // Shard registry — available to all models via SHARD_ROOT
  if (shard_reg.ok) {
    oss << ", \"shard_registry\": true"
        << ", \"shard_count\": " << shard_reg.shards.size();
  } else {
    oss << ", \"shard_registry\": false";
  }
  oss << " }";
  return oss.str();
}
