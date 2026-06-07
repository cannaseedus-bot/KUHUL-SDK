#include "kbc1.h"
#include "runtime.h"
#include "xml_cluster.h"
#include <iostream>
#include <string>

extern KBC1_Program compile_minimal_16layer_moe();

static std::string join_args(int argc, char** argv, int start) {
  std::string out;
  for (int i = start; i < argc; ++i) {
    if (!out.empty()) out += " ";
    out += argv[i];
  }
  return out;
}

int main(int argc, char** argv) {
  if (argc >= 2) {
    const std::string cmd = argv[1];
    if (cmd == "infer") {
      std::string prompt = join_args(argc, argv, 2);
      std::cout << run_inference(prompt) << "\n";
      return 0;
    }
    if (cmd == "train_step") {
      if (argc < 4) {
        std::cerr << "Usage: scx_runtime.exe train_step <target_id> <prompt>\n";
        return 2;
      }
      int target_id = std::stoi(argv[2]);
      std::string prompt = join_args(argc, argv, 3);
      std::cout << run_train_step(prompt, target_id) << "\n";
      return 0;
    }
  }

  auto prog = compile_minimal_16layer_moe();

  bool use_gpu = init_gpu_if_available();
  if (use_gpu) {
    std::cout << "Running on D3D12 GPU path\n";
    run_d3d12(prog, nullptr);
  } else {
    std::cout << "Running on CPU fallback\n";
    run_cpu(prog);
  }

  return 0;
}
