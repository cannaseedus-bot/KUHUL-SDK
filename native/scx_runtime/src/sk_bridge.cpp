#include "runtime.h"
#include <string>
#include <iostream>

// Placeholder SK integration; in a real build, include Semantic Kernel headers.
// Expose a simple function to call run_inference; keeps interface stable.
std::string sk_run(const std::string& prompt) {
  std::cout << "[SK bridge] dispatching to SCX runtime\n";
  return run_inference(prompt);
}
