#include "runtime_mesh.h"
#include <thread>
#include <iostream>

int get_lane_for_expert(const Cluster& c, int expert_id) {
  for (auto& e : c.expert_map) {
    if (e.expert_id == expert_id) return e.lane_id;
  }
  return 0;
}

void dispatch_experts_mesh(const Cluster& cluster,
                           std::vector<int> active_experts) {
  std::vector<std::thread> workers;
  for (int e : active_experts) {
    workers.emplace_back([&, e]() {
      int lane = get_lane_for_expert(cluster, e);
      // Placeholder: would build and submit command list to lane queue.
      std::cout << "expert " << e << " -> lane " << lane << "\n";
    });
  }
  for (auto& w : workers) w.join();
}
