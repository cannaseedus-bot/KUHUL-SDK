#include "xml_cluster.h"
#include <fstream>
#include <sstream>
#include <string>

namespace {

int getIntAttr(const std::string& line, const std::string& key, int def = 0) {
  auto pos = line.find(key + "=\"");
  if (pos == std::string::npos) return def;
  pos += key.size() + 2;
  auto end = line.find('"', pos);
  if (end == std::string::npos) return def;
  return std::stoi(line.substr(pos, end - pos));
}

std::string getStrAttr(const std::string& line, const std::string& key, const std::string& def = "") {
  auto pos = line.find(key + "=\"");
  if (pos == std::string::npos) return def;
  pos += key.size() + 2;
  auto end = line.find('"', pos);
  if (end == std::string::npos) return def;
  return line.substr(pos, end - pos);
}

} // namespace

Cluster load_cluster(const char* path) {
  Cluster c;
  std::ifstream in(path);
  if (!in) return c;
  std::string line;
  while (std::getline(in, line)) {
    if (line.find("<node") != std::string::npos) {
      Node n;
      n.id = getIntAttr(line, "id", 0);
      n.role = getStrAttr(line, "role", "cpu");
      n.memory = getStrAttr(line, "memory", "");
      n.threads = getIntAttr(line, "threads", 0);
      c.nodes.push_back(n);
    } else if (line.find("<gpu_lane") != std::string::npos) {
      GPULane l;
      l.id = getIntAttr(line, "id", 0);
      l.device = getIntAttr(line, "device", 0);
      l.queue = getStrAttr(line, "queue", "compute");
      c.lanes.push_back(l);
    } else if (line.find("<map") != std::string::npos && line.find("expert=") != std::string::npos) {
      ExpertMap m;
      m.expert_id = getIntAttr(line, "expert", 0);
      m.lane_id = getIntAttr(line, "lane", 0);
      c.expert_map.push_back(m);
    }
  }
  return c;
}
