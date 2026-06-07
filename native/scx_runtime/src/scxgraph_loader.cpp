#include "scxgraph.h"
#include <string>
#include <vector>

namespace scx {

// ---------------------------------------------------------------------------
// Minimal JSON field extractors — no external dependencies.
// ---------------------------------------------------------------------------

static std::string extract_string_field(const std::string& json, const std::string& key) {
  const std::string pat = "\"" + key + "\"";
  auto p = json.find(pat);
  if (p == std::string::npos) return "";
  auto colon = json.find(':', p + pat.size());
  if (colon == std::string::npos) return "";
  auto q1 = json.find('"', colon);
  if (q1 == std::string::npos) return "";
  auto q2 = json.find('"', q1 + 1);
  if (q2 == std::string::npos) return "";
  return json.substr(q1 + 1, q2 - q1 - 1);
}

static float extract_float_field(const std::string& json, const std::string& key) {
  const std::string pat = "\"" + key + "\"";
  auto p = json.find(pat);
  if (p == std::string::npos) return 0.f;
  auto colon = json.find(':', p + pat.size());
  if (colon == std::string::npos) return 0.f;
  size_t vs = colon + 1;
  while (vs < json.size() && (json[vs] == ' ' || json[vs] == '\t' || json[vs] == '\n' || json[vs] == '\r'))
    ++vs;
  try { return std::stof(json.substr(vs)); } catch (...) { return 0.f; }
}

// Extract the first balanced block delimited by open/close after key.
static std::string extract_block(const std::string& json, const std::string& key, char open, char close) {
  const std::string pat = "\"" + key + "\"";
  auto p = json.find(pat);
  if (p == std::string::npos) return "";
  auto ob = json.find(open, p + pat.size());
  if (ob == std::string::npos) return "";
  int depth = 1;
  size_t i = ob + 1;
  while (i < json.size() && depth > 0) {
    if (json[i] == open)  ++depth;
    if (json[i] == close) --depth;
    ++i;
  }
  return json.substr(ob, i - ob);
}

// Parse [x, y] from a "pos": [x, y] field inside obj.
static bool parse_pos(const std::string& obj, float pos[2]) {
  auto p = obj.find("\"pos\"");
  if (p == std::string::npos) return false;
  auto bracket = obj.find('[', p);
  if (bracket == std::string::npos) return false;
  auto comma = obj.find(',', bracket + 1);
  auto endb  = obj.find(']', bracket + 1);
  if (comma == std::string::npos || endb == std::string::npos || comma > endb) return false;
  try {
    pos[0] = std::stof(obj.substr(bracket + 1, comma - bracket - 1));
    pos[1] = std::stof(obj.substr(comma  + 1, endb  - comma  - 1));
    return true;
  } catch (...) { return false; }
}

// Split a JSON array string into individual { } object strings.
static std::vector<std::string> split_array_objects(const std::string& arr) {
  std::vector<std::string> out;
  int depth = 0;
  size_t start = std::string::npos;
  for (size_t i = 0; i < arr.size(); ++i) {
    if (arr[i] == '{') {
      if (depth == 0) start = i;
      ++depth;
    } else if (arr[i] == '}') {
      --depth;
      if (depth == 0 && start != std::string::npos) {
        out.push_back(arr.substr(start, i - start + 1));
        start = std::string::npos;
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------

ScxGraph load_scxgraph(const std::string& content) {
  ScxGraph g;

  const std::string scx_block = extract_block(content, "scxGraph", '{', '}');
  if (scx_block.empty()) return g;

  g.coord_frame = extract_string_field(scx_block, "coordFrame");
  if (g.coord_frame.empty()) g.coord_frame = "triangle";

  // Nodes
  const std::string nodes_block = extract_block(scx_block, "nodes", '[', ']');
  for (const auto& obj : split_array_objects(nodes_block)) {
    ScxNode n;
    n.id     = extract_string_field(obj, "id");
    n.role   = extract_string_field(obj, "role");
    n.q      = extract_string_field(obj, "q");
    n.device = extract_string_field(obj, "device");
    parse_pos(obj, n.pos);
    if (!n.id.empty()) g.nodes.push_back(std::move(n));
  }

  // Edges
  const std::string edges_block = extract_block(scx_block, "edges", '[', ']');
  for (const auto& obj : split_array_objects(edges_block)) {
    ScxEdge e;
    e.id      = extract_string_field(obj, "id");
    e.from    = extract_string_field(obj, "from");
    e.to      = extract_string_field(obj, "to");
    e.type    = extract_string_field(obj, "type");
    e.metric  = extract_float_field(obj, "metric");
    e.entropy = extract_float_field(obj, "entropy");
    e.phase   = extract_float_field(obj, "phase");
    if (!e.id.empty()) g.edges.push_back(std::move(e));
  }

  g.ok = !g.nodes.empty();
  return g;
}

} // namespace scx
