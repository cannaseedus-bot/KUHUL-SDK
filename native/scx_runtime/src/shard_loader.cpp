#include "shard_loader.h"
#include "dds.h"
#include <fstream>
#include <iterator>
#include <sstream>

// ---------------------------------------------------------------------------
// load_sqdds_shard
//
// Wire format (matches shard-artifacts.js packShardAsSqdds):
//
//   [Header  20 bytes]  magic(4) version(2) bits(2) tensor_count(4) data_offset(8)
//   [TensorR 40 bytes]  id(4) dtype(2) dims(2) shape[4](16) offset(8) size(8)
//   [Payload N bytes ]  raw INT8 token bytes
//
// Only the first tensor record is read; dtype must be 8 (INT8).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Minimal JSON helpers (no external deps) — extract a string value for a key.
// Only handles flat string / number fields, sufficient for shard-index.json.
// ---------------------------------------------------------------------------
static std::string json_string(const std::string& src, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  auto pos = src.find(needle);
  if (pos == std::string::npos) return {};
  pos = src.find(':', pos + needle.size());
  if (pos == std::string::npos) return {};
  pos = src.find('"', pos + 1);
  if (pos == std::string::npos) return {};
  const auto end = src.find('"', pos + 1);
  if (end == std::string::npos) return {};
  return src.substr(pos + 1, end - pos - 1);
}

static uint64_t json_uint64(const std::string& src, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  auto pos = src.find(needle);
  if (pos == std::string::npos) return 0;
  pos = src.find(':', pos + needle.size());
  if (pos == std::string::npos) return 0;
  while (pos < src.size() && (src[pos] == ':' || src[pos] == ' ')) ++pos;
  return static_cast<uint64_t>(std::stoull(src.substr(pos)));
}

// Split a JSON array literal "[{...},{...}]" into individual "{...}" strings.
static std::vector<std::string> split_json_objects(const std::string& arr) {
  std::vector<std::string> out;
  int depth = 0;
  std::size_t start = std::string::npos;
  for (std::size_t i = 0; i < arr.size(); ++i) {
    if (arr[i] == '{') {
      if (depth++ == 0) start = i;
    } else if (arr[i] == '}') {
      if (--depth == 0 && start != std::string::npos) {
        out.push_back(arr.substr(start, i - start + 1));
        start = std::string::npos;
      }
    }
  }
  return out;
}

// Extract the raw JSON array string for a given key.
static std::string json_array(const std::string& src, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  auto pos = src.find(needle);
  if (pos == std::string::npos) return {};
  pos = src.find('[', pos + needle.size());
  if (pos == std::string::npos) return {};
  int depth = 0;
  for (std::size_t i = pos; i < src.size(); ++i) {
    if (src[i] == '[') ++depth;
    else if (src[i] == ']' && --depth == 0) return src.substr(pos, i - pos + 1);
  }
  return {};
}

// ---------------------------------------------------------------------------

ShardData load_sqdds_shard(const std::string& file_path) {
  ShardData sd;
  sd.path = file_path;

  std::ifstream f(file_path, std::ios::binary);
  if (!f) return sd;

  const std::vector<uint8_t> raw(
      (std::istreambuf_iterator<char>(f)),
      std::istreambuf_iterator<char>());

  // Minimum: 20 (header) + 40 (one tensor record) = 60 bytes
  if (raw.size() < 60) return sd;

  // ── Header ────────────────────────────────────────────────────────────────
  SCX_DDS_Header hdr = parse_dds_header(raw.data(), raw.size());

  static constexpr uint32_t SCXQDDS_MAGIC = 0x53444453u;  // 'SDDS' LE
  if (hdr.magic != SCXQDDS_MAGIC) return sd;
  if (hdr.tensor_count == 0)      return sd;

  // ── Tensor record ─────────────────────────────────────────────────────────
  size_t off = 4 + 2 + 2 + 4 + 8;  // sizeof header fields

  auto rd16 = [&](uint16_t& v) {
    v = static_cast<uint16_t>(raw[off] | (raw[off + 1] << 8));
    off += 2;
  };
  auto rd32 = [&](uint32_t& v) {
    v = raw[off] | (raw[off+1]<<8) | (raw[off+2]<<16) | (raw[off+3]<<24);
    off += 4;
  };
  auto rd64 = [&](uint64_t& v) {
    v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(raw[off + i]) << (8 * i);
    off += 8;
  };

  SCX_Tensor t{};
  rd32(t.id);
  rd16(t.dtype);
  rd16(t.dims);
  rd32(t.shape[0]); rd32(t.shape[1]); rd32(t.shape[2]); rd32(t.shape[3]);
  rd64(t.offset);
  rd64(t.size);

  // Only accept INT8 (bits == 8)
  if (t.dtype != 8) return sd;
  if (t.size == 0)  return sd;
  if (t.offset + t.size > raw.size()) return sd;

  sd.tokens.assign(raw.begin() + static_cast<ptrdiff_t>(t.offset),
                   raw.begin() + static_cast<ptrdiff_t>(t.offset + t.size));
  sd.token_count = sd.tokens.size();
  sd.ok          = true;
  return sd;
}

// ---------------------------------------------------------------------------
// load_shard_registry
//
// Reads artifacts/training/shard-index.json (written by shard-artifacts.js).
// Resolves each shard path relative to shard_root_dir so callers get absolute
// paths ready to pass to load_sqdds_shard().
// ---------------------------------------------------------------------------
ShardRegistry load_shard_registry(const std::string& shard_root_dir) {
  ShardRegistry reg;
  reg.shard_root = shard_root_dir;

  const std::string index_path = shard_root_dir + "/shard-index.json";
  std::ifstream f(index_path);
  if (!f) return reg;

  const std::string src((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

  // Verify schema
  if (src.find("xcfe-shard-index-1") == std::string::npos) return reg;

  const std::string arr = json_array(src, "shards");
  for (const std::string& obj : split_json_objects(arr)) {
    ShardEntry e;
    e.id          = json_string(obj, "id");
    const std::string rel = json_string(obj, "path");
    e.abs_path    = shard_root_dir + "/" + rel;
    e.token_bytes = json_uint64(obj, "token_bytes");
    e.sources     = static_cast<uint32_t>(json_uint64(obj, "sources"));
    if (!e.id.empty() && !rel.empty()) reg.shards.push_back(std::move(e));
  }

  reg.ok = !reg.shards.empty();
  return reg;
}

std::vector<ShardData> ShardRegistry::load_all() const {
  std::vector<ShardData> out;
  for (const auto& entry : shards) {
    ShardData sd = load_sqdds_shard(entry.abs_path);
    if (sd.ok) out.push_back(std::move(sd));
  }
  return out;
}
