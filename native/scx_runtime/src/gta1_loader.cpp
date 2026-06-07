#include "gta1_loader.h"
#include <fstream>
#include <cstdint>
#include <vector>
#include <cstring>
#include <string>
#include <sstream>

#ifdef SCX_HAVE_ZSTD
#include <zstd.h>
#endif

namespace {
struct Header {
  char magic[4];
  uint16_t version;
  uint16_t flags;
  uint64_t reserved;
  uint64_t index_offset;
  uint64_t index_count;
  uint8_t uuid[16];
  uint64_t checksum;
};

struct IndexEntry {
  uint32_t kind;
  uint32_t reserved;
  uint64_t offset;
  uint64_t length;
  uint64_t checksum;
};

template<typename T>
bool read(std::ifstream& f, T& out) {
  f.read(reinterpret_cast<char*>(&out), sizeof(T));
  return f.good();
}

std::string decompress_if_needed(const std::vector<uint8_t>& input, bool compressed){
#ifdef SCX_HAVE_ZSTD
  if(compressed){
    unsigned long long size = ZSTD_getFrameContentSize(input.data(), input.size());
    if(size == ZSTD_CONTENTSIZE_ERROR || size == ZSTD_CONTENTSIZE_UNKNOWN) return {};
    std::string out;
    out.resize(static_cast<size_t>(size));
    const size_t rc = ZSTD_decompress(out.data(), out.size(), input.data(), input.size());
    if(ZSTD_isError(rc)) return {};
    out.resize(rc);
    return out;
  }
#else
  if(compressed) return {};
#endif
  return std::string(reinterpret_cast<const char*>(input.data()), input.size());
}

std::string get_attr(const std::string& line, const std::string& key){
  const std::string pat = key + "=\"";
  auto p = line.find(pat);
  if(p==std::string::npos) return "";
  p += pat.size();
  auto q = line.find('"', p);
  if(q==std::string::npos) return "";
  return line.substr(p, q-p);
}

std::vector<uint32_t> parse_shape(const std::string& s){
  std::vector<uint32_t> out;
  if(s.empty()) return out;
  std::stringstream ss(s);
  std::string tok;
  while(std::getline(ss,tok,'x')){
    out.push_back(static_cast<uint32_t>(std::stoul(tok)));
  }
  return out;
}
} // namespace

bool load_gta1(const std::string& path, Gta1Info& out) {
  std::ifstream f(path, std::ios::binary);
  if(!f) return false;

  Header h{};
  if(!read(f,h)) return false;
  if(std::strncmp(h.magic,"GTA1",4)!=0) return false;
  if(h.version != 1) return false;

  f.seekg(static_cast<std::streamoff>(h.index_offset), std::ios::beg);
  if(!f.good()) return false;

  std::vector<IndexEntry> entries(h.index_count);
  for(size_t i=0;i<h.index_count;i++){
    if(!read(f, entries[i])) return false;
  }

  // helper to load a block
  auto loadBlock = [&](const IndexEntry& e)->std::vector<uint8_t>{
    std::vector<uint8_t> buf(static_cast<size_t>(e.length));
    std::streampos pos = static_cast<std::streamoff>(e.offset);
    f.seekg(pos, std::ios::beg);
    if(!f.good()) return {};
    f.read(reinterpret_cast<char*>(buf.data()), buf.size());
    if(!f.good()) return {};
    return buf;
  };

  const bool fields_zstd = (h.flags & 0x1) != 0;
  const bool topo_zstd   = (h.flags & 0x2) != 0;

  for(const auto& e: entries){
    if(e.kind==2){ // topology
      auto buf = loadBlock(e);
      if(buf.empty()) return false;
      auto topo = decompress_if_needed(buf, topo_zstd);
      if(topo.empty()) return false;
      out.topology_xml = std::move(topo);
      out.ok = true;
    } else if(e.kind==1){ // fields
      auto buf = loadBlock(e);
      if(buf.empty()) return false;
      auto fields = decompress_if_needed(buf, fields_zstd);
      if(fields.empty()) return false;
      const uint8_t* p = reinterpret_cast<const uint8_t*>(fields.data());
      const uint8_t* end = p + fields.size();
      if(end - p < 4) return false;
      uint32_t count = *reinterpret_cast<const uint32_t*>(p); p += 4;
      for(uint32_t i=0;i<count;i++){
        if(end - p < 4+2+2) break;
        GtaTensor t{};
        t.id = *reinterpret_cast<const uint32_t*>(p); p+=4;
        t.dtype = *reinterpret_cast<const uint16_t*>(p); p+=2;
        uint16_t dims = *reinterpret_cast<const uint16_t*>(p); p+=2;
        if(end - p < 4*dims + 8 + 8 + 2 + 2 + 4) break;
        t.shape.resize(dims);
        for(uint16_t d=0; d<dims; d++){
          t.shape[d] = *reinterpret_cast<const uint32_t*>(p); p+=4;
        }
        uint64_t data_offset = *reinterpret_cast<const uint64_t*>(p); p+=8;
        uint64_t data_len = *reinterpret_cast<const uint64_t*>(p); p+=8;
        t.q_scheme = *reinterpret_cast<const uint16_t*>(p); p+=2;
        p+=2; // reserved
        t.scale = *reinterpret_cast<const float*>(p); p+=4;
        (void)data_offset; (void)data_len;
        out.tensors.push_back(std::move(t));
      }
    }
  }

  // Parse topology XML for nodes/edges/coord-frame
  if(out.ok){
    std::stringstream ss(out.topology_xml);
    std::string line;
    while(std::getline(ss,line)){
      if(line.find("<brain-node") != std::string::npos){
        GtaNode n{};
        n.id = get_attr(line,"id");
        n.role = get_attr(line,"role");
        n.q_scheme = get_attr(line,"q_scheme");
        n.device = get_attr(line,"device");
        n.shape = parse_shape(get_attr(line,"shape"));
        out.nodes.push_back(std::move(n));
      } else if(line.find("<brain-edge") != std::string::npos || line.find("<geodesic-entropy-arc") != std::string::npos){
        GtaEdge e{};
        e.type = (line.find("geodesic-entropy-arc")!=std::string::npos) ? "geodesic-entropy-arc" : "brain-edge";
        e.id = get_attr(line,"id");
        e.from = get_attr(line,"from");
        e.to = get_attr(line,"to");
        const auto m = get_attr(line,"metric"); if(!m.empty()) e.metric = std::stod(m);
        const auto ent = get_attr(line,"entropy"); if(!ent.empty()) e.entropy = std::stod(ent);
        const auto ph = get_attr(line,"phase"); if(!ph.empty()) e.phase = std::stod(ph);
        out.edges.push_back(std::move(e));
      } else if(line.find("brain-graph") != std::string::npos){
        out.coord_frame = get_attr(line,"coord-frame");
      }
    }
  }

  return out.ok;
}
