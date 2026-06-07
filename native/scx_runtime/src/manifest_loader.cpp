#include "manifest_loader.h"
#include "scxgraph.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cctype>
#include <filesystem>

#ifdef SCX_HAVE_ZSTD
#include <zstd.h>
#endif

static std::string read_file(const std::string& path){
  std::ifstream f(path, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static std::string get_attr(const std::string& line, const std::string& key){
  const std::string pat = key + "=\"";
  auto p = line.find(pat);
  if(p==std::string::npos) return "";
  p += pat.size();
  auto q = line.find('"', p);
  if(q==std::string::npos) return "";
  return line.substr(p, q-p);
}

static std::vector<uint32_t> parse_shape(const std::string& s){
  std::vector<uint32_t> out;
  if(s.empty()) return out;
  std::stringstream ss(s);
  std::string tok;
  while(std::getline(ss,tok,'x')){
    out.push_back(static_cast<uint32_t>(std::stoul(tok)));
  }
  return out;
}

static inline bool is_base64_char(char c){
  return std::isalnum(static_cast<unsigned char>(c)) || c=='+' || c=='/' || c=='=';
}

static std::vector<uint8_t> decode_base64(const std::string& b64){
  static const std::string alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  auto val = [&](char c)->int{
    if(c=='=') return 0;
    auto p = alphabet.find(c);
    return p==std::string::npos ? -1 : static_cast<int>(p);
  };
  std::vector<uint8_t> out;
  out.reserve((b64.size()*3)/4);
  for(size_t i=0; i+3 < b64.size(); i+=4){
    int v0 = val(b64[i]);
    int v1 = val(b64[i+1]);
    int v2 = val(b64[i+2]);
    int v3 = val(b64[i+3]);
    if(v0<0 || v1<0 || v2<0 || v3<0) break;
    uint32_t triple = (v0<<18)|(v1<<12)|(v2<<6)|v3;
    out.push_back((triple>>16)&0xFF);
    if(b64[i+2] != '=') out.push_back((triple>>8)&0xFF);
    if(b64[i+3] != '=') out.push_back(triple&0xFF);
  }
  return out;
}

#ifdef SCX_HAVE_ZSTD
static std::vector<uint8_t> maybe_zstd(const std::vector<uint8_t>& in, bool compressed){
  if(!compressed) return in;
  unsigned long long sz = ZSTD_getFrameContentSize(in.data(), in.size());
  if(sz == ZSTD_CONTENTSIZE_ERROR || sz == ZSTD_CONTENTSIZE_UNKNOWN) return {};
  std::vector<uint8_t> out(sz);
  size_t rc = ZSTD_decompress(out.data(), out.size(), in.data(), in.size());
  if(ZSTD_isError(rc)) return {};
  out.resize(rc);
  return out;
}
#else
static std::vector<uint8_t> maybe_zstd(const std::vector<uint8_t>& in, bool){ return in; }
#endif

bool load_manifest(const std::string& path, ManifestInfo& out){
  const std::string content = read_file(path);
  if(content.find("\"schema\"") == std::string::npos) return false;
  if(content.find("xcfe-model-1") == std::string::npos) return false;

  // extract topology string value
  std::string topo;
  {
    const std::string key = "\"topology\"";
    auto p = content.find(key);
    if(p != std::string::npos){
      p = content.find('"', p + key.size());
      if(p != std::string::npos){
        auto q = content.find('"', p+1);
        if(q != std::string::npos){
          topo = content.substr(p+1, q-p-1);
        }
      }
    }
  }
  if(topo.empty()) return false;

  // scan topology for nodes/edges/coord-frame
  std::stringstream ss(topo);
  std::string line;
  while(std::getline(ss,line,'>')){
    if(line.find("<brain-node") != std::string::npos){
      out.nodes++;
    } else if(line.find("<brain-edge") != std::string::npos || line.find("<geodesic-entropy-arc") != std::string::npos){
      out.edges++;
    } else if(line.find("brain-graph") != std::string::npos){
      out.coord_frame = get_attr(line,"coord-frame");
    }
  }

  // tensors: count entries and decode inline base64 lengths
  {
    const std::string tkey = "\"tensors\"";
    auto p = content.find(tkey);
    if(p != std::string::npos){
      // naive scan for "data": "base64:..."
      auto pos = p;
      while(true){
        auto d = content.find("\"data\"", pos);
        if(d == std::string::npos) break;
        auto colon = content.find(':', d);
        auto quote = content.find('"', colon);
        auto endq = content.find('"', quote+1);
        if(quote != std::string::npos && endq != std::string::npos){
          std::string v = content.substr(quote+1, endq-quote-1);
          const std::string p1 = "base64:";
          const std::string p2 = "base64+zstd:";
          bool zstd = false;
          if(v.rfind(p2,0)==0){ v = v.substr(p2.size()); zstd = true; }
          else if(v.rfind(p1,0)==0){ v = v.substr(p1.size()); zstd = false; }
          else { v.clear(); }
          if(!v.empty()){
            bool ok=true;
            for(char c: v){ if(!is_base64_char(c)){ ok=false; break; } }
            if(ok){
              auto raw = decode_base64(v);
              auto dec = maybe_zstd(raw, zstd);
              ManifestTensor t{};
              t.data = std::move(dec);
              out.tensors.push_back(std::move(t));
            }
          }
        }
        pos = endq == std::string::npos ? d+6 : endq;
      }

      // scan for external source paths
      pos = p;
      while(true){
        auto s = content.find("\"source\"", pos);
        if(s == std::string::npos) break;
        auto colon = content.find(':', s);
        auto q1 = content.find('"', colon);
        auto q2 = content.find('"', q1+1);
        if(q1!=std::string::npos && q2!=std::string::npos){
          std::string src = content.substr(q1+1, q2-q1-1);
          ManifestTensor t{};
          t.source = src;
          // try reading file relative to manifest dir
          std::filesystem::path base(path);
          auto full = base.parent_path() / src;
          if(std::filesystem::exists(full)){
            std::ifstream rf(full, std::ios::binary);
            t.data.assign(std::istreambuf_iterator<char>(rf), {});
          }
          out.tensors.push_back(std::move(t));
        }
        pos = q2 == std::string::npos ? s+8 : q2;
      }
    }
  }

  // kbc1 (optional) as base64 or base64+zstd
  {
    const std::string key = "\"kbc1\"";
    auto p = content.find(key);
    if(p != std::string::npos){
      auto colon = content.find(':', p);
      auto q1 = content.find('"', colon);
      auto q2 = content.find('"', q1+1);
      if(q1!=std::string::npos && q2!=std::string::npos){
        std::string v = content.substr(q1+1, q2-q1-1);
        const std::string p1 = "base64:";
        const std::string p2 = "base64+zstd:";
        bool zstd = false;
        if(v.rfind(p2,0)==0){ v = v.substr(p2.size()); zstd = true; }
        else if(v.rfind(p1,0)==0){ v = v.substr(p1.size()); zstd = false; }
        else { v.clear(); }
        if(!v.empty()){
          bool ok=true;
          for(char c: v){ if(!is_base64_char(c)){ ok=false; break; } }
          if(ok){
            auto raw = decode_base64(v);
            out.kbc1_bytes = maybe_zstd(raw, zstd);
          }
        }
      }
    }
  }

  if(out.coord_frame.empty()) out.coord_frame = "triangle";

  // Parse scxGraph JSON field — structured graph for XVM compilation
  out.scx_graph = scx::load_scxgraph(content);

  out.ok = true;
  return true;
}
