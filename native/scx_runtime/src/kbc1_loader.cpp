#include "kbc1_loader.h"
#include <fstream>

bool load_kbc1(const std::string& path, KBC1_Program& out) {
  std::ifstream f(path, std::ios::binary);
  if(!f) return false;
  out.inst.clear();
  while(true){
    uint16_t op, argc;
    uint32_t args[4];
    f.read(reinterpret_cast<char*>(&op), sizeof(op));
    if(!f.good()) break;
    f.read(reinterpret_cast<char*>(&argc), sizeof(argc));
    if(!f.good()) break;
    f.read(reinterpret_cast<char*>(args), sizeof(args));
    if(!f.good()) break;
    KBC1_Inst i{op, argc, {args[0], args[1], args[2], args[3]}};
    out.inst.push_back(i);
  }
  return !out.inst.empty();
}
