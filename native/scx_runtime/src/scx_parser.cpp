#include "scx_parser.h"
#include <cctype>

SCXProgram parse_scx(const std::string& src) {
  SCXProgram prog;
  for (size_t i = 0; i < src.size();) {
    if (src[i] == '⟁') {
      ++i;
      int id = 0;
      while (i < src.size() && std::isdigit(static_cast<unsigned char>(src[i]))) {
        id = id * 10 + (src[i++] - '0');
      }
      SCXInstruction inst;
      inst.op = SCXOp::ROUTE;
      inst.args.push_back({static_cast<uint32_t>(id), 0.0f, {}});
      prog.instructions.push_back(inst);
    } else {
      ++i;
    }
  }
  return prog;
}
