#include "kbc1.h"
#include "manifest_loader.h"

KBC1_Program compile_minimal_16layer_moe() {
  KBC1_Program p;
  auto push = [&](uint16_t op) {
    KBC1_Inst i{op, 0, {0, 0, 0, 0}};
    p.inst.push_back(i);
  };

  push(OP_INPUT);
  for (int l = 0; l < 16; ++l) {
    push(OP_LAYERNORM);
    push(OP_ATTN_QKV);
    push(OP_ATTN_SOFTMAX);
    push(OP_ATTN_OUT);
    push(OP_MOE_ROUTE);
    push(OP_MOE_DISPATCH);
    push(OP_MOE_COMBINE);
  }
  push(OP_OUTPUT);
  return p;
}

// Compile from a manifest-derived graph: simple mapping of nodes/edges to ops.
KBC1_Program compile_from_manifest(const ManifestInfo& m) {
  if(!m.ok) return compile_minimal_16layer_moe();
  KBC1_Program p;
  auto push = [&](uint16_t op, uint32_t a0=0, uint32_t a1=0, uint32_t a2=0, uint32_t a3=0) {
    KBC1_Inst i{op, 0, {a0,a1,a2,a3}};
    p.inst.push_back(i);
  };

  push(OP_INPUT);

  // Treat each edge as a routing + dispatch + combine step.
  for(size_t i=0; i<m.edges; ++i){
    push(OP_MOE_ROUTE);
    push(OP_MOE_DISPATCH);
    push(OP_MOE_COMBINE);
  }

  push(OP_OUTPUT);
  return p;
}
