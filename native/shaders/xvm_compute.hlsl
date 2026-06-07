struct Fiber {
  uint pc;
  uint sp;
  uint phase;
  uint flags;
  uint r0;
  uint r1;
  uint r2;
  uint r3;
};

StructuredBuffer<uint> Code : register(t0);
RWStructuredBuffer<Fiber> Fibers : register(u0);
RWStructuredBuffer<uint> Shared : register(u1);
RWStructuredBuffer<uint> Stack : register(u2);
RWStructuredBuffer<uint> Trace : register(u3);
RWStructuredBuffer<uint> TraceIndex : register(u4);

uint read_reg(Fiber f, uint idx) {
  if (idx == 0) return f.r0;
  if (idx == 1) return f.r1;
  if (idx == 2) return f.r2;
  return f.r3;
}

void write_reg(inout Fiber f, uint idx, uint value) {
  if (idx == 0) f.r0 = value;
  if (idx == 1) f.r1 = value;
  if (idx == 2) f.r2 = value;
  if (idx == 3) f.r3 = value;
}

void write_trace(uint fid, uint op, Fiber f) {
  uint base;
  InterlockedAdd(TraceIndex[0], 7, base);
  Trace[base + 0] = fid;
  Trace[base + 1] = op;
  Trace[base + 2] = f.r0;
  Trace[base + 3] = f.r1;
  Trace[base + 4] = f.r2;
  Trace[base + 5] = f.r3;
  Trace[base + 6] = f.pc;
}

[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
  uint fid = id.x;
  Fiber f = Fibers[fid];

  if (f.flags == 0) {
    return;
  }

  uint instr = Code[f.pc++];
  uint op = instr & 0x3F;

  switch (op) {
    case 0x01: { // LOAD_CONST reg, imm32
      uint reg = Code[f.pc++];
      uint imm = Code[f.pc++];
      write_reg(f, reg, imm);
      break;
    }
    case 0x02: { // MOV dst, src
      uint dst = Code[f.pc++];
      uint src = Code[f.pc++];
      write_reg(f, dst, read_reg(f, src));
      break;
    }
    case 0x03: { // ADD dst, src
      uint dst = Code[f.pc++];
      uint src = Code[f.pc++];
      write_reg(f, dst, read_reg(f, dst) + read_reg(f, src));
      break;
    }
    case 0x04: { // SUB dst, src
      uint dst = Code[f.pc++];
      uint src = Code[f.pc++];
      write_reg(f, dst, read_reg(f, dst) - read_reg(f, src));
      break;
    }
    case 0x05: { // MUL dst, src
      uint dst = Code[f.pc++];
      uint src = Code[f.pc++];
      write_reg(f, dst, read_reg(f, dst) * read_reg(f, src));
      break;
    }
    case 0x06: { // DIV dst, src
      uint dst = Code[f.pc++];
      uint src = Code[f.pc++];
      uint div = read_reg(f, src);
      write_reg(f, dst, div == 0 ? 0 : (read_reg(f, dst) / div));
      break;
    }
    case 0x10: {
      uint idx = Code[f.pc++];
      uint val = Code[f.pc++];
      InterlockedAdd(Shared[idx], val);
      break;
    }
    case 0x20: { // JMP target
      uint target = Code[f.pc++];
      f.pc = target;
      break;
    }
    case 0x21: { // JMP_IF r0 != 0, target
      uint target = Code[f.pc++];
      if (f.r0 != 0) {
        f.pc = target;
      }
      break;
    }
    case 0x22: { // CMP_EQ a, b -> r0
      uint a = Code[f.pc++];
      uint b = Code[f.pc++];
      f.r0 = (read_reg(f, a) == read_reg(f, b)) ? 1 : 0;
      break;
    }
    case 0x30: { // LOAD_SHARED reg, idx
      uint reg = Code[f.pc++];
      uint idx = Code[f.pc++];
      write_reg(f, reg, Shared[idx]);
      break;
    }
    case 0x31: { // STORE_SHARED idx, reg
      uint idx = Code[f.pc++];
      uint reg = Code[f.pc++];
      Shared[idx] = read_reg(f, reg);
      break;
    }
    case 0x3f: { // RETURN
      f.flags = 0;
      break;
    }
    default: {
      f.flags = 0;
      break;
    }
  }

  write_trace(fid, op, f);
  Fibers[fid] = f;
}
