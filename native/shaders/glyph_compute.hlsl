// ============================================================
// KUHUL CORE SHADER — GLYPH / GRAM EXECUTION (v1)
// ============================================================
// Shader 1: Symbolic Execution Engine
// Replaces/generalizes HTML shader
// Executes SCXQ2 INT4 lanes (true runtime core)
//
// Compile: dxc -T cs_6_0 -E CS_GlyphExec -O3 glyph_compute.hlsl -Fo glyph_compute.cso
// ============================================================

cbuffer DispatchParams : register(b0)
{
    uint laneCount;        // Number of parallel lanes
    uint tokensPerLane;    // Tokens per lane
    uint mode;             // 0=GRAM, 1=TENSOR
    uint param1;           // Reserved
};

// Buffers
StructuredBuffer<uint>   gramBuffer   : register(t0);   // INT4 packed SCXQ2 instructions
RWStructuredBuffer<float> stateBuffer : register(u0);   // Lane state (float32)
RWStructuredBuffer<uint> outBuffer    : register(u1);   // Route/Call signals
RWStructuredBuffer<uint> debugBuffer  : register(u2);   // Debug trace (optional)

// ============================================================================
// INT4 DECODE
// ============================================================================
uint DecodeToken(uint word, uint idx)
{
    return (word >> (idx * 4)) & 0xF;
}

// ============================================================================
// ISA v1 (INT4 OPCODES)
// ============================================================================
#define OP_NOP   0x0
#define OP_LOAD  0x1
#define OP_STORE 0x2
#define OP_ADD   0x3
#define OP_MUL   0x4
#define OP_DOT   0x5
#define OP_NORM  0x6
#define OP_EXP   0x8
#define OP_SUM   0x9
#define OP_MAX   0xA
#define OP_MIN   0xB
#define OP_MOV   0xC
#define OP_ESC   0xF  // Signal XCFE routing

// ============================================================================
// GLYPH EXECUTION KERNEL
// ============================================================================
[numthreads(64, 1, 1)]
void CS_GlyphExec(uint3 DTid : SV_DispatchThreadID)
{
    // Thread-local registers (must be inside function)
    float r0 = 0;
    float r1 = 0;
    float acc = 0;
    uint pc = 0;  // Program counter
    
    uint lane = DTid.x;
    if (lane >= laneCount) return;

    // Base offset in gramBuffer (INT4 packed: 8 tokens per uint32)
    uint base = lane * ((tokensPerLane + 7) / 8);
    
    // Initialize lane state
    r0 = stateBuffer[lane * 4 + 0];
    r1 = stateBuffer[lane * 4 + 1];
    acc = stateBuffer[lane * 4 + 2];
    pc = 0;

    uint out_flag = 0;
    uint debug_idx = 0;

    // Execute INT4 token stream
    for (uint i = 0; i < tokensPerLane; i++)
    {
        uint word = gramBuffer[base + (i >> 3)];
        uint token = DecodeToken(word, i & 7);

        // ISA v1 dispatch
        switch(token)
        {
            case OP_NOP: // 0x0 — No operation
                break;

            case OP_LOAD: // 0x1 — Load from state
                r0 = stateBuffer[lane * 4 + 0];
                break;

            case OP_STORE: // 0x2 — Store to state
                stateBuffer[lane * 4 + 0] = r0;
                stateBuffer[lane * 4 + 1] = r1;
                stateBuffer[lane * 4 + 2] = acc;
                break;

            case OP_ADD: // 0x3 — Add registers
                r0 = r0 + r1;
                break;

            case OP_MUL: // 0x4 — Multiply registers
                r0 = r0 * r1;
                break;

            case OP_DOT: // 0x5 — Dot product accumulator
                acc = r0 * r1;
                break;

            case OP_NORM: // 0x6 — RMS normalize
                {
                    float mag = sqrt(r0 * r0 + r1 * r1 + 1e-6);
                    r0 = r0 / mag;
                    r1 = r1 / mag;
                }
                break;

            case OP_EXP: // 0x8 — Exponential
                r0 = exp(r0);
                break;

            case OP_SUM: // 0x9 — Accumulate sum
                acc += r0;
                break;

            case OP_MAX: // 0xA — Maximum
                r0 = max(r0, r1);
                break;

            case OP_MIN: // 0xB — Minimum
                r0 = min(r0, r1);
                break;

            case OP_MOV: // 0xC — Move register
                r1 = r0;
                break;

            case OP_ESC: // 0xF — Escape to XCFE routing
                out_flag = 1;
                outBuffer[lane] = lane;  // Signal lane index for routing
                break;

            default: // Undefined opcode — NOP
                break;
        }

        // Debug trace (optional, can be disabled)
        if (debug_idx < 64)
        {
            debugBuffer[lane * 64 + debug_idx] = token;
            debug_idx++;
        }

        pc++;
    }

    // Final state store
    stateBuffer[lane * 4 + 0] = r0;
    stateBuffer[lane * 4 + 1] = r1;
    stateBuffer[lane * 4 + 2] = acc;
    stateBuffer[lane * 4 + 3] = pc;

    // Final output signal
    if (out_flag)
    {
        outBuffer[lane] = lane | 0x80000000;  // High bit = escape signal
    }
}

// ============================================================================
// XCFE MODE DISPATCH
// ============================================================================
// Sek(field):
//   if mode == GRAM (0):
//       → CS_GlyphExec
//   if mode == TENSOR (1):
//       → CS_AttnPass1 → reduce → CS_AttnPass2
// ============================================================================
