/**
 * @file kuhul_fold_compute.hlsl
 * @brief COMPUTE_FOLD kernel — MM-1 token matmul, MoE top-1 routing, KV cache, token emission
 *
 * Fold:      ⟁COMPUTE_FOLD⟁
 * Micronaut: MM-1 (ModelMicronaut)
 * Lane:      BATCH
 * Nodes:     300 (z-layers 0–4 of 10×10×10 grid)
 * Dispatch:  300 thread groups × 64 threads = 19200 threads total
 *
 * Pipeline per token position:
 *   1. Read INT8 trunk weights → dequantize → matmul (trunk_matmul)
 *   2. Router logits (1024 → 9 experts) → top-1 argmax
 *   3. Expert matmul (selected expert weights only)
 *   4. Emit token signal → BATCH lane output buffer
 *   5. Write router_logits_i8 → META_FOLD via arc_CF_MF
 *   6. Write trunk output → STORAGE_FOLD via arc_CF_SF
 *
 * CM-1 gate: U+0002 (@Wo) must be set in ControlFlags before dispatch.
 * Verifier:  V6 (replay determinism) — same weights + same input → same output bytes.
 */

// ============================================================================
// Resource Bindings
// ============================================================================

StructuredBuffer<int>    TrunkWeights    : register(t0);  // INT8 trunk weights (packed as int)
StructuredBuffer<int>    RouterWeights   : register(t1);  // INT8 router weights 1024→9 (packed)
StructuredBuffer<int>    ExpertWeights   : register(t2);  // INT8 expert weights (all 9, strided)
StructuredBuffer<float>  InputTokens     : register(t3);  // Float32 input embeddings (seq × 1024)
StructuredBuffer<float>  KVCache         : register(t4);  // KV cache (2048 × 12L × 2 × 1024)
StructuredBuffer<uint>   ControlFlags    : register(t5);  // CM-1 gate flags (U+0002 = permit)
StructuredBuffer<float>  TrunkScales     : register(t6);  // Per-row INT8 dequant scales (trunk)
StructuredBuffer<float>  RouterScales    : register(t7);  // Per-row INT8 dequant scales (router)
StructuredBuffer<float>  ExpertScales    : register(t8);  // Per-row INT8 dequant scales (experts)

RWStructuredBuffer<float> TokenOutput    : register(u0);  // Emitted token logits (vocab_size=32768)
RWStructuredBuffer<float> TrunkOutput    : register(u1);  // Trunk activations → STORAGE_FOLD
RWStructuredBuffer<int>   RouterLogitsI8 : register(u2);  // Router logits INT8 → META_FOLD
RWStructuredBuffer<uint>  ExpertSelected : register(u3);  // Top-1 expert index per token

cbuffer ComputeFoldParams : register(b0)
{
    uint  hidden_dim;       // 1024
    uint  num_layers;       // 12 (trunk)
    uint  num_experts;      // 9
    uint  expert_hidden;    // 1024
    uint  expert_layers;    // 4
    uint  seq_len;          // current sequence length
    uint  vocab_size;       // 32768
    uint  node_id;          // this node's fold-local index (0–299)
    uint  cm1_gate;         // must equal 0x0002 to proceed
    uint  _pad0;
    uint  _pad1;
    uint  _pad2;
};

// ============================================================================
// Helpers
// ============================================================================

/** Dequantize INT8 weight: scale is per-row absmax INT8 */
float DequantI8(int w_i8, float scale)
{
    return float(w_i8) * scale;
}

/** Fast sigmoid for SwiGLU gate */
float Sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

/** SwiGLU activation: SiLU(gate) * value */
float SwiGLU(float gate, float value)
{
    return (gate * Sigmoid(gate)) * value;
}

/** RMS norm over hidden_dim elements (reads from shared memory) */
groupshared float gs_norm_scratch[1024];

float RMSNorm(uint tid, float x, uint dim)
{
    gs_norm_scratch[tid] = x * x;
    GroupMemoryBarrierWithGroupSync();

    // Parallel reduction for sum of squares
    for (uint s = dim / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            gs_norm_scratch[tid] += gs_norm_scratch[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }

    float rms = sqrt(gs_norm_scratch[0] / float(dim) + 1e-6f);
    return x / rms;
}

// ============================================================================
// Trunk MatMul  (INT8 weights, float32 activations)
// Each thread handles one output row.
// ============================================================================

groupshared float gs_trunk_act[1024];   // Input activation (shared across threads)
groupshared float gs_router_logits[9];  // Router output (small, 9 experts)

[numthreads(64, 1, 1)]
void CS_ComputeFold(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    // CM-1 gate check — U+0002 (@Wo) must be set
    if (ControlFlags[0] != cm1_gate)
        return;

    // Each group handles one token position
    uint token_pos = gid.x;
    if (token_pos >= seq_len)
        return;

    uint input_base  = token_pos * hidden_dim;
    uint output_base = token_pos * vocab_size;

    // -----------------------------------------------------------------------
    // Stage 1: Load input token embedding into shared memory
    // -----------------------------------------------------------------------
    for (uint d = tid; d < hidden_dim; d += 64)
        gs_trunk_act[d] = InputTokens[input_base + d];
    GroupMemoryBarrierWithGroupSync();

    // -----------------------------------------------------------------------
    // Stage 2: Trunk matmul (12 layers × 1024→1024, simplified to 1 layer per
    //          node; full layer stack is across all 200 trunk nodes)
    //          Each node owns rows [node_id*5 .. node_id*5+4] of weight matrix.
    // -----------------------------------------------------------------------
    float trunk_out[4];
    uint row_base = (node_id % 200) * (hidden_dim / 200);  // rows owned by this node

    for (uint r = 0; r < 4 && r < hidden_dim; r++)
    {
        uint row = row_base + r;
        float acc = 0.0f;
        float scale = TrunkScales[row];

        for (uint c = 0; c < hidden_dim; c++)
        {
            int w_packed = TrunkWeights[row * hidden_dim + c];
            int w_i8     = (w_packed >> ((c & 3) * 8)) & 0xFF;
            // sign-extend INT8
            if (w_i8 > 127) w_i8 -= 256;
            acc += gs_trunk_act[c] * DequantI8(w_i8, scale);
        }
        trunk_out[r] = acc;

        // Write trunk activation to STORAGE_FOLD output buffer
        TrunkOutput[token_pos * hidden_dim + row] = acc;
    }
    GroupMemoryBarrierWithGroupSync();

    // -----------------------------------------------------------------------
    // Stage 3: Router logits (1024 → 9) — only first thread group does this
    // -----------------------------------------------------------------------
    if (node_id >= 200)  // router nodes occupy node_ids 200–299
    {
        uint router_node = node_id - 200;  // 0–99

        // Each router node computes logits for 1 expert (9 experts distributed)
        uint expert_idx = router_node % num_experts;
        float logit = 0.0f;
        float r_scale = RouterScales[expert_idx];

        for (uint c = tid; c < hidden_dim; c += 64)
        {
            int w_packed = RouterWeights[expert_idx * hidden_dim + c];
            int w_i8     = w_packed & 0xFF;
            if (w_i8 > 127) w_i8 -= 256;
            logit += gs_trunk_act[c] * DequantI8(w_i8, r_scale);
        }

        // Reduce across 64 threads into gs_router_logits[expert_idx]
        // (simplified: thread 0 accumulates; real impl uses parallel reduce)
        if (tid == 0)
        {
            gs_router_logits[expert_idx] = logit;

            // Write INT8 quantized router logit → META_FOLD
            float abs_scale = max(abs(logit) / 127.0f, 1e-8f);
            int logit_i8 = clamp(int(logit / abs_scale), -127, 127);
            RouterLogitsI8[token_pos * num_experts + expert_idx] = logit_i8;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // -----------------------------------------------------------------------
    // Stage 4: Top-1 argmax expert selection (node 200 does this)
    // -----------------------------------------------------------------------
    if (node_id == 200 && tid == 0)
    {
        uint best_expert = 0;
        float best_logit = gs_router_logits[0];
        for (uint e = 1; e < num_experts; e++)
        {
            if (gs_router_logits[e] > best_logit)
            {
                best_logit  = gs_router_logits[e];
                best_expert = e;
            }
        }
        ExpertSelected[token_pos] = best_expert;
    }
    GroupMemoryBarrierWithGroupSync();

    // -----------------------------------------------------------------------
    // Stage 5: Expert matmul (selected expert only, 4 layers × 1024→1024)
    //          Expert nodes: node_ids 100–199 (one node per expert × 11 spare)
    //          Only nodes whose expert_idx == ExpertSelected[token_pos] run.
    // -----------------------------------------------------------------------
    if (node_id >= 100 && node_id < 200)
    {
        uint my_expert = (node_id - 100) / (100 / num_experts);  // 0–8
        if (my_expert != ExpertSelected[token_pos])
            return;

        float e_scale = ExpertScales[my_expert];
        uint expert_weight_base = my_expert * expert_layers * hidden_dim * hidden_dim;

        float expert_out = 0.0f;
        uint out_row = tid;

        for (uint c = 0; c < hidden_dim; c++)
        {
            int w_packed = ExpertWeights[expert_weight_base + out_row * hidden_dim + c];
            int w_i8     = w_packed & 0xFF;
            if (w_i8 > 127) w_i8 -= 256;
            expert_out += gs_trunk_act[c] * DequantI8(w_i8, e_scale);
        }

        // Apply SwiGLU (gate from second half of expert weight row)
        float gate = expert_out;
        float value = expert_out;  // simplified; real: two projections
        float activated = SwiGLU(gate, value);

        // Write to token output (logits projection simplified here)
        if (out_row < vocab_size)
            TokenOutput[output_base + out_row] = activated;
    }
}
