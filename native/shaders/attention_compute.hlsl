// ============================================================
// KUHUL TENSOR SHADER — ATTENTION (v1)
// ============================================================
// Shader 2: Neural/Tensor Engine
// Transformer backend with KV cache
// Works with XCFE routing
// INT4-ready (currently float32 for precision)
//
// Compile: dxc -T cs_6_0 -E CS_AttnPass1 -O3 attention_compute.hlsl -Fo attention_pass1.cso
//          dxc -T cs_6_0 -E CS_AttnPass2 -O3 attention_compute.hlsl -Fo attention_pass2.cso
// ============================================================

cbuffer AttentionParams : register(b0)
{
    uint  T;             // Sequence length
    uint  D;             // Dimension (hidden size)
    uint  H;             // Number of heads
    uint  _pad0;
    float scale;         // 1.0 / sqrt(D)
    float dropout;       // Dropout rate (0.0 = disabled)
    uint  cache_offset;  // KV cache write offset
    uint  cache_enabled; // 1 = use KV cache, 0 = disabled
};

// ============================================================================
// BUFFERS
// ============================================================================
// Q, K, V inputs (float32)
StructuredBuffer<float> Q : register(t0);  // [T × D]
StructuredBuffer<float> K : register(t1);  // [T × D] or [cache_len × D]
StructuredBuffer<float> V : register(t2);  // [T × D] or [cache_len × D]

// KV Cache (read-write for incremental decoding/streaming memory)
RWStructuredBuffer<float> KVCache : register(u3);  // [max_len × D × 2]

// Outputs
RWStructuredBuffer<float> Out      : register(u0);  // [T × D] attention output
RWStructuredBuffer<float> Scores   : register(u1);  // [T × T] attention scores (softmax)
RWStructuredBuffer<float> Logits   : register(u2);  // [T × T] raw logits (before softmax)

// Optional: per-head outputs (for multi-head attention)
RWStructuredBuffer<float> HeadOut  : register(u4);  // [H × T × (D/H)]

// ============================================================================
// PASS 1: Compute Q·K scores + store KV cache
// ============================================================================
[numthreads(64, 1, 1)]
void CS_AttnPass1(uint3 DTid : SV_DispatchThreadID)
{
    uint q_idx = DTid.x;  // Query position
    if (q_idx >= T) return;

    uint q_base = q_idx * D;
    
    // Compute dot product with all keys
    for (uint k_idx = 0; k_idx < T; k_idx++)
    {
        uint k_base = k_idx * D;
        float dot = 0.0;

        // Dot product Q·K
        for (uint d = 0; d < D; d++)
        {
            dot += Q[q_base + d] * K[k_base + d];
        }

        // Scale by 1/sqrt(D)
        float scaled = dot * scale;
        
        // Store raw logit
        Logits[q_idx * T + k_idx] = scaled;

        // Apply causal mask (optional, controlled by sign of scale)
        if (scale < 0 && k_idx > q_idx)
        {
            scaled = -1e9;  // Mask future positions
        }

        // Exponential for softmax
        float score = exp(scaled);
        Scores[q_idx * T + k_idx] = score;
    }

    // Write KV cache (for incremental decoding/streaming memory)
    if (cache_enabled) {
        uint cache_base = (cache_offset + q_idx) * D * 2;
        for (uint d = 0; d < D; d++)
        {
            KVCache[cache_base + d]         = K[q_base + d];  // Key
            KVCache[cache_base + D + d]     = V[q_base + d];  // Value
        }
    }
}

// ============================================================================
// PASS 2: Normalize scores + apply V
// ============================================================================
[numthreads(64, 1, 1)]
void CS_AttnPass2(uint3 DTid : SV_DispatchThreadID)
{
    uint q_idx = DTid.x;  // Query position
    if (q_idx >= T) return;

    // -----------------------------------------------------------------------
    // Step 1: Compute denominator (sum of scores)
    // -----------------------------------------------------------------------
    float denom = 0.0;
    for (uint k_idx = 0; k_idx < T; k_idx++)
    {
        denom += Scores[q_idx * T + k_idx];
    }
    
    // Avoid division by zero
    denom = max(denom, 1e-9);
    float inv_denom = 1.0 / denom;

    // -----------------------------------------------------------------------
    // Step 2: Weighted sum of values (each thread handles one output element)
    // -----------------------------------------------------------------------
    uint out_idx = q_idx * D + DTid.y;
    if (DTid.y >= D) return;
    
    float sum = 0.0;
    for (uint k_idx = 0; k_idx < T; k_idx++)
    {
        float score = Scores[q_idx * T + k_idx] * inv_denom;
        uint v_base = k_idx * D;
        sum += score * V[v_base + DTid.y];
    }
    
    Out[out_idx] = sum;
}

// ============================================================================
// PASS 3 (Optional): Multi-head split
// ============================================================================
[numthreads(64, 1, 1)]
void CS_MultiHeadSplit(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= T * D) return;

    uint q_idx = idx / D;
    uint d_idx = idx % D;
    uint head_dim = D / H;
    uint head_idx = d_idx / head_dim;
    uint head_offset = d_idx % head_dim;

    // Write to per-head output
    uint head_base = head_idx * T * head_dim;
    HeadOut[head_base + q_idx * head_dim + head_offset] = Out[idx];
}

// ============================================================================
// XCFE INTEGRATION
// ============================================================================
// Attention dispatch sequence:
//   1. CS_AttnPass1(T, D, H, scale, cache_enabled=1) → scores + KV cache
//   2. [CPU/XCFE: optional reduce] → causal mask, custom masking
//   3. CS_AttnPass2(T, D, H) → normalize + apply V
//   4. [Optional] CS_MultiHeadSplit → split into H heads
//
// Out buffer → next layer input or lm_head projection
// ============================================================================
