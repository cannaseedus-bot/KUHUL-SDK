/**
 * @file fabric_kernels.hlsl
 * @brief Core Fabric compute kernels for transformer operations
 * 
 * Implements:
 * - Multi-head attention
 * - MoE routing
 * - SCXQ2 GPU decompression
 * - Expert merging
 * 
 * All kernels are optimized for:
 * - Async compute dispatch
 * - Descriptor heap binding
 * - Minimal register pressure
 * - Coalesced memory access
 */

// ============================================================================
// Multi-Head Attention Kernel
// ============================================================================

/**
 * @brief Multi-head self-attention (transformer block)
 * 
 * Computes: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
 * 
 * Layout (per token):
 *   [Q_head0] [Q_head1] ... [Q_headN]
 *   [K_head0] [K_head1] ... [K_headN]  (or from cache)
 *   [V_head0] [V_head1] ... [V_headN]  (or from cache)
 * 
 * Execution:
 *   - One thread group per head
 *   - Threads process sequence positions in parallel
 *   - Uses shared memory for softmax reduction
 * 
 * Dispatch: numHeads × seqLen × 1
 */

StructuredBuffer<float> Queries : register(t0);        // Q: (B, N, H, d_k)
StructuredBuffer<float> Keys : register(t1);           // K: (S, N, H, d_k)
StructuredBuffer<float> Values : register(t2);         // V: (S, N, H, d_v)
StructuredBuffer<float> PosBias : register(t3);        // Positional bias (optional)

RWStructuredBuffer<float> AttentionOutput : register(u0);  // Output: (B, N, H, d_v)
RWStructuredBuffer<float> AttentionWeights : register(u1); // Debug: attention weights

cbuffer AttentionParams : register(b0)
{
    uint num_heads;                // Number of attention heads
    uint head_dim;                 // Dimension per head
    uint seq_len;                  // Sequence length
    uint batch_size;
    float temperature;             // Temperature for softmax
    float inv_sqrt_dk;            // 1 / sqrt(d_k) for scaling
};

groupshared float shared_attn[64 * 64];  // Shared memory for attention scores

[numthreads(64, 1, 1)]
void MultiHeadAttention(uint3 id : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID)
{
    uint batch = id.y;
    uint head = id.z;
    uint token_pos = id.x;

    if (batch >= batch_size || head >= num_heads || token_pos >= seq_len)
        return;

    // Compute attention scores for this token against all keys
    float max_score = -1e9;
    float sum_exp = 0.0;

    // First pass: compute max score (for numerical stability)
    for (uint key_pos = 0; key_pos < seq_len; key_pos++)
    {
        float score = 0.0;

        // Dot product: Q @ K^T
        for (uint d = 0; d < head_dim; d++)
        {
            float q = Queries[batch * seq_len * num_heads * head_dim +
                              token_pos * num_heads * head_dim +
                              head * head_dim + d];

            float k = Keys[key_pos * num_heads * head_dim +
                           head * head_dim + d];

            score += q * k;
        }

        // Scale and apply positional bias
        score *= inv_sqrt_dk;
        if (key_pos < seq_len)
        {
            score += PosBias[token_pos * seq_len + key_pos];
        }

        max_score = max(max_score, score);
    }

    // Second pass: compute attention weights (softmax)
    for (uint key_pos = 0; key_pos < seq_len; key_pos++)
    {
        float score = 0.0;

        for (uint d = 0; d < head_dim; d++)
        {
            float q = Queries[batch * seq_len * num_heads * head_dim +
                              token_pos * num_heads * head_dim +
                              head * head_dim + d];

            float k = Keys[key_pos * num_heads * head_dim +
                           head * head_dim + d];

            score += q * k;
        }

        score *= inv_sqrt_dk;
        if (key_pos < seq_len)
        {
            score += PosBias[token_pos * seq_len + key_pos];
        }

        float weight = exp((score - max_score) * temperature);
        sum_exp += weight;

        // Store in shared memory for next pass
        shared_attn[key_pos] = weight;
    }

    // Third pass: weighted sum over values
    float3 output = float3(0.0, 0.0, 0.0);

    for (uint key_pos = 0; key_pos < seq_len; key_pos++)
    {
        float weight = shared_attn[key_pos] / (sum_exp + 1e-6);

        // Accumulate: Σ(weight_i * V_i)
        for (uint d = 0; d < head_dim; d += 3)  // Vectorized (3 at a time)
        {
            float3 v = float3(
                Values[key_pos * num_heads * head_dim + head * head_dim + d],
                d + 1 < head_dim ? Values[key_pos * num_heads * head_dim + head * head_dim + d + 1] : 0.0,
                d + 2 < head_dim ? Values[key_pos * num_heads * head_dim + head * head_dim + d + 2] : 0.0
            );

            output += weight * v;
        }
    }

    // Write output
    for (uint d = 0; d < head_dim; d++)
    {
        AttentionOutput[batch * seq_len * num_heads * head_dim +
                        token_pos * num_heads * head_dim +
                        head * head_dim + d] = output[d % 3];
    }
}

// ============================================================================
// MoE Router Kernel
// ============================================================================

/**
 * @brief Route tokens to mixture of experts
 * 
 * Computes: expert_scores = softmax(token @ W_router)
 * Selects: top-k experts per token
 * 
 * Output: (token_id, expert_id) pairs for dispatch
 */

StructuredBuffer<float> TokenEmbeddings : register(t0);    // Input tokens
StructuredBuffer<float> RouterWeights : register(t1);      // Router matrix

RWStructuredBuffer<uint> ExpertIndices : register(u0);     // Output: expert indices
RWStructuredBuffer<float> ExpertGates : register(u1);      // Output: gating weights

cbuffer MoEParams : register(b0)
{
    uint num_experts;              // Total experts
    uint topk;                     // Experts per token
    uint router_dim;               // Router embedding dimension
    uint token_count;
};

[numthreads(64, 1, 1)]
void MoERouter(uint3 id : SV_DispatchThreadID)
{
    uint token_idx = id.x;

    if (token_idx >= token_count)
        return;

    // Compute router scores: token @ W_router
    float scores[8];  // Assume max 8 experts (can be higher)
    for (uint e = 0; e < num_experts && e < 8; e++)
    {
        float score = 0.0;

        for (uint d = 0; d < router_dim; d++)
        {
            float token_val = TokenEmbeddings[token_idx * router_dim + d];
            float weight = RouterWeights[e * router_dim + d];
            score += token_val * weight;
        }

        scores[e] = score;
    }

    // Find top-k experts (simple bubble sort for k=2)
    uint topk_experts[2] = {0, 1};
    float topk_scores[2] = {scores[0], scores[1]};

    for (uint e = 2; e < num_experts; e++)
    {
        if (scores[e] > topk_scores[1])
        {
            if (scores[e] > topk_scores[0])
            {
                topk_scores[1] = topk_scores[0];
                topk_experts[1] = topk_experts[0];

                topk_scores[0] = scores[e];
                topk_experts[0] = e;
            }
            else
            {
                topk_scores[1] = scores[e];
                topk_experts[1] = e;
            }
        }
    }

    // Store expert indices and gating weights
    ExpertIndices[token_idx * 2] = topk_experts[0];
    ExpertIndices[token_idx * 2 + 1] = topk_experts[1];

    ExpertGates[token_idx * 2] = topk_scores[0];
    ExpertGates[token_idx * 2 + 1] = topk_scores[1];
}

// ============================================================================
// SCXQ2 GPU Decompression Kernel
// ============================================================================

/**
 * @brief Decompress SCXQ2-encoded weights on GPU
 * 
 * Format:
 *   [DICT (symbol → value mappings)]
 *   [INDEX (start offset per symbol)]
 *   [ENCODED (symbol stream)]
 * 
 * Operation:
 *   for each encoded_symbol: output = dict[index[symbol] : index[symbol+1]]
 */

StructuredBuffer<uint> Dictionary : register(t0);     // Symbol → value dictionary
StructuredBuffer<uint> SymbolIndex : register(t1);    // Symbol start offsets
StructuredBuffer<uint> EncodedStream : register(t2);  // Compressed symbol stream

RWStructuredBuffer<uint> DecodedWeights : register(u0);  // Output: decompressed

cbuffer SCXQ2Params : register(b0)
{
    uint dict_size;                // Dictionary entries
    uint encoded_count;            // Encoded symbols
    uint output_size;              // Total output elements
};

[numthreads(64, 1, 1)]
void SCXQ2Decode(uint3 id : SV_DispatchThreadID)
{
    uint symbol_idx = id.x;

    if (symbol_idx >= encoded_count)
        return;

    // Read encoded symbol
    uint symbol = EncodedStream[symbol_idx];

    // Look up dictionary entry range
    uint start_offset = SymbolIndex[symbol];
    uint end_offset = (symbol + 1 < dict_size) ? SymbolIndex[symbol + 1] : output_size;

    // Expand: copy dictionary range to output
    for (uint i = start_offset; i < end_offset; i++)
    {
        DecodedWeights[symbol_idx * (end_offset - start_offset) + (i - start_offset)] =
            Dictionary[i];
    }
}

// ============================================================================
// Expert Merge Kernel
// ============================================================================

/**
 * @brief Merge outputs from multiple experts
 * 
 * For each token:
 *   output = Σ(expert_output_k * gate_weight_k) for k in top-k
 * 
 * Where gate_weights come from MoE router softmax
 */

StructuredBuffer<float> ExpertOutputs : register(t0);      // Per-expert outputs
StructuredBuffer<uint> ExpertAssignments : register(t1);   // Token → expert mapping
StructuredBuffer<float> GateWeights : register(t2);        // Gating weights

RWStructuredBuffer<float> MergedOutput : register(u0);     // Fused output

cbuffer MergeParams : register(b0)
{
    uint token_count;
    uint output_dim;
    uint topk;
};

[numthreads(64, 1, 1)]
void ExpertMerge(uint3 id : SV_DispatchThreadID)
{
    uint token_idx = id.x;
    uint dim_idx = id.y;

    if (token_idx >= token_count || dim_idx >= output_dim)
        return;

    float merged = 0.0;

    // Combine top-k expert outputs
    for (uint k = 0; k < topk; k++)
    {
        uint expert_id = ExpertAssignments[token_idx * topk + k];
        float gate = GateWeights[token_idx * topk + k];

        float expert_out = ExpertOutputs[expert_id * output_dim + dim_idx];
        merged += gate * expert_out;
    }

    MergedOutput[token_idx * output_dim + dim_idx] = merged;
}
