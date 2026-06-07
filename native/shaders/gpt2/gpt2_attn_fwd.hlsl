// gpt2_attn_fwd.hlsl — Single-head causal self-attention forward
// Inputs: Q, K, V each [seq, head_dim] (pre-projected, one head)
// Output: out [seq, n_embd] (writes to head_idx slice), P [seq, seq] softmax weights (saved for backward)
// Dispatch(seq_len, 1, 1)  numthreads(1,1,1) — one group per query position

cbuffer AttnFwdParams : register(b0) {
    uint  seq_len;
    uint  head_dim;
    uint  n_embd;      // full embedding dim — used to index into Out correctly
    uint  head_idx;    // which head this dispatch handles
    float scale;       // 1/sqrt(head_dim)
    float pad0;
    float pad1;
    float pad2;
};

StructuredBuffer<float>   Q   : register(t0);  // [seq, head_dim]
StructuredBuffer<float>   K   : register(t1);  // [seq, head_dim]
StructuredBuffer<float>   V   : register(t2);  // [seq, head_dim]

RWStructuredBuffer<float> Out : register(u0);  // [seq, n_embd]
RWStructuredBuffer<float> P   : register(u1);  // [seq, seq] — softmax weights for bwd

[numthreads(1, 1, 1)]
void CSMain(uint3 gid : SV_GroupID) {
    const uint i = gid.x;   // query position

    // Compute raw scores for all key positions <= i (causal mask)
    float scores[512];   // max seq 512; increase if needed
    float max_score = -1e30f;
    for (uint j = 0; j <= i; ++j) {
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d)
            dot += Q[i * head_dim + d] * K[j * head_dim + d];
        scores[j] = dot * scale;
        if (scores[j] > max_score) max_score = scores[j];
    }

    // Softmax (causal — positions j > i are masked to -inf)
    float sum_exp = 0.0f;
    for (uint j = 0; j <= i; ++j) {
        scores[j] = exp(scores[j] - max_score);
        sum_exp += scores[j];
    }
    for (uint j = 0; j <= i; ++j) {
        P[i * seq_len + j] = scores[j] / sum_exp;
    }
    for (uint j = i + 1; j < seq_len; ++j) {
        P[i * seq_len + j] = 0.0f;
    }

    // out[i, head_idx*head_dim .. (head_idx+1)*head_dim] = sum_j P[i,j] * V[j]
    const uint out_base = i * n_embd + head_idx * head_dim;
    for (uint d = 0; d < head_dim; ++d) {
        float acc = 0.0f;
        for (uint j = 0; j <= i; ++j)
            acc += P[i * seq_len + j] * V[j * head_dim + d];
        Out[out_base + d] = acc;
    }
}
