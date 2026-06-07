// gpt2_attn_merge.hlsl — Merge head-major [n_head, seq, head_dim] → [seq, n_embd]
// Dispatch(ceil(seq/8), ceil(n_head/4), 1)  numthreads(8,4,1)

cbuffer MergeParams : register(b0) {
    uint seq_len;
    uint n_head;
    uint head_dim;
    uint n_embd;
};

StructuredBuffer<float>   attn_heads : register(t0);  // [n_head, seq, head_dim]
RWStructuredBuffer<float> merged     : register(u0);  // [seq, n_embd]

[numthreads(8, 4, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    const uint s = tid.x;
    const uint h = tid.y;
    if (s >= seq_len || h >= n_head) return;

    const uint in_base   = h * seq_len * head_dim + s * head_dim;
    const uint out_base  = s * n_embd + h * head_dim;

    for (uint d = 0; d < head_dim; ++d)
        merged[out_base + d] = attn_heads[in_base + d];
}
