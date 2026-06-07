// gpt2_qkv_split.hlsl — Split [seq, 3*n_embd] QKV into head-major layout
// Output: q/k/v each [n_head, seq, head_dim] — contiguous per head
// Dispatch(ceil(seq/8), ceil(n_head/4), 1)  numthreads(8,4,1)

cbuffer QKVParams : register(b0) {
    uint seq_len;
    uint n_embd;
    uint n_head;
    uint head_dim;
};

StructuredBuffer<float>   qkv_all : register(t0);  // [seq, 3*n_embd]
RWStructuredBuffer<float> q_out   : register(u0);  // [n_head, seq, head_dim]
RWStructuredBuffer<float> k_out   : register(u1);
RWStructuredBuffer<float> v_out   : register(u2);

[numthreads(8, 4, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    const uint s = tid.x;
    const uint h = tid.y;
    if (s >= seq_len || h >= n_head) return;

    const uint qkv_row = s * 3 * n_embd;
    const uint head_off = h * head_dim;
    const uint out_base = h * seq_len * head_dim + s * head_dim;

    for (uint d = 0; d < head_dim; ++d) {
        q_out[out_base + d] = qkv_all[qkv_row + head_off + d];
        k_out[out_base + d] = qkv_all[qkv_row + n_embd   + head_off + d];
        v_out[out_base + d] = qkv_all[qkv_row + 2*n_embd + head_off + d];
    }
}
