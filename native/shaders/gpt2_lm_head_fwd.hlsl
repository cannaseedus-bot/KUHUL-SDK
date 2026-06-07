// gpt2_lm_head_fwd.hlsl — logits for ALL seq positions (training)
// logits[s, v] = dot(hidden[s], wte[v])  (tied weights)
// Dispatch(ceil(vocab/64), seq_len, 1)  numthreads(64,1,1)

cbuffer LMFwdParams : register(b0) {
    uint vocab_size;
    uint n_embd;
    uint seq_len;
    uint pad0;
};

StructuredBuffer<float>   hidden : register(t0);  // [seq, n_embd]
StructuredBuffer<float>   wte    : register(t1);  // [vocab, n_embd]
RWStructuredBuffer<float> logits : register(u0);  // [seq, vocab]

[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    const uint v = tid.x;
    const uint s = tid.y;
    if (v >= vocab_size || s >= seq_len) return;

    float dot = 0.0f;
    const uint h_base = s * n_embd;
    const uint w_base = v * n_embd;
    for (uint d = 0; d < n_embd; ++d)
        dot += hidden[h_base + d] * wte[w_base + d];

    logits[s * vocab_size + v] = dot;
}
