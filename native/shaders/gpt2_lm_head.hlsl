// gpt2_lm_head.hlsl — unembedding: logits = hidden[last] @ wte.T
// GPT-2 ties lm_head to wte — no separate weight.
// Only scores the LAST sequence position (LM generation / training target).
// Dispatch(ceil(vocab_size/64), 1, 1)  numthreads(64,1,1)

cbuffer LMHeadParams : register(b0) {
    uint vocab_size;
    uint n_embd;
    uint last_pos;   // seq_len - 1
    uint pad0;
};

StructuredBuffer<float>   hidden : register(t0);  // [seq, n_embd]
StructuredBuffer<float>   wte    : register(t1);  // [vocab, n_embd]

RWStructuredBuffer<float> logits : register(u0);  // [vocab_size]

[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    const uint v = tid.x;
    if (v >= vocab_size) return;

    const uint h_base = last_pos * n_embd;
    const uint w_base = v * n_embd;

    float dot = 0.0f;
    for (uint d = 0; d < n_embd; ++d)
        dot += hidden[h_base + d] * wte[w_base + d];

    logits[v] = dot;
}
