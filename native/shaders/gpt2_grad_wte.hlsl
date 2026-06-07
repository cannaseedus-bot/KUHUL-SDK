// gpt2_grad_wte.hlsl — wte gradient from lm_head backward
// grad_wte[v, d] += sum_s(dlogits[s, v] * hidden[s, d])
// Dispatch(ceil(vocab/8), ceil(n_embd/8), 1)  numthreads(8,8,1)

cbuffer GradWTEParams : register(b0) {
    uint vocab_size;
    uint n_embd;
    uint seq_len;
    uint pad0;
};

StructuredBuffer<float>   dlogits : register(t0);  // [seq, vocab]
StructuredBuffer<float>   hidden  : register(t1);  // [seq, n_embd]
RWStructuredBuffer<float> grad_w  : register(u0);  // [vocab, n_embd] +=

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    const uint v = tid.x;
    const uint d = tid.y;
    if (v >= vocab_size || d >= n_embd) return;

    float acc = 0.0f;
    for (uint s = 0; s < seq_len; ++s)
        acc += dlogits[s * vocab_size + v] * hidden[s * n_embd + d];
    grad_w[v * n_embd + d] += acc;
}
