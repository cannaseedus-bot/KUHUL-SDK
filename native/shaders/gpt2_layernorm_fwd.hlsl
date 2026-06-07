// gpt2_layernorm_fwd.hlsl — LayerNorm forward, saves x_hat for backward
// y = gamma * (x - mean) / sqrt(var + eps) + beta
// Dispatch(seq_len, 1, 1)  numthreads(1,1,1) — one group per position

cbuffer LNFwdParams : register(b0) {
    uint n_embd;
    uint seq_len;
    float eps;
    uint pad0;
};

StructuredBuffer<float>   x      : register(t0);  // [seq, n_embd]
StructuredBuffer<float>   gamma  : register(t1);  // [n_embd]
StructuredBuffer<float>   beta   : register(t2);  // [n_embd]

RWStructuredBuffer<float> y      : register(u0);  // [seq, n_embd] output
RWStructuredBuffer<float> x_hat  : register(u1);  // [seq, n_embd] saved for bwd

[numthreads(1, 1, 1)]
void CSMain(uint3 gid : SV_GroupID) {
    const uint s    = gid.x;
    const uint base = s * n_embd;

    float mean = 0.0f;
    for (uint i = 0; i < n_embd; ++i) mean += x[base + i];
    mean /= (float)n_embd;

    float var = 0.0f;
    for (uint i = 0; i < n_embd; ++i) {
        float d = x[base + i] - mean;
        var += d * d;
    }
    var /= (float)n_embd;

    const float inv_std = 1.0f / sqrt(var + eps);

    for (uint i = 0; i < n_embd; ++i) {
        const float xh = (x[base + i] - mean) * inv_std;
        x_hat[base + i] = xh;
        y[base + i]     = gamma[i] * xh + beta[i];
    }
}
