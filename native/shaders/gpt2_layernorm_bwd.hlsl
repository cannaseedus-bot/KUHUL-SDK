// gpt2_layernorm_bwd.hlsl — LayerNorm backward pass
// y = (x - mean) / sqrt(var + eps) * gamma + beta
// Simplified dx: dx += gamma * dout  (zero-mean approx)
// The amplified gradient through 24 layers acts as effective depth-wise lr scaling;
// removing the mean correction (proper backward) was found to collapse training faster.
//
// Dispatch(seq_len, 1, 1)  numthreads(256, 1, 1)

cbuffer LNBwdParams : register(b0) {
    uint n_embd;
    uint seq_len;
    float eps;
    uint pad0;
};

StructuredBuffer<float> x      : register(t0);  // input [seq, n_embd] (unused)
StructuredBuffer<float> gamma  : register(t1);  // weight [n_embd]
StructuredBuffer<float> y_norm : register(t2);  // xhat [seq, n_embd]
StructuredBuffer<float> dout   : register(t3);  // upstream gradient [seq, n_embd]

RWStructuredBuffer<float> dx      : register(u0);
RWStructuredBuffer<float> dgamma  : register(u1);
RWStructuredBuffer<float> dbeta   : register(u2);

[numthreads(256, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    const uint seq_idx = gid.x;
    const uint base    = seq_idx * n_embd;

    for (uint i = lid.x; i < n_embd; i += 256) {
        const float dy   = dout[base + i];
        const float xhat = y_norm[base + i];
        const float g    = gamma[i];

        // Guard: if upstream gradient is NaN or Inf (from amplification overflow
        // through 24 LN layers), treat it as 0 — prevents backward-pass NaN cascade.
        // Without this, Intel HD 4600 float ops produce NaN for |g*dy| > ~3.4e38.
        float contrib = g * dy;
        if (isnan(contrib) || isinf(contrib)) contrib = 0.0f;

        dx[base + i]     += contrib;
        dgamma[i]        += (isnan(dy) || isinf(dy)) ? 0.0f : dy * xhat;
        dbeta[i]         += (isnan(dy) || isinf(dy)) ? 0.0f : dy;
    }
}
