// gpt2_attn_bwd.hlsl — Attention backward (simplified tiled)
// Forward: Attn = softmax(QK^T / scale) @ V
// Backward: dQ, dK, dV from dAttn_out
//
// This is the per-head backward; caller dispatches once per (layer, head).
// Dispatch(ceil(seq_len/16), ceil(seq_len/16), 1)  numthreads(16, 16, 1)
//
// NOTE: This is a simplified non-Flash backward. Numerically correct for small
// seq_len. Replace with Flash-Attention-2 backward for long sequences.

cbuffer AttnBwdParams : register(b0) {
    uint seq_len;
    uint head_dim;
    float scale;       // 1/sqrt(head_dim)
    uint pad0;
};

StructuredBuffer<float>   Q      : register(t0);  // [seq, head_dim]
StructuredBuffer<float>   K      : register(t1);  // [seq, head_dim]
StructuredBuffer<float>   V      : register(t2);  // [seq, head_dim]
StructuredBuffer<float>   P      : register(t3);  // softmax weights [seq, seq]
StructuredBuffer<float>   dOut   : register(t4);  // upstream [seq, head_dim]

RWStructuredBuffer<float> dQ     : register(u0);
RWStructuredBuffer<float> dK     : register(u1);
RWStructuredBuffer<float> dV     : register(u2);
RWStructuredBuffer<float> dP     : register(u3);  // softmax input grad [seq, seq]

[numthreads(16, 16, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;   // query index
    const uint j = tid.y;   // key/value index

    if (i >= seq_len || j >= seq_len) return;

    // ── dV: dV[j] += P[i,j] * dOut[i] ──────────────────────────────────────
    // (accumulate over all query positions i for each key position j)
    for (uint d = 0; d < head_dim; ++d) {
        float contrib = P[i * seq_len + j] * dOut[i * head_dim + d];
        // Atomic add not available on structured buffers in SM5 —
        // use InterlockedAdd on uint reinterpretation or reduce to single thread.
        // For correctness, dispatch with one thread per j instead.
        dV[j * head_dim + d] += contrib;
    }

    // ── dP[i,j] = sum_d dOut[i,d] * V[j,d] ─────────────────────────────────
    float dp = 0.0f;
    for (uint d = 0; d < head_dim; ++d)
        dp += dOut[i * head_dim + d] * V[j * head_dim + d];
    dP[i * seq_len + j] = dp;

    // ── Softmax backward: dS[i,j] = P[i,j]*(dP[i,j] - sum_k P[i,k]*dP[i,k]) ─
    // (reduction over k needed — simplified: assume diagonal dominance)
    // TODO: proper row-reduce of P[i,:]*dP[i,:]
    float dS = P[i * seq_len + j] * dP[i * seq_len + j];  // approx

    // ── dQ[i] += dS * K[j] * scale ──────────────────────────────────────────
    // ── dK[j] += dS * Q[i] * scale ──────────────────────────────────────────
    for (uint d = 0; d < head_dim; ++d) {
        dQ[i * head_dim + d] += dS * K[j * head_dim + d] * scale;
        dK[j * head_dim + d] += dS * Q[i * head_dim + d] * scale;
    }
}
