// gpt2_attn_bwd.hlsl — Attention backward pass
// Forward: Attn = softmax(QK^T / scale) @ V
// Backward: dQ, dK, dV from dAttn_out
//
// Three-pass algorithm for correctness:
// Pass 1: Compute dP[i,j] = sum_d dOut[i,d] * V[j,d]
// Pass 2: Compute softmax backward: dS[i,j] = P[i,j]*(dP[i,j] - sum_k P[i,k]*dP[i,k])
// Pass 3: Compute dQ, dK, dV from dS
//
// Dispatch(seq_len, seq_len, 1)  numthreads(1, 1, 1) per (i,j) pair
// For large seq_len, use tiled Flash Attention backward instead

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
RWStructuredBuffer<float> dP     : register(u3);  // [seq, seq]

groupshared float gs_row_sum[256];

// Pass 1: Compute dP[i,j] = dot(dOut[i], V[j])
[numthreads(1, 1, 1)]
void CSMain_dP(uint3 gid : SV_GroupID) {
    const uint i = gid.x;  // query position
    const uint j = gid.y;  // key/value position
    
    if (i >= seq_len || j >= seq_len) return;
    
    float dp = 0.0f;
    for (uint d = 0; d < head_dim; ++d) {
        dp += dOut[i * head_dim + d] * V[j * head_dim + d];
    }
    dP[i * seq_len + j] = dp;
}

// Pass 2: Softmax backward - compute row sums then dS
[numthreads(1, 1, 1)]
void CSMain_dS(uint3 gid : SV_GroupID) {
    const uint i = gid.x;  // query position
    if (i >= seq_len) return;
    
    // Compute sum_k P[i,k] * dP[i,k] for this row
    float row_sum = 0.0f;
    for (uint k = 0; k < seq_len; ++k) {
        row_sum += P[i * seq_len + k] * dP[i * seq_len + k];
    }
    
    // Compute dS[i,j] = P[i,j] * (dP[i,j] - row_sum)
    for (uint j = 0; j < seq_len; ++j) {
        float dS = P[i * seq_len + j] * (dP[i * seq_len + j] - row_sum);
        dP[i * seq_len + j] = dS * scale;  // store scaled dS back in dP
    }
}

// Pass 3: Compute dQ, dK, dV from dS
// dQ[i] = sum_j dS[i,j] * K[j]
// dK[j] = sum_i dS[i,j] * Q[i]
// dV[j] = sum_i P[i,j] * dOut[i]
[numthreads(1, 1, 1)]
void CSMain_dQKV(uint3 gid : SV_GroupID) {
    const uint i = gid.x;  // query position
    const uint j = gid.y;  // key/value position
    
    if (i >= seq_len || j >= seq_len) return;
    
    const float dS_ij = dP[i * seq_len + j];  // scaled dS from pass 2
    const float P_ij = P[i * seq_len + j];
    
    // dQ[i] += dS[i,j] * K[j]
    for (uint d = 0; d < head_dim; ++d) {
        dQ[i * head_dim + d] += dS_ij * K[j * head_dim + d];
    }
    
    // dK[j] += dS[i,j] * Q[i]
    for (uint d = 0; d < head_dim; ++d) {
        dK[j * head_dim + d] += dS_ij * Q[i * head_dim + d];
    }
    
    // dV[j] += P[i,j] * dOut[i]
    for (uint d = 0; d < head_dim; ++d) {
        dV[j * head_dim + d] += P_ij * dOut[i * head_dim + d];
    }
}
