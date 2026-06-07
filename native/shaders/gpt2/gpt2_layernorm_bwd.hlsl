// gpt2_layernorm_bwd.hlsl — LayerNorm backward, cs_5_0
//
// Two entry points — call BOTH in order:
//
// CSMain      — dx pass (parallel per sequence position)
//   Dispatch(seq_len, 1, 1)  numthreads(256, 1, 1)
//   Computes dx for each seq position using saved xhat and inv_std.
//   Does NOT touch dgamma/dbeta — avoids cross-group race condition.
//
// CSMain_params — gamma/beta gradient pass (parallel per embedding dim)
//   Dispatch(ceil(n_embd/256), 1, 1)  numthreads(256, 1, 1)
//   Each thread owns a unique [j] index, loops over all seq positions.
//   No race: each thread's j is unique across all dispatch groups.
//
// Bug fixed (v0.1.1 → v0.1.2):
//   Old single-entry-point wrote dgamma[i]/dbeta[i] inside the per-seq dispatch,
//   causing data races between groups. Also missing inv_std scaling on dx.
//
// Formula:
//   dx[i] = inv_std/N * (N*g[i]*dy[i] - Σ_j(g[j]*dy[j]) - xhat[i]*Σ_j(g[j]*dy[j]*xhat[j]))

cbuffer LNBwdParams : register(b0) {
    uint  n_embd;   // N
    uint  seq_len;
    float eps;      // kept for cbuffer compatibility; inv_std is passed as buffer
    uint  pad0;
};

StructuredBuffer<float>   xhat    : register(t0);  // normalized input [seq, N]
StructuredBuffer<float>   gamma   : register(t1);  // weight [N]
StructuredBuffer<float>   dout    : register(t2);  // upstream gradient [seq, N]
StructuredBuffer<float>   inv_std : register(t3);  // [seq] saved from forward pass

RWStructuredBuffer<float> dx      : register(u0);  // [seq, N]  (accumulate +=)
RWStructuredBuffer<float> dgamma  : register(u1);  // [N]       (accumulate +=)
RWStructuredBuffer<float> dbeta   : register(u2);  // [N]       (accumulate +=)

groupshared float gs_buf[512];  // [0..255] = sum_dg, [256..511] = sum_dgx

// ── CSMain: compute dx, one group per sequence position ──────────────────────
[numthreads(256, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    const uint seq_idx = gid.x;
    const uint tid     = lid.x;
    const uint base    = seq_idx * n_embd;
    const float N      = (float)n_embd;
    const float istd   = inv_std[seq_idx];

    float sum_dg  = 0.0f;
    float sum_dgx = 0.0f;
    for (uint i = tid; i < n_embd; i += 256) {
        const float g   = gamma[i];
        const float dy  = dout[base + i];
        const float xh  = xhat[base + i];
        const float gdy = g * dy;
        sum_dg  += gdy;
        sum_dgx += gdy * xh;
        // dgamma/dbeta NOT updated here — see CSMain_params
    }

    gs_buf[tid]       = sum_dg;
    gs_buf[256 + tid] = sum_dgx;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = 128; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            gs_buf[tid]       += gs_buf[tid + stride];
            gs_buf[256 + tid] += gs_buf[256 + tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    const float tot_dg  = gs_buf[0];
    const float tot_dgx = gs_buf[256];
    const float inv_N   = 1.0f / N;

    for (uint i = tid; i < n_embd; i += 256) {
        const float g  = gamma[i];
        const float dy = dout[base + i];
        const float xh = xhat[base + i];
        dx[base + i] += istd * inv_N * (N * g * dy - tot_dg - xh * tot_dgx);
    }
}

// ── CSMain_params: dgamma + dbeta, race-free ─────────────────────────────────
// Dispatch(ceil(n_embd/256), 1, 1)  numthreads(256, 1, 1)
[numthreads(256, 1, 1)]
void CSMain_params(uint3 tid : SV_DispatchThreadID) {
    const uint j = tid.x;
    if (j >= n_embd) return;

    float dg = 0.f, db = 0.f;
    for (uint s = 0; s < seq_len; ++s) {
        const float dy = dout[s * n_embd + j];
        const float xh = xhat[s * n_embd + j];
        dg += dy * xh;
        db += dy;
    }
    dgamma[j] += dg;
    dbeta[j]  += db;
}
