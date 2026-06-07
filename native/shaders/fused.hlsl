// ============================================================
// FUSED QKV + ATTENTION + FORCE + MoE ROUTING  (cs_6_0)
// Pass 1 of 3 — single-pass, branchless, wave-friendly.
//
// Compile: dxc -T cs_6_0 -E main -O3 fused.hlsl -Fo fused.cso
// ============================================================

#define MAX_NEIGHBORS 32
#define EXPERTS 8
#define TOPK 2

// ------------------------------------------------------------
// ROOT SIGNATURE
// ------------------------------------------------------------
#define ROOT_SIG \
    "RootConstants(b0, num32BitConstants=4), \
     SRV(t0), \
     UAV(u1), UAV(u2), UAV(u3), UAV(u4), UAV(u5), \
     SRV(t6), SRV(t7), SRV(t8), \
     UAV(u9), UAV(u10)"

cbuffer FusedCB : register(b0)
{
    uint EntityCount;
    uint GridW;
    uint FrameIdx;
    uint _pad;
};

// ------------------------------------------------------------
// BUFFERS
// ------------------------------------------------------------
StructuredBuffer<uint>       entities      : register(t0);
RWStructuredBuffer<float4>   position      : register(u1);
RWStructuredBuffer<float4>   velocity      : register(u2);
RWStructuredBuffer<float>    signal        : register(u3);
RWStructuredBuffer<float4> axes          : register(u4);
RWStructuredBuffer<float4>   force         : register(u5);

StructuredBuffer<uint>       grid_offsets  : register(t6);
StructuredBuffer<uint>       grid_counts   : register(t7);
StructuredBuffer<uint>       grid_indices  : register(t8);

RWStructuredBuffer<uint>     events        : register(u9);
RWStructuredBuffer<float4>   event_params  : register(u10);

// ------------------------------------------------------------
// HELPERS
// ------------------------------------------------------------
float3 normalize_safe(float3 v)
{
    return v / max(length(v), 1e-6f);
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
[RootSignature(ROOT_SIG)]
[numthreads(128,1,1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint i = DTid.x;
    if (i >= EntityCount) return;

    float3 pos_i = position[i].xyz;
    float3 vel_i = velocity[i].xyz;
    float  sig_i = signal[i];

    uint GridCells = GridW * GridW * GridW;
    uint cx = min((uint)max(pos_i.x, 0.0f), GridW - 1);
    uint cy = min((uint)max(pos_i.y, 0.0f), GridW - 1);
    uint cz = min((uint)max(pos_i.z, 0.0f), GridW - 1);
    uint cell = min(cx + cy * GridW + cz * (GridW * GridW), GridCells - 1);

    uint offset = grid_offsets[cell];
    uint count  = min(grid_counts[cell], MAX_NEIGHBORS);

    float4 Q = float4(pos_i, sig_i);

    float3 force_acc = float3(0,0,0);
    float  attn_sum  = 0.0f;

    float top_score[TOPK];
    uint  top_index[TOPK];

    [unroll]
    for (uint k = 0; k < TOPK; k++) {
        top_score[k] = -1e9f;
        top_index[k] = 0;
    }

    [unroll]
    for (uint n = 0; n < MAX_NEIGHBORS; n++)
    {
        float active = (n < count) ? 1.0f : 0.0f;
        uint  j      = grid_indices[min(offset + n, EntityCount - 1)];

        float3 pos_j = position[j].xyz;
        float  sig_j = signal[j];

        float4 K     = float4(pos_j, sig_j);
        float3 V     = velocity[j].xyz;

        float score  = dot(Q, K) * active;
        float w      = exp(clamp(score, -20.0f, 20.0f));

        attn_sum    += w;
        force_acc   += w * normalize_safe(pos_j - pos_i) * active;

        uint  expert_id = j % EXPERTS;
        float s         = score;

        [unroll]
        for (uint k = 0; k < TOPK; k++)
        {
            float better    = (s > top_score[k]) ? 1.0f : 0.0f;
            float old_score = top_score[k];
            uint  old_index = top_index[k];

            top_score[k]    = lerp(top_score[k], s, better);
            top_index[k]    = (better > 0.5f) ? expert_id : top_index[k];

            s               = lerp(s, old_score, better);
        }
    }

    float  inv    = 1.0f / max(attn_sum, 1e-5f);
    force_acc    *= inv;

    force[i]         = float4(force_acc, 0.0f);
    signal[i]        = (float)top_index[0] * 0.5f + (float)top_index[1] * 0.5f;
    events[i]        = top_index[0];
    event_params[i]  = float4(force_acc, (float)top_index[0]);
}
