// ============================================================
// EXPERT SPECIALIZATION KERNEL  (cs_5_1 fallback)
// Replaces WaveActiveSum / WaveIsFirstLane with
// a simple two-step groupshared tree reduction for stats.
//
// Compile: dxc -T cs_5_1 -E main -O3 experts_51.hlsl -Fo experts_51.cso
// ============================================================

#define EXPERTS    8
#define DT         0.016f
#define OMEGA_0    0.05f
#define K_COUPLING 0.12f
#define QUANT_BINS 255.0f
#define GROUP_SIZE 64

#define ROOT_SIG \
    "RootConstants(b0, num32BitConstants=4), \
     SRV(t11), SRV(t12), \
     UAV(u1),  UAV(u2),  UAV(u3),  UAV(u4), UAV(u5), UAV(u9), UAV(u10)"

cbuffer ExpertCB : register(b0)
{
    uint ExpertId;
    uint EntityCount;
    uint ListStride;
    uint FrameIdx;
};

StructuredBuffer<uint>       expert_counts  : register(t11);
StructuredBuffer<uint>       expert_lists   : register(t12);
RWStructuredBuffer<float4>   position       : register(u1);
RWStructuredBuffer<float4>   velocity       : register(u2);
RWStructuredBuffer<float>    signal         : register(u3);
RWStructuredBuffer<float4> axes           : register(u4);
RWStructuredBuffer<float4>   force          : register(u5);
RWStructuredBuffer<uint>     events         : register(u9);
RWStructuredBuffer<float4>   event_params   : register(u10);

// ── Groupshared tree reduction (64 threads) ───────────────────
groupshared float gs_val[GROUP_SIZE];   // for signal sum
groupshared float gs_sq [GROUP_SIZE];   // for signal^2

float3 normalize_safe(float3 v)
{
    return v / max(length(v), 1e-6f);
}

// ── Expert functions (identical to cs_6_0 version) ────────────

float3 expert_geometry(uint eid)
{
    float3 axis  = normalize_safe(float3(axes[eid].x, axes[eid].y, axes[eid].z));
    float3 pos   = position[eid].xyz;
    return -(pos - axis * dot(pos, axis)) * 0.05f;
}

float expert_temporal(uint eid)
{
    float  phase  = event_params[eid].w;
    float3 f_dir  = normalize_safe(force[eid].xyz);
    float  nbr_ph = atan2(f_dir.y, f_dir.x);
    float  new_ph = phase + (OMEGA_0 + K_COUPLING * sin(nbr_ph - phase)) * DT;
    return new_ph - floor(new_ph / 6.2831853f) * 6.2831853f;
}

float expert_amplify(uint eid)
{
    float align = saturate(dot(normalize_safe(velocity[eid].xyz), normalize_safe(force[eid].xyz)));
    return signal[eid] * (1.0f + align * 0.5f);
}

float expert_compress(uint eid)
{
    float s = signal[eid], lo = event_params[eid].x, hi = event_params[eid].y;
    float q = round(saturate((s - lo) / max(hi - lo, 1e-5f)) * QUANT_BINS) / QUANT_BINS;
    return lo + q * max(hi - lo, 1e-5f);
}

float3 expert_focus(uint eid)
{
    float3 axis = normalize_safe(float3(axes[eid].x, axes[eid].y, axes[eid].z));
    return axis * max(dot(force[eid].xyz, axis), 0.0f);
}

void expert_integrate(uint eid)
{
    float3 vel = (velocity[eid].xyz + force[eid].xyz * DT) * 0.98f;
    float3 pos = position[eid].xyz + vel * DT;
    position[eid] = float4(pos, position[eid].w);
    velocity[eid] = float4(vel, velocity[eid].w);
}

float expert_pattern(uint eid)
{
    return signal[eid] * (1.0f + saturate(event_params[eid].w) * 0.3f);
}

float expert_novelty(uint eid, float mean, float std_dev)
{
    float dev  = abs(signal[eid] - mean);
    float norm = (std_dev > 1e-5f) ? dev / std_dev : 0.0f;
    return lerp(signal[eid], signal[eid] * 2.5f, saturate(norm * 0.5f - 1.0f));
}

// ── Main ──────────────────────────────────────────────────────

[RootSignature(ROOT_SIG)]
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID)
{
    uint ltid     = GTid.x;
    uint slot     = DTid.x;
    uint my_count = expert_counts[ExpertId];

    // Load signal for stats (use 0 if out of range)
    float sig_val = 0.0f;
    uint  eid     = 0;
    bool  active  = (slot < my_count);

    if (active) {
        eid     = expert_lists[ExpertId * ListStride + slot];
        sig_val = signal[eid];
    }

    // ── Groupshared tree reduction for mean/std ───────────────
    gs_val[ltid] = active ? sig_val        : 0.0f;
    gs_sq [ltid] = active ? sig_val*sig_val: 0.0f;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (ltid < stride) {
            gs_val[ltid] += gs_val[ltid + stride];
            gs_sq [ltid] += gs_sq [ltid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float n     = max((float)min(my_count, GROUP_SIZE), 1.0f);
    float mean  = gs_val[0] / n;
    float var   = max(gs_sq[0] / n - mean * mean, 0.0f);
    float std   = sqrt(var);

    if (!active) return;

    // ── Expert dispatch ───────────────────────────────────────
    float  new_signal = sig_val;
    float3 new_force  = force[eid].xyz;
    float  new_phase  = event_params[eid].w;

    if      (ExpertId == 0) new_force  = force[eid].xyz + expert_geometry(eid);
    else if (ExpertId == 1) new_phase  = expert_temporal(eid);
    else if (ExpertId == 2) new_signal = expert_amplify(eid);
    else if (ExpertId == 3) new_signal = expert_compress(eid);
    else if (ExpertId == 4) new_force  = expert_focus(eid);
    else if (ExpertId == 5) expert_integrate(eid);
    else if (ExpertId == 6) new_signal = expert_pattern(eid);
    else if (ExpertId == 7) new_signal = expert_novelty(eid, mean, std);

    signal[eid]         = new_signal;
    force[eid]          = float4(new_force, 0.0f);
    event_params[eid].w = new_phase;

    uint packed = (ExpertId << 24) | (uint(saturate(new_signal) * 0xFFFFFF) & 0xFFFFFF);
    events[eid] = packed;
}
