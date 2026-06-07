// ============================================================
// EXPERT SPECIALIZATION KERNEL  (cs_6_0)
// Pass 3 of 3 — dispatched 8x via ExecuteIndirect.
//
// Each dispatch targets one expert (ExpertId via root constant).
// Reads compacted entity lists from the orchestrate pass.
//
//   E0  geometry   -- bone-axis position correction
//   E1  temporal   -- Kuramoto phase oscillator step
//   E2  amplify    -- resonance-driven signal boost
//   E3  compress   -- SCXQ2 lane quantization (8-bit round-trip)
//   E4  focus      -- principal-axis force alignment
//   E5  integrate  -- symplectic Euler pos/vel update
//   E6  pattern    -- n-gram frequency weighted signal
//   E7  novelty    -- surprise spike for anomalous signal
//
// Compile: dxc -T cs_6_0 -E main -O3 experts.hlsl -Fo experts.cso
// ============================================================

#define EXPERTS    8
#define DT         0.016f          // 60 Hz timestep
#define OMEGA_0    0.05f           // Kuramoto natural frequency
#define K_COUPLING 0.12f           // Kuramoto coupling strength
#define QUANT_BINS 255.0f          // SCXQ2 8-bit quantization bins

// ------------------------------------------------------------
// ROOT SIGNATURE
// ------------------------------------------------------------
#define ROOT_SIG \
    "RootConstants(b0, num32BitConstants=4), \
     SRV(t11), SRV(t12), \
     UAV(u1),  UAV(u2),  UAV(u3),  UAV(u4), UAV(u5), UAV(u9), UAV(u10)"

// ------------------------------------------------------------
// PER-DISPATCH CONSTANTS (pushed by ExecuteIndirect per expert)
// ------------------------------------------------------------
cbuffer ExpertCB : register(b0)
{
    uint ExpertId;     // 0-7: which expert this dispatch runs
    uint EntityCount;  // not used directly -- use expert_counts[ExpertId]
    uint ListStride;   // per-expert list capacity in expert_lists buffer
    uint FrameIdx;     // global frame counter
};

// ------------------------------------------------------------
// BUFFERS  (register layout mirrors fused kernel + orchestrate)
// ------------------------------------------------------------
RWStructuredBuffer<uint>     events         : register(u9);   // top-1 expert in / packed trace out
StructuredBuffer<uint>       expert_counts  : register(t11);  // [EXPERTS] entity counts
StructuredBuffer<uint>       expert_lists   : register(t12);  // [EXPERTS * ListStride]

RWStructuredBuffer<float4>   position       : register(u1);
RWStructuredBuffer<float4>   velocity       : register(u2);
RWStructuredBuffer<float>    signal         : register(u3);
RWStructuredBuffer<float4> axes           : register(u4);
RWStructuredBuffer<float4>   force          : register(u5);
RWStructuredBuffer<float4>   event_params   : register(u10);  // carries [force_acc | phase]

// ------------------------------------------------------------
// GROUPSHARED  (wave stats for E7 novelty detection)
// float atomic via uint bit-reinterpret trick
// ------------------------------------------------------------
// Fixed-point float stats: scale float to int for InterlockedAdd.
// Max meaningful signal range: [-1000, 1000] * STAT_SCALE = fits uint32 for 64 threads.
groupshared uint gs_signal_bits;    // sum(signal)   * STAT_SCALE, as uint
groupshared uint gs_signal_sq_bits; // sum(signal^2) * STAT_SCALE, as uint
groupshared uint gs_count;

static const float STAT_SCALE = 100.0f;  // signal clamped to [-100,100] safe

// ------------------------------------------------------------
// HELPER
// ------------------------------------------------------------
float3 normalize_safe(float3 v)
{
    return v / max(length(v), 1e-6f);
}

// ------------------------------------------------------------
// EXPERT  E0 -- Geometry
// Pull position toward the entity's principal bone axis.
// Correction injected into the force accumulator, not pos directly,
// so E5 (integrate) applies it next frame.
// ------------------------------------------------------------
float3 expert_geometry(uint eid)
{
    float3 axis      = normalize_safe(axes[eid].xyz);
    float3 pos       = position[eid].xyz;
    float  proj      = dot(pos, axis);
    float3 on_axis   = axis * proj;
    float3 off_axis  = pos - on_axis;
    return -off_axis * 0.05f;               // gentle restoration toward axis
}

// ------------------------------------------------------------
// EXPERT  E1 -- Temporal (Kuramoto)
// Advances the phase oscillator.  Phase is stored in event_params.w
// ------------------------------------------------------------
float expert_temporal(uint eid)
{
    float  phase    = event_params[eid].w;
    float3 f_dir    = normalize_safe(force[eid].xyz);
    float  nbr_ph   = atan2(f_dir.y, f_dir.x);    // encode neighbor phase in force dir
    float  coupling = K_COUPLING * sin(nbr_ph - phase);
    float  new_ph   = phase + (OMEGA_0 + coupling) * DT;
    return new_ph - floor(new_ph / 6.2831853f) * 6.2831853f;   // wrap to [0, 2pi]
}

// ------------------------------------------------------------
// EXPERT  E2 -- Amplify
// Boost signal when velocity and force point the same direction.
// ------------------------------------------------------------
float expert_amplify(uint eid)
{
    float3 vel   = normalize_safe(velocity[eid].xyz);
    float3 f     = normalize_safe(force[eid].xyz);
    float  align = saturate(dot(vel, f));
    return signal[eid] * (1.0f + align * 0.5f);
}

// ------------------------------------------------------------
// EXPERT  E3 -- Compress (SCXQ2 lane quantization)
// Round-trip through 8-bit quantization.
// Range hint stored in event_params.xy by host each frame.
// ------------------------------------------------------------
float expert_compress(uint eid)
{
    float s     = signal[eid];
    float lo    = event_params[eid].x;
    float hi    = event_params[eid].y;
    float range = max(hi - lo, 1e-5f);
    float q     = round(saturate((s - lo) / range) * QUANT_BINS) / QUANT_BINS;
    return lo + q * range;
}

// ------------------------------------------------------------
// EXPERT  E4 -- Focus
// Project force onto principal axis; discard off-axis noise.
// ------------------------------------------------------------
float3 expert_focus(uint eid)
{
    float3 axis = normalize_safe(axes[eid].xyz);
    float3 f    = force[eid].xyz;
    return axis * max(dot(f, axis), 0.0f);
}

// ------------------------------------------------------------
// EXPERT  E5 -- Integrate (symplectic Euler)
// Applies force -> velocity -> position.  Writes pos + vel directly.
// ------------------------------------------------------------
void expert_integrate(uint eid)
{
    float3 pos = position[eid].xyz;
    float3 vel = velocity[eid].xyz;
    float3 f   = force[eid].xyz;

    vel = (vel + f * DT) * 0.98f;          // integrate + linear damping
    pos = pos + vel * DT;

    position[eid] = float4(pos, position[eid].w);
    velocity[eid] = float4(vel, velocity[eid].w);
}

// ------------------------------------------------------------
// EXPERT  E6 -- Pattern (n-gram frequency weight)
// event_params.w encodes a frequency hint set by the host
// (bigram count normalized to [0,1]).
// ------------------------------------------------------------
float expert_pattern(uint eid)
{
    float freq = saturate(event_params[eid].w);
    return signal[eid] * (1.0f + freq * 0.3f);
}

// ------------------------------------------------------------
// EXPERT  E7 -- Novelty (anomaly spike)
// Amplify signal that deviates > 2 sigma from the wave mean.
// ------------------------------------------------------------
float expert_novelty(uint eid, float mean, float std_dev)
{
    float s    = signal[eid];
    float dev  = abs(s - mean);
    float norm = (std_dev > 1e-5f) ? dev / std_dev : 0.0f;
    float spike = saturate(norm * 0.5f - 1.0f);    // > 2 sigma -> spike > 0
    return lerp(s, s * 2.5f, spike);
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
[RootSignature(ROOT_SIG)]
[numthreads(64, 1, 1)]
void main(
    uint3 DTid : SV_DispatchThreadID,
    uint3 GTid : SV_GroupThreadID
)
{
    // ── Init groupshared ──────────────────────────────────────
    if (GTid.x == 0)
    {
        gs_signal_bits    = asuint(0.0f);
        gs_signal_sq_bits = asuint(0.0f);
        gs_count          = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    uint slot     = DTid.x;
    uint my_count = expert_counts[ExpertId];

    if (slot >= my_count) return;

    uint  eid     = expert_lists[ExpertId * ListStride + slot];
    float sig_val = signal[eid];

    // ── Wave-level accumulation for stats ─────────────────────
    float wave_sum = WaveActiveSum(sig_val);
    float wave_sq  = WaveActiveSum(sig_val * sig_val);
    uint  wave_n   = WaveActiveCountBits(true);

    if (WaveIsFirstLane())
    {
        // Bias by +100 so negative signals map to positive uint range
        InterlockedAdd(gs_signal_bits,    (uint)((wave_sum    + 100.0f * (float)wave_n) * STAT_SCALE));
        InterlockedAdd(gs_signal_sq_bits, (uint)(max(wave_sq,  0.0f) * STAT_SCALE));
        InterlockedAdd(gs_count, wave_n);
    }
    GroupMemoryBarrierWithGroupSync();

    float n    = max((float)gs_count, 1.0f);
    float mean = ((float)gs_signal_bits / STAT_SCALE) / n - 100.0f;   // unbias
    float sq   = ((float)gs_signal_sq_bits / STAT_SCALE) / n;
    float std  = sqrt(max(sq - mean * mean, 0.0f));

    // ── Expert dispatch ───────────────────────────────────────
    float  new_signal = sig_val;
    float3 new_force  = force[eid].xyz;
    float  new_phase  = event_params[eid].w;

    switch (ExpertId)
    {
        case 0:  new_force  = force[eid].xyz + expert_geometry(eid);  break;
        case 1:  new_phase  = expert_temporal(eid);                    break;
        case 2:  new_signal = expert_amplify(eid);                     break;
        case 3:  new_signal = expert_compress(eid);                    break;
        case 4:  new_force  = expert_focus(eid);                       break;
        case 5:  expert_integrate(eid);                                break;
        case 6:  new_signal = expert_pattern(eid);                     break;
        case 7:  new_signal = expert_novelty(eid, mean, std);          break;
        default: break;
    }

    // ── Write outputs ─────────────────────────────────────────
    signal[eid]           = new_signal;
    force[eid]            = float4(new_force, 0.0f);
    event_params[eid].w   = new_phase;

    // Emit routing trace: packed (ExpertId | quantized_signal) for SCXQ2 proof
    uint packed = (ExpertId << 24) | (uint(saturate(new_signal) * 0xFFFFFF) & 0xFFFFFF);
    events[eid] = packed;
}
