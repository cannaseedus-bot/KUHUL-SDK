// ============================================================
// ORCHESTRATOR  (cs_6_0)
// Pass 2 of 3 — runs after the fused QKV/Force/MoE kernel.
//
// Reads events[] (expert routing) written by the fused kernel,
// uses wave intrinsics to compact entities per expert,
// then writes:
//   expert_counts[8]            — how many entities each expert owns
//   expert_lists[8 * N]         — compacted entity index lists
//   dispatch_args[8]            — DispatchIndirectArgs for ExecuteIndirect
//
// Compile: dxc -T cs_6_0 -E main -O3 orchestrate.hlsl -Fo orchestrate.cso
// ============================================================

#define EXPERTS      8
#define GROUP_SIZE   128
#define MAX_ENTITIES 65536

// ------------------------------------------------------------
// ROOT SIGNATURE
// ------------------------------------------------------------
#define ROOT_SIG \
    "RootConstants(b0, num32BitConstants=4), \
     SRV(t0), \
     SRV(t9), \
     UAV(u11), \
     UAV(u12), \
     UAV(u13)"

// ------------------------------------------------------------
// CONSTANTS
// ------------------------------------------------------------
cbuffer PassCB : register(b0)
{
    uint  EntityCount;   // total entities this dispatch
    uint  ExpertCount;   // always EXPERTS (8)
    uint  ListStride;    // per-expert list capacity = ceil(EntityCount/EXPERTS)*2
    uint  _pad0;
};

// ------------------------------------------------------------
// BUFFERS  (register layout mirrors fused kernel)
// ------------------------------------------------------------
StructuredBuffer<uint>            entities       : register(t0);  // entity IDs
StructuredBuffer<uint>            events         : register(t9);  // top-1 expert from fused pass

RWStructuredBuffer<uint>          expert_counts  : register(u11); // [8]
RWStructuredBuffer<uint>          expert_lists   : register(u12); // [8 * ListStride]
RWStructuredBuffer<uint3>         dispatch_args  : register(u13); // [8] DispatchIndirectArgs

// ------------------------------------------------------------
// GROUPSHARED  — per-wave compaction, then cross-wave merge
// GROUP_SIZE/WaveGetLaneCount() waves max; lane count can be 4
// on WARP so allocate GROUP_SIZE slots to be safe.
// ------------------------------------------------------------
groupshared uint gs_expert_count[EXPERTS];             // accumulator per expert
groupshared uint gs_expert_base[EXPERTS];               // global base offset (prefix sum)
groupshared uint gs_wave_offsets[EXPERTS][GROUP_SIZE];  // wave-local offsets

// ------------------------------------------------------------
// HELPERS
// ------------------------------------------------------------
uint ExpertIndex(uint entity_id)
{
    return events[entity_id] % EXPERTS;
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
[RootSignature(ROOT_SIG)]
[numthreads(GROUP_SIZE, 1, 1)]
void main(
    uint3 DTid : SV_DispatchThreadID,
    uint3 GTid : SV_GroupThreadID,
    uint3 Gid  : SV_GroupID
)
{
    uint tid    = DTid.x;
    uint ltid   = GTid.x;
    uint waveId = ltid / WaveGetLaneCount();

    // ── Init groupshared ──────────────────────────────────────
    if (ltid < EXPERTS)
    {
        gs_expert_count[ltid] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    // ── Classify this thread's entity ─────────────────────────
    bool  active    = (tid < EntityCount);
    uint  entity_id = active ? entities[tid] : 0;
    uint  eid       = active ? ExpertIndex(entity_id) : EXPERTS;  // EXPERTS = out-of-range

    // ── Wave-level compaction: count entities per expert ──────
    //    For each expert, ballot which lanes in this wave own it.
    [unroll]
    for (uint e = 0; e < EXPERTS; e++)
    {
        bool owns = (eid == e);

        // Lane offset within wave for this expert
        uint lane_offset = WavePrefixCountBits(owns);

        // Total in this wave
        uint wave_count  = WaveActiveCountBits(owns);

        // First lane of this expert grabs a slot in groupshared
        if (owns && lane_offset == 0)
        {
            uint gs_slot;
            InterlockedAdd(gs_expert_count[e], wave_count, gs_slot);
            gs_wave_offsets[e][waveId] = gs_slot;  // store base for this wave
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // ── Thread 0..7: build global prefix sums & write dispatch args ──
    if (ltid < EXPERTS)
    {
        uint e     = ltid;
        uint total = gs_expert_count[e];

        // Accumulate global offset across groups (atomic on u11)
        uint global_base;
        InterlockedAdd(expert_counts[e], total, global_base);
        gs_expert_base[e] = global_base;

        // Write DispatchIndirectArgs { ThreadGroupCountX, 1, 1 }
        uint groups = (total + 63) / 64;   // expert kernel uses [numthreads(64,1,1)]
        dispatch_args[e] = uint3(max(groups, 1), 1, 1);
    }

    GroupMemoryBarrierWithGroupSync();

    // ── Each active thread writes its entity into its expert list ──
    if (active && eid < EXPERTS)
    {
        // Wave-local position for this expert in this wave
        uint lane_offset = WavePrefixCountBits(eid == eid);   // position among same-expert lanes

        // Recompute properly per expert
        uint my_lane_pos = 0;
        [unroll]
        for (uint e = 0; e < EXPERTS; e++)
        {
            bool owns = (eid == e);
            my_lane_pos = owns ? WavePrefixCountBits(owns) : my_lane_pos;
        }

        uint base = gs_expert_base[eid] + gs_wave_offsets[eid][waveId] + my_lane_pos;
        uint slot = base % (ListStride * EXPERTS);   // guard against overflow

        expert_lists[eid * ListStride + (slot % ListStride)] = entity_id;
    }
}
