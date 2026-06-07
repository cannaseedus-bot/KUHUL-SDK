// ============================================================
// ORCHESTRATOR  (cs_5_1 fallback — no wave intrinsics)
//
// Each active thread atomically grabs a unique slot in its
// expert's list.  dispatch_args are set conservatively by the
// C++ host (all experts get ceil(EntityCount/64) groups).
//
// Compile: dxc -T cs_5_1 -E main -O3 orchestrate_51.hlsl -Fo orchestrate_51.cso
// ============================================================

#define EXPERTS    8
#define GROUP_SIZE 128

#define ROOT_SIG \
    "RootConstants(b0, num32BitConstants=4), \
     SRV(t0), \
     SRV(t9), \
     UAV(u11), \
     UAV(u12), \
     UAV(u13)"

cbuffer PassCB : register(b0)
{
    uint EntityCount;
    uint ExpertCount;
    uint ListStride;
    uint _pad0;
};

StructuredBuffer<uint>    entities      : register(t0);
StructuredBuffer<uint>    events        : register(t9);
RWStructuredBuffer<uint>  expert_counts : register(u11);
RWStructuredBuffer<uint>  expert_lists  : register(u12);
RWStructuredBuffer<uint3> dispatch_args : register(u13);

[RootSignature(ROOT_SIG)]
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint tid = DTid.x;
    if (tid >= EntityCount) return;

    uint entity_id = entities[tid];
    uint eid       = events[entity_id] % EXPERTS;

    // Atomic slot grab — O(1) per thread, no wave ops needed
    uint slot;
    InterlockedAdd(expert_counts[eid], 1, slot);

    if (slot < ListStride)
        expert_lists[eid * ListStride + slot] = entity_id;

    // dispatch_args written by C++ host after this pass completes
}
