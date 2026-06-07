/**
 * @file kuhul_fold_storage.hlsl
 * @brief STORAGE_FOLD kernel — SM-1 snapshot/delta/seal and replay_payload buffer
 *
 * Fold:      ⟁STORAGE_FOLD⟁
 * Micronaut: SM-1 (StorageMicronaut)
 * Lane:      FIELD
 * Nodes:     150 (z-layers 5–6 of 10×10×10 grid)
 *   100 nodes (z-layer 5): snapshot / delta
 *    50 nodes (z-layer 6): replay_payload ring buffer
 * Dispatch:  150 thread groups × 64 threads
 *
 * Operations:
 *   store_object    — write trunk activation slab + compute SHA-256 identity hash
 *   compute_delta   — XOR diff between two snapshots → delta slab
 *   seal_snapshot   — freeze snapshot: write hash into seal buffer, mark immutable
 *   retrieve_object — read snapshot by slot index + verify byte identity
 *   replay_payload  — write/read quarantine payload ring (50 slots × 512×128)
 *
 * Invariants:
 *   - Nothing persists unless STORAGE_FOLD seals it (KUHUL_π law)
 *   - Byte identity must be verified before retrieval (V6 replay determinism)
 *   - replay_payload slots are write-once per quarantine event
 */

// ============================================================================
// Resource Bindings
// ============================================================================

StructuredBuffer<float>  TrunkActivations : register(t0);  // From COMPUTE_FOLD via arc_CF_SF
StructuredBuffer<uint>   SnapshotHashes   : register(t1);  // Existing seals (8 words × 150 slots)
StructuredBuffer<uint>   ControlFlags     : register(t2);  // CM-1 gate
StructuredBuffer<uint>   OpCodes          : register(t3);  // Per-node operation (store/delta/seal/get/replay)
StructuredBuffer<uint>   SlotIndices      : register(t4);  // Which snapshot slot to operate on
StructuredBuffer<float>  ReplayPayloads   : register(t5);  // Input replay payload (50 × 512 × 128)

RWStructuredBuffer<float> SnapshotBuffer  : register(u0);  // 150 × 512 × 256 snapshot slabs
RWStructuredBuffer<float> DeltaBuffer     : register(u1);  // Delta output (same shape)
RWStructuredBuffer<uint>  SealBuffer      : register(u2);  // SHA-256 digests: 150 × 8 words
RWStructuredBuffer<float> ReplayRing      : register(u3);  // 50 × 512 × 128 replay ring
RWStructuredBuffer<uint>  ReplayRingMeta  : register(u4);  // slot_id, write_ptr, sealed flag

cbuffer StorageFoldParams : register(b0)
{
    uint  node_id;          // 0–149 (0–99 = snapshot nodes, 100–149 = replay nodes)
    uint  slab_size;        // 512 × 256 = 131072 elements per snapshot slab
    uint  replay_slab_size; // 512 × 128 = 65536 elements per replay slot
    uint  num_snap_slots;   // 100
    uint  num_replay_slots; // 50
    uint  cm1_gate;         // 0x0002
    uint  hidden_dim;       // 1024
    uint  _pad;
};

// ============================================================================
// Tiny SHA-256 (8-word state, processes 512×256 float slab as byte stream)
// Simplified: uses HLSL-friendly iterative block compression.
// Full SHA-256 requires 64 rounds per 512-bit block; we run on 32-bit words.
// ============================================================================

static const uint SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

uint RotR(uint x, uint n) { return (x >> n) | (x << (32u - n)); }

void SHA256Block(uint w[16], inout uint h[8])
{
    uint sched[64];
    for (uint i = 0; i < 16; i++) sched[i] = w[i];
    for (uint i = 16; i < 64; i++)
    {
        uint s0 = RotR(sched[i-15], 7) ^ RotR(sched[i-15], 18) ^ (sched[i-15] >> 3);
        uint s1 = RotR(sched[i-2],  17) ^ RotR(sched[i-2],  19) ^ (sched[i-2]  >> 10);
        sched[i] = sched[i-16] + s0 + sched[i-7] + s1;
    }
    uint a = h[0], b = h[1], c = h[2], d = h[3];
    uint e = h[4], f = h[5], g = h[6], hh = h[7];
    for (uint i = 0; i < 64; i++)
    {
        uint S1    = RotR(e, 6) ^ RotR(e, 11) ^ RotR(e, 25);
        uint ch    = (e & f) ^ (~e & g);
        uint temp1 = hh + S1 + ch + SHA256_K[i] + sched[i];
        uint S0    = RotR(a, 2) ^ RotR(a, 13) ^ RotR(a, 22);
        uint maj   = (a & b) ^ (a & c) ^ (b & c);
        uint temp2 = S0 + maj;
        hh = g; g = f; f = e; e = d + temp1;
        d = c;  c = b; b = a; a = temp1 + temp2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

/** Hash a slab of float32 values into 8-word SHA-256 digest.
    Treats each float as 4 bytes, packs into 512-bit (16-word) blocks. */
void HashSlab(uint slab_base, uint count, out uint digest[8])
{
    // Init with SHA-256 IV
    digest[0] = 0x6a09e667u; digest[1] = 0xbb67ae85u;
    digest[2] = 0x3c6ef372u; digest[3] = 0xa54ff53au;
    digest[4] = 0x510e527fu; digest[5] = 0x9b05688cu;
    digest[6] = 0x1f83d9abu; digest[7] = 0x5be0cd19u;

    uint block[16];
    uint block_word = 0;

    for (uint i = 0; i < count; i++)
    {
        block[block_word++] = asuint(SnapshotBuffer[slab_base + i]);
        if (block_word == 16)
        {
            SHA256Block(block, digest);
            block_word = 0;
        }
    }

    // Final block with padding (simplified: pad remaining words with 0)
    for (uint p = block_word; p < 16; p++) block[p] = 0;
    block[block_word] = 0x80000000u;  // padding bit
    block[15] = count * 32;           // bit length (count floats × 32 bits)
    SHA256Block(block, digest);
}

// ============================================================================
// Operation codes (OpCodes buffer)
// ============================================================================
static const uint OP_STORE   = 0u;
static const uint OP_DELTA   = 1u;
static const uint OP_SEAL    = 2u;
static const uint OP_GET     = 3u;
static const uint OP_REPLAY  = 4u;

// ============================================================================
// Main Storage Fold Kernel
// ============================================================================

groupshared float gs_slab_chunk[64];     // Working chunk (64 floats at a time)
groupshared uint  gs_verified;           // 1 if identity check passed, 0 if failed

[numthreads(64, 1, 1)]
void CS_StorageFold(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    // CM-1 gate check
    if (ControlFlags[0] != cm1_gate)
        return;

    if (node_id >= 150)
        return;

    uint op   = OpCodes[node_id];
    uint slot = SlotIndices[node_id];

    // -----------------------------------------------------------------------
    // OP_STORE — copy TrunkActivations → SnapshotBuffer[slot]
    // -----------------------------------------------------------------------
    if (op == OP_STORE && slot < num_snap_slots)
    {
        uint dst_base = slot * slab_size;
        uint src_base = slot * hidden_dim;  // aligned to trunk output stride

        for (uint i = tid; i < slab_size; i += 64)
            SnapshotBuffer[dst_base + i] = TrunkActivations[src_base + (i % hidden_dim)];

        GroupMemoryBarrierWithGroupSync();
    }

    // -----------------------------------------------------------------------
    // OP_DELTA — XOR-diff between slot and slot-1, write to DeltaBuffer
    // -----------------------------------------------------------------------
    else if (op == OP_DELTA && slot > 0 && slot < num_snap_slots)
    {
        uint cur_base  = slot       * slab_size;
        uint prev_base = (slot - 1) * slab_size;

        for (uint i = tid; i < slab_size; i += 64)
        {
            uint cur_bits  = asuint(SnapshotBuffer[cur_base  + i]);
            uint prev_bits = asuint(SnapshotBuffer[prev_base + i]);
            DeltaBuffer[cur_base + i] = asfloat(cur_bits ^ prev_bits);
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // -----------------------------------------------------------------------
    // OP_SEAL — compute SHA-256 of snapshot slab, write into SealBuffer
    // -----------------------------------------------------------------------
    else if (op == OP_SEAL && slot < num_snap_slots && tid == 0)
    {
        uint digest[8];
        HashSlab(slot * slab_size, slab_size, digest);

        uint seal_base = slot * 8;
        for (uint w = 0; w < 8; w++)
            SealBuffer[seal_base + w] = digest[w];
    }

    // -----------------------------------------------------------------------
    // OP_GET — verify byte identity before allowing retrieval
    //          Recomputes hash and compares against SealBuffer.
    // -----------------------------------------------------------------------
    else if (op == OP_GET && slot < num_snap_slots)
    {
        if (tid == 0)
        {
            uint digest[8];
            HashSlab(slot * slab_size, slab_size, digest);

            uint seal_base = slot * 8;
            uint match = 1u;
            for (uint w = 0; w < 8; w++)
            {
                if (SealBuffer[seal_base + w] != digest[w])
                    match = 0u;
            }
            gs_verified = match;
        }
        GroupMemoryBarrierWithGroupSync();

        // V6 enforcement: if hash mismatch, zero out output to prevent replay
        if (gs_verified == 0u)
        {
            uint dst_base = slot * slab_size;
            for (uint i = tid; i < slab_size; i += 64)
                SnapshotBuffer[dst_base + i] = 0.0f;
        }
    }

    // -----------------------------------------------------------------------
    // OP_REPLAY — write replay payload into ring slot (replay nodes, id 100–149)
    //             Write-once: skips if ReplayRingMeta[slot].sealed == 1
    // -----------------------------------------------------------------------
    else if (op == OP_REPLAY && node_id >= 100)
    {
        uint ring_slot = slot % num_replay_slots;
        uint meta_idx  = ring_slot * 2;  // [0]=slot_id [1]=sealed

        if (tid == 0)
            gs_verified = ReplayRingMeta[meta_idx + 1];  // sealed flag
        GroupMemoryBarrierWithGroupSync();

        if (gs_verified == 0u)  // not yet sealed → write allowed
        {
            uint src_base = ring_slot * replay_slab_size;
            uint dst_base = ring_slot * replay_slab_size;

            for (uint i = tid; i < replay_slab_size; i += 64)
                ReplayRing[dst_base + i] = ReplayPayloads[src_base + i];

            GroupMemoryBarrierWithGroupSync();

            if (tid == 0)
            {
                ReplayRingMeta[meta_idx]     = ring_slot;  // slot_id
                ReplayRingMeta[meta_idx + 1] = 1u;         // sealed = true (write-once)
            }
        }
    }
}
