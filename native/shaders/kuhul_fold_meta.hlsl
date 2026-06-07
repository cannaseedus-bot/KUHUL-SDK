/**
 * @file kuhul_fold_meta.hlsl
 * @brief META_FOLD kernel — VM-2 per-token step hash, router logits hash,
 *        Merkle root verification, chain_hash accumulation
 *
 * Fold:      ⟁META_FOLD⟁
 * Micronaut: VM-2 (VerificationMicronaut)
 * Lane:      DICT
 * Nodes:     100 (z-layers 7–8 of 10×10×10 grid)
 *   64 nodes (z-layer 7): step_hash / chain_hash accumulators
 *   36 nodes (z-layer 8): Merkle proof verifier
 * Dispatch:  100 thread groups × 64 threads
 *
 * Per-token proof pipeline:
 *   1. Receive router_logits_i8 from COMPUTE_FOLD (arc_CF_MF)
 *   2. SHA-256(router_logits_i8[token]) → router_logits_hash
 *   3. SHA-256(trunk_output[token])     → token_content_hash
 *   4. step_hash[token] = SHA-256(router_logits_hash ‖ token_content_hash ‖ token_pos)
 *   5. chain_hash = SHA-256(chain_hash_prev ‖ step_hash[token])   ← rolling proof chain
 *   6. Receive sealed snapshot hash from STORAGE_FOLD (arc_SF_MF)
 *   7. Verify snapshot_hash ∈ MerkleRoots[trunk|router|expert0-8]
 *   8. Emit verified chain_hash → COMPUTE_FOLD feedback (arc_MF_CF_feedback)
 *
 * Verifier rules enforced here:
 *   V7: all proofs include abi_hash + policy_hash + meta_hash
 *   V6: replay determinism — step_hash must be identical given same inputs
 *   V2: all mutations require explicit control gate records
 *
 * authority="none" — VM-2 observes and attests but cannot block execution.
 */

// ============================================================================
// Resource Bindings
// ============================================================================

StructuredBuffer<int>   RouterLogitsI8   : register(t0); // From COMPUTE_FOLD, seq × 9 INT8
StructuredBuffer<float> TrunkOutput      : register(t1); // From COMPUTE_FOLD, seq × 1024
StructuredBuffer<uint>  SnapshotSeal     : register(t2); // From STORAGE_FOLD, 100 × 8 words
StructuredBuffer<uint>  MerkleRoots      : register(t3); // 11 roots × 8 words (trunk+router+expert0-8)
StructuredBuffer<uint>  ControlFlags     : register(t4); // CM-1 gate
StructuredBuffer<uint>  AbiHash          : register(t5); // ABI contract hash (8 words, fixed)
StructuredBuffer<uint>  PolicyHash       : register(t6); // Policy engine hash (8 words, from pool eval)
StructuredBuffer<uint>  PrevChainHash    : register(t7); // chain_hash from previous token (8 words)

RWStructuredBuffer<uint> StepHashes      : register(u0); // seq × 8 words per step
RWStructuredBuffer<uint> ChainHash       : register(u1); // Rolling chain: 8 words (latest)
RWStructuredBuffer<uint> MetaHash        : register(u2); // SHA-256(abi_hash ‖ policy_hash ‖ chain_hash)
RWStructuredBuffer<uint> MerkleProofLog  : register(u3); // Per-token Merkle verify result (0/1)
RWStructuredBuffer<uint> AttestationLog  : register(u4); // Full V7 attestation record

cbuffer MetaFoldParams : register(b0)
{
    uint  node_id;       // 0–99 (0–63 = step_hash nodes, 64–99 = merkle nodes)
    uint  seq_len;       // current sequence length
    uint  num_experts;   // 9
    uint  num_roots;     // 11 (trunk + router + 9 experts)
    uint  token_pos;     // which token position this dispatch covers
    uint  cm1_gate;      // 0x0002
    uint  _pad0;
    uint  _pad1;
};

// ============================================================================
// SHA-256 (reused from storage fold — defined inline here for self-containment)
// ============================================================================

static const uint SHA256_K_M[64] = {
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

uint RotR_M(uint x, uint n) { return (x >> n) | (x << (32u - n)); }

void SHA256_IV(out uint h[8])
{
    h[0] = 0x6a09e667u; h[1] = 0xbb67ae85u;
    h[2] = 0x3c6ef372u; h[3] = 0xa54ff53au;
    h[4] = 0x510e527fu; h[5] = 0x9b05688cu;
    h[6] = 0x1f83d9abu; h[7] = 0x5be0cd19u;
}

void SHA256_Block(uint w[16], inout uint h[8])
{
    uint sched[64];
    for (uint i = 0; i < 16; i++) sched[i] = w[i];
    for (uint i = 16; i < 64; i++)
    {
        uint s0 = RotR_M(sched[i-15], 7) ^ RotR_M(sched[i-15], 18) ^ (sched[i-15] >> 3);
        uint s1 = RotR_M(sched[i-2], 17) ^ RotR_M(sched[i-2], 19)  ^ (sched[i-2]  >> 10);
        sched[i] = sched[i-16] + s0 + sched[i-7] + s1;
    }
    uint a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (uint i = 0; i < 64; i++)
    {
        uint S1    = RotR_M(e, 6) ^ RotR_M(e, 11) ^ RotR_M(e, 25);
        uint ch    = (e & f) ^ (~e & g);
        uint temp1 = hh + S1 + ch + SHA256_K_M[i] + sched[i];
        uint S0    = RotR_M(a, 2) ^ RotR_M(a, 13) ^ RotR_M(a, 22);
        uint maj   = (a & b) ^ (a & c) ^ (b & c);
        uint temp2 = S0 + maj;
        hh=g; g=f; f=e; e=d+temp1;
        d=c;  c=b; b=a; a=temp1+temp2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d;
    h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
}

/** Hash exactly N uint32 words using SHA-256 */
void HashWords(uint words[16], uint count, out uint digest[8])
{
    SHA256_IV(digest);
    uint block[16];
    for (uint i = 0; i < 16; i++)
        block[i] = (i < count) ? words[i] : 0u;
    block[count] = 0x80000000u;
    block[15]    = count * 32u;
    SHA256_Block(block, digest);
}

/** Concatenate two 8-word hashes and hash them (SHA-256 of 512-bit input) */
void HashConcat8x8(uint a[8], uint b[8], out uint digest[8])
{
    uint block[16];
    for (uint i = 0; i < 8; i++) block[i]   = a[i];
    for (uint i = 0; i < 8; i++) block[i+8] = b[i];
    SHA256_IV(digest);
    SHA256_Block(block, digest);
}

/** Concatenate three 8-word hashes (fits in two SHA-256 blocks) */
void HashConcat8x8x8(uint a[8], uint b[8], uint c[8], out uint digest[8])
{
    // First block: a ‖ b (256 bits)
    uint mid[8];
    HashConcat8x8(a, b, mid);
    // Second block: mid ‖ c (256 bits)
    HashConcat8x8(mid, c, digest);
}

// ============================================================================
// Merkle Proof: verify leaf_hash ∈ MerkleRoots
// Returns 1 if any root matches the leaf, 0 otherwise.
// (Flat check — full Merkle path not stored here, just root comparison.)
// ============================================================================

uint MerkleVerify(uint leaf[8])
{
    for (uint r = 0; r < num_roots; r++)
    {
        uint match = 1u;
        for (uint w = 0; w < 8; w++)
        {
            if (MerkleRoots[r * 8 + w] != leaf[w])
            {
                match = 0u;
                break;
            }
        }
        if (match) return 1u;
    }
    return 0u;
}

// ============================================================================
// Shared memory
// ============================================================================

groupshared uint gs_router_hash[8];    // SHA-256(router_logits_i8[token])
groupshared uint gs_content_hash[8];   // SHA-256(trunk_output[token])
groupshared uint gs_step_hash[8];      // step_hash[token] = SHA-256(router‖content‖pos)
groupshared uint gs_chain_new[8];      // chain_hash updated for this token
groupshared uint gs_snap_hash[8];      // snapshot seal hash for this token's slot
groupshared uint gs_merkle_ok;         // 1 = snapshot in Merkle tree

// ============================================================================
// Main META_FOLD Kernel
// ============================================================================

[numthreads(64, 1, 1)]
void CS_MetaFold(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    // CM-1 gate check
    if (ControlFlags[0] != cm1_gate)
        return;

    if (node_id >= 100)
        return;

    bool is_hash_node   = (node_id < 64);
    bool is_merkle_node = (node_id >= 64);

    // -----------------------------------------------------------------------
    // STEP-HASH NODES (0–63): compute per-token step_hash and chain_hash
    // -----------------------------------------------------------------------
    if (is_hash_node)
    {
        // Stage A: SHA-256 of router logits INT8 for this token
        // router_logits_i8 is seq × 9 ints; pack 9 values into one 512-bit block
        if (tid == 0)
        {
            uint block[16];
            for (uint e = 0; e < 9; e++)
                block[e] = uint(RouterLogitsI8[token_pos * 9 + e] & 0xFF);
            for (uint p = 9; p < 15; p++) block[p] = 0u;
            block[9]  = 0x80000000u;
            block[15] = 9u * 8u;  // 9 bytes in bits

            SHA256_IV(gs_router_hash);
            SHA256_Block(block, gs_router_hash);
        }
        GroupMemoryBarrierWithGroupSync();

        // Stage B: SHA-256 of trunk output for this token (1024 floats = 32 × 16-word blocks)
        // Each of 64 hash-nodes processes one 16-word block of trunk output
        {
            uint block_idx  = tid;          // each thread covers one 16-word chunk
            uint chunk_base = token_pos * 1024u + block_idx * 16u;

            uint local_h[8];
            SHA256_IV(local_h);

            if (chunk_base + 16u <= uint(seq_len) * 1024u)
            {
                uint block[16];
                for (uint w = 0; w < 16; w++)
                    block[w] = asuint(TrunkOutput[chunk_base + w]);
                SHA256_Block(block, local_h);
            }

            // Thread 0 accumulates all chunk hashes (simplified: XOR fold)
            // Real: Merkle tree over chunks; here we chain all 64 hashes serially.
            // Using gs_content_hash updated sequentially per thread is not
            // GPU-safe for arbitrary order — use atomic XOR as approximation.
            for (uint w = 0; w < 8; w++)
                InterlockedXor(gs_content_hash[w], local_h[w]);
        }
        GroupMemoryBarrierWithGroupSync();

        // Stage C: step_hash = SHA-256(router_hash ‖ content_hash ‖ token_pos_word)
        if (tid == 0)
        {
            // Combine: pack 8+8+1 = 17 words into two SHA-256 blocks
            uint h_mid[8];
            HashConcat8x8(gs_router_hash, gs_content_hash, h_mid);

            uint pos_block[16];
            for (uint w = 0; w < 8; w++)  pos_block[w]   = h_mid[w];
            for (uint w = 0; w < 7; w++)  pos_block[w+8] = 0u;
            pos_block[8]  = token_pos;    // token position as nonce
            pos_block[9]  = 0x80000000u;
            pos_block[15] = 17u * 32u;

            SHA256_IV(gs_step_hash);
            SHA256_Block(pos_block, gs_step_hash);

            // Write step_hash to output
            uint step_base = token_pos * 8u;
            for (uint w = 0; w < 8; w++)
                StepHashes[step_base + w] = gs_step_hash[w];
        }
        GroupMemoryBarrierWithGroupSync();

        // Stage D: chain_hash = SHA-256(prev_chain_hash ‖ step_hash)
        if (tid == 0)
        {
            uint prev[8], step[8];
            for (uint w = 0; w < 8; w++)
            {
                prev[w] = PrevChainHash[w];
                step[w] = gs_step_hash[w];
            }
            HashConcat8x8(prev, step, gs_chain_new);

            for (uint w = 0; w < 8; w++)
                ChainHash[w] = gs_chain_new[w];
        }
        GroupMemoryBarrierWithGroupSync();

        // Stage E: V7 meta_hash = SHA-256(abi_hash ‖ policy_hash ‖ chain_hash)
        if (tid == 0)
        {
            uint abi[8], pol[8], chain[8], meta[8];
            for (uint w = 0; w < 8; w++)
            {
                abi[w]   = AbiHash[w];
                pol[w]   = PolicyHash[w];
                chain[w] = gs_chain_new[w];
            }
            HashConcat8x8x8(abi, pol, chain, meta);
            for (uint w = 0; w < 8; w++)
                MetaHash[w] = meta[w];
        }
    }

    // -----------------------------------------------------------------------
    // MERKLE NODES (64–99): verify snapshot seal ∈ Merkle tree
    // Each Merkle node handles one snapshot slot (up to 36 simultaneous).
    // -----------------------------------------------------------------------
    else if (is_merkle_node)
    {
        uint snap_slot = node_id - 64u;

        // Load snapshot seal from STORAGE_FOLD
        if (tid == 0)
        {
            for (uint w = 0; w < 8; w++)
                gs_snap_hash[w] = SnapshotSeal[snap_slot * 8u + w];

            gs_merkle_ok = MerkleVerify(gs_snap_hash);

            // Log result per token
            MerkleProofLog[snap_slot] = gs_merkle_ok;
        }
        GroupMemoryBarrierWithGroupSync();

        // Emit attestation record (V7: abi_hash ‖ policy_hash ‖ meta_hash present)
        if (tid == 0 && gs_merkle_ok == 1u)
        {
            uint attest_base = snap_slot * 32u;  // 4 × 8-word fields

            // field 0: abi_hash
            for (uint w = 0; w < 8; w++)
                AttestationLog[attest_base + w]      = AbiHash[w];
            // field 1: policy_hash
            for (uint w = 0; w < 8; w++)
                AttestationLog[attest_base + 8 + w]  = PolicyHash[w];
            // field 2: snapshot seal hash (what was verified)
            for (uint w = 0; w < 8; w++)
                AttestationLog[attest_base + 16 + w] = gs_snap_hash[w];
            // field 3: current chain_hash (live proof linkage)
            for (uint w = 0; w < 8; w++)
                AttestationLog[attest_base + 24 + w] = ChainHash[w];
        }
    }
}
