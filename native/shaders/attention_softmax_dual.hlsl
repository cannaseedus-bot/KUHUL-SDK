// attention_softmax_dual.hlsl
// Dual-head compute softmax over dot(Q, K) for two heads in parallel
// Assumes per-head dims: Lq=64, Lk=64, Dh=64. Total D = H * Dh (H=2)
// Outputs: P0 and P1 matrices (Lq x Lk) for head0 and head1

Texture2D<float> texQ : register(t0);   // Q laid out as [d, q] for head-major packing
Texture2D<float> texK : register(t1);
RWTexture2D<float> outP0 : register(u0);
RWTexture2D<float> outP1 : register(u1);

cbuffer Constants : register(b0) {
    uint Lq; // 64
    uint Lk; // 64
    uint Dh; // 64 (per head)
    float scale; // 1/sqrt(Dh)
}

// Threading: one group per query row. Each group has 64 threads (one per k index)
[numthreads(64, 1, 1)]
void CSMain(uint3 DTid : SV_DispatchThreadID, uint3 Gid : SV_GroupID, uint3 GTid : SV_GroupThreadID) {
    uint q = Gid.x;    // query row
    uint k = GTid.x;   // key column handled by this thread

    // Shared memory for partial dot products per head
    groupshared float partial0[64];
    groupshared float partial1[64];

    float dot0 = 0.0f;
    float dot1 = 0.0f;

    // iterate over Dh, striding by threadcount (64)
    for (uint d = k; d < Dh; d += 64) {
        // index layout assumption: tex.Load(int3(x=d + head*Dh, y=q, z=0))
        float q0 = texQ.Load(int3(d + 0 * Dh, q, 0));
        float k0v = texK.Load(int3(d + 0 * Dh, k, 0));
        dot0 += q0 * k0v;
        float q1 = texQ.Load(int3(d + 1 * Dh, q, 0));
        float k1v = texK.Load(int3(d + 1 * Dh, k, 0));
        dot1 += q1 * k1v;
    }

    partial0[k] = dot0;
    partial1[k] = dot1;
    GroupMemoryBarrierWithGroupSync();

    // thread 0 computes softmax for both heads across k=0..Lk-1
    if (k == 0) {
        // head0 softmax
        float mx0 = -3.4e38f;
        for (uint i = 0; i < Lk; ++i) mx0 = max(mx0, partial0[i] * scale);
        float sum0 = 0.0f;
        float tmp0[64];
        for (uint i = 0; i < Lk; ++i) { tmp0[i] = exp(partial0[i] * scale - mx0); sum0 += tmp0[i]; }

        // head1 softmax
        float mx1 = -3.4e38f;
        for (uint i = 0; i < Lk; ++i) mx1 = max(mx1, partial1[i] * scale);
        float sum1 = 0.0f;
        float tmp1[64];
        for (uint i = 0; i < Lk; ++i) { tmp1[i] = exp(partial1[i] * scale - mx1); sum1 += tmp1[i]; }

        // write normalized probabilities for both heads for row q
        for (uint i = 0; i < Lk; ++i) {
            float p0 = tmp0[i] / sum0;
            float p1 = tmp1[i] / sum1;
            outP0[int2(i, q)] = p0; // x=col k, y=row q
            outP1[int2(i, q)] = p1;
        }
    }
}

// Notes:
// - This shader computes softmax per query row for two heads in parallel. It assumes texQ/texK
//   are packed so that head 0 occupies d=[0..Dh-1] and head1 occupies d=[Dh..2*Dh-1].
// - For performance, replace tex.Load with StructuredBuffer reads and tune threadgroup sizes.
// - Use fp16 and native dot-product instructions where supported to reduce memory and compute.
