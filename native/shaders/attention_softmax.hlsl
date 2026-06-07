// attention_softmax.hlsl
// Compute softmax over dot(Q, K) for single-head attention
// Assumes: Q: [Lq, D]  (Lq=64, D=64), K: [Lk, D] (Lk=64, D=64)
// Output: P: [Lq, Lk]

Texture2D<float> texQ : register(t0);   // Q as 2D texture or buffer (row-major)
Texture2D<float> texK : register(t1);
RWTexture2D<float> outP : register(u0);

cbuffer Constants : register(b0) {
    uint Lq; // 64
    uint Lk; // 64
    uint D;  // 64
    float scale; // 1/sqrt(D)
}

// We dispatch with [Lq, 1, 1] thread groups and [64,1,1] threads per group
[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID, uint3 gidx : SV_GroupID, uint3 gt : SV_GroupThreadID) {
    uint q = gidx.x; // which query row
    uint k = gt.x;   // thread computes partial dot for this k across D by striding
    // We'll compute full dot per thread by looping D with striding of thread count (64)

    // Compute dot products into shared memory
    groupshared float partial[64]; // one per k
    float dot = 0.0f;
    // Each thread computes dot(Q[q,:], K[k,:]) by iterating d
    for (uint d = k; d < D; d += 64) {
        float qv = texQ.Load(int3(d, q, 0)); // depends on how Q stored; adapt indexing
        float kv = texK.Load(int3(d, k, 0));
        dot += qv * kv;
    }
    // reduce across threads in group to get full dot for this k
    // simple binary tree reduction across gt.x - but here gt.x is up to 64
    // store partial in shared
    partial[k] = dot;
    GroupMemoryBarrierWithGroupSync();

    // Now thread 0 in group computes softmax over all k using partial[0..Lk-1]
    if (gt.x == 0) {
        // find max
        float mx = -3.4e38f;
        for (uint i = 0; i < Lk; ++i) mx = max(mx, partial[i] * scale);
        // compute exp and sum
        float sum = 0.0f;
        float tmp[64];
        for (uint i = 0; i < Lk; ++i) { tmp[i] = exp(partial[i] * scale - mx); sum += tmp[i]; }
        // write normalized probs to outP at row q
        for (uint i = 0; i < Lk; ++i) {
            float p = tmp[i] / sum;
            outP[int2(i, q)] = p; // (x=col k, y=row q)
        }
    }
}

// NOTE: This shader is a starting point. For correctness and performance:
// - Store Q and K in structured buffers or SRVs with proper layout
// - Use shared memory reductions and parallel sums
// - Consider splitting dot computation across threads differently
// - Use FP16 where possible and native instructions for faster throughput
