// gpt2_qkv_split.hlsl — Split QKV buffer into per-head Q, K, V
// QKV: [seq, 3*embd] where embd = heads * head_dim
// Q, K, V: [seq, head_dim] for one head
// Dispatch(ceil(seq*head_dim/256), 1, 1)

cbuffer QKVSplitParams : register(b0) {
    uint seq_len;
    uint n_head;
    uint head_dim;   // embd / n_head
    uint head_idx;   // which head to extract
};

StructuredBuffer<float> qkv : register(t0);  // [seq, 3*embd]

RWStructuredBuffer<float> Q : register(u0);  // [seq, head_dim]
RWStructuredBuffer<float> K : register(u1);  // [seq, head_dim]
RWStructuredBuffer<float> V : register(u2);  // [seq, head_dim]

[numthreads(256, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    const uint total_dim = seq_len * head_dim;
    const uint i = tid.x;
    if (i >= total_dim) return;
    
    const uint pos = i / head_dim;      // sequence position
    const uint dim = i % head_dim;      // dimension within head
    
    const uint embd = n_head * head_dim;
    const uint qkv_offset = pos * 3 * embd;
    const uint head_offset = head_idx * head_dim;
    
    // Q: first embd dimensions
    Q[i] = qkv[qkv_offset + head_offset + dim];
    
    // K: second embd dimensions
    K[i] = qkv[qkv_offset + embd + head_offset + dim];
    
    // V: third embd dimensions
    V[i] = qkv[qkv_offset + 2*embd + head_offset + dim];
}
