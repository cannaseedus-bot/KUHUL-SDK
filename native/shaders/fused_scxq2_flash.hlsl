// Stub fused SCXQ2 + FlashAttention kernel (compile with dxc)
// Actual SCXQ2 decode and flash attention math to be completed.

cbuffer Scales : register(b0)
{
    float scale;
    float zero;
};

StructuredBuffer<uint>  W_packed : register(t0); // packed int4 weights (2 vals/byte packed into uint32 stream)
StructuredBuffer<float> X        : register(t1); // activations
RWStructuredBuffer<float> Out    : register(u0);

// Simple int4 decode (2 vals per byte). Replace with groupwise scale/zero if available.
float2 decode_int4(uint packed, uint idx)
{
    // idx: which uint32 lane
    uint v = W_packed[idx];
    float a = ((v >> 0) & 0xF);
    float b = ((v >> 4) & 0xF);
    a = (a - zero) * scale;
    b = (b - zero) * scale;
    return float2(a, b);
}

// Placeholder flash-attn: computes dot for a tiny tile.
[numthreads(128,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint lane = tid.x;
    float2 w = decode_int4(0, lane); // packed lane
    // Toy compute: Out = sum(w) * X[lane]
    float x = X[lane];
    Out[lane] = (w.x + w.y) * x;
    // TODO: replace with tiled QKV matmul + softmax for real flash attention.
}
