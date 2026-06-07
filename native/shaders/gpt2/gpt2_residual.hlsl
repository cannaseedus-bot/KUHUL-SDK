// gpt2_residual.hlsl — Residual connection: result = x + y
// Dispatch(ceil(numel/256), 1, 1)  numthreads(256, 1, 1)

cbuffer ResidualParams : register(b0) {
    uint numel;
    uint3 pad;
};

StructuredBuffer<float>   x : register(t0);  // [numel] first input
StructuredBuffer<float>   y : register(t1);  // [numel] second input

RWStructuredBuffer<float> result : register(u0); // [numel] output (= x + y)

[numthreads(256, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= numel) return;
    
    result[i] = x[i] + y[i];
}
