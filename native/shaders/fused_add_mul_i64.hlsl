// fused_add_mul_i64.hlsl

RWStructuredBuffer<int64_t> A   : register(u0);
RWStructuredBuffer<int64_t> B   : register(u1);
RWStructuredBuffer<int64_t> C   : register(u2);
RWStructuredBuffer<int64_t> OUT : register(u3);

cbuffer Params : register(b0)
{
    uint offsetA;
    uint offsetB;
    uint offsetC;
    uint offsetOut;
    uint count;
};

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= count) return;
    int64_t v = A[offsetA + i] + B[offsetB + i];
    OUT[offsetOut + i] = v * C[offsetC + i];
}
