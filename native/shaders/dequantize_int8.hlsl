// dequantize_int8.hlsl
// Dequantize packed INT8 -> float outputs
// Inputs:
//   StructuredBuffer<uint> IN_PACKED   : register(t0);
//   StructuredBuffer<float> IN_SCALE    : register(t1);
// Outputs:
//   RWStructuredBuffer<float> OUT       : register(u0);

[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint idx4 = tid.x;
    uint packed = IN_PACKED[idx4];
    float scale = IN_SCALE[idx4];

    int qx = (packed << 24) >> 24;
    int qy = (packed << 16) >> 24;
    int qz = (packed << 8)  >> 24;
    int qw = packed >> 24;

    uint base = idx4 * 4;
    OUT[base + 0] = (float)qx * scale;
    OUT[base + 1] = (float)qy * scale;
    OUT[base + 2] = (float)qz * scale;
    OUT[base + 3] = (float)qw * scale;
}
