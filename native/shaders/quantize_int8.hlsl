// quantize_int8.hlsl
// Per-tensor symmetric INT8 quantization kernel (GPU)
// Inputs:
//   RWStructuredBuffer<float> IN         : register(t0);  // input activations
// Outputs:
//   RWStructuredBuffer<uint> OUT_PACKED   : register(u0);  // packed 4x int8 per uint
//   RWStructuredBuffer<float> OUT_SCALE   : register(u1);  // one scale per 4-element group

[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint idx4 = tid.x;
    uint base = idx4 * 4;

    float4 v = float4(0,0,0,0);
    v.x = IN[base + 0];
    v.y = IN[base + 1];
    v.z = IN[base + 2];
    v.w = IN[base + 3];

    float4 av = abs(v);
    float maxv = max(max(av.x, av.y), max(av.z, av.w));
    float scale = maxv / 127.0 + 1e-6;

    int4 q = int4(round(v / scale));
    q = clamp(q, -127, 127);

    uint packed = (uint)((q.x & 0xFF)) | ((uint)(q.y & 0xFF) << 8) | ((uint)(q.z & 0xFF) << 16) | ((uint)(q.w & 0xFF) << 24);

    OUT_PACKED[idx4] = packed;
    OUT_SCALE[idx4] = scale;
}
