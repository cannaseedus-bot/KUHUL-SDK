// HLSL quantized decompress shader (decompress_quant.hlsl)
// Input: ByteAddressBuffer quantData (uint8 quantized values)
// Output: RWStructuredBuffer<float> outWeights

RWStructuredBuffer<float> outWeights : register(u0);
ByteAddressBuffer quantData : register(t0);

cbuffer Dequant : register(b0) {
    float scale;    // scale per tile
    float zero_point; // zero point
    uint outCount;   // number of output elements
}

[numthreads(64,1,1)]
void CSMain(uint3 DTid : SV_DispatchThreadID) {
    uint idx = DTid.x;
    if (idx >= outCount) return;
    uint byteOffset = idx;
    // Read a 32-bit dword containing the byte (aligned read) and extract the desired byte
    uint dwordOffset = byteOffset & ~3u; // align to 4
    uint shift = (byteOffset & 3u) * 8u;
    uint dword = quantData.Load(dwordOffset);
    uint v = (dword >> shift) & 0xFFu;
    float val = (float(v) - zero_point) * scale;
    outWeights[idx] = val;
}
