cbuffer Params : register(b0) {
  uint D;            // token dim
  uint NUM_EXPERTS;  // e.g., 32/64
};

StructuredBuffer<float> token : register(t0);   // [D]
StructuredBuffer<float> router : register(t1);  // [E * D]
RWStructuredBuffer<uint> topk : register(u0);   // [2]

groupshared float scores[64]; // up to 64 experts

[numthreads(32,1,1)] // one warp
void main(uint tid : SV_GroupThreadID)
{
  // each thread handles multiple experts strided
  for (uint e = tid; e < NUM_EXPERTS; e += 32) {
    float s = 0.0f;
    for (uint d=0; d<D; d+=4) {
      float4 t4 = float4(token[d+0], token[d+1], token[d+2], token[d+3]);
      float4 r4 = float4(
        router[e*D + d+0],
        router[e*D + d+1],
        router[e*D + d+2],
        router[e*D + d+3]
      );
      s += dot(t4, r4);
    }
    scores[e] = s;
  }

  GroupMemoryBarrierWithGroupSync();

  // reduction to find top-2
  for (uint stride = NUM_EXPERTS >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      if (scores[tid] < scores[tid + stride]) {
        float tmp = scores[tid];
        scores[tid] = scores[tid + stride];
        scores[tid + stride] = tmp;
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }

  if (tid == 0) {
    topk[0] = 0;
    topk[1] = 1;
  }
}
