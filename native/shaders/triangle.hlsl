// Triangle mesh propagation compute shader
struct Triangle {
  uint v0;
  uint v1;
  uint v2;
  uint pad0;
  float w0;
  float w1;
  float w2;
  float pad1;
};

StructuredBuffer<float4> Vertices : register(t0);
StructuredBuffer<int> Neighbors : register(t1);
StructuredBuffer<float> PhaseIn : register(t2);
StructuredBuffer<float> OutputIn : register(t3);
StructuredBuffer<float> TensorWeights : register(t4);

RWStructuredBuffer<Triangle> Triangles : register(u0);
RWStructuredBuffer<float> PhaseOut : register(u1);
RWStructuredBuffer<float> OutputOut : register(u2);

[numthreads(64,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
  uint i = id.x;
  if (i >= Triangles.Length) return;

  Triangle t = Triangles[i];
  float v0 = Vertices[t.v0].x;
  float v1 = Vertices[t.v1].x;
  float v2 = Vertices[t.v2].x;

  const uint maxNeighbors = 6;
  uint base = i * maxNeighbors;
  float sumOut = 0.0;
  float sumPhase = 0.0;
  float count = 0.0;

  [unroll]
  for (uint n = 0; n < maxNeighbors; n++) {
    int idx = Neighbors[base + n];
    if (idx >= 0) {
      sumOut += OutputIn[idx];
      sumPhase += PhaseIn[idx];
      count += 1.0;
    }
  }

  float neighborOut = (count > 0.0) ? (sumOut / count) : 0.0;
  float neighborPhase = (count > 0.0) ? (sumPhase / count) : PhaseIn[i];
  float phaseCoupling = neighborPhase - PhaseIn[i];

  float omega = 0.05;
  float couplingStrength = 0.12;
  float tw = (TensorWeights.Length > 0) ? TensorWeights[0] : 1.0;
  float signal = (v0 * t.w0 + v1 * t.w1 + v2 * t.w2 + sin(PhaseIn[i]) + neighborOut) * tw;

  OutputOut[i] = max(signal, 0.0);
  PhaseOut[i] = PhaseIn[i] + omega + phaseCoupling * couplingStrength;
}
