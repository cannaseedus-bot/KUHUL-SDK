// gpt2_gelu_fwd.hlsl — GELU forward, saves pre-activation for backward
// GELU(x) ≈ 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
// Dispatch(ceil(numel/256), 1, 1)

static const float SQRT_2_OVER_PI = 0.7978845608f;
static const float COEFF = 0.044715f;

cbuffer GeluParams : register(b0) {
    uint numel;
    uint3 pad;
};

StructuredBuffer<float>   x_in  : register(t0);  // [numel] input
RWStructuredBuffer<float> x_pre : register(u0);  // [numel] saved pre-activation (= x_in)
RWStructuredBuffer<float> y     : register(u1);  // [numel] output

[numthreads(256, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= numel) return;

    const float x = x_in[i];
    x_pre[i] = x;  // save for backward

    const float k = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    y[i] = 0.5f * x * (1.0f + tanh(k));
}
