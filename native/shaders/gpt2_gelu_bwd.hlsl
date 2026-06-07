// gpt2_gelu_bwd.hlsl — GELU backward
// GPT-2 uses the tanh approximation:
// GELU(x) ≈ 0.5*x*(1 + tanh(sqrt(2/π)*(x + 0.044715*x³)))
// d/dx GELU(x) = 0.5*(1 + tanh(k)) + 0.5*x*(1-tanh²(k))*sqrt(2/π)*(1 + 3*0.044715*x²)
// where k = sqrt(2/π) * (x + 0.044715*x³)
//
// Dispatch(ceil(numel/256), 1, 1)

static const float SQRT_2_OVER_PI = 0.7978845608f;  // sqrt(2/pi)
static const float COEFF = 0.044715f;

StructuredBuffer<float>  pre_gelu : register(t0);  // x before GELU [numel]
StructuredBuffer<float>  dout     : register(t1);  // upstream gradient [numel]
RWStructuredBuffer<float> dx      : register(u0);  // output gradient [numel]

cbuffer GeluBwdParams : register(b0) {
    uint numel;
    uint3 pad;
};

[numthreads(256, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= numel) return;

    const float x = pre_gelu[i];
    const float x3 = x * x * x;
    // Clamp k before tanh — Intel HD 4600 driver overflows exp(2k) for |k| >= 88
    const float k  = clamp(SQRT_2_OVER_PI * (x + COEFF * x3), -15.0f, 15.0f);
    const float t  = tanh(k);

    // dgelu/dx = 0.5*(1+t) + 0.5*x*(1-t*t)*SQRT_2_OVER_PI*(1 + 3*COEFF*x*x)
    const float dgelu_dx = 0.5f * (1.0f + t)
        + 0.5f * x * (1.0f - t * t) * SQRT_2_OVER_PI * (1.0f + 3.0f * COEFF * x * x);

    dx[i] += dout[i] * dgelu_dx;
}
