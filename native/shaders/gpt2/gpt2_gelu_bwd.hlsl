// gpt2_gelu_bwd.hlsl — GELU backward, cs_5_0
// GPT-2 uses the tanh approximation:
//   GELU(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
//   d/dx = 0.5*(1+t) + 0.5*x*(1-t^2)*sqrt(2/pi)*(1+3*0.044715*x^2)
//   where t = tanh(sqrt(2/pi)*(x + 0.044715*x^3))
//
// Dispatch(ceil(numel/256), 1, 1)  numthreads(256, 1, 1)
//
// Bug fixed (v0.1.1 -> v0.1.2):
//   Old version used t0=pre_gelu, t1=dout, u0=dx.
//   When the C++ host bound the same buffer as both t1 (SRV) and u0 (UAV),
//   D3D11 silently nulled one binding, producing zero gradients.
//   Fix: in-place update — u0 carries both dout (input) and dx (output).
//   Caller writes upstream gradient into u0 before dispatch, reads dx from u0 after.

static const float SQRT_2_OVER_PI = 0.7978845608f;
static const float COEFF = 0.044715f;

StructuredBuffer<float>   pre_gelu : register(t0);  // x before GELU [numel]
RWStructuredBuffer<float> dx       : register(u0);  // in-place: dout in, dx out

cbuffer GeluBwdParams : register(b0) {
    uint numel;
    uint3 pad;
};

[numthreads(256, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= numel) return;

    const float x  = pre_gelu[i];
    const float x2 = x * x;
    const float k  = SQRT_2_OVER_PI * (x + COEFF * x2 * x);
    const float t  = tanh(k);

    const float dgelu_dx = 0.5f * (1.0f + t)
        + 0.5f * x * (1.0f - t * t) * SQRT_2_OVER_PI * (1.0f + 3.0f * COEFF * x2);

    dx[i] *= dgelu_dx;  // in-place: d_pre = d_post * gelu'(x)
}
