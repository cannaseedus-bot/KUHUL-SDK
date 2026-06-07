// gpt2_adam.hlsl — GPU Adam optimizer update
// Dispatch(ceil(numel/256), 1, 1)
// cs_5_0 compatible (D3D11 compute)

cbuffer AdamParams : register(b0) {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float bias_corr1;   // 1 / (1 - beta1^t)
    float bias_corr2;   // 1 / (1 - beta2^t)
    uint  numel;
};

RWStructuredBuffer<float> weights : register(u0);
RWStructuredBuffer<float> grads   : register(u1);
RWStructuredBuffer<float> m       : register(u2);   // first moment
RWStructuredBuffer<float> v       : register(u3);   // second moment

[numthreads(256, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= numel) return;

    float g = grads[i];

    // NaN/Inf guard — skip corrupt gradients rather than infecting weights
    if (isnan(g) || isinf(g)) { grads[i] = 0.0f; return; }

    // Per-element gradient clipping
    g = clamp(g, -10.0f, 10.0f);

    // L2 weight decay folded into gradient
    g += weight_decay * weights[i];

    // Moment updates (EMA)
    const float mi = beta1 * m[i] + (1.0f - beta1) * g;
    const float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;

    // Bias-corrected update
    const float m_hat = mi * bias_corr1;
    const float v_hat = vi * bias_corr2;

    weights[i] -= lr * m_hat / (sqrt(v_hat) + eps);

    // Zero gradient for next step
    grads[i] = 0.0f;
}
