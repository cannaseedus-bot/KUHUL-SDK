// gpt2_matmul.hlsl — y = x @ W + b
// x: [M, K]   W: [K, N]   b: [N]   y: [M, N]
// Dispatch(ceil(M/8), ceil(N/8), 1)  numthreads(8,8,1)

cbuffer MatmulParams : register(b0) {
    uint M;   // rows of x (seq_len or batch*seq)
    uint K;   // cols of x / rows of W (n_embd or d_ff)
    uint N;   // cols of W (output dim)
    uint use_bias;  // 1 = add bias, 0 = skip
};

StructuredBuffer<float>   x    : register(t0);  // [M, K]
StructuredBuffer<float>   W    : register(t1);  // [K, N]
StructuredBuffer<float>   bias : register(t2);  // [N]

RWStructuredBuffer<float> y    : register(u0);  // [M, N]  — add into (+=)

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    const uint row = tid.x;
    const uint col = tid.y;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (uint k = 0; k < K; ++k)
        acc += x[row * K + k] * W[k * N + col];

    if (use_bias) acc += bias[col];
    y[row * N + col] += acc;
}
