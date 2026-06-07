// gpt2_matmul_dW.hlsl — dW[K,N] += X.T @ dY  where X is [M,K], dY is [M,N]
// Used for weight gradients: dW += input_activation.T @ upstream_gradient
// Dispatch(ceil(K/8), ceil(N/8), 1)  numthreads(8,8,1)

cbuffer CB : register(b0) { uint M, K, N, pad; };

StructuredBuffer<float>   X  : register(t0);   // [M, K]  input activation
StructuredBuffer<float>   dY : register(t1);   // [M, N]  output gradient
RWStructuredBuffer<float> dW : register(u0);   // [K, N]  weight gradient (accumulated)

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    const uint k = tid.x, n = tid.y;
    if (k >= K || n >= N) return;
    float acc = 0.0f;
    for (uint m = 0; m < M; ++m)
        acc += X[m * K + k] * dY[m * N + n];
    dW[k * N + n] += acc;
}
