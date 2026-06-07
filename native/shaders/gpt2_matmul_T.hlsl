// gpt2_matmul_T.hlsl — y[M,N] += x[M,K] @ W.T  where W is stored as [N,K]
// Used for input gradients: dX = dY @ W.T
// Dispatch(ceil(M/8), ceil(N/8), 1)  numthreads(8,8,1)

cbuffer CB : register(b0) { uint M, K, N, pad; };

StructuredBuffer<float>   x : register(t0);   // [M, K]
StructuredBuffer<float>   W : register(t1);   // [N, K]  — accessed as transposed
RWStructuredBuffer<float> y : register(u0);   // [M, N]  — accumulated

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    const uint m = tid.x, n = tid.y;
    if (m >= M || n >= N) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k)
        acc += x[m * K + k] * W[n * K + k];   // W.T[k,n] = W[n,k]
    y[m * N + n] += acc;
}
