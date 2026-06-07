// matmul_tiled.hlsl

#define TILE 16

StructuredBuffer<float> A : register(t0); // M x K
StructuredBuffer<float> B : register(t1); // K x N
RWStructuredBuffer<float> C : register(u0); // M x N

cbuffer Params : register(b0)
{
    uint M;
    uint N;
    uint K;
};

groupshared float As[TILE][TILE];
groupshared float Bs[TILE][TILE];

[numthreads(TILE, TILE, 1)]
void main(uint3 gid  : SV_GroupID,
          uint3 tid  : SV_GroupThreadID,
          uint3 dtid : SV_DispatchThreadID)
{
    uint row = gid.y * TILE + tid.y;
    uint col = gid.x * TILE + tid.x;

    float acc = 0.0f;

    for (uint t = 0; t < (K + TILE - 1) / TILE; t++)
    {
        uint tiledCol = t * TILE + tid.x;
        uint tiledRow = t * TILE + tid.y;

        if (row < M && tiledCol < K)
            As[tid.y][tid.x] = A[row * K + tiledCol];
        else
            As[tid.y][tid.x] = 0.0;

        if (tiledRow < K && col < N)
            Bs[tid.y][tid.x] = B[tiledRow * N + col];
        else
            Bs[tid.y][tid.x] = 0.0;

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0; k < TILE; k++)
        {
            acc += As[tid.y][k] * Bs[k][tid.x];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < M && col < N)
    {
        C[row * N + col] = acc;
    }
}
