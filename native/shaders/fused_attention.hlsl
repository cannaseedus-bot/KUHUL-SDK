// fused_attention.hlsl

#define Br 16
#define Bc 16

StructuredBuffer<float> Q : register(t0); // [M, D]
StructuredBuffer<float> K : register(t1); // [N, D]
StructuredBuffer<float> V : register(t2); // [N, D]
RWStructuredBuffer<float> O : register(u0); // [M, D]

cbuffer Params : register(b0)
{
    uint M;
    uint N;
    uint D;
    float scale;
};

groupshared float Ks[Bc][64]; // D must be <= 64 for this simple shader
groupshared float Vs[Bc][64];

[numthreads(Br,1,1)]
void main(uint3 gid : SV_GroupID,
          uint3 tid : SV_GroupThreadID)
{
    uint q_row = gid.x * Br + tid.x;
    if (q_row >= M) return;

    float m_i = -3.402823e38; // -FLT_MAX
    float l_i = 0.0;

    // assume D <= 64 for local arrays
    float acc[64];
    [unroll] for (uint d = 0; d < D; d++) acc[d] = 0.0;

    for (uint tile = 0; tile < (N + Bc - 1) / Bc; tile++)
    {
        uint base = tile * Bc;

        for (uint j = tid.x; j < Bc * D; j += Br)
        {
            uint r = j / D;
            uint c = j % D;
            uint idx = base + r;
            if (idx < N)
            {
                Ks[r][c] = K[idx * D + c];
                Vs[r][c] = V[idx * D + c];
            }
            else
            {
                Ks[r][c] = 0.0;
                Vs[r][c] = 0.0;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        float scores[Bc];
        for (uint j = 0; j < Bc; j++)
        {
            float dotv = 0.0;
            [unroll]
            for (uint d = 0; d < D; d++){
                float qv = Q[q_row * D + d];
                dotv += qv * Ks[j][d];
            }
            scores[j] = dotv * scale;
        }

        float m_new = m_i;
        for (uint j = 0; j < Bc; j++) m_new = max(m_new, scores[j]);

        float l_new = 0.0;
        for (uint j = 0; j < Bc; j++){
            float e = exp(scores[j] - m_new);
            scores[j] = e; l_new += e;
        }

        float exp_scale = exp(m_i - m_new);
        for (uint d = 0; d < D; d++) acc[d] *= exp_scale;

        for (uint j = 0; j < Bc; j++){
            float p = scores[j];
            for (uint d = 0; d < D; d++){
                acc[d] += p * Vs[j][d];
            }
        }

        l_i = l_i * exp_scale + l_new;
        m_i = m_new;

        GroupMemoryBarrierWithGroupSync();
    }

    for (uint d = 0; d < D; d++){
        O[q_row * D + d] = acc[d] / l_i;
    }
}
