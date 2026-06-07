// Attention kernel using compressed INT4 K/V cache (SCXQ2-lite layout).
// cs_5_0 compatible path for iGPU bandwidth reduction.

cbuffer Params : register(b0)
{
    uint seq_len;
    uint head_dim;
    float scale;
    float _pad0;
};

StructuredBuffer<float> Q : register(t0);
ByteAddressBuffer KCache : register(t1);
ByteAddressBuffer VCache : register(t2);
StructuredBuffer<float> KScale : register(t3);
StructuredBuffer<float> VScale : register(t4);
RWStructuredBuffer<float> Out : register(u0);

#define TILE 64
#define HEAD 64

groupshared float Ks[TILE][HEAD];
groupshared float Vs[TILE][HEAD];

float unpack_int4(uint packed, bool high)
{
    int v = high ? ((packed >> 4) & 0xF) : (packed & 0xF);
    if (v >= 8) v -= 16;
    return (float)v;
}

[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID)
{
    const uint q_idx = tid.x;
    const bool active = (q_idx < seq_len) && (head_dim <= HEAD);

    float q_vec[HEAD];
    [unroll]
    for (uint d = 0; d < HEAD; ++d) q_vec[d] = 0.0f;
    if (active) {
        for (uint d = 0; d < head_dim; ++d) {
            q_vec[d] = Q[q_idx * head_dim + d];
        }
    }

    float acc[HEAD];
    [unroll]
    for (uint i = 0; i < HEAD; ++i) acc[i] = 0.0f;

    float max_score = -1e20f;
    float sum_exp = 0.0f;

    for (uint t = 0; t < seq_len; t += TILE) {
        const uint kv_idx = t + lid.x;
        if (kv_idx < seq_len) {
            const float ks = KScale[kv_idx];
            const float vs = VScale[kv_idx];
            for (uint d = 0; d < head_dim; d += 2) {
                const uint packedOffset = ((kv_idx * head_dim + d) / 2); // byte offset
                const uint byteInWord = packedOffset & 3;
                const uint wordAddr = packedOffset & ~3;
                const uint shift = byteInWord * 8;

                const uint kWord = KCache.Load(wordAddr);
                const uint vWord = VCache.Load(wordAddr);
                const uint kp = (kWord >> shift) & 0xFF;
                const uint vp = (vWord >> shift) & 0xFF;

                Ks[lid.x][d] = unpack_int4(kp, false) * ks;
                Vs[lid.x][d] = unpack_int4(vp, false) * vs;
                if (d + 1 < head_dim) {
                    Ks[lid.x][d + 1] = unpack_int4(kp, true) * ks;
                    Vs[lid.x][d + 1] = unpack_int4(vp, true) * vs;
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();

        float scores[TILE];
        [unroll]
        for (uint i = 0; i < TILE; ++i) scores[i] = 0.0f;

        const uint tile_count = min(TILE, seq_len - t);
        if (active) {
            for (uint j = 0; j < tile_count; ++j) {
                float dot = 0.0f;
                for (uint d0 = 0; d0 < head_dim; ++d0) {
                    dot += q_vec[d0] * Ks[j][d0];
                }
                dot *= scale;
                scores[j] = dot;
                max_score = max(max_score, dot);
            }

            float tile_sum = 0.0f;
            for (uint j = 0; j < tile_count; ++j) {
                scores[j] = exp(scores[j] - max_score);
                tile_sum += scores[j];
            }

            if (tile_sum > 0.0f) {
                for (uint j = 0; j < tile_count; ++j) {
                    const float w = scores[j] / tile_sum;
                    for (uint d1 = 0; d1 < head_dim; ++d1) {
                        acc[d1] += w * Vs[j][d1];
                    }
                }
                sum_exp += tile_sum;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (active && sum_exp > 0.0f) {
        for (uint d = 0; d < head_dim; ++d) {
            Out[q_idx * head_dim + d] = acc[d];
        }
    }
}
