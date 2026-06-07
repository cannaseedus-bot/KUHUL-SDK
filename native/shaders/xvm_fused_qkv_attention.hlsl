// Fused QKV + attention kernel (cs_5_0 compatible).
// Designed for shared-memory fallback path on legacy iGPU hardware.

cbuffer Params : register(b0)
{
    uint seq_len;
    uint model_dim;
    uint head_dim;
    float scale;
};

StructuredBuffer<float> X : register(t0);
StructuredBuffer<float> Wqkv : register(t1);
RWStructuredBuffer<float> Out : register(u0);

#define TILE 64
#define HEAD 64

groupshared float Ks[TILE][HEAD];
groupshared float Vs[TILE][HEAD];

[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID)
{
    const uint q_idx = tid.x;
    if (q_idx >= seq_len) return;
    if (head_dim > HEAD) return;

    float q_vec[HEAD];
    [unroll]
    for (uint d = 0; d < HEAD; ++d) q_vec[d] = 0.0f;

    for (uint d = 0; d < head_dim; ++d) {
        float sum = 0.0f;
        for (uint m = 0; m < model_dim; ++m) {
            const float x = X[q_idx * model_dim + m];
            const float wq = Wqkv[m * (3 * head_dim) + d];
            sum += x * wq;
        }
        q_vec[d] = sum;
    }

    float acc[HEAD];
    [unroll]
    for (uint i = 0; i < HEAD; ++i) acc[i] = 0.0f;

    float max_score = -1e20f;
    float sum_exp = 0.0f;

    for (uint t = 0; t < seq_len; t += TILE) {
        const uint k_idx = t + lid.x;
        if (k_idx < seq_len) {
            for (uint d = 0; d < head_dim; ++d) {
                float k_sum = 0.0f;
                float v_sum = 0.0f;

                for (uint m = 0; m < model_dim; ++m) {
                    const float x = X[k_idx * model_dim + m];
                    k_sum += x * Wqkv[m * (3 * head_dim) + head_dim + d];
                    v_sum += x * Wqkv[m * (3 * head_dim) + 2 * head_dim + d];
                }
                Ks[lid.x][d] = k_sum;
                Vs[lid.x][d] = v_sum;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        float scores[TILE];
        [unroll]
        for (uint i = 0; i < TILE; ++i) scores[i] = 0.0f;

        const uint tile_count = min(TILE, seq_len - t);
        for (uint j = 0; j < tile_count; ++j) {
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                dot += q_vec[d] * Ks[j][d];
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
                for (uint d = 0; d < head_dim; ++d) {
                    acc[d] += w * Vs[j][d];
                }
            }
            sum_exp += tile_sum;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (sum_exp <= 0.0f) return;
    for (uint d = 0; d < head_dim; ++d) {
        Out[q_idx * head_dim + d] = acc[d];
    }
}
