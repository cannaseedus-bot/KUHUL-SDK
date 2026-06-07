// gpt2_embed.hlsl — token + position embedding lookup
// hidden[i] = wte[token[i]] + wpe[i]
// Dispatch(ceil(seq_len/64), 1, 1)  numthreads(64,1,1)

cbuffer EmbedParams : register(b0) {
    uint seq_len;
    uint n_embd;
    uint2 pad;
};

StructuredBuffer<float> wte    : register(t0);  // [vocab, n_embd]
StructuredBuffer<float> wpe    : register(t1);  // [n_ctx, n_embd]
StructuredBuffer<int>   tokens : register(t2);  // [seq_len]

RWStructuredBuffer<float> hidden : register(u0); // [seq_len, n_embd]

[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    const uint pos = tid.x;
    if (pos >= seq_len) return;

    const int tok = tokens[pos];
    const uint wte_base = (uint)tok * n_embd;
    const uint wpe_base = pos * n_embd;
    const uint out_base = pos * n_embd;

    for (uint d = 0; d < n_embd; ++d)
        hidden[out_base + d] = wte[wte_base + d] + wpe[wpe_base + d];
}
