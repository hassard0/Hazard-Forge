// ImGui UI fragment shader. Samples the font atlas (material set 0: binding 0 = sampled image,
// binding 1 = sampler — matches the engine's material descriptor layout / BindTexture) and
// modulates by the per-vertex color. Output is straight (non-premultiplied) alpha; the UI pipeline
// uses src_alpha / one_minus_src_alpha blending.
[[vk::binding(0, 0)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 0)]] SamplerState gSmp : register(s0);

struct PSInput {
    float4 pos : SV_Position;
    [[vk::location(0)]] float4 col : COLOR0;
    [[vk::location(1)]] float2 uv  : TEXCOORD0;
};

float4 main(PSInput i) : SV_Target {
    return i.col * gTex.Sample(gSmp, i.uv);
}
