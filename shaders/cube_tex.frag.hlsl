// DXC emits Texture2D and SamplerState as two separate descriptors (a sampled image
// and a sampler), not one combined image sampler — so they get distinct bindings.
[[vk::binding(0, 0)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 0)]] SamplerState gSmp : register(s0);
struct PSInput {
    float4 pos   : SV_Position;
    [[vk::location(0)]] float3 color : COLOR;
    [[vk::location(1)]] float2 uv    : TEXCOORD0;
};
float4 main(PSInput i) : SV_Target {
    float4 t = gTex.Sample(gSmp, i.uv);
    return float4(t.rgb * i.color, 1.0);
}
