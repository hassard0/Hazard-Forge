struct FrameData { float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos; };
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(0, 1)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 1)]] SamplerState gSmp : register(s0);
struct PSInput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float3 color  : COLOR;
    [[vk::location(1)]] float2 uv     : TEXCOORD0;
    [[vk::location(2)]] float3 wnormal: NORMAL;
    [[vk::location(3)]] float3 wpos    : POSITION0;
};
float4 main(PSInput i) : SV_Target {
    float3 N = normalize(i.wnormal);
    float3 L = normalize(-f.lightDir.xyz);
    float  diff = max(dot(N, L), 0.0);
    float3 V = normalize(f.viewPos.xyz - i.wpos);
    float3 H = normalize(L + V);
    float  spec = pow(max(dot(N, H), 0.0), 32.0);
    float  ambient = 0.15;
    float3 tex = gTex.Sample(gSmp, i.uv).rgb * i.color;
    float3 rgb = tex * ((ambient + diff) * f.lightColor.rgb) + spec * f.lightColor.rgb * 0.4;
    return float4(rgb, 1.0);
}
