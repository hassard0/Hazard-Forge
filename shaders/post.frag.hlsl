[[vk::binding(0, 0)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 0)]] SamplerState gSmp : register(s0);
struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };
float4 main(PSInput i) : SV_Target {
    float3 c = gTex.Sample(gSmp, i.uv).rgb;
    c = c / (c + 1.0);                         // Reinhard tonemap
    c = pow(c, 1.0 / 2.2);                     // gamma
    float2 d = i.uv - 0.5;
    float vig = smoothstep(0.8, 0.35, length(d));  // darken corners
    return float4(c * vig, 1.0);
}
