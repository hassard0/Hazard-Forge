// Slice AK — probe REFLECTION bake fragment shader. Writes the wall's flat color (the bound 1x1
// colored texture) into the reflection atlas tile, with a simple Lambert term from a fixed fill light
// so the captured walls have gentle shading (not flat) — enough to read as a room, while the wall
// COLOR (red left / green right / neutral else) dominates so reflections clearly show which wall.
// Linear HDR out (the atlas is RGBA16F); no tonemap. Vertex color is intentionally IGNORED so the
// shared cube mesh's per-face tints don't fight the bound wall color.
[[vk::binding(0, 0)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 0)]] SamplerState gSmp : register(s0);
struct PSInput {
    float4 clip : SV_Position;
    [[vk::location(0)]] float2 uv      : TEXCOORD0;
    [[vk::location(1)]] float3 wnormal : NORMAL;
    [[vk::location(2)]] float3 wpos    : POSITION0;
};
float3 SrgbToLinear(float3 c) { return pow(saturate(c), 2.2); }
float4 main(PSInput i) : SV_Target {
    float3 albedo = SrgbToLinear(gTex.Sample(gSmp, i.uv).rgb);
    float3 N = normalize(i.wnormal);
    // Fixed fill light from above-front so walls are gently shaded (a flat fill keeps color dominant).
    float3 L = normalize(float3(-0.3, 0.9, 0.4));
    float ndl = saturate(dot(N, L)) * 0.6 + 0.4;   // 0.4 ambient + up to 0.6 diffuse
    return float4(albedo * ndl, 1.0);
}
