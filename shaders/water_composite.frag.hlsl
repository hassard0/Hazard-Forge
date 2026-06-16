// Water rendering — composite + tonemap (Slice CF). The FINAL pass for the --water-shot showcase:
// samples the lit HDR opaque scene (gScene, binding 0/1) and the water surface RT (gWater, binding 3/4
// — the same second material slot the bloom/SSAO/SSR composites use): rgb = water shading, a = 1 where
// the water grid rasterized, 0 elsewhere (cleared). It blends `lerp(scene, water.rgb, water.a)` so the
// water plane covers the lower scene while the sky + above-water object tops keep the plain lit scene,
// then applies the SAME displayed-image pipeline as post.frag/ssr_composite — exposure 1.7, ACES
// filmic tonemap, gamma, cinematic grade, film grain, vignette — and writes the LDR swapchain. The
// tonemap/grade math is replicated verbatim (no shared include in the toolchain). post.frag/
// ssr_composite + their goldens are untouched.
[[vk::binding(0, 0)]] Texture2D    gScene : register(t0);   // lit HDR opaque scene color
[[vk::binding(1, 0)]] SamplerState gSmp   : register(s0);
[[vk::binding(3, 0)]] Texture2D    gWater : register(t3);   // water shading (rgb) + coverage (a)
[[vk::binding(4, 0)]] SamplerState gWaterSmp : register(s3);

struct WaterCompParams { float2 texel; float intensity; float pad; };
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer WaterCompPC { WaterCompParams cp; };
#define HF_CP cp
#else
[[vk::push_constant]] struct { WaterCompParams p; } pc;
#define HF_CP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };
static const float3 kLuma = float3(0.299, 0.587, 0.114);

float3 ACES(float3 x) {
    float a = 2.51, b = 0.03, c2 = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c2 * x + d) + e));
}
float3 ColorGrade(float3 c) {
    float luma = dot(c, kLuma);
    const float3 shadowTint = float3(0.90, 1.00, 1.06);
    const float3 highTint   = float3(1.06, 1.00, 0.92);
    const float  gradeAmt   = 0.08;
    float3 tint = lerp(shadowTint, highTint, luma);
    c = lerp(c, c * tint, gradeAmt);
    c = lerp(c, smoothstep(0.0, 1.0, c), 0.15);
    float gluma = dot(c, kLuma);
    c = lerp(gluma.xxx, c, 1.08);
    return c;
}
float3 FilmGrain(float3 c, float2 uv, float2 res) {
    float n = frac(sin(dot(uv * res, float2(12.9898, 78.233))) * 43758.5453);
    float grain = (n - 0.5) * 2.0 * 0.025;
    float luma = dot(c, kLuma);
    grain *= 1.0 - smoothstep(0.5, 1.0, luma);
    return c + grain;
}

float4 main(PSInput i) : SV_Target {
    float w, h;
    gScene.GetDimensions(w, h);

    float3 scene = gScene.Sample(gSmp, i.uv).rgb;     // linear HDR lit opaque scene
    float4 water = gWater.Sample(gWaterSmp, i.uv);    // rgb = water shading, a = coverage

    // Debug (pad < 0): show the raw water RT (no tonemap) so the pipeline can be inspected.
    if (HF_CP.pad < 0.0) return float4(water.rgb, 1.0);

    // Blend the water over the scene by its coverage. a=0 (no water) leaves the plain lit scene.
    float3 c = lerp(scene, water.rgb, water.a);

    c *= HF_CP.intensity;                             // exposure (1.7), matches post.frag
    c = ACES(c);
    c = pow(c, 1.0 / 2.2);
    c = ColorGrade(c);
    c = FilmGrain(c, i.uv, float2(w, h));

    float2 d = i.uv - 0.5;
    float vig = smoothstep(0.8, 0.35, length(d));
    return float4(saturate(c * vig), 1.0);
}
