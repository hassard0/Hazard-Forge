// Screen-space reflections — composite + tonemap (Slice AH). The FINAL pass for the SSR showcase:
// samples the lit HDR scene (gScene, binding 0/1) and the SSR reflection (gSsr, binding 3/4 — the
// same second material slot bloom/SSAO composites use): rgb = reflected radiance, a = blended weight
// (reflectivity * Fresnel * edge/march fade). It blends `lerp(scene, ssr.rgb, ssr.a)` so reflective
// floor pixels show the reflection over the scene while everything else (and SSR misses, a=0) keeps
// the plain lit scene (whose floor already carries the existing procedural IBL sky sheen — the SSR
// fallback). Then it applies the SAME displayed-image pipeline as post.frag — exposure 1.7, ACES
// filmic tonemap, gamma, cinematic grade, film grain, vignette — and writes the LDR swapchain.
// post.frag/bloom_composite and their goldens are untouched; the tonemap/grade math is replicated
// verbatim here (no shared include in the toolchain).
[[vk::binding(0, 0)]] Texture2D    gScene : register(t0);   // lit HDR scene color
[[vk::binding(1, 0)]] SamplerState gSmp   : register(s0);
[[vk::binding(3, 0)]] Texture2D    gSsr   : register(t3);   // SSR reflection (rgb) + weight (a)
[[vk::binding(4, 0)]] SamplerState gSsrSmp : register(s3);

struct SsrCompParams { float2 texel; float intensity; float pad; };
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer SsrCompPC { SsrCompParams cp; };
#define HF_CP cp
#else
[[vk::push_constant]] struct { SsrCompParams p; } pc;
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

    float3 scene = gScene.Sample(gSmp, i.uv).rgb;   // linear HDR lit scene
    float4 ssr   = gSsr.Sample(gSsrSmp, i.uv);      // rgb = reflection, a = blend weight

    // Debug (pad < 0): show the raw SSR RT (no tonemap) so the ray-march pipeline can be inspected.
    if (HF_CP.pad < 0.0) return float4(ssr.rgb, 1.0);

    // Blend the reflection over the scene by its weight. Misses (a=0) leave the plain scene, which
    // already includes the floor's procedural IBL sky reflection (the documented SSR fallback).
    float3 c = lerp(scene, ssr.rgb, ssr.a);

    c *= HF_CP.intensity;                           // exposure (1.7), matches post.frag
    c = ACES(c);
    c = pow(c, 1.0 / 2.2);
    c = ColorGrade(c);
    c = FilmGrain(c, i.uv, float2(w, h));

    float2 d = i.uv - 0.5;
    float vig = smoothstep(0.8, 0.35, length(d));
    return float4(saturate(c * vig), 1.0);
}
