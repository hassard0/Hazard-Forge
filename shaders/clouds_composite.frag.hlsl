// Volumetric clouds — composite + tonemap (Slice CH). The FINAL pass for the --clouds-shot showcase:
// samples the lit HDR scene (gScene, binding 0/1 — which already contains the procedural sky in the
// background + the lit/shadowed scene in front) and the cloud RT (gClouds, binding 3/4 — the same
// second material slot the bloom/SSAO/SSR/water composites use): rgb = cloud radiance, a = coverage
// (0 = clear sky, 1 = opaque cloud; 0 where the lit scene is in front, so the scene shows through).
// It blends `lerp(scene, clouds.rgb, clouds.a)` so the cumulus layer covers the sky background while
// the lit scene + clear sky keep the plain lit scene, then applies the SAME displayed-image pipeline
// as post.frag / water_composite — exposure, ACES filmic tonemap, gamma, cinematic grade, film grain,
// vignette — and writes the LDR swapchain. post.frag / water_composite / volumetric_composite + their
// goldens are untouched; the tonemap/grade math is replicated verbatim (no shared include).
[[vk::binding(0, 0)]] Texture2D    gScene  : register(t0);   // lit HDR scene color (incl. sky bg)
[[vk::binding(1, 0)]] SamplerState gSmp    : register(s0);
[[vk::binding(3, 0)]] Texture2D    gClouds : register(t3);   // cloud radiance (rgb) + coverage (a)
[[vk::binding(4, 0)]] SamplerState gCloudsSmp : register(s3);

struct CloudCompParams { float2 texel; float intensity; float pad; };
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer CloudCompPC { CloudCompParams cp; };
#define HF_CP cp
#else
[[vk::push_constant]] struct { CloudCompParams p; } pc;
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

    float3 scene  = gScene.Sample(gSmp, i.uv).rgb;        // linear HDR lit scene (incl. sky bg)
    float4 clouds = gClouds.Sample(gCloudsSmp, i.uv);     // rgb = cloud radiance, a = coverage

    // Debug (pad < 0): show the raw cloud RT (no tonemap) so the raymarch can be inspected.
    if (HF_CP.pad < 0.0) return float4(clouds.rgb, 1.0);

    // Blend the cloud layer over the scene by its coverage. a=0 (clear sky / scene in front) leaves
    // the plain lit scene; a=1 (opaque cumulus) shows the cloud radiance.
    float3 c = lerp(scene, clouds.rgb, clouds.a);

    c *= HF_CP.intensity;                                 // exposure, matches post.frag
    c = ACES(c);
    c = pow(c, 1.0 / 2.2);
    c = ColorGrade(c);
    c = FilmGrain(c, i.uv, float2(w, h));

    float2 d = i.uv - 0.5;
    float vig = smoothstep(0.8, 0.35, length(d));
    return float4(saturate(c * vig), 1.0);
}
