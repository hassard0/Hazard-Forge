// Screen-space global illumination — composite + tonemap (Slice BP). The FINAL pass for the
// --ssgi-shot showcase: samples the lit HDR scene (gScene, binding 0/1) and the SSGI indirect-diffuse
// irradiance (gSsgi, binding 3/4 — the same second material slot bloom/SSAO/SSR composite use), then
// ADDS the indirect over the scene: c = scene + indirect * albedo. The albedo modulation: SSGI's
// indirect irradiance must be multiplied by the RECEIVER's diffuse albedo to become reflected radiance
// (Cornell-style color bleed = the colored panel's light * the neutral receiver's albedo). The
// showcase receivers (a white floor + neutral box) are near-WHITE (albedo ~= 1), and we deliberately
// keep no separate albedo g-buffer (out of scope — SSAO/SSR/decal share only normal+depth), so for
// these near-white receivers `indirect * albedo ~= indirect` and we add the indirect irradiance
// directly. (A future albedo g-buffer would let arbitrary-color receivers modulate the bleed; the
// estimator + bindings are unchanged.) Then it applies the SAME displayed-image pipeline as
// post.frag/ssr_composite — exposure 1.7, ACES filmic tonemap, gamma, cinematic grade, film grain,
// vignette — and writes the LDR swapchain. post.frag/ssr_composite + their goldens are untouched; the
// tonemap/grade math is replicated verbatim here (no shared include in the toolchain).
[[vk::binding(0, 0)]] Texture2D    gScene : register(t0);   // lit HDR scene color
[[vk::binding(1, 0)]] SamplerState gSmp   : register(s0);
[[vk::binding(3, 0)]] Texture2D    gSsgi  : register(t3);   // SSGI indirect diffuse (rgb); a = 1
[[vk::binding(4, 0)]] SamplerState gSsgiSmp : register(s3);

struct SsgiCompParams { float2 texel; float intensity; float pad; };
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer SsgiCompPC { SsgiCompParams cp; };
#define HF_CP cp
#else
[[vk::push_constant]] struct { SsgiCompParams p; } pc;
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

    float3 scene = gScene.Sample(gSmp, i.uv).rgb;     // linear HDR lit scene
    float3 indirect = gSsgi.Sample(gSsgiSmp, i.uv).rgb; // SSGI indirect diffuse irradiance

    // Debug (pad < 0): show the raw SSGI indirect (no tonemap) so the gather pipeline can be inspected.
    if (HF_CP.pad < 0.0) return float4(indirect, 1.0);

    // ADD the indirect diffuse over the scene (albedo ~= 1 for the near-white receivers; documented).
    float3 c = scene + indirect;

    c *= HF_CP.intensity;                             // exposure (1.7), matches post.frag/ssr_composite
    c = ACES(c);
    c = pow(c, 1.0 / 2.2);
    c = ColorGrade(c);
    c = FilmGrain(c, i.uv, float2(w, h));

    float2 d = i.uv - 0.5;
    float vig = smoothstep(0.8, 0.35, length(d));
    return float4(saturate(c * vig), 1.0);
}
