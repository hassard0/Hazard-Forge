// SSAO composite + tonemap (Slice Y). The FINAL pass for the SSAO showcase: samples the lit HDR scene
// (gTex, binding 0/1) and the blurred AO (gAO, binding 3/4 — the same second material slot the bloom
// composite uses), darkens the scene by the AO (litColor * lerp(1, ao, aoStrength)), then applies the
// SAME displayed-image pipeline as post.frag — exposure 1.7, ACES filmic tonemap, gamma, cinematic
// grade, film grain, vignette — and writes the LDR swapchain. aoStrength = 0 renders SSAO-OFF (AO
// forced to 1) through the IDENTICAL pipeline, giving a clean on/off comparison. post.frag and its
// golden are untouched; the tonemap/grade math is replicated verbatim (no shared include).
[[vk::binding(0, 0)]] Texture2D    gTex : register(t0);   // lit HDR scene color
[[vk::binding(1, 0)]] SamplerState gSmp : register(s0);
[[vk::binding(3, 0)]] Texture2D    gAO  : register(t3);   // blurred AO (binding 3/4 slot)
[[vk::binding(4, 0)]] SamplerState gAOSmp : register(s3);

struct SsaoCompParams { float2 texel; float aoStrength; float intensity; };
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer SsaoCompPC { SsaoCompParams cp; };
#define HF_CP cp
#else
[[vk::push_constant]] struct { SsaoCompParams p; } pc;
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
    gTex.GetDimensions(w, h);

    float3 scene = gTex.Sample(gSmp, i.uv).rgb;     // linear HDR lit scene
    float  ao    = gAO.Sample(gAOSmp, i.uv).r;      // blurred AO (1 = open)

    // Debug: aoStrength < 0 outputs the raw AO term as grayscale (no tonemap) for inspection.
    if (HF_CP.aoStrength < 0.0) return float4(ao, ao, ao, 1.0);

    // Darken by AO. aoStrength interpolates between no-AO (1) and full AO; 0 = SSAO OFF.
    float aoFactor = lerp(1.0, ao, HF_CP.aoStrength);
    float3 c = scene * aoFactor;

    c *= HF_CP.intensity;                           // exposure (1.7), matches post.frag
    c = ACES(c);
    c = pow(c, 1.0 / 2.2);
    c = ColorGrade(c);
    c = FilmGrain(c, i.uv, float2(w, h));

    float2 d = i.uv - 0.5;
    float vig = smoothstep(0.8, 0.35, length(d));
    return float4(saturate(c * vig), 1.0);
}
