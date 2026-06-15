// HDR bloom — composite + tonemap (Slice U). The FINAL post pass for the bloom showcase: samples
// the HDR scene (gTex, binding 0/1) and the accumulated bloom result (gTex2, binding 3/4), adds the
// bloom (scaled by strength) into the scene, then applies the SAME displayed-image pipeline as
// post.frag — exposure 1.7, ACES filmic tonemap, gamma, cinematic color grade, film grain, vignette
// — and writes the LDR swapchain. This REPLACES post.frag for the bloom showcase only; post.frag and
// its golden are untouched. The tonemap/grade math is replicated here verbatim (no shared include in
// the toolchain) so the displayed look matches the existing post pass exactly aside from the bloom.
[[vk::binding(0, 0)]] Texture2D    gTex  : register(t0);   // HDR scene color
[[vk::binding(1, 0)]] SamplerState gSmp  : register(s0);
[[vk::binding(3, 0)]] Texture2D    gTex2 : register(t3);   // accumulated bloom (binding 3/4 slot)
[[vk::binding(4, 0)]] SamplerState gSmp2 : register(s3);

struct BloomParams { float2 texel; float threshold; float knee; float strength; float intensity; };
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer BloomPC { BloomParams bp; };
#define HF_BP bp
#else
[[vk::push_constant]] struct { BloomParams p; } pc;
#define HF_BP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };
static const float3 kLuma = float3(0.299, 0.587, 0.114);

// ACES filmic tonemapping (Narkowicz approximation) — identical to post.frag.
float3 ACES(float3 x) {
    float a = 2.51, b = 0.03, c2 = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c2 * x + d) + e));
}

// Cinematic color grade — identical to post.frag.
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

// Subtle static film grain — identical to post.frag.
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

    float3 scene = gTex.Sample(gSmp, i.uv).rgb;     // linear HDR scene
    float3 bloom = gTex2.Sample(gSmp2, i.uv).rgb;   // accumulated bloom (linear HDR)

    // Add the bloom into the scene (strength controls overall bloom presence), then expose.
    float3 c = scene + bloom * HF_BP.strength;
    c *= HF_BP.intensity;                           // exposure (1.7), matches post.frag

    c = ACES(c);                                    // ACES filmic tonemap
    c = pow(c, 1.0 / 2.2);                          // gamma -> displayed LDR

    c = ColorGrade(c);
    c = FilmGrain(c, i.uv, float2(w, h));

    float2 d = i.uv - 0.5;
    float vig = smoothstep(0.8, 0.35, length(d));   // darken corners
    return float4(saturate(c * vig), 1.0);
}
