// Slice CW — Auto-Exposure: the TONEMAP-WITH-EXPOSURE fullscreen variant (NEW path behind
// --autoexposure-shot). IDENTICAL to the default post.frag tonemap (FXAA -> exposure -> bloom glow ->
// ACES -> gamma -> color grade -> film grain -> vignette) EXCEPT the fixed `kExposure = 1.7` is replaced
// by the `exposure` value the autoexposure_reduce pass wrote to the 1-entry exposure SSBO. EVERYTHING
// ELSE is byte-for-byte the same as post.frag, so:
//
//   * With adaptationEnabled == false the reduce writes E0 == 1.7 (the default's fixed exposure), so this
//     shader applies EXACTLY 1.7 -> its output is BYTE-IDENTICAL to the default post.frag tonemap of the
//     same scene (the adaptation-off no-op proof — no constant bias, no exposure drift). The existing
//     post.frag + its golden are UNTOUCHED; this is a separate variant.
//   * With adaptationEnabled == true the reduce writes the histogram-derived exposure -> a bright scene
//     darkens + a dark scene brightens (the eye adapting).
//
// SEAM DISCIPLINE: above the RHI seam; the only vk/MSL mentions are the HF_MSL_GEN gen-path guards +
// [[vk::binding]] decorations. The exposure SSBO reuses the Slice-AG cluster fragment-storage binding
// (binding 13, space 3 on Vulkan; a flat fragment buffer on Metal) bound via BindLightClusters — NO new
// RHI seam (same plumbing as froxel_apply's volume).
//
// BINDING LAYOUT (mirrors froxel_apply.frag so the exposure SSBO lands at set 3 behind the frame(set0) +
// material(set1) sets): a frame UBO at set 0 (DECLARED but UNUSED — present only so the material set
// sits at index 1 and the cluster set at index 3), the HDR scene texture at the MATERIAL set (set 1,
// binding 0 — the base slot BindTexture fills), and the exposure SSBO at the cluster set (set 3,
// binding 13).
struct FrameDummy { float4x4 a; float4x4 b; float4 c[6]; };  // never read; matches the frame UBO size class
[[vk::binding(0, 0)]] cbuffer Frame { FrameDummy gFrameUnused; };

[[vk::binding(0, 1)]] Texture2D    gTex : register(t0);   // the HDR scene color (linear)
[[vk::binding(1, 1)]] SamplerState gSmp : register(s0);

// The 1-entry exposure SSBO (the autoexposure_reduce pass wrote it) at the cluster-set fragment-storage
// binding 13/space3 (Vulkan) / flat fragment buffer (Metal), bound via BindLightClusters(exposure,...).
[[vk::binding(13, 3)]] StructuredBuffer<float> gExposure : register(t13, space3);

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

static const float3 kLuma = float3(0.299, 0.587, 0.114);

// Lightweight FXAA (NVIDIA FXAA3 "console" style) — copied VERBATIM from post.frag.
float3 Fxaa(float2 uv, float2 texel) {
    float3 rgbM  = gTex.Sample(gSmp, uv).rgb;
    float3 rgbNW = gTex.Sample(gSmp, uv + float2(-1.0, -1.0) * texel).rgb;
    float3 rgbNE = gTex.Sample(gSmp, uv + float2( 1.0, -1.0) * texel).rgb;
    float3 rgbSW = gTex.Sample(gSmp, uv + float2(-1.0,  1.0) * texel).rgb;
    float3 rgbSE = gTex.Sample(gSmp, uv + float2( 1.0,  1.0) * texel).rgb;

    float lumaM  = dot(rgbM,  kLuma);
    float lumaNW = dot(rgbNW, kLuma);
    float lumaNE = dot(rgbNE, kLuma);
    float lumaSW = dot(rgbSW, kLuma);
    float lumaSE = dot(rgbSE, kLuma);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    float2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.25 * 0.125, 1.0 / 128.0);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcpDirMin, -8.0, 8.0) * texel;

    float3 rgbA = 0.5 * (gTex.Sample(gSmp, uv + dir * (1.0 / 3.0 - 0.5)).rgb +
                         gTex.Sample(gSmp, uv + dir * (2.0 / 3.0 - 0.5)).rgb);
    float3 rgbB = rgbA * 0.5 + 0.25 * (gTex.Sample(gSmp, uv + dir * -0.5).rgb +
                                       gTex.Sample(gSmp, uv + dir *  0.5).rgb);
    float lumaB = dot(rgbB, kLuma);
    return (lumaB < lumaMin || lumaB > lumaMax) ? rgbA : rgbB;
}

// ACES filmic tonemapping (Narkowicz) — copied VERBATIM from post.frag.
float3 ACES(float3 x) {
    float a = 2.51, b = 0.03, c2 = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c2 * x + d) + e));
}

// Cinematic color grade — copied VERBATIM from post.frag.
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

// Subtle static film grain — copied VERBATIM from post.frag.
float3 FilmGrain(float3 c, float2 uv, float2 res) {
    float n = frac(sin(dot(uv * res, float2(12.9898, 78.233))) * 43758.5453);
    float grain = (n - 0.5) * 2.0 * 0.025;
    float luma = dot(c, kLuma);
    grain *= 1.0 - smoothstep(0.5, 1.0, luma);
    return c + grain;
}

// Screen-space glow / cheap bloom — copied VERBATIM from post.frag.
float3 Glow(float2 uv, float2 texel) {
    const float threshold = 0.8;
    const int   kDirs   = 8;
    const float radii[3] = { 2.0, 4.0, 7.0 };
    const float rw[3]    = { 1.0, 0.65, 0.35 };

    float3 sum = 0.0.xxx;
    float  wsum = 0.0;
    for (int r = 0; r < 3; ++r) {
        for (int s = 0; s < kDirs; ++s) {
            float ang = (6.2831853 / kDirs) * (float)s;
            float2 off = float2(cos(ang), sin(ang)) * radii[r] * texel;
            float3 tap = gTex.Sample(gSmp, uv + off).rgb;
            float  luma = dot(tap, kLuma);
            float3 bright = tap * max(luma - threshold, 0.0);
            sum  += bright * rw[r];
            wsum += rw[r];
        }
    }
    return sum / max(wsum, 1e-4);
}

float4 main(PSInput i) : SV_Target {
    float w, h;
    gTex.GetDimensions(w, h);
    float2 texel = 1.0 / float2(w, h);

    float3 c = Fxaa(i.uv, texel);              // anti-alias base scene (linear/HDR-ish)

    // EXPOSURE: the ONLY difference from post.frag — read the auto-exposure value the reduce pass wrote
    // instead of the fixed kExposure = 1.7. With adaptationEnabled=false the reduce wrote E0 == 1.7, so
    // this is the EXACT same multiply the default tonemap does -> byte-identical output (the no-op proof).
    float kExposure = gExposure[0];
    c *= kExposure;

    const float glowStrength = 0.65;
    float3 glow = Glow(i.uv, texel);
    c += glow * glowStrength;                  // add bloom before tonemapping

    c = ACES(c);                               // ACES filmic tonemap
    c = pow(c, 1.0 / 2.2);                     // gamma -> displayed LDR color

    c = ColorGrade(c);                         // cinematic split-tone + contrast + saturation
    c = FilmGrain(c, i.uv, float2(w, h));      // subtle static grain

    float2 d = i.uv - 0.5;
    float vig = smoothstep(0.8, 0.35, length(d));  // darken corners
    return float4(saturate(c * vig), 1.0);
}
