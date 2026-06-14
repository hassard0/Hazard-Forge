[[vk::binding(0, 0)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 0)]] SamplerState gSmp : register(s0);
struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

static const float3 kLuma = float3(0.299, 0.587, 0.114);

// Lightweight FXAA (NVIDIA FXAA3 "console" style): detect luma edges from the 4 corner
// neighbours, then blend along the edge direction. Smooths geometry aliasing.
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

// ACES filmic tonemapping (Narkowicz approximation). Richer, more cinematic
// roll-off in the highlights than Reinhard, with a gentle toe in the shadows.
float3 ACES(float3 x) {
    float a = 2.51, b = 0.03, c2 = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c2 * x + d) + e));
}

// Cinematic color grade applied to the displayed LDR color (0..1, post-gamma):
//  - split-tone: lift shadows slightly cool/teal, push highlights slightly warm
//  - gentle contrast S-curve and a small saturation boost around luma
float3 ColorGrade(float3 c) {
    float luma = dot(c, kLuma);

    // Split-tone: cool/teal in shadows, warm in highlights.
    const float3 shadowTint = float3(0.90, 1.00, 1.06);  // cool / teal
    const float3 highTint   = float3(1.06, 1.00, 0.92);  // warm
    const float  gradeAmt   = 0.08;                       // overall split-tone strength
    float3 tint = lerp(shadowTint, highTint, luma);
    c = lerp(c, c * tint, gradeAmt);

    // Gentle contrast S-curve.
    c = lerp(c, smoothstep(0.0, 1.0, c), 0.15);

    // Small saturation boost around luma.
    float gluma = dot(c, kLuma);
    c = lerp(gluma.xxx, c, 1.08);

    return c;
}

// Subtle static film grain. Hash the pixel coordinate to ~+/-0.025 and add it,
// scaling the grain down in highlights so it lives mostly in midtones/shadows.
float3 FilmGrain(float3 c, float2 uv, float2 res) {
    float n = frac(sin(dot(uv * res, float2(12.9898, 78.233))) * 43758.5453);
    float grain = (n - 0.5) * 2.0 * 0.025;        // ~ +/-0.025
    float luma = dot(c, kLuma);
    grain *= 1.0 - smoothstep(0.5, 1.0, luma);    // fade out in highlights
    return c + grain;
}

// Screen-space glow / cheap bloom: bright-pass a ring kernel around the texel,
// keeping only luma above a threshold, then average into a soft bloom color.
float3 Glow(float2 uv, float2 texel) {
    const float threshold = 0.8;   // bright-pass cutoff
    // 24 taps: 8 directions across 3 radii (in texels). Inner taps weighted
    // a touch higher so the bloom falls off softly with distance.
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
            float3 bright = tap * max(luma - threshold, 0.0);  // keep only the bright excess
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
