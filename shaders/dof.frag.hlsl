// Depth of field — thin-lens circle-of-confusion depth gather (Slice CG). A fullscreen pass (reuse
// post.vert) reading the resolved HDR scene color (t0/s0) + the view-space normal+linear-depth
// G-buffer (t3/s3, the SAME g-buffer SSR/SSAO/SSGI write/read, bound via BindTexturePair). For each
// pixel it computes the CENTER circle of confusion from the thin-lens model (engine/render/dof.h
// CircleOfConfusion, copied verbatim below), then GATHERS a fixed Vogel-spiral disk of neighbour taps:
// each tap's own CoC is computed from its depth and it contributes to the centre with the scatter-as-
// gather BlurWeight (engine/render/dof.h, copied verbatim) — a tap only deposits colour where ITS CoC
// disk reaches the centre. This is DEPTH-AWARE: a sharp focal subject (CoC ~ 0) does not bleed onto the
// blurred background (its taps can't reach neighbours), and a near blurred object correctly spreads
// over the focal subject (its large CoC reaches across). The disk RADIUS in pixels is the centre CoC,
// so in-focus pixels gather only themselves (sharp) and defocused pixels gather a wide blur. After the
// gather the result is exposure/ACES/grade/grain/vignette tonemapped to the LDR swapchain (the SAME
// displayed-image pipeline as post.frag / ssr_composite, replicated here — no shared HLSL include).
//
// DETERMINISM: the Vogel disk + a baked per-pixel dither rotation (like SSR) — NO time/RNG. Two runs
// byte-identical. DoF is a NEW path behind --dof-shot; existing post/ssr/ssgi shaders + their goldens
// are UNTOUCHED.
[[vk::binding(0, 0)]] Texture2D    gScene : register(t0);   // resolved HDR scene colour
[[vk::binding(1, 0)]] SamplerState gSmp   : register(s0);
[[vk::binding(3, 0)]] Texture2D    gGbuf  : register(t3);   // view normal.xyz + linDepth.w
[[vk::binding(4, 0)]] SamplerState gGSmp  : register(s3);

struct DofParams {
    float2 texel;        // 1/size
    float  focalDist;    // view-linear depth of the focal plane
    float  aperture;     // entrance-pupil diameter (folds in the view-units -> px scale)
    float  focalLength;  // lens focal length (same view units)
    float  maxCoCpx;     // CoC clamp / gather-disk radius cap (pixels)
    float2 pad;
};
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer DofPC { DofParams dp; };
#define HF_DP dp
#else
[[vk::push_constant]] struct { DofParams p; } pc;
#define HF_DP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };
static const float3 kLuma = float3(0.299, 0.587, 0.114);

// Thin-lens CoC RADIUS in pixels — IDENTICAL to engine/render/dof.h CircleOfConfusion:
//   coc = aperture * |focalLength*(depth - focalDist)| / (depth * (focalDist - focalLength))
// guarded at depth -> 0 + a degenerate (focalDist==focalLength) lens, clamped to [0, maxCoCpx].
float CircleOfConfusion(float depth) {
    float d = max(depth, 1e-3);
    float denomFocus = HF_DP.focalDist - HF_DP.focalLength;
    if (abs(denomFocus) < 1e-4) denomFocus = (denomFocus < 0.0) ? -1e-4 : 1e-4;
    float numer = HF_DP.aperture * abs(HF_DP.focalLength * (depth - HF_DP.focalDist));
    float denom = d * denomFocus;
    float coc = numer / denom;
    if (!(coc == coc) || coc < 0.0) coc = 0.0;   // NaN/negative guard (coc==coc is false for NaN)
    return min(coc, HF_DP.maxCoCpx);
}

// Scatter-as-gather weight — IDENTICAL to engine/render/dof.h BlurWeight: a neighbour reaches the
// centre iff its CoC disk covers it (tapDistPx <= tapCoCpx), normalized by the disk area.
float BlurWeight(float tapCoCpx, float tapDistPx) {
    if (tapDistPx > tapCoCpx + 1e-4) return 0.0;
    float r = max(tapCoCpx, 1.0);
    return 1.0 / (3.14159265358979 * r * r);
}

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

static const int kTaps = 48;                       // Vogel-spiral disk taps
static const float kGoldenAngle = 2.39996322972865; // radians: pi*(3 - sqrt(5))
static const float kExposure = 1.7;                 // matches post.frag / ssr_composite

float4 main(PSInput i) : SV_Target {
    float2 res = float2(1.0 / HF_DP.texel.x, 1.0 / HF_DP.texel.y);
    float3 centerCol = gScene.Sample(gSmp, i.uv).rgb;
    float  centerDepth = gGbuf.Sample(gGSmp, i.uv).w;

    // The centre's own CoC sets the gather-disk radius. Background (depth==0) = max blur (far field).
    float centerCoC = (centerDepth > 0.0001)
        ? CircleOfConfusion(centerDepth)
        : HF_DP.maxCoCpx;

    float3 sum = float3(0.0, 0.0, 0.0);
    float  wsum = 0.0;

    // The centre tap always contributes itself (so an in-focus pixel == its own colour, sharp).
    {
        float w = BlurWeight(max(centerCoC, 1.0), 0.0);
        sum += w * centerCol;
        wsum += w;
    }

    // Baked per-pixel dither rotation (Bayer/16 -> a fraction of a turn) breaks the spiral's banding
    // without any RNG/time — deterministic per pixel, byte-stable golden (same idea as ssr.frag).
    int2 px = int2(i.uv * res);
    const float kDither[16] = {
        0.0/16.0,  8.0/16.0,  2.0/16.0, 10.0/16.0,
       12.0/16.0,  4.0/16.0, 14.0/16.0,  6.0/16.0,
        3.0/16.0, 11.0/16.0,  1.0/16.0,  9.0/16.0,
       15.0/16.0,  7.0/16.0, 13.0/16.0,  5.0/16.0,
    };
    float rot = kDither[(px.x & 3) + ((px.y & 3) << 2)] * 6.2831853;

    // Gather a Vogel-spiral disk whose radius is the centre CoC (sharp pixels => ~0 radius => no blur).
    float radius = max(centerCoC, 0.0);
    [loop] for (int t = 0; t < kTaps; ++t) {
        float fi = (float)t + 0.5;
        float r = radius * sqrt(fi / (float)kTaps);    // area-uniform spiral
        float a = fi * kGoldenAngle + rot;
        float2 offPx = float2(cos(a), sin(a)) * r;
        float2 tuv = i.uv + offPx * HF_DP.texel;

        float tapDepth = gGbuf.Sample(gGSmp, tuv).w;
        float tapCoC = (tapDepth > 0.0001) ? CircleOfConfusion(tapDepth) : HF_DP.maxCoCpx;

        // Depth-aware scatter-as-gather: the tap contributes only if ITS CoC disk reaches the centre.
        float dist = length(offPx);
        float w = BlurWeight(tapCoC, dist);
        sum  += w * gScene.Sample(gSmp, tuv).rgb;
        wsum += w;
    }

    float3 c = (wsum > 1e-8) ? (sum / wsum) : centerCol;

    // Same displayed-image pipeline as post.frag / ssr_composite.
    c *= kExposure;
    c = ACES(c);
    c = pow(c, 1.0 / 2.2);
    c = ColorGrade(c);
    c = FilmGrain(c, i.uv, res);
    float2 d = i.uv - 0.5;
    float vig = smoothstep(0.8, 0.35, length(d));
    return float4(saturate(c * vig), 1.0);
}
