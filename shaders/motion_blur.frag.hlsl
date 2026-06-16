// Per-object + camera motion blur — velocity-gather post pass (Slice CN). A fullscreen pass (reuse
// post.vert) reading the resolved HDR scene color (t0/s0) + the view-space normal+linear-depth
// G-buffer (t3/s3, the SAME g-buffer SSR/SSAO/DoF write/read, bound via BindTexturePair). For each
// pixel it reconstructs the VIEW-space position from the G-buffer linear depth (exactly like SSR/DoF),
// computes the screen-space VELOCITY from the prev/cur view-proj (render::motionblur::ScreenVelocity,
// mirrored below), clamps it to maxBlurPx (ClampVelocity), then GATHERS N fixed taps stepping along
// the velocity, each weighted by the depth-aware TapWeight (render/motion_blur.h, copied verbatim) so
// a moving foreground streaks over a static background while a static foreground is not smeared by a
// moving background. The normalized weighted sum is exposure/ACES/grade/grain/vignette tonemapped to
// the LDR swapchain (the SAME displayed-image pipeline as post.frag / ssr_composite / dof.frag).
//
// THE ZERO-VELOCITY EQUIVALENCE PROOF: the prev pixel position is computed by projecting THIS pixel's
// reconstructed VIEW position through `prevClipFromView` == prevViewProj * inverse(curView); the cur
// pixel position is THIS pixel's own SV_Position. When prevViewProj == curViewProj the composed matrix
// reconstructs the SAME clip position, so prevPx == curPx and the velocity is EXACTLY zero -> every
// tap lands on the center (tapDist 0 <= velLen 0) with TapWeight 1 only at the center -> the normalized
// sum is the center color, BYTE-IDENTICAL to the un-blurred scene. (Same pass-through discipline as
// dof.frag's in-focus center tap.)
//
// DETERMINISM: a baked 4x4 ordered-dither offsets the tap parameter by a sub-step fraction (like
// ssr/dof) to break the fixed-step banding — NO time/RNG. Two runs byte-identical. Motion blur is a
// NEW path behind --motionblur-shot; existing post/ssr/ssgi/dof shaders + their goldens are UNTOUCHED.
[[vk::binding(0, 0)]] Texture2D    gScene : register(t0);   // resolved HDR scene color
[[vk::binding(1, 0)]] SamplerState gSmp   : register(s0);
[[vk::binding(3, 0)]] Texture2D    gGbuf  : register(t3);   // view normal.xyz + linDepth.w
[[vk::binding(4, 0)]] SamplerState gGSmp  : register(s3);

struct MbParams {
    float4 prevClip0;    // prevViewProj * inverse(curView), column 0  (a column-major mat4 in 4 float4s)
    float4 prevClip1;    // column 1
    float4 prevClip2;    // column 2
    float4 prevClip3;    // column 3
    float2 texel;        // 1/size
    float  tanHalfFovY;  // view-space reconstruction (matches SSR/DoF)
    float  aspect;       // width/height
    float  maxBlurPx;    // velocity-length clamp + gather extent cap (pixels)
    float  taps;         // gather tap count (float-packed; cast to int)
    float  velScale;     // extra velocity scale folded onto ScreenVelocity (shutter strength)
    float  pad;
};
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer MbPC { MbParams mb; };
#define HF_MB mb
#else
[[vk::push_constant]] struct { MbParams p; } pc;
#define HF_MB pc.p
#endif
// 96 bytes (4*16 + 8 + 4*5 + 4 pad rounded) <= 128B push-constant budget on the integrated AMD GPU.

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };
static const float3 kLuma = float3(0.299, 0.587, 0.114);

// Y-flip sign mapping screen UV.y <-> view-space Y — IDENTICAL convention to ssr.frag / ssao.frag.
#ifdef HF_MSL_GEN
static const float HF_YS = 1.0;
#else
static const float HF_YS = -1.0;
#endif

// Reconstruct the VIEW-space position from the screen UV + linear depth (mirrors ssr.frag).
float3 ReconstructViewPos(float2 uv, float linDepth) {
    float2 ndc = uv * 2.0 - 1.0;
    float vx = ndc.x * (HF_MB.aspect * HF_MB.tanHalfFovY) * linDepth;
    float vy = HF_YS * ndc.y * (HF_MB.tanHalfFovY) * linDepth;
    float vz = -linDepth;                   // RH view space: -Z forward
    return float3(vx, vy, vz);
}

// Multiply a column-major mat4 (4 columns) by a homogeneous point (w=1); returns clip xyzw.
float4 MulPrevClip(float3 vp) {
    return HF_MB.prevClip0 * vp.x + HF_MB.prevClip1 * vp.y + HF_MB.prevClip2 * vp.z + HF_MB.prevClip3;
}

// Depth-aware gather tap weight — IDENTICAL to engine/render/motion_blur.h TapWeight: an in-extent
// foreground/co-planar tap contributes (weight 1); a beyond-extent or strictly-farther background tap
// contributes 0. Normalized so zero velocity -> the center tap (dist 0) alone is weight 1.
float TapWeight(float tapDepth, float centerDepth, float tapDistPx, float velLenPx) {
    if (tapDistPx > velLenPx + 1e-4) return 0.0;
    if (tapDepth > centerDepth + 1e-4) return 0.0;
    return 1.0;
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

static const float kExposure = 1.7;                 // matches post.frag / ssr_composite / dof.frag
// Baked 4x4 ordered-dither (Bayer/16) in [0,1): offsets the tap parameter by a sub-step fraction so the
// fixed tap grid doesn't band. Deterministic per pixel -> byte-stable golden (same idea as ssr/dof).
static const float kDither[16] = {
    0.0 / 16.0,  8.0 / 16.0,  2.0 / 16.0, 10.0 / 16.0,
   12.0 / 16.0,  4.0 / 16.0, 14.0 / 16.0,  6.0 / 16.0,
    3.0 / 16.0, 11.0 / 16.0,  1.0 / 16.0,  9.0 / 16.0,
   15.0 / 16.0,  7.0 / 16.0, 13.0 / 16.0,  5.0 / 16.0,
};

float4 main(PSInput i) : SV_Target {
    float2 res = float2(1.0 / HF_MB.texel.x, 1.0 / HF_MB.texel.y);
    float3 centerCol = gScene.Sample(gSmp, i.uv).rgb;
    float  centerDepth = gGbuf.Sample(gGSmp, i.uv).w;

    // --- Screen-space velocity (cur - prev) in PIXELS. ---
    // cur pixel position == this fragment's own pixel coordinate (SV_Position is pixel-centered, in the
    // backend's screen orientation, matching how the prev clip below maps to pixels).
    float2 curPx = i.uv * res;
    float2 vel = float2(0.0, 0.0);
    if (centerDepth > 0.0001) {
        float3 vpos = ReconstructViewPos(i.uv, centerDepth);
        float4 prevClip = MulPrevClip(vpos);
        if (prevClip.w > 1e-6) {
            float2 prevNdc = prevClip.xy / prevClip.w;
            float2 prevPx = (prevNdc * 0.5 + 0.5) * res;
            vel = (curPx - prevPx) * HF_MB.velScale;
        }
    }
    // Clamp the velocity length to maxBlurPx, preserving direction (ClampVelocity).
    float velLen = length(vel);
    if (velLen > HF_MB.maxBlurPx && velLen > 1e-8) {
        vel *= HF_MB.maxBlurPx / velLen;
        velLen = HF_MB.maxBlurPx;
    }

    int   N = (int)(HF_MB.taps + 0.5);
    int2  px = int2(curPx);
    float dither = kDither[(px.x & 3) + ((px.y & 3) << 2)];

    // --- Gather N taps stepping along the velocity, weighted by the depth-aware TapWeight. ---
    // Tap j samples at the center +/- a fraction of the velocity vector. ZERO-VELOCITY SHORT-CIRCUIT:
    // when the velocity is (sub-pixel) zero every tap would land on the center anyway; we output the
    // center color DIRECTLY so the result is BIT-EXACT the un-blurred scene color (no N-fold average
    // rounding) — this is the byte-identical pass-through proof. A non-zero velocity runs the gather.
    float3 c;
    if (velLen <= 1e-6) {
        c = centerCol;
    } else {
        float3 sum = float3(0.0, 0.0, 0.0);
        float  wsum = 0.0;
        [loop] for (int j = 0; j < N; ++j) {
            // Parameter in [-1,1] across the taps, dithered by a sub-step fraction to break banding.
            float t = (N > 1) ? ((((float)j + dither) / (float)(N - 1)) * 2.0 - 1.0) : 0.0;
            float2 offPx = vel * (0.5 * t);            // sweep +/- half the velocity around the center
            float2 tuv = i.uv + offPx * HF_MB.texel;
            float  tapDepth = gGbuf.Sample(gGSmp, tuv).w;
            float  tapDist = length(offPx);
            // Background taps (depth==0) read as "very far" so they never smear a nearer center.
            float  tapDepthCmp = (tapDepth > 0.0001) ? tapDepth : 1e9;
            float  centerCmp   = (centerDepth > 0.0001) ? centerDepth : 1e9;
            float  w = TapWeight(tapDepthCmp, centerCmp, tapDist, velLen);
            sum  += w * gScene.Sample(gSmp, tuv).rgb;
            wsum += w;
        }
        c = (wsum > 1e-8) ? (sum / wsum) : centerCol;
    }

    // Same displayed-image pipeline as post.frag / ssr_composite / dof.frag.
    c *= kExposure;
    c = ACES(c);
    c = pow(c, 1.0 / 2.2);
    c = ColorGrade(c);
    c = FilmGrain(c, i.uv, res);
    float2 d = i.uv - 0.5;
    float vig = smoothstep(0.8, 0.35, length(d));
    return float4(saturate(c * vig), 1.0);
}
