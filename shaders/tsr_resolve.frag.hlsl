// Temporal SUPER-RESOLUTION resolve pass (Slice US2, issue #20). A fork of taa_resolve.frag generalized
// to a RESOLUTION MISMATCH: the current frame (gCurrent) is the HALF-res jittered HDR scene, the history
// (gHistory) is the FULL-res accumulated HDR. Each accumulation frame the jittered low-res render fills a
// different sub-pixel slot of the high-res grid; folding N=8 jittered frames into the full-res history via
// a TRUE running-average (alpha = 1/(n+1)) supersamples the upscaled image — sharper than US1's single-
// frame bilinear. For each FULL-res output pixel it:
//   1. samples the current HALF-res HDR color at i.uv (the linear sampler bilinear-upscales it),
//   2. builds the 3x3 neighborhood color AABB over the LOW-res image using curTexel (=1/halfRes — the
//      clamp source is the low-res image, FSR2 convention),
//   3. samples the FULL-res reprojected history at i.uv (identity reprojection — static camera; the
//      reprojection plumbing stays present so US3's moving-camera shot exercises it without a seam change),
//   4. clamps the history color into that neighborhood AABB (ghosting suppression — per-channel clamp,
//      EXACTLY taa_resolve / render::taa::ClipHistoryToNeighborhood), and
//   5. blends lerp(clampedHistory, current, alpha) with alpha = 1/(n+1) so the N jittered samples average
//      into a clean supersample (alpha=1 on the first frame -> current unblended, defined start).
// Writes the resolved FULL-res HDR color, which becomes the next frame's history and (on the final frame)
// the input to the existing tonemap/post chain. NEW shader (fork of taa_resolve.frag); taa_resolve.frag /
// taa.h and all existing shaders/goldens untouched.
[[vk::binding(0, 0)]] Texture2D    gCurrent : register(t0);   // current HALF-res jittered HDR scene
[[vk::binding(1, 0)]] SamplerState gSmp     : register(s0);
[[vk::binding(3, 0)]] Texture2D    gHistory : register(t3);   // FULL-res accumulated history (HDR)
[[vk::binding(4, 0)]] SamplerState gHSmp    : register(s3);

// curTexel = 1/halfRes (the LOW-res neighborhood taps); histTexel = 1/fullRes (carried for symmetry /
// future full-res neighborhood use — present so the params match the US3 layout). alpha = 1/(n+1) running
// average; firstFrame!=0 forces the current frame through unblended even if a (stale) history is bound.
struct TsrParams { float2 curTexel; float2 histTexel; float alpha; float firstFrame; };
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer TsrPC { TsrParams tp; };
#define HF_TP tp
#else
[[vk::push_constant]] struct { TsrParams p; } pc;
#define HF_TP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

float4 main(PSInput i) : SV_Target {
    float3 current = gCurrent.Sample(gSmp, i.uv).rgb;

    // First frame (empty history): output the current frame unblended so accumulation has a defined
    // start (matches render::taa::ResolveBlend with alpha=1.0).
    if (HF_TP.firstFrame != 0.0) return float4(current, 1.0);

    // Build the 3x3 neighborhood color AABB over the LOW-res current frame (center + 8 neighbors), using
    // curTexel = 1/halfRes — the clamp source is the low-res image (FSR2 convention).
    float3 boxMin = current;
    float3 boxMax = current;
    [unroll] for (int dy = -1; dy <= 1; ++dy) {
        [unroll] for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            float2 off = float2((float)dx, (float)dy) * HF_TP.curTexel;
            float3 n = gCurrent.Sample(gSmp, i.uv + off).rgb;
            boxMin = min(boxMin, n);
            boxMax = max(boxMax, n);
        }
    }

    // Reprojected FULL-res history. Static shot => motion vectors are zero => the reprojected UV is
    // identity (sample the same UV). A moving-camera shot (US3) would map i.uv back through prevViewProj.
    float3 history = gHistory.Sample(gHSmp, i.uv).rgb;

    // Neighborhood clamp (per-channel AABB) — suppress ghosting by pulling drifted history back into the
    // locally-plausible color range. Mirrors render::taa::ClipHistoryToNeighborhood.
    float3 clampedHistory = clamp(history, boxMin, boxMax);

    // Running-average blend: lerp(clampedHistory, current, alpha) with alpha = 1/(n+1) so N jittered
    // samples average into a clean supersample. Mirrors render::taa::ResolveBlend.
    float3 resolved = lerp(clampedHistory, current, HF_TP.alpha);
    return float4(resolved, 1.0);
}
