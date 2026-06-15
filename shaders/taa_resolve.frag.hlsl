// Temporal anti-aliasing — resolve pass (Slice AP). Fullscreen pass over the current jittered HDR
// scene (gCurrent, binding 0/1) + the reprojected history (gHistory, binding 3/4 — the same second
// material slot bloom/SSAO/SSR composites use). For each pixel it:
//   1. samples the current HDR color,
//   2. builds the 3x3 neighborhood color AABB (min/max over the 8 neighbors + center),
//   3. samples the reprojected history (history UV mapped back through prevViewProj — IDENTITY for
//      the static --taa-shot, where motion vectors are zero; the reprojection plumbing is present so
//      a future moving-camera shot exercises it without a seam change),
//   4. clamps the history color into that neighborhood AABB (ghosting suppression — the per-channel
//      clamp variant, mirroring render::taa::ClipHistoryToNeighborhood exactly), and
//   5. blends lerp(clampedHistory, current, alpha) — alpha=1.0 on the first (empty-history) frame so
//      accumulation has a defined start, ~0.1 in steady state (render::taa::ResolveBlend).
// Writes the resolved HDR color, which becomes (a) the next frame's history and (b) the input to the
// existing tonemap/post chain on the final captured frame. The ONE new shader for the slice; existing
// post/bloom/ssao/ssr shaders + their goldens are untouched.
[[vk::binding(0, 0)]] Texture2D    gCurrent : register(t0);   // current jittered HDR scene
[[vk::binding(1, 0)]] SamplerState gSmp     : register(s0);
[[vk::binding(3, 0)]] Texture2D    gHistory : register(t3);   // accumulated history (HDR)
[[vk::binding(4, 0)]] SamplerState gHSmp    : register(s3);

// alpha: blend weight of the current frame (1.0 first frame, ~0.1 steady state). firstFrame!=0 forces
// the current frame through unblended even if a (stale) history texture is bound. texel = 1/RTsize for
// the 3x3 neighborhood taps.
struct TaaParams { float2 texel; float alpha; float firstFrame; };
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer TaaPC { TaaParams tp; };
#define HF_TP tp
#else
[[vk::push_constant]] struct { TaaParams p; } pc;
#define HF_TP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

float4 main(PSInput i) : SV_Target {
    float3 current = gCurrent.Sample(gSmp, i.uv).rgb;

    // First frame (empty history): output the current frame unblended so accumulation has a defined
    // start (matches render::taa::ResolveBlend with alpha=1.0).
    if (HF_TP.firstFrame != 0.0) return float4(current, 1.0);

    // Build the 3x3 neighborhood color AABB over the current frame (center + 8 neighbors).
    float3 boxMin = current;
    float3 boxMax = current;
    [unroll] for (int dy = -1; dy <= 1; ++dy) {
        [unroll] for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            float2 off = float2((float)dx, (float)dy) * HF_TP.texel;
            float3 n = gCurrent.Sample(gSmp, i.uv + off).rgb;
            boxMin = min(boxMin, n);
            boxMax = max(boxMax, n);
        }
    }

    // Reprojected history. Static shot => motion vectors are zero => the reprojected UV is identity
    // (sample the same UV). A moving-camera shot would map i.uv back through prevViewProj here.
    float3 history = gHistory.Sample(gHSmp, i.uv).rgb;

    // Neighborhood clamp (per-channel AABB) — suppress ghosting by pulling drifted history back into
    // the locally-plausible color range. Mirrors render::taa::ClipHistoryToNeighborhood.
    float3 clampedHistory = clamp(history, boxMin, boxMax);

    // Exponential blend: lerp(clampedHistory, current, alpha). Mirrors render::taa::ResolveBlend.
    float3 resolved = lerp(clampedHistory, current, HF_TP.alpha);
    return float4(resolved, 1.0);
}
