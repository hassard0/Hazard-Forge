#pragma once
// Temporal anti-aliasing math — pure CPU (header-only, no device, no backend symbols). Same pattern
// as engine/render/ssr.h. Namespace hf::render::taa. Shared by the --taa-shot showcase (which builds
// the per-frame jittered projection from Jitter) AND tests/taa_test.cpp, so the unit test exercises
// the SAME jitter / neighborhood-clamp / resolve-blend math the in-shader resolve (taa_resolve.frag)
// uses.
//
// TAA accumulates several sub-pixel-jittered frames into a smooth, alias-free image:
//   * each accumulation frame jitters the projection by a sub-pixel Halton offset (Jitter),
//   * the resolve pass blends the current frame with reprojected history, clamped to the current
//     3x3 neighborhood AABB (ClipHistoryToNeighborhood) to suppress ghosting,
//   * the blend is an exponential moving average (ResolveBlend): alpha=1 on the first (empty-history)
//     frame so accumulation has a defined start, ~0.1 in steady state.
// The jitter sequence is a pure function of the frame index (Halton(2,3), no time, no RNG), so two
// runs are bit-identical and the same index gives the same offset on Vulkan and Metal.

#include "math/math.h"
#include <algorithm>
#include <cmath>

namespace hf::render::taa {

// math.h has no Vec2; a TAA jitter offset is a 2D NDC pair. Plain aggregate so the test reads .x/.y.
struct Vec2 { float x = 0, y = 0; };

// N = fixed accumulation-frame count for the deterministic golden shot.
inline constexpr int kAccumFrames = 8;
// Exponential-blend weight in steady state (fraction of the CURRENT frame mixed into history).
inline constexpr float kSteadyAlpha = 0.1f;

// Halton low-discrepancy sequence: the radical inverse of `index` in `base`. `index` is 1-based
// (Halton(base,0) would be 0), so the showcase passes frameIndex+1. Returns a value in [0,1).
// Halton(2,i) and Halton(3,i) give the X/Y bases of the (2,3) jitter sequence. Pure + deterministic.
inline float Halton(int base, int index) {
    float result = 0.0f;
    float f = 1.0f / (float)base;
    int i = index;
    while (i > 0) {
        result += f * (float)(i % base);
        i /= base;
        f /= (float)base;
    }
    return result;
}

// Sub-pixel jitter offset in NDC space for a given accumulation frame. Halton(.,frameIndex+1) is in
// [0,1); centering by -0.5 yields [-0.5,0.5), and *2/dim converts a sub-pixel shift to NDC (NDC spans
// [-1,1] across `dim` pixels => 2/dim NDC per pixel). The result is added into the projection matrix's
// m[2][0]/m[2][1] (clip-space XY translation per unit W) AFTER the base projection is built, so it
// jitters screen XY without disturbing depth. |x| <= 1/width, |y| <= 1/height (one sub-pixel).
inline Vec2 Jitter(int frameIndex, int width, int height) {
    float hx = Halton(2, frameIndex + 1) - 0.5f;
    float hy = Halton(3, frameIndex + 1) - 0.5f;
    return Vec2{hx * 2.0f / (float)width, hy * 2.0f / (float)height};
}

// Clip the history color toward the current 3x3 neighborhood AABB. We use the simple per-channel
// clamp (NOT the full clip-toward-center variant — YAGNI for a static shot) — this is EXACTLY what
// taa_resolve.frag does (HLSL clamp(history, boxMin, boxMax)). Suppresses ghosting: history that has
// drifted outside the locally-plausible color range is pulled back to the nearest neighborhood face.
inline math::Vec3 ClipHistoryToNeighborhood(const math::Vec3& history,
                                            const math::Vec3& boxMin,
                                            const math::Vec3& boxMax) {
    return math::Vec3{
        std::clamp(history.x, boxMin.x, boxMax.x),
        std::clamp(history.y, boxMin.y, boxMax.y),
        std::clamp(history.z, boxMin.z, boxMax.z),
    };
}

// Exponential resolve blend: lerp(clampedHistory, current, alpha) = clampedHistory*(1-alpha) +
// current*alpha. alpha=1.0 on the first frame (empty history -> output the current frame unblended),
// ~0.1 in steady state (heavily weight accumulated history, fold in a little of the new frame).
// Mirrors taa_resolve.frag's final lerp exactly.
inline math::Vec3 ResolveBlend(const math::Vec3& current,
                               const math::Vec3& clampedHistory,
                               float alpha) {
    return math::Vec3{
        clampedHistory.x + (current.x - clampedHistory.x) * alpha,
        clampedHistory.y + (current.y - clampedHistory.y) * alpha,
        clampedHistory.z + (current.z - clampedHistory.z) * alpha,
    };
}

} // namespace hf::render::taa
