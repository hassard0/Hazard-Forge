#pragma once
// Slice CN — Per-Object + Camera Motion Blur math — pure CPU (header-only, no device, no backend
// symbols). Shared by the --motionblur-shot showcase reasoning, the in-shader velocity gather
// (shaders/motion_blur.frag.hlsl runs the SAME ScreenVelocity + ClampVelocity + TapWeight), AND
// tests/motion_blur_test.cpp — so the unit test exercises the EXACT velocity + gather weight the GPU
// pass uses.
//
// Mirrors ssr.h / dof.h / cluster.h: a tiny shared-math header above the RHI seam (ZERO vk*/MTL*/
// mtl::/Backend::Metal symbols). The motion-blur shader copies these three functions verbatim so the
// CPU test and the GPU pass agree on the velocity field + the depth-aware gather.
//
// THE TECHNIQUE (velocity-gather motion blur, McGuire-style, simplified, single fixed shutter):
//   A surface point seen this frame was at a DIFFERENT screen position last frame (the camera and/or
//   the object moved). Its screen-space VELOCITY v is the cur-frame pixel position minus the prev-frame
//   pixel position. We gather N taps stepping along v, each weighted by a depth-aware TapWeight, and
//   output the normalized weighted sum: a moving surface SMEARS along v, a static surface (v == 0) does
//   not move at all so every tap lands on the center and the result is the center color UNCHANGED.
//
// THE ZERO-VELOCITY EQUIVALENCE PROOF (what makes this golden-safe): when prevViewProj == curViewProj
//   (and the point is static), ScreenVelocity returns EXACTLY (0,0). With v == 0 every gather tap lands
//   on the center pixel (tapDistPx == 0 <= velLenPx == 0) and TapWeight is NORMALIZED so the center tap
//   alone has weight 1 and every other tap has weight 0 — so the normalized sum is the center color,
//   BYTE-IDENTICAL to the un-blurred scene. No energy added/lost, no off-by-one. The showcase asserts
//   exactly this (motionblur(zeroVel) == sceneColor, SHA), the same internal-assert discipline as the
//   CL clustered==brute-force and CJ Hi-Z==frustum proofs.
//
// CONVENTIONS (must match engine/math + the shader EXACTLY):
//   * Mat4 is column-major; MulPointDivide(m, p, w) applies the perspective divide and exposes clip w.
//   * NDC -> PIXEL convention: a clip-space point projects to NDC xy in [-1,1] (post-divide). We map it
//     to PIXEL coordinates with the STANDARD half-pixel-extent scaling, INDEPENDENT of any backend
//     Y-flip baked into the projection:
//         pxX = (ndcX * 0.5 + 0.5) * screenW
//         pxY = (ndcY * 0.5 + 0.5) * screenH
//     Because the velocity is the DIFFERENCE of two points pushed through the SAME pair of matrices in
//     the SAME NDC->pixel mapping, the constant +0.5 offsets cancel and any consistent clip-Y sign
//     cancels too — the screen-space delta is identical regardless of the backend's clip-Y convention,
//     so the velocity field (hence the golden) is bit-identical on Vulkan + Metal. We deliberately keep
//     the screen-space delta a SIGNED pixel vector (cur - prev): its sign is the on-screen motion
//     direction, its length the per-frame travel in pixels.
//   * A point BEHIND either camera (clip w <= 0) has no well-defined screen velocity; we return (0,0)
//     there (it is off-screen / occluded anyway) so the gather degenerates to a pass-through rather
//     than smearing along a garbage direction.

#include "math/math.h"

#include <algorithm>
#include <cmath>

namespace hf::render::motionblur {

// --- Screen-space velocity -----------------------------------------------------------------------
// ScreenVelocity(worldPos, prevViewProj, curViewProj, screenW, screenH) -> the per-frame screen-space
// motion of `worldPos` in PIXELS, as the SIGNED delta (cur - prev). Project the SAME world point by
// both view-proj matrices to NDC, map each to pixels with the documented NDC->pixel convention, and
// subtract. Properties the test + the shader rely on:
//   * prevViewProj == curViewProj  ->  EXACTLY (0,0) for ANY point (the two projections are identical,
//     so cur-pixel == prev-pixel bit-for-bit). This is the heart of the zero-velocity proof.
//   * a camera/object motion produces a velocity whose direction is the on-screen displacement and
//     whose magnitude grows with the motion magnitude (and the frame dt the caller folds into the
//     prev matrix).
//   * a point behind either camera (clip w <= 0) -> (0,0) (no smear along an ill-defined direction).
inline math::Vec2 ScreenVelocity(const math::Vec3& worldPos, const math::Mat4& prevViewProj,
                                 const math::Mat4& curViewProj, int screenW, int screenH) {
    float wPrev = 0.0f, wCur = 0.0f;
    math::Vec3 ndcPrev = math::MulPointDivide(prevViewProj, worldPos, wPrev);
    math::Vec3 ndcCur  = math::MulPointDivide(curViewProj,  worldPos, wCur);
    // Behind either camera -> no valid screen velocity (treat as static -> pass-through).
    if (wPrev <= 1e-6f || wCur <= 1e-6f) return math::Vec2{0.0f, 0.0f};

    float sw = static_cast<float>(screenW);
    float sh = static_cast<float>(screenH);
    // NDC -> pixel (the +0.5 offsets cancel in the subtraction; kept explicit to document the mapping).
    float pxPrevX = (ndcPrev.x * 0.5f + 0.5f) * sw;
    float pxPrevY = (ndcPrev.y * 0.5f + 0.5f) * sh;
    float pxCurX  = (ndcCur.x  * 0.5f + 0.5f) * sw;
    float pxCurY  = (ndcCur.y  * 0.5f + 0.5f) * sh;
    return math::Vec2{pxCurX - pxPrevX, pxCurY - pxPrevY};
}

// --- Velocity clamp ------------------------------------------------------------------------------
// ClampVelocity(vPx, maxBlurPx) -> the velocity with its LENGTH clamped to maxBlurPx, DIRECTION
// preserved. Caps the blur extent (perf bound + an artifact bound — a huge motion would otherwise
// gather across the whole frame). A zero or sub-threshold velocity passes through unchanged (and a
// zero velocity stays exactly (0,0), preserving the pass-through proof).
inline math::Vec2 ClampVelocity(const math::Vec2& vPx, float maxBlurPx) {
    float len = std::sqrt(vPx.x * vPx.x + vPx.y * vPx.y);
    if (len <= maxBlurPx || len <= 1e-8f) return vPx;           // within budget (or zero) -> unchanged
    float s = maxBlurPx / len;                                  // scale to the cap, keep direction
    return math::Vec2{vPx.x * s, vPx.y * s};
}

// --- Depth-aware gather tap weight ---------------------------------------------------------------
// TapWeight(tapDepth, centerDepth, tapDistPx, velLenPx) -> the (already-normalized) contribution of a
// tap at screen distance `tapDistPx` along the velocity to the center pixel, given the tap's and the
// center's VIEW-LINEAR depths (positive = in front of the camera; the SAME .w the g-buffer stores) and
// the center pixel's blur extent `velLenPx` (== |clamped velocity|).
//
// GATHER MODEL (McGuire-style fore/background classification, simplified):
//   * A tap contributes only if it lies WITHIN the center's blur extent: tapDistPx <= velLenPx (a tiny
//     slack so the center tap, dist 0, always counts itself).
//   * DEPTH-AWARE foreground/background rule — so a moving NEARER foreground streaks OVER a static
//     background while a static foreground is NOT smeared by a moving background:
//       - If the tap is NEARER than (or equal to, within epsilon) the center (tapDepth <= centerDepth):
//         the tap is a foreground sample sweeping across the center -> it CONTRIBUTES (a near moving
//         object correctly streaks over whatever is behind it).
//       - If the tap is FARTHER than the center (a background sample): it must NOT bleed onto a nearer
//         (possibly static) center, so it contributes ~0.
//   * NORMALIZED so zero velocity -> the center tap (dist 0, depth == center) has weight 1 and every
//     non-center tap (dist > 0 > velLenPx == 0) has weight 0. Each contributing tap returns a UNIFORM
//     weight 1; the caller divides by the accumulated weight sum, so with v == 0 the sum is exactly the
//     center color (weight 1 / total 1) — byte-identical to the input. A uniform box weight over the
//     in-extent foreground taps keeps the gather a simple, golden-stable average (no shutter curve;
//     YAGNI per the spec).
inline float TapWeight(float tapDepth, float centerDepth, float tapDistPx, float velLenPx) {
    // Outside the blur extent -> no contribution (sharp regions don't gather distant taps).
    if (tapDistPx > velLenPx + 1e-4f) return 0.0f;
    // Background tap (strictly farther than the center) must not smear a nearer center.
    if (tapDepth > centerDepth + 1e-4f) return 0.0f;
    // In-extent foreground/co-planar tap: uniform box weight (normalized average by the caller).
    return 1.0f;
}

}  // namespace hf::render::motionblur
