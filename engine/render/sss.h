#pragma once
// Slice CZ — Subsurface Scattering (screen-space separable SSS; Jimenez et al. 2015, "Separable
// Subsurface Scattering") math — pure CPU (header-only, no device, no backend symbols). Namespace
// hf::render::sss. Mirrors dof.h / gtao.h / auto_exposure.h: a tiny shared-math header ABOVE the RHI
// seam (ZERO vk*/MTL*/mtl::/Backend::Metal CODE symbols — the only mentions of "vk"/"MTL" anywhere in
// this slice's above-seam files are seam-discipline doc comments + the [[vk::binding]] HLSL
// decorations). The SSS fragment shader (shaders/sss_blur.frag.hlsl) copies DiffusionWeight +
// DepthFalloff + BlurAxis VERBATIM, so tests/sss_test.cpp exercises the EXACT diffusion gather the GPU
// pass runs — which is what makes the sssStrength=0 (or sssWidth=0) render byte-identical to the
// non-SSS lit render AND bit-identical cross-backend.
//
// THE TECHNIQUE (screen-space separable SSS): light entering translucent skin/wax/marble scatters a
// short distance UNDER the surface before exiting, so the diffuse lit color of a flagged subsurface
// material should be SOFTENED by a small radius — the soft translucent glow that wraps the terminator
// and fills hard shadow edges. The full subsurface diffusion is a 2D convolution of the lit color with
// a radially-symmetric skin diffusion profile; Jimenez's key insight is that the profile is
// well-approximated as SEPARABLE, so the 2D blur is done as a HORIZONTAL 1D pass followed by a VERTICAL
// 1D pass (two cheap passes instead of one expensive 2D gather). Each 1D pass gathers `taps` samples
// stepping along the pass axis, weighted by the diffusion profile (DiffusionWeight) AND a depth-aware
// falloff (DepthFalloff) so the blur does NOT bleed across a silhouette / depth discontinuity, AND the
// subsurface mask so non-subsurface pixels pass through unchanged. The step length scales with
// sssStrength * sssWidth / centerDepth (a closer surface subtends more pixels per world unit, so the
// screen-space radius grows as it approaches the camera — perspective-correct diffusion).
//
// THE sssStrength=0 / sssWidth=0 NO-OP PROOF (what makes this golden-safe — like CR radius=0==no-AO,
// CT maxDist=0==no-contact, CW adaptation-off==fixed-exposure): the per-tap step is
// sssStrength * sssWidth / centerDepth. With sssStrength == 0 (or sssWidth == 0) the step is 0 → every
// one of the `taps` samples lands on the CENTER pixel → the normalized weighted sum of identical
// center colors equals the center color EXACTLY → BlurAxis returns the center color, and the two passes
// compose to a pure pass-through. So the showcase renders the SSS pass at sssStrength=0 and asserts it
// is BYTE-IDENTICAL (SHA) to the engine's standard lit render of the same scene (no constant bias, no
// normalization error, no off-by-one) — then renders the real sssStrength>0 version as the golden. To
// make this UNCONDITIONAL + branch-clean we ALSO short-circuit sssStrength<=0 || sssWidthPx<=0 to
// return the center color directly (so the identity is exact regardless of the depth/mask field). The
// unit test pins DiffusionWeight (positive, peaks at 0, strictly decreasing, wider width spreads
// weight), DepthFalloff (1 at equal depth, → 0 across a depth discontinuity, monotone), and
// BlurAxis(sssStrength=0)==center / BlurAxis(sssWidthPx=0)==center EXACTLY.
//
// CONVENTIONS (match the SSAO/SSR/DoF G-buffer EXACTLY):
//   * `depth` is VIEW-SPACE LINEAR depth = -vpos.z (positive in front of the camera), the value
//     gbuffer.frag stores in .w and the shader reconstructs from. centerDepth==0 (cleared G-buffer /
//     background) is treated as "no surface" -> no diffusion.
//   * All kernel offsets are in PIXELS along the pass axis (the gather walks the screen by pixels).
//     `axisPx` is the per-tap-step screen offset for THIS pass (horizontal: (1,0); vertical: (0,1)),
//     scaled inside by the perspective-correct step length.
//   * The subsurface MASK is in [0,1]: 1 = flagged subsurface material, 0 = opaque (pass-through). The
//     shader sources it from a dedicated mask RT (the subsurface objects rendered white, everything
//     else black) — documented in sss_blur.frag.hlsl.
//
// Pure, deterministic functions: no RNG, no time, fixed tap count.

#include "math/math.h"

#include <algorithm>
#include <cmath>

namespace hf::render::sss {

inline constexpr float kPi = 3.14159265358979323846f;

// --- Skin diffusion profile weight ---------------------------------------------------------------
// DiffusionWeight(distancePx, sssWidthPx) -> the (un-normalized) subsurface diffusion weight at a tap
// `distancePx` pixels from the center, for a profile whose characteristic width is `sssWidthPx` pixels.
//
// PROFILE (documented): a single normalized Gaussian falloff
//     w(d) = exp( -kFalloff * (d / width)^2 )
// with kFalloff = 3.0. This is the standard screen-space-SSS choice: a Gaussian is the separable
// building block Jimenez fits the multi-Gaussian skin profile from, and a single Gaussian of the
// surface's subsurface width captures the soft single-scatter glow this slice targets (the full
// sum-of-Gaussians authoring is explicit YAGNI in the spec). kFalloff = 3 places the profile's value at
// d == width at exp(-3) ≈ 0.0498 (a ~5% tail at one width), i.e. the diffusion has essentially decayed
// by one width — a tight, skin-like falloff. The peak (d == 0) is 1.
//
// Properties the unit test pins:
//   * w(0) == 1 (peaks at the center tap).
//   * w(d) > 0 for every finite d (a Gaussian never reaches 0 -> every tap contributes a little, so the
//     normalized gather is always well-defined).
//   * w(d) is STRICTLY decreasing in |d| (a farther tap always contributes less).
//   * a WIDER sssWidthPx spreads the weight: for a fixed d > 0, w(d) increases with width (the same tap
//     is weighted more under a wider profile) — the wider the subsurface width, the farther light
//     bleeds.
// width is guarded to a small positive epsilon so sssWidthPx -> 0 does not divide by zero (the caller
// short-circuits sssWidthPx<=0 before ever weighting, but the guard keeps the function total).
inline float DiffusionWeight(float distancePx, float sssWidthPx) {
    const float kFalloff = 3.0f;
    float width = std::max(sssWidthPx, 1e-6f);
    float x = distancePx / width;
    return std::exp(-kFalloff * x * x);
}

// --- Depth-aware falloff (silhouette stop) -------------------------------------------------------
// DepthFalloff(tapDepth, centerDepth, depthScale) -> a weight in [0,1] that CUTS a tap whose view-space
// linear depth differs from the center by more than ~the subsurface width, so the diffusion does NOT
// bleed across a silhouette / depth discontinuity (skin in front must not smear onto the background
// behind it).
//
// FORM (documented): a Gaussian of the depth difference, normalized by `depthScale` (the depth window,
// in view-space linear-depth units, over which a tap is still considered the same surface — set by the
// caller to ~the subsurface width in world units):
//     f = exp( -( (tapDepth - centerDepth) / depthScale )^2 )
// Properties the unit test pins:
//   * f == 1 EXACTLY at tapDepth == centerDepth (a co-planar tap is fully trusted).
//   * f -> 0 as |tapDepth - centerDepth| grows large vs depthScale (a tap across a big depth step is
//     rejected -> no bleed across the silhouette).
//   * f is MONOTONE decreasing in |tapDepth - centerDepth| (the farther in depth, the less it counts).
// depthScale is guarded to a small positive epsilon so depthScale -> 0 (an infinitely sharp edge-stop)
// is finite: any non-zero depth difference -> f == 0, an exact difference -> f == 1.
inline float DepthFalloff(float tapDepth, float centerDepth, float depthScale) {
    float ds = std::max(depthScale, 1e-6f);
    float diff = (tapDepth - centerDepth) / ds;
    return std::exp(-(diff * diff));
}

// --- One separable SSS blur pass -----------------------------------------------------------------
// BlurAxis(colorAt, depthAt, maskAt, uv, axisPx, sssWidthPx, sssStrength, taps, depthScale) -> the
// blurred color at `uv` for ONE separable pass along `axisPx`.
//
// `colorAt(u,v)`  -> math::Vec3, the resolved lit scene color at screen UV (the shader: a texture
//                    sample; the test: a procedural field).
// `depthAt(u,v)`  -> float, the VIEW-SPACE LINEAR depth (G-buffer .w) at screen UV.
// `maskAt(u,v)`   -> float in [0,1], the subsurface mask at screen UV.
// `uv`            -> the center pixel UV.
// `axisPx`        -> the PER-TAP-STEP screen offset for this pass, in PIXELS, as a UV-space delta
//                    (i.e. {1/width, 0} horizontal or {0, 1/height} vertical times one pixel). The
//                    gather steps integer multiples of this from -halfTaps..+halfTaps.
// `sssWidthPx`    -> the subsurface profile width in pixels (drives DiffusionWeight + the step length).
// `sssStrength`   -> the global SSS strength multiplier (drives the step length; 0 -> no diffusion).
// `taps`          -> the number of samples gathered along the axis (clamped to >= 1; the center tap is
//                    always included, the rest are symmetric around it).
// `depthScale`    -> the DepthFalloff depth window (view-space linear-depth units).
//
// THE GATHER (documented): the per-tap STEP length in pixels is
//     step = sssStrength * sssWidthPx / max(centerDepth, eps)
// (perspective-correct: a closer surface — smaller centerDepth — diffuses over MORE pixels). For each
// of `taps` taps i in [-half, +half] the sample is at uv + axisPx * (step * i); its contribution is
//     weight_i = DiffusionWeight(|step*i| px, sssWidthPx) * DepthFalloff(tapDepth, centerDepth, depthScale) * maskAt(tap)
// and the result is sum(weight_i * colorAt_i) / sum(weight_i). The mask factor makes a tap that lands on
// a NON-subsurface neighbor contribute nothing, so SSS color only pools within the subsurface region.
//
// THE NO-OP EARLY-OUT (the proof): if sssStrength <= 0 OR sssWidthPx <= 0 we return colorAt(uv)
// directly. (Even without the early-out, step would be 0 -> every tap samples the center -> the
// normalized weighted sum of identical colors == the center color; the early-out makes the identity
// EXACT + branch-clean regardless of the depth/mask field, so the two passes compose to a byte-identical
// pass-through at zero strength/width.) A non-subsurface center pixel (maskAt(uv) == 0) ALSO returns the
// center color: its own center-tap weight uses mask 0, and to keep it an exact pass-through we
// short-circuit it (the shader does the same), so the existing lit/post path stays byte-identical.
template <typename ColorFn, typename DepthFn, typename MaskFn>
inline math::Vec3 BlurAxis(ColorFn colorAt, DepthFn depthAt, MaskFn maskAt,
                           math::Vec2 uv, math::Vec2 axisPx, float sssWidthPx, float sssStrength,
                           int taps, float depthScale) {
    math::Vec3 centerCol = colorAt(uv.x, uv.y);
    // NO-OP early-out: zero strength or zero width -> pure pass-through (the byte-identical proof).
    if (sssStrength <= 0.0f || sssWidthPx <= 0.0f) return centerCol;

    float centerMask = maskAt(uv.x, uv.y);
    // A non-subsurface pixel passes through unchanged (keeps the lit/post path byte-identical).
    if (centerMask <= 0.0f) return centerCol;

    float centerDepth = depthAt(uv.x, uv.y);
    // Background / no surface (cleared G-buffer w == 0): nothing to diffuse, pass through.
    if (centerDepth <= 1e-4f) return centerCol;

    if (taps < 1) taps = 1;
    int half = taps / 2;

    // Perspective-correct per-tap step in pixels (closer surface -> wider screen-space diffusion).
    float step = sssStrength * sssWidthPx / std::max(centerDepth, 1e-4f);

    math::Vec3 sum{0.0f, 0.0f, 0.0f};
    float wsum = 0.0f;
    for (int i = -half; i <= half; ++i) {
        float distPx = step * static_cast<float>(i);
        math::Vec2 tuv{uv.x + axisPx.x * distPx, uv.y + axisPx.y * distPx};
        float tapDepth = depthAt(tuv.x, tuv.y);
        float tapMask  = maskAt(tuv.x, tuv.y);
        float w = DiffusionWeight(std::fabs(distPx), sssWidthPx)
                * DepthFalloff(tapDepth, centerDepth, depthScale)
                * tapMask;
        math::Vec3 c = colorAt(tuv.x, tuv.y);
        sum.x += w * c.x; sum.y += w * c.y; sum.z += w * c.z;
        wsum += w;
    }
    // The center tap (i==0) always carries DiffusionWeight(0)=1 * DepthFalloff(equal)=1 * centerMask>0,
    // so wsum > 0 here; the guard keeps it total.
    if (wsum <= 1e-8f) return centerCol;
    return math::Vec3{sum.x / wsum, sum.y / wsum, sum.z / wsum};
}

}  // namespace hf::render::sss
