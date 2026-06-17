#pragma once
// Slice DF — Contrast-Adaptive Sharpening (CAS; AMD FidelityFX CAS) math — pure CPU (header-only,
// no device, no backend symbols). Namespace hf::render::cas. Mirrors dof.h / gtao.h / color_grade.h:
// a tiny shared-math header ABOVE the RHI seam (ZERO vk*/MTL*/mtl::/Backend::Metal CODE symbols — the
// only mentions of "vk"/"MTL" anywhere in this slice's above-seam files are seam-discipline doc
// comments + the [[vk::binding]] HLSL decorations). The CAS fragment shader (shaders/cas.frag.hlsl)
// copies CasWeight + CasSharpen VERBATIM, so tests/cas_test.cpp exercises the EXACT per-pixel sharpen
// the GPU pass runs — which is what makes the sharpness=0 render BYTE-IDENTICAL to the unsharpened
// render AND bit-identical cross-backend.
//
// THE TECHNIQUE (AMD FidelityFX Contrast-Adaptive Sharpening): a fullscreen post pass that crisps a
// rendered (SDR) image WITHOUT the ringing halos a naive unsharp mask produces. For each pixel it reads
// the center color plus its 4 cross neighbors (up/down/left/right — the FidelityFX CAS "ring 0" cross
// kernel) and forms a normalized weighted difference:
//
//     out = (center + w*(up + down + left + right)) / (1 + 4*w)
//
// where w is a SMALL NEGATIVE weight (a discrete-Laplacian sharpen: subtracting a fraction of the
// neighborhood's average from the center boosts local contrast). The "contrast-adaptive" part is that w
// is driven by the local neighborhood luminance min/max so the pass sharpens MORE in flat / low-contrast
// regions and LESS near a hard edge — and the result is then CLAMPED to the neighborhood [min,max] per
// channel so a sharpen can NEVER overshoot past the brightest / darkest neighbor (the no-ringing clamp,
// the property that distinguishes CAS from plain unsharp masking). CAS runs on the FINAL SDR result
// (post-tonemap), exactly like the FidelityFX reference and the color-grade pass (documented; sharpening
// before tonemap is explicit YAGNI in the spec).
//
// THE sharpness=0 NO-OP PROOF (what makes this golden-safe — like GTAO radius=0==no-AO, SSS
// strength=0==no-SSS, color-grade identity==ungraded): the sharpen contribution scales with `sharpness`,
// and CasWeight returns EXACTLY 0 at sharpness == 0 (the short-circuit below). With w == 0 the formula
// degenerates to out = (center + 0) / (1 + 0) = center, and clamping center to its own neighborhood
// [min,max] (center always lies within the min/max of {center,up,down,left,right}) is a no-op — so
// CasSharpen(..., 0) == center EXACTLY (no constant bias, no division drift, no clamp bias). The showcase
// renders the CAS pass at sharpness 0 and asserts BYTE-IDENTICAL (SHA) to the engine's standard
// (unsharpened) render of the same scene — the SAME cas shader at sharpness 0 vs the unsharpened render,
// so the proof is backend-portable — then renders the real sharpness > 0 version as the golden. The unit
// test additionally pins the kernel: identity at sharpness 0, the adaptive weight, and the no-overshoot
// clamp.
//
// CONVENTIONS:
//   * CAS operates on the DISPLAYED SDR color (post-tonemap, nominally in [0,1]). It does NOT tonemap.
//   * The adaptive weight uses the per-channel neighborhood min/max (CasSharpen derives minL/maxL from
//     the MAX-luminance / MIN-luminance channel envelope of the cross neighborhood — the FidelityFX CAS
//     scheme — so a bright detail on a dark field still adapts; see CasSharpen).
//   * The peak weight range lerp(-0.125, -0.2, sharpness) is the standard FidelityFX CAS peak: at the
//     low end (-1/8) a gentle sharpen, at the high end (-1/5) the maximum CAS sharpen. sharpness is the
//     user knob in [0,1]; at sharpness == 0 the weight is forced to EXACTLY 0 (a true pass-through), not
//     -1/8 — so the no-op proof is exact. (Documented divergence from a literal lerp at the zero end:
//     the literal FidelityFX lerp never reaches 0, but a sharpen pass needs a clean OFF state, so we gate
//     the whole adaptive weight to 0 at sharpness 0.)
//
// Pure, deterministic functions: no RNG, no time.

#include "math/math.h"

#include <algorithm>
#include <cmath>

namespace hf::render::cas {

// Rec.601-ish luma weights used to derive the per-pixel luminance envelope for the adaptive weight.
// (CAS only needs a monotone brightness measure to drive the adaptive amount; the exact weights are not
// load-bearing for the no-op proof — at sharpness 0 the weight is 0 regardless.)
inline constexpr float kLumaR = 0.299f;
inline constexpr float kLumaG = 0.587f;
inline constexpr float kLumaB = 0.114f;

// The FidelityFX CAS peak-weight endpoints (the negative Laplacian lobe). lerp(kPeakLo, kPeakHi,
// sharpness): a gentle -1/8 at the low end, the maximum -1/5 at the high end.
inline constexpr float kPeakLo = -0.125f;  // -1/8 : gentle sharpen
inline constexpr float kPeakHi = -0.200f;  // -1/5 : max CAS sharpen
inline constexpr float kEps    = 1e-4f;     // guards the max(maxL, eps) division on a black neighborhood

// --- The CAS adaptive weight --------------------------------------------------------------------
// CasWeight(minL, maxL, sharpness) -> the per-pixel negative-lobe sharpen weight w applied to EACH of
// the 4 cross neighbors. `minL`/`maxL` are the local neighborhood luminance (or per-channel) MIN and MAX.
//
// THE FIDELITYFX ADAPTIVE AMPLITUDE: the sharpen amplitude is driven by how much "headroom" the local
// region has — a region whose values sit far below the local max (low local contrast / smooth) gets a
// HIGH amplitude; a region already spanning a hard edge (maxL near 1, minL near 0) gets a LOW amplitude
// so the sharpen does not amplify the edge into a ring:
//
//     amp = sqrt( clamp( min(minL, 1 - maxL) / max(maxL, eps), 0, 1 ) )
//     w   = amp * lerp(kPeakLo, kPeakHi, sharpness)
//
// Reading min(minL, 1-maxL): the SMALLER of "how far the darkest sample is above black" and "how far the
// brightest sample is below white" — i.e. the distance to the nearer clipping rail. Divided by maxL it
// is a normalized local-contrast headroom; the sqrt softens it (the FidelityFX curve); amp ∈ [0,1].
// Multiplying by the (negative) peak weight gives the per-neighbor lobe w ≤ 0.
//
// THE EXACT-ZERO PATH (the no-op proof's core): at sharpness == 0 we SHORT-CIRCUIT to return 0.0f
// EXACTLY — branch-clean, no dependence on the luminance envelope. (A literal lerp(kPeakLo, kPeakHi, 0)
// would be kPeakLo = -1/8, NOT zero; but a sharpen pass needs a clean OFF, and the byte-identical proof
// requires w == 0 -> out == center exactly, so sharpness 0 forces w = 0.) For sharpness > 0 the weight
// is the smooth FidelityFX amplitude above; w is monotone in sharpness (the |peak| grows from 1/8 to 1/5
// as sharpness goes 0+ -> 1, so |w| increases -> more sharpening), and larger (more negative) in a
// low-contrast region than near a hard edge (the adaptive property).
inline float CasWeight(float minL, float maxL, float sharpness) {
    if (sharpness <= 0.0f) return 0.0f;   // exact OFF: sharpness 0 -> w 0 -> out == center (no-op proof)
    float headroom = std::min(minL, 1.0f - maxL);
    float denom = std::max(maxL, kEps);
    float ratio = headroom / denom;
    if (ratio < 0.0f) ratio = 0.0f;
    if (ratio > 1.0f) ratio = 1.0f;
    float amp = std::sqrt(ratio);
    float peak = math::lerp(kPeakLo, kPeakHi, sharpness);  // -1/8 .. -1/5
    return amp * peak;                                     // the negative per-neighbor lobe (w <= 0)
}

// --- The CAS 3x3 (cross) sharpen ----------------------------------------------------------------
// CasSharpen(center, up, down, left, right, sharpness) -> the contrast-adaptive-sharpened center color.
//
// Derives the neighborhood luminance min/max from the 5 cross samples, gets the adaptive weight w via
// CasWeight, then applies the normalized CAS sharpen and the no-ringing clamp PER CHANNEL:
//
//     out_ch = clamp( (center_ch + w*(up_ch + down_ch + left_ch + right_ch)) / (1 + 4*w),
//                     min_ch, max_ch )
//
// where (min_ch, max_ch) are the per-channel min/max over {center, up, down, left, right}. The clamp is
// what makes CAS ring-free: the sharpened value can never exceed the brightest neighbor's channel nor
// fall below the darkest — a high-contrast edge gets crisper but no overshoot halo appears.
//
// THE sharpness=0 IDENTITY: CasWeight(..., 0) == 0 -> w == 0 -> numerator = center + 0, denominator =
// 1 -> out = center EXACTLY; the clamp to [min,max] is a no-op because center always lies within the
// neighborhood's own min/max. So CasSharpen(c, ..., 0) == c EXACTLY for ANY neighborhood (the
// byte-identical sharpness=0 == unsharpened render proof's per-pixel core). We compute the weight FIRST
// and, when it is exactly 0, return center directly so the identity is exact + branch-clean regardless
// of any min/max rounding.
inline math::Vec3 CasSharpen(const math::Vec3& center, const math::Vec3& up, const math::Vec3& down,
                             const math::Vec3& left, const math::Vec3& right, float sharpness) {
    // Per-channel neighborhood min/max over the 5 cross samples (the no-ringing clamp bounds).
    auto mn = [](float a, float b, float c, float d, float e) {
        return std::min(std::min(std::min(a, b), std::min(c, d)), e);
    };
    auto mx = [](float a, float b, float c, float d, float e) {
        return std::max(std::max(std::max(a, b), std::max(c, d)), e);
    };
    math::Vec3 lo{mn(center.x, up.x, down.x, left.x, right.x),
                  mn(center.y, up.y, down.y, left.y, right.y),
                  mn(center.z, up.z, down.z, left.z, right.z)};
    math::Vec3 hi{mx(center.x, up.x, down.x, left.x, right.x),
                  mx(center.y, up.y, down.y, left.y, right.y),
                  mx(center.z, up.z, down.z, left.z, right.z)};

    // The adaptive weight from the neighborhood luminance envelope: minL = luma of the per-channel min,
    // maxL = luma of the per-channel max (the FidelityFX "min/max box" luminance, so a bright detail on
    // a dark field still adapts). Both in [0,1] for an SDR neighborhood.
    float minL = lo.x * kLumaR + lo.y * kLumaG + lo.z * kLumaB;
    float maxL = hi.x * kLumaR + hi.y * kLumaG + hi.z * kLumaB;
    float w = CasWeight(minL, maxL, sharpness);

    // EXACT OFF: w == 0 -> out == center (the byte-identical no-op proof; branch-clean exact identity).
    if (w == 0.0f) return center;

    float denom = 1.0f + 4.0f * w;
    auto channel = [&](float c, float u, float d, float l, float r, float loc, float hic) {
        float v = (c + w * (u + d + l + r)) / denom;     // normalized CAS sharpen
        if (v < loc) v = loc;                            // no-ringing clamp: never below the darkest
        if (v > hic) v = hic;                            // never above the brightest neighbor
        return v;
    };
    return math::Vec3{channel(center.x, up.x, down.x, left.x, right.x, lo.x, hi.x),
                      channel(center.y, up.y, down.y, left.y, right.y, lo.y, hi.y),
                      channel(center.z, up.z, down.z, left.z, right.z, lo.z, hi.z)};
}

}  // namespace hf::render::cas
