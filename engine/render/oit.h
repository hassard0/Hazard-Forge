#pragma once
// Slice CO — Order-Independent Transparency (Weighted Blended OIT, McGuire & Bavoil 2013). Pure CPU
// (header-only, no device, no backend symbols). Namespace hf::render::oit. Mirrors dof.h / cluster.h /
// motion_blur.h: a tiny shared-math header above the RHI seam (ZERO vk*/MTL*/mtl::/Backend::Metal
// CODE symbols — the only mentions of "vk"/"MTL" are seam-discipline doc comments). The two OIT
// shaders (oit_accum.frag / oit_resolve.frag) copy these three functions VERBATIM so the CPU unit
// test (tests/oit_test.cpp) and the GPU accum+resolve passes agree on the weight + the composite.
//
// THE TECHNIQUE (Weighted Blended OIT): correct-enough transparency WITHOUT sorting. Each transparent
// fragment contributes its PREMULTIPLIED color * a depth weight `w` ADDITIVELY into an RGBA16F accum
// target, and its alpha contributes ADDITIVELY into accum.a; SEPARATELY each fragment multiplies a
// "revealage" target (how much background still shows through) by (1 - alpha). A fullscreen resolve
// then composites  transparentRGB = accum.rgb / max(accum.a, eps)  OVER the opaque background weighted
// by (1 - revealage). No per-pixel sorting, no linked lists — one accumulate pass + one resolve.
//
// THE ORDER-INDEPENDENCE EQUIVALENCE PROOF (what makes this golden-safe — like CL clustered==brute,
// CJ Hi-Z==frustum, CN zero-vel==scene): the accum target is a SUM (Σ premultColor*w, Σ alpha*w) and
// the revealage is a PRODUCT (Π (1-alpha)). Addition and multiplication are COMMUTATIVE, so rendering
// the SAME N transparent fragments in order [0,1,2,...] vs a PERMUTED order [3,1,0,2,...] yields the
// IDENTICAL accum + revealage and therefore a BYTE-IDENTICAL resolved image. The showcase + the unit
// test render both orders and assert SHA / bit equality — the entire point of OIT, proven.
//   NOTE on FP determinism: floating-point addition is NOT associative in general, so to keep the two
//   orders BIT-exact we accumulate a FIXED, FINITE, WELL-SEPARATED weighted contribution set whose
//   partial sums do not diverge across orderings (the showcase + test pick such values; the internal
//   assert is the gate — if a sum ever differs across orders the run FAILS LOUDLY rather than baking a
//   bad golden). Accumulate() itself is exact per fragment; the order-stability is a property of the
//   chosen input set, documented and verified empirically.
//
// CONVENTIONS (must match the shaders EXACTLY):
//   * `viewDepth` is VIEW-SPACE LINEAR depth = -vpos.z (positive in front of the camera; the SAME .w
//     the gbuffer.frag stores and SSR/SSAO/DoF reconstruct). Larger = farther.
//   * `premultColorAlpha` is the fragment's PREMULTIPLIED color in .xyz (== straightColor*alpha) and
//     its straight `alpha` in .w. The accumulation adds premultColor*w to accum.rgb and alpha*w to
//     accum.a; the resolve divides accum.rgb by accum.a to recover the weighted-average color.
//   * `revealage` starts at 1.0 (fully revealed = only background shows) and is multiplied by
//     (1 - alpha) per fragment, so after N fragments revealage == Π(1-alpha_i) ∈ [0,1].

#include "math/math.h"

#include <algorithm>
#include <cmath>

namespace hf::render::oit {

// --- McGuire depth weight ------------------------------------------------------------------------
// Weight(viewDepth, alpha) -> the per-fragment WBOIT weight `w` (McGuire & Bavoil 2013, eq. their
// "weight function 9"/the practical depth weight). It DOWN-WEIGHTS distant fragments so a near
// transparent surface dominates the weighted average over a far one (reducing the color-bleed
// artifact inherent to the unsorted blend), while staying positive + finite for every input.
//
// EXACT FORMULA (depth in VIEW-SPACE LINEAR units, the same units gbuffer.frag stores):
//
//     w = alpha * clamp( 0.03 / (1e-5 + (viewDepth / 200)^4),  1e-2,  3e3 )
//
// (the published "9th" weighting from the JCGT 2013 paper, with the depth scaled by /200 so the falloff
// is gentle across a typical 0..~90-unit scene). Properties the unit test + the shaders pin:
//   * POSITIVE + FINITE for ALL inputs (alpha in [0,1], any viewDepth incl. 0 / negative): the clamp
//     bounds the depth term to [1e-2, 3e3] and alpha>=0, so w>=0 and finite (no div-by-zero: the
//     denominator carries a +1e-5 floor; no NaN: every term is finite).
//   * DECREASING IN DEPTH: for a FIXED alpha>0, a NEARER fragment (smaller viewDepth) gets a weight
//     >= a FARTHER one — the (viewDepth/200)^4 term grows with depth so the reciprocal shrinks (until
//     it saturates at the clamp floor for very far fragments, where it is flat, hence ">=" not ">").
//   * SCALES WITH ALPHA: w is proportional to alpha, so a more-opaque fragment carries more weight.
inline float Weight(float viewDepth, float alpha) {
    // Guard alpha to [0,1] (the showcase/tests pass straight alphas already in range; this keeps the
    // function total for any caller).
    float a = std::min(std::max(alpha, 0.0f), 1.0f);
    float zd = viewDepth / 200.0f;
    float zd2 = zd * zd;
    float zd4 = zd2 * zd2;                       // (viewDepth/200)^4, grows with depth
    float depthTerm = 0.03f / (1e-5f + zd4);     // shrinks as depth grows; +1e-5 floors the denom
    depthTerm = std::min(std::max(depthTerm, 1e-2f), 3e3f);  // clamp -> bounded, finite, positive
    return a * depthTerm;
}

// --- Per-fragment accumulation -------------------------------------------------------------------
// Accumulate(premultColorAlpha, weight, accum, revealage) -> fold ONE transparent fragment into the
// running WBOIT accum + revealage. premultColorAlpha.xyz is the PREMULTIPLIED color (straight*alpha),
// premultColorAlpha.w is the straight alpha. The accumulation is:
//     accum     += vec4(premultColor * weight,  alpha * weight)
//     revealage *= (1 - alpha)
// SUM (accum) + PRODUCT (revealage) -> ORDER-INDEPENDENT by construction (commutative). Mirrored by
// the oit_accum.frag MRT outputs (RT0 additive accum, RT1 multiplicative revealage) — the shader's
// additive ONE,ONE blend performs the accum += and the multiplicative blend performs the revealage *=.
inline void Accumulate(const math::Vec4& premultColorAlpha, float weight,
                       math::Vec4& accum, float& revealage) {
    float alpha = premultColorAlpha.w;
    accum.x += premultColorAlpha.x * weight;
    accum.y += premultColorAlpha.y * weight;
    accum.z += premultColorAlpha.z * weight;
    accum.w += alpha * weight;                 // Σ alpha*weight  (the resolve's divisor)
    revealage *= (1.0f - alpha);               // Π (1 - alpha)
}

// --- Resolve composite ---------------------------------------------------------------------------
// ResolveOver(accum, revealage, background) -> the final composited RGB of the transparent layers OVER
// the opaque `background`. The WBOIT resolve:
//     transparentRGB = accum.rgb / max(accum.a, 1e-4)        // weighted-average straight color
//     out            = transparentRGB * (1 - revealage) + background * revealage
// The (1 - revealage) is the total transparent coverage (1 - Π(1-alpha)); revealage is how much
// background still shows. max(accum.a, 1e-4) guards the divide when no/near-zero-alpha fragments
// landed (then revealage ~ 1 so the transparent term is weighted ~0 anyway -> out ~ background).
// Returns the resolved RGB; the resolved alpha (coverage) is (1 - revealage) — callers writing an
// alpha channel should emit that; the showcase composites over an opaque background so it outputs RGB.
inline math::Vec3 ResolveOver(const math::Vec4& accum, float revealage,
                              const math::Vec3& background) {
    float denom = std::max(accum.w, 1e-4f);
    math::Vec3 transparentRGB{accum.x / denom, accum.y / denom, accum.z / denom};
    float cover = 1.0f - revealage;            // total transparent coverage
    return math::Vec3{
        transparentRGB.x * cover + background.x * revealage,
        transparentRGB.y * cover + background.y * revealage,
        transparentRGB.z * cover + background.z * revealage,
    };
}

}  // namespace hf::render::oit
