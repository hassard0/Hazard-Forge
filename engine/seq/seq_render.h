#pragma once
// Slice SEQ-S6 — LIT 3D render bridge (FLOAT, render-only — the money-shot capstone of FLAGSHIP #25 SEQ,
// the DETERMINISTIC CINEMATIC SEQUENCER). THE ONE FLOAT CROSSING of the whole flagship. seq.h's S1-S5 stay
// STRICT INTEGER + <cmath>-free (the cheap clang-hash proof — scalar/easing/event/transform tracks +
// lockstep/scrub via net::Session); here — and ONLY here — we cross to float to turn the bit-exact Q16.16
// transform timeline into per-sample render transforms for the rasterizer. This is the documented FLOAT
// visresolve-bar (the FPX6/FR6/JT6/VH6/ECON6 precedent): the STATE (the sampled FxTransform trail) is
// bit-exact integer, the final raster/shade is float (cross-vendor ~the engine baseline, NOT held to the
// integer zero-diff bar). The provenance proof: every render instance derives from a bit-exact
// seq::SampleTransform output along the FIXED cutscene TransformTrack.
//
// SEPARATE HEADER ON PURPOSE: seq.h is the 6-include self-contained <cmath>-free bit-exact header (only
// <cstddef>/<cstdint>/<vector> + sim/fpx.h + net/session.h + flow/flow.h) so its standalone-clang
// cross-platform proof stays cheap. The render bridge needs math::Mat4/FromTRS/Vec3/Quat/Normalize +
// fpx::FxToFloat, which would break that — so the bridge lives HERE, NOT in seq.h (mirrors econ_render.h /
// wfc_render.h, the known-good twins: the core header is self-contained, the render bridge is a separate
// _render.h). This header is NOT standalone-clang (it's the render layer, only used by the showcase) —
// that's fine. NO backend symbols, header-only.

#include <cstdint>
#include <vector>

#include "seq/seq.h"        // S1-S5 SEQ core (TransformTrack / SampleTransform / MakeShowcaseTransform) — UNMODIFIED
#include "math/math.h"      // render bridge: math::Mat4 / Quat / Vec3 / FromTRS / Normalize (float)
#include "sim/fpx.h"        // fpx::FxToFloat (the single fixed-point->float convention; mirrors econ_render.h)

namespace hf::seq {

// Namespace aliases so the render bridge reads like the econ_render.h / FxBodyTransform precedent: `fpx::`
// is hf::sim::fpx (the Q16.16 toolbox + FxToFloat); `math::` (hf::math) is already reachable as a sibling of
// hf::seq. These are render-layer-only conveniences (seq.h itself stays float-free).
namespace fpx = hf::sim::fpx;

// SeqTransformToMat4(x): one math::Mat4 from a bit-exact FxTransform — the FPX6 FxBodyTransform pattern for
// a FULL TRS pose (translate * rotate * scale), the ONE float crossing. Each Q16.16 field crosses to float
// via fpx::FxToFloat; the (possibly slightly off-unit nlerp) quaternion is re-normalized via math::Normalize;
// the matrix is the standard T*R*S via math::FromTRS. Render-only; NO sim mutation. Pure deterministic host
// float (no RNG, no clock).
inline math::Mat4 SeqTransformToMat4(const FxTransform& x) {
    const math::Vec3 t{ fpx::FxToFloat(x.t.x), fpx::FxToFloat(x.t.y), fpx::FxToFloat(x.t.z) };
    const math::Quat q = math::Normalize(math::Quat{
        fpx::FxToFloat(x.r.x), fpx::FxToFloat(x.r.y), fpx::FxToFloat(x.r.z), fpx::FxToFloat(x.r.w) });
    const math::Vec3 s{ fpx::FxToFloat(x.s.x), fpx::FxToFloat(x.s.y), fpx::FxToFloat(x.s.z) };
    return math::FromTRS(t, q, s);
}

// SeqToRenderInstances(samples): one instance per sampled transform, in sample order. Empty input -> empty
// output (the empty no-op). Pure deterministic host float (no RNG/clock), render-only. Provenance: each Mat4
// derives from a bit-exact seq::SampleTransform output (the canonical provenance function — the showcase
// memcmp-compares its bytes against a recompute of the same bit-exact trail).
inline std::vector<math::Mat4> SeqToRenderInstances(const std::vector<FxTransform>& samples) {
    std::vector<math::Mat4> out;
    out.reserve(samples.size());
    for (const FxTransform& x : samples) out.push_back(SeqTransformToMat4(x));
    return out;
}

// MakeCutsceneTrail(n): the hero cutscene scene — a FIXED, enriched TransformTrack sampled at n successive
// times into a ghosted MOTION TRAIL of the animated hero object. Pure integer (seq::SampleTransform is
// bit-exact), so the sampled trail is byte-identical run-to-run AND cross-backend by construction (the Vulkan
// --seq-render-shot and the Metal --seq-render call THIS verbatim). FIXED FOREVER — the render provenance
// rests on it.
//
// The track ENRICHES the S4 MakeShowcaseTransform(): the S4 rotation keys (so the hero visibly TURNS through
// the 90deg/180deg +Y arc) + a wider translation SWEEP (so the trail SPREADS across the ground, not a tight
// cluster) + a subtle scale pulse on Y. Sampled at t_i = (fx)((int64)i * dt) for i in [0,n) over a fixed dt
// spanning the [0,2]s keyed range. Returns the vector of bit-exact FxTransform — the provenance source.
inline std::vector<FxTransform> MakeCutsceneTrail(uint32_t n) {
    // Start from the S4 fixed showcase track (its rotation keys + scale pulse) and WIDEN the translation
    // sweep so the trail spreads readably across the ground (a clear arc from one side to the other). The
    // rotation + scale channels are the S4 fixture verbatim (the turn + the pulse); only the translation
    // amplitude is enriched (still a FIXED, deterministic track — pure integer).
    TransformTrack tr = MakeShowcaseTransform();

    // Wider X sweep: 0 -> 3 -> 6 over [0,2]s (vs S4's 0->1->2) so the trail strides across the ground.
    tr.translation.x.times  = {0, kOne, 2 * kOne};
    tr.translation.x.values = {0, 3 * kOne, 6 * kOne};
    tr.translation.x.easing = Easing::Linear;
    // Z drifts the trail forward in depth: 0 -> -3/2 -> -3 (vs S4's 0->-1/2->-1).
    tr.translation.z.times  = {0, kOne, 2 * kOne};
    tr.translation.z.values = {0, -3 * kOne / 2, -3 * kOne};
    tr.translation.z.easing = Easing::Linear;
    // A gentle vertical arc so the trail rises then settles: 0 -> 1 -> 0 (vs S4's 0->1/2->0).
    tr.translation.y.times  = {0, kOne, 2 * kOne};
    tr.translation.y.values = {0, kOne, 0};
    tr.translation.y.easing = Easing::EaseInOutSine;
    // tr.rotation + tr.scale are the S4 fixture verbatim (the +Y turn + the subtle Y scale pulse).

    // Sample n successive times over the [0,2]s keyed range. dt is chosen so the last sample (i==n-1) lands
    // at the end of the keyed range when n>1 (and a single sample lands at t=0); the i*dt product is formed
    // in int64 to avoid overflow (the SampleSweep discipline). dt = (2s) / (n-1) for n>1.
    const fx span = 2 * kOne;                                  // the [0,2]s keyed range
    const fx dt   = (n > 1) ? (fx)((int64_t)span / (int64_t)(n - 1)) : 0;
    std::vector<FxTransform> out;
    out.reserve((std::size_t)n);
    for (uint32_t i = 0; i < n; ++i) {
        const fx t = (fx)((int64_t)i * (int64_t)dt);
        out.push_back(SampleTransform(tr, t));
    }
    return out;
}

}  // namespace hf::seq
