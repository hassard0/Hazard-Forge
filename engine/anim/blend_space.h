#pragma once
// Slice AN1 — DETERMINISTIC BLEND SPACES (1D + 2D parameter-driven animation blending; the parity++
// audit item #4 / the long-open gap-roadmap AN1 row). The anim stack ships clips/sampling/cross-fade
// (anim::BlendAnimations) and the FSM (state_machine.h), and MM1 ships the SEARCH-based locomotion
// sibling (motion_match.h) — but nothing PARAMETRIC: the standard "idle/walk/run by speed" (1D) and
// "strafe by speed x direction" (2D) locomotion primitive UE5 authors as Blend Spaces. AN1 builds it
// deterministically: INTEGER parameters, INTEGER weights, INTEGER point-in-triangle/barycentric math
// (the NAV/convex orientation discipline, int64 throughout — the WH7 overflow lesson), and tick-based
// parameter slewing (NOT wall-clock — an FSM-driven param ramp is replayable; UE5's is frame-rate
// coupled). Namespace hf::anim::bs, header-only, pure CPU, NO device/backend symbols, NO RNG, NO clock.
// engine/anim/animation.h + skeleton.h + state_machine.h + motion_match.h are #included READ-ONLY
// (byte-untouched).
//
// THE FLOAT / INTEGER SPLIT (documented honestly, the MM1 quantize-boundary discipline):
//   * INTEGER (bit-exact cross-platform BY CONSTRUCTION): parameters (Q16.16), the 1D segment search +
//     lerp factor, the 2D point-in-triangle orientation tests + barycentric weights (int64 Cramer),
//     the outside-hull clamp projection, the per-tick parameter slew, the normalized-phase
//     accumulator, and every digest below. Two peers holding the same (space, param stream) agree on
//     every WEIGHT and every SELECTED sample bit-for-bit.
//   * FLOAT (the EXISTING anim-stack pose class, reused as-is): the pose sampling + per-joint blend —
//     EvaluatePose* converts the exact integer weights to float (exact: every Q16.16 value in
//     [0, kOne] fits a float mantissa) and calls animation.h's SampleLocalPose + BlendLocalPoses (THE
//     REUSED SEAM — BlendAnimations' per-joint blend core: lerp t/s, Slerp r; NOT reimplemented).
//     The float path is deterministic per platform; the pinned DIGESTS quantize pose samples ONCE via
//     mm::QuantizeFx (llround, round-half-away — the MM1 boundary) and the test fixtures use exact
//     binary-fraction keys/params (translation-only rotations) so the pinned digests are ALSO
//     cross-compiler (MSVC == clang) — the same guarantee MM1 ships. Rotation blending inherits
//     animation.h Slerp (nlerp) float semantics.
//
// TIME SYNC (the part naive implementations get wrong): NORMALIZED-PHASE SYNCHRONIZATION — the
// standard blend-space rule. A blend space does NOT sample its clips at the same absolute time; it
// samples each clip at the SAME NORMALIZED PHASE (time_i = phase * duration_i), so a 1.0 s walk loop
// and a 0.5 s run loop blend with their foot-plants ALIGNED (feet don't slide through the blend).
// The phase is a Q16.16 accumulator in [0, kOne) advanced by a fixed per-tick rate and wrapped
// (mod kOne) — tick-based, replayable. Naive same-absolute-time blending is what BlendAnimations
// does when handed equal times; blend_space_test pins BOTH values on a constructed fixture to prove
// the distinction (phase-synced foot-z == the aligned key, naive == the cancelled 0).
//
// THE 2D TRIANGULATION (v1, DOCUMENTED CHOICE): a FIXED AUTHORED triangulation — the author provides
// triangles over the sample points. Delaunay is float-hostile (in-circle predicates need exact
// arithmetic to be deterministic); the honest deterministic v1 is authored triangles, exactly like
// the authored FSM graph. Integer auto-triangulation (exact-predicate Delaunay) is a future slice.
// Authored winding does NOT matter: every predicate canonicalizes by the triangle's signed area.
//
// 2D CONVENTIONS (all pinned by blend_space_test):
//   * CONTAINMENT: boundary-INCLUSIVE integer orientation tests (a point on an edge/vertex is inside);
//     the FIRST triangle in authored array order that contains the point wins (the deterministic
//     shared-edge tie-break — lower triangle index).
//   * BARYCENTRIC: int64 Cramer — w0 = Orient(p,b,c)/D, w1 = Orient(a,p,c)/D with D = Orient(a,b,c)
//     (2x signed area), each as a truncating ((num << kFrac) / D) Q16.16 division; w2 = kOne - w0 - w1
//     (THE RESIDUAL CONVENTION — the three weights sum to kOne EXACTLY; the <= 2^-16 truncation error
//     lands on w2). At a vertex the weights are exactly {kOne, 0, 0}.
//   * OUTSIDE THE HULL: clamp to the NEAREST EDGE/VERTEX — scan ALL triangle edges, project the point
//     onto each segment (integer t in [0, kOne]), take the int64 squared-distance argmin with the
//     strict (dist2, triIndex, edgeIndex) scan-order tie-break (first wins), and return the 2-sample
//     edge lerp at the projected t (a projection past an endpoint clamps to that VERTEX -> weight
//     kOne on one sample). `clamped` is reported so callers/showcases can tell.
//   * OVERFLOW BUDGET (pinned discipline, the WH7 lesson): |parameter| <= 2^20 Q16.16 (+-16.0 in
//     parameter units — blend parameters are speeds/directions, far inside this). Then coordinate
//     diffs are < 2^21, an orientation cross product < 2^43, and the barycentric (num << 16) < 2^59 —
//     comfortably inside int64. Same budget for the projection dot products.
//
// THE 3-WAY BLEND COMPOSITION (pinned, honesty over green): pose = BlendLocalPoses(
// BlendLocalPoses(P(a), P(b), w1/(w0+w1)), P(c), w2) — a LEFT FOLD in the triangle's AUTHORED vertex
// order (a, b, c), reusing the existing 2-pose seam twice; the inner weight is the truncating Q16.16
// division ((w1 << kFrac) / (w0 + w1)). For translations/scales (linear lerps) this equals the true
// weighted average up to the inner-weight truncation: per-component deviation <= (component range) *
// 2^-16 * (1 - w2). For ROTATIONS the engine's Slerp is nlerp, which is NON-ASSOCIATIVE: nested
// nlerp != the normalized weighted quaternion sum in general. The deviation is ZERO when the three
// rotations are equal (the pinned fixtures) and grows O(angular-spread^2) — for locomotion-scale
// spreads (< ~30 deg between neighboring samples) it is far below visual/quantization thresholds,
// but it is NOT claimed exact; the pinned composition ORDER is the determinism contract. When any
// barycentric weight == kOne (and in the 1D w == 0 / w == kOne cases) the pose is the DIRECT clip
// sample — identity-at-sample-point holds EXACTLY (no blend call, no float rounding).
//
// PARAMETER SLEWING (tick-based, the GAS/UE5-can't-replay contrast): SlewParam moves the current
// parameter toward the target by at most `rate` Q16.16 PER TICK (rate 0 = disabled -> instant).
// SlewParam2 slews each axis INDEPENDENTLY (a box clamp, documented — a radial clamp needs an integer
// sqrt and is a future refinement). Deterministic: a pure function of (current, target, rate).
//
// FSM INTEGRATION (the thin adapter, item 4): state_machine.h has no pose-source seam (states are
// clip indices), so the adapter lives HERE as a helper the caller drives: BlendDriver1D binds an FSM
// float parameter index; TickBlendDriver1D reads fsm.GetParam(idx), quantizes it ONCE via
// mm::QuantizeFx (THE one float->fx boundary at this seam — the FSM's param model is float and is
// reused read-only), slews toward it, and advances the wrapped phase. EvaluateBlendDriver1D turns the
// driver state into a pose. The FSM keeps driving state SELECTION exactly as before; the driver adds
// the parametric pose source next to it.
//
// REUSE MAP: animation.h SampleLocalPose/BlendLocalPoses/PaletteFromLocalPose (the pose source + THE
// 2-pose blend seam) read-only; skeleton.h (rest pose) read-only; state_machine.h (the FSM the driver
// binds) read-only; motion_match.h mm::fx/kOne/kFrac/QuantizeFx + mm::detail::Fnv1a64Word (the anim
// family's Q16.16 + digest currency — same bits as MM1's pinned goldens) + the MakeMMTestRig/
// MakeMMIdleClip/MakeMMWalkClip fixture assets (the shared deterministic fixtures; AN1 adds the run
// clip). NO new RHI, NO shader, NO device — both showcase backends run the IDENTICAL scenario +
// raster below (the WV1/SP1 header-shared-scenario zero-copy-drift pattern).

#include <cstdint>
#include <vector>

#include "anim/animation.h"      // READ-ONLY: SampleLocalPose/BlendLocalPoses — the pose source + blend seam
#include "anim/motion_match.h"   // READ-ONLY: mm::fx/kOne/QuantizeFx/Fnv1a64Word + the MM fixture assets
#include "anim/skeleton.h"       // READ-ONLY: Skeleton/Joint — the rest pose
#include "anim/state_machine.h"  // READ-ONLY: StateMachine — the float-param FSM the driver binds

namespace hf::anim {
namespace bs {

using mm::fx;
using mm::kFrac;
using mm::kOne;
using mm::QuantizeFx;   // THE float->integer boundary (llround, round-half-away — the MM1 discipline)

// ===================== The 1D blend space (idle/walk/run by speed) ====================================

// One 1D sample: a Q16.16 parameter position + a clip index into a caller-provided
// std::vector<Animation> (the state_machine.h AnimState convention). `samples` MUST be sorted
// ascending by param (authored order == evaluation order; equal params are legal — the zero-width
// segment is skipped by the w==0 convention below).
struct BlendSample1D {
    fx      param = 0;
    int32_t clip = 0;
};

struct BlendSpace1D {
    std::vector<BlendSample1D> samples;   // sorted ascending by param
};

// The 1D weight result: pose = (1 - w) * samples[i0] + w * samples[i1], w in [0, kOne]. i0 == i1
// (w == 0) outside the sample range (UE5 clamp semantics) and AT the exact sample points; {-1,-1,0}
// for an empty space.
struct Weights1D {
    int32_t i0 = -1;
    int32_t i1 = -1;
    fx      w = 0;
};

// EvaluateWeights1D: find the bracketing pair + the EXACT integer lerp factor. Convention (pinned):
// param <= first -> {0,0,0}; param >= last -> {n-1,n-1,0}; else the segment [i, i+1) with
// p[i] <= param < p[i+1] and w = ((param - p[i]) << kFrac) / (p[i+1] - p[i]) (truncating int64
// division; exact when the span divides the offset). A param AT an interior sample lands the segment
// starting there with w == 0 -> the identity-at-sample path.
inline Weights1D EvaluateWeights1D(const BlendSpace1D& s, fx param) {
    const int n = (int)s.samples.size();
    if (n == 0) return Weights1D{};
    if (n == 1 || param <= s.samples[0].param) return Weights1D{0, 0, 0};
    if (param >= s.samples[(size_t)(n - 1)].param) return Weights1D{n - 1, n - 1, 0};
    int hi = 1;
    while (hi < n - 1 && s.samples[(size_t)hi].param <= param) ++hi;   // first p[hi] > param
    const int i0 = hi - 1;
    const fx p0 = s.samples[(size_t)i0].param;
    const fx p1 = s.samples[(size_t)hi].param;
    const int64_t span = (int64_t)p1 - (int64_t)p0;
    const fx w = (span > 0) ? (fx)((((int64_t)param - (int64_t)p0) << kFrac) / span) : 0;
    return Weights1D{i0, hi, w};
}

// ===================== Phase-synchronized clip sampling ===============================================

// PhaseTime: normalized phase (Q16.16 in [0, kOne]) -> this clip's local sample time in seconds
// (phase * duration). Computed in IEEE double then narrowed once — exact when phase and duration are
// binary fractions (the fixture discipline).
inline float PhaseTime(const Animation& clip, fx phase) {
    return (float)((double)phase * (double)clip.duration / 65536.0);
}

namespace detail {

// RestPose: the skeleton's rest local TRS (what SampleLocalPose starts from) — the deterministic
// fallback for an empty space / out-of-range clip index.
inline std::vector<JointPose> RestPose(const Skeleton& sk) {
    std::vector<JointPose> pose(sk.joints.size());
    for (size_t j = 0; j < sk.joints.size(); ++j) {
        pose[j].t = sk.joints[j].t;
        pose[j].r = sk.joints[j].r;
        pose[j].s = sk.joints[j].s;
    }
    return pose;
}

// SamplePhase: one clip sampled at the synchronized phase (bounds-checked; out-of-range -> rest).
inline std::vector<JointPose> SamplePhase(const Skeleton& sk, const std::vector<Animation>& clips,
                                          int32_t clip, fx phase) {
    if (clip < 0 || (size_t)clip >= clips.size()) return RestPose(sk);
    const Animation& a = clips[(size_t)clip];
    return SampleLocalPose(sk, a, PhaseTime(a, phase));
}

// WeightToFloat: exact conversion of a Q16.16 weight in [0, kOne] to float (values <= 2^16 always
// fit the 24-bit mantissa; no rounding).
inline float WeightToFloat(fx w) { return (float)w / 65536.0f; }

}  // namespace detail

// EvaluatePose1D: the 1D blend-space pose — the two bracketing clips sampled at the SAME normalized
// phase (the sync rule), blended by the exact integer weight via the EXISTING BlendLocalPoses (the
// reused 2-pose seam). w == 0 / w == kOne take the DIRECT sample path (identity-at-sample EXACT).
// Compose to a palette with the existing PaletteFromLocalPose when skinning.
inline std::vector<JointPose> EvaluatePose1D(const Skeleton& sk, const std::vector<Animation>& clips,
                                             const BlendSpace1D& s, fx param, fx phase) {
    const Weights1D w = EvaluateWeights1D(s, param);
    if (w.i0 < 0) return detail::RestPose(sk);
    if (w.i0 == w.i1 || w.w == 0)
        return detail::SamplePhase(sk, clips, s.samples[(size_t)w.i0].clip, phase);
    if (w.w == kOne)
        return detail::SamplePhase(sk, clips, s.samples[(size_t)w.i1].clip, phase);
    return BlendLocalPoses(detail::SamplePhase(sk, clips, s.samples[(size_t)w.i0].clip, phase),
                           detail::SamplePhase(sk, clips, s.samples[(size_t)w.i1].clip, phase),
                           detail::WeightToFloat(w.w));
}

// ===================== The 2D blend space (strafe by speed x direction) ===============================

// One 2D sample: a Q16.16 (x, y) parameter position + a clip index. |x|,|y| <= 2^20 (the overflow
// budget in the banner).
struct BlendSample2D {
    fx      x = 0;
    fx      y = 0;
    int32_t clip = 0;
};

// One authored triangle: three indices into BlendSpace2D::samples. Winding does not matter (every
// predicate canonicalizes by the signed area); DEGENERATE (zero-area) triangles are skipped for
// containment but their edges still participate in the outside-hull clamp scan.
struct BlendTri {
    int32_t a = 0;
    int32_t b = 0;
    int32_t c = 0;
};

// The 2D blend space: sample points + the FIXED AUTHORED triangulation (the banner's v1 choice).
struct BlendSpace2D {
    std::vector<BlendSample2D> samples;
    std::vector<BlendTri>      tris;
};

// The 2D weight result. Inside a triangle: tri >= 0, {i0,i1,i2} = that triangle's authored (a,b,c),
// w0+w1+w2 == kOne EXACTLY (w2 is the residual), clamped == false. Outside the hull: clamped == true,
// the result is the 2-sample edge lerp {i0,i1,w0,w1} (i2 == -1, w2 == 0, w0+w1 == kOne) on the
// nearest edge (tri = the edge's triangle). Empty space: tri == -1.
struct Weights2D {
    int32_t tri = -1;
    int32_t i0 = -1, i1 = -1, i2 = -1;
    fx      w0 = 0, w1 = 0, w2 = 0;
    bool    clamped = false;
};

namespace detail {

// Orient: the int64 orientation cross product (b - a) x (c - a) — the NAV/convex integer discipline.
inline int64_t Orient(fx ax, fx ay, fx bx, fx by, fx cx, fx cy) {
    return ((int64_t)bx - ax) * ((int64_t)cy - ay) - ((int64_t)by - ay) * ((int64_t)cx - ax);
}

}  // namespace detail

// EvaluateWeights2D: point-in-triangle (boundary-inclusive, first triangle in authored order wins)
// -> int64 Cramer barycentric weights (w2 = the exact residual); outside the hull -> the nearest-edge
// clamp (the banner's pinned conventions).
inline Weights2D EvaluateWeights2D(const BlendSpace2D& s, fx px, fx py) {
    Weights2D out;
    const int triCount = (int)s.tris.size();
    const int sampleCount = (int)s.samples.size();
    auto validTri = [&](const BlendTri& t) {
        return t.a >= 0 && t.a < sampleCount && t.b >= 0 && t.b < sampleCount && t.c >= 0 &&
               t.c < sampleCount;
    };
    // (1) Containment scan: the FIRST triangle whose three canonicalized edge functions are >= 0.
    for (int ti = 0; ti < triCount; ++ti) {
        const BlendTri& t = s.tris[(size_t)ti];
        if (!validTri(t)) continue;
        const BlendSample2D& A = s.samples[(size_t)t.a];
        const BlendSample2D& B = s.samples[(size_t)t.b];
        const BlendSample2D& C = s.samples[(size_t)t.c];
        const int64_t D = detail::Orient(A.x, A.y, B.x, B.y, C.x, C.y);   // 2x signed area
        if (D == 0) continue;                                             // degenerate: skip
        const int64_t sgn = D > 0 ? 1 : -1;
        const int64_t n0 = sgn * detail::Orient(px, py, B.x, B.y, C.x, C.y);   // -> w0 (vertex a)
        const int64_t n1 = sgn * detail::Orient(A.x, A.y, px, py, C.x, C.y);   // -> w1 (vertex b)
        const int64_t n2 = sgn * detail::Orient(A.x, A.y, B.x, B.y, px, py);   // -> w2 (vertex c)
        if (n0 < 0 || n1 < 0 || n2 < 0) continue;                              // outside this tri
        const int64_t aD = sgn * D;
        out.tri = ti;
        out.i0 = t.a; out.i1 = t.b; out.i2 = t.c;
        out.w0 = (fx)((n0 << kFrac) / aD);            // truncating Q16.16 (budget: < 2^59 — banner)
        out.w1 = (fx)((n1 << kFrac) / aD);
        out.w2 = kOne - out.w0 - out.w1;              // THE RESIDUAL — the sum is kOne EXACTLY
        return out;
    }
    // (2) Outside the hull (or no valid triangle contains p): the nearest-edge clamp. Scan ALL edges
    // of ALL valid triangles; strict '<' keeps the first (tri, edge) on ties (the pinned tie-break).
    int64_t bestD2 = INT64_MAX;
    for (int ti = 0; ti < triCount; ++ti) {
        const BlendTri& t = s.tris[(size_t)ti];
        if (!validTri(t)) continue;
        const int32_t idx[3] = {t.a, t.b, t.c};
        for (int e = 0; e < 3; ++e) {
            const BlendSample2D& A = s.samples[(size_t)idx[e]];
            const BlendSample2D& B = s.samples[(size_t)idx[(e + 1) % 3]];
            const int64_t ex = (int64_t)B.x - A.x, ey = (int64_t)B.y - A.y;
            const int64_t len2 = ex * ex + ey * ey;
            fx tq = 0;
            if (len2 > 0) {
                const int64_t dot = ((int64_t)px - A.x) * ex + ((int64_t)py - A.y) * ey;
                int64_t tn = (dot << kFrac) / len2;                    // truncating projection
                if (tn < 0) tn = 0;
                if (tn > kOne) tn = kOne;
                tq = (fx)tn;
            }
            // The projected point (per-component truncating Q16.16) and its int64 squared distance.
            const fx qx = (fx)(A.x + ((ex * tq) >> kFrac));
            const fx qy = (fx)(A.y + ((ey * tq) >> kFrac));
            const int64_t dx = (int64_t)px - qx, dy = (int64_t)py - qy;
            const int64_t d2 = dx * dx + dy * dy;
            if (d2 < bestD2) {
                bestD2 = d2;
                out.tri = ti;
                out.i0 = idx[e]; out.i1 = idx[(e + 1) % 3]; out.i2 = -1;
                out.w0 = kOne - tq; out.w1 = tq; out.w2 = 0;
                out.clamped = true;
            }
        }
    }
    return out;
}

// EvaluatePose2D: the 2D blend-space pose at the synchronized phase. Inside a triangle: the pinned
// LEFT-FOLD 3-way composition over the EXISTING BlendLocalPoses (see the banner — inner weight
// (w1 << kFrac)/(w0 + w1) truncating, outer weight w2; NOT claimed equal to the true weighted
// quaternion average — nlerp is non-associative, the ORDER is the contract). Any weight == kOne (and
// the clamped-edge w == 0/kOne cases) takes the DIRECT sample path — identity-at-vertex EXACT.
inline std::vector<JointPose> EvaluatePose2D(const Skeleton& sk, const std::vector<Animation>& clips,
                                             const BlendSpace2D& s, fx px, fx py, fx phase) {
    const Weights2D w = EvaluateWeights2D(s, px, py);
    if (w.tri < 0 || w.i0 < 0) return detail::RestPose(sk);
    auto clipOf = [&](int32_t sample) { return s.samples[(size_t)sample].clip; };
    if (w.w0 == kOne) return detail::SamplePhase(sk, clips, clipOf(w.i0), phase);
    if (w.w1 == kOne) return detail::SamplePhase(sk, clips, clipOf(w.i1), phase);
    if (w.i2 < 0) {                                          // the clamped-edge 2-sample lerp
        return BlendLocalPoses(detail::SamplePhase(sk, clips, clipOf(w.i0), phase),
                               detail::SamplePhase(sk, clips, clipOf(w.i1), phase),
                               detail::WeightToFloat(w.w1));
    }
    if (w.w2 == kOne) return detail::SamplePhase(sk, clips, clipOf(w.i2), phase);
    // The 3-way LEFT FOLD in authored (a, b, c) order.
    const int64_t w01 = (int64_t)w.w0 + (int64_t)w.w1;
    std::vector<JointPose> inner;
    if (w01 <= 0) {
        inner = detail::SamplePhase(sk, clips, clipOf(w.i1), phase);   // degenerate: all weight on c
    } else {
        const fx wIn = (fx)((((int64_t)w.w1) << kFrac) / w01);         // truncating inner weight
        inner = (wIn == 0)
                    ? detail::SamplePhase(sk, clips, clipOf(w.i0), phase)
                    : (wIn == kOne
                           ? detail::SamplePhase(sk, clips, clipOf(w.i1), phase)
                           : BlendLocalPoses(detail::SamplePhase(sk, clips, clipOf(w.i0), phase),
                                             detail::SamplePhase(sk, clips, clipOf(w.i1), phase),
                                             detail::WeightToFloat(wIn)));
    }
    if (w.w2 == 0) return inner;
    return BlendLocalPoses(inner, detail::SamplePhase(sk, clips, clipOf(w.i2), phase),
                           detail::WeightToFloat(w.w2));
}

// ===================== Parameter slewing (tick-based, deterministic) ==================================

// SlewParam: move `current` toward `target` by at most `rate` Q16.16 PER TICK. rate <= 0 -> disabled
// (instant — the identity with an unslewed set). Pure integer, a pure function of its arguments.
inline fx SlewParam(fx current, fx target, fx rate) {
    if (rate <= 0) return target;
    const int64_t d = (int64_t)target - (int64_t)current;
    if (d > rate) return (fx)(current + rate);
    if (d < -(int64_t)rate) return (fx)(current - rate);
    return target;
}

// A 2D parameter point (the 2D blend-space cursor).
struct BlendParam2 {
    fx x = 0;
    fx y = 0;
};

// SlewParam2: per-axis INDEPENDENT slew (the documented box clamp — radial is a future refinement).
inline BlendParam2 SlewParam2(BlendParam2 current, BlendParam2 target, fx rate) {
    return BlendParam2{SlewParam(current.x, target.x, rate), SlewParam(current.y, target.y, rate)};
}

// AdvancePhase: the wrapped normalized-phase accumulator — phase' = (phase + rate) mod kOne. `rate`
// is the normalized phase advance PER TICK (Q16.16; rate = tickDt / loopDuration, host-authored);
// both inputs are expected in [0, kOne).
inline fx AdvancePhase(fx phase, fx rate) {
    int64_t p = ((int64_t)phase + (int64_t)rate) % kOne;
    if (p < 0) p += kOne;
    return (fx)p;
}

// ===================== The FSM adapter (item 4 — the thin driver) =====================================

// BlendDriver1D: a 1D blend space whose parameter is BOUND to a state_machine.h float parameter.
// state_machine.h has no pose-source seam (states are clip indices), so this helper is the documented
// adapter shape: the caller ticks the FSM as usual, then TickBlendDriver1D pulls the bound param
// (quantized ONCE — the float->fx boundary at this seam), slews, and advances the phase;
// EvaluateBlendDriver1D is the pose source. POD state {param, phase} — snapshot-friendly.
struct BlendDriver1D {
    int32_t fsmParam = -1;   // StateMachine parameter index (ParamIndex/AddParam), -1 = unbound
    fx      slewRate = 0;    // Q16.16/tick, 0 = instant
    fx      phaseRate = 0;   // normalized phase advance per tick
    fx      param = 0;       // the slewed current parameter (state)
    fx      phase = 0;       // the wrapped normalized phase (state)
};

inline void TickBlendDriver1D(BlendDriver1D& d, const StateMachine& fsm) {
    const fx target = (d.fsmParam >= 0) ? QuantizeFx(fsm.GetParam(d.fsmParam)) : d.param;
    d.param = SlewParam(d.param, target, d.slewRate);
    d.phase = AdvancePhase(d.phase, d.phaseRate);
}

inline std::vector<JointPose> EvaluateBlendDriver1D(const BlendDriver1D& d, const Skeleton& sk,
                                                    const std::vector<Animation>& clips,
                                                    const BlendSpace1D& s) {
    return EvaluatePose1D(sk, clips, s, d.param, d.phase);
}

// ===================== Digests (the pinned-golden currency) ===========================================

// DigestPoseQ: quantize every JointPose field ONCE (Q16.16 via mm::QuantizeFx — THE boundary) and
// fold the words through the MM1 FNV-1a-64 primitive (field-wise, layout/endianness-independent).
inline uint64_t DigestPoseQ(const std::vector<JointPose>& pose) {
    uint64_t h = 14695981039346656037ull;
    auto word = [&](float v) { h = mm::detail::Fnv1a64Word(h, (uint32_t)QuantizeFx(v)); };
    for (const JointPose& p : pose) {
        word(p.t.x); word(p.t.y); word(p.t.z);
        word(p.r.x); word(p.r.y); word(p.r.z); word(p.r.w);
        word(p.s.x); word(p.s.y); word(p.s.z);
    }
    return h;
}

// ===================== The shared showcase scenario (--an1-blend-shot, both backends) ================
// The WV1/SP1 header-shared-scenario pattern: RunBlendShotScenario + RenderBlendShot are the ONE
// implementation both the Vulkan --an1-blend-shot and the Metal --an1-blend call — strict-zero
// cross-backend BY CONSTRUCTION (pure CPU integer scenario; the raster consumes only integers).

// MakeBsRunClip: the RUN stride — the MM1 walk loop at EXACTLY 2x rate (all keyframe times halved,
// duration 1.0 s; root advances 0 -> 3.0 in +z at 3.0 u/s; feet swing +-0.375 in antiphase with a
// 0.25 s half-period). All binary fractions — every float lerp in sampling is EXACT (the MM1
// cross-compiler fixture discipline). Run at phase p == walk at phase p (same normalized pose), which
// is what makes the phase-sync proof analytic.
inline Animation MakeBsRunClip() {
    Animation a;
    a.name = "bs_run";
    a.duration = 1.0f;
    Channel root;
    root.jointIndex = 0; root.path = Channel::Path::Translation;
    root.times = {0.0f, 1.0f};
    root.values = {0.0f, 0.0f, 0.0f,   0.0f, 0.0f, 3.0f};
    a.channels.push_back(root);
    Channel lf;
    lf.jointIndex = 1; lf.path = Channel::Path::Translation;
    lf.times = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    lf.values = {-0.25f, 0.0f, -0.375f,
                 -0.25f, 0.0f, +0.375f,
                 -0.25f, 0.0f, -0.375f,
                 -0.25f, 0.0f, +0.375f,
                 -0.25f, 0.0f, -0.375f};
    a.channels.push_back(lf);
    Channel rf;
    rf.jointIndex = 2; rf.path = Channel::Path::Translation;
    rf.times = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    rf.values = {0.25f, 0.0f, +0.375f,
                 0.25f, 0.0f, -0.375f,
                 0.25f, 0.0f, +0.375f,
                 0.25f, 0.0f, -0.375f,
                 0.25f, 0.0f, +0.375f};
    a.channels.push_back(rf);
    return a;
}

// MakeShowcaseClips: {idle, walk, run} — the MM1 fixture clips + the AN1 run (identical bits in the
// test and BOTH showcase harnesses).
inline std::vector<Animation> MakeShowcaseClips() {
    return std::vector<Animation>{mm::MakeMMIdleClip(), mm::MakeMMWalkClip(), MakeBsRunClip()};
}

// MakeShowcaseSpace2D: the locomotion diamond — speed on +y, strafe direction on x. The strafe
// samples ALIAS the walk clip (a fixture simplification, documented: a real rig authors strafe
// clips; the SPACE math is what AN1 pins). 5 samples, 4 authored triangles.
//   4 run(0,2)
//   1 strafeL(-1,1)   2 walk(0,1)   3 strafeR(1,1)
//   0 idle(0,0)
inline BlendSpace2D MakeShowcaseSpace2D() {
    BlendSpace2D s;
    s.samples = {
        BlendSample2D{0, 0, 0},              // 0: idle
        BlendSample2D{-kOne, kOne, 1},       // 1: strafeL (walk clip aliased)
        BlendSample2D{0, kOne, 1},           // 2: walk
        BlendSample2D{kOne, kOne, 1},        // 3: strafeR (walk clip aliased)
        BlendSample2D{0, 2 * kOne, 2},       // 4: run
    };
    s.tris = {
        BlendTri{0, 1, 2},   // idle / strafeL / walk
        BlendTri{0, 2, 3},   // idle / walk / strafeR
        BlendTri{1, 4, 2},   // strafeL / run / walk
        BlendTri{2, 4, 3},   // walk / run / strafeR
    };
    return s;
}

inline constexpr int kShotSteps = 240;                 // the scripted parameter sweep length
inline constexpr fx  kShotSlewRate = kOne / 32;        // 2048/tick — a full unit ramps in 32 ticks
inline constexpr fx  kShotPhaseRate = kOne / 64;       // one walk loop per 64 ticks (2.0 s @ 32 tps)

// The scripted target for tick t (held 60 ticks each): accelerate to run -> strafe right -> strafe
// left -> a target BELOW the hull (0, -1) — the last leg deliberately overshoots to exercise the
// nearest-edge clamp live in the shot.
inline BlendParam2 ShotTarget(int t) {
    if (t < 60) return BlendParam2{0, 2 * kOne};
    if (t < 120) return BlendParam2{kOne, kOne};
    if (t < 180) return BlendParam2{-kOne, kOne};
    return BlendParam2{0, -kOne};
}

// One recorded tick of the shot: the slewed parameter, the active triangle + weights, and the
// quantized blended pose probes (root z + left-foot z — the live pose-path proof).
struct BlendShotStep {
    fx      x = 0, y = 0;
    int32_t tri = -1;
    int32_t i0 = -1, i1 = -1, i2 = -1;
    fx      w0 = 0, w1 = 0, w2 = 0;
    int32_t clamped = 0;
    fx      rootZq = 0;
    fx      footZq = 0;
};

struct BlendShotRun {
    BlendSpace2D               space;
    std::vector<BlendShotStep> steps;
    uint64_t                   pathDigest = 0;      // FNV over the slewed (x, y) per tick
    uint64_t                   weightsDigest = 0;   // FNV over (tri, clamped, i0..i2, w0..w2) per tick
    uint64_t                   poseDigest = 0;      // FNV over the quantized pose probes per tick
    uint64_t                   digest = 0;          // the combined two-run comparison currency
};

// RunBlendShotScenario: the pure function both backends call — the fixed 240-tick slewed parameter
// sweep through the locomotion diamond, phase-synchronized pose evaluation each tick.
inline BlendShotRun RunBlendShotScenario() {
    BlendShotRun run;
    run.space = MakeShowcaseSpace2D();
    const Skeleton sk = mm::MakeMMTestRig();
    const std::vector<Animation> clips = MakeShowcaseClips();
    BlendParam2 cur{0, 0};
    fx phase = 0;
    uint64_t hp = 14695981039346656037ull, hw = hp, hq = hp;
    run.steps.reserve((size_t)kShotSteps);
    for (int t = 0; t < kShotSteps; ++t) {
        cur = SlewParam2(cur, ShotTarget(t), kShotSlewRate);
        const Weights2D w = EvaluateWeights2D(run.space, cur.x, cur.y);
        const std::vector<JointPose> pose = EvaluatePose2D(sk, clips, run.space, cur.x, cur.y, phase);
        BlendShotStep st;
        st.x = cur.x; st.y = cur.y;
        st.tri = w.tri; st.i0 = w.i0; st.i1 = w.i1; st.i2 = w.i2;
        st.w0 = w.w0; st.w1 = w.w1; st.w2 = w.w2;
        st.clamped = w.clamped ? 1 : 0;
        st.rootZq = (pose.size() > 0) ? QuantizeFx(pose[0].t.z) : 0;
        st.footZq = (pose.size() > 1) ? QuantizeFx(pose[1].t.z) : 0;
        run.steps.push_back(st);
        hp = mm::detail::Fnv1a64Word(hp, (uint32_t)st.x);
        hp = mm::detail::Fnv1a64Word(hp, (uint32_t)st.y);
        hw = mm::detail::Fnv1a64Word(hw, (uint32_t)st.tri);
        hw = mm::detail::Fnv1a64Word(hw, (uint32_t)st.clamped);
        hw = mm::detail::Fnv1a64Word(hw, (uint32_t)st.i0);
        hw = mm::detail::Fnv1a64Word(hw, (uint32_t)st.i1);
        hw = mm::detail::Fnv1a64Word(hw, (uint32_t)st.i2);
        hw = mm::detail::Fnv1a64Word(hw, (uint32_t)st.w0);
        hw = mm::detail::Fnv1a64Word(hw, (uint32_t)st.w1);
        hw = mm::detail::Fnv1a64Word(hw, (uint32_t)st.w2);
        hq = mm::detail::Fnv1a64Word(hq, (uint32_t)st.rootZq);
        hq = mm::detail::Fnv1a64Word(hq, (uint32_t)st.footZq);
        phase = AdvancePhase(phase, kShotPhaseRate);
    }
    run.pathDigest = hp;
    run.weightsDigest = hw;
    run.poseDigest = hq;
    uint64_t h = 14695981039346656037ull;
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(hp & 0xffffffffu));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(hp >> 32));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(hw & 0xffffffffu));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(hw >> 32));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(hq & 0xffffffffu));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(hq >> 32));
    run.digest = h;
    return run;
}

// RenderBlendShot: the PURE-INTEGER raster both backends call — strict-zero cross-backend BY
// CONSTRUCTION. Top: the blend-space diagram (authored triangulation edges, sample discs, the slewed
// parameter path — amber inside the hull, red when clamped). Bottom: the per-tick weight-bar strip
// (2 px per tick, segments stacked w0/w1/w2 in the SAMPLE-index palette color; a red cap marks
// clamped ticks). 520x420 BGRA8, fixed constant colors, integer DDA/discs only.
inline void RenderBlendShot(const BlendShotRun& run, std::vector<uint8_t>& bgra, uint32_t& outW,
                            uint32_t& outH) {
    const int W = 520, H = 420;
    outW = (uint32_t)W; outH = (uint32_t)H;
    bgra.assign((size_t)W * H * 4, 0);
    for (size_t p = 0; p < (size_t)W * H; ++p) {   // deep slate ground
        bgra[p * 4 + 0] = 24; bgra[p * 4 + 1] = 18; bgra[p * 4 + 2] = 14; bgra[p * 4 + 3] = 255;
    }
    auto putPx = [&](int ix, int iy, uint8_t r, uint8_t g, uint8_t b) {
        if (ix < 0 || ix >= W || iy < 0 || iy >= H) return;
        uint8_t* dst = &bgra[((size_t)iy * W + ix) * 4];
        dst[0] = b; dst[1] = g; dst[2] = r; dst[3] = 255;
    };
    // Param window: x in [-1.6, 1.6] at 150 px/u around px 260; y in [-1.1, 2.4] at 80 px/u, py 300
    // is y == -1.1 (the diagram occupies py [10, 300]).
    auto mapX = [&](fx x) { return 260 + (int)(((int64_t)x * 150) >> kFrac); };
    auto mapY = [&](fx y) {
        return 300 - (int)((((int64_t)y + (kOne + kOne / 10)) * 80) >> kFrac);
    };
    auto line = [&](int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b) {
        const int dx = x1 - x0, dy = y1 - y0;
        int steps = dx >= 0 ? dx : -dx;
        const int ady = dy >= 0 ? dy : -dy;
        if (ady > steps) steps = ady;
        if (steps == 0) { putPx(x0, y0, r, g, b); return; }
        for (int i = 0; i <= steps; ++i) {   // integer DDA (rounded interpolation — the SP1 raster)
            const int px = x0 + (int)(((int64_t)dx * i * 2 + steps) / (2 * (int64_t)steps));
            const int py = y0 + (int)(((int64_t)dy * i * 2 + steps) / (2 * (int64_t)steps));
            putPx(px, py, r, g, b);
        }
    };
    auto disc = [&](int cx, int cy, int rr, uint8_t r, uint8_t g, uint8_t b) {
        for (int dy = -rr; dy <= rr; ++dy)
            for (int dx = -rr; dx <= rr; ++dx)
                if (dx * dx + dy * dy <= rr * rr) putPx(cx + dx, cy + dy, r, g, b);
    };
    // The sample-index palette (5 showcase samples; wraps for larger spaces).
    const uint8_t pal[5][3] = {
        {214, 196, 120},   // 0 idle  — warm sand
        {140, 200, 120},   // 1 strafeL — sage
        {96, 214, 214},    // 2 walk  — cyan
        {226, 160, 84},    // 3 strafeR — amber
        {196, 120, 214},   // 4 run   — violet
    };
    auto palOf = [&](int32_t i) { return pal[(i >= 0 ? i : 0) % 5]; };
    // (1) Triangulation edges.
    for (const BlendTri& t : run.space.tris) {
        const int32_t idx[3] = {t.a, t.b, t.c};
        for (int e = 0; e < 3; ++e) {
            const BlendSample2D& A = run.space.samples[(size_t)idx[e]];
            const BlendSample2D& B = run.space.samples[(size_t)idx[(e + 1) % 3]];
            line(mapX(A.x), mapY(A.y), mapX(B.x), mapY(B.y), 96, 104, 118);
        }
    }
    // (2) The parameter path (per-tick dots; red when clamped, amber inside).
    for (const BlendShotStep& st : run.steps) {
        const int cx = mapX(st.x), cy = mapY(st.y);
        if (st.clamped) disc(cx, cy, 1, 226, 84, 84);
        else            disc(cx, cy, 1, 232, 190, 96);
    }
    // (3) Sample discs over the path.
    for (size_t i = 0; i < run.space.samples.size(); ++i) {
        const uint8_t* c = palOf((int32_t)i);
        disc(mapX(run.space.samples[i].x), mapY(run.space.samples[i].y), 5, c[0], c[1], c[2]);
    }
    // (4) The per-tick weight-bar strip: py [312, 412), 2 px per tick from px 20 — segments stacked
    // bottom-up i0/i1/i2 in sample colors; a 3-px red cap for clamped ticks.
    const int barBase = 412, barH = 100;
    for (size_t t = 0; t < run.steps.size(); ++t) {
        const BlendShotStep& st = run.steps[t];
        const int x0 = 20 + (int)t * 2;
        const int h0 = (int)(((int64_t)st.w0 * barH) >> kFrac);
        const int h1 = (int)(((int64_t)st.w1 * barH) >> kFrac);
        const int h2 = (int)(((int64_t)st.w2 * barH) >> kFrac);
        int y = barBase;
        const int32_t ids[3] = {st.i0, st.i1, st.i2};
        const int hs[3] = {h0, h1, h2};
        for (int k = 0; k < 3; ++k) {
            if (ids[k] < 0 || hs[k] <= 0) continue;
            const uint8_t* c = palOf(ids[k]);
            for (int yy = 0; yy < hs[k]; ++yy)
                for (int xx = 0; xx < 2; ++xx) putPx(x0 + xx, y - 1 - yy, c[0], c[1], c[2]);
            y -= hs[k];
        }
        if (st.clamped) {
            for (int yy = 0; yy < 3; ++yy)
                for (int xx = 0; xx < 2; ++xx)
                    putPx(x0 + xx, barBase - barH - 2 - yy, 226, 84, 84);
        }
    }
}

}  // namespace bs
}  // namespace hf::anim
