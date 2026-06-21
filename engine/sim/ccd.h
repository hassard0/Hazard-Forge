#pragma once
// Slice CD1 — Deterministic Integer CCD: THE TIME-OF-IMPACT PRIMITIVE (conservative advancement core, the
// BEACHHEAD of FLAGSHIP #24: DETERMINISTIC INTEGER CONTINUOUS COLLISION DETECTION, hf::sim::ccd). A discrete
// solver lets a fast/thin body TUNNEL through geometry in one tick. CD1 builds the core primitive that
// prevents it: a deterministic integer TIME-OF-IMPACT (TOI) between a moving pair of convex hulls, computed by
// CONSERVATIVE ADVANCEMENT — repeatedly query the closest distance (the FROZEN gjk::Gjk, which already returns
// the gap + witnesses for a separated pair), advance the bodies by a step guaranteed not to overshoot, and
// stop when the gap closes. Bit-identical CPU<->Vulkan<->Metal.
//
// Header-only, namespace hf::sim::ccd, #include "sim/broad.h" READ-ONLY (transitively gjk/convex/fric/persist/
// fpx/grain — all BYTE-FROZEN; this header REUSES their helpers, it does NOT redefine the fixed-point format).
//
// THE int64 REALITY (the GJ2/CX1/FPX3 lesson, the honest proof-strength call): the conservative-advance loop
// is a FROZEN gjk::Gjk call (int64 FxRotate/FxDot/fxdiv/FxISqrt) + IntegrateBodyFull (int64 fxmul) + fxdiv/
// FxLength (int64). DXC compiles int64 (the Vulkan path); glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT
// parse int64_t in HLSL. So shaders/ccd_toi.comp.hlsl is VULKAN-SPIR-V-ONLY (in the Vulkan compile list, NOT
// in the Metal hf_gen_msl list); the Metal --ccd-toi runs the CPU ConservativeAdvance -> byte-identical to the
// Vulkan GPU result BY CONSTRUCTION (the gjk_distance.comp / convex_sat.comp split), while the Vulkan side
// carries the GPU==CPU memcmp. ccd_toi.comp copies ConservativeAdvance (incl. the embedded Gjk) VERBATIM, so
// the GPU exercises the EXACT integer ops -> a divergence is exactly what the host GPU==CPU memcmp catches.
//
// DESIGN CRUX (the conservative closing-speed bound, the determinism + correctness make-or-break): for a
// separated convex pair gjk::Gjk returns `separation` (origin -> closest CSO point of A-B; FxLength = the gap)
// + the witnesses. The contact normal is n = FxNormalize(separation). The FASTEST the gap can close is bounded
// ABOVE by the linear closing speed PLUS the angular contribution of BOTH bodies:
//   bound = FxDot(bodyB.vel - bodyA.vel, n) + |bodyA.angVel|*rMaxA + |bodyB.angVel|*rMaxB
// where rMaxA = the max FxLength(worldVert - bodyA.pos) over hullA's world verts (the body bounding radius).
// THE ANGULAR TERMS ARE MANDATORY — without them a rotating pair's gap can close faster than the linear bound
// predicts and the loop OVERSHOOTS the true TOI (the one subtle correctness bug; the test hand-checks a
// rotating case). advance = fxdiv(gap, bound) is a LOWER bound on the true TOI -> never overshoots.

#include <cstdint>

#include "sim/broad.h"   // read-only: transitively gjk::Gjk/FxHull/Support + fpx::FxBody/IntegrateBodyFull/
                         // FxLength/FxNormalize/fxdiv + convex::FxDot + the Q16.16 toolbox.

namespace hf::sim {
namespace ccd {

// Pull the Q16.16 helpers into this namespace (REUSE, do NOT redefine the fixed-point format).
using convex::fx;
using convex::kOne;
using convex::kFrac;
using convex::FxVec3;
using convex::FxDot;        // the Q16.16 dot (int64 intermediate)
using fpx::FxBody;
using fpx::FxLength;        // int64 FxISqrt magnitude
using fpx::FxNormalize;     // int64 integer normalize
using fpx::fxdiv;           // int64 Q16.16 divide
using fpx::FxAdd;
using fpx::FxSub;
using fpx::FxRotate;
using fpx::IntegrateBodyFull;

// kContactEps: the gap (in Q16.16 world units) at or below which the pair is "in contact" -> a HIT. ~1/64 of a
// world unit — well above the integer FxISqrt truncation jitter (~1) yet a tight clearance (the scene gaps are
// ~1 world unit, the gjk_distance band lineage). FIXED, identical CPU/GPU.
inline constexpr fx kContactEps = kOne / 64;

// kToiMaxIter: the FIXED conservative-advancement iteration bound. Conservative advancement converges from
// BELOW (each step is a lower bound on the true TOI), so a fixed budget can stop sub-eps before contact — a
// documented within-band gap (the GJ2-GJ4 EPA-band lineage). 32 sub-steps is ample for the CD1 scenes; the
// loop is a fixed bound so the GPU thread is deterministic + TDR-free.
inline constexpr uint32_t kToiMaxIter = 32u;

// ----- FxToi: the time-of-impact result. toi = the impact time in [0, dt] (Q16.16); hit = 0/1 (a uint for the
// std430 GPU mirror); iterations = the conservative-advancement steps taken (<= kToiMaxIter, for diagnostics).
// std430-packable as { int toi; uint hit; uint iterations; }.
struct FxToi {
    fx       toi        = 0;   // Q16.16 impact time within [0, dt]
    uint32_t hit        = 0;   // 0 = no impact this dt; 1 = impact (gap closed to <= kContactEps) / already overlapping
    uint32_t iterations = 0;   // conservative-advancement steps taken
};

// ----- BodyMaxRadius(hull, body): the body's bounding radius = the max FxLength(worldVert - body.pos) over the
// hull's WORLD verts. World vert = FxRotate(body.orient, localVert) + body.pos, so worldVert - body.pos =
// FxRotate(body.orient, localVert) — a rotation is length-preserving, so this equals max FxLength(localVert)
// (computed directly, no pos needed) up to Q16.16 truncation; we compute it via the rotated world verts to
// match the showcase's literal "world vert minus pos" definition byte-for-byte. count==0 -> 0 (a point).
// int64 (FxRotate + FxISqrt). The shader copies THIS body VERBATIM.
inline fx BodyMaxRadius(const gjk::FxHull& hull, const FxBody& body) {
    fx best = 0;
    for (uint32_t i = 0; i < hull.count; ++i) {
        const FxVec3 worldV = FxAdd(FxRotate(body.orient, hull.verts[i]), body.pos);
        const fx r = FxLength(FxSub(worldV, body.pos));
        if (r > best) best = r;   // strict-greater scan (the lowest-index tie idiom; only the MAX matters)
    }
    return best;
}

// ----- ClosingSpeedBound(bodyA, rMaxA, bodyB, rMaxB, n): the conservative upper bound on the rate the gap can
// close along the contact normal n. = FxDot(bodyB.vel - bodyA.vel, n) + |bodyA.angVel|*rMaxA + |bodyB.angVel|*
// rMaxB. The linear term is the relative-velocity projection onto n; the angular terms bound the surface speed
// a rotating body can add (|omega| * boundingRadius). MANDATORY — without them rotating pairs overshoot. <= 0
// means the pair is RECEDING (or rotating apart faster than it closes) -> no impact this dt. int64 (FxDot/
// FxLength/fxmul). The shader copies THIS body VERBATIM.
inline fx ClosingSpeedBound(const FxBody& bodyA, fx rMaxA,
                            const FxBody& bodyB, fx rMaxB, const FxVec3& n) {
    const FxVec3 relVel = FxSub(bodyB.vel, bodyA.vel);
    const fx linClose = FxDot(relVel, n);                 // relative linear closing speed along n
    const fx angA = fpx::fxmul(FxLength(bodyA.angVel), rMaxA);   // |omegaA| * rMaxA
    const fx angB = fpx::fxmul(FxLength(bodyB.angVel), rMaxB);   // |omegaB| * rMaxB
    return linClose + angA + angB;
}

// ----- ConservativeAdvance(hullA, bodyA, hullB, bodyB, dt): the integer TOI loop using each body's OWN vel/
// angVel. Statics (zero vel + zero angVel) contribute nothing. Pure integer, FIXED iteration bound.
//   - If Gjk reports OVERLAP at the start (already touching/penetrating) -> {toi=0, hit=1}.
//   - Else loop (up to kToiMaxIter): gap = FxLength(separation); if gap <= kContactEps -> {t, hit=1}.
//     n = FxNormalize(separation); bound = ClosingSpeedBound(...); if bound <= 0 -> RECEDING -> {dt, hit=0}.
//     advance = fxdiv(gap, bound) (a LOWER bound on the true TOI -> never overshoots). t += advance; if t >= dt
//     -> {dt, hit=0}. Integrate COPIES of both bodies forward by `advance` via IntegrateBodyFull with ZERO
//     gravity (CCD advances along the current velocity only; the resolve happens in a later slice). Re-Gjk at
//     the new pose. If the loop hits kToiMaxIter -> {t, current hit} (deterministic, within-band).
// ConservativeAdvancePose: the SAME loop, additionally writing the bodies at their FINAL advanced pose (the
// pose the returned toi corresponds to) into poseA/poseB. The no-overshoot proof checks gjk::Gjk at THIS pose
// — the loop's own internally-advanced pose — NOT a single-IntegrateBodyFull reconstruction (which would
// diverge for a ROTATING body, since quaternion integration is non-linear + renormalized per substep, so
// cumulative substeps != one big step). ConservativeAdvance delegates here so the FxToi is identical.
inline FxToi ConservativeAdvancePose(const gjk::FxHull& hullA, const FxBody& bodyA,
                                     const gjk::FxHull& hullB, const FxBody& bodyB, fx dt,
                                     FxBody& poseA, FxBody& poseB) {
    FxToi out;

    // Working COPIES advanced through the substeps (the originals are immutable).
    FxBody a = bodyA;
    FxBody b = bodyB;

    // The body bounding radii are pose-invariant (a rotation preserves length), so compute them once from the
    // start pose — they bound the angular surface speed for the whole advance.
    const fx rMaxA = BodyMaxRadius(hullA, a);
    const fx rMaxB = BodyMaxRadius(hullB, b);

    const FxVec3 kZeroG{0, 0, 0};   // CCD advances along velocity only — ZERO gravity for IntegrateBodyFull.

    fx t = 0;
    for (uint32_t iter = 0; iter < kToiMaxIter; ++iter) {
        out.iterations = iter;
        const gjk::GjkResult g = gjk::Gjk(hullA, a, hullB, b);
        if (g.overlap) {
            // Already overlapping (at t==0 this is "already touching" -> toi 0; deeper in the loop the gap
            // closed past the eps band between substeps -> a hit at the current t).
            out.toi = t;
            out.hit = 1u;
            poseA = a; poseB = b;
            return out;
        }
        const fx gap = FxLength(g.separation);
        if (gap <= kContactEps) {            // close enough -> contact -> HIT at the current time
            out.toi = t;
            out.hit = 1u;
            poseA = a; poseB = b;
            return out;
        }
        const FxVec3 n = FxNormalize(g.separation);
        const fx bound = ClosingSpeedBound(a, rMaxA, b, rMaxB, n);
        if (bound <= 0) {                    // receding (or rotating apart faster than closing) -> no impact
            out.toi = dt;
            out.hit = 0u;
            poseA = a; poseB = b;
            return out;
        }
        const fx advance = fxdiv(gap, bound);   // LOWER bound on the true TOI (never overshoots)
        t += advance;
        if (t >= dt) {                       // the gap would not close within this dt -> no impact
            out.toi = dt;
            out.hit = 0u;
            poseA = a; poseB = b;
            return out;
        }
        // Advance COPIES of both bodies forward by `advance` (ZERO gravity), then re-query at the new pose.
        IntegrateBodyFull(a, kZeroG, advance);
        IntegrateBodyFull(b, kZeroG, advance);
    }
    // Hit the fixed iteration budget without converging -> the current (t, hit) — deterministic, within-band.
    out.toi = t;
    out.hit = 0u;
    out.iterations = kToiMaxIter;
    poseA = a; poseB = b;
    return out;
}

inline FxToi ConservativeAdvance(const gjk::FxHull& hullA, const FxBody& bodyA,
                                 const gjk::FxHull& hullB, const FxBody& bodyB, fx dt) {
    FxBody poseA, poseB;
    return ConservativeAdvancePose(hullA, bodyA, hullB, bodyB, dt, poseA, poseB);
}

// ----- CcdToiMeasure: a deterministic summary the showcase/test asserts. Over a fixed (hullA, bodyA, hullB,
// bodyB, dt) pair set: the number of pairs, the number of HITS, and the SUM of the returned toi values. A PURE
// function of the inputs -> two MeasureCcdToi calls over the same inputs are byte-equal.
struct CcdToiMeasure {
    uint32_t pairs   = 0;
    uint32_t hits    = 0;
    fx       toiSum  = 0;
};

// A CCD candidate pair: two hulls + their moving bodies + the timestep. The showcase/test scene is an array of
// these (the GjkPair analog with a dt).
struct CcdPair {
    gjk::FxHull hullA;
    FxBody      bodyA;
    gjk::FxHull hullB;
    FxBody      bodyB;
    fx          dt = 0;
};

// MeasureCcdToi(pairs): ConservativeAdvance over each pair in FIXED order, accumulating the summary. Pure
// integer, deterministic -> two calls over the same pairs are byte-equal.
inline CcdToiMeasure MeasureCcdToi(const CcdPair* pairs, uint32_t count) {
    CcdToiMeasure m;
    m.pairs = count;
    for (uint32_t i = 0; i < count; ++i) {
        const FxToi r = ConservativeAdvance(pairs[i].hullA, pairs[i].bodyA,
                                            pairs[i].hullB, pairs[i].bodyB, pairs[i].dt);
        if (r.hit) ++m.hits;
        m.toiSum += r.toi;
    }
    return m;
}

}  // namespace ccd
}  // namespace hf::sim
