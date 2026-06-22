#pragma once
// Slice HF1 — Hull Friction + Joints: THE TAGGED FRICTION MANIFOLD ON THE EPA NORMAL (the friction BEACHHEAD of
// FLAGSHIP #30: HULL FRICTION + HULL JOINTS — finishing the deterministic convex-rigid story, hf::sim::hullfric).
// The general convex-hull world (gjk::HullWorld / warmhull) has warm-started, sleeping, robustly-stacking contacts
// — but they are FRICTIONLESS (normal-only) and the bodies cannot be JOINTED. Flagship #30 brings the two
// capabilities that already exist FOR BOXES (fric.h tangential friction, joint.h ball/hinge/limit joints) to
// general convex hulls. HF1 is the friction beachhead: the GJK/EPA keyed hull manifold gains a per-pair integer
// TANGENT BASIS (built ONCE per pair on the sign-corrected EPA normal via fric::MakeTangentBasis) + the
// tangent-impulse accumulator fields + the warm-start cache fields (incl. the basis-axis index for the safe
// warm-start across a basis flip). This is the data the HF2 cone solver consumes. STRICT INTEGER, Q16.16.
//
// THE DESIGN: a NEW friction header that WRAPS the FROZEN warmhull manifold; the basis on the EPA normal.
// warmhull.h is a completed flagship's header — BYTE-FROZEN, #included read-only by gjk/ccd/etc. HF1 must NOT grow
// warmhull::KeyedHullManifoldWH2 / CachedHullContact in place (that changes byte layout -> breaks the wh* goldens).
// Instead hullfric.h defines NEW structs that WRAP the warmhull data + add the friction fields, and a NEW
// BuildHullFrictionManifold that CALLS the frozen warmhull/gjk manifold builder read-only and appends the tangent
// basis. The friction itself is the fric.h / persist.h machinery, applied to the hull manifold:
//   - The tangent basis is built on the CONTINUOUS EPA normal (NOT a SAT axis). fric::MakeTangentBasis(n) (the
//     integer Gram-Schmidt from a unit normal) is fed the SAME sign-corrected A->B normal warmhull::
//     SolveHullManifoldWarm computes (warmhull.h:655-656) — ONCE per pair (all contact points of a manifold share
//     the basis). The basis is discontinuous in n (it flips when the least-aligned cardinal axis changes), which
//     is the HF2 warm-start crux — HF1 lays the groundwork by CACHING the basis-axis index so HF2 can detect a
//     flip and safely cold-start the tangents (a deterministic WH1-style safe miss, integer-clean).
//   - CachedHullFrictionContact mirrors persist::CachedContact (key + 3 impulses) PLUS the basis-axis index;
//     MatchHullFrictionCache / UpdateHullFrictionCache mirror persist::MatchCache / persist::UpdateCache
//     (warm-seed all three impulses when the key matches AND the basis axis is unchanged; else cold-start tangents).
//
// THE int64 REALITY (the WH2/FC1 lesson): the manifold this wraps is built by the int64 manifold::HullContactMulti
// (GJK/EPA + the Sutherland-Hodgman clip), and the tangent basis itself uses int64 FxNormalize/FxDot/FxCross.
// DXC -spirv compiles int64; glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL, so
// shaders/hullfric_points.comp.hlsl is VULKAN-SPIR-V-ONLY (in the Vulkan compile list, NOT in the metal_headless
// hf_gen_msl list). The Metal --hf1-points runs the CPU BuildAllHullFrictionManifolds -> byte-identical to the
// Vulkan GPU result BY CONSTRUCTION (the warmhull_cache.comp convention), while the Vulkan side carries the
// GPU==CPU memcmp proof. hullfric_points.comp copies the BuildHullFrictionManifold tangent-basis body VERBATIM
// (consuming the host-built frozen warmhull manifold positions/depths/normal/keys) so the GPU friction manifold
// is byte-identical to the CPU reference -> the host GPU==CPU memcmp catches any divergence.
//
// Header-only, namespace hf::sim::hullfric, #include warmhull.h / gjk.h / fric.h / persist.h / manifold.h READ-ONLY
// (ALL BYTE-FROZEN — hullfric.h is a brand-new additive sibling that NEVER edits a frozen header).

#include <cstdint>
#include <vector>

#include "sim/warmhull.h"   // read-only/BYTE-FROZEN: KeyedHullManifoldWH2 / BuildKeyedHullManifold /
                            // SolveHullManifoldWarm (the sign-corrected EPA normal) / HullContactKey /
                            // HullContactKeysEqual / MakeHullContactKey + the frozen narrowphase.
#include "sim/gjk.h"        // read-only: HullWorld / FxBody / FxHull (the body+hull access).
#include "sim/fric.h"       // read-only: MakeTangentBasis / LeastAlignedAxis / TangentBasis (the integer basis).
#include "sim/persist.h"    // read-only: the CachedContact + MatchCache/UpdateCache friction-cache TEMPLATE.
#include "sim/manifold.h"   // read-only: WorldInvInertiaFull (HF2's full inertia; HF1 only builds the manifold).

namespace hf::sim {
namespace hullfric {

// Pull the frozen helpers into this namespace (REUSE, do NOT redefine).
using gjk::fx;
using gjk::kOne;
using gjk::FxVec3;
using gjk::FxHull;
using gjk::FxBody;
using gjk::HullWorld;
using convex::FxDot;
using fpx::FxSub;
using warmhull::HullContactKey;
using warmhull::KeyedHullManifoldWH2;
using warmhull::BuildKeyedHullManifold;
using warmhull::HullContactKeysEqual;

// ----- HullFrictionPoint: one contact point + its per-point impulse accumulators (the fric::FrictionPoint shape) -
// point = a warmhull manifold contact point; the three impulse accumulators are ALL ZEROED at build (the warm-start
// hooks HF2's cone solver fills). The shared per-manifold tangent basis lives on HullFrictionManifold (one per pair),
// NOT per point — every contact point of a manifold shares the SAME A->B normal + basis. std430-packable as 4 x int32
// (16 bytes) — the GPU HullFrictionPoint mirror memcmp's against.
struct HullFrictionPoint {
    FxVec3 point;
    fx     normalImpulse   = 0;
    fx     tangentImpulse1 = 0;
    fx     tangentImpulse2 = 0;
};

// ----- HullFrictionManifold: the warmhull manifold data + the per-pair friction basis + per-point accumulators ----
// count / normal / points[] / depths[] = the warmhull KeyedHullManifoldWH2 manifold geometry (BYTE-EQUAL to the
// frozen warmhull::BuildKeyedHullManifold by construction — HF1 ADDS the basis, it does NOT perturb the geometry).
// normal is SIGN-CORRECTED A->B ONCE (== warmhull::SolveHullManifoldWarm's flip). t1/t2 = fric::MakeTangentBasis(
// normal) (computed ONCE per pair — the normal is shared by every point). basisAxis = fric::LeastAlignedAxis(normal)
// (the cardinal-axis index the Gram-Schmidt picked — the crux field HF2 caches to detect a basis FLIP). keys[] =
// the WH1 HullContactKey per point (carried through so HF2's cache can match). pts[] = the per-point accumulators.
// std430-packable (POD; the GPU HullFrictionManifoldGpu mirror compares).
struct HullFrictionManifold {
    uint32_t          count = 0;
    FxVec3            normal;                 // SIGN-CORRECTED A->B (the basis source)
    FxVec3            points[4];              // == warmhull manifold positions
    fx                depths[4] = {0, 0, 0, 0};   // == warmhull manifold depths
    FxVec3            t1, t2;                 // the per-pair tangent basis (one per manifold)
    int32_t           basisAxis = 0;          // fric::LeastAlignedAxis(normal) — the warm-start flip-detect field
    HullContactKey    keys[4];               // keys[i] <-> points[i] (carried through for HF2's cache)
    HullFrictionPoint pts[4];                // the per-point zeroed impulse accumulators
};

// ----- BuildHullFrictionManifold(bodyAIdx, bodyA, hullA, bodyBIdx, bodyB, hullB) -> HullFrictionManifold ----------
// The PINNED steps (the shader copies the tangent-basis body VERBATIM over the host-built manifold):
//   (1) keyed = warmhull::BuildKeyedHullManifold(...) (the FROZEN narrowphase — positions/depths/normal/keys);
//   (2) keyed.manifold.count == 0 -> return {count = 0} (separated/degenerate);
//   (3) n = keyed.manifold.normal SIGN-CORRECTED A->B ONCE (== warmhull::SolveHullManifoldWarm, warmhull.h:655-656);
//   (4) tb = fric::MakeTangentBasis(n) (the FC1 integer Gram-Schmidt, computed ONCE — the normal is shared);
//       basisAxis = fric::LeastAlignedAxis(n) (the cardinal-axis the basis picked);
//   (5) copy the manifold geometry + keys; zero the three accumulators of every point.
// PURE INTEGER, FIXED order -> bit-identical CPU<->Vulkan<->Metal. HF1 only BUILDS the state — applies NO impulse
// (that is HF2); the accumulators stay zero.
inline HullFrictionManifold BuildHullFrictionManifold(uint32_t bodyAIdx, const FxBody& bodyA, const FxHull& hullA,
                                                      uint32_t bodyBIdx, const FxBody& bodyB, const FxHull& hullB) {
    HullFrictionManifold out;   // count 0, geometry/keys zeroed, accumulators 0
    const KeyedHullManifoldWH2 keyed = BuildKeyedHullManifold(bodyAIdx, bodyA, hullA, bodyBIdx, bodyB, hullB);
    if (keyed.manifold.count == 0) return out;   // separated -> {count = 0}

    // (3) sign-correct the EPA normal A->B ONCE (== warmhull::SolveHullManifoldWarm step 0).
    FxVec3 n = keyed.manifold.normal;
    if (FxDot(n, FxSub(bodyB.pos, bodyA.pos)) < 0) n = FxVec3{-n.x, -n.y, -n.z};

    // (4) the FC1 tangent basis at the (shared) A->B normal, computed ONCE; + the basis-axis index (the flip field).
    const fric::TangentBasis tb = fric::MakeTangentBasis(n);
    out.count     = keyed.manifold.count;
    out.normal    = n;
    out.t1        = tb.t1;
    out.t2        = tb.t2;
    out.basisAxis = (int32_t)fric::LeastAlignedAxis(n);

    // (5) copy the manifold geometry + keys; zero every point's accumulators (the cold-start contract).
    const uint32_t cnt = keyed.manifold.count < 4u ? keyed.manifold.count : 4u;
    for (uint32_t i = 0; i < cnt; ++i) {
        out.points[i]            = keyed.manifold.points[i];
        out.depths[i]            = keyed.manifold.depths[i];
        out.keys[i]              = keyed.keys[i];
        out.pts[i].point         = keyed.manifold.points[i];
        out.pts[i].normalImpulse   = 0;   // ZEROED at build (the HF2 warm-start hooks)
        out.pts[i].tangentImpulse1 = 0;
        out.pts[i].tangentImpulse2 = 0;
    }
    return out;
}

// ----- CachedHullFrictionContact: one cached hull contact's key + its 3 persisted impulses + the basis-axis -------
// key = the WH1 HullContactKey (the deterministic identity); normalImpulse / tangentImpulse1 / tangentImpulse2 =
// LAST tick's accumulated impulses (the persist::CachedContact fields); basisAxis = the cardinal-axis index the
// tangent basis used LAST tick (fric::LeastAlignedAxis of last tick's normal). The basisAxis is THE crux field —
// HF2 warm-seeds the tangents ONLY when the key matches AND the basisAxis matches (a basis flip across the frame
// makes last tick's tangent impulses meaningless in the new frame, so they must cold-start). std430-packable as
// 4 x uint32 (key) + 3 x int32 (impulses) + 1 x int32 (basisAxis) = 8 x int32 (32 bytes).
struct CachedHullFrictionContact {
    HullContactKey key;
    fx             normalImpulse   = 0;
    fx             tangentImpulse1 = 0;
    fx             tangentImpulse2 = 0;
    int32_t        basisAxis       = 0;
};

// ----- HullFrictionCache: the store — LAST tick's accumulated impulses + basis axes keyed by HullContactKey -------
// A flat vector + a FIXED linear scan (the contact sets are tiny — the warmhull::HullCache / persist::PersistentCache
// shape). UpdateHullFrictionCache rewrites it to EXACTLY this tick's contacts (absent keys evicted);
// MatchHullFrictionCache reads it (read-only). FIXED order -> deterministic.
struct HullFrictionCache {
    std::vector<CachedHullFrictionContact> entries;
};

// ----- MatchHullFrictionCache(cache, manifold) -> mutates manifold.pts: inherit prior impulses by key+basis -------
// For each contact point i in FIXED order [0, manifold.count): scan the cache in FIXED order for the FIRST entry
// whose key equals manifold.keys[i] (HullContactKeysEqual) AND whose basisAxis equals manifold.basisAxis. If found,
// copy all THREE cached impulses into manifold.pts[i] (the WARM inherit). If the key is absent OR the basis axis
// FLIPPED, leave the accumulators at zero (a cold-start — BuildHullFrictionManifold already zeroed them). The
// basis-axis match is the HF1 groundwork that keeps the warm-start safe across a basis flip (the discontinuity in
// MakeTangentBasis): a flipped basis means t1/t2 changed direction, so last tick's tangent impulses are stale ->
// cold-start. FIXED scan order -> deterministic. The shader copies THIS body VERBATIM (the GPU==CPU memcmp
// make-or-break — though HF1's shot only exercises the basis build; the cache match is HF2's solver path).
inline void MatchHullFrictionCache(const HullFrictionCache& cache, HullFrictionManifold& manifold) {
    const uint32_t cnt = manifold.count < 4u ? manifold.count : 4u;
    for (uint32_t i = 0; i < cnt; ++i) {
        for (size_t e = 0; e < cache.entries.size(); ++e) {
            if (HullContactKeysEqual(cache.entries[e].key, manifold.keys[i])
                && cache.entries[e].basisAxis == manifold.basisAxis) {
                manifold.pts[i].normalImpulse   = cache.entries[e].normalImpulse;
                manifold.pts[i].tangentImpulse1 = cache.entries[e].tangentImpulse1;
                manifold.pts[i].tangentImpulse2 = cache.entries[e].tangentImpulse2;
                break;   // the FIRST matching entry wins (fixed scan order)
            }
        }
    }
}

// ----- UpdateHullFrictionCache(cache, manifold) -> rewrites cache: store THIS tick's contacts, evict absent keys ---
// Mirrors persist::UpdateCache / warmhull::UpdateHullCache: clear the cache, then for each point i in FIXED order
// append {key, normalImpulse, tangentImpulse1, tangentImpulse2, basisAxis}. Keys (or basis axes) not present this
// tick are thereby EVICTED (the new cache IS exactly this tick's set). FIXED order -> deterministic.
inline void UpdateHullFrictionCache(HullFrictionCache& cache, const HullFrictionManifold& manifold) {
    const uint32_t cnt = manifold.count < 4u ? manifold.count : 4u;
    for (uint32_t i = 0; i < cnt; ++i) {
        cache.entries.push_back(CachedHullFrictionContact{
            manifold.keys[i], manifold.pts[i].normalImpulse, manifold.pts[i].tangentImpulse1,
            manifold.pts[i].tangentImpulse2, manifold.basisAxis});
    }
}

// ----- HullFrictionPair: one pair's two bodies + hulls + their GLOBAL body indices (the warmhull::HullKeyPair
// shape, re-exposed here so the showcase/test can feed BuildAllHullFrictionManifolds a fixed pair set). ------------
struct HullFrictionPair {
    uint32_t bodyAIdx; FxBody bodyA; FxHull hullA;
    uint32_t bodyBIdx; FxBody bodyB; FxHull hullB;
};

// ----- BuildAllHullFrictionManifolds(world) -> one HullFrictionManifold per active i<j hull pair ------------------
// The fixed-order all-pairs of warmhull::StepWarmHullWorld (warmhull.h:750-762): for each i<j hull pair in FIXED
// order, skip static-static pairs, BuildHullFrictionManifold from the CURRENT positions; KEEP only pairs with a
// contact (count > 0). Pure integer, FIXED order -> deterministic. (HF1 builds the manifolds; the cone solve over
// them is HF2.)
inline std::vector<HullFrictionManifold> BuildAllHullFrictionManifolds(const HullWorld& world) {
    std::vector<HullFrictionManifold> out;
    const size_t n = world.bodies.size();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (world.bodies[i].invMass == 0 && world.bodies[j].invMass == 0) continue;   // static-static
            HullFrictionManifold m = BuildHullFrictionManifold(
                (uint32_t)i, world.bodies[i], world.hulls[i],
                (uint32_t)j, world.bodies[j], world.hulls[j]);
            if (m.count == 0) continue;
            out.push_back(m);
        }
    }
    return out;
}

// ----- BuildAllHullFrictionManifoldsPairs(pairs) -> one HullFrictionManifold per pair (the fixed pair-list form) --
// The standalone-loop form (the warmhull::BuildHullContactKeys over a HullKeyPair list shape) the HF1 showcase
// feeds a CURATED fixed pair set. One manifold per pair in FIXED order (count may be 0 for a separated pair — kept
// in slot so the GPU thread-per-pair indexing aligns). Pure integer, FIXED order -> deterministic.
inline std::vector<HullFrictionManifold> BuildAllHullFrictionManifoldsPairs(
        const std::vector<HullFrictionPair>& pairs) {
    std::vector<HullFrictionManifold> out;
    out.reserve(pairs.size());
    for (const HullFrictionPair& p : pairs) {
        out.push_back(BuildHullFrictionManifold(p.bodyAIdx, p.bodyA, p.hullA,
                                                p.bodyBIdx, p.bodyB, p.hullB));
    }
    return out;
}

// ----- HullFrictionMeasure: the deterministic basis-orthonormality summary over a set of friction manifolds -------
// pairs = total manifolds; pairsWithContact = manifolds with count > 0; totalPoints = the sum of all contact points;
// maxDotErr = the max over all manifolds of |n.t1| / |n.t2| / |t1.t2| (the basis orthogonality residual — 0 for a
// perfect basis, a small integer for the fixed-point drift). Pure integer, fixed order -> deterministic. The
// showcase + test print + assert (maxDotErr below the fixed-point band, both tangents ~ unit length).
struct HullFrictionMeasure {
    uint32_t pairs            = 0;
    uint32_t pairsWithContact = 0;
    uint32_t totalPoints      = 0;
    fx       maxDotErr        = 0;   // max |n.t1| / |n.t2| / |t1.t2| over all manifolds
    fx       minLen           = 0;   // min FxLength over {t1, t2} of all contact manifolds
    fx       maxLen           = 0;   // max FxLength over {t1, t2} of all contact manifolds
};

// MeasureHullFriction(manifolds): accumulate the deterministic basis-orthonormality summary. Pure integer, fixed
// order. Only manifolds with a contact (count > 0) contribute a basis.
inline HullFrictionMeasure MeasureHullFriction(const std::vector<HullFrictionManifold>& manifolds) {
    auto absfx = [](fx v) { return v < 0 ? -v : v; };
    HullFrictionMeasure m;
    m.pairs = (uint32_t)manifolds.size();
    bool first = true;
    for (const HullFrictionManifold& hm : manifolds) {
        if (hm.count == 0) continue;
        ++m.pairsWithContact;
        m.totalPoints += hm.count;
        const fx d0 = absfx(FxDot(hm.normal, hm.t1));
        const fx d1 = absfx(FxDot(hm.normal, hm.t2));
        const fx d2 = absfx(FxDot(hm.t1, hm.t2));
        if (d0 > m.maxDotErr) m.maxDotErr = d0;
        if (d1 > m.maxDotErr) m.maxDotErr = d1;
        if (d2 > m.maxDotErr) m.maxDotErr = d2;
        const fx l1 = fpx::FxLength(hm.t1);
        const fx l2 = fpx::FxLength(hm.t2);
        if (first) { m.minLen = l1; m.maxLen = l1; first = false; }
        if (l1 < m.minLen) m.minLen = l1;
        if (l1 > m.maxLen) m.maxLen = l1;
        if (l2 < m.minLen) m.minLen = l2;
        if (l2 > m.maxLen) m.maxLen = l2;
    }
    return m;
}

// ============================================================================================================
// ===== Slice HF2 — THE WARM CONE SOLVER (APPEND; HF1 above is BYTE-FROZEN) ===================================
// ============================================================================================================
// HF2 adds the accumulated, warm-started, cone-clamped Coulomb friction SOLVE over the HF1 hull manifold — the
// persist::SolveFrictionWarm machinery (proven bit-exact for boxes), with the box inverse-inertia swapped for the
// hull's manifold::WorldInvInertiaFull tensor, EXACTLY how warmhull::SolveHullManifoldWarm adapted the normal-only
// warm solver. The accumulators are HF1's HullFrictionManifold.pts[].normalImpulse/tangentImpulse1/2; the tangent
// frame is HF1's per-pair m.t1/m.t2 on the sign-corrected EPA normal m.normal; the warm seed is HF1's
// MatchHullFrictionCache (consumed here). STRICT INTEGER, FIXED order -> bit-identical CPU<->Vulkan<->Metal.

// Pull the full-inertia helpers HF2's solver needs (REUSE, do NOT redefine).
using convex::FxMat3;
using convex::FxCross;
using convex::FxMat3MulVec;
using fpx::FxScale;
using fpx::FxAdd;
using fpx::fxmul;
using fpx::fxdiv;
using manifold::FxHullFaces;
using manifold::BuildCanonicalFaces;
using manifold::FxHullInertiaBodyFull;
using manifold::WorldInvInertiaFull;

// ----- HullFrictionConfig: the cone coefficient + the warm-solve knobs (the persist::FrictionStepConfig subset HF2
// reads). mu = the Coulomb cone coefficient (0 = frictionless); restitution = the normal bounce factor; iters = the
// accumulated Gauss-Seidel sweep count; slop/beta = the de-pen knobs (UNUSED by the SOLVE-ONLY HF2 — they belong to
// the HF3 step — carried for the HF3 forward-compat). -----
struct HullFrictionConfig {
    fx       mu          = 0;          // Q16.16 Coulomb friction coefficient (0 = frictionless control)
    fx       restitution = 0;          // Q16.16 bounce factor (0 = fully inelastic)
    uint32_t iters       = 8;          // accumulated Gauss-Seidel sweeps
    fx       slop        = kOne / 64;  // (HF3) allowed penetration before pushing apart
    fx       beta        = (fx)((int64_t)8 * kOne / 10);   // (HF3) position-correction fraction
};

// ----- SolveHullFrictionWarm: the persist::SolveFrictionWarm body with the hull WorldInvInertiaFull tensor --------
// Mutates a/b vel+angVel AND m.pts[] accumulators. The accumulators ARRIVE SEEDED (MatchHullFrictionCache for a
// matched contact, or zero for a cold one). The tangent frame (m.t1, m.t2) + the A->B normal (m.normal) are HF1's
// per-pair basis (one per manifold). The PINNED steps (the shader copies THIS body VERBATIM):
//   (1) PRIME ONCE: per point apply the seeded TOTAL impulse J = n*jn + t1*jt1 + t2*jt2 to both bodies (re-inject
//       last tick's converged impulse). A zero seed primes nothing (a cold contact).
//   (2) cfg.iters Gauss-Seidel sweeps; per point in FIXED order, NORMAL then t1 then t2 (the persist order),
//       ACCUMULATED:
//       - NORMAL: vn = (vpB-vpA).n; kn = invMa+invMb + the angular terms (the full WorldInvInertiaFull tensor);
//         djn = fxdiv(-(1+e)*vn, kn); newTotal = max(0, normalImpulse + djn); apply J = n*(newTotal-normalImpulse);
//         normalImpulse = newTotal.
//       - TANGENT t (t1 then t2): vt = (vpB-vpA).t; kt = the same effective-mass form on t; djt = fxdiv(-vt, kt);
//         newTotal = clamp(tangentImpulse + djt, -mu*normalImpulse, +mu*normalImpulse) (the cone on the ACCUMULATED
//         normal); apply Jt = t*(newTotal-tangentImpulse); tangentImpulse = newTotal.
// PURE INTEGER, FIXED order -> bit-identical CPU<->Vulkan<->Metal. (Mirrors warmhull::SolveHullManifoldWarm's
// inertia wiring + adds the persist::SolveFrictionWarm tangent blocks.)
inline void SolveHullFrictionWarm(FxBody& a, FxBody& b, HullFrictionManifold& m,
                                  const FxMat3& invIaW, const FxMat3& invIbW,
                                  const HullFrictionConfig& cfg) {
    if (m.count == 0) return;

    const fx invMassA = a.invMass;
    const fx invMassB = b.invMass;
    const fx mu  = cfg.mu;
    const fx rest = cfg.restitution;
    const uint32_t cnt = m.count < 4u ? m.count : 4u;
    const FxVec3 n  = m.normal;   // already A->B (HF1 sign-corrected it ONCE at build)
    const FxVec3 t1 = m.t1;
    const FxVec3 t2 = m.t2;

    // ---- (1) PRIME ONCE: re-inject the seeded TOTAL impulse at every point (the warm-start kick). ----
    for (uint32_t pi = 0; pi < cnt; ++pi) {
        HullFrictionPoint& fp = m.pts[pi];
        const FxVec3 p  = fp.point;
        const FxVec3 rA = fpx::FxSub(p, a.pos);
        const FxVec3 rB = fpx::FxSub(p, b.pos);
        const FxVec3 J = FxAdd(FxScale(n, fp.normalImpulse),
                               FxAdd(FxScale(t1, fp.tangentImpulse1),
                                     FxScale(t2, fp.tangentImpulse2)));
        a.vel    = fpx::FxSub(a.vel, FxScale(J, invMassA));
        a.angVel = fpx::FxSub(a.angVel, FxMat3MulVec(invIaW, FxCross(rA, J)));
        b.vel    = FxAdd(b.vel, FxScale(J, invMassB));
        b.angVel = FxAdd(b.angVel, FxMat3MulVec(invIbW, FxCross(rB, J)));
    }

    // ---- (2) the accumulated Gauss-Seidel sweeps (apply only the DELTA each time). ----
    for (uint32_t it = 0; it < cfg.iters; ++it) {
        for (uint32_t pi = 0; pi < cnt; ++pi) {
            HullFrictionPoint& fp = m.pts[pi];
            const FxVec3 p  = fp.point;
            const FxVec3 rA = fpx::FxSub(p, a.pos);
            const FxVec3 rB = fpx::FxSub(p, b.pos);

            // ---- NORMAL impulse (the effective-mass form, ACCUMULATED, clamp >= 0) ----
            {
                const FxVec3 vpA = FxAdd(a.vel, FxCross(a.angVel, rA));
                const FxVec3 vpB = FxAdd(b.vel, FxCross(b.angVel, rB));
                const fx vn = FxDot(fpx::FxSub(vpB, vpA), n);
                const FxVec3 raxn = FxCross(rA, n);
                const FxVec3 rbxn = FxCross(rB, n);
                const fx angA = FxDot(n, FxCross(FxMat3MulVec(invIaW, raxn), rA));
                const fx angB = FxDot(n, FxCross(FxMat3MulVec(invIbW, rbxn), rB));
                const fx kn = invMassA + invMassB + angA + angB;
                if (kn > 0) {
                    const fx djn = fxdiv(-fxmul(kOne + rest, vn), kn);
                    fx newTotal = fp.normalImpulse + djn;
                    if (newTotal < 0) newTotal = 0;           // clamp the ACCUMULATED total >= 0
                    const fx applied = newTotal - fp.normalImpulse;
                    const FxVec3 J = FxScale(n, applied);
                    a.vel    = fpx::FxSub(a.vel, FxScale(J, invMassA));
                    a.angVel = fpx::FxSub(a.angVel, FxMat3MulVec(invIaW, FxCross(rA, J)));
                    b.vel    = FxAdd(b.vel, FxScale(J, invMassB));
                    b.angVel = FxAdd(b.angVel, FxMat3MulVec(invIbW, FxCross(rB, J)));
                    fp.normalImpulse = newTotal;
                }
            }

            // ---- TANGENT friction impulses (t1 then t2), the cone on the ACCUMULATED normal ----
            const fx coneLo = -fxmul(mu, fp.normalImpulse);   // -mu * the accumulated jn
            const fx coneHi =  fxmul(mu, fp.normalImpulse);   // +mu * the accumulated jn
            const FxVec3 tangents[2] = {t1, t2};
            fx* tacc[2] = {&fp.tangentImpulse1, &fp.tangentImpulse2};
            for (int ti = 0; ti < 2; ++ti) {
                const FxVec3 t = tangents[ti];
                const FxVec3 vpA = FxAdd(a.vel, FxCross(a.angVel, rA));
                const FxVec3 vpB = FxAdd(b.vel, FxCross(b.angVel, rB));
                const fx vt = FxDot(fpx::FxSub(vpB, vpA), t);
                const FxVec3 raxt = FxCross(rA, t);
                const FxVec3 rbxt = FxCross(rB, t);
                const fx angA = FxDot(t, FxCross(FxMat3MulVec(invIaW, raxt), rA));
                const fx angB = FxDot(t, FxCross(FxMat3MulVec(invIbW, rbxt), rB));
                const fx kt = invMassA + invMassB + angA + angB;
                if (kt <= 0) continue;
                const fx djt = fxdiv(-vt, kt);
                fx newTotal = *tacc[ti] + djt;
                if (newTotal < coneLo) newTotal = coneLo;          // CLAMP the ACCUMULATED tangent to the cone
                else if (newTotal > coneHi) newTotal = coneHi;
                const fx applied = newTotal - *tacc[ti];
                const FxVec3 Jt = FxScale(t, applied);
                a.vel    = fpx::FxSub(a.vel, FxScale(Jt, invMassA));
                a.angVel = fpx::FxSub(a.angVel, FxMat3MulVec(invIaW, FxCross(rA, Jt)));
                b.vel    = FxAdd(b.vel, FxScale(Jt, invMassB));
                b.angVel = FxAdd(b.angVel, FxMat3MulVec(invIbW, FxCross(rB, Jt)));
                *tacc[ti] = newTotal;
            }
        }
    }
}

// ----- HullContactResidual(a, b, m): the post-solve constraint residual (the warm<cold lever metric) -------------
// The max over the manifold's contact points of |a NORMAL velocity that should be >= 0 but is < 0| (a penetrating
// approach the solve failed to remove) + |a TANGENT velocity beyond the friction cone| (a sticking violation). A
// LOWER residual = a better-converged solve. The warm (cache-primed) solve reaches a STRICTLY lower residual than
// the cold (zero-seeded) solve at equal iters — the warm-start lever the whole slice rests on. Evaluated from the
// post-solve body velocities at the contact points (read-only). PURE INTEGER, FIXED order.
inline fx HullContactResidual(const FxBody& a, const FxBody& b, const HullFrictionManifold& m) {
    auto absfx = [](fx v) { return v < 0 ? -v : v; };
    if (m.count == 0) return 0;
    const FxVec3 n  = m.normal;
    const FxVec3 t1 = m.t1;
    const FxVec3 t2 = m.t2;
    const uint32_t cnt = m.count < 4u ? m.count : 4u;
    fx maxRes = 0;
    for (uint32_t pi = 0; pi < cnt; ++pi) {
        const HullFrictionPoint& fp = m.pts[pi];
        const FxVec3 p  = fp.point;
        const FxVec3 rA = fpx::FxSub(p, a.pos);
        const FxVec3 rB = fpx::FxSub(p, b.pos);
        const FxVec3 vpA = FxAdd(a.vel, FxCross(a.angVel, rA));
        const FxVec3 vpB = FxAdd(b.vel, FxCross(b.angVel, rB));
        const FxVec3 dv  = fpx::FxSub(vpB, vpA);
        const fx vn = FxDot(dv, n);
        if (vn < 0) { const fx r = -vn; if (r > maxRes) maxRes = r; }   // a residual approach (penetration)
        const fx vt1 = FxDot(dv, t1);
        const fx vt2 = FxDot(dv, t2);
        const fx a1 = absfx(vt1);
        const fx a2 = absfx(vt2);
        if (a1 > maxRes) maxRes = a1;   // a residual tangential slip (un-arrested sliding)
        if (a2 > maxRes) maxRes = a2;
    }
    return maxRes;
}

// ----- StepHullFrictionSolveOnly(world, cache, cfg, invIW): the HF3 step MINUS integrate/de-pen ------------------
// Build the manifolds (HF1 BuildAllHullFrictionManifolds), warm-seed each from `cache` (MatchHullFrictionCache),
// run SolveHullFrictionWarm per pair in the FIXED i<j order (mutation IN PLACE so later pairs see earlier updates),
// then REWRITE `cache` to EXACTLY this tick's solved contacts (UpdateHullFrictionCache, absent keys evicted). The
// per-body world inverse inertias (the FULL tensor) are precomputed ONCE (== warmhull::StepWarmHullWorld step 2);
// HF2 solves a FIXED manifold set in place — NO integrate, NO de-pen (that is HF3). The solved manifolds are also
// returned (the showcase/test read the converged accumulators). PURE INTEGER, FIXED order -> identical CPU/GPU.
inline std::vector<HullFrictionManifold> StepHullFrictionSolveOnly(
        HullWorld& world, HullFrictionCache& cache, const HullFrictionConfig& cfg) {
    const size_t n = world.bodies.size();

    // World inverse inertias once — the FULL tensor (== warmhull::StepWarmHullWorld step 2, VERBATIM).
    std::vector<FxMat3> invIW(n);
    for (size_t i = 0; i < n; ++i) {
        const FxHullFaces faces = BuildCanonicalFaces(world.hulls[i]);
        const FxMat3 invIbody = FxHullInertiaBodyFull(world.hulls[i], faces, world.bodies[i].invMass);
        invIW[i] = WorldInvInertiaFull(world.bodies[i], invIbody);
    }

    // The FIXED i<j pair order: build the HF1 friction manifold -> warm-seed from the cache -> warm-solve in place.
    std::vector<HullFrictionManifold> solved;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (world.bodies[i].invMass == 0 && world.bodies[j].invMass == 0) continue;   // static-static
            HullFrictionManifold m = BuildHullFrictionManifold(
                (uint32_t)i, world.bodies[i], world.hulls[i],
                (uint32_t)j, world.bodies[j], world.hulls[j]);
            if (m.count == 0) continue;
            MatchHullFrictionCache(cache, m);   // warm-seed the accumulators from last tick (key + basis-axis)
            SolveHullFrictionWarm(world.bodies[i], world.bodies[j], m, invIW[i], invIW[j], cfg);
            solved.push_back(m);
        }
    }

    // Rewrite the cache to EXACTLY this tick's solved contacts (absent keys evicted) — the accumulators persist.
    cache.entries.clear();
    for (const HullFrictionManifold& m : solved) UpdateHullFrictionCache(cache, m);
    return solved;
}

// ============================================================================================================
// ===== Slice HF3 — THE FRICTION-LOCKED HULL WORLD STEP (APPEND; HF1+HF2 above is BYTE-FROZEN) ================
// ============================================================================================================
// HF2 solved a FIXED manifold set in isolation (no integrate / no de-pen). HF3 assembles the FULL per-tick
// WORLD STEP: the warmhull::StepWarmHullWorld 5-pass tick (predict-integrate -> all-pairs broadphase -> warm
// solve -> position de-penetration -> orientation) with the normal-only SolveHullManifoldWarm SWAPPED for the
// HF2 SolveHullFrictionWarm (the cone solver) + the per-pair tangent basis built each tick (HF1
// BuildHullFrictionManifold) + mu threaded + the friction cache PERSISTED across ticks (MatchHullFrictionCache
// warm-seeds each tick, UpdateHullFrictionCache rewrites it). The de-pen + integrate passes are the WH3 passes
// VERBATIM. The money beat: a hull released on a tilted static hull GRIPS and rests at mu>0, but SLIDES off at
// mu=0 — friction is the thing holding it. STRICT INTEGER, FIXED order -> bit-exact CPU<->Vulkan<->Metal.
//
// THE int64 REALITY (the HF1/HF2 lesson): the whole chain is int64 (GJK/EPA + the SH clip + the full inertia +
// the cone solve + the FxLength). DXC -spirv compiles int64; glslc CANNOT, so shaders/hullfric_step.comp is
// VULKAN-SPIR-V-ONLY (NOT in hf_gen_msl), single-thread over the small scene copying StepWarmFrictionHullWorldN
// VERBATIM, chunked 1 tick/dispatch (the Windows ~2s TDR rule). The Metal --hf3-step runs the CPU
// StepWarmFrictionHullWorldN (byte-identical by construction), while the Vulkan side carries the GPU==CPU memcmp.

// Pull the de-pen helper (REUSE, do NOT redefine).
using manifold::HullContactMulti;

// ----- HullFrictionStepConfig: the full-step knobs (HF2's HullFrictionConfig is byte-frozen — a NEW struct) ---
// mu = the Coulomb cone coefficient (0 = frictionless control); gravity = the per-tick acceleration; dt = the
// tick; solveIters = the warm cone-solver Gauss-Seidel sweeps (== HullFrictionConfig.iters); posIters = the
// position de-pen sweeps; restitution = the normal bounce; slop/beta = the de-pen knobs; linDamp/angDamp =
// the per-tick velocity RETAIN factors (kOne == no damping). The WH3 ConvexStepConfig shape + mu, repackaged.
struct HullFrictionStepConfig {
    fx       mu          = 0;                              // Q16.16 Coulomb friction coefficient (0 = frictionless)
    FxVec3   gravity     = FxVec3{0, -(fx)((int64_t)98 * kOne / 10), 0};   // -9.8 host-snapped
    fx       dt          = kOne / 60;
    uint32_t solveIters  = 8;                              // warm cone-solver sweeps
    uint32_t posIters    = 4;                              // position de-pen sweeps
    fx       restitution = 0;
    fx       slop        = kOne / 64;                      // allowed penetration before pushing apart
    fx       beta        = (fx)((int64_t)8 * kOne / 10);   // 0.8 position-correction fraction
    fx       linDamp     = kOne;                           // kOne == no linear damping
    fx       angDamp     = kOne;                           // kOne == no angular damping
};

// ----- StepWarmFrictionHullWorld(world, cache, cfg): ONE deterministic friction-locked WARM-started tick ------
// The warmhull::StepWarmHullWorld 5-pass tick (warmhull.h:722), with step (3) the warm solve SWAPPED to the HF2
// cone solver over the HF1 friction manifold + cache. The cache PERSISTS across ticks (carried in/out). Steps
// (1) predict-integrate + damp, (2) FULL inertia, (4) position de-pen are the WH3 passes VERBATIM. Step (3):
// per overlapping i<j pair in FIXED order: BuildHullFrictionManifold (re-derived from CURRENT positions, builds
// the tangent basis this tick) -> MatchHullFrictionCache (warm-seed key+basisAxis) -> SolveHullFrictionWarm
// (prime once + cfg.solveIters accumulated cone sweeps, IN PLACE) -> capture the converged manifolds, then
// rewrite `cache` to EXACTLY this tick's contacts (UpdateHullFrictionCache, absent keys/basis-flips evicted).
// PURE INTEGER, FIXED order -> identical CPU/GPU. The shader copies THIS body VERBATIM.
inline void StepWarmFrictionHullWorld(HullWorld& world, HullFrictionCache& cache,
                                      const HullFrictionStepConfig& cfg) {
    const size_t n = world.bodies.size();

    // (1) predict-integrate dynamic bodies + per-tick damping (== StepWarmHullWorld step 1, VERBATIM).
    for (size_t i = 0; i < n; ++i) {
        if (convex::IsDynamic(world.bodies[i])) {
            fpx::IntegrateBodyFull(world.bodies[i], cfg.gravity, cfg.dt);
            if (cfg.linDamp != kOne) world.bodies[i].vel = FxScale(world.bodies[i].vel, cfg.linDamp);
            if (cfg.angDamp != kOne) world.bodies[i].angVel = FxScale(world.bodies[i].angVel, cfg.angDamp);
        }
    }

    // (2) world inverse inertias once/tick — the FULL tensor (== StepWarmHullWorld step 2, VERBATIM).
    std::vector<FxMat3> invIW(n);
    for (size_t i = 0; i < n; ++i) {
        const FxHullFaces faces = BuildCanonicalFaces(world.hulls[i]);
        const FxMat3 invIbody = FxHullInertiaBodyFull(world.hulls[i], faces, world.bodies[i].invMass);
        invIW[i] = WorldInvInertiaFull(world.bodies[i], invIbody);
    }

    // (3 — THE SWAP) the warm cone solve over the persistent friction cache, FIXED i<j order, IN PLACE. The HF2
    // config the cone solver reads (mu/restitution/solveIters) carried from the step config.
    HullFrictionConfig solveCfg;
    solveCfg.mu          = cfg.mu;
    solveCfg.restitution = cfg.restitution;
    solveCfg.iters       = cfg.solveIters;
    std::vector<HullFrictionManifold> solved;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (world.bodies[i].invMass == 0 && world.bodies[j].invMass == 0) continue;   // static-static
            HullFrictionManifold m = BuildHullFrictionManifold(
                (uint32_t)i, world.bodies[i], world.hulls[i],
                (uint32_t)j, world.bodies[j], world.hulls[j]);
            if (m.count == 0) continue;
            MatchHullFrictionCache(cache, m);   // warm-seed the accumulators from last tick (key + basis-axis)
            SolveHullFrictionWarm(world.bodies[i], world.bodies[j], m, invIW[i], invIW[j], solveCfg);
            solved.push_back(m);
        }
    }

    // (5a) Rewrite the cache to EXACTLY this tick's solved contacts (absent keys/basis-flips evicted) — the
    // accumulators persist (the warm-start that compounds for a resting contact).
    cache.entries.clear();
    for (const HullFrictionManifold& m : solved) UpdateHullFrictionCache(cache, m);

    // (4) position de-penetration (== StepWarmHullWorld step 4, VERBATIM — re-derives HullContactMulti).
    for (uint32_t pit = 0; pit < cfg.posIters; ++pit) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                const fx invSum = world.bodies[i].invMass + world.bodies[j].invMass;
                if (invSum == 0) continue;
                const convex::ContactManifold m = HullContactMulti(world.bodies[i], world.hulls[i],
                                                                   world.bodies[j], world.hulls[j]);
                if (m.count == 0) continue;
                FxVec3 nrm = m.normal;
                if (FxDot(nrm, FxSub(world.bodies[j].pos, world.bodies[i].pos)) < 0)
                    nrm = FxVec3{-nrm.x, -nrm.y, -nrm.z};
                const fx excess = m.depths[0] - cfg.slop;
                if (excess <= 0) continue;
                const fx corrected = fxmul(excess, cfg.beta);
                const fx wi = fxdiv(world.bodies[i].invMass, invSum);
                const fx wj = kOne - wi;
                const FxVec3 ci = FxScale(nrm, fxmul(corrected, wi));
                const FxVec3 cj = FxScale(nrm, fxmul(corrected, wj));
                world.bodies[i].pos = FxSub(world.bodies[i].pos, ci);
                world.bodies[j].pos = FxAdd(world.bodies[j].pos, cj);
            }
        }
    }
    // (5) orientation was already integrated in step (1).
}

// ----- StepWarmFrictionHullWorldN(world, cache, cfg, ticks): run `ticks` friction-locked steps ----------------
// The cache carries the accumulated friction impulses ACROSS ticks (the warm-start that locks a resting grip).
// Mirrors the chunked 1-tick/dispatch the GPU does (TDR-safe) — bit-identical to one big run by construction.
inline void StepWarmFrictionHullWorldN(HullWorld& world, HullFrictionCache& cache,
                                       const HullFrictionStepConfig& cfg, uint32_t ticks) {
    for (uint32_t t = 0; t < ticks; ++t) StepWarmFrictionHullWorld(world, cache, cfg);
}

// ----- HullGripMeasure: is the dynamic hull at REST on the ramp (the gripped-on-ramp proof) ------------------
// speed   = the dynamic body's max |vel|+|angVel| component magnitude (FxLength of vel + angVel — the rest
// metric); onRamp = the dynamic body is still ABOVE/ON the ramp (its world position has NOT slid past the ramp
// extent along the slide axis); rested = (speed below restThreshold AND onRamp) — the body GRIPPED. A SLID hull
// reads onRamp=false (it left the ramp footprint) and/or a non-zero speed (it kept sliding). Pure integer,
// fixed order -> deterministic.
struct HullGripMeasure {
    fx   speed     = 0;       // FxLength(vel) + FxLength(angVel) of the dynamic body
    bool onRamp    = false;   // still within the ramp footprint along the slide axis
    bool rested    = false;   // speed below threshold AND onRamp (the GRIP)
};

// MeasureHullGrip(world, dynIdx, rampIdx, restThreshold, slideLimit): evaluate the grip of body `dynIdx` resting
// on the ramp body `rampIdx`. The body is ON the ramp if |dynPos - rampPos| along x AND z stays within
// slideLimit (a slid body translates far past the ramp footprint along the downhill axis). Read-only, integer.
inline HullGripMeasure MeasureHullGrip(const HullWorld& world, uint32_t dynIdx, uint32_t rampIdx,
                                       fx restThreshold, fx slideLimit) {
    HullGripMeasure g;
    if (dynIdx >= world.bodies.size() || rampIdx >= world.bodies.size()) return g;
    const FxBody& dyn  = world.bodies[dynIdx];
    const FxBody& ramp = world.bodies[rampIdx];
    g.speed = fpx::FxLength(dyn.vel) + fpx::FxLength(dyn.angVel);
    auto absfx = [](fx v) { return v < 0 ? -v : v; };
    const fx dx = absfx(dyn.pos.x - ramp.pos.x);
    const fx dz = absfx(dyn.pos.z - ramp.pos.z);
    g.onRamp = (dx <= slideLimit) && (dz <= slideLimit);
    g.rested = (g.speed <= restThreshold) && g.onRamp;
    return g;
}

}  // namespace hullfric
}  // namespace hf::sim
