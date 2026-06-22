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

}  // namespace hullfric
}  // namespace hf::sim
