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

// ===== Slice CD2 — THE SWEPT-AABB BROADPHASE (the candidate-pair generator that feeds CD1's TOI) ==========
// CD1 built the time-of-impact PRIMITIVE; CD2 builds the BROADPHASE that decides WHICH pairs to feed it. A
// DISCRETE broadphase keys on each body's INSTANTANEOUS world AABB — so a FAST mover whose box at the tick's
// START and its box at the tick's END don't overlap an obstacle in between is MISSED (the bodies tunnel). The
// fix: each body's broadphase box is the UNION of its AABB at the start pose and at the integrated end pose —
// the SWEPT AABB — so the candidate set covers every pair the sweep crosses. The grid + 27-cell-stencil +
// count->scan->emit machinery is reused VERBATIM from the frozen #23 broadphase (broad::MakeBodyGrid /
// BuildBodyCellTable / the BuildBroadphasePairs stencil), keyed on the swept AABBs instead of pos+radius boxes.
//
// THE int32/int64 SPLIT (the proof-tier call, the established "int64 boundary on the CPU-ref, int32 keyed on
// the GPU" pattern). The swept-AABB PRECOMPUTE — integrate the end pose (fpx::IntegrateBodyFull, int64 fxmul +
// quaternion) + 2x broad::BuildHullAabb (int64 gjk::Support) + the per-axis union — is done HOST-side and the
// resulting int32 lo/hi AABBs are fed IDENTICALLY to the CPU reference (BuildSweptBroadphasePairs) AND the GPU
// (ccd_swept_{count,scan,emit}.comp). The grid+pair work over those int32 AABBs is PURE int32 (FloorDiv + the
// six-compare AabbOverlap + an ascending scatter — NO int64, NO fxmul, NO sqrt), so the three new shaders
// MSL-GENERATE NATIVELY (a TRUE GPU pass on BOTH backends, the broad_pair_* tier). The GPU==CPU memcmp is over
// the int32 pair list + offsets, exactly like BP2.
//
// THE SUPERSET PROOF (the make-or-break): the swept-pair set CONTAINS every pair the discrete (instantaneous-
// AABB) broadphase produces, AND for a fast mover it additionally contains the pairs the discrete set MISSES (a
// sweep that crosses an obstacle non-overlapping at both endpoints). SweptPairsSupersetOfDiscrete returns true
// iff swept ⊇ discrete and reports M = how many pairs the swept set has that the discrete set lacks.
//
// CAVEATS (honest, in scope): the swept AABB is a CONSERVATIVE straight-line union of two pose-AABBs — a
// ROTATING body's true swept volume is larger than the union of its two endpoint AABBs (a documented limit;
// the integrated end pose includes the body's own rotation over dt, so the endpoint boxes already bracket the
// rotation, but the IN-BETWEEN sweep of a fast spinner can bulge slightly past the union — a future-slice
// margin refinement). The swept AABBs inflate the candidate count (more pairs than discrete — that is the
// POINT). Convex hulls only. cellSize must be >= the max swept-AABB diameter so the ±1 stencil stays exact.

// AabbUnion(a, b): the per-axis min/max union of two AABBs (the smallest AABB containing both). Pure int32.
inline fpx::FxAabb AabbUnion(const fpx::FxAabb& a, const fpx::FxAabb& b) {
    return fpx::FxAabb{
        fpx::FxVec3{a.lo.x < b.lo.x ? a.lo.x : b.lo.x,
                    a.lo.y < b.lo.y ? a.lo.y : b.lo.y,
                    a.lo.z < b.lo.z ? a.lo.z : b.lo.z},
        fpx::FxVec3{a.hi.x > b.hi.x ? a.hi.x : b.hi.x,
                    a.hi.y > b.hi.y ? a.hi.y : b.hi.y,
                    a.hi.z > b.hi.z ? a.hi.z : b.hi.z},
    };
}

// AabbCenter(a): the AABB centre = (lo+hi)/2 per axis (Q16.16 integer halve, matching broad.h's proxy-body pos).
// This is the cell-assignment key for the grid (the same value the GPU computes from the lo/hi it reads). int32.
inline fpx::FxVec3 AabbCenter(const fpx::FxAabb& a) {
    return fpx::FxVec3{(a.lo.x + a.hi.x) / 2, (a.lo.y + a.hi.y) / 2, (a.lo.z + a.hi.z) / 2};
}

// SweptHullAabb(hull, body, dt): the body's SWEPT world AABB over [0, dt] = broad::BuildHullAabb at the START
// pose UNION broad::BuildHullAabb at the integrated END pose. bodyEnd = a COPY of `body` advanced forward by dt
// via fpx::IntegrateBodyFull with ZERO gravity (sweep along the current velocity/angVel only). A STATIC body
// (no kFlagDynamic) does not move -> IntegrateBodyFull is a no-op -> the swept AABB == the instantaneous AABB
// (statics don't sweep). The end-pose support queries are int64 (the int64 BOUNDARY); the RESULT is an FxAabb
// of int32 Q16.16 coords. Every world vertex of the body at EITHER endpoint lies inside this AABB.
inline fpx::FxAabb SweptHullAabb(const gjk::FxHull& hull, const fpx::FxBody& body, fx dt) {
    const fpx::FxAabb startBox = broad::BuildHullAabb(hull, body);
    fpx::FxBody bodyEnd = body;                          // a COPY, advanced along the current velocity
    const fpx::FxVec3 kZeroG{0, 0, 0};                   // CCD sweeps along velocity only — ZERO gravity
    IntegrateBodyFull(bodyEnd, kZeroG, dt);
    const fpx::FxAabb endBox = broad::BuildHullAabb(hull, bodyEnd);
    return AabbUnion(startBox, endBox);
}

// BuildSweptAabbs(world, dt, aabbsOut): the host-side swept-AABB precompute — one SweptHullAabb per body, in
// world-index order. THIS is the int64 boundary (integrate + support); the resulting int32 lo/hi AABBs are fed
// to BOTH the CPU pair generator below AND the GPU ccd_swept shaders (the established split).
inline void BuildSweptAabbs(const gjk::HullWorld& world, fx dt, std::vector<fpx::FxAabb>& aabbsOut) {
    const uint32_t n = (uint32_t)world.bodies.size();
    aabbsOut.assign((size_t)n, fpx::FxAabb{});
    for (uint32_t i = 0; i < n; ++i)
        aabbsOut[i] = SweptHullAabb(world.hulls[i], world.bodies[i], dt);
}

// ----- The PURE-int32 grid+pair generator over EXPLICIT AABBs (mirrors broad::BuildBroadphasePairs, keyed on
// lo/hi AABBs instead of pos+radius). The grid is built from the AABB CENTRES (broad::MakeBodyGrid over synthetic
// centre proxies — reused VERBATIM); the candidate predicate is fpx::AabbOverlap over the SWEPT AABBs. The
// stencil order + j>i de-dup + count->scan->emit are byte-for-byte the BP2 machinery -> the ccd_swept_*.comp
// shaders reproduce this exactly. This is what GPU==CPU memcmps against. -----------------------------------

// CenterProxyBodies(aabbs): synthetic pos-only FxBody per AABB (pos = AabbCenter), so the frozen
// broad::MakeBodyGrid / BuildBodyCellTable (which key on body.pos) bucket the swept AABBs by their centre cell.
// radius is unused by the grid cell math (only pos.FloorDiv matters); we leave it 0. Pure int32.
inline void CenterProxyBodies(const std::vector<fpx::FxAabb>& aabbs, std::vector<fpx::FxBody>& proxiesOut) {
    proxiesOut.assign(aabbs.size(), fpx::FxBody{});
    for (size_t i = 0; i < aabbs.size(); ++i) proxiesOut[i].pos = AabbCenter(aabbs[i]);
}

// CountSweptPairs(aabbs, proxies, grid, table, perBodyOut): per body i (ascending), scan i's 3x3x3 stencil over
// the centre-proxy cell table (FIXED (dz,dy,dx) order, clamped to the grid) and count j>i with fpx::AabbOverlap
// over the SWEPT AABBs. Per-body-disjoint (race-free). The ccd_swept_count.comp computes THIS per thread.
inline uint32_t CountSweptPairs(const std::vector<fpx::FxAabb>& aabbs,
                                const std::vector<fpx::FxBody>& proxies, const broad::BodyGrid& grid,
                                const broad::BodyCellTable& table, std::vector<uint32_t>& perBodyOut) {
    const uint32_t n = (uint32_t)aabbs.size();
    perBodyOut.assign((size_t)n, 0u);
    uint32_t total = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const fpx::FxCell ci = broad::BodyCellOf(proxies[i].pos, grid.cellSize);
        uint32_t c = 0;
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const fpx::FxCell nc{ci.x + dx, ci.y + dy, ci.z + dz};
            if (nc.x < grid.cellMin.x || nc.x >= grid.cellMin.x + grid.gridDim.x) continue;
            if (nc.y < grid.cellMin.y || nc.y >= grid.cellMin.y + grid.gridDim.y) continue;
            if (nc.z < grid.cellMin.z || nc.z >= grid.cellMin.z + grid.gridDim.z) continue;
            const uint32_t cell = broad::FlatBodyCellId(nc, grid);
            for (uint32_t s = table.cellStart[cell]; s < table.cellStart[cell + 1u]; ++s) {
                const uint32_t j = table.cellBodies[s];
                if (j <= i) continue;                                  // canonical de-dup: emit (i,j) once
                if (fpx::AabbOverlap(aabbs[i], aabbs[j])) ++c;
            }
        }
        perBodyOut[i] = c;
        total += c;
    }
    return total;
}

// BuildSweptPairsFromAabbs(aabbs, proxies, grid, table, perBodyOffset, pairsOut): the full count->scan->emit over
// the swept AABBs (the BP2 BuildBroadphasePairs shape, keyed on the explicit AABBs). The pair list is grouped by
// i (ascending), then stencil cell (FIXED order), then j (ascending within a cell) -> fully deterministic. The
// GPU ccd_swept_{count,scan,emit} mirror this byte-for-byte.
inline void BuildSweptPairsFromAabbs(const std::vector<fpx::FxAabb>& aabbs,
                                     const std::vector<fpx::FxBody>& proxies, const broad::BodyGrid& grid,
                                     const broad::BodyCellTable& table, std::vector<uint32_t>& perBodyOffset,
                                     std::vector<fpx::FxPair>& pairsOut) {
    const uint32_t n = (uint32_t)aabbs.size();
    std::vector<uint32_t> counts;
    const uint32_t total = CountSweptPairs(aabbs, proxies, grid, table, counts);
    perBodyOffset.assign((size_t)n, 0u);
    uint32_t running = 0;
    for (uint32_t i = 0; i < n; ++i) { perBodyOffset[i] = running; running += counts[i]; }
    pairsOut.assign((size_t)total, fpx::FxPair{0u, 0u});
    for (uint32_t i = 0; i < n; ++i) {
        const fpx::FxCell ci = broad::BodyCellOf(proxies[i].pos, grid.cellSize);
        uint32_t local = 0;
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const fpx::FxCell nc{ci.x + dx, ci.y + dy, ci.z + dz};
            if (nc.x < grid.cellMin.x || nc.x >= grid.cellMin.x + grid.gridDim.x) continue;
            if (nc.y < grid.cellMin.y || nc.y >= grid.cellMin.y + grid.gridDim.y) continue;
            if (nc.z < grid.cellMin.z || nc.z >= grid.cellMin.z + grid.gridDim.z) continue;
            const uint32_t cell = broad::FlatBodyCellId(nc, grid);
            for (uint32_t s = table.cellStart[cell]; s < table.cellStart[cell + 1u]; ++s) {
                const uint32_t j = table.cellBodies[s];
                if (j <= i) continue;
                if (fpx::AabbOverlap(aabbs[i], aabbs[j])) {
                    pairsOut[perBodyOffset[i] + local] = fpx::FxPair{i, j};
                    ++local;
                }
            }
        }
    }
}

// BuildSweptBroadphasePairs(world, dt, cellSize, perBodyOffset, pairsOut): THE CD2 swept-AABB candidate-pair
// generator. (1) precompute the swept AABB per body (SweptHullAabb — the int64 boundary); (2) build the BP1
// grid + CSR cell table over the AABB-centre proxies (broad::MakeBodyGrid / BuildBodyCellTable reused VERBATIM);
// (3) run the pure-int32 count->scan->emit pair generation over the swept AABBs. The candidate set is a SUPERSET
// of the discrete broadphase (the sweep covers more). cellSize >= the max swept-AABB diameter so the ±1 stencil
// stays exact. perBodyOffset is the exclusive prefix-sum (no sentinel — the caller appends pairs.size()).
inline void BuildSweptBroadphasePairs(const gjk::HullWorld& world, fx dt, fx cellSize,
                                      std::vector<uint32_t>& perBodyOffset,
                                      std::vector<fpx::FxPair>& pairsOut) {
    std::vector<fpx::FxAabb> aabbs;
    BuildSweptAabbs(world, dt, aabbs);
    std::vector<fpx::FxBody> proxies;
    CenterProxyBodies(aabbs, proxies);
    const broad::BodyGrid grid = broad::MakeBodyGrid(proxies, cellSize);
    const broad::BodyCellTable table = broad::BuildBodyCellTable(proxies, grid);
    BuildSweptPairsFromAabbs(aabbs, proxies, grid, table, perBodyOffset, pairsOut);
}

// BuildDiscreteBroadphasePairs(world, cellSize, perBodyOffset, pairsOut): the DISCRETE reference — the SAME
// pure-int32 grid+pair generator but keyed on each body's INSTANTANEOUS world AABB (broad::BuildHullAabb, NO
// sweep == dt 0 collapse). Used by the superset proof: the discrete set is what a tick-snapshot broadphase
// produces (and what a fast mover would tunnel past). Same grid machinery, same predicate -> a clean SUBSET
// relationship: instantaneous AABB ⊆ swept AABB on every axis, so every discrete pair is a swept pair.
inline void BuildDiscreteBroadphasePairs(const gjk::HullWorld& world, fx cellSize,
                                         std::vector<uint32_t>& perBodyOffset,
                                         std::vector<fpx::FxPair>& pairsOut) {
    const uint32_t n = (uint32_t)world.bodies.size();
    std::vector<fpx::FxAabb> aabbs((size_t)n);
    for (uint32_t i = 0; i < n; ++i) aabbs[i] = broad::BuildHullAabb(world.hulls[i], world.bodies[i]);
    std::vector<fpx::FxBody> proxies;
    CenterProxyBodies(aabbs, proxies);
    const broad::BodyGrid grid = broad::MakeBodyGrid(proxies, cellSize);
    const broad::BodyCellTable table = broad::BuildBodyCellTable(proxies, grid);
    BuildSweptPairsFromAabbs(aabbs, proxies, grid, table, perBodyOffset, pairsOut);
}

// ----- SweptPairMeasure + MeasureSweptPairs: a deterministic summary the showcase/test asserts. -----------
struct SweptPairMeasure {
    uint32_t bodies      = 0;   // the body count
    uint32_t sweptPairs  = 0;   // the swept candidate-pair count (the superset)
    uint32_t discretePairs = 0; // the discrete (instantaneous-AABB) candidate-pair count
    uint32_t missedByDiscrete = 0;  // M = swept pairs the discrete set LACKS (>0 for a fast-mover scene)
    bool     supersetOfDiscrete = false;  // true iff swept ⊇ discrete (every discrete pair is a swept pair)
};

// SweptPairsSupersetOfDiscrete(world, dt, cellSize): THE SUPERSET PROOF. Build both the swept and the discrete
// pair sets (sorted canonical); return true iff the swept set CONTAINS every discrete pair. Reports M (via the
// out-param) = how many swept pairs the discrete set MISSES (a sweep that crosses an obstacle non-overlapping at
// either endpoint instant). For a fast-mover scene M > 0 — the swept broadphase catches what the discrete one
// would tunnel past. EXACT (integer set containment over the sorted lists), NOT within-band.
inline bool SweptPairsSupersetOfDiscrete(const gjk::HullWorld& world, fx dt, fx cellSize,
                                         uint32_t* missedOut = nullptr) {
    std::vector<uint32_t> sOff, dOff;
    std::vector<fpx::FxPair> swept, discrete;
    BuildSweptBroadphasePairs(world, dt, cellSize, sOff, swept);
    BuildDiscreteBroadphasePairs(world, cellSize, dOff, discrete);
    broad::SortPairsCanonical(swept);
    broad::SortPairsCanonical(discrete);
    // swept ⊇ discrete: every discrete pair appears in swept (a merge-style containment over sorted lists).
    bool superset = true;
    size_t si = 0;
    for (size_t di = 0; di < discrete.size(); ++di) {
        const fpx::FxPair& d = discrete[di];
        while (si < swept.size() && (swept[si].i < d.i || (swept[si].i == d.i && swept[si].j < d.j))) ++si;
        if (si >= swept.size() || swept[si].i != d.i || swept[si].j != d.j) { superset = false; break; }
    }
    // M = swept pairs NOT in discrete (the fast-mover catch). Count via set-difference over the sorted lists.
    uint32_t missed = 0;
    {
        size_t di = 0;
        for (size_t k = 0; k < swept.size(); ++k) {
            const fpx::FxPair& s = swept[k];
            while (di < discrete.size() && (discrete[di].i < s.i ||
                   (discrete[di].i == s.i && discrete[di].j < s.j))) ++di;
            const bool inDiscrete = di < discrete.size() && discrete[di].i == s.i && discrete[di].j == s.j;
            if (!inDiscrete) ++missed;
        }
    }
    if (missedOut) *missedOut = missed;
    return superset;
}

// MeasureSweptPairs(world, dt, cellSize): the full deterministic summary — {bodies, sweptPairs, discretePairs,
// missedByDiscrete (M), supersetOfDiscrete}. A PURE function of the inputs (two calls byte-equal).
inline SweptPairMeasure MeasureSweptPairs(const gjk::HullWorld& world, fx dt, fx cellSize) {
    SweptPairMeasure m;
    m.bodies = (uint32_t)world.bodies.size();
    std::vector<uint32_t> sOff, dOff;
    std::vector<fpx::FxPair> swept, discrete;
    BuildSweptBroadphasePairs(world, dt, cellSize, sOff, swept);
    BuildDiscreteBroadphasePairs(world, cellSize, dOff, discrete);
    m.sweptPairs = (uint32_t)swept.size();
    m.discretePairs = (uint32_t)discrete.size();
    uint32_t missed = 0;
    m.supersetOfDiscrete = SweptPairsSupersetOfDiscrete(world, dt, cellSize, &missed);
    m.missedByDiscrete = missed;
    return m;
}

// ===== Slice CD3 — THE SUBSTEPPED CCD WORLD STEP (the mechanism beat: prevents tunneling) ==================
// CD1 built the time-of-impact PRIMITIVE (ConservativeAdvance); CD2 built the swept-AABB BROADPHASE
// (BuildSweptBroadphasePairs, a proven superset of the discrete candidate set). CD3 ASSEMBLES them into a
// SUBSTEPPED CCD world step: each tick, sweep-broadphase the candidate pairs, compute each pair's TOI, advance
// ALL bodies to the EARLIEST impact, resolve that contact via the FROZEN solver (gjk::HullContact ->
// convex::SolveManifoldImpulse + position de-pen, reused VERBATIM from broad::StepHullWorldBP), then consume
// the remaining dt and repeat — so a FAST body STOPS at the first thing it would hit instead of tunneling
// through it in one big tick. Bounded by maxSubsteps. All integer, FIXED substep + pair + body order ->
// bit-identical CPU<->Vulkan<->Metal.
//
// THE NO-TUNNEL HEADLINE: in a scene with a fast mover, StepHullWorldCCD lands the body on the CORRECT (near)
// side of a thin wall, while the discrete broad::StepHullWorldBP on the SAME scene tunnels it through — the two
// final states DIFFER, and the CCD one passes a noTunnel predicate (final pos on the approach side, gap >= 0).
//
// THE int64 REALITY (the CD1 lesson): a CCD tick = swept broadphase + N TOI loops (each an embedded gjk::Gjk)
// + a resolve (gjk::HullContact = Gjk->Epa), all int64. DXC compiles int64 (the Vulkan path); glslc cannot.
// So shaders/ccd_step.comp.hlsl is VULKAN-SPIR-V-ONLY (NOT in hf_gen_msl); the Metal --ccd-step runs the CPU
// StepHullWorldCCDN -> byte-identical to the Vulkan GPU result BY CONSTRUCTION. Because a CCD tick is the
// HEAVIEST single-thread dispatch in the suite, the GPU shader CHUNKS at 1 TICK/DISPATCH from the start
// (barrier between, TDR-safe by construction, bit-identical to one big dispatch).
//
// CAVEATS (honest, in scope): conservative advancement is within the CD1 TOI band; maxSubsteps is a bounded
// budget (a very-fast body past the cap can still tunnel — a documented deterministic limit); the swept
// broadphase inflates the candidate set; the GJ within-band caveats inherited (single-point manifold, diagonal
// inertia). Convex hulls only.

// CcdStepConfig = a broad::BroadStepConfig (the gravity/dt/iters/slop/beta/damp + cellSize) + the substep cap.
struct CcdStepConfig {
    broad::BroadStepConfig bcfg;            // the broadphase-driven hull step config (reused verbatim)
    uint32_t               maxSubsteps = 8; // the bounded substep budget per tick (a body past the cap can tunnel)
};

// ----- ResolveTouchingPairCCD: the FROZEN at-TOI contact resolve over ONE candidate pair (i,j), reusing the
// broad::StepHullWorldBP solve + de-pen machinery VERBATIM at the bodies' CURRENT (just-advanced-to-TOI) poses.
// Runs solveIters impulse sweeps then posIters de-pen sweeps over the single pair. invIW are the per-body world
// inverse inertias (computed once from the post-advance orient, exactly as StepHullWorldBP pass (2)). Identical
// loop body to broad::StepHullWorldBP passes (3)+(4); only the pair source is this one pair. Mutates in place.
inline void ResolveTouchingPairCCD(gjk::HullWorld& world, const broad::BroadStepConfig& bcfg,
                                   const std::vector<convex::FxMat3>& invIW, uint32_t i, uint32_t j) {
    const hf::sim::convex::ConvexStepConfig& cfg = bcfg.cfg;
    // (3) impulse solve — solveIters Gauss-Seidel sweeps over THIS pair (== StepHullWorldBP pass 3 body).
    for (uint32_t sweep = 0; sweep < cfg.solveIters; ++sweep) {
        if (world.bodies[i].invMass == 0 && world.bodies[j].invMass == 0) break;  // static-static
        const convex::ContactManifold m = gjk::HullContact(world.bodies[i], world.hulls[i],
                                                           world.bodies[j], world.hulls[j]);
        if (m.count == 0) continue;
        convex::SolveManifoldImpulse(world.bodies[i], world.bodies[j], invIW[i], invIW[j], m,
                                     cfg.restitution, 1);
    }
    // (4) position de-penetration — posIters sweeps over THIS pair (== StepHullWorldBP pass 4 body).
    for (uint32_t pit = 0; pit < cfg.posIters; ++pit) {
        const fx invSum = world.bodies[i].invMass + world.bodies[j].invMass;
        if (invSum == 0) break;   // both static -> skip
        const convex::ContactManifold m = gjk::HullContact(world.bodies[i], world.hulls[i],
                                                           world.bodies[j], world.hulls[j]);
        if (m.count == 0) continue;
        convex::FxVec3 nrm = m.normal;
        if (convex::FxDot(nrm, fpx::FxSub(world.bodies[j].pos, world.bodies[i].pos)) < 0)
            nrm = convex::FxVec3{-nrm.x, -nrm.y, -nrm.z};
        const fx excess = m.depths[0] - cfg.slop;
        if (excess <= 0) continue;
        const fx corrected = convex::fxmul(excess, cfg.beta);
        const fx wi = convex::fxdiv(world.bodies[i].invMass, invSum);
        const fx wj = fpx::kOne - wi;
        const convex::FxVec3 ci = convex::FxScale(nrm, convex::fxmul(corrected, wi));
        const convex::FxVec3 cj = convex::FxScale(nrm, convex::fxmul(corrected, wj));
        world.bodies[i].pos = fpx::FxSub(world.bodies[i].pos, ci);
        world.bodies[j].pos = convex::FxAdd(world.bodies[j].pos, cj);
    }
}

// ----- StepHullWorldCCD(world, cfg): ONE CCD tick, bounded by cfg.maxSubsteps. The substep loop:
//   (1) BuildSweptBroadphasePairs(world, remainingDt, cellSize) -> the (i,j)-sorted candidate list.
//   (2) per candidate pair ConservativeAdvance(hullA, bodyA, hullB, bodyB, remainingDt) -> FxToi; track the
//       GLOBAL-MIN TOI over hit pairs (deterministic min; ties -> the lowest pair index, the first encountered).
//   (3a) NO pair hit within remainingDt -> integrate ALL dynamic bodies by the FULL remainingDt (fpx::
//        IntegrateBodyFull, the tick's gravity) + per-tick damping -> the tick is done.
//   (3b) else -> integrate ALL dynamic bodies by minToi (sub-dt = minToi, the tick's gravity) + damping.
//   (4)  RESOLVE at minToi: compute the per-body world inverse inertias (once, post-advance), then run the
//        frozen solve over every candidate pair whose ConservativeAdvance HIT (the touching pairs).
//   (5)  remainingDt -= minToi; loop from (1) until remainingDt <= 0 or maxSubsteps is hit.
// All integer, FIXED substep + pair + body order. The candidate pairs are re-derived each substep from the
// CURRENT positions (the broadphase is bit-transparent / not snapshotted state). NOTE the per-substep damping
// matches StepHullWorldBP pass (1): velocity retention applied ONCE per substep advance.
inline void StepHullWorldCCD(gjk::HullWorld& world, const CcdStepConfig& cfg) {
    const hf::sim::convex::ConvexStepConfig& scfg = cfg.bcfg.cfg;
    const uint32_t n = (uint32_t)world.bodies.size();
    fx remainingDt = scfg.dt;

    for (uint32_t sub = 0; sub < cfg.maxSubsteps && remainingDt > 0; ++sub) {
        // (1) swept broadphase over the remaining dt -> the (i,j)-sorted candidate pairs.
        std::vector<uint32_t> offsets;
        std::vector<fpx::FxPair> pairs;
        BuildSweptBroadphasePairs(world, remainingDt, cfg.bcfg.cellSize, offsets, pairs);

        // (2) per-pair TOI -> the global-minimum TOI over hit pairs (lowest-pair-index tie-break = first seen).
        fx minToi = remainingDt;
        bool anyHit = false;
        std::vector<uint8_t> pairHit((size_t)pairs.size(), 0u);
        for (uint32_t p = 0; p < (uint32_t)pairs.size(); ++p) {
            const uint32_t i = pairs[p].i, j = pairs[p].j;
            const FxToi r = ConservativeAdvance(world.hulls[i], world.bodies[i],
                                                world.hulls[j], world.bodies[j], remainingDt);
            if (r.hit) {
                pairHit[p] = 1u;
                if (!anyHit || r.toi < minToi) { minToi = r.toi; }   // strict-less keeps the lowest-index tie
                anyHit = true;
            }
        }

        // (3) advance ALL dynamic bodies by the advance step (minToi on a hit, the full remainingDt otherwise) +
        // per-substep velocity retention (== StepHullWorldBP pass 1). The tick's gravity is applied over the step.
        const fx advance = anyHit ? minToi : remainingDt;
        for (uint32_t i = 0; i < n; ++i) {
            if (convex::IsDynamic(world.bodies[i])) {
                fpx::IntegrateBodyFull(world.bodies[i], scfg.gravity, advance);
                if (scfg.linDamp != fpx::kOne) world.bodies[i].vel = convex::FxScale(world.bodies[i].vel, scfg.linDamp);
                if (scfg.angDamp != fpx::kOne) world.bodies[i].angVel = convex::FxScale(world.bodies[i].angVel, scfg.angDamp);
            }
        }

        if (!anyHit) { remainingDt = 0; break; }   // no impact this tick -> the full step was consumed.

        // (4) resolve at minToi: the per-body world inverse inertias once (post-advance orient), then the frozen
        // solve over every TOUCHING (hit) candidate pair, in the (i,j)-ascending pair order.
        std::vector<convex::FxMat3> invIW((size_t)n);
        for (uint32_t i = 0; i < n; ++i) {
            const convex::FxVec3 invIbody = gjk::FxHullInvInertiaBody(world.hulls[i], world.bodies[i].invMass);
            invIW[i] = convex::WorldInvInertia(world.bodies[i], invIbody);
        }
        for (uint32_t p = 0; p < (uint32_t)pairs.size(); ++p) {
            if (!pairHit[p]) continue;
            ResolveTouchingPairCCD(world, cfg.bcfg, invIW, pairs[p].i, pairs[p].j);
        }

        // (5) consume the substep + loop.
        remainingDt -= minToi;
    }
}

// ----- StepHullWorldCCDN(world, cfg, ticks): run `ticks` CCD ticks. The shader runs THIS verbatim (chunked).
inline void StepHullWorldCCDN(gjk::HullWorld& world, const CcdStepConfig& cfg, uint32_t ticks) {
    for (uint32_t t = 0; t < ticks; ++t) StepHullWorldCCD(world, cfg);
}

// ----- CcdMeasure + MeasureCcd: a deterministic rest/no-tunnel summary of a CCD-stepped world. maxSpeed = max
// FxLength(vel) over the dynamic bodies (the rest test); maxPenetration = max HullContact depth over all i<j
// pairs (the held test); dynamicCount. Pure integer, fixed order (the gjk::MeasureHullStack analog over a
// HullWorld). Two MeasureCcd calls over the same world are byte-equal.
struct CcdMeasure {
    fx       maxSpeed       = 0;
    fx       maxPenetration = 0;
    uint32_t dynamicCount   = 0;
};

inline CcdMeasure MeasureCcd(const gjk::HullWorld& world) {
    CcdMeasure ms;
    const uint32_t n = (uint32_t)world.bodies.size();
    for (uint32_t i = 0; i < n; ++i) {
        if (convex::IsDynamic(world.bodies[i])) {
            ++ms.dynamicCount;
            const fx sp = fpx::FxLength(world.bodies[i].vel);
            if (sp > ms.maxSpeed) ms.maxSpeed = sp;
        }
    }
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = i + 1; j < n; ++j) {
            if (world.bodies[i].invMass == 0 && world.bodies[j].invMass == 0) continue;
            const convex::ContactManifold m = gjk::HullContact(world.bodies[i], world.hulls[i],
                                                               world.bodies[j], world.hulls[j]);
            if (m.count != 0 && m.depths[0] > ms.maxPenetration) ms.maxPenetration = m.depths[0];
        }
    return ms;
}

// ===== Slice CD4 — A BULLET THROUGH A THIN WALL STOPS (the new-physics HEADLINE beat) =======================
// CD1-CD3 built the time-of-impact primitive, the swept-AABB broadphase, and the substepped CCD world step
// (proven byte-identical + no-tunnel vs the discrete step). CD4 is the money-shot DEMONSTRATION: a fast
// projectile fired at a thin static wall is ARRESTED at the wall's near face — the per-tick travel (|vel|*dt) is
// MANY times the wall thickness, so a DISCRETE solver (broad::StepHullWorldBP) steps the projectile from before
// the wall to past it in one move (a guaranteed tunnel), while the CCD step (StepHullWorldCCD) stops it exactly
// at impact, deterministically and bit-identically CPU<->Vulkan<->Metal. CD4 REUSES CD3's StepHullWorldCCD +
// ccd_step.comp ENTIRELY — it adds ONLY the dedicated bullet-wall SCENE + the impact measurement + the
// side-by-side discrete control. NO new shader, NO new RHI. APPEND-only (CD1-CD3 byte-frozen).
//
// CAVEAT (honest, in scope): the projectile speed is chosen below the maxSubsteps budget (a projectile fast
// enough to exceed the budget can still tunnel — a documented deterministic limit, the CD3 lineage); the CD1-CD3
// within-band caveats are inherited. Convex hulls only.

// ----- MakeBulletWallScene(): the deterministic bullet-wall world. A THIN static WALL (a box hull, small
// thickness along the projectile's +X travel axis, large in Y/Z) centred at x=+5; a fast DYNAMIC PROJECTILE (a
// small box) at x=0 with a +X velocity whose per-tick travel (|vel|*dt) is MANY times the wall thickness; plus a
// wide static FLOOR + a couple of slow drops for context. Body indices are FIXED:
//   0 = projectile (dynamic, fast +X), 1 = thin wall (static), 2-3 = slow drops (dynamic), 4 = floor (static).
// The PROJECTILE/WALL geometry matches the showcase + the test (one canonical scene). All integer, fixed.
// (The dt the scene is intended to be stepped at is kBulletDt — a big tick so |vel|*dt >> the wall thickness.)
inline gjk::HullWorld MakeBulletWallScene() {
    const fx kOneL = convex::kOne;
    auto fi   = [&](int v)    { return (fx)((int64_t)v * (int64_t)kOneL); };
    auto fd   = [&](double v) { return (fx)(v * (double)kOneL); };
    auto from = [&](int v)    { return (fx)((int64_t)v << convex::kFrac); };

    auto body = [&](fx x, fx y, fx z, fx vx, bool dyn) {
        FxBody b;
        b.pos    = {x, y, z};
        b.orient = fpx::FxQuat{0, 0, 0, kOneL};
        b.invMass = dyn ? kOneL : 0;
        b.flags   = dyn ? fpx::kFlagDynamic : 0u;
        b.vel    = {vx, 0, 0};
        b.angVel = {0, 0, 0};
        b.radius = 0;
        return b;
    };

    gjk::HullWorld w;
    // 0 = the fast PROJECTILE: a small box at x=0 moving +X at 100 u/s. At the intended dt=1/10 the per-tick
    // travel is 10 world units — ~50x the 0.2-thick wall (a guaranteed discrete tunnel).
    w.bodies.push_back(body(0, 0, 0, from(100), true));
    w.hulls.push_back(gjk::MakeBox(fd(0.4), fd(0.4), fd(0.4)));
    // 1 = the THIN static WALL: a box centred at x=5, half-extent 0.1 on X (0.2 thick), tall on Y/Z.
    w.bodies.push_back(body(from(5), 0, 0, 0, false));
    w.hulls.push_back(gjk::MakeBox(fd(0.1), fi(2), fi(2)));
    // 2-3 = a couple of SLOW drops for context (no horizontal motion; they fall + settle on the floor).
    w.bodies.push_back(body(from(-3), fd(1.7), 0, 0, true));
    w.hulls.push_back(gjk::MakeBox(fd(0.5), fd(0.5), fd(0.5)));
    w.bodies.push_back(body(from(-3), fd(3.0), 0, 0, true));
    w.hulls.push_back(gjk::MakeOcta(fd(0.55)));
    // 4 = a wide static FLOOR (box-hull half-extent 4) for the slow drops.
    w.bodies.push_back(body(0, from(-2), 0, 0, false));
    w.hulls.push_back(gjk::MakeBox(fi(4), kOneL, fi(4)));
    return w;
}

// kBulletDt: the timestep MakeBulletWallScene is intended to be stepped at — a BIG tick (1/10 s) so the
// projectile's per-tick travel (|vel|*dt = 10 world units) is ~50x the 0.2-thick wall: a guaranteed tunnel for
// the discrete solver, the headline the CCD step prevents.
inline constexpr fx kBulletDt = convex::kOne / 10;

// MakeBulletWallConfig(): the CcdStepConfig the bullet-wall scene is stepped at — a PURE HORIZONTAL shot (zero
// gravity so the projectile travels straight at the wall along +X) with dt = kBulletDt and the CD3 settling
// params. maxSubsteps is the CD3 budget (8) — the projectile speed is chosen below it, so the CCD step arrests
// it rather than tunneling (the documented deterministic limit).
inline CcdStepConfig MakeBulletWallConfig() {
    CcdStepConfig c;
    c.bcfg.cfg.gravity     = convex::FxVec3{0, 0, 0};   // pure horizontal shot — isolate the tunnel mechanism
    c.bcfg.cfg.dt          = kBulletDt;
    c.bcfg.cfg.solveIters  = 20;
    c.bcfg.cfg.restitution = 0;
    c.bcfg.cfg.slop        = convex::kOne / 64;
    c.bcfg.cfg.beta        = (fx)((int64_t)4 * convex::kOne / 10);    // 0.4
    c.bcfg.cfg.linDamp     = (fx)((int64_t)90 * convex::kOne / 100);  // 0.90
    c.bcfg.cfg.angDamp     = (fx)((int64_t)5 * convex::kOne / 100);   // 0.05
    c.bcfg.cfg.posIters    = 4;
    c.bcfg.cellSize        = (fx)((int64_t)64 << convex::kFrac);      // >= the max swept-AABB diameter
    c.maxSubsteps          = 8;
    return c;
}

// ----- BulletMeasure: the deterministic bullet-wall impact summary the showcase/test asserts. `tunneled` = did
// the projectile pass through to the wall's FAR face (its centre on the far side of the wall centre)? `impactTick`
// = the FIRST tick (1-based) at which the projectile's centre is within an impact band of the wall's near face
// (the arrest tick; 0 == never reached the wall within the stepped ticks). `arrestX` = the projectile's final
// centre x (the arrested pose). A PURE function of the inputs.
struct BulletMeasure {
    uint32_t tunneled    = 0;   // 0/1 — is the projectile past the wall's FAR face? (the headline: CCD -> 0)
    uint32_t impactTick  = 0;   // the 1-based tick at which the projectile arrested at the wall (0 = none)
    fx       arrestX     = 0;   // the projectile's final centre x (the arrested-at-wall pose)
};

// MeasureBullet(world, wallIndex, projectileIndex): given a stepped bullet-wall world, report whether the
// projectile TUNNELED — its centre x past the wall centre x (the far side). The arrest pose is the projectile's
// current centre. tunneled iff projectileCentreX > wallCentreX (it passed the wall). PURE integer.
inline BulletMeasure MeasureBullet(const gjk::HullWorld& world, uint32_t wallIndex, uint32_t projectileIndex) {
    BulletMeasure m;
    const fx wallX = world.bodies[wallIndex].pos.x;
    const fx projX = world.bodies[projectileIndex].pos.x;
    m.arrestX  = projX;
    m.tunneled = (projX > wallX) ? 1u : 0u;
    return m;
}

// StepBulletImpactTick(world, cfg, wallIndex, projectileIndex, ticks): step the bullet-wall world `ticks` ticks
// with StepHullWorldCCD, returning the 1-based tick at which the projectile FIRST comes to rest against the
// wall (its centre stops advancing toward the wall AND it has not tunneled) — the impactTick — or 0 if it never
// arrested. The world is mutated to its final stepped state. Pure integer, deterministic. Used to populate
// BulletMeasure::impactTick for the showcase/test (the headline {tunneled:false, impactTick:<T>} line).
inline uint32_t StepBulletImpactTick(gjk::HullWorld& world, const CcdStepConfig& cfg,
                                     uint32_t wallIndex, uint32_t projectileIndex, uint32_t ticks) {
    const fx wallX = world.bodies[wallIndex].pos.x;
    uint32_t impactTick = 0;
    fx prevX = world.bodies[projectileIndex].pos.x;
    for (uint32_t t = 0; t < ticks; ++t) {
        StepHullWorldCCD(world, cfg);
        const fx nowX = world.bodies[projectileIndex].pos.x;
        // The FIRST tick the projectile stops advancing toward the wall (its +X progress halts) on the near side
        // -> the arrest tick. nowX <= prevX (no further +X progress) AND it has not passed the wall centre.
        if (impactTick == 0 && nowX <= prevX && nowX < wallX) impactTick = t + 1u;
        prevX = nowX;
    }
    return impactTick;
}

// =========================================================================================================
// Slice CD5 — Deterministic Integer CCD: LOCKSTEP + ROLLBACK (the netcode headline). APPENDED after CD4's
// MeasureBullet/StepBulletImpactTick (CD1-CD4's lines above are BYTE-FROZEN). PURE CPU — NO shader, NO RHI.
// Because StepHullWorldCCD (CD3) is a fully deterministic integer tick whose ONLY mutable replayable state is
// the `bodies` vector (the `hulls` are immutable/shared geometry, and the swept grid + candidate-pair list +
// the per-substep time-of-impact are RE-DERIVED each tick from the CURRENT positions/velocities — so they are
// NOT state to snapshot, they are RECOMPUTED, which is exactly why the lockstep holds THROUGH the swept
// continuous solve), CD5 is the direct BP5/GJ5 twin: the SAME command/snapshot/lockstep/rollback shapes
// broad::RunBroadLockstep/RunBroadRollback use, with StepHullWorldBP -> StepHullWorldCCD. Two peers fed only an
// input-command stream re-derive the entire CCD-driven hull world byte-identical (re-deriving the swept
// broadphase + the per-substep TOIs EACH tick), and a rollback re-sims from a snapshot bit-for-bit. THE
// HEADLINE: mainstream float engines' CCD is non-deterministic root-finding (DISABLED in lockstep netcode);
// HF's is bit-identical and replayable.
//
// REUSE (do NOT redefine): the frozen gjk:: command/snapshot/equality machinery — gjk::ApplyHullCommands,
// gjk::HullSnapshot / gjk::SnapshotHull / gjk::RestoreHull / gjk::HullBodiesEqual (bodies only, hulls
// immutable), convex::ConvexCommand. The ONLY new thing is SimCcdTick (= ApplyHullCommands + StepHullWorldCCD)
// and the two harness functions over it. Both backends run THIS identical CPU harness -> the converged-state
// golden is bit-identical BY CONSTRUCTION (cross-vendor 0 px).

// ----- SimCcdTick(world, cfg, commands, tick): ONE CCD-driven deterministic tick with its inputs -----------
// (1) gjk::ApplyHullCommands(world, commands, tick) — this tick's perturbations, in array order, BEFORE the
// step so the impulse/spin integrates this tick; (2) StepHullWorldCCD(world, cfg) — the CD3 substepped CCD
// tick reused VERBATIM (re-derives the swept broadphase + per-substep TOIs from CURRENT positions/velocities).
// Pure integer, fixed order -> bit-identical on every peer/platform. (The broad::SimBroadTick analog with
// StepHullWorldBP -> StepHullWorldCCD.)
inline void SimCcdTick(gjk::HullWorld& world, const CcdStepConfig& cfg,
                       const std::vector<convex::ConvexCommand>& commands, uint32_t tick) {
    gjk::ApplyHullCommands(world, commands, tick);
    StepHullWorldCCD(world, cfg);
}

// ----- RunCcdLockstep(world0, cfg, commands, ticks, outIdentical): two peers converge from inputs alone -----
// THE peer entry point (the broad::RunBroadLockstep control flow over SimCcdTick). Two independent peers
// (authority + replica) BOTH start from `world0`, BOTH run SimCcdTick for `ticks` with the SAME command stream
// (INPUTS ONLY — no state shared; each re-derives the swept broadphase + per-substep TOIs each tick
// independently -> the SAME impact times -> byte-identical) -> sets *outIdentical (if non-null) to whether the
// two final body vectors are byte-identical (the make-or-break lockstep-THROUGH-the-swept-solve proof) +
// returns the converged AUTHORITY world (for the golden). The peer step order is PINNED.
inline gjk::HullWorld RunCcdLockstep(const gjk::HullWorld& world0, const CcdStepConfig& cfg,
                                     const std::vector<convex::ConvexCommand>& commands,
                                     uint32_t ticks, bool* outIdentical = nullptr) {
    gjk::HullWorld authority = world0;   // a fresh copy
    gjk::HullWorld replica   = world0;   // the second peer fed the SAME inputs
    for (uint32_t t = 0; t < ticks; ++t) {
        SimCcdTick(authority, cfg, commands, t);
        SimCcdTick(replica,   cfg, commands, t);
    }
    if (outIdentical) *outIdentical = gjk::HullBodiesEqual(authority.bodies, replica.bodies);
    return authority;
}

// ----- RunCcdRollback(world0, cfg, authStream, mispredictStream, ticks, rollbackAt, ...) -------------------
// The rollback harness (the broad::RunBroadRollback control flow over SimCcdTick).
// (1) advance ticks 0..rollbackAt from `world0` applying authStream; (2) SAVE a gjk::HullSnapshot AT
// rollbackAt (SnapshotHull — the body world; the swept grid/pairs/TOIs are recomputed, NOT stored); (2b)
// speculatively advance a few ticks (<=3) with the MISPREDICTED stream (a WRONG/extra impulse — the client
// prediction that diverges), capturing that diverged intermediate; (3) ROLLBACK — RestoreHull to the snapshot
// + RE-SIMULATE rollbackAt..ticks with the CORRECT authStream -> the corrected final world. Returns the
// corrected world; sets *outCorrectedEqAuthority (if non-null) to whether it == RunCcdLockstep(world0, cfg,
// authStream, ticks) byte-for-byte, and *outMispredictDiverged (if non-null) to whether the speculative
// pre-rollback state DIFFERED from the authority at the same tick (proving a REAL divergence was corrected,
// not a no-op). cfg + the streams are CONSTANT, NOT snapshotted.
inline gjk::HullWorld RunCcdRollback(const gjk::HullWorld& world0, const CcdStepConfig& cfg,
                                     const std::vector<convex::ConvexCommand>& authStream,
                                     const std::vector<convex::ConvexCommand>& mispredictStream,
                                     uint32_t ticks, uint32_t rollbackAt,
                                     bool* outCorrectedEqAuthority = nullptr,
                                     bool* outMispredictDiverged = nullptr) {
    gjk::HullWorld w = world0;
    // (1) advance 0..rollbackAt with the authoritative stream.
    for (uint32_t t = 0; t < rollbackAt; ++t)
        SimCcdTick(w, cfg, authStream, t);
    // (2) SAVE the snapshot at rollbackAt (the rollback restore point — just the body world).
    const gjk::HullSnapshot snap = gjk::SnapshotHull(w, rollbackAt);
    // (2b) speculatively advance a few ticks with the MISPREDICTED stream (the wrong/extra impulse — the
    // client prediction that diverges). Bounded to the remaining ticks (<=3). Capture the diverged state.
    uint32_t specTicks = ticks - rollbackAt;
    if (specTicks > 3u) specTicks = 3u;
    for (uint32_t s = 0; s < specTicks; ++s)
        SimCcdTick(w, cfg, mispredictStream, rollbackAt + s);
    gjk::HullWorld speculative = w;   // the diverged pre-rollback intermediate (for the "real divergence" proof)
    // (3) ROLLBACK: restore the snapshot (the body world) + re-sim rollbackAt..ticks with the authStream.
    gjk::RestoreHull(w, snap);
    for (uint32_t t = rollbackAt; t < ticks; ++t)
        SimCcdTick(w, cfg, authStream, t);

    if (outCorrectedEqAuthority || outMispredictDiverged) {
        // The authority advanced the SAME number of speculative ticks (rollbackAt + specTicks) with the
        // CORRECT stream — the apples-to-apples comparison point for the misprediction-diverged proof.
        gjk::HullWorld authAtSpec = world0;
        for (uint32_t t = 0; t < rollbackAt + specTicks; ++t)
            SimCcdTick(authAtSpec, cfg, authStream, t);
        if (outMispredictDiverged)
            *outMispredictDiverged = !gjk::HullBodiesEqual(speculative.bodies, authAtSpec.bodies);
        if (outCorrectedEqAuthority) {
            const gjk::HullWorld authFinal = RunCcdLockstep(world0, cfg, authStream, ticks, nullptr);
            *outCorrectedEqAuthority = gjk::HullBodiesEqual(w.bodies, authFinal.bodies);
        }
    }
    return w;
}

}  // namespace ccd
}  // namespace hf::sim
