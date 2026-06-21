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

}  // namespace ccd
}  // namespace hf::sim
