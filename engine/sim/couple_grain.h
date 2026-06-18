#pragma once
// Slice CG1 — Deterministic Rigid<->Grain Coupling: the UNIFIED bodies+grains WORLD + the BODY->GRAIN grid-hash
// QUERY (the BEACHHEAD of FLAGSHIP #12: DETERMINISTIC TWO-WAY RIGID<->GRAIN COUPLING). The SECOND
// material-interaction pairing (after the rigid<->fluid CP flagship): a dynamic fpx::FxBody coupled to the
// bit-exact FRICTIONAL granular pile (engine/sim/grain.h). Drop a heavy body onto/into a poured sand bed — it
// SINKS, the sand piles around and SUPPORTS it, it settles half-buried, and two peers re-simulate the whole
// sink + pile bit-for-bit. Strictly harder than CP: grain has the extra friction physics (GR4) and supports
// the body through a MANY-CONTACT bed (not a buoyant volume). UE5 has no deterministic granular at all, let
// alone deterministic rigid<->granular coupling. CG1 is ONLY the unified world + the body->grain neighbour
// QUERY (which grains each body contains) — the link, NO momentum exchange yet (CG2 support/drag, CG3
// displacement, CG4 the coupled step, CG5 lockstep, CG6 render). Pure CPU, header-only, NO device, NO backend
// symbols, NO <cmath>. Namespace hf::sim::cgrain. The whole flagship reuses the proven engine/sim/fpx.h +
// engine/sim/grain.h toolbox.
//
// THE CP1 TWIN (the one new shape is the GRAIN grid instead of the FLUID grid): couple.h's CP1
// GatherBodyParticles finds, per BODY, the fluid particles inside the body's sphere over the FL2 fluid cell
// table; CG1's GatherBodyGrains finds, per BODY, the GRAINS inside the body's sphere over the GR2 grain cell
// table. A body radius is typically MANY grain cells wide, so the body spans a RANGE of cells (its
// fpx::BodyAabb in cell space), NOT a fixed 27-cell stencil. So GatherBodyGrains iterates the cell range
// [GrainCellOf(body.pos - radius) .. GrainCellOf(body.pos + radius)] over the GR2 grain cell table, and for
// each grain in those cells accepts iff the per-axis box reject |body.pos.axis - g.pos.axis| < body.radius
// passes (a box, NOT a sphere). Built by count->scan->emit (CSR bodyStart[bodyCount+1] + bodyGrains[], grouped
// by body, ascending grain index — the CP1/GR2 EMIT-order discipline, fully deterministic). Pure int32.
//
// THE int32 DECISION (the CP1/FL2/GR2 precedent): the body->grain query is integer index arithmetic + the
// per-axis |body.pos.axis - g.pos.axis| < body.radius box reject (fx is int32 -> a PURE INT32 compare, NO
// products, NO int64, NO sqrt). So the GPU cgrain_body_{count,scan,emit}.comp shaders MSL-generate NATIVELY ->
// a TRUE GPU pass on both Vulkan AND Metal (the strongest cross-vendor proof, like CP1/GR2 — strict
// zero-differing-pixel). The exact radial sphere cull (|g - body| < radius) is DEFERRED to CG2's force / CG3's
// projection (the support/displacement is 0 outside the sphere), so the over-inclusive box candidate is
// correct — exactly the CP1/FL2/GR2 "over-inclusive box, exact cull deferred" discipline.
//
// REUSE MAP (file:line — read-only, NOT modified; couple_grain is the additive sibling):
//   * engine/sim/grain.h: GrainParticle (grain.h:85-92 — pos, prev, vel, invMass, radius, flags),
//     GrainGrid/MakeGrainGrid/GrainCellOf/FlatGrainCellId/GrainCellCount (GR2, grain.h:238-260),
//     GrainCellTable/BuildGrainCellTable (GR2, grain.h:296-328) — the GR2 grid-hash + cell table this query
//     iterates. The GR2 grain_cell_{count,scan,emit}.comp shaders (already MSL-native) are REUSED for the cell
//     table build; CG1 adds the 3 cgrain_body_{count,scan,emit}.comp passes (the SAME shape as CP1's
//     couple_body_* over the grain cell table).
//   * engine/sim/fpx.h: FxBody (pos, vel, invMass, flags, radius), FxVec3, FxAabb/BodyAabb (the body's integer
//     AABB), kFlagDynamic, FxCell. DO NOT modify fpx.h/grain.h/fluid.h/cloth.h/couple.h — couple_grain is the
//     additive sibling.

#include <cstdint>
#include <vector>

#include "sim/fpx.h"     // read-only: fx / FxVec3 / FxBody / FxAabb / BodyAabb / kFlagDynamic / FxCell
#include "sim/grain.h"   // read-only: GrainParticle / GrainGrid / MakeGrainGrid / GrainCellOf / FlatGrainCellId /
                         // GrainCellCount / GrainCellTable / BuildGrainCellTable (the GR2 grid-hash + cell table)

namespace hf::sim {
namespace cgrain {

// Reuse the fpx Q16.16 scalar + vector toolbox verbatim (NO new fixed-point primitives).
using fpx::fx;
using fpx::FxVec3;
using fpx::FxCell;
inline constexpr int kFrac = fpx::kFrac;   // Q16.16 fractional bits (== fpx::kFrac, MUST match the shader)
inline constexpr fx  kOne  = fpx::kOne;    // 1.0 in Q16.16 (65536)

// ----- The unified bodies+grains world (rigid + grain in one Q16.16 frame) ---------------------------------
// The bodies and the grains share the SAME world units (the GR3 GrainSphereFromBody precedent — fpx::FxBody and
// grain::GrainParticle already interoperate). CG1 only needs `bodies` + `grains` + the grain `hSearch`
// cell-size; gravity/dt/groundY are carried for CG2-CG6 (the coupled step + lockstep).
struct CGrainWorld {
    std::vector<fpx::FxBody>            bodies;       // the dynamic rigid bodies (the FPX sim members)
    std::vector<grain::GrainParticle>  grains;       // the granular pool (the GR sim members)
    FxVec3                             gravity;      // carried for CG2-CG6 (the coupled step)
    fx                                 dt = 0;       // carried for CG4-CG6 (the coupled step / lockstep)
    fx                                 groundY = 0;  // carried for CG2-CG6 (the ground clamp)
    fx                                 hSearch = 0;  // CG1: the grain grid cell-size (== the GR2 search radius)
};

// ----- The body->grain reject (the PURE INT32 per-axis |dx| < radius box test) -----------------------------
// BodyGrainAccept(b, g): accept grain g as a candidate of body b iff |b.pos.axis - g.pos.axis| < b.radius on
// EVERY axis (a box, NOT a sphere — the over-inclusive candidate set CG2's support/CG3's displacement culls).
// PURE INT32: an integer subtract + abs + compare per axis, NO products, NO int64, NO sqrt. The shader copies
// THIS verbatim. (== couple.h::BodyParticleAccept with grain::GrainParticle instead of fluid::FluidParticle.)
inline bool BodyGrainAccept(const fpx::FxBody& b, const grain::GrainParticle& g) {
    fx dx = b.pos.x - g.pos.x; if (dx < 0) dx = -dx;
    fx dy = b.pos.y - g.pos.y; if (dy < 0) dy = -dy;
    fx dz = b.pos.z - g.pos.z; if (dz < 0) dz = -dz;
    return dx < b.radius && dy < b.radius && dz < b.radius;
}

// ----- The CSR body->grain query result --------------------------------------------------------------------
// bodyStart[b..] is the exclusive prefix-sum of per-body gathered counts (bodyStart has bodyCount+1 entries;
// bodyStart[b]..bodyStart[b+1] is body b's slice), and bodyGrains[] holds the gathered grain indices grouped
// by body, ASCENDING grain index within each body (deterministic). The GPU cgrain_body_{count,scan,emit}
// mirror this byte-for-byte.
struct CGrainQuery {
    std::vector<uint32_t> bodyStart;    // bodyCount+1 exclusive prefix-sum offsets (CSR row pointers)
    std::vector<uint32_t> bodyGrains;   // gathered grain indices grouped by body (ascending)
};

// ----- CountBodyGrains: the per-body count over the body's AABB cell range (count) -------------------------
// For each body i, iterate the cell range [GrainCellOf(body.pos - radius) .. GrainCellOf(body.pos + radius)]
// (the fpx::BodyAabb quantised to grain cells at cell-size grid.hSearch); for each grain in those cells, count
// it iff BodyGrainAccept(body, g). perBodyOut[i] = #gathered; returns the total. The GPU cgrain_body_count
// mirrors THIS per-thread (one thread per body i). The cell range is the ONE delta vs GR2's fixed 27-cell
// stencil — a body spans MANY cells. Cells outside the grid are skipped (clamp). Pure int32. (Iterating the
// AABB corners in cell space is correct because BodyGrainAccept's box is exactly [pos-radius, pos+radius] per
// axis, which the cell range [GrainCellOf(pos-radius), GrainCellOf(pos+radius)] fully covers — every accepted
// grain lies in one of those cells.)
inline uint32_t CountBodyGrains(const CGrainWorld& world, const grain::GrainGrid& grid,
                                const grain::GrainCellTable& table, std::vector<uint32_t>& perBodyOut) {
    const uint32_t bodyCount = (uint32_t)world.bodies.size();
    perBodyOut.assign((size_t)bodyCount, 0u);
    uint32_t total = 0;
    for (uint32_t i = 0; i < bodyCount; ++i) {
        const fpx::FxBody& b = world.bodies[(size_t)i];
        const fpx::FxAabb aabb = fpx::BodyAabb(b);
        const FxCell loCell = grain::GrainCellOf(aabb.lo, grid.hSearch);
        const FxCell hiCell = grain::GrainCellOf(aabb.hi, grid.hSearch);
        uint32_t c = 0u;
        for (int cz = loCell.z; cz <= hiCell.z; ++cz)
        for (int cy = loCell.y; cy <= hiCell.y; ++cy)
        for (int cx = loCell.x; cx <= hiCell.x; ++cx) {
            // Skip cells outside the bounded grid (clamp — the body may overhang the grain AABB).
            if (cx < grid.cellMin.x || cx >= grid.cellMin.x + grid.gridDim.x) continue;
            if (cy < grid.cellMin.y || cy >= grid.cellMin.y + grid.gridDim.y) continue;
            if (cz < grid.cellMin.z || cz >= grid.cellMin.z + grid.gridDim.z) continue;
            const uint32_t cell = grain::FlatGrainCellId(FxCell{cx, cy, cz}, grid);
            for (uint32_t s = table.cellStart[cell]; s < table.cellStart[cell + 1u]; ++s) {
                const uint32_t j = table.cellGrains[s];
                if (BodyGrainAccept(b, world.grains[(size_t)j])) ++c;
            }
        }
        perBodyOut[(size_t)i] = c;
        total += c;
    }
    return total;
}

// ----- GatherBodyGrains: the full body->grain query (count->scan->emit) ------------------------------------
// (1) Build the grain GrainGrid + GrainCellTable (reuse GR2 MakeGrainGrid/BuildGrainCellTable at hSearch). (2)
// CountBodyGrains -> per-body counts. (3) exclusive prefix-sum -> bodyStart. (4) EMIT each accepted grain index
// into body i's disjoint slice in the FIXED order: ascending cell (cz,cy,cx) over the body's AABB range, then
// ascending grain index within a cell (cellGrains is already ascending-index per cell) -> fully deterministic.
// Each body writes into its OWN DISJOINT [bodyStart[i], bodyStart[i+1]) range -> the GPU emit is race-free, NO
// atomics (the per-body-disjoint pattern). The GPU does the SAME three passes -> the GPU bodyGrains + bodyStart
// memcmp against this byte-for-byte. (DET-CRUX, the CP1/GR2 lesson: the per-body EMIT scatter is fixed
// ascending order; the count + the per-body lists are per-body-disjoint -> race-free. The reused grain
// cell-EMIT is single-thread ascending, already correct in GR2.) Pure int32.
inline CGrainQuery GatherBodyGrains(const CGrainWorld& world) {
    const uint32_t bodyCount = (uint32_t)world.bodies.size();
    CGrainQuery q;

    // (1) the GR2 grain grid + cell table over the grain pool (cell-size = hSearch, the GR2 search radius).
    // Empty pool -> a 1x1x1 grid (deterministic degenerate); every body then gathers 0.
    const grain::GrainGrid grid = grain::MakeGrainGrid(world.grains, world.hSearch);
    const grain::GrainCellTable table = grain::BuildGrainCellTable(world.grains, grid);

    // (2) COUNT: per-body gathered count over the body's AABB cell range.
    std::vector<uint32_t> counts;
    const uint32_t total = CountBodyGrains(world, grid, table, counts);

    // (3) SCAN: exclusive prefix-sum -> bodyStart (bodyCount+1 entries; the last == total).
    q.bodyStart.assign((size_t)bodyCount + 1u, 0u);
    uint32_t running = 0;
    for (uint32_t i = 0; i < bodyCount; ++i) {
        q.bodyStart[(size_t)i] = running;
        running += counts[(size_t)i];
    }
    q.bodyStart[(size_t)bodyCount] = running;   // sentinel == total

    // (4) EMIT: each body writes its gathered grain indices into its disjoint slice (ascending cell, then
    // ascending grain index within a cell -> the GR2/CP1 deterministic emit order).
    q.bodyGrains.assign((size_t)total, 0u);
    for (uint32_t i = 0; i < bodyCount; ++i) {
        const fpx::FxBody& b = world.bodies[(size_t)i];
        const fpx::FxAabb aabb = fpx::BodyAabb(b);
        const FxCell loCell = grain::GrainCellOf(aabb.lo, grid.hSearch);
        const FxCell hiCell = grain::GrainCellOf(aabb.hi, grid.hSearch);
        uint32_t base  = q.bodyStart[(size_t)i];
        uint32_t local = 0u;
        for (int cz = loCell.z; cz <= hiCell.z; ++cz)
        for (int cy = loCell.y; cy <= hiCell.y; ++cy)
        for (int cx = loCell.x; cx <= hiCell.x; ++cx) {
            if (cx < grid.cellMin.x || cx >= grid.cellMin.x + grid.gridDim.x) continue;
            if (cy < grid.cellMin.y || cy >= grid.cellMin.y + grid.gridDim.y) continue;
            if (cz < grid.cellMin.z || cz >= grid.cellMin.z + grid.gridDim.z) continue;
            const uint32_t cell = grain::FlatGrainCellId(FxCell{cx, cy, cz}, grid);
            for (uint32_t s = table.cellStart[cell]; s < table.cellStart[cell + 1u]; ++s) {
                const uint32_t j = table.cellGrains[s];
                if (BodyGrainAccept(b, world.grains[(size_t)j])) {
                    q.bodyGrains[(size_t)(base + local)] = j;
                    ++local;
                }
            }
        }
    }
    return q;
}

// CountGathered(q): the total gathered body-grain pairs (== bodyGrains.size() == bodyStart.back()) — a
// reporting/stat helper for the showcase coverage proof. Pure integer.
inline uint32_t CountGathered(const CGrainQuery& q) {
    return q.bodyStart.empty() ? 0u : q.bodyStart.back();
}

// MaxPerBody(q): the largest per-body gathered count over the CSR — a reporting/stat helper. Pure integer.
inline uint32_t MaxPerBody(const CGrainQuery& q) {
    uint32_t m = 0;
    for (size_t i = 0; i + 1 < q.bodyStart.size(); ++i) {
        const uint32_t c = q.bodyStart[i + 1] - q.bodyStart[i];
        if (c > m) m = c;
    }
    return m;
}

// ===== Slice CG2 — CONTACT SUPPORT + DRAG (grain->body, the CRUX) ====================================
// THE FIRST MOMENTUM EXCHANGE of FLAGSHIP #12: each fpx::FxBody sums, over its CG1 gathered grain list (in
// the FIXED CG1 emit order, ascending), a CONTACT-SUPPORT impulse (the grain bed pushes the body OUT/up
// along each overlapping grain's contact normal, ∝ the penetration) + a DRAG impulse (toward the local
// grain velocity, which CG2 damps since the grains are held STATIC), and the body decelerates and RESTS on
// the bed at an emergent rest line. ONE-WAY for now (grain -> body; the body->grain reaction is CG3).
// LINEAR only — a sphere body's contact support acts through its centre (no torque), so angVel is untouched
// until a future asymmetric-bed refinement. The CP2 buoyancy/drag twin, with contact-support physics (Σ
// penetration·normal over overlapping grains) instead of buoyancy (∝ the gathered count).
//
// THE int64 DECISION (the CP2/GR3/FL4 split): the support/drag math is int64 (FxLength/FxNormalize/fxmul/
// fxdiv) -> shaders/cgrain_support.comp is VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) + the Metal
// --cgrain-support showcase runs the CPU StepCGrainSupport (byte-identical by construction). The CG1
// grid-hash query (re-run each step) stays int32 MSL-native.
//
// THE CRUX — the order-sensitive body-force reduction: summing N grain impulses onto one body is a
// reduction; the integer zero-diff bar demands a PINNED summation order. CG2 sums each body's contributions
// in the FIXED CG1 gathered order (bodyGrains[bodyStart[i] .. bodyStart[i+1]), ascending). The GPU dispatch
// is ONE THREAD PER BODY (a serial inner loop over that body's gathered list) -> multi-thread OVER bodies,
// NOT over grains. With a TINY body count (1-few, << the ~2s watchdog), no body's inner loop approaches the
// single-thread TDR ceiling. CAVEAT (the new wrinkle vs CP2): a body resting ON a sand bed gathers MORE
// grains than a body floating in fluid (a bed supports through many simultaneous contacts), so the per-body
// inner loop is LONGER than CP2's case — still bounded and far under the ~2s watchdog for a tiny body count,
// but flagged. IF body counts ever scale up, a deterministic integer atomic-add reduction is needed —
// explicitly OUT of scope for CG1-CG6 (the swraster 64-bit-atomics caveat).

using fpx::FxAdd;          // read-only: the fpx.h vector toolbox (reused verbatim, no new primitives)
using fpx::FxSub;
using fpx::FxScale;
using fpx::FxLength;       // read-only: the int64 length (FxISqrt of the sum of squares)
using fpx::FxNormalize;    // read-only: the int64 normalize (+Y fallback on length 0)
using fpx::fxmul;          // read-only: the int64-intermediate Q16.16 multiply

// The host-snapped Q16.16 coupling coefficients. Tuned so a unit-mass FxBody (invMass=kOne, radius 2)
// dropped above the CG1 grain bed settles at an emergent REST LINE on the bed (the net upward contact
// support balances gravity) — NOT crashing through, NOT bouncing out. SUPPORT: F_support per overlapping
// grain ∝ its penetration depth, summed over the gathered contacts, so a body pressed DEEPER into the bed
// overlaps MORE grains by MORE penetration and feels a larger restoring push -> a stable equilibrium rest
// line (the contact-support proxy: the bed pushes harder the more the body sinks). DRAG: a linear damping
// toward the (static) grains' mean velocity, so the body settles instead of bouncing. The rest line is
// EMERGENT/iterative (the GR4/CP2 caveat), NOT an exact sink depth.
//   kSupport = 6.0 (Q16.16): a firm per-penetration contact stiffness. With a settled bed of ~0.25-radius
//     grains a body overlaps MANY grains at small penetrations; the summed Σ pen·n builds the upward force
//     that balances |g|=9.8 a short distance into the bed surface, NOT crashing through. (host-snapped.)
//   kDrag = 3.0 (Q16.16): a firm linear damping so the body settles within K steps instead of ringing.
inline constexpr fx kSupport = (fx)(12.0 * (double)kOne + 0.5);  // 786432 (Q16.16, ~12 per-pen stiffness)
inline constexpr fx kDrag    = (fx)(6.0  * (double)kOne + 0.5);  // 393216 (Q16.16, ~6.0 damping)

// AccumBodyGrainForces(world, query, dt): the per-body contact-support+drag impulse accumulate (the
// FIXED-ORDER reduction). For each DYNAMIC body i (static / non-dynamic skipped), over its CG1 gathered list
// in ASCENDING order: SUPPORT — for each overlapping grain (d = body.pos − grain.pos, dist = FxLength(d),
// pen = (body.radius + grain.radius) − dist > 0): F_support += FxScale(FxNormalize(d), fxmul(kSupport, pen))
// (the bed pushes the body AWAY from the grain along the contact normal, ∝ penetration, summed). DRAG —
// vGrainAvg = (Σ grain.vel)/count per axis (the fixed-order int64 sum / the integer count); F_drag =
// fxmul(kDrag, vGrainAvg − body.vel) per axis. Apply body.vel += FxScale(F_support + F_drag, body.invMass)·dt.
// A body that gathers 0 grains feels no support/drag (free-fall). LINEAR only (angVel untouched). int64.
// cgrain_support.comp copies THIS body VERBATIM (one thread per body i). Deterministic (fixed gathered
// order, fixed op order) -> bit-identical to the GPU memcmp.
inline void AccumBodyGrainForces(CGrainWorld& world, const CGrainQuery& query, fx dt) {
    const uint32_t bodyCount = (uint32_t)world.bodies.size();
    for (uint32_t i = 0; i < bodyCount; ++i) {
        fpx::FxBody& b = world.bodies[(size_t)i];
        if (!(b.flags & fpx::kFlagDynamic)) continue;   // static/kinematic -> no support/drag (the pinned case)
        const uint32_t s0 = query.bodyStart[(size_t)i];
        const uint32_t s1 = query.bodyStart[(size_t)i + 1u];
        const uint32_t count = s1 - s0;
        if (count == 0u) continue;                      // gathers nothing (clear of the bed) -> free-fall

        // SUPPORT: Σ over the gathered grains of the contact push (∝ penetration along the contact normal),
        // summed in the FIXED CG1 gathered (ascending) order. A grain only contributes when it OVERLAPS the
        // body (pen > 0) — the exact radial cull deferred from CG1's over-inclusive box happens HERE.
        FxVec3 fSupport{0, 0, 0};
        // DRAG: vGrainAvg = (Σ_j grain[j].vel) / count, the fixed-order int64 sum / the integer count.
        int64_t sumX = 0, sumY = 0, sumZ = 0;
        for (uint32_t s = s0; s < s1; ++s) {
            const grain::GrainParticle& g = world.grains[(size_t)query.bodyGrains[(size_t)s]];
            const FxVec3 d = FxSub(b.pos, g.pos);       // body relative to the grain (push body AWAY from grain)
            const fx dist = FxLength(d);
            const fx pen = (b.radius + g.radius) - dist;
            if (pen > 0) {
                const FxVec3 n = FxNormalize(d);        // contact normal (dist==0 -> the +Y fallback)
                fSupport = FxAdd(fSupport, FxScale(n, fpx::fxmul(kSupport, pen)));
            }
            sumX += (int64_t)g.vel.x;
            sumY += (int64_t)g.vel.y;
            sumZ += (int64_t)g.vel.z;
        }
        const FxVec3 vGrainAvg{
            (fx)(sumX / (int64_t)count), (fx)(sumY / (int64_t)count), (fx)(sumZ / (int64_t)count)};
        // F_drag = kDrag * (vGrainAvg - body.vel) per axis.
        const FxVec3 dv = FxSub(vGrainAvg, b.vel);
        const FxVec3 fDrag{fpx::fxmul(kDrag, dv.x), fpx::fxmul(kDrag, dv.y), fpx::fxmul(kDrag, dv.z)};

        // Apply the impulse as a velocity delta: vel += (F_support + F_drag) * invMass * dt (linear only).
        const FxVec3 fTotal = FxAdd(fSupport, fDrag);
        const FxVec3 dvel = FxScale(FxScale(fTotal, b.invMass), dt);
        b.vel = FxAdd(b.vel, dvel);
    }
}

// StepCGrainSupport(world, dt): ONE coupled step (the driver — the grains are held STATIC). (1) re-query the
// body->grain neighbour lists from the bodies' CURRENT positions (CG1 GatherBodyGrains) -> (2) accumulate
// the contact-support+drag velocity delta (AccumBodyGrainForces) -> (3) IntegrateBody per body (gravity +
// the support-adjusted vel — the fpx.h integrate VERBATIM) THEN ResolveGround (the fpx.h RADIUS-AWARE floor:
// the body's SURFACE rests on the floor at groundY + radius — the bed-floor clamp, VERBATIM). The grains are
// NOT moved (the reaction is CG3). Over K steps a body dropped above the bed falls, contacts the bed, support
// builds as it overlaps grains, and it settles to an emergent rest line damped by drag (or sinks to the bed
// floor groundY + radius if it never gathers grains — the support=0 control). Both IntegrateBody +
// ResolveGround are copied VERBATIM by cgrain_support.comp's host driver (the GPU runs the SAME per-step seq).
inline void StepCGrainSupport(CGrainWorld& world, fx dt) {
    const CGrainQuery query = GatherBodyGrains(world);   // CG1 re-query (grains static -> the bed unchanged)
    AccumBodyGrainForces(world, query, dt);              // CG2 contact support + drag velocity delta
    for (fpx::FxBody& b : world.bodies) {                // fpx.h integrate + radius-aware floor, VERBATIM
        fpx::IntegrateBody(b, world.gravity, world.groundY, dt);
        fpx::ResolveGround(b, world.groundY);            // the body SURFACE rests at groundY + radius
    }
}

// StepCGrainSupportSteps(world, dt, steps): run K StepCGrainSupport steps. The CPU reference the GPU body
// state memcmp's against byte-for-byte. Pure integer -> two runs byte-identical, cross-backend identical.
inline void StepCGrainSupportSteps(CGrainWorld& world, fx dt, int steps) {
    for (int s = 0; s < steps; ++s) StepCGrainSupport(world, dt);
}

// MeasureRestLine(world): the honest rest-line metric — the mean settled body pos.y over the DYNAMIC bodies
// (a deterministic Q16.16 stat for the rest-line proof). 0 dynamic bodies -> 0. The rest line is
// EMERGENT/iterative (the GR4/CP2 caveat), NOT an exact sink depth: the proof asserts the body rests by a
// margin above the bed floor (groundY + radius) + is bounded above + is deterministic + two-run byte-identical.
inline fx MeasureRestLine(const CGrainWorld& world) {
    int64_t sum = 0; uint32_t n = 0;
    for (const fpx::FxBody& b : world.bodies)
        if (b.flags & fpx::kFlagDynamic) { sum += (int64_t)b.pos.y; ++n; }
    return n == 0u ? 0 : (fx)(sum / (int64_t)n);
}

// ===== Slice CG3 — GRAIN REACTION / DISPLACEMENT (body->grain, Newton's 3rd law to CG2) ==============
// THE SECOND HALF of FLAGSHIP #12's two-way exchange: the body now pushes BACK on the sand. Each grain
// INSIDE a body is projected out to the body surface (the body PARTS the sand — a cavity) AND receives the
// equal-opposite DRAG-REACTION impulse (the body imparts its momentum to the surrounding grains). Completes
// the two-way exchange (CG2 grain->body, CG3 body->grain). The CP3 ApplyBodyToFluid twin, with
// grain::GrainParticle, and the positional push is LITERALLY grain.h::CollideGrainSphere(g,
// GrainSphereFromBody(b)) — the grain-out-of-FxBody bridge already exists.
//
// THE MIRROR OF CG2 (per-GRAIN over the body set): CG2's body-force reduction was per-BODY (over its
// gathered grains). CG3's grain displacement is the mirror — ONE thread per GRAIN, each grain iterates the
// tiny body list (fixed order), and for each body that CONTAINS it accumulates the surface-snap push into a
// SEPARATE dp[] (JACOBI) + applies the drag-reaction velocity impulse. Each grain writes ONLY its own
// dp/vel -> per-grain-disjoint, race-free, NO atomics, [numthreads(64,1,1)] MULTI-THREAD, NO TDR (the
// GR3/CP3 win; the EXACT shape of grain.h::CollideGrainSpheres, body as the sphere). int64
// (FxLength/FxNormalize/fxmul) -> cgrain_displace.comp is VULKAN-SPIR-V-ONLY + the Metal --cgrain-displace
// showcase runs THIS CPU ApplyBodyToGrains (byte-identical by construction, the CG2/GR3/CP3 split). The CG1
// query passes stay int32 MSL-native.

// kDragReaction: the body->grain drag-reaction coefficient (the CG2 kDrag partner — the equal-opposite of
// the body's CG2 drag, now imparting the body's momentum to the grains). Host-snapped Q16.16 (~1.5, the CP3
// kDragReaction value). The showcase + the CPU reference + the GPU shader share THIS exact constant.
inline constexpr fx kDragReaction = (fx)(1.5 * (double)kOne + 0.5);   // 98304 (Q16.16, ~1.5 reaction drag)

// ApplyBodyToGrains(world, dt): the per-grain projection-out-of-body + drag reaction (the GR3
// CollideGrainSphere mold over the body set + a Jacobi dp[]). For each grain g (skip STATIC), over each
// DYNAMIC body b (fixed order), with d = g.pos − b.pos, dist = FxLength(d), surf = b.radius + g.radius: if
// dist < surf (the grain is INSIDE the body) accumulate the surface-snap push into a SEPARATE dp[] buffer
//   dp_g += FxAdd(b.pos, FxScale(FxNormalize(d), surf)) − g.pos    // snap to the body surface (sphereR+grainR)
// (dist==0 -> the FxNormalize +Y fallback — EXACTLY grain.h::CollideGrainSphere(g, GrainSphereFromBody(b)))
// AND apply the drag-reaction velocity impulse per axis
//   g.vel += fxmul(kDragReaction, (b.vel − g.vel)) · dt           // toward the body velocity
// then apply g.pos += dp_g for ALL grains after (JACOBI — each grain reads the iteration-start body state,
// NOT the in-progress positions). STATIC grains (kFlagStatic / boundary) -> dp 0, vel untouched. Non-dynamic
// bodies are SKIPPED (they hold; CG3 is the dynamic-body reaction). int64 (FxLength/FxNormalize/fxmul).
// cgrain_displace.comp copies THIS body VERBATIM (one thread per grain). Deterministic (the fixed body
// order, fixed op order) -> bit-identical to the GPU memcmp + two-run byte-identical.
inline void ApplyBodyToGrains(CGrainWorld& world, fx dt) {
    const size_t n = world.grains.size();
    const size_t bodyCount = world.bodies.size();
    std::vector<FxVec3> dp((size_t)n, FxVec3{0, 0, 0});   // the Jacobi double-buffer (per-grain Δp)
    for (size_t i = 0; i < n; ++i) {
        grain::GrainParticle& g = world.grains[i];
        if (g.flags & grain::kFlagStatic) continue;       // boundary grain -> dp 0, vel untouched
        FxVec3 accum{0, 0, 0};
        for (size_t bi = 0; bi < bodyCount; ++bi) {
            const fpx::FxBody& b = world.bodies[bi];
            if (!(b.flags & fpx::kFlagDynamic)) continue; // non-dynamic body -> holds (the pinned case)
            const FxVec3 d = FxSub(g.pos, b.pos);         // grain relative to the body centre (outward)
            const fx dist = FxLength(d);
            const fx surf = b.radius + g.radius;          // the surfaces-touch distance (sphereR + grainR)
            if (dist >= surf) continue;                   // outside the (expanded) body sphere -> no push
            // (1) POSITIONAL DISPLACEMENT: snap the grain centre to the body surface (the body parts the sand).
            const FxVec3 nrm = FxNormalize(d);            // outward normal (dist==0 -> {0,kOne,0} fallback)
            const FxVec3 surfPt = FxAdd(b.pos, FxScale(nrm, surf));   // the surface point along the normal
            accum = FxAdd(accum, FxSub(surfPt, g.pos));   // into the Jacobi dp[] (the CollideGrainSphere push)
            // (2) DRAG REACTION: the body imparts momentum to the grain (the equal-opposite of CG2's drag).
            const FxVec3 dv = FxSub(b.vel, g.vel);        // toward the body velocity
            g.vel.x += fpx::fxmul(fpx::fxmul(kDragReaction, dv.x), dt);
            g.vel.y += fpx::fxmul(fpx::fxmul(kDragReaction, dv.y), dt);
            g.vel.z += fpx::fxmul(fpx::fxmul(kDragReaction, dv.z), dt);
        }
        dp[i] = accum;
    }
    // Apply pos += dp for all grains (Jacobi — disjoint per-grain writes, race-free).
    for (size_t i = 0; i < n; ++i) {
        if (world.grains[i].flags & grain::kFlagStatic) continue;
        world.grains[i].pos = FxAdd(world.grains[i].pos, dp[i]);
    }
}

// MeasureGrainBodyPenetration(world): the honest no-penetration metric (the FL4/GR3/CP3 caveat) — over every
// grain / dynamic body pair, sum/max the grain-into-body penetration pen = (b.radius + g.radius) −
// |g.pos − b.pos| > 0. Returns {peak, summed} in Q16.16 (int64 accumulator). DETERMINISTIC + bit-exact. The
// showcase's "sand parted" proof compares this BEFORE vs AFTER ApplyBodyToGrains (penAfter < penBefore — the
// FL4/GR3 honesty: relieved, NOT zero; Jacobi single-projection so a grain inside MULTIPLE bodies leaves a
// deterministic-but-nonzero residual). Static grains are counted (they CAN sit inside a body; CG3 does not
// move them, so their penetration is part of the honest residual). Non-dynamic bodies skipped.
struct GrainBodyPenetration { int64_t peak = 0; int64_t summed = 0; };
inline GrainBodyPenetration MeasureGrainBodyPenetration(const CGrainWorld& world) {
    GrainBodyPenetration out;
    for (const grain::GrainParticle& g : world.grains)
        for (const fpx::FxBody& b : world.bodies) {
            if (!(b.flags & fpx::kFlagDynamic)) continue;
            const fx pen = (b.radius + g.radius) - FxLength(FxSub(g.pos, b.pos));
            if (pen > 0) { out.summed += (int64_t)pen; if ((int64_t)pen > out.peak) out.peak = (int64_t)pen; }
        }
    return out;
}

// CountDisplacedGrains(world): the deterministic count of grains INSIDE at least one dynamic body (dist <
// b.radius + g.radius) — the "displaced > 0" coverage stat for the showcase proof. Static grains ARE counted
// (they sit inside the body too). Pure int64 compare -> bit-exact CPU<->GPU.
inline uint32_t CountDisplacedGrains(const CGrainWorld& world) {
    uint32_t c = 0;
    for (const grain::GrainParticle& g : world.grains) {
        bool inside = false;
        for (const fpx::FxBody& b : world.bodies) {
            if (!(b.flags & fpx::kFlagDynamic)) continue;
            if (FxLength(FxSub(g.pos, b.pos)) < (b.radius + g.radius)) { inside = true; break; }
        }
        if (inside) ++c;
    }
    return c;
}

// ===== Slice CG4 — THE COUPLED STEP (the body sinking into sand, the INTEGRATED two-way solver) ==========
// The FOURTH slice of FLAGSHIP #12: ONE deterministic tick that runs the grain's own frictional pile dynamics
// (GR3 non-penetration + GR4 Coulomb friction) AND both exchange directions (CG2 grain->body support, CG3
// body->grain displacement) AND the rigid integrate — a dynamic sand bed AND a dynamic body in one
// bidirectional loop. The result: a body SINKS into a dynamic, self-piling sand bed under gravity, the sand
// piles around it (holding its angle of repose) and SUPPORTS it, and it settles half-buried — no script. The
// CP4 twin with the GRAIN sim. CG4 ORCHESTRATES the existing bit-exact pieces — it composes the GR3/GR4
// SUB-passes (NOT StepGrainFriction wholesale, which would re-predict + re-build the neighbour list and skip
// the coupling) interleaved with CG2 AccumBodyGrainForces + CG3 ApplyBodyToGrains + the rigid IntegrateBody/
// ResolveGround, in the LOCKED order below. NO new shader, NO new RHI: the GPU showcase is a host-driven
// multi-pass driver over the EXISTING GR3 grain_contact_* + GR4 grain_friction + CG2 cgrain_support + CG3
// cgrain_displace shaders.
//
// THE COUPLED TICK (StepCGrain, the locked make-or-break order):
//   (1) PREDICT:  grains IntegrateGrains(grains) [GR1: prev=pos, predict pos; radius-aware ground rest]; each
//                 DYNAMIC body vel += gravity·dt  (velocity ONLY — the position integrates at (5)).
//   (2) BUILD:    grid = MakeGrainGrid; table = BuildGrainCellTable; nbr = BuildGrainNeighborList (GR2) — once
//                 per step, FIXED across the K iters (the standard PBF choice); query = GatherBodyGrains (CG1),
//                 also built ONCE per step (from the predicted grains + the current body pos).
//   (3) ITERATE (K JACOBI iters), EACH:
//         (3a) GR3 normal:   SolveGrainContact -> apply pos+=Δp   (grain-grain non-penetration).
//         (3b) GR4 friction: SolveGrainFriction -> apply pos+=Δp  (the angle-of-repose tangential clamp).
//         (3c) CG3 body->grain: ApplyBodyToGrains (grains pushed out of the bodies + the drag reaction).
//         (3d) CG2 grain->body: AccumBodyGrainForces (support + drag -> body.vel delta, over the CG1 query).
//   (4) GRAIN VELOCITY: vel = (pos − prev)/dt   (the GR-step grain velocity update).
//   (5) BODY INTEGRATE: each DYNAMIC body pos += vel·dt; ResolveGround(body, groundY)   (the bed/floor clamp).
//                 grains: CollideGrainPlane(grains, groundY)   (the grains rest on the ground floor).
// The body accumulates gravity (1) + support/drag over the K iters (3d), THEN its position integrates (5); the
// sand self-piles (3a/3b) and parts around the body (3c). Over many steps the body sinks in, the sand supports
// it, and it settles half-buried — emergent, no script. The neighbour list + the CG1 query are built ONCE per
// step (the PBF choice). Pure integer, fixed op order -> two runs bit-identical AND bit-exact GPU==CPU. Bodies
// do not collide with each other (single body / non-overlapping; body-body contacts are out of scope).
//
// THE CP4 COMPENSATION (the invMass = kOne/iters trick): CG2 AccumBodyGrainForces runs in EACH of the K iters
// (3d), so its impulse is applied K× per step; balance it against once-per-step gravity with the body
// invMass = kOne/iters (mathematically exact: K·F·(1/K) = F) — the same clean compensation CP4 used. The
// CALLER sets the body invMass (the showcase / test sets it to kOne/iters); StepCGrain itself does NOT touch
// invMass — it simply applies CG2's force scaled by the body's own invMass each iter (the AccumBodyGrainForces
// contract), so a body built with invMass=kOne/iters gets the exactly-balanced K-fold accumulation.
//
// The friction coefficient μ is grain::kGrainMu (the GR4 default — the angle-of-repose constant); the grains
// hold their repose around the sinking body exactly as the GR4 showcase pile does.

inline void StepCGrain(CGrainWorld& world, fx dt, int iters) {
    const size_t n = world.grains.size();
    // (1) PREDICT: the grain integrate (GR1 — vel += g·dt; prev = pos; pos += vel·dt; radius-aware ground rest)
    // AND each dynamic body's gravity velocity integrate (velocity ONLY; the position integrates at (5)).
    grain::IntegrateGrains(world.grains, world.gravity, dt, world.groundY);
    for (fpx::FxBody& b : world.bodies) {
        if (!(b.flags & fpx::kFlagDynamic)) continue;
        b.vel.x += fpx::fxmul(world.gravity.x, dt);
        b.vel.y += fpx::fxmul(world.gravity.y, dt);
        b.vel.z += fpx::fxmul(world.gravity.z, dt);
    }
    // (2) BUILD: the GR2 neighbour list from the PREDICTED positions (built ONCE, fixed across the K iters) +
    // the CG1 body->grain query (built ONCE per step, from the predicted grains + the current body pos).
    const grain::GrainGrid grid = grain::MakeGrainGrid(world.grains, world.hSearch);
    const grain::GrainCellTable table = grain::BuildGrainCellTable(world.grains, grid);
    const grain::GrainNeighborList list =
        grain::BuildGrainNeighborList(world.grains, grid, table, world.hSearch);
    const CGrainQuery query = GatherBodyGrains(world);
    // (3) K JACOBI iterations: GR3 normal -> GR4 friction -> CG3 body->grain -> CG2 grain->body, each iteration.
    std::vector<FxVec3> dp;
    for (int it = 0; it < iters; ++it) {
        // (3a) GR3 NORMAL push (the grain-grain non-penetration; Δp_i into a SEPARATE dp buffer -> apply).
        grain::SolveGrainContact(world.grains, list, dp);
        for (size_t i = 0; i < n; ++i) {
            if (world.grains[i].flags & grain::kFlagStatic) continue;
            world.grains[i].pos = FxAdd(world.grains[i].pos, dp[i]);
        }
        // (3b) GR4 TANGENTIAL friction (reads the POST-normal positions; the angle-of-repose clamp).
        grain::SolveGrainFriction(world.grains, list, grain::kGrainMu, dp);
        for (size_t i = 0; i < n; ++i) {
            if (world.grains[i].flags & grain::kFlagStatic) continue;
            world.grains[i].pos = FxAdd(world.grains[i].pos, dp[i]);
        }
        // (3c) CG3 body->grain: displace the grains out of the bodies (+ the drag reaction).
        ApplyBodyToGrains(world, dt);
        // (3d) CG2 grain->body: accumulate support + drag into the body velocities (over the CG1 query).
        AccumBodyGrainForces(world, query, dt);
    }
    // (4) GRAIN VELOCITY: vel = (pos − prev)/dt (the GR-step PBF grain velocity update).
    if (dt != 0) {
        for (size_t i = 0; i < n; ++i) {
            if (world.grains[i].flags & grain::kFlagStatic) continue;
            const FxVec3 dpos = FxSub(world.grains[i].pos, world.grains[i].prev);
            world.grains[i].vel =
                FxVec3{fpx::fxdiv(dpos.x, dt), fpx::fxdiv(dpos.y, dt), fpx::fxdiv(dpos.z, dt)};
        }
    }
    // (5) BODY INTEGRATE: pos += vel·dt for each dynamic body THEN ResolveGround (the radius-aware bed clamp);
    // the grains rest on the ground floor (CollideGrainPlane — pos.y >= groundY + radius).
    for (fpx::FxBody& b : world.bodies) {
        if (!(b.flags & fpx::kFlagDynamic)) continue;
        b.pos.x += fpx::fxmul(b.vel.x, dt);
        b.pos.y += fpx::fxmul(b.vel.y, dt);
        b.pos.z += fpx::fxmul(b.vel.z, dt);
        fpx::ResolveGround(b, world.groundY);
    }
    grain::CollideGrainPlane(world.grains, world.groundY);
}

// StepCGrainSteps(world, dt, iters, steps): run K coupled ticks. The CPU reference the GPU multi-pass driver
// memcmp's against byte-for-byte (the grain + body state). Pure integer -> two runs byte-identical,
// cross-backend identical (the GPU runs the GR3/GR4/CG2/CG3 int64 passes Vulkan-only; Metal runs THIS).
inline void StepCGrainSteps(CGrainWorld& world, fx dt, int iters, int steps) {
    for (int s = 0; s < steps; ++s) StepCGrain(world, dt, iters);
}

// MeasureCGrainState(world): the honest emergent metrics helper (the CP4 MeasureCoupleState twin with the
// grain sim). Returns the mean DYNAMIC-body rest line (pos.y), the GR4 grain repose slope (the bed-coherence
// stat — the pile still holds an angle of repose around the body), the body's sink (the drop from its start),
// and the dynamic body count. Deterministic Q16.16 stats for the proofs (the GR4/CP2 caveat shape — the rest
// line + sink are EMERGENT/within-band, NOT exact depths). The repose is computed from the CURRENT grains via
// grain::MeasureGrainRepose, bit-exact CPU<->GPU. `startY` is the body's initial pos.y (the caller passes the
// scene's drop height) so sink = startY − restY is the body's net descent.
struct CGrainState {
    fx       restY = 0;          // the mean settled DYNAMIC-body pos.y (the emergent rest line)
    fx       repose = 0;         // the GR4 grain repose slope (the bed-coherence stat)
    fx       sink = 0;           // the body's drop from its start (startY − restY)
    uint32_t dynamicBodies = 0;  // the count of dynamic bodies (restY is their mean)
};
inline CGrainState MeasureCGrainState(const CGrainWorld& world, fx startY) {
    CGrainState s;
    int64_t sum = 0;
    for (const fpx::FxBody& b : world.bodies)
        if (b.flags & fpx::kFlagDynamic) { sum += (int64_t)b.pos.y; ++s.dynamicBodies; }
    s.restY = s.dynamicBodies == 0u ? 0 : (fx)(sum / (int64_t)s.dynamicBodies);
    s.sink  = startY - s.restY;
    // The grain repose slope over the current bed (the GR4 coherence metric). Empty pool -> 0.
    if (!world.grains.empty())
        s.repose = grain::MeasureGrainRepose(world.grains, world.groundY).slope;
    return s;
}

}  // namespace cgrain
}  // namespace hf::sim
