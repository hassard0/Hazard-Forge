#pragma once
// Slice CP1 — Deterministic Rigid<->Fluid Coupling: the UNIFIED COUPLED WORLD + the BODY->FLUID grid-hash
// QUERY (the BEACHHEAD of FLAGSHIP #11: DETERMINISTIC TWO-WAY RIGID<->FLUID COUPLING). The natural 5th act
// of the deterministic-sim arc: not another isolated body, but making the EXISTING bodies INTERACT. The four
// sim members (rigid fpx, cloth, fluid, grain) each live in their own world and only touch STATIC kinematic
// colliders — no two simulated bodies ever exchange momentum. This flagship couples a dynamic fpx::FxBody to
// the bit-exact PBF fluid (buoyancy + drag + displacement, one Q16.16 world, lockstep/rollback-replayable).
// CP1 is ONLY the unified world + the body->fluid neighbour QUERY (which fluid particles each body contains)
// — the link, NO momentum exchange yet (CP2 buoyancy/drag, CP3 displacement, CP4 the coupled step, CP5
// lockstep, CP6 render). Pure CPU, header-only, NO device, NO backend symbols, NO <cmath>. Namespace
// hf::sim::couple. The whole flagship reuses the proven engine/sim/fpx.h + engine/sim/fluid.h toolbox.
//
// THE QUERY (the one new shape vs GR2's grain->grain neighbour search): GR2 finds, per GRAIN, its neighbour
// grains in a fixed 27-cell 3x3x3 stencil (the search radius ~ a grain diameter). CP1 finds, per BODY, the
// fluid particles inside the body's sphere — and a body radius is typically MANY fluid cells wide, so the
// body spans a RANGE of cells (its fpx::BodyAabb in cell space), NOT a fixed 27-cell stencil. So
// GatherBodyParticles iterates the cell range [CellOf(body.pos - radius) .. CellOf(body.pos + radius)] over
// the FL2 fluid cell table, and for each fluid particle in those cells accepts iff the per-axis box reject
// |body.pos.axis - p.pos.axis| < body.radius passes (a box, NOT a sphere). Built by count->scan->emit (CSR
// bodyStart[bodyCount+1] + bodyParticles[], grouped by body, ascending particle index — the GR2/FL2
// EMIT-order discipline, fully deterministic). Pure int32.
//
// THE int32 DECISION (the FL2/GR2 precedent): the body->fluid query is integer index arithmetic + the
// per-axis |body.pos.axis - p.pos.axis| < body.radius box reject (fx is int32 -> a PURE INT32 compare, NO
// products, NO int64, NO sqrt). So the GPU couple_body_{count,scan,emit}.comp shaders MSL-generate NATIVELY
// -> a TRUE GPU pass on both Vulkan AND Metal (the strongest cross-vendor proof, like GR2 — strict
// zero-differing-pixel). The exact radial sphere cull (|p - body| < radius) is DEFERRED to CP2's force (the
// buoyant/drag impulse is 0 outside the sphere), so the over-inclusive box candidate is correct — exactly
// the FL2/GR2 "over-inclusive box, exact cull deferred" discipline.
//
// REUSE MAP (file:line — read-only, NOT modified; couple is the additive sibling):
//   * engine/sim/fluid.h: FluidParticle (fluid.h:82-88), FluidKernel (fluid.h:441 — the cell-size h),
//     FluidGrid/MakeGrid/CellOf/FlatCellId/CellCount (fluid.h:226-274), FluidCellTable/BuildCellTable
//     (fluid.h:283-315) — the FL2 grid-hash + cell table this query iterates.
//   * engine/sim/fpx.h: FxBody (fpx.h:116-131 — pos, vel, invMass, flags, radius, orient, angVel), FxVec3,
//     FxAabb/BodyAabb (fpx.h:210-220 — the body's integer AABB), kFlagDynamic (fpx.h:133), CellId/FxCell.
//   * engine/sim/grain.h GR2 (the closest twin to MIRROR): GrainNeighborAccept (the per-axis box reject),
//     CountGrainNeighbors/BuildGrainNeighborList (the count->scan->emit CSR) — CP1's CountBodyParticles/
//     GatherBodyParticles are the SAME shape with a body-AABB cell RANGE (not a 27-cell stencil) and
//     body.radius (not hSearch).

#include <cstdint>
#include <vector>

#include "sim/fpx.h"     // read-only: fx / FxVec3 / FxBody / FxAabb / BodyAabb / kFlagDynamic / FloorDiv / FxCell
#include "sim/fluid.h"   // read-only: FluidParticle / FluidKernel / FluidGrid / MakeGrid / CellOf / FlatCellId /
                         // CellCount / FluidCellTable / BuildCellTable (the FL2 grid-hash + cell table)

namespace hf::sim {
namespace couple {

// Reuse the fpx Q16.16 scalar + vector toolbox verbatim (NO new fixed-point primitives).
using fpx::fx;
using fpx::FxVec3;
using fpx::FxCell;
using fpx::FloorDiv;
using fpx::FxAdd;     // Slice CP2: the buoyancy/drag vector toolbox (reused from fpx.h, no new primitives)
using fpx::FxSub;
using fpx::FxScale;
inline constexpr int kFrac = fpx::kFrac;   // Q16.16 fractional bits (== fpx::kFrac, MUST match the shader)
inline constexpr fx  kOne  = fpx::kOne;    // 1.0 in Q16.16 (65536)

// ----- The unified coupled world (bodies + fluid in one Q16.16 frame) -------------------------------------
// The bodies and the fluid share the SAME world units (the CL4/GR3 deformable-meets-rigid precedent —
// fpx::FxBody and fluid::FluidParticle are both Q16.16). CP1 only needs `bodies` + `particles` + the grid
// cell-size from `kernel.h`; gravity/dt/groundY are carried for CP2-CP6 (the coupled step + lockstep).
struct CoupleWorld {
    std::vector<fpx::FxBody>            bodies;       // the dynamic rigid bodies (the FPX sim members)
    std::vector<fluid::FluidParticle>  particles;    // the PBF fluid particle pool (the FL sim members)
    fluid::FluidKernel                 kernel;       // CP1 uses ONLY kernel.h (the grid cell-size)
    FxVec3                             gravity;      // carried for CP2-CP6 (the coupled step)
    fx                                 dt = 0;       // carried for CP4-CP6 (the coupled step / lockstep)
    fx                                 groundY = 0;  // carried for CP2-CP6 (the ground clamp)
};

// ----- The body->particle reject (the PURE INT32 per-axis |dx| < radius box test) -------------------------
// BodyParticleAccept(b, p): accept fluid particle p as a candidate of body b iff |b.pos.axis - p.pos.axis| <
// b.radius on EVERY axis (a box, NOT a sphere — the over-inclusive candidate set CP2's buoyant/drag force
// culls). PURE INT32: an integer subtract + abs + compare per axis, NO products, NO int64, NO sqrt. The
// shader copies THIS verbatim. (== fluid.h::NeighborAccept / grain.h::GrainNeighborAccept with b.radius for h
// and the body centre for `a`.)
inline bool BodyParticleAccept(const fpx::FxBody& b, const fluid::FluidParticle& p) {
    fx dx = b.pos.x - p.pos.x; if (dx < 0) dx = -dx;
    fx dy = b.pos.y - p.pos.y; if (dy < 0) dy = -dy;
    fx dz = b.pos.z - p.pos.z; if (dz < 0) dz = -dz;
    return dx < b.radius && dy < b.radius && dz < b.radius;
}

// ----- The CSR body->particle query result ----------------------------------------------------------------
// bodyStart[b..] is the exclusive prefix-sum of per-body gathered counts (bodyStart has bodyCount+1 entries;
// bodyStart[b]..bodyStart[b+1] is body b's slice), and bodyParticles[] holds the gathered fluid-particle
// indices grouped by body, ASCENDING particle index within each body (deterministic). The GPU
// couple_body_{count,scan,emit} mirror this byte-for-byte.
struct CoupleQuery {
    std::vector<uint32_t> bodyStart;       // bodyCount+1 exclusive prefix-sum offsets (CSR row pointers)
    std::vector<uint32_t> bodyParticles;   // gathered fluid-particle indices grouped by body (ascending)
};

// ----- CountBodyParticles: the per-body count over the body's AABB cell range (count) ---------------------
// For each body i, iterate the cell range [CellOf(body.pos - radius) .. CellOf(body.pos + radius)] (the
// fpx::BodyAabb quantised to fluid cells at cell-size grid.h); for each fluid particle in those cells, count
// it iff BodyParticleAccept(body, p). perBodyOut[i] = #gathered; returns the total. The GPU
// couple_body_count mirrors THIS per-thread (one thread per body i). The cell range is the ONE delta vs
// GR2's fixed 27-cell stencil — a body spans MANY cells. Cells outside the grid are skipped (clamp). Pure
// int32. (Iterating the AABB corners in cell space is correct because BodyParticleAccept's box is exactly
// [pos-radius, pos+radius] per axis, which the cell range [CellOf(pos-radius), CellOf(pos+radius)] fully
// covers — every accepted particle lies in one of those cells.)
inline uint32_t CountBodyParticles(const CoupleWorld& world, const fluid::FluidGrid& grid,
                                   const fluid::FluidCellTable& table, std::vector<uint32_t>& perBodyOut) {
    const uint32_t bodyCount = (uint32_t)world.bodies.size();
    perBodyOut.assign((size_t)bodyCount, 0u);
    uint32_t total = 0;
    for (uint32_t i = 0; i < bodyCount; ++i) {
        const fpx::FxBody& b = world.bodies[(size_t)i];
        const fpx::FxAabb aabb = fpx::BodyAabb(b);
        const FxCell loCell = fluid::CellOf(aabb.lo, grid.h);
        const FxCell hiCell = fluid::CellOf(aabb.hi, grid.h);
        uint32_t c = 0u;
        for (int cz = loCell.z; cz <= hiCell.z; ++cz)
        for (int cy = loCell.y; cy <= hiCell.y; ++cy)
        for (int cx = loCell.x; cx <= hiCell.x; ++cx) {
            // Skip cells outside the bounded grid (clamp — the body may overhang the particle AABB).
            if (cx < grid.cellMin.x || cx >= grid.cellMin.x + grid.gridDim.x) continue;
            if (cy < grid.cellMin.y || cy >= grid.cellMin.y + grid.gridDim.y) continue;
            if (cz < grid.cellMin.z || cz >= grid.cellMin.z + grid.gridDim.z) continue;
            const uint32_t cell = fluid::FlatCellId(FxCell{cx, cy, cz}, grid);
            for (uint32_t s = table.cellStart[cell]; s < table.cellStart[cell + 1u]; ++s) {
                const uint32_t j = table.cellParticles[s];
                if (BodyParticleAccept(b, world.particles[(size_t)j])) ++c;
            }
        }
        perBodyOut[(size_t)i] = c;
        total += c;
    }
    return total;
}

// ----- GatherBodyParticles: the full body->fluid query (count->scan->emit) --------------------------------
// (1) Build the fluid FluidGrid + FluidCellTable (reuse FL2). (2) CountBodyParticles -> per-body counts. (3)
// exclusive prefix-sum -> bodyStart. (4) EMIT each accepted particle index into body i's disjoint slice in
// the FIXED order: ascending cell (cz,cy,cx) over the body's AABB range, then ascending particle index
// within a cell (cellParticles is already ascending-index per cell) -> fully deterministic. Each body writes
// into its OWN DISJOINT [bodyStart[i], bodyStart[i+1]) range -> the GPU emit is race-free, NO atomics (the
// per-body-disjoint pattern). The GPU does the SAME three passes -> the GPU bodyParticles + bodyStart memcmp
// against this byte-for-byte. (DET-CRUX, the GR2/FL2 lesson: the per-body EMIT scatter is fixed ascending
// order; the count + the per-body lists are per-body-disjoint -> race-free. The reused fluid cell-EMIT is
// single-thread ascending, already correct in FL2.) Pure int32.
inline CoupleQuery GatherBodyParticles(const CoupleWorld& world) {
    const uint32_t bodyCount = (uint32_t)world.bodies.size();
    CoupleQuery q;

    // (1) the FL2 fluid grid + cell table over the particle pool (cell-size = kernel.h, the FL2 smoothing
    // radius). Empty pool -> a 1x1x1 grid (deterministic degenerate); every body then gathers 0.
    const fluid::FluidGrid grid = fluid::MakeGrid(world.particles, world.kernel.h);
    const fluid::FluidCellTable table = fluid::BuildCellTable(world.particles, grid);

    // (2) COUNT: per-body gathered count over the body's AABB cell range.
    std::vector<uint32_t> counts;
    const uint32_t total = CountBodyParticles(world, grid, table, counts);

    // (3) SCAN: exclusive prefix-sum -> bodyStart (bodyCount+1 entries; the last == total).
    q.bodyStart.assign((size_t)bodyCount + 1u, 0u);
    uint32_t running = 0;
    for (uint32_t i = 0; i < bodyCount; ++i) {
        q.bodyStart[(size_t)i] = running;
        running += counts[(size_t)i];
    }
    q.bodyStart[(size_t)bodyCount] = running;   // sentinel == total

    // (4) EMIT: each body writes its gathered particle indices into its disjoint slice (ascending cell,
    // then ascending particle index within a cell -> the FL2/GR2 deterministic emit order).
    q.bodyParticles.assign((size_t)total, 0u);
    for (uint32_t i = 0; i < bodyCount; ++i) {
        const fpx::FxBody& b = world.bodies[(size_t)i];
        const fpx::FxAabb aabb = fpx::BodyAabb(b);
        const FxCell loCell = fluid::CellOf(aabb.lo, grid.h);
        const FxCell hiCell = fluid::CellOf(aabb.hi, grid.h);
        uint32_t base  = q.bodyStart[(size_t)i];
        uint32_t local = 0u;
        for (int cz = loCell.z; cz <= hiCell.z; ++cz)
        for (int cy = loCell.y; cy <= hiCell.y; ++cy)
        for (int cx = loCell.x; cx <= hiCell.x; ++cx) {
            if (cx < grid.cellMin.x || cx >= grid.cellMin.x + grid.gridDim.x) continue;
            if (cy < grid.cellMin.y || cy >= grid.cellMin.y + grid.gridDim.y) continue;
            if (cz < grid.cellMin.z || cz >= grid.cellMin.z + grid.gridDim.z) continue;
            const uint32_t cell = fluid::FlatCellId(FxCell{cx, cy, cz}, grid);
            for (uint32_t s = table.cellStart[cell]; s < table.cellStart[cell + 1u]; ++s) {
                const uint32_t j = table.cellParticles[s];
                if (BodyParticleAccept(b, world.particles[(size_t)j])) {
                    q.bodyParticles[(size_t)(base + local)] = j;
                    ++local;
                }
            }
        }
    }
    return q;
}

// CountGathered(q): the total gathered body-particle pairs (== bodyParticles.size() == bodyStart.back()) —
// a reporting/stat helper for the showcase coverage proof. Pure integer.
inline uint32_t CountGathered(const CoupleQuery& q) {
    return q.bodyStart.empty() ? 0u : q.bodyStart.back();
}

// MaxPerBody(q): the largest per-body gathered count over the CSR — a reporting/stat helper. Pure integer.
inline uint32_t MaxPerBody(const CoupleQuery& q) {
    uint32_t m = 0;
    for (size_t i = 0; i + 1 < q.bodyStart.size(); ++i) {
        const uint32_t c = q.bodyStart[i + 1] - q.bodyStart[i];
        if (c > m) m = c;
    }
    return m;
}

// ===== Slice CP2 — BUOYANCY + DRAG (fluid->body, the CRUX) ==========================================
// THE FIRST MOMENTUM EXCHANGE of FLAGSHIP #11: each fpx::FxBody sums, over its CP1 gathered fluid-particle
// list (in the FIXED CP1 emit order, ascending), a BUOYANT impulse (the displaced-fluid restoring force, up
// = -normalize(gravity), proportional to the gathered count = the displaced volume) + a DRAG impulse (toward
// the local fluid velocity, which CP2 damps since the fluid is held STATIC), and floats/sinks/damps. ONE-WAY
// for now (fluid -> body; the body->fluid reaction is CP3). LINEAR only — a sphere body's buoyancy acts
// through its centre (no torque), so angVel is untouched until CP4's non-uniform coupling.
//
// THE int64 DECISION (the GR3/FL4 split): the buoyancy/drag math is int64 (fxmul/fxdiv/FxScale) ->
// shaders/couple_buoyancy.comp is VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) + the Metal
// --couple-buoyancy showcase runs the CPU StepCoupleBuoyancy (byte-identical by construction). The CP1
// grid-hash query (re-run each step) stays int32 MSL-native.
//
// THE CRUX — the order-sensitive body-force reduction: summing N fluid-particle impulses onto one body is a
// reduction; the integer zero-diff bar demands a PINNED summation order. CP2 sums each body's contributions
// in the FIXED CP1 gathered order (bodyParticles[bodyStart[i] .. bodyStart[i+1]), ascending). The GPU
// dispatch is ONE THREAD PER BODY (a serial inner loop over that body's short gathered list) -> multi-thread
// OVER bodies, NOT over particles. With a TINY body count (1-few, << the ~2s watchdog), no body's inner loop
// approaches the single-thread TDR ceiling that sank CL3/FPX3. (Integer addition is exact regardless of
// order, but the fixed order keeps it provably bit-identical to the CPU reference and future-proof.) IF body
// counts ever scale up, a deterministic integer atomic-add reduction would be needed — explicitly OUT of
// scope for CP1-CP6 (flag loudly, like the swraster 64-bit-atomics caveat).

// The host-snapped Q16.16 coupling coefficients. Tuned so a unit-mass FxBody (invMass=kOne, radius 2)
// dropped above the CP1 pool settles at an emergent FLOAT LINE inside the pool (buoyancy ~ gravity) — NOT
// sunk to the bed, NOT flying out. BUOYANCY: F_buoy per displaced particle, summed over the gathered count,
// so a deeper (more-submerged) body feels MORE upward force -> a stable equilibrium depth (the Archimedes
// proxy: displaced volume ∝ gathered count). DRAG: a linear damping toward the (static) fluid's mean
// velocity, so the body settles instead of oscillating. The float line is EMERGENT/iterative (the GR4/FL4
// caveat), NOT an exact Archimedes depth.
//   kBuoyPerParticle = 0.42 (Q16.16): with a ~30-particle submerged gather the upward force ~12.6 >> |g|=9.8
//     near the surface but the gather SHRINKS as the body rises out of the pool, so it balances gravity at a
//     partial-submersion float line. (host-snapped: 0.42 * 65536 ~ 27525.)
//   kDrag = 1.5 (Q16.16): a firm linear damping so the bob settles within K steps instead of ringing.
inline constexpr fx kBuoyPerParticle = (fx)(0.42 * (double)kOne + 0.5);   // 27525 (Q16.16, ~0.42/particle)
inline constexpr fx kDrag            = (fx)(1.5  * (double)kOne + 0.5);   // 98304 (Q16.16, ~1.5 damping)

// AccumBodyForces(world, query, dt): the per-body buoyancy+drag impulse accumulate (the FIXED-ORDER
// reduction). For each DYNAMIC body i (static / non-dynamic skipped), over its CP1 gathered list in ASCENDING
// order: up = -FxNormalize(gravity); F_buoy = fxmul(kBuoyPerParticle, count<<kFrac) along up (count<<kFrac
// promotes the integer gathered count to Q16.16 so the buoyant force is kBuoyPerParticle PER displaced
// particle); vFluidAvg = (Σ particle.vel)/count per axis (the fixed-order int64 sum / the integer count);
// F_drag = fxmul(kDrag, vFluidAvg - body.vel) per axis; body.vel += FxScale(F_buoy + F_drag, body.invMass)·dt.
// A body that gathers 0 particles feels no buoyancy/drag (free-fall). LINEAR only (angVel untouched). int64.
// couple_buoyancy.comp copies THIS body VERBATIM (one thread per body i). Deterministic (fixed gathered order,
// fixed op order) -> bit-identical to the GPU memcmp.
inline void AccumBodyForces(CoupleWorld& world, const CoupleQuery& query, fx dt) {
    const uint32_t bodyCount = (uint32_t)world.bodies.size();
    // up = -normalize(gravity) (the buoyant direction). gravity 0 -> a deterministic +Y fallback.
    const FxVec3 up = fpx::FxNormalize(FxVec3{-world.gravity.x, -world.gravity.y, -world.gravity.z});
    for (uint32_t i = 0; i < bodyCount; ++i) {
        fpx::FxBody& b = world.bodies[(size_t)i];
        if (!(b.flags & fpx::kFlagDynamic)) continue;   // static/kinematic -> no buoyancy/drag (the pinned case)
        const uint32_t s0 = query.bodyStart[(size_t)i];
        const uint32_t s1 = query.bodyStart[(size_t)i + 1u];
        const uint32_t count = s1 - s0;
        if (count == 0u) continue;                      // gathers nothing (out of the pool) -> free-fall

        // BUOYANCY: F_buoy = kBuoyPerParticle * count along up. count<<kFrac promotes the integer count to
        // Q16.16 so fxmul yields kBuoyPerParticle PER particle (the displaced-volume proxy).
        const fx countFx = (fx)((int32_t)count << kFrac);
        const fx buoyMag = fpx::fxmul(kBuoyPerParticle, countFx);
        const FxVec3 fBuoy = fpx::FxScale(up, buoyMag);

        // DRAG: vFluidAvg = (Σ_j particle[j].vel) / count, summed in the FIXED CP1 gathered (ascending) order.
        // int64 accumulate (a sum of ~count velocities never overflows int64), then a truncating int64 divide
        // by the integer count -> deterministic, bit-identical CPU<->shader.
        int64_t sumX = 0, sumY = 0, sumZ = 0;
        for (uint32_t s = s0; s < s1; ++s) {
            const fluid::FluidParticle& p = world.particles[(size_t)query.bodyParticles[(size_t)s]];
            sumX += (int64_t)p.vel.x;
            sumY += (int64_t)p.vel.y;
            sumZ += (int64_t)p.vel.z;
        }
        const FxVec3 vFluidAvg{
            (fx)(sumX / (int64_t)count), (fx)(sumY / (int64_t)count), (fx)(sumZ / (int64_t)count)};
        // F_drag = kDrag * (vFluidAvg - body.vel) per axis.
        const FxVec3 dv = FxSub(vFluidAvg, b.vel);
        const FxVec3 fDrag{fpx::fxmul(kDrag, dv.x), fpx::fxmul(kDrag, dv.y), fpx::fxmul(kDrag, dv.z)};

        // Apply the impulse as a velocity delta: vel += (F_buoy + F_drag) * invMass * dt (linear only).
        const FxVec3 fTotal = FxAdd(fBuoy, fDrag);
        const FxVec3 dvel = FxScale(FxScale(fTotal, b.invMass), dt);
        b.vel = FxAdd(b.vel, dvel);
    }
}

// StepCoupleBuoyancy(world, dt): ONE coupled step (the driver — the fluid is held STATIC). (1) re-query the
// body->fluid neighbour lists from the bodies' CURRENT positions (CP1 GatherBodyParticles) -> (2) accumulate
// the buoyancy+drag velocity delta (AccumBodyForces) -> (3) IntegrateBody per body (gravity + the buoyancy-
// adjusted vel — the fpx.h integrate VERBATIM) THEN ResolveGround (the fpx.h RADIUS-AWARE floor: the body's
// SURFACE rests on the floor at groundY + radius — the fpx.h ground contact VERBATIM). The fluid particles
// are NOT moved (the reaction is CP3). Over K steps a body dropped above the pool falls, enters, buoyancy
// builds as it submerges, and it settles to an emergent float line damped by drag (or sinks to the bed
// groundY + radius if it never gathers particles — the buoy=0 control). Both IntegrateBody + ResolveGround
// are copied VERBATIM by couple_buoyancy.comp's host driver (the GPU runs the SAME per-step sequence).
inline void StepCoupleBuoyancy(CoupleWorld& world, fx dt) {
    const CoupleQuery query = GatherBodyParticles(world);   // CP1 re-query (fluid static -> the pool unchanged)
    AccumBodyForces(world, query, dt);                      // CP2 buoyancy + drag velocity delta
    for (fpx::FxBody& b : world.bodies) {                   // fpx.h integrate + radius-aware floor, VERBATIM
        fpx::IntegrateBody(b, world.gravity, world.groundY, dt);
        fpx::ResolveGround(b, world.groundY);               // the body SURFACE rests at groundY + radius
    }
}

// StepCoupleBuoyancySteps(world, dt, steps): run K StepCoupleBuoyancy steps. The CPU reference the GPU body
// state memcmp's against byte-for-byte. Pure integer -> two runs byte-identical, cross-backend identical.
inline void StepCoupleBuoyancySteps(CoupleWorld& world, fx dt, int steps) {
    for (int s = 0; s < steps; ++s) StepCoupleBuoyancy(world, dt);
}

// MeasureFloatLine(world): the honest float-line metric — the mean settled body pos.y over the DYNAMIC bodies
// (a deterministic Q16.16 stat for the float-line proof). 0 dynamic bodies -> 0. The float line is
// EMERGENT/iterative (the GR4/FL4 caveat), NOT an exact Archimedes depth: the proof asserts the body floats
// by a margin above the bed (groundY + radius) + is bounded above + is deterministic + two-run byte-identical.
inline fx MeasureFloatLine(const CoupleWorld& world) {
    int64_t sum = 0; uint32_t n = 0;
    for (const fpx::FxBody& b : world.bodies)
        if (b.flags & fpx::kFlagDynamic) { sum += (int64_t)b.pos.y; ++n; }
    return n == 0u ? 0 : (fx)(sum / (int64_t)n);
}

}  // namespace couple
}  // namespace hf::sim
