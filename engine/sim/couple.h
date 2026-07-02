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

// ===== Slice CP3 — FLUID REACTION / DISPLACEMENT (body->fluid, Newton's 3rd law to CP2) ==============
// The Newton's-3rd-law HALF of CP2: the body now pushes BACK on the fluid. Each fluid particle INSIDE a
// dynamic body is (1) projected OUT to the body surface — the body DISPLACES the fluid (a parting / wake /
// cavity where the body is) — and (2) receives the equal-opposite DRAG impulse — the body imparts its
// momentum to the surrounding fluid. Completes the two-way exchange (CP2 fluid->body, CP3 body->fluid).
//
// THE MIRROR OF CP2 (the per-PARTICLE-over-the-body-set shape): CP2's body-force reduction was per-BODY
// (over its gathered particles). CP3's fluid displacement is the mirror — ONE thread per FLUID PARTICLE,
// each particle iterates the tiny body list (fixed order), and for each body that CONTAINS it accumulates
// the surface-snap push into a SEPARATE dp[] (JACOBI) + applies the drag-reaction velocity impulse. Each
// particle writes ONLY its own dp/vel -> per-particle-disjoint, race-free, NO atomics, [numthreads(64,1,1)]
// MULTI-THREAD, NO TDR (the FL4/GR3 win). This is the EXACT shape of GR3 grain.h::CollideGrainSpheres
// (which iterates per-particle per-sphere), with fpx::FxBody as the sphere (project the fluid POINT to
// b.radius) + the drag-reaction term + the Jacobi dp[] double-buffer.
//
// int64 (FxLength/FxNormalize/fxmul) -> couple_displace.comp is VULKAN-SPIR-V-ONLY (DXC compiles int64;
// glslc cannot) + the Metal --couple-displace showcase runs THIS CPU ApplyBodyToFluid (byte-identical by
// construction, the CP2/GR3/FL4 split). The CP1 query passes stay int32 MSL-native.

using fpx::FxLength;       // read-only: the int64 length (FxISqrt of the sum of squares)
using fpx::FxNormalize;    // read-only: the int64 normalize (+Y fallback on length 0)
using fpx::fxmul;          // read-only: the int64-intermediate Q16.16 multiply

// kDragReaction: the body->fluid drag-reaction coefficient (the CP2 kDrag partner — the equal-opposite of
// the body's CP2 drag, now imparting the body's momentum to the fluid). Host-snapped Q16.16 (~1.5, matching
// kDrag's firm linear damping). The showcase + the CPU reference + the GPU shader share THIS exact constant.
inline constexpr fx kDragReaction = (fx)(1.5 * (double)kOne + 0.5);   // 98304 (Q16.16, ~1.5 reaction drag)

// ApplyBodyToFluid(world): the per-fluid-particle projection-out-of-body + drag reaction (the GR3
// CollideGrainSpheres mold over the body set + a Jacobi dp[]). For each fluid particle p (skip STATIC),
// over each DYNAMIC body b (fixed order), with d = p.pos − b.pos, dist = FxLength(d): if dist < b.radius
// (the particle is INSIDE the body) accumulate the surface-snap push into a SEPARATE dp[] buffer
//   dp_p += FxAdd(b.pos, FxScale(FxNormalize(d), b.radius)) − p.pos    // snap to the body surface
// (dist==0 -> the FxNormalize +Y fallback) AND apply the drag-reaction velocity impulse per axis
//   p.vel += fxmul(kDragReaction, (b.vel − p.vel)) · dt                // toward the body velocity
// then apply p.pos += dp_p for ALL particles after (JACOBI — each particle reads the iteration-start body
// state, NOT the in-progress positions). STATIC particles (kFlagStatic / boundary) -> dp 0, vel untouched.
// Non-dynamic bodies are SKIPPED (they hold; CP3 is the dynamic-body reaction). int64 (FxLength/FxNormalize/
// fxmul). couple_displace.comp copies THIS body VERBATIM (one thread per fluid particle). Deterministic
// (the fixed body order, fixed op order) -> bit-identical to the GPU memcmp + two-run byte-identical.
inline void ApplyBodyToFluid(CoupleWorld& world, fx dt) {
    const size_t n = world.particles.size();
    const size_t bodyCount = world.bodies.size();
    std::vector<FxVec3> dp((size_t)n, FxVec3{0, 0, 0});   // the Jacobi double-buffer (per-particle Δp)
    for (size_t i = 0; i < n; ++i) {
        fluid::FluidParticle& p = world.particles[i];
        if (p.flags & fluid::kFlagStatic) continue;       // boundary particle -> dp 0, vel untouched
        FxVec3 accum{0, 0, 0};
        for (size_t bi = 0; bi < bodyCount; ++bi) {
            const fpx::FxBody& b = world.bodies[bi];
            if (!(b.flags & fpx::kFlagDynamic)) continue; // non-dynamic body -> holds (the pinned case)
            const FxVec3 d = FxSub(p.pos, b.pos);         // particle relative to the body centre
            const fx dist = FxLength(d);
            if (dist >= b.radius) continue;               // outside the body sphere -> no displacement
            // (1) POSITIONAL DISPLACEMENT: snap the particle to the body surface (the body parts the fluid).
            const FxVec3 nrm = FxNormalize(d);            // outward normal (dist==0 -> {0,kOne,0} fallback)
            const FxVec3 surf = FxAdd(b.pos, FxScale(nrm, b.radius));   // the surface point along the normal
            accum = FxAdd(accum, FxSub(surf, p.pos));     // into the Jacobi dp[] (the CollideParticleSphere push)
            // (2) DRAG REACTION: the body imparts momentum to the fluid (the equal-opposite of CP2's drag).
            const FxVec3 dv = FxSub(b.vel, p.vel);        // toward the body velocity
            p.vel.x += fxmul(fxmul(kDragReaction, dv.x), dt);
            p.vel.y += fxmul(fxmul(kDragReaction, dv.y), dt);
            p.vel.z += fxmul(fxmul(kDragReaction, dv.z), dt);
        }
        dp[i] = accum;
    }
    // Apply pos += dp for all particles (Jacobi — disjoint per-particle writes, race-free).
    for (size_t i = 0; i < n; ++i) {
        if (world.particles[i].flags & fluid::kFlagStatic) continue;
        world.particles[i].pos = FxAdd(world.particles[i].pos, dp[i]);
    }
}

// MeasureFluidPenetration(world): the honest no-penetration metric (the FL4/GR3 caveat) — over every fluid
// particle / dynamic body pair, sum/max the particle-into-body penetration pen = b.radius − |p.pos − b.pos|
// > 0. Returns {peak, summed} in Q16.16 (int64 accumulator). DETERMINISTIC + bit-exact. The showcase's
// "fluid parted" proof compares this BEFORE vs AFTER ApplyBodyToFluid (penAfter < penBefore — the FL4/GR3
// honesty: relieved, NOT zero; Jacobi single-projection so a particle inside MULTIPLE bodies leaves a
// deterministic-but-nonzero residual). Static particles are counted (they CAN sit inside a body; CP3 does
// not move them, so their penetration is part of the honest residual). Non-dynamic bodies skipped.
struct FluidPenetration { int64_t peak = 0; int64_t summed = 0; };
inline FluidPenetration MeasureFluidPenetration(const CoupleWorld& world) {
    FluidPenetration out;
    for (const fluid::FluidParticle& p : world.particles)
        for (const fpx::FxBody& b : world.bodies) {
            if (!(b.flags & fpx::kFlagDynamic)) continue;
            const fx pen = b.radius - FxLength(FxSub(p.pos, b.pos));
            if (pen > 0) { out.summed += (int64_t)pen; if ((int64_t)pen > out.peak) out.peak = (int64_t)pen; }
        }
    return out;
}

// CountDisplaced(world): the deterministic count of fluid particles INSIDE at least one dynamic body
// (dist < b.radius) — the "displaced > 0" coverage stat for the showcase proof. Static particles ARE
// counted (they sit inside the body too). Pure int64 compare -> bit-exact CPU<->GPU.
inline uint32_t CountDisplaced(const CoupleWorld& world) {
    uint32_t c = 0;
    for (const fluid::FluidParticle& p : world.particles) {
        bool inside = false;
        for (const fpx::FxBody& b : world.bodies) {
            if (!(b.flags & fpx::kFlagDynamic)) continue;
            if (FxLength(FxSub(p.pos, b.pos)) < b.radius) { inside = true; break; }
        }
        if (inside) ++c;
    }
    return c;
}

// ===== Slice CP4 — THE COUPLED STEP (the bobbing barrel, the INTEGRATED two-way solver) ==============
// The fourth slice of FLAGSHIP #11: ONE deterministic tick that runs the fluid's own incompressibility
// (FL4) AND both exchange directions (CP2 fluid->body, CP3 body->fluid) AND the rigid integrate — a dynamic
// fluid AND a dynamic body in one bidirectional loop. The result: a barrel BOBS under emergent buoyancy in
// an INCOMPRESSIBLE fluid (no script). CP4 ORCHESTRATES the existing bit-exact pieces — it composes the FL4
// SUB-passes (NOT StepFluid wholesale, which would re-predict + re-build the neighbour list and skip the
// coupling) interleaved with CP2 AccumBodyForces + CP3 ApplyBodyToFluid + the rigid IntegrateBody/
// ResolveGround, in the LOCKED order below. NO new shader, NO new RHI: the GPU showcase is a host-driven
// multi-pass driver over the EXISTING FL4 fluid_* + CP2 couple_buoyancy + CP3 couple_displace shaders.
//
// THE COUPLED TICK (StepCouple, the locked make-or-break order):
//   (1) PREDICT:  fluid IntegrateFluid(particles) [FL1: prev=pos, predict pos]; each DYNAMIC body
//                 vel += gravity·dt  (velocity ONLY — the position integrates at (5)).
//   (2) BUILD:    grid = MakeGrid; table = BuildCellTable; nbr = BuildNeighborList (FL2) — once per step,
//                 FIXED across the K iters (the standard PBF choice); query = GatherBodyParticles (CP1),
//                 also built ONCE per step (from the predicted fluid + the current body pos).
//   (3) ITERATE (K JACOBI iters), EACH:
//         (3a) FL4 density: ComputeDensity -> ComputeLambda -> SolveDensityConstraint -> apply pos+=Δp.
//         (3b) CP3 body->fluid: ApplyBodyToFluid (displace fluid out of bodies + drag reaction).
//         (3c) CP2 fluid->body: AccumBodyForces (buoyancy + drag -> body.vel delta, over the CP1 query).
//   (4) FLUID VELOCITY: vel = (pos − prev)/dt   (the FL4 PBF velocity update).
//   (5) BODY INTEGRATE: each DYNAMIC body pos += vel·dt; ResolveGround(body, groundY)   (the bed clamp).
// The body accumulates gravity (1) + buoyancy/drag over the K iters (3c), THEN its position integrates (5);
// over many steps it falls in, buoyancy builds, and it BOBS around the float line (damped by drag) —
// emergent, no script. Pure integer, fixed op order -> two runs bit-identical AND bit-exact GPU==CPU. Bodies
// do not collide with each other (single body / non-overlapping; body-body contacts are out of scope).
//
// The neighbour list + the CP1 query are built ONCE per step (the PBF choice); CP4 reuses the FL2/CP1
// helpers VERBATIM. The kernel LUT (kernel.W/gradW/restDensity/epsilon/bins) the caller built (BuildKernelTable)
// drives the FL4 density solve — CP4 reads world.kernel exactly as StepFluid reads its kernel.

inline void StepCouple(CoupleWorld& world, fx dt, int iters) {
    const size_t n = world.particles.size();
    // (1) PREDICT: the fluid integrate (FL1 — vel += g·dt; prev = pos; pos += vel·dt; floor clamp) AND each
    // dynamic body's gravity velocity integrate (velocity ONLY; the position integrates at (5)).
    fluid::IntegrateFluid(world.particles, world.gravity, dt, world.groundY);
    for (fpx::FxBody& b : world.bodies) {
        if (!(b.flags & fpx::kFlagDynamic)) continue;
        b.vel.x += fpx::fxmul(world.gravity.x, dt);
        b.vel.y += fpx::fxmul(world.gravity.y, dt);
        b.vel.z += fpx::fxmul(world.gravity.z, dt);
    }
    // (2) BUILD: the FL2 neighbour list from the PREDICTED positions (built ONCE, fixed across the K iters)
    // + the CP1 body->fluid query (built ONCE per step, from the predicted fluid + the current body pos).
    const fluid::FluidGrid grid = fluid::MakeGrid(world.particles, world.kernel.h);
    const fluid::FluidCellTable table = fluid::BuildCellTable(world.particles, grid);
    const fluid::FluidNeighborList list = fluid::BuildNeighborList(world.particles, grid, table, world.kernel.h);
    const CoupleQuery query = GatherBodyParticles(world);
    // (3) K JACOBI iterations: FL4 density solve -> CP3 body->fluid -> CP2 fluid->body, each iteration.
    std::vector<fx> density, lambda;
    std::vector<FxVec3> dp;
    for (int it = 0; it < iters; ++it) {
        // (3a) FL4 density: ρ_i for ALL -> λ_i for ALL -> Δp_i for ALL (separate dp buffer) -> apply p+=Δp.
        fluid::ComputeDensity(world.particles, list, world.kernel, density);
        fluid::ComputeLambda(world.particles, list, world.kernel, density, lambda);
        fluid::SolveDensityConstraint(world.particles, list, world.kernel, lambda, dp);
        for (size_t i = 0; i < n; ++i) {
            if (world.particles[i].flags & fluid::kFlagStatic) continue;
            world.particles[i].pos = FxAdd(world.particles[i].pos, dp[i]);
        }
        // (3b) CP3 body->fluid: displace the fluid out of the bodies (+ drag reaction).
        ApplyBodyToFluid(world, dt);
        // (3c) CP2 fluid->body: accumulate buoyancy + drag into the body velocities (over the CP1 query).
        AccumBodyForces(world, query, dt);
    }
    // (4) FLUID VELOCITY: vel = (pos − prev)/dt (the FL4 PBF velocity update).
    if (dt != 0) {
        for (size_t i = 0; i < n; ++i) {
            if (world.particles[i].flags & fluid::kFlagStatic) continue;
            const FxVec3 dpos = FxSub(world.particles[i].pos, world.particles[i].prev);
            world.particles[i].vel =
                FxVec3{fpx::fxdiv(dpos.x, dt), fpx::fxdiv(dpos.y, dt), fpx::fxdiv(dpos.z, dt)};
        }
    }
    // (5) BODY INTEGRATE: pos += vel·dt for each dynamic body THEN ResolveGround (the radius-aware bed clamp).
    for (fpx::FxBody& b : world.bodies) {
        if (!(b.flags & fpx::kFlagDynamic)) continue;
        b.pos.x += fpx::fxmul(b.vel.x, dt);
        b.pos.y += fpx::fxmul(b.vel.y, dt);
        b.pos.z += fpx::fxmul(b.vel.z, dt);
        fpx::ResolveGround(b, world.groundY);
    }
}

// StepCoupleSteps(world, dt, iters, steps): run K coupled ticks. The CPU reference the GPU multi-pass driver
// memcmp's against byte-for-byte (the fluid + body state). Pure integer -> two runs byte-identical,
// cross-backend identical (the GPU runs the FL4/CP2/CP3 int64 passes Vulkan-only; Metal runs THIS).
inline void StepCoupleSteps(CoupleWorld& world, fx dt, int iters, int steps) {
    for (int s = 0; s < steps; ++s) StepCouple(world, dt, iters);
}

// MeasureCoupleState(world): the honest emergent metrics helper. Returns the mean DYNAMIC-body float line
// (pos.y), the FL4 summed |ρ−ρ0| density residual (the incompressibility coherence stat), and the dynamic
// body count. Deterministic Q16.16 stats for the proofs (the GR4/FL4 caveat shape — the float line + the
// bob are EMERGENT/within-band, NOT exact depths). The density residual is computed from the CURRENT fluid
// (a fresh grid/neighbour list + ComputeDensity), bit-exact CPU<->GPU.
struct CoupleState {
    fx      floatY = 0;          // the mean settled DYNAMIC-body pos.y (the emergent float line)
    int64_t densityResidual = 0; // the FL4 summed |ρ_i − ρ0| (the incompressibility coherence stat)
    uint32_t dynamicBodies = 0;  // the count of dynamic bodies (floatY is their mean)
};
inline CoupleState MeasureCoupleState(const CoupleWorld& world) {
    CoupleState s;
    int64_t sum = 0;
    for (const fpx::FxBody& b : world.bodies)
        if (b.flags & fpx::kFlagDynamic) { sum += (int64_t)b.pos.y; ++s.dynamicBodies; }
    s.floatY = s.dynamicBodies == 0u ? 0 : (fx)(sum / (int64_t)s.dynamicBodies);
    // The density residual over the current fluid (the FL4 coherence metric). Empty pool -> 0.
    if (!world.particles.empty() && world.kernel.bins > 0 && world.kernel.restDensity > 0) {
        const fluid::FluidGrid grid = fluid::MakeGrid(world.particles, world.kernel.h);
        const fluid::FluidCellTable table = fluid::BuildCellTable(world.particles, grid);
        const fluid::FluidNeighborList list =
            fluid::BuildNeighborList(world.particles, grid, table, world.kernel.h);
        std::vector<fx> density;
        fluid::ComputeDensity(world.particles, list, world.kernel, density);
        s.densityResidual = fluid::DensityResidual(density, world.kernel.restDensity);
    }
    return s;
}

// ===== Slice CP5 — LOCKSTEP + ROLLBACK (the MULTI-BODY netcode HEADLINE) =============================
// Prove the bit-exact CP4 coupled step (StepCouple, itself bit-identical Vulkan/Metal) is true cross-platform
// LOCKSTEP + ROLLBACK — and the arc's FIRST MULTI-BODY lockstep: a peer fed the INPUT command stream ALONE
// (NOT full state) re-derives the authority's exact COUPLED state — BOTH the rigid bodies AND the fluid —
// bit-for-bit, and a mispredicted input is corrected by rolling back to a saved snapshot + re-simulating the
// authoritative stream. PURE CPU, 0 backend symbols, NO new shader / RHI: a determinism PROPERTY of the
// existing bit-exact StepCouple. The DIRECT COMPOSITION of the FPX5 (rigid) + FL5 (fluid) harnesses over
// CoupleWorld — the fluid.h FluidCommand/ApplyFluidCommand/SimFluidTick/SnapshotFluid/RestoreFluid/
// RunFluidLockstep/RunFluidRollback shape (and grain.h GR5) over CoupleWorld + StepCouple. NO <cmath>, NO
// RNG, NO clock. The trilogy's netcode story now spans a COUPLED multi-material system: shove the barrel, and
// two peers re-simulate the bob AND the splash bit-for-bit.
//
// THE MULTI-BODY TWIST (the new thing vs FL5/GR5): the world has TWO heterogeneous body sets — the rigid
// std::vector<fpx::FxBody> bodies AND the std::vector<fluid::FluidParticle> particles. SnapshotCouple
// deep-copies BOTH; RunCoupleLockstep's replica==authority must memcmp BOTH; a CoupleCommand can target a
// rigid body OR a fluid particle. This is the FIRST lockstep over a coupled multi-material system — strictly
// more than FL5 (fluid alone) or FPX5 (rigid alone).
//
// A CoupleCommand is the deterministic per-tick INPUT a netcode layer would put on the wire (NOT full state).
// kCmdBodyShove adds `arg` (a delta-velocity) to a rigid body's velocity (the "shove the barrel" headline);
// kCmdBodyMove adds `arg` (a delta-position) to a body's position; kCmdFluidWind adds `arg` (a delta-velocity)
// to a fluid particle's velocity. Integer adds; static bodies (non-dynamic) / static particles (kFlagStatic)
// are never mutated; out-of-range target / unknown kind -> a no-op (deterministic). A std::vector<CoupleCommand>
// is the command STREAM, processed in ARRAY ORDER per tick (the deterministic-order contract — the same order
// on every peer/platform), so authority + replica fed the same stream re-derive the same coupled state exactly.

inline constexpr uint32_t kCmdBodyShove = 0u;   // arg added to target rigid body's velocity (the barrel shove)
inline constexpr uint32_t kCmdBodyMove  = 1u;   // arg added to target rigid body's position (a position nudge)
inline constexpr uint32_t kCmdFluidWind = 2u;   // arg added to target fluid particle's velocity (a wind gust)

struct CoupleCommand {
    uint32_t tick   = 0;   // the tick this input applies on
    uint32_t kind   = 0;   // kCmdBodyShove / kCmdBodyMove / kCmdFluidWind
    uint32_t target = 0;   // the target index (a rigid body index for shove/move; a fluid particle for wind)
    FxVec3   arg;          // the Q16.16 payload (delta-velocity for shove/wind; delta-position for move)
};

// ApplyCoupleCommand(world, c): apply ONE input command to the coupled world (pure integer — add to a body's
// vel/pos or a fluid particle's vel). Out-of-range target is a no-op (deterministic); unknown kind is a no-op;
// static bodies (non-dynamic) and static fluid particles (kFlagStatic) are never mutated (they hold). The
// input event the lockstep/rollback streams are made of. The fluid.h::ApplyFluidCommand / grain.h::
// ApplyGrainCommand twin, extended to the multi-body world.
inline void ApplyCoupleCommand(CoupleWorld& world, const CoupleCommand& c) {
    if (c.kind == kCmdBodyShove || c.kind == kCmdBodyMove) {
        if (c.target >= (uint32_t)world.bodies.size()) return;     // out-of-range body target -> no-op
        fpx::FxBody& b = world.bodies[(size_t)c.target];
        if (!(b.flags & fpx::kFlagDynamic)) return;                // static/kinematic body holds — no input moves it
        if (c.kind == kCmdBodyShove) { b.vel.x += c.arg.x; b.vel.y += c.arg.y; b.vel.z += c.arg.z; }
        else                         { b.pos.x += c.arg.x; b.pos.y += c.arg.y; b.pos.z += c.arg.z; }
    } else if (c.kind == kCmdFluidWind) {
        if (c.target >= (uint32_t)world.particles.size()) return;  // out-of-range fluid target -> no-op
        fluid::FluidParticle& p = world.particles[(size_t)c.target];
        if (p.flags & fluid::kFlagStatic) return;                  // static boundary particle holds
        p.vel.x += c.arg.x; p.vel.y += c.arg.y; p.vel.z += c.arg.z;
    }
    // unknown kind -> a no-op (deterministic).
}

// SimCoupleTick(world, stream, tick, dt, iters): the deterministic per-tick step. (1) apply ALL commands in
// `stream` whose .tick == `tick`, in ARRAY ORDER (the deterministic input-order contract); (2) StepCouple one
// step (CP4 — the bit-exact coupled tick over bodies + fluid). Pure integer, fixed order -> bit-identical on
// every peer/platform. The fluid.h::SimFluidTick / grain.h::SimGrainTick twin over CoupleWorld.
inline void SimCoupleTick(CoupleWorld& world, const std::vector<CoupleCommand>& stream, uint32_t tick,
                          fx dt, int iters) {
    for (const CoupleCommand& c : stream)
        if (c.tick == tick) ApplyCoupleCommand(world, c);
    StepCouple(world, dt, iters);
}

// CoupleSnapshot: the FIRST multi-body snapshot — a deep copy of BOTH the rigid `bodies` AND the fluid
// `particles` vectors (the rollback primitive — a lossless saved tick across the coupled multi-material
// state). The fluid.h::SnapshotFluid result extended to the two heterogeneous body sets.
struct CoupleSnapshot {
    std::vector<fpx::FxBody>           bodies;       // a deep copy of the rigid bodies
    std::vector<fluid::FluidParticle>  particles;    // a deep copy of the fluid particles
};

// SnapshotCouple(world): a deep copy of BOTH the bodies AND the particles vectors (std::vector copy is a deep
// copy). The MULTI-BODY twist: the snapshot covers BOTH body sets. The fluid.h::SnapshotFluid twin.
inline CoupleSnapshot SnapshotCouple(const CoupleWorld& world) {
    CoupleSnapshot s;
    s.bodies    = world.bodies;       // value copy: deep-copies the rigid bodies
    s.particles = world.particles;    // value copy: deep-copies the fluid particles
    return s;
}

// RestoreCouple(world, snap): restore BOTH the bodies AND the particles to a saved snapshot (the rollback).
// Bit-exact round-trip with SnapshotCouple across BOTH vectors. The fluid.h::RestoreFluid twin.
inline void RestoreCouple(CoupleWorld& world, const CoupleSnapshot& snap) {
    world.bodies    = snap.bodies;
    world.particles = snap.particles;
}

// RunCoupleLockstep(init, stream, ticks, dt, iters): THE peer entry point. Run `ticks` SimCoupleTicks from a
// COPY of `init`, applying the command stream -> the final coupled state (bodies + fluid). authority =
// RunCoupleLockstep(...); replica = RunCoupleLockstep(...) from the SAME init + stream (inputs ONLY — no state
// shared) -> BIT-IDENTICAL by determinism (the lockstep proof memcmps BOTH the bodies AND the particles). The
// fluid.h::RunFluidLockstep / grain.h::RunGrainLockstep twin over CoupleWorld.
inline CoupleWorld RunCoupleLockstep(const CoupleWorld& init, const std::vector<CoupleCommand>& stream,
                                     int ticks, fx dt, int iters) {
    CoupleWorld world = init;
    for (int t = 0; t < ticks; ++t)
        SimCoupleTick(world, stream, (uint32_t)t, dt, iters);
    return world;
}

// RunCoupleRollback(init, authStream, mispredictStream, ticks, mispredictTick, dt, iters): the rollback
// harness. (1) run ticks 0..mispredictTick from init applying authStream, SAVING a snapshot AT mispredictTick
// (before that tick is simulated); (2) speculatively advance a few ticks from the snapshot with the MISPREDICTED
// stream (the wrong input) — the client prediction that diverges; (3) "receive" the authoritative input ->
// RestoreCouple to the snapshot + RE-SIMULATE mispredictTick..ticks with the CORRECT authStream -> the final
// corrected coupled state. The proof asserts this == RunCoupleLockstep(init, authStream, ticks) (rollback
// corrected the misprediction EXACTLY, BOTH bodies AND fluid) AND that the mispredicted-before-rollback state
// DIFFERED from the authority (a real divergence was fixed). The fluid.h::RunFluidRollback twin over CoupleWorld.
inline CoupleWorld RunCoupleRollback(const CoupleWorld& init, const std::vector<CoupleCommand>& authStream,
                                     const std::vector<CoupleCommand>& mispredictStream, int ticks,
                                     int mispredictTick, fx dt, int iters) {
    CoupleWorld world = init;
    // (1) advance 0..mispredictTick with the authoritative stream.
    for (int t = 0; t < mispredictTick; ++t)
        SimCoupleTick(world, authStream, (uint32_t)t, dt, iters);
    // (2) SAVE the snapshot at mispredictTick (the rollback restore point — BOTH bodies AND fluid).
    const CoupleSnapshot snap = SnapshotCouple(world);
    // (2b) speculatively advance a few ticks with the MISPREDICTED stream (the wrong input) — the client
    // prediction that diverges from authority. Bounded to the remaining ticks.
    int specTicks = ticks - mispredictTick;
    if (specTicks > 3) specTicks = 3;
    for (int s = 0; s < specTicks; ++s)
        SimCoupleTick(world, mispredictStream, (uint32_t)(mispredictTick + s), dt, iters);
    // (3) ROLLBACK: restore the snapshot + re-simulate mispredictTick..ticks with the CORRECT authStream.
    RestoreCouple(world, snap);
    for (int t = mispredictTick; t < ticks; ++t)
        SimCoupleTick(world, authStream, (uint32_t)t, dt, iters);
    return world;
}

// ===== Slice CP6 — LIT 3D RENDER CAPSTONE (the money-shot, COMPLETES FLAGSHIP #11; FLOAT, render-only) =====
// The SIXTH and FINAL slice RENDERS the bit-exact coupled state as a lit 3D scene — the FxBody barrel floating
// among the fluid droplets as lit 3D INSTANCED SPHERES. The CP1-CP5 sim above stays strict integer/bit-exact;
// here — and ONLY here — we cross to float to build the per-instance render transforms for the rasterizer.
// This is the documented FLOAT visresolve-bar (the FPX6/FL6/GR6 precedent): the SIM is bit-exact, the final
// raster/shade is float (cross-vendor ~the engine baseline, NOT held to the integer zero-diff bar).
//
// CoupleToRenderInstances builds a COMBINED instance set DIRECTLY from the bit-exact CoupleWorld state — one
// LARGE sphere per fpx::FxBody (the barrel, via fpx::FxBodyTransform: translate(pos/kOne) * rotate(orient) *
// scale(radius), the FPX6 render bridge VERBATIM) followed by one SMALL sphere per fluid::FluidParticle (a
// droplet, via fluid::FluidToRenderInstances: translate(pos/kOne) * scale(dropletRadius), the FL6 render
// bridge VERBATIM). The bodies come FIRST so the caller can distinguish them (the body is a LARGE sphere
// among small droplets — NO per-instance color, NO new shader). The instance count is bodies.size() +
// particles.size(). Pure deterministic host float (no RNG, no clock). The provenance: every body transform
// derives from FxBody::pos/orient/radius (the settled output of StepCouple) + every droplet from
// FluidParticle::pos. Empty world (no bodies + no particles) -> empty output (the empty no-op: the cleared
// base scene). Render-only, NO sim mutation — this is the ONLY float crossing of the whole flagship.
inline std::vector<math::Mat4> CoupleToRenderInstances(const CoupleWorld& world, float dropletRadius) {
    std::vector<math::Mat4> out;
    out.reserve(world.bodies.size() + world.particles.size());
    // The bodies FIRST (the LARGE barrel spheres, scaled by body radius via the FPX6 bridge).
    for (const fpx::FxBody& b : world.bodies)
        out.push_back(fpx::FxBodyTransform(b));
    // Then the fluid droplets (the SMALL spheres, the FL6 bridge — translate(pos/kOne) * scale(dropletRadius)).
    const std::vector<math::Mat4> droplets = fluid::FluidToRenderInstances(world.particles, dropletRadius);
    out.insert(out.end(), droplets.begin(), droplets.end());
    return out;
}

// ===== Slice CP7 — SUBMERGED-VOLUME BUOYANCY + SEALED CONTAINMENT (Track-R R5, the CP refinement) =====
// FLAGSHIP #11 (CP1-CP6) shipped with TWO documented caveats: (1) "linear buoyancy only" — CP2's buoyant
// force is kBuoyPerParticle * gatheredCount, a linear proxy of the displaced volume (NOT Archimedes-exact);
// (2) "static-wall containment leaks" — the CP4 basin is a shell of STATIC boundary particles whose ONLY
// hold on the dynamic fluid is the SOFT, UNILATERAL FL4 density constraint (λ=0 when under-dense) plus the
// CP4 predict-time floor clamp; there is NO hard positional barrier, so an agitated particle (the barrel
// dropped hard) can (a) be splashed OVER the one-particle rim (y > wall top) and land OUTSIDE the basin
// footprint where nothing brings it back, or (b) be density-pushed laterally INTO/THROUGH the
// one-particle-thick wall layer between solver iterations (the constraint is not a projection against a
// plane). CP7 closes BOTH caveats, APPEND-ONLY below CP6 (CP1-CP6 code byte-untouched; fluid.h/fpx.h
// byte-untouched):
//   (1) SUBMERGED-VOLUME BUOYANCY (the Archimedes upgrade): the body proxy is a SPHERE (the CP
//       convention — fpx::FxBody carries pos + radius); the EXACT submerged volume at depth d below the
//       fluid surface is the spherical-cap volume V(d) = π d² (3r − d) / 3 for 0 <= d <= 2r (0 above,
//       the full (4/3)πr³ below — V(2r) IS the full-sphere volume, same formula). Implemented in pure
//       Q16.16 integer (SphereCapVolume below); buoyant force = ρ_fluid · |g| · V_submerged along
//       -normalize(gravity). The fluid SURFACE height is derived deterministically from the particles:
//       the POOL-SURFACE QUANTILE — the 90th-percentile pos.y over ALL NON-STATIC particles (a sorted
//       integer select, FluidSurfaceY below; static basin walls are excluded so a wall top is never
//       read as "water"), gated per body by water CONTACT (the body's CP1 gathered list must contain a
//       non-static particle, else free-fall). THE ESTIMATOR LESSON (documented — the first design was
//       rejected empirically): a LOCAL max-y over the body's gathered list reads a body-RIDING splash
//       plume as "surface" (CP3's drag reaction accelerates nearby particles with the body, so a rising
//       body carries a column of riders whose max-y tracks the body) — a positive feedback that rockets
//       the body out of the pool. The global bulk quantile is immune: a few dozen airborne riders
//       cannot shift the 90th percentile of the pool. The Archimedes floating condition falls out
//       exactly: a body settles where V(d)/V_total == ρ_body/ρ_fluid (ρ_b = 0.5 -> half-submerged, d = r).
//   (2) SEALED CONTAINMENT: after the composed step, every NON-STATIC fluid particle is CLAMPED to the
//       basin's AABB interior (per-axis position clamp + the into-wall velocity component zeroed — the
//       FL1 floor-clamp form applied to all six faces). A hard, deterministic seal: ZERO particles
//       outside the basin AABB after every StepCoupleV2 tick, BY CONSTRUCTION.
//   (3) StepCoupleV2: CP4's locked order with the NEW buoyancy swapped in for the old per-iteration
//       linear model (buoyancy+drag run ONCE per step, post the fluid velocity derivation — the GF4
//       "velocity couplings run once" lesson) + the sealed containment as the final pass. The IDENTITY
//       FLAG: CoupleV2Params::legacy != 0 -> StepCoupleV2 delegates VERBATIM to the frozen CP4
//       StepCouple (byte-identical by construction — the identity proof), basin/params ignored.
//   (4) LOCKSTEP: RunCoupleLockstepV2 via the CP5 mold (CoupleCommand/ApplyCoupleCommand/SnapshotCouple
//       reused VERBATIM; only the step fn is swapped).
//
// THE GPU CONVENTION FOR CP7 (documented decision — the CF1/FR8/HR1 precedent): CP7 ships PURE CPU on
// BOTH backends (NO GPU dispatch, NO new shader, NO new RHI) — both Vulkan-Windows and Metal-Mac run
// THIS identical CPU code, so the integer showcase golden is bit-identical cross-backend BY
// CONSTRUCTION. CP2's GPU path (couple_buoyancy.comp, the OLD linear model) stays UNTOUCHED and keeps
// serving the frozen CP2/CP4 showcases; the V2 math is int64 (fxmul/fxdiv/FxLength) so a future GPU
// kernel would be DXC/Vulkan-only + Metal CPU-reference (the established fluid_dp.comp convention).
//
// HONEST CAVEATS: (a) the surface estimator is the max PARTICLE-CENTER y over the gathered box — it is
// quantized by the particle spacing, splash-sensitive, and a particle riding the body's top can inflate
// it; the equilibrium depth is therefore pinned within an honest band around the analytic Archimedes
// depth, not exactly on it. (b) The equilibrium is a damped BOB (the CP4 caveat) — the drag relaxes it
// but the settled depth oscillates within a small band. (c) The V2 drag is a MASS-INDEPENDENT velocity
// relaxation toward the local fluid mean (a documented model choice — physical drag scales with area,
// not mass; it keeps heavy bodies from ringing for hundreds of steps), unlike CP2's invMass-scaled
// drag. (d) SphereCapVolume's int64 triple product bounds the body radius to r <= 16 world units
// (4r³ < 2^63 in raw Q16.16³).

// kPiOver3: π/3 in Q16.16, host-snapped ONCE at compile time (round-to-nearest — the cloth.h restLen /
// fluid.h BuildKernelTable snap discipline). The ONE transcendental constant of the cap volume.
inline constexpr fx kPiOver3 = (fx)(1.0471975511965976 * (double)kOne + 0.5);   // 68629 (Q16.16 π/3)

// SphereCapVolume(r, d): the EXACT spherical-cap volume V(d) = (π/3) · d² · (3r − d), Q16.16 integer.
// d is the submersion depth measured from the sphere's BOTTOM; d <= 0 -> 0 (fully above the surface);
// d >= 2r -> clamped to 2r, where V(2r) = (π/3)(2r)²(r) = (4/3)πr³ IS the full-sphere volume (the same
// formula — no separate branch). ROUNDING (documented): the triple product P = d·d·(3r−d) is formed
// EXACTLY in int64 (Q48.48; bound 4r³ -> r <= 16 world units), then P >> 32 floors it to Q16.16, then
// the kPiOver3 multiply floors again (the fxmul arithmetic shift) — two floor truncations of <= 1 LSB
// each, and P is exactly monotone non-decreasing in d on [0, 2r] (the real cubic is non-decreasing
// there), so the integer V(d) is MONOTONE NON-DECREASING by construction. Pure integer, deterministic.
inline fx SphereCapVolume(fx r, fx d) {
    if (r <= 0 || d <= 0) return 0;
    const fx full = r + r;                                     // 2r (the full-submersion depth)
    if (d > full) d = full;                                    // below the surface -> the full sphere
    const int64_t dd  = (int64_t)d * (int64_t)d;               // d² exactly (Q32.32)
    const int64_t rem = 3 * (int64_t)r - (int64_t)d;           // (3r − d) >= r > 0 (Q16.16)
    const int64_t P   = dd * rem;                              // d²(3r−d) exactly (Q48.48; bound 4r³)
    const int64_t v16 = P >> 32;                               // floor to Q16.16 of d²(3r−d)
    return (fx)(((int64_t)kPiOver3 * v16) >> kFrac);           // · π/3 (Q16.16, floor)
}

// SphereVolume(r): the full sphere volume (4/3)πr³ == SphereCapVolume(r, 2r) (the d=2r cap IS the whole
// sphere). The V_total of the Archimedes floating condition V_sub/V_total == ρ_b/ρ_f.
inline fx SphereVolume(fx r) { return SphereCapVolume(r, r + r); }

// BodyFromDensity(pos, radius, rhoBody): build a DYNAMIC sphere FxBody whose mass follows from its
// density: m = ρ_body · V_sphere(radius), invMass = fxdiv(kOne, m). This is what makes the Archimedes
// floating condition exact in the per-step balance: gravity Δv = g·dt; buoyancy Δv = ρ_f·g·V(d)·invMass·dt;
// equal exactly when V(d) = m/ρ_f, i.e. V(d)/V_total = ρ_b/ρ_f. rhoBody/radius must be > 0 (else a
// static body is returned — invMass 0, deterministic degenerate). Pure integer (fxdiv/fxmul).
inline fpx::FxBody BodyFromDensity(const FxVec3& pos, fx radius, fx rhoBody) {
    fpx::FxBody b;
    b.pos = pos;
    b.radius = radius;
    const fx vol = SphereVolume(radius);
    const fx mass = fpx::fxmul(rhoBody, vol);
    if (mass > 0) {
        b.invMass = fpx::fxdiv(kOne, mass);
        b.flags = fpx::kFlagDynamic;
    }
    return b;
}

// ----- The deterministic fluid-surface probe (the documented estimator) -----------------------------------
// FluidSurfaceY(particles): the POOL SURFACE height = the 90th-percentile pos.y over the NON-STATIC
// particles — the k-th smallest y with k = (count−1)·9/10, found by a VALUE-SPACE BINARY SEARCH (count
// how many y's are <= mid, halve the [minY, maxY] range; <= 32 passes; NO sort, NO <algorithm>, NO
// allocation — pure integer compares, exactly deterministic). STATIC particles (the basin walls/floor)
// are EXCLUDED — a wall top is never read as "water". THE ESTIMATOR CHOICE (documented, empirically
// forced): the naive LOCAL max-y over a body's gathered list reads a body-riding splash plume as
// "surface" (CP3's drag reaction makes nearby particles ride a moving body), a positive feedback that
// rockets the body; the BULK quantile is immune (airborne splash is < 10% of the pool) and matches the
// basin scene's flat shared surface. samples == 0 (no dynamic water at all) -> surfaceY invalid.
// HONEST: the quantile of particle CENTERS sits ~half a particle spacing below the visual surface and
// is spacing-quantized — the Archimedes proof band covers this.
struct SurfaceProbe {
    uint32_t samples = 0;    // the number of NON-STATIC particles the probe saw (the pool size)
    fx       surfaceY = 0;   // the 90th-percentile pos.y over them (valid iff samples > 0)
};
inline SurfaceProbe FluidSurfaceY(const std::vector<fluid::FluidParticle>& particles) {
    SurfaceProbe p;
    fx lo = 0, hi = 0;
    for (const fluid::FluidParticle& fp : particles) {
        if (fp.flags & fluid::kFlagStatic) continue;
        if (p.samples == 0u) { lo = hi = fp.pos.y; }
        else { if (fp.pos.y < lo) lo = fp.pos.y; if (fp.pos.y > hi) hi = fp.pos.y; }
        ++p.samples;
    }
    if (p.samples == 0u) return p;
    // The k-th smallest (0-based k = (count−1)·9/10) by value-space bisection: the smallest value v in
    // [lo, hi] with #{y <= v} >= k+1. Integer floor midpoint (int64 to avoid overflow on the sum).
    const uint32_t k = (uint32_t)(((uint64_t)(p.samples - 1u) * 9u) / 10u);
    while (lo < hi) {
        const fx mid = (fx)(((int64_t)lo + (int64_t)hi) >> 1);   // floor midpoint (never == hi)
        uint32_t cnt = 0;
        for (const fluid::FluidParticle& fp : particles)
            if (!(fp.flags & fluid::kFlagStatic) && fp.pos.y <= mid) ++cnt;
        if (cnt >= k + 1u) hi = mid;
        else lo = mid + 1;
    }
    p.surfaceY = lo;
    return p;
}

// CountBodyWaterContacts(world, query, i): the per-body water-contact GATE — the number of NON-STATIC
// particles in body i's CP1 gathered list (the FIXED ascending order). 0 -> the body is clear of the
// water -> free-fall (no buoyancy, no drag — the CP2 no-gather contract). Deterministic, pure integer.
inline uint32_t CountBodyWaterContacts(const CoupleWorld& world, const CoupleQuery& query, uint32_t i) {
    uint32_t contacts = 0;
    const uint32_t s0 = query.bodyStart[(size_t)i];
    const uint32_t s1 = query.bodyStart[(size_t)i + 1u];
    for (uint32_t s = s0; s < s1; ++s)
        if (!(world.particles[(size_t)query.bodyParticles[(size_t)s]].flags & fluid::kFlagStatic))
            ++contacts;
    return contacts;
}

// SubmergedDepth(body, surfaceY): the sphere's submersion depth d = surfaceY − (pos.y − radius), measured
// from the sphere's BOTTOM up to the fluid surface, clamped to [0, 2r] (the SphereCapVolume domain).
inline fx SubmergedDepth(const fpx::FxBody& b, fx surfaceY) {
    fx d = surfaceY - (b.pos.y - b.radius);
    if (d < 0) d = 0;
    const fx full = b.radius + b.radius;
    if (d > full) d = full;
    return d;
}

// ----- The V2 coupling parameters + the sealed basin ------------------------------------------------------
// rhoFluid: the fluid density ρ_f of the Archimedes force (Q16.16; 0 -> buoyancy OFF). drag: the
// MASS-INDEPENDENT velocity-relaxation coefficient toward the local fluid mean velocity (per second,
// Q16.16; 0 -> drag OFF; see the honest-caveat note — this is a model choice, not CP2's invMass-scaled
// drag). legacy != 0 -> StepCoupleV2 delegates VERBATIM to the frozen CP4 StepCouple (the identity flag;
// basin + the other params are ignored on that path).
struct CoupleV2Params {
    fx       rhoFluid  = 0;  // ρ_f (Q16.16); buoyant force = ρ_f · |g| · V_submerged
    fx       drag      = 0;  // velocity relaxation toward the local fluid mean (per second, Q16.16)
    fx       viscosity = 0;  // FL7 XSPH coefficient applied to the POOL each step (0 -> exact no-op, the
                             // FL7 identity-at-zero) — calms the PBF slosh so the float line settles
    uint32_t legacy    = 0;  // != 0 -> byte-identical CP4 StepCouple delegation (the identity proof)
};

// CoupleBasin: the sealed AABB interior the fluid is clamped to (Q16.16 world corners, lo <= hi). The
// CP4 scene's basin of static wall particles becomes this hard interior box (the walls stay in the scene
// for the density interaction; the AABB is the final positional authority).
struct CoupleBasin {
    FxVec3 lo, hi;
};

// UnboundedBasin(): the no-op basin (the whole int32 range) — ClampToBasin never fires (the containment
// off-switch, for the identity/ablation runs).
inline CoupleBasin UnboundedBasin() {
    constexpr fx kMin = (fx)0x80000000;   // INT32_MIN
    constexpr fx kMax = 0x7FFFFFFF;       // INT32_MAX
    return CoupleBasin{FxVec3{kMin, kMin, kMin}, FxVec3{kMax, kMax, kMax}};
}

// ClampToBasin(particles, basin): the SEAL — clamp every NON-STATIC particle to the basin AABB interior,
// per axis: pos < lo -> pos = lo (and the into-wall velocity component, if still pointing out, is zeroed
// — the FL1 floor-clamp form); pos > hi mirrored. STATIC particles (the wall shell itself, which lives
// OUTSIDE the interior) are untouched. Returns the number of particles that needed at least one clamp
// (a coverage stat). Deterministic (per-particle independent), pure integer compare.
inline uint32_t ClampToBasin(std::vector<fluid::FluidParticle>& particles, const CoupleBasin& basin) {
    uint32_t clamped = 0;
    const size_t n = particles.size();
    for (size_t i = 0; i < n; ++i) {
        fluid::FluidParticle& p = particles[i];
        if (p.flags & fluid::kFlagStatic) continue;
        bool hit = false;
        if (p.pos.x < basin.lo.x) { p.pos.x = basin.lo.x; if (p.vel.x < 0) p.vel.x = 0; hit = true; }
        if (p.pos.x > basin.hi.x) { p.pos.x = basin.hi.x; if (p.vel.x > 0) p.vel.x = 0; hit = true; }
        if (p.pos.y < basin.lo.y) { p.pos.y = basin.lo.y; if (p.vel.y < 0) p.vel.y = 0; hit = true; }
        if (p.pos.y > basin.hi.y) { p.pos.y = basin.hi.y; if (p.vel.y > 0) p.vel.y = 0; hit = true; }
        if (p.pos.z < basin.lo.z) { p.pos.z = basin.lo.z; if (p.vel.z < 0) p.vel.z = 0; hit = true; }
        if (p.pos.z > basin.hi.z) { p.pos.z = basin.hi.z; if (p.vel.z > 0) p.vel.z = 0; hit = true; }
        if (hit) ++clamped;
    }
    return clamped;
}

// CountOutsideBasin(particles, basin): the containment proof metric — the number of NON-STATIC particles
// STRICTLY outside the basin AABB on any axis. StepCoupleV2 ends every tick with this == 0 BY
// CONSTRUCTION (the seal); the OLD StepCouple under agitation is the leaking negative control. Pure
// integer compare, deterministic.
inline uint32_t CountOutsideBasin(const std::vector<fluid::FluidParticle>& particles,
                                  const CoupleBasin& basin) {
    uint32_t outside = 0;
    for (const fluid::FluidParticle& p : particles) {
        if (p.flags & fluid::kFlagStatic) continue;
        if (p.pos.x < basin.lo.x || p.pos.x > basin.hi.x ||
            p.pos.y < basin.lo.y || p.pos.y > basin.hi.y ||
            p.pos.z < basin.lo.z || p.pos.z > basin.hi.z) ++outside;
    }
    return outside;
}

// ----- AccumBodyForcesV2: the ARCHIMEDES buoyancy + drag exchange (fluid->body, run ONCE per step) --------
// probe = FluidSurfaceY(particles) ONCE (the pool-surface quantile). Then for each DYNAMIC body i over
// its CP1 gathered list (the FIXED ascending order — the CP2 reduction discipline):
//   * contacts = CountBodyWaterContacts. 0 -> free-fall (no buoyancy, no drag — the body is clear of
//     the water; the CP2 no-gather contract). This gate also kills the airborne case: a body carried
//     above the pool computes d = clamp(surface − (pos.y − r)) = 0 -> zero buoyant force anyway.
//   * d = SubmergedDepth(body, probe.surfaceY); V = SphereCapVolume(radius, d).
//   * BUOYANCY (a force): up = -FxNormalize(gravity) (the CP2 direction, +Y fallback at gravity 0);
//     buoyMag = fxmul(fxmul(ρ_f, |g|), V) with |g| = FxLength(gravity);
//     vel += up · buoyMag · invMass · dt (the invMass-scaled force -> the Archimedes balance).
//   * DRAG (a velocity relaxation, mass-INDEPENDENT — the documented V2 model choice): vFluidAvg = the
//     fixed-order int64 mean velocity of the NON-STATIC gathered particles;
//     vel += fxmul(drag, vFluidAvg − vel) · dt per axis.
// int64 (fxmul/fxdiv/FxLength/FxNormalize). Deterministic (fixed gathered order, fixed op order).
inline void AccumBodyForcesV2(CoupleWorld& world, const CoupleQuery& query, const CoupleV2Params& params,
                              fx dt) {
    const uint32_t bodyCount = (uint32_t)world.bodies.size();
    const FxVec3 up = fpx::FxNormalize(FxVec3{-world.gravity.x, -world.gravity.y, -world.gravity.z});
    const fx gmag = FxLength(world.gravity);
    const SurfaceProbe probe = FluidSurfaceY(world.particles);   // the pool surface, ONCE per step
    if (probe.samples == 0u) return;                             // no dynamic water -> nothing to exchange
    for (uint32_t i = 0; i < bodyCount; ++i) {
        fpx::FxBody& b = world.bodies[(size_t)i];
        if (!(b.flags & fpx::kFlagDynamic)) continue;       // static/kinematic -> untouched
        if (CountBodyWaterContacts(world, query, i) == 0u) continue;   // clear of the water -> free-fall

        // BUOYANCY: F = ρ_f · |g| · V_submerged along up; Δv = F · invMass · dt.
        if (params.rhoFluid > 0) {
            const fx d = SubmergedDepth(b, probe.surfaceY);
            const fx vol = SphereCapVolume(b.radius, d);
            const fx buoyMag = fxmul(fxmul(params.rhoFluid, gmag), vol);
            const FxVec3 dvel = FxScale(FxScale(FxScale(up, buoyMag), b.invMass), dt);
            b.vel = FxAdd(b.vel, dvel);
        }

        // DRAG: Δv = drag · (vFluidAvg − vel) · dt (mass-independent relaxation; the fixed-order int64
        // mean over the NON-STATIC gathered particles — the CP2 sum discipline, statics excluded).
        if (params.drag > 0) {
            int64_t sumX = 0, sumY = 0, sumZ = 0, cnt = 0;
            const uint32_t s0 = query.bodyStart[(size_t)i];
            const uint32_t s1 = query.bodyStart[(size_t)i + 1u];
            for (uint32_t s = s0; s < s1; ++s) {
                const fluid::FluidParticle& p = world.particles[(size_t)query.bodyParticles[(size_t)s]];
                if (p.flags & fluid::kFlagStatic) continue;
                sumX += (int64_t)p.vel.x; sumY += (int64_t)p.vel.y; sumZ += (int64_t)p.vel.z;
                ++cnt;
            }
            if (cnt > 0) {
                const FxVec3 vAvg{(fx)(sumX / cnt), (fx)(sumY / cnt), (fx)(sumZ / cnt)};
                const FxVec3 dv = FxSub(vAvg, b.vel);
                b.vel.x += fxmul(fxmul(params.drag, dv.x), dt);
                b.vel.y += fxmul(fxmul(params.drag, dv.y), dt);
                b.vel.z += fxmul(fxmul(params.drag, dv.z), dt);
            }
        }
    }
}

// ----- StepCoupleV2: the composed V2 tick (Archimedes buoyancy swapped in + the sealed containment) -------
// params.legacy != 0 -> VERBATIM delegation to the frozen CP4 StepCouple (byte-identical by construction
// — the identity flag; basin/params ignored). Otherwise the CP4 locked order with the buoyancy exchange
// moved OUT of the K iterations (run ONCE, post the fluid velocity derivation — the GF4 lesson) and the
// seal appended:
//   (1) PREDICT:  fluid IntegrateFluid; each DYNAMIC body vel += gravity·dt (velocity only)   [CP4 (1)]
//   (2) BUILD:    FL2 grid/table/neighbour list + the CP1 query, ONCE per step                [CP4 (2)]
//   (3) ITERATE (K JACOBI iters): FL4 density solve -> CP3 ApplyBodyToFluid                   [CP4 (3a,3b);
//                 the old per-iteration CP2 AccumBodyForces (3c) is NOT run — the V2 swap]
//   (4) FLUID VELOCITY: vel = (pos − prev)/dt                                                 [CP4 (4)]
//   (4b) POOL VISCOSITY: FL7 StepFluidViscosity (XSPH; viscosity==0 -> EXACT no-op)           [NEW]
//   (4c) EXCHANGE: AccumBodyForcesV2 ONCE (Archimedes buoyancy + drag, over the step's query)  [NEW]
//   (5) BODY INTEGRATE: pos += vel·dt; ResolveGround                                          [CP4 (5)]
//   (6) SEAL: ClampToBasin (the hard AABB clamp — the final authority on fluid positions)     [NEW]
// The query is built at (2) from the predicted fluid + the pre-step body pos (the CP4 convention); (4b)
// reads the CURRENT particle state through those fixed indices — deterministic. Pure integer, fixed op
// order -> two runs bit-identical, cross-backend identical (pure CPU on both — the CP7 GPU convention).
inline void StepCoupleV2(CoupleWorld& world, const CoupleBasin& basin, const CoupleV2Params& params,
                         fx dt, int iters) {
    if (params.legacy != 0u) { StepCouple(world, dt, iters); return; }   // the identity flag (CP4 verbatim)
    const size_t n = world.particles.size();
    // (1) PREDICT.
    fluid::IntegrateFluid(world.particles, world.gravity, dt, world.groundY);
    for (fpx::FxBody& b : world.bodies) {
        if (!(b.flags & fpx::kFlagDynamic)) continue;
        b.vel.x += fpx::fxmul(world.gravity.x, dt);
        b.vel.y += fpx::fxmul(world.gravity.y, dt);
        b.vel.z += fpx::fxmul(world.gravity.z, dt);
    }
    // (2) BUILD.
    const fluid::FluidGrid grid = fluid::MakeGrid(world.particles, world.kernel.h);
    const fluid::FluidCellTable table = fluid::BuildCellTable(world.particles, grid);
    const fluid::FluidNeighborList list = fluid::BuildNeighborList(world.particles, grid, table, world.kernel.h);
    const CoupleQuery query = GatherBodyParticles(world);
    // (3) K JACOBI iterations: FL4 density solve -> CP3 body->fluid (NO per-iteration buoyancy — V2).
    std::vector<fx> density, lambda;
    std::vector<FxVec3> dp;
    for (int it = 0; it < iters; ++it) {
        fluid::ComputeDensity(world.particles, list, world.kernel, density);
        fluid::ComputeLambda(world.particles, list, world.kernel, density, lambda);
        fluid::SolveDensityConstraint(world.particles, list, world.kernel, lambda, dp);
        for (size_t i = 0; i < n; ++i) {
            if (world.particles[i].flags & fluid::kFlagStatic) continue;
            world.particles[i].pos = FxAdd(world.particles[i].pos, dp[i]);
        }
        ApplyBodyToFluid(world, dt);
    }
    // (4) FLUID VELOCITY: vel = (pos − prev)/dt.
    if (dt != 0) {
        for (size_t i = 0; i < n; ++i) {
            if (world.particles[i].flags & fluid::kFlagStatic) continue;
            const FxVec3 dpos = FxSub(world.particles[i].pos, world.particles[i].prev);
            world.particles[i].vel =
                FxVec3{fpx::fxdiv(dpos.x, dt), fpx::fxdiv(dpos.y, dt), fpx::fxdiv(dpos.z, dt)};
        }
    }
    // (4b) POOL VISCOSITY: the FL7 XSPH smooth (fluid.h::StepFluidViscosity, reused AS-IS — its own
    // neighbour build + the two Jacobi passes + the (vel, prev) re-encode; params.viscosity == 0 -> the
    // FL7 identity-at-zero EXACT no-op). Calms the PBF slosh so the Archimedes float line settles.
    fluid::StepFluidViscosity(world.particles, world.kernel, params.viscosity, dt);
    // (4c) THE V2 EXCHANGE: Archimedes buoyancy + drag, ONCE per step (post the smoothed velocities).
    AccumBodyForcesV2(world, query, params, dt);
    // (5) BODY INTEGRATE + the radius-aware bed clamp.
    for (fpx::FxBody& b : world.bodies) {
        if (!(b.flags & fpx::kFlagDynamic)) continue;
        b.pos.x += fpx::fxmul(b.vel.x, dt);
        b.pos.y += fpx::fxmul(b.vel.y, dt);
        b.pos.z += fpx::fxmul(b.vel.z, dt);
        fpx::ResolveGround(b, world.groundY);
    }
    // (6) THE SEAL: the hard basin clamp (zero particles outside, by construction).
    ClampToBasin(world.particles, basin);
}

// StepCoupleV2Steps(world, basin, params, dt, iters, steps): run K V2 ticks (the showcase/test driver).
inline void StepCoupleV2Steps(CoupleWorld& world, const CoupleBasin& basin, const CoupleV2Params& params,
                              fx dt, int iters, int steps) {
    for (int s = 0; s < steps; ++s) StepCoupleV2(world, basin, params, dt, iters);
}

// MeasureSubmergedDepth(world, bodyIndex): the reporting stat — the body's SubmergedDepth against the
// FluidSurfaceY pool-surface estimate, gated by water contact (0 if the body gathers no water, exactly
// like the force path). Deterministic Q16.16; the Archimedes proof pins this against the analytic depth
// within the honest band.
inline fx MeasureSubmergedDepth(const CoupleWorld& world, uint32_t bodyIndex) {
    if (bodyIndex >= (uint32_t)world.bodies.size()) return 0;
    const CoupleQuery query = GatherBodyParticles(world);
    if (CountBodyWaterContacts(world, query, bodyIndex) == 0u) return 0;
    const SurfaceProbe probe = FluidSurfaceY(world.particles);
    if (probe.samples == 0u) return 0;
    return SubmergedDepth(world.bodies[(size_t)bodyIndex], probe.surfaceY);
}

// ----- CP7 LOCKSTEP: the CP5 mold with the step fn swapped (commands/snapshot reused VERBATIM) ------------
// SimCoupleTickV2: apply the tick's commands in ARRAY ORDER (ApplyCoupleCommand — CP5 verbatim), then ONE
// StepCoupleV2. RunCoupleLockstepV2: run `ticks` V2 ticks from a COPY of init — authority and replica fed
// the SAME init + stream re-derive the coupled state (bodies AND fluid) bit-for-bit.
inline void SimCoupleTickV2(CoupleWorld& world, const CoupleBasin& basin, const CoupleV2Params& params,
                            const std::vector<CoupleCommand>& stream, uint32_t tick, fx dt, int iters) {
    for (const CoupleCommand& c : stream)
        if (c.tick == tick) ApplyCoupleCommand(world, c);
    StepCoupleV2(world, basin, params, dt, iters);
}

inline CoupleWorld RunCoupleLockstepV2(const CoupleWorld& init, const CoupleBasin& basin,
                                       const CoupleV2Params& params,
                                       const std::vector<CoupleCommand>& stream, int ticks, fx dt,
                                       int iters) {
    CoupleWorld world = init;
    for (int t = 0; t < ticks; ++t)
        SimCoupleTickV2(world, basin, params, stream, (uint32_t)t, dt, iters);
    return world;
}

// ----- CoupleDigest: the deterministic FNV-1a-64 digest of the whole coupled state (the pin) --------------
// Hashes every body's int32 words FIELD-WISE (pos/vel/invMass/flags/radius/orient/angVel) then every
// fluid particle's (pos/prev/vel/invMass/flags) — explicit byte shifts, NO reinterpret_cast, so the
// digest is layout/padding-independent and identical on every compiler/platform (the fluid.h FluidDigest
// mold over the TWO heterogeneous body sets). The golden-discipline pin the CP7 tests + showcase print.
inline uint64_t CoupleDigest(const CoupleWorld& world) {
    uint64_t h = 1469598103934665603ull;                    // FNV-1a 64 offset basis
    auto mix = [&h](uint32_t v) {
        for (int b = 0; b < 4; ++b) {
            h ^= (uint64_t)((v >> (b * 8)) & 0xFFu);
            h *= 1099511628211ull;                          // FNV-1a 64 prime
        }
    };
    for (const fpx::FxBody& b : world.bodies) {
        mix((uint32_t)b.pos.x);    mix((uint32_t)b.pos.y);    mix((uint32_t)b.pos.z);
        mix((uint32_t)b.vel.x);    mix((uint32_t)b.vel.y);    mix((uint32_t)b.vel.z);
        mix((uint32_t)b.invMass);  mix(b.flags);              mix((uint32_t)b.radius);
        mix((uint32_t)b.orient.x); mix((uint32_t)b.orient.y); mix((uint32_t)b.orient.z);
        mix((uint32_t)b.orient.w);
        mix((uint32_t)b.angVel.x); mix((uint32_t)b.angVel.y); mix((uint32_t)b.angVel.z);
    }
    for (const fluid::FluidParticle& p : world.particles) {
        mix((uint32_t)p.pos.x);  mix((uint32_t)p.pos.y);  mix((uint32_t)p.pos.z);
        mix((uint32_t)p.prev.x); mix((uint32_t)p.prev.y); mix((uint32_t)p.prev.z);
        mix((uint32_t)p.vel.x);  mix((uint32_t)p.vel.y);  mix((uint32_t)p.vel.z);
        mix((uint32_t)p.invMass); mix(p.flags);
    }
    return h;
}

}  // namespace couple
}  // namespace hf::sim
