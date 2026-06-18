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

}  // namespace couple
}  // namespace hf::sim
