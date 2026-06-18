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

}  // namespace cgrain
}  // namespace hf::sim
