#pragma once
// Slice GF1 — Deterministic Grain<->Fluid Coupling: the UNIFIED TWO-POOL WORLD + the SHARED-GRID CROSS QUERY
// (the BEACHHEAD of FLAGSHIP #13: DETERMINISTIC TWO-WAY GRAIN<->FLUID COUPLING, hf::sim::cgf). The THIRD
// material-interaction pairing — and the FIRST particle<->particle coupling (vs the two rigid<->particle ones:
// rigid<->fluid CP, rigid<->grain CG). Two emergent materials co-resident in ONE Q16.16 world: a frictional
// sand bed (engine/sim/grain.h) + an incompressible PBF fluid (engine/sim/fluid.h) — WET SAND / MUD / SLURRY.
// GF1 is ONLY the unified two-pool world + the cross-pool neighbour QUERY (each grain's nearby fluid particles
// AND each fluid particle's nearby grains) — the link, NO exchange yet (GF2 buoyancy/seepage, GF3 contact
// reaction, GF4 the coupled step, GF5 lockstep, GF6 render). The GR2/FL2 27-cell-stencil grid-hash neighbour
// search applied CROSS-POOL. Pure CPU, header-only, NO device, NO backend symbols, NO <cmath>. Namespace
// hf::sim::cgf. couple_gf is the NEW additive sibling #include-ing grain.h + fluid.h read-only.
//
// THE GR2/FL2 TWIN (the one new shape is ONE SHARED grid over BOTH pools + a CROSS query): GR2's
// BuildGrainNeighborList finds, per GRAIN, the grains in its 27-cell stencil over the GR2 grain cell table;
// FL2's BuildNeighborList finds, per FLUID, the fluid particles in its 27-cell stencil over the FL2 fluid cell
// table. GF1's two CROSS lists find, per GRAIN, the FLUID particles in its 27-cell stencil over the FLUID cell
// table (gfNeighbors), AND per FLUID, the GRAINS in its 27-cell stencil over the GRAIN cell table (fgNeighbors)
// — the query pool scans the OTHER pool's cell table, over ONE shared grid so both cells are in the same
// coordinate frame. Built by count->scan->emit (CSR gfStart[grainCount+1] + gfNeighbors[], grouped by grain;
// CSR fgStart[fluidCount+1] + fgNeighbors[], grouped by fluid — the GR2/FL2 EMIT-order discipline, fully
// deterministic). Pure int32. There is NO j==i self-skip: the two pools are DISTINCT (a grain index and a
// fluid index never collide), so every cross-pool candidate that passes the box reject is emitted.
//
// THE int32 DECISION (the FL2/GR2/CP1/CG1 precedent): the cross-pool query is integer index arithmetic + the
// per-axis |dx| < h box reject (fx is int32 -> a PURE INT32 compare, NO products, NO int64, NO sqrt). So the
// GPU cgf_gf_{count,scan,emit}.comp + cgf_fg_{count,scan,emit}.comp shaders MSL-generate NATIVELY -> a TRUE GPU
// pass on both Vulkan AND Metal (the strongest cross-vendor proof, like GR2/FL2 — strict zero-differing-pixel).
// The exact radial cull (the exchange force/contact is 0 beyond h) is DEFERRED to GF2/GF3, so the
// over-inclusive box candidate is correct — exactly the FL2/GR2 "over-inclusive box, exact cull deferred"
// discipline.
//
// REUSE MAP (file:line — read-only, NOT modified; couple_gf is the additive sibling):
//   * engine/sim/grain.h: GrainParticle (grain.h:85-92), GrainGrid/MakeGrainGrid/GrainCellOf/FlatGrainCellId/
//     GrainCellCount (GR2, grain.h:238-260), GrainCellTable/BuildGrainCellTable (GR2, grain.h:296-328),
//     GrainNeighborAccept (the per-axis box reject, grain.h:335). The GR2 grain_cell_{count,scan,emit}.comp
//     shaders (already MSL-native) are REUSED for the grain cell table build (sized to the shared grid).
//   * engine/sim/fluid.h: FluidParticle (fluid.h:82-88), FluidGrid/MakeGrid/CellOf/FlatCellId/CellCount (FL2,
//     fluid.h:226-249), FluidCellTable/BuildCellTable (FL2, fluid.h:283-315), NeighborAccept (the per-axis box
//     reject, fluid.h:321). The FL2 fluid_cell_{count,scan,emit}.comp shaders (already MSL-native) are REUSED
//     for the fluid cell table build (sized to the shared grid).
//   * engine/sim/fpx.h: fx / FxVec3 / FloorDiv / FxCell / CellId. DO NOT modify fpx.h/grain.h/fluid.h/cloth.h/
//     couple.h/couple_grain.h — couple_gf is the additive sibling.
//   * engine/sim/couple_grain.h: CGrainWorld + CGrainQuery{bodyStart, bodyGrains} — the additive-sibling
//     unified-world pattern + the per-query CSR. GF1's CGFWorld + CGFNeighbors{gfStart, gfNeighbors, fgStart,
//     fgNeighbors} are the SAME shape, two PARTICLE pools.

#include <cstdint>
#include <vector>

#include "sim/fpx.h"     // read-only: fx / FxVec3 / FloorDiv / FxCell / CellId
#include "sim/grain.h"   // read-only: GrainParticle / GrainGrid / GrainCellTable / GrainNeighborAccept (GR2)
#include "sim/fluid.h"   // read-only: FluidParticle / FluidGrid / FluidCellTable / NeighborAccept (FL2)

namespace hf::sim {
namespace cgf {

// Reuse the fpx Q16.16 scalar + vector toolbox verbatim (NO new fixed-point primitives).
using fpx::fx;
using fpx::FxVec3;
using fpx::FxCell;
using fpx::FloorDiv;
inline constexpr int kFrac = fpx::kFrac;   // Q16.16 fractional bits (== fpx::kFrac, MUST match the shader)
inline constexpr fx  kOne  = fpx::kOne;    // 1.0 in Q16.16 (65536)

// ----- The unified two-pool world (a grain pool + a fluid pool in one Q16.16 frame) ------------------------
// Both pools share the SAME world units (both #include sim/fpx.h; FluidParticle and GrainParticle are
// near-identical std430 packings). GF1 needs `grains` + `fluid` + the coupling cell-size `h`; gravity/dt/
// groundY are carried for GF2-GF6 (the coupled step + lockstep).
struct CGFWorld {
    std::vector<grain::GrainParticle>  grains;       // the granular pool (the GR sim members)
    std::vector<fluid::FluidParticle>  fluid;        // the fluid pool (the FL sim members)
    FxVec3                             gravity;      // carried for GF2-GF6 (the coupled step)
    fx                                 dt = 0;       // carried for GF4-GF6 (the coupled step / lockstep)
    fx                                 groundY = 0;  // carried for GF2-GF6 (the ground clamp)
    fx                                 h = 0;        // GF1: the shared coupling cell-size (== the search radius)
};

// ----- The shared grid over BOTH pools' union AABB (the one new shape — two pools, ONE grid) ---------------
// CGFGrid is structurally identical to GrainGrid / FluidGrid (FloorDiv cell coords + CellId linearization),
// but its cell bounds cover the UNION of both pools' cell ranges, so a grain's cell AND a fluid's cell land in
// [0,gridDim) under the SAME cellMin/gridDim -> the CellId linearization is total + collision-free over BOTH
// pools (pin this edge case). Cell-size h is the coupling radius.
struct CGFGrid {
    fx     h = 0;        // Q16.16 cell size (== the coupling search radius)
    FxCell cellMin;      // the integer cell coord of the grid's (0,0,0) corner (the union lower cell)
    FxCell gridDim;      // the grid extent in cells per axis (cellCount = x*y*z)
};

// CGFCellOf(pos, h): the integer grid cell a position falls in, FloorDiv per axis. Pure int32. (== the GR2
// GrainCellOf / FL2 CellOf — the SAME quantization for BOTH pools, which is what makes the shared grid total.)
inline FxCell CGFCellOf(const FxVec3& pos, fx h) {
    return FxCell{FloorDiv(pos.x, h), FloorDiv(pos.y, h), FloorDiv(pos.z, h)};
}

// CGFCellCount(grid): the total number of cells in the shared dense grid (gridDim.x * y * z).
inline uint32_t CGFCellCount(const CGFGrid& grid) {
    return (uint32_t)(grid.gridDim.x * grid.gridDim.y * grid.gridDim.z);
}

// MakeCGFGrid(world): the shared bounded dense grid covering the UNION of both pools' cell ranges at cell-size
// world.h. cellMin = the per-axis min cell over ALL grains AND ALL fluid; gridDim = (maxCell - minCell + 1) per
// axis. BOTH pools empty -> a 1x1x1 grid at origin (deterministic degenerate). Either pool empty -> the grid
// covers the OTHER pool's bounds. Pure int32. (The union of the two MakeGrainGrid/MakeGrid cell bounds — the
// edge case the spec pins: the linearization stays total + collision-free over BOTH pools.)
inline CGFGrid MakeCGFGrid(const CGFWorld& world) {
    CGFGrid grid;
    grid.h = world.h;
    if (world.grains.empty() && world.fluid.empty()) {
        grid.cellMin = FxCell{0, 0, 0};
        grid.gridDim = FxCell{1, 1, 1};
        return grid;
    }
    bool seeded = false;
    FxCell lo{0, 0, 0}, hi{0, 0, 0};
    auto fold = [&](const FxVec3& pos) {
        const FxCell c = CGFCellOf(pos, world.h);
        if (!seeded) { lo = c; hi = c; seeded = true; return; }
        if (c.x < lo.x) lo.x = c.x; if (c.x > hi.x) hi.x = c.x;
        if (c.y < lo.y) lo.y = c.y; if (c.y > hi.y) hi.y = c.y;
        if (c.z < lo.z) lo.z = c.z; if (c.z > hi.z) hi.z = c.z;
    };
    for (const grain::GrainParticle& g : world.grains) fold(g.pos);
    for (const fluid::FluidParticle& f : world.fluid)  fold(f.pos);
    grid.cellMin = lo;
    grid.gridDim = FxCell{hi.x - lo.x + 1, hi.y - lo.y + 1, hi.z - lo.z + 1};
    return grid;
}

// FlatCGFCellId(cell, grid): the flat id of an absolute cell coord into the shared dense grid (offset by
// cellMin into [0,gridDim), then the CellId linearization). The caller guarantees the cell is in range (the
// grid was sized to BOTH pools); returns the linear cell index in [0, gridDim.x*y*z).
inline uint32_t FlatCGFCellId(const FxCell& cell, const CGFGrid& grid) {
    const FxCell local{cell.x - grid.cellMin.x, cell.y - grid.cellMin.y, cell.z - grid.cellMin.z};
    return fpx::CellId(local, grid.gridDim);
}

// ----- The grain / fluid cell tables over the SHARED grid (reuse GR2/FL2 by adapting their grids) ----------
// GF1 buckets each pool into its OWN cell table over the SHARED grid. The GR2 BuildGrainCellTable / FL2
// BuildCellTable take a GrainGrid / FluidGrid; we build those FROM the shared CGFGrid (same h/cellMin/gridDim)
// so both tables are linearized in the SAME coordinate frame. This is the EXACT reuse the GPU does: the reused
// grain_cell_{count,scan,emit} / fluid_cell_{count,scan,emit} passes run with the SHARED grid's
// hSearch/cellMin/gridDim in their params.
inline grain::GrainGrid GrainGridFromCGF(const CGFGrid& g) {
    grain::GrainGrid out; out.hSearch = g.h; out.cellMin = g.cellMin; out.gridDim = g.gridDim; return out;
}
inline fluid::FluidGrid FluidGridFromCGF(const CGFGrid& g) {
    fluid::FluidGrid out; out.h = g.h; out.cellMin = g.cellMin; out.gridDim = g.gridDim; return out;
}

// ----- The CSR cross-pool query result (the two cross lists) -----------------------------------------------
// gfStart[i..] is the exclusive prefix-sum of per-grain FLUID-neighbour counts (grainCount+1 entries;
// gfStart[i]..gfStart[i+1] is grain i's slice), gfNeighbors[] holds the gathered FLUID indices grouped by
// grain, ASCENDING stencil-cell then ascending fluid index. fgStart/fgNeighbors are the mirror, per FLUID over
// the GRAIN cell table. The GPU cgf_gf_*/cgf_fg_* mirror these byte-for-byte.
struct CGFNeighbors {
    std::vector<uint32_t> gfStart;       // grainCount+1 exclusive prefix-sum offsets (CSR, grain -> fluid)
    std::vector<uint32_t> gfNeighbors;   // gathered FLUID indices grouped by grain (stencil order)
    std::vector<uint32_t> fgStart;       // fluidCount+1 exclusive prefix-sum offsets (CSR, fluid -> grain)
    std::vector<uint32_t> fgNeighbors;   // gathered GRAIN indices grouped by fluid (stencil order)
};

// ----- The cross-pool count (per query particle, the 27-cell stencil over the TARGET pool's cell table) ----
// CountCross<QueryPos, TargetPos>: for each query particle i, scan the 27 cells of its 3x3x3 stencil over the
// SHARED grid; for each TARGET particle j in those cells (read from the target cell table), accept iff the
// per-axis |query_i.axis - target_j.axis| < h box reject passes (== GrainNeighborAccept / NeighborAccept). NO
// j==i self-skip (the pools are distinct). perQueryOut[i] = #accepted; returns the total. The GPU
// cgf_gf_count / cgf_fg_count mirror THIS per-thread (one thread per query particle i). Pure int32.
template <typename QueryVec, typename TargetVec, typename QueryArr, typename TargetArr>
inline uint32_t CountCross(const QueryArr& query, const TargetArr& target, const CGFGrid& grid,
                           const std::vector<uint32_t>& cellStart, const std::vector<uint32_t>& cellTargets,
                           std::vector<uint32_t>& perQueryOut,
                           QueryVec qpos, TargetVec tpos) {
    const uint32_t n = (uint32_t)query.size();
    const fx h = grid.h;
    perQueryOut.assign((size_t)n, 0u);
    uint32_t total = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const FxVec3 pi = qpos(query[(size_t)i]);
        const FxCell ci = CGFCellOf(pi, h);
        uint32_t c = 0;
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const FxCell nc{ci.x + dx, ci.y + dy, ci.z + dz};
            if (nc.x < grid.cellMin.x || nc.x >= grid.cellMin.x + grid.gridDim.x) continue;
            if (nc.y < grid.cellMin.y || nc.y >= grid.cellMin.y + grid.gridDim.y) continue;
            if (nc.z < grid.cellMin.z || nc.z >= grid.cellMin.z + grid.gridDim.z) continue;
            const uint32_t cell = FlatCGFCellId(nc, grid);
            for (uint32_t s = cellStart[cell]; s < cellStart[cell + 1u]; ++s) {
                const uint32_t j = cellTargets[s];
                const FxVec3 pj = tpos(target[(size_t)j]);
                // The per-axis box reject (== GrainNeighborAccept / fluid::NeighborAccept), PURE INT32.
                fx ax = pi.x - pj.x; if (ax < 0) ax = -ax;
                fx ay = pi.y - pj.y; if (ay < 0) ay = -ay;
                fx az = pi.z - pj.z; if (az < 0) az = -az;
                if (ax < h && ay < h && az < h) ++c;
            }
        }
        perQueryOut[(size_t)i] = c;
        total += c;
    }
    return total;
}

// ----- The cross-pool emit (per query particle, into its DISJOINT CSR slice) -------------------------------
// EmitCross: re-scan the SAME 27-cell stencil in the SAME fixed order, read the write base start[i] (the
// prefix-sum), and EMIT each accepted target index j into neighbors[] at start[i] + local++. Each query
// particle writes into its OWN DISJOINT slice -> race-free, NO atomics. Order: ascending stencil-cell
// (dz,dy,dx) then ascending j (the target cell table is ascending-index per cell) -> deterministic. The GPU
// cgf_gf_emit / cgf_fg_emit mirror THIS. Pure int32.
template <typename QueryVec, typename TargetVec, typename QueryArr, typename TargetArr>
inline void EmitCross(const QueryArr& query, const TargetArr& target, const CGFGrid& grid,
                      const std::vector<uint32_t>& cellStart, const std::vector<uint32_t>& cellTargets,
                      const std::vector<uint32_t>& start, std::vector<uint32_t>& neighbors,
                      QueryVec qpos, TargetVec tpos) {
    const uint32_t n = (uint32_t)query.size();
    const fx h = grid.h;
    for (uint32_t i = 0; i < n; ++i) {
        const FxVec3 pi = qpos(query[(size_t)i]);
        const FxCell ci = CGFCellOf(pi, h);
        uint32_t base  = start[(size_t)i];
        uint32_t local = 0u;
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const FxCell nc{ci.x + dx, ci.y + dy, ci.z + dz};
            if (nc.x < grid.cellMin.x || nc.x >= grid.cellMin.x + grid.gridDim.x) continue;
            if (nc.y < grid.cellMin.y || nc.y >= grid.cellMin.y + grid.gridDim.y) continue;
            if (nc.z < grid.cellMin.z || nc.z >= grid.cellMin.z + grid.gridDim.z) continue;
            const uint32_t cell = FlatCGFCellId(nc, grid);
            for (uint32_t s = cellStart[cell]; s < cellStart[cell + 1u]; ++s) {
                const uint32_t j = cellTargets[s];
                const FxVec3 pj = tpos(target[(size_t)j]);
                fx ax = pi.x - pj.x; if (ax < 0) ax = -ax;
                fx ay = pi.y - pj.y; if (ay < 0) ay = -ay;
                fx az = pi.z - pj.z; if (az < 0) az = -az;
                if (ax < h && ay < h && az < h) { neighbors[(size_t)(base + local)] = j; ++local; }
            }
        }
    }
}

// Position accessors (the QueryVec/TargetVec functors — pure, return the particle's pos by value).
inline FxVec3 GrainPos(const grain::GrainParticle& g) { return g.pos; }
inline FxVec3 FluidPos(const fluid::FluidParticle& f) { return f.pos; }

// ----- BuildCGFNeighbors: the two cross-pool count->scan->emit lists (the GF1 headline) --------------------
// (1) Build the SHARED grid + the grain cell table + the fluid cell table over it (reuse GR2/FL2). (2) gf:
// per grain, COUNT the fluid 27-cell-stencil candidates -> SCAN -> gfStart, then EMIT -> gfNeighbors. (3) fg:
// per fluid, COUNT the grain 27-cell-stencil candidates -> SCAN -> fgStart, then EMIT -> fgNeighbors. Both
// pure int32, both grouped by the query particle ascending in the fixed stencil order -> deterministic. The
// GPU cgf_gf_* (grain->fluid) + cgf_fg_* (fluid->grain) mirror this byte-for-byte. (DET-CRUX, the GR2/FL2
// lesson: the per-query EMIT scatter is fixed ascending order; the count + the per-query lists are
// per-query-disjoint -> race-free. The reused cell-EMIT is single-thread ascending, already correct in
// GR2/FL2.)
inline CGFNeighbors BuildCGFNeighbors(const CGFWorld& world) {
    const uint32_t grainCount = (uint32_t)world.grains.size();
    const uint32_t fluidCount = (uint32_t)world.fluid.size();
    CGFNeighbors out;

    // (1) the SHARED grid + the two cell tables (reuse GR2/FL2 over the shared grid).
    const CGFGrid grid = MakeCGFGrid(world);
    const grain::GrainGrid gGrid = GrainGridFromCGF(grid);
    const fluid::FluidGrid fGrid = FluidGridFromCGF(grid);
    const grain::GrainCellTable gTable = grain::BuildGrainCellTable(world.grains, gGrid);
    const fluid::FluidCellTable fTable = fluid::BuildCellTable(world.fluid, fGrid);

    // (2) gf (grain -> fluid): per grain, the FLUID cell table 27-cell stencil, |dx|<h accept.
    {
        std::vector<uint32_t> counts;
        const uint32_t total = CountCross(world.grains, world.fluid, grid,
            fTable.cellStart, fTable.cellParticles, counts, &GrainPos, &FluidPos);
        out.gfStart.assign((size_t)grainCount + 1u, 0u);
        uint32_t running = 0;
        for (uint32_t i = 0; i < grainCount; ++i) { out.gfStart[(size_t)i] = running; running += counts[(size_t)i]; }
        out.gfStart[(size_t)grainCount] = running;   // sentinel == total
        out.gfNeighbors.assign((size_t)total, 0u);
        EmitCross(world.grains, world.fluid, grid, fTable.cellStart, fTable.cellParticles,
                  out.gfStart, out.gfNeighbors, &GrainPos, &FluidPos);
    }

    // (3) fg (fluid -> grain): per fluid, the GRAIN cell table 27-cell stencil, |dx|<h accept.
    {
        std::vector<uint32_t> counts;
        const uint32_t total = CountCross(world.fluid, world.grains, grid,
            gTable.cellStart, gTable.cellGrains, counts, &FluidPos, &GrainPos);
        out.fgStart.assign((size_t)fluidCount + 1u, 0u);
        uint32_t running = 0;
        for (uint32_t i = 0; i < fluidCount; ++i) { out.fgStart[(size_t)i] = running; running += counts[(size_t)i]; }
        out.fgStart[(size_t)fluidCount] = running;   // sentinel == total
        out.fgNeighbors.assign((size_t)total, 0u);
        EmitCross(world.fluid, world.grains, grid, gTable.cellStart, gTable.cellGrains,
                  out.fgStart, out.fgNeighbors, &FluidPos, &GrainPos);
    }

    return out;
}

// CountGF(nbr): the total grain->fluid cross pairs (== gfNeighbors.size() == gfStart.back()). Pure integer.
inline uint32_t CountGF(const CGFNeighbors& nbr) {
    return nbr.gfStart.empty() ? 0u : nbr.gfStart.back();
}

// CountFG(nbr): the total fluid->grain cross pairs (== fgNeighbors.size() == fgStart.back()). Pure integer.
inline uint32_t CountFG(const CGFNeighbors& nbr) {
    return nbr.fgStart.empty() ? 0u : nbr.fgStart.back();
}

// MaxGFPerGrain(nbr): the largest per-grain fluid-neighbour count (a reporting/stat helper). Pure integer.
inline uint32_t MaxGFPerGrain(const CGFNeighbors& nbr) {
    uint32_t m = 0;
    for (size_t i = 0; i + 1 < nbr.gfStart.size(); ++i) {
        const uint32_t c = nbr.gfStart[i + 1] - nbr.gfStart[i];
        if (c > m) m = c;
    }
    return m;
}

// MaxFGPerFluid(nbr): the largest per-fluid grain-neighbour count (a reporting/stat helper). Pure integer.
inline uint32_t MaxFGPerFluid(const CGFNeighbors& nbr) {
    uint32_t m = 0;
    for (size_t i = 0; i + 1 < nbr.fgStart.size(); ++i) {
        const uint32_t c = nbr.fgStart[i + 1] - nbr.fgStart[i];
        if (c > m) m = c;
    }
    return m;
}

}  // namespace cgf
}  // namespace hf::sim
