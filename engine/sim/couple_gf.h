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

// ===== Slice GF2 — BUOYANCY / SEEPAGE (fluid->grain, the CRUX; the 2nd slice of FLAGSHIP #13) =========
// THE FIRST MOMENTUM EXCHANGE of FLAGSHIP #13: each grain sums, over its GF1 fluid-neighbour list (in the
// FIXED GF1 gfNeighbors emit order, ascending), a BUOYANT impulse (∝ the surrounding fluid count, opposing
// gravity — the submerging fluid lifts it) + a DRAG impulse (toward the local fluid velocity, which GF2 damps
// since the fluid is held STATIC), so SUBMERGED grains LIGHTEN / FLUIDIZE (a wet/dry contrast) — the sand
// under the fluid loosens toward slurry, while DRY grains pack normally. ONE-WAY for now (fluid -> grain; the
// grain->fluid reaction is GF3). LINEAR only — grains are points, no torque. The CG2 contact-support reduction
// twin, with a COUNT-based buoyancy (the fluid-submersion proxy) instead of a penetration-based support, and
// per-PARTICLE (one thread per grain over its OWN short fluid list) instead of per-body.
//
// THE int64 DECISION (the CG2/GR3/FL4 split): the buoyancy/drag math is int64 (FxNormalize/FxScale/fxmul/
// fxdiv) -> shaders/cgf_buoyancy.comp is VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) + the Metal
// --cgf-buoyancy showcase runs the CPU StepCGFBuoyancy (byte-identical by construction). The GF1 cross-query
// (re-run each step) stays int32 MSL-native.
//
// THE CRUX — but EASIER than CG2 (per-GRAIN, not per-BODY): each grain sums over its OWN short gfNeighbors
// fluid list (the GF1 over-inclusive box list), writing ONLY its own velocity -> per-grain-disjoint, race-free,
// NO atomics, [numthreads(64,1,1)] MULTI-THREAD over grains, NO single-thread TDR (the FL4/GR3 Jacobi pattern).
// The fixed gfNeighbors ascending order keeps the int64 sum bit-identical CPU<->shader.

using fpx::FxAdd;          // read-only: the fpx.h vector toolbox (reused verbatim, no new primitives)
using fpx::FxSub;
using fpx::FxScale;
using fpx::FxNormalize;    // read-only: the int64 normalize (+Y fallback on length 0)
using fpx::fxmul;          // read-only: the int64-intermediate Q16.16 multiply

// The host-snapped Q16.16 coupling coefficients. Tuned so SUBMERGED grains (cnt>0 fluid neighbours) LIGHTEN /
// rise / loosen (the sand fluidizes under the fluid) while DRY grains (cnt==0) pack normally — an emergent
// WET/DRY contrast, NOT an exact buoyancy depth (the CP2/GR4 within-band caveat). BUOYANCY: F_buoy ∝ the
// submerging fluid count, opposing gravity (more surrounding fluid -> more lift). DRAG: a linear damping toward
// the (static) fluid's mean velocity, so the grain settles instead of ringing.
//   kBuoyPerFluid = 0.3 (Q16.16): the per-fluid-neighbour lift. A submerged grain has MANY fluid neighbours
//     (a packed bed under a fluid block gathers ~tens-120), so a SMALL per-neighbour lift is the right scale:
//     a grain near the fluid surface (fewer neighbours) feels just enough lift to LIGHTEN/rise, but as it
//     rises it LOSES neighbours -> the lift drops -> it stabilises at a buoyed line where buoyancy ≈ gravity
//     (a SELF-LIMITING suspension), NOT a runaway launch. Host-snapped, tuned (with kDrag) for a clear stable
//     wet>dry margin (~2 units) with the wet grains bounded a few units up, never flying out.
//   kDrag = 4.0 (Q16.16): a firm linear damping toward the static fluid (vFluidAvg≈0) so the rising wet grains
//     hit a low terminal velocity and settle loose at the buoyed line instead of overshooting/ringing — the
//     stabiliser that keeps the self-limiting suspension from oscillating.
inline constexpr fx kBuoyPerFluid = (fx)(0.3 * (double)kOne + 0.5);   // 19661 (Q16.16, ~0.3 per-fluid lift)
inline constexpr fx kDrag         = (fx)(4.0 * (double)kOne + 0.5);   // 262144 (Q16.16, ~4.0 damping)

// AccumGrainBuoyancy(world, neighbors, buoyPerFluid): the per-grain buoyancy+drag impulse accumulate (the
// per-PARTICLE Jacobi reduction). For each grain i (static skipped), over its GF1 fluid-neighbour list
// gfNeighbors[gfStart[i]..gfStart[i+1]) in ASCENDING order: cnt = the count; cnt==0 -> dry, skip (free GR
// sim). BUOYANCY F_buoy = FxScale(up, fxmul(buoyPerFluid, cnt<<kFrac)) with up = -FxNormalize(gravity). DRAG
// vFluidAvg = (Σ_j fluid[j].vel)/cnt (the fixed-order int64 sum / the integer count); F_drag =
// fxmul(kDrag, vFluidAvg - grain.vel) per axis. Apply grain.vel += FxScale(F_buoy + F_drag, grain.invMass)·dt
// (linear). int64. cgf_buoyancy.comp copies THIS body VERBATIM (one thread per grain i). Deterministic (the
// fixed gfNeighbors order, fixed op order) -> bit-identical to the GPU memcmp + two-run byte-identical. The
// buoyPerFluid coefficient is a parameter so the showcase's buoy=0 control reuses this exact body.
inline void AccumGrainBuoyancy(CGFWorld& world, const CGFNeighbors& neighbors, fx buoyPerFluid) {
    const uint32_t grainCount = (uint32_t)world.grains.size();
    const fx dt = world.dt;
    // up = -normalize(gravity) (the buoyant direction, opposing gravity). Computed ONCE (shared by all grains).
    const FxVec3 g = world.gravity;
    const FxVec3 gn = FxNormalize(g);
    const FxVec3 up{-gn.x, -gn.y, -gn.z};
    for (uint32_t i = 0; i < grainCount; ++i) {
        grain::GrainParticle& gi = world.grains[(size_t)i];
        if (gi.flags & grain::kFlagStatic) continue;   // static boundary grain -> untouched (the pinned case)
        const uint32_t s0 = neighbors.gfStart[(size_t)i];
        const uint32_t s1 = neighbors.gfStart[(size_t)i + 1u];
        const uint32_t cnt = s1 - s0;
        if (cnt == 0u) continue;                        // dry grain (no fluid neighbours) -> free GR sim

        // BUOYANCY: F_buoy = up * (buoyPerFluid * (cnt in Q16.16)) — ∝ the submerging fluid count, opposing
        // gravity. The count<<kFrac Q16.16 promotion is the CG2 precedent.
        const fx buoyMag = fpx::fxmul(buoyPerFluid, (fx)(cnt << kFrac));
        const FxVec3 fBuoy = FxScale(up, buoyMag);
        // DRAG: vFluidAvg = (Σ_j fluid[j].vel) / cnt, the fixed-order int64 sum / the integer count.
        int64_t sumX = 0, sumY = 0, sumZ = 0;
        for (uint32_t s = s0; s < s1; ++s) {
            const fluid::FluidParticle& f = world.fluid[(size_t)neighbors.gfNeighbors[(size_t)s]];
            sumX += (int64_t)f.vel.x;
            sumY += (int64_t)f.vel.y;
            sumZ += (int64_t)f.vel.z;
        }
        const FxVec3 vFluidAvg{
            (fx)(sumX / (int64_t)cnt), (fx)(sumY / (int64_t)cnt), (fx)(sumZ / (int64_t)cnt)};
        // F_drag = kDrag * (vFluidAvg - grain.vel) per axis.
        const FxVec3 dv = FxSub(vFluidAvg, gi.vel);
        const FxVec3 fDrag{fpx::fxmul(kDrag, dv.x), fpx::fxmul(kDrag, dv.y), fpx::fxmul(kDrag, dv.z)};

        // Apply the impulse as a velocity delta: vel += (F_buoy + F_drag) * invMass * dt (linear only).
        const FxVec3 fTotal = FxAdd(fBuoy, fDrag);
        const FxVec3 dvel = FxScale(FxScale(fTotal, gi.invMass), dt);
        gi.vel = FxAdd(gi.vel, dvel);
    }
}

// StepCGFBuoyancy(world, dt): ONE coupled step (the driver — the fluid is held STATIC). (1) re-query the
// grain->fluid neighbour lists from the grains' CURRENT positions (GF1 BuildCGFNeighbors) -> (2) accumulate
// the buoyancy+drag velocity delta (AccumGrainBuoyancy with the default kBuoyPerFluid) -> (3) IntegrateGrains
// (the grain integrate + radius-aware ground rest, the GR1 step VERBATIM — gravity + the buoyancy-adjusted
// vel). The fluid is NOT moved (the reaction is GF3). Over K steps the SUBMERGED grains lighten/rise while
// the DRY grains pack. IntegrateGrains is copied VERBATIM by cgf_buoyancy.comp's host driver (the GPU runs
// the SAME per-step sequence).
inline void StepCGFBuoyancy(CGFWorld& world, fx dt) {
    const CGFNeighbors nbr = BuildCGFNeighbors(world);    // GF1 re-query (fluid static -> the block unchanged)
    AccumGrainBuoyancy(world, nbr, kBuoyPerFluid);        // GF2 buoyancy + drag velocity delta
    grain::IntegrateGrains(world.grains, world.gravity, dt, world.groundY);   // GR1 integrate, VERBATIM
}

// StepCGFBuoyancyControl(world, dt, buoyPerFluid): the parameterized step (the buoy=0 control reuses the
// SAME driver with buoyPerFluid=0 -> the wet and dry grains pack the same, proving buoyancy does the work).
inline void StepCGFBuoyancyControl(CGFWorld& world, fx dt, fx buoyPerFluid) {
    const CGFNeighbors nbr = BuildCGFNeighbors(world);
    AccumGrainBuoyancy(world, nbr, buoyPerFluid);
    grain::IntegrateGrains(world.grains, world.gravity, dt, world.groundY);
}

// StepCGFBuoyancySteps(world, dt, steps): run K StepCGFBuoyancy steps. The CPU reference the GPU grain state
// memcmp's against byte-for-byte. Pure integer -> two runs byte-identical, cross-backend identical.
inline void StepCGFBuoyancySteps(CGFWorld& world, fx dt, int steps) {
    for (int s = 0; s < steps; ++s) StepCGFBuoyancy(world, dt);
}

// StepCGFBuoyancyControlSteps(world, dt, buoyPerFluid, steps): K parameterized steps (the buoy=0 control).
inline void StepCGFBuoyancyControlSteps(CGFWorld& world, fx dt, fx buoyPerFluid, int steps) {
    for (int s = 0; s < steps; ++s) StepCGFBuoyancyControl(world, dt, buoyPerFluid);
}

// MeasureWetDry(world): the honest WET/DRY metric — the mean pos.y of SUBMERGED grains (cnt>0 fluid
// neighbours) vs DRY grains (cnt==0), a deterministic Q16.16 stat for the wet/dry-contrast proof. Re-queries
// GF1 from the current positions to classify each grain. 0 wet (or 0 dry) -> that mean is 0. The wet/dry
// contrast is EMERGENT/within-band (the CP2/GR4 caveat): the proof asserts wetY > dryY by a margin +
// deterministic, NOT an exact buoyancy depth.
struct WetDry {
    fx       wetY = 0;     // the mean pos.y of submerged grains (cnt>0)
    fx       dryY = 0;     // the mean pos.y of dry grains (cnt==0)
    uint32_t wet  = 0;     // the submerged-grain count
    uint32_t dry  = 0;     // the dry-grain count
};
inline WetDry MeasureWetDry(const CGFWorld& world) {
    WetDry out;
    const CGFNeighbors nbr = BuildCGFNeighbors(world);
    const uint32_t grainCount = (uint32_t)world.grains.size();
    int64_t wetSum = 0, drySum = 0;
    for (uint32_t i = 0; i < grainCount; ++i) {
        const uint32_t cnt = nbr.gfStart[(size_t)i + 1u] - nbr.gfStart[(size_t)i];
        if (cnt > 0u) { wetSum += (int64_t)world.grains[(size_t)i].pos.y; ++out.wet; }
        else          { drySum += (int64_t)world.grains[(size_t)i].pos.y; ++out.dry; }
    }
    out.wetY = out.wet == 0u ? 0 : (fx)(wetSum / (int64_t)out.wet);
    out.dryY = out.dry == 0u ? 0 : (fx)(drySum / (int64_t)out.dry);
    return out;
}

// ===== Slice GF3 — CONTACT REACTION / DISPLACEMENT (grain->fluid, Newton's 3rd law to GF2) ============
// THE SECOND HALF of FLAGSHIP #13's two-way exchange: the grains now push BACK on the fluid. Each fluid
// particle INSIDE a grain is projected out to the grain surface (the sand DISPLACES the fluid — the fluid
// sits ON / seeps AROUND the bed, not inside the grain volumes) AND receives the equal-opposite DRAG-REACTION
// impulse (the grain imparts its momentum to the surrounding fluid). Completes the two-way exchange (GF2
// fluid->grain buoyancy, GF3 grain->fluid displacement). The CG3 ApplyBodyToGrains / CP3 ApplyBodyToFluid twin,
// per-FLUID-particle over its GF1 fgNeighbors grain list, and the positional push is LITERALLY the grain.h
// CollideGrainSphere sphere-projection with each GRAIN as the sphere (surf = g.radius; fluid particles are
// points, so NO grain-radius+particle-radius — just g.radius).
//
// THE MIRROR OF GF2 (per-FLUID over the grain set): GF2's buoyancy was per-GRAIN (over its gfNeighbors fluid
// list). GF3's displacement is the mirror — ONE thread per FLUID PARTICLE, each fluid particle iterates its GF1
// fgNeighbors grain list (fixed order), and for each grain that CONTAINS it accumulates the surface-snap push
// into a SEPARATE dp[] (JACOBI) + applies the drag-reaction velocity impulse. Each fluid particle writes ONLY
// its own dp/vel -> per-fluid-disjoint, race-free, NO atomics, [numthreads(64,1,1)] MULTI-THREAD, NO TDR (the
// GR3/CG3/CP3 win; the EXACT shape of grain.h::CollideGrainSpheres, grain as the sphere). int64
// (FxLength/FxNormalize/fxmul) -> cgf_displace.comp is VULKAN-SPIR-V-ONLY + the Metal --cgf-displace showcase
// runs THIS CPU ApplyGrainsToFluid (byte-identical by construction, the GF2/CG3 split). The GF1 cross-query
// passes stay int32 MSL-native.

using fpx::FxLength;       // read-only: the int64 length (FxISqrt of the sum of squares)

// kDragReaction: the grain->fluid drag-reaction coefficient (the GF2 kDrag partner — the equal-opposite of the
// grain's GF2 drag, now imparting the grain's momentum to the fluid). Host-snapped Q16.16 (~1.5, the CG3/CP3
// kDragReaction value). The showcase + the CPU reference + the GPU shader share THIS exact constant.
inline constexpr fx kDragReaction = (fx)(1.5 * (double)kOne + 0.5);   // 98304 (Q16.16, ~1.5 reaction drag)

// ApplyGrainsToFluid(world, neighbors): the per-fluid-particle projection-out-of-grains + drag reaction (the GR3
// CollideGrainSphere mold over the grain set + a Jacobi dp[]). For each fluid particle p (skip STATIC), over
// each of its GF1 grain neighbours g (fgNeighbors[fgStart[p]..fgStart[p+1]), the FIXED GF1 emit order), with
// d = p.pos − g.pos, dist = FxLength(d), surf = g.radius: if dist < surf (the fluid particle is INSIDE the grain)
// accumulate the surface-snap push into a SEPARATE dp[] buffer
//   dp_p += FxAdd(g.pos, FxScale(FxNormalize(d), surf)) − p.pos    // snap to the grain surface (== g.radius)
// (dist==0 -> the FxNormalize +Y fallback — EXACTLY grain.h::CollideGrainSphere's snap with the grain as the
// sphere) AND apply the drag-reaction velocity impulse per axis
//   p.vel += fxmul(kDragReaction, (g.vel − p.vel)) · dt           // toward the grain velocity
// then apply p.pos += dp_p for ALL fluid particles after (JACOBI — each fluid particle reads the iteration-start
// grain state, NOT the in-progress positions). STATIC fluid particles (kFlagStatic / boundary) -> dp 0, vel
// untouched. int64 (FxLength/FxNormalize/fxmul). cgf_displace.comp copies THIS body VERBATIM (one thread per
// fluid particle). Deterministic (the fixed fgNeighbors order, fixed op order) -> bit-identical to the GPU
// memcmp + two-run byte-identical.
inline void ApplyGrainsToFluid(CGFWorld& world, const CGFNeighbors& neighbors) {
    const uint32_t fluidCount = (uint32_t)world.fluid.size();
    const fx dt = world.dt;
    std::vector<FxVec3> dp((size_t)fluidCount, FxVec3{0, 0, 0});   // the Jacobi double-buffer (per-fluid Δp)
    for (uint32_t i = 0; i < fluidCount; ++i) {
        fluid::FluidParticle& p = world.fluid[(size_t)i];
        if (p.flags & fluid::kFlagStatic) continue;       // boundary fluid -> dp 0, vel untouched
        const uint32_t s0 = neighbors.fgStart[(size_t)i];
        const uint32_t s1 = neighbors.fgStart[(size_t)i + 1u];
        FxVec3 accum{0, 0, 0};
        for (uint32_t s = s0; s < s1; ++s) {
            const grain::GrainParticle& g = world.grains[(size_t)neighbors.fgNeighbors[(size_t)s]];
            const FxVec3 d = FxSub(p.pos, g.pos);         // fluid relative to the grain centre (outward)
            const fx dist = FxLength(d);
            const fx surf = g.radius;                     // the grain exclusion radius (fluid particles are points)
            if (dist >= surf) continue;                   // outside the grain sphere -> no push
            // (1) POSITIONAL DISPLACEMENT: snap the fluid particle to the grain surface (the sand parts the fluid).
            const FxVec3 nrm = FxNormalize(d);            // outward normal (dist==0 -> {0,kOne,0} fallback)
            const FxVec3 surfPt = FxAdd(g.pos, FxScale(nrm, surf));   // the surface point along the normal
            accum = FxAdd(accum, FxSub(surfPt, p.pos));   // into the Jacobi dp[] (the CollideGrainSphere push)
            // (2) DRAG REACTION: the grain imparts momentum to the fluid (the equal-opposite of GF2's drag).
            const FxVec3 dv = FxSub(g.vel, p.vel);        // toward the grain velocity
            p.vel.x += fpx::fxmul(fpx::fxmul(kDragReaction, dv.x), dt);
            p.vel.y += fpx::fxmul(fpx::fxmul(kDragReaction, dv.y), dt);
            p.vel.z += fpx::fxmul(fpx::fxmul(kDragReaction, dv.z), dt);
        }
        dp[(size_t)i] = accum;
    }
    // Apply pos += dp for all fluid particles (Jacobi — disjoint per-fluid writes, race-free).
    for (uint32_t i = 0; i < fluidCount; ++i) {
        if (world.fluid[(size_t)i].flags & fluid::kFlagStatic) continue;
        world.fluid[(size_t)i].pos = FxAdd(world.fluid[(size_t)i].pos, dp[(size_t)i]);
    }
}

// MeasureFluidGrainPenetration(world): the honest no-penetration metric (the FL4/GR3/CG3 caveat) — over every
// fluid / grain pair, sum/max the fluid-into-grain penetration pen = g.radius − |p.pos − g.pos| > 0. Returns
// {peak, summed} in Q16.16 (int64 accumulator). DETERMINISTIC + bit-exact. The showcase's "fluid parted" proof
// compares this BEFORE vs AFTER ApplyGrainsToFluid (penAfter < penBefore — the FL4/GR3 honesty: relieved, NOT
// zero; Jacobi single-projection so a fluid particle inside MULTIPLE grains leaves a deterministic-but-nonzero
// residual). Static fluid particles are counted (they CAN sit inside a grain; GF3 does not move them, so their
// penetration is part of the honest residual).
struct FluidGrainPenetration { int64_t peak = 0; int64_t summed = 0; };
inline FluidGrainPenetration MeasureFluidGrainPenetration(const CGFWorld& world) {
    FluidGrainPenetration out;
    for (const fluid::FluidParticle& p : world.fluid)
        for (const grain::GrainParticle& g : world.grains) {
            const fx pen = g.radius - FxLength(FxSub(p.pos, g.pos));
            if (pen > 0) { out.summed += (int64_t)pen; if ((int64_t)pen > out.peak) out.peak = (int64_t)pen; }
        }
    return out;
}

// CountDisplacedFluid(world): the deterministic count of fluid particles INSIDE at least one grain (dist <
// g.radius) — the "displaced > 0" coverage stat for the showcase proof. Static fluid ARE counted (they sit
// inside a grain too). Pure int64 compare -> bit-exact CPU<->GPU.
inline uint32_t CountDisplacedFluid(const CGFWorld& world) {
    uint32_t c = 0;
    for (const fluid::FluidParticle& p : world.fluid) {
        bool inside = false;
        for (const grain::GrainParticle& g : world.grains)
            if (FxLength(FxSub(p.pos, g.pos)) < g.radius) { inside = true; break; }
        if (inside) ++c;
    }
    return c;
}

// StepCGFDisplace(world): ONE grain->fluid displacement pass (the showcase / GPU driver mirror). Re-query the
// GF1 cross-lists from the CURRENT state (BuildCGFNeighbors) -> ApplyGrainsToFluid. The GPU runs the SAME
// sequence (the int32 GF1 query passes + the int64 cgf_displace.comp). Pure integer -> two runs byte-identical,
// cross-backend identical.
inline void StepCGFDisplace(CGFWorld& world) {
    const CGFNeighbors nbr = BuildCGFNeighbors(world);
    ApplyGrainsToFluid(world, nbr);
}

// ===== Slice GF4 — THE COUPLED STEP (StepCGF, the convergence; the 4th slice of FLAGSHIP #13) =========
// ONE deterministic forward grain+fluid settling tick: the four GF arcs run TOGETHER. A frictional sand bed
// (GR3 non-penetration + GR4 Coulomb friction) co-settles with an incompressible PBF fluid (FL4 density) in
// ONE Q16.16 world, exchanging momentum two ways (GF3 grain->fluid displacement + GF2 fluid->grain buoyancy)
// — WET SAND / MUD / SLURRY: the fluid pools on/seeps around the bed, submerged grains lighten, no script.
// The CG4 StepCGrain mold with TWO PARTICLE pools (fluid + grain) instead of grains + rigid bodies. StepCGF
// ORCHESTRATES the already-bit-exact pieces (FL4 density sub-passes, GR3/GR4 contact/friction, GF2/GF3
// couplings) in the LOCKED order below. NO new shader, NO new RHI: the GPU showcase is a host-driven
// multi-pass driver over the EXISTING fluid_* + grain_* + cgf_displace/cgf_buoyancy + GF1 cross-query shaders.
//
// THE MAKE-OR-BREAK DECISION (a documented refinement of the roadmap sketch): both pools are PBF/PBD, so the
// final velocity is DERIVED as (pos − prev)/dt AFTER the position solve. Therefore any velocity impulse
// applied INSIDE the K position iters is CLOBBERED by that derivation. So the cross-pool VELOCITY couplings
// (GF2 buoyancy + GF3 displacement) run ONCE, in a POST pass AFTER the velocity update — exactly as PBF
// XSPH-viscosity / vorticity confinement are applied post-velocity-update. The K Jacobi iters hold ONLY each
// pool's POSITIONAL self-constraints (FL4 density for the fluid; GR3 normal + GR4 friction for the grains).
// THREE wins: (1) correctness — nothing clobbered; (2) maximal reuse — GF2 AND GF3 called VERBATIM (no
// splitting, no new coupling math); (3) no K-compensation — GF2/GF3 run once, so the CG4 invMass=kOne/iters
// trick is UNNEEDED (grains + fluid keep their natural mass). GF1/GF2/GF3 code stays BYTE-FROZEN.
//
// THE COUPLED TICK (StepCGF, the locked (1)-(6) order):
//   (1) PREDICT both pools: fluid::IntegrateFluid (FL1) + grain::IntegrateGrains (GR1).
//   (2) BUILD ONCE from the predicted positions (fixed across the K iters — the standard PBF choice): the FL2
//       fluid neighbour list, the GR2 grain neighbour list, the GF1 cross lists, and the FluidKernel LUT.
//   (3) K JACOBI iters, each pool's POSITIONAL self-constraints ONLY, in this fixed sub-order:
//         (3a) FL4 density : ComputeDensity -> ComputeLambda -> SolveDensityConstraint -> fluid.pos += dp.
//         (3b) GR3 normal  : SolveGrainContact -> grains.pos += dp.
//         (3c) GR4 friction: SolveGrainFriction(kGrainMu) -> grains.pos += dp.
//   (4) VELOCITY UPDATE both pools: vel = (pos − prev)/dt (static skipped, as the sub-passes already do).
//   (5) CROSS-POOL VELOCITY COUPLING — ONCE, AFTER (4) so it survives (reuse the step-(2) cross lists):
//         (5a) GF3 grain->fluid : ApplyGrainsToFluid(world, nbr)  VERBATIM (snap out + drag reaction).
//         (5b) GF2 fluid->grain : AccumGrainBuoyancy(world, nbr, kBuoyPerFluid)  VERBATIM (lift + drag).
//   (6) GROUND CLAMPS both pools: fluid::CollidePlane + grain::CollideGrainPlane.
// Pure integer, fixed op order -> two runs bit-identical AND bit-exact GPU==CPU (every constituent pass is
// already proven so; the driver only sequences them). The grain->fluid displacement happening once-per-step
// (not re-projected each iter) is exactly how GF3's StepCGFDisplace already works — the fluid's own K-iter
// density solve keeps it incompressible, and the post pass parts it out of the sand each step.
//
// HONEST CAVEATS (the FL4/GR3/GF2/GF3 caveat shape): the K Jacobi iters leave a deterministic-but-nonzero
// incompressibility + non-penetration residual (single projection); the wet/dry lift + the pool-above margin
// are EMERGENT/within-band (NOT exact buoyancy depths). The claim is DETERMINISM + cross-platform bit-identity
// composing the bit-exact pieces, NOT a validated continuum two-phase model.

using fpx::fxdiv;          // read-only: the int64-intermediate Q16.16 divide (the PBF velocity derivation)

// StepCGF(world, dt, iters): the (1)-(6) locked-order host driver. Reuses FL4/GR3/GR4 in the K iters and
// GF2/GF3 VERBATIM in the POST pass. The neighbour lists + the kernel are built ONCE per step from the
// PREDICTED positions (the PBF fixed-neighbour choice). Pure integer, fixed op order. The kernel is the FL
// scene's kernel (the caller passes it via world — but the CGFWorld has no kernel field, so StepCGF rebuilds
// it ONCE per step from world.h/restDensity probe — NO, the caller passes it as `kernel`). To keep CGFWorld
// byte-frozen, the kernel is a STEP parameter (the showcase/test builds it once from the scene + passes it).
inline void StepCGF(CGFWorld& world, const fluid::FluidKernel& kernel, fx dt, int iters) {
    const size_t nf = world.fluid.size();
    const size_t ng = world.grains.size();

    // (1) PREDICT both pools (each: vel += g·dt; prev = pos; pos += vel·dt; floor clamp). FL1 + GR1, VERBATIM.
    fluid::IntegrateFluid(world.fluid, world.gravity, dt, world.groundY);
    grain::IntegrateGrains(world.grains, world.gravity, dt, world.groundY);

    // (2) BUILD ONCE from the PREDICTED positions (fixed across the K iters).
    const fluid::FluidGrid fGrid = fluid::MakeGrid(world.fluid, kernel.h);
    const fluid::FluidCellTable fTable = fluid::BuildCellTable(world.fluid, fGrid);
    const fluid::FluidNeighborList fList = fluid::BuildNeighborList(world.fluid, fGrid, fTable, kernel.h);
    const grain::GrainGrid gGrid = grain::MakeGrainGrid(world.grains, world.h);
    const grain::GrainCellTable gTable = grain::BuildGrainCellTable(world.grains, gGrid);
    const grain::GrainNeighborList gList =
        grain::BuildGrainNeighborList(world.grains, gGrid, gTable, world.h);
    const CGFNeighbors nbr = BuildCGFNeighbors(world);   // GF1 cross lists from the predicted positions

    // (3) K JACOBI iters — POSITIONAL self-constraints ONLY (their effect carried by the (4) pos−prev derive).
    std::vector<fx> density, lambda;
    std::vector<FxVec3> fdp, gdp;
    for (int it = 0; it < iters; ++it) {
        // (3a) FL4 density (the incompressible pool): ρ -> λ -> Δp -> apply (Jacobi, separate dp buffer).
        fluid::ComputeDensity(world.fluid, fList, kernel, density);
        fluid::ComputeLambda(world.fluid, fList, kernel, density, lambda);
        fluid::SolveDensityConstraint(world.fluid, fList, kernel, lambda, fdp);
        for (size_t i = 0; i < nf; ++i) {
            if (world.fluid[i].flags & fluid::kFlagStatic) continue;
            world.fluid[i].pos = FxAdd(world.fluid[i].pos, fdp[i]);
        }
        // (3b) GR3 normal push (grain-grain non-penetration; Δp into a SEPARATE dp buffer -> apply).
        grain::SolveGrainContact(world.grains, gList, gdp);
        for (size_t i = 0; i < ng; ++i) {
            if (world.grains[i].flags & grain::kFlagStatic) continue;
            world.grains[i].pos = FxAdd(world.grains[i].pos, gdp[i]);
        }
        // (3c) GR4 TANGENTIAL friction (reads the POST-normal positions; the angle-of-repose clamp).
        grain::SolveGrainFriction(world.grains, gList, grain::kGrainMu, gdp);
        for (size_t i = 0; i < ng; ++i) {
            if (world.grains[i].flags & grain::kFlagStatic) continue;
            world.grains[i].pos = FxAdd(world.grains[i].pos, gdp[i]);
        }
    }

    // (4) VELOCITY UPDATE both pools (PBF: derives the constraint-corrected velocity). Static skipped.
    if (dt != 0) {
        for (size_t i = 0; i < nf; ++i) {
            if (world.fluid[i].flags & fluid::kFlagStatic) continue;
            const FxVec3 dpos = FxSub(world.fluid[i].pos, world.fluid[i].prev);
            world.fluid[i].vel = FxVec3{fxdiv(dpos.x, dt), fxdiv(dpos.y, dt), fxdiv(dpos.z, dt)};
        }
        for (size_t i = 0; i < ng; ++i) {
            if (world.grains[i].flags & grain::kFlagStatic) continue;
            const FxVec3 dpos = FxSub(world.grains[i].pos, world.grains[i].prev);
            world.grains[i].vel = FxVec3{fxdiv(dpos.x, dt), fxdiv(dpos.y, dt), fxdiv(dpos.z, dt)};
        }
    }

    // (5) CROSS-POOL VELOCITY COUPLING — ONCE, AFTER (4) so the impulses survive the pos−prev derivation.
    // Reuse the step-(2) cross lists (fixed; the post-pass parts the fluid out + buoys the submerged grains).
    ApplyGrainsToFluid(world, nbr);                   // (5a) GF3 grain->fluid (VERBATIM)
    AccumGrainBuoyancy(world, nbr, kBuoyPerFluid);    // (5b) GF2 fluid->grain (VERBATIM)

    // (6) GROUND CLAMPS both pools (the fluid floor + the radius-aware grain floor rest).
    fluid::CollidePlane(world.fluid, world.groundY);
    grain::CollideGrainPlane(world.grains, world.groundY);
}

// StepCGFSteps(world, kernel, dt, iters, steps): run K coupled ticks. The CPU reference the GPU multi-pass
// driver memcmp's against byte-for-byte (the final fluid + grain arrays). Pure integer -> two runs
// byte-identical, cross-backend identical (the GPU runs the FL4/GR3/GR4/GF2/GF3 int64 passes Vulkan-only +
// the int32 GF1 cross-query / cell passes; Metal runs THIS).
inline void StepCGFSteps(CGFWorld& world, const fluid::FluidKernel& kernel, fx dt, int iters, int steps) {
    for (int s = 0; s < steps; ++s) StepCGF(world, kernel, dt, iters);
}

// MeasureCGFState(world): the honest emergent-metrics helper (the CG4 MeasureCGrainState twin with two
// particle pools). Returns the mean DYNAMIC-grain pos.y (bedY), the mean DYNAMIC-fluid pos.y (fluidY), the
// GR4 grain repose slope (repose — the bed still holds an angle of repose), and the GF2 wet/dry split
// (wetY = mean y of submerged grains, dryY = mean y of dry grains). Deterministic Q16.16 stats for the
// proofs. The pool-above + wet/dry margins are EMERGENT/within-band (the CP2/GR4/GF2 caveat), NOT exact
// depths. Re-queries GF1 from the current positions to classify wet/dry (reuse MeasureWetDry verbatim).
struct CGFState {
    fx       bedY   = 0;     // the mean settled DYNAMIC-grain pos.y (the bed line)
    fx       fluidY = 0;     // the mean settled DYNAMIC-fluid pos.y (the pool line — pools ABOVE the bed)
    fx       repose = 0;     // the GR4 grain repose slope (the bed-coherence stat)
    fx       wetY   = 0;     // the mean pos.y of submerged grains (cnt>0 fluid neighbours)
    fx       dryY   = 0;     // the mean pos.y of dry grains (cnt==0)
    uint32_t dynamicGrains = 0;   // the count of dynamic grains (bedY is their mean)
    uint32_t dynamicFluid  = 0;   // the count of dynamic fluid particles (fluidY is their mean)
};
inline CGFState MeasureCGFState(const CGFWorld& world) {
    CGFState s;
    int64_t bedSum = 0, fluidSum = 0;
    for (const grain::GrainParticle& g : world.grains)
        if (!(g.flags & grain::kFlagStatic)) { bedSum += (int64_t)g.pos.y; ++s.dynamicGrains; }
    for (const fluid::FluidParticle& f : world.fluid)
        if (!(f.flags & fluid::kFlagStatic)) { fluidSum += (int64_t)f.pos.y; ++s.dynamicFluid; }
    s.bedY   = s.dynamicGrains == 0u ? 0 : (fx)(bedSum / (int64_t)s.dynamicGrains);
    s.fluidY = s.dynamicFluid  == 0u ? 0 : (fx)(fluidSum / (int64_t)s.dynamicFluid);
    if (!world.grains.empty())
        s.repose = grain::MeasureGrainRepose(world.grains, world.groundY).slope;
    const WetDry wd = MeasureWetDry(world);   // GF2 re-query classify (submerged vs dry grains)
    s.wetY = wd.wetY; s.dryY = wd.dryY;
    return s;
}

}  // namespace cgf
}  // namespace hf::sim
