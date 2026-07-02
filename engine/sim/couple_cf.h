#pragma once
// Slice CF1 — Deterministic Cloth<->Fluid Coupling: the WET-CLOTH CORE (Track-S S1 of
// docs/SUPERIORITY_ROADMAP.md — the FOURTH material-interaction pairing after rigid<->fluid (couple.h),
// rigid<->grain (couple_grain.h) and grain<->fluid (couple_gf.h), and the SECOND particle<->particle one).
// A constraint-connected CLOTH sheet (engine/sim/cloth.h) and an incompressible PBF FLUID
// (engine/sim/fluid.h) co-resident in ONE Q16.16 world, exchanging momentum TWO WAYS: the fluid DRAGS /
// PUSHES the cloth (it sags, it is carried), the cloth BLOCKS / DRAGS the fluid (a falling stream is
// caught by a pinned sheet) — bit-identical CPU/Vulkan/Metal AND lockstep/rollback-replayable. No major
// engine ships ANY deterministic two-way coupling; this is Hazard Forge's fourth. Pure CPU, header-only,
// NO device, NO backend symbols. Namespace hf::sim::cfl. couple_cf is the NEW additive sibling
// #include-ing cloth.h + fluid.h READ-ONLY (both stay byte-unchanged — the couple_gf.h discipline).
//
// THE FOUR PIECES (the couple_gf.h GF1-GF5 mold, compressed into the one coupling-core slice):
//   (1) CROSS-QUERY (the GF1 twin): ONE shared bounded dense grid over BOTH pools' union AABB at
//       cell-size h (the coupling radius); each pool bucketed into its OWN cell table over the SHARED
//       grid; two CSR cross lists via count->scan->emit — per CLOTH VERT the fluid particles in its
//       27-cell stencil (cfNeighbors) + per FLUID particle the cloth verts in its 27-cell stencil
//       (fcNeighbors), each accepted iff the per-axis |dx| < h box reject passes (PURE INT32, the
//       over-inclusive box; the exact radial cull happens in the drag kernel / the contact FxLength —
//       the FL2/GF1 split). NO self-skip: the pools are DISTINCT.
//   (2) TWO-WAY DRAG (the momentum exchange — the FL7 XSPH discipline applied CROSS-POOL, symmetric):
//       both pools' velocities are DERIVED as v = (pos - prev)/dt (per-axis fxdiv, the PBF implicit
//       velocity); per cross pair within the kernel radius each side accumulates
//         acc_self += fxmul(v_other - v_self, W[bin(r^2)])          (the FL3 host-snapped poly6 LUT)
//       Jacobi into a scratch buffer (ALL reads before ANY write), then
//         v'_self = v_self + fxmul(invMass_self, fxmul(kDrag, acc_self))
//       and BOTH pools are RE-ENCODED vel = v', prev = pos - v'*dt (the FL7 read-derive-modify-reencode
//       discipline — the state stays the (pos, prev, vel) triple, so the FL5/CL5 snapshot machinery
//       applies unchanged). PINNED cloth verts (invMass 0) and STATIC fluid never move (share 0 /
//       skipped). MOMENTUM: the exchange is pairwise-antisymmetric with the inverse-mass scaling
//       (m*dv = invMass*m*J = J each side, opposite signs), so total momentum is conserved in EXACT
//       arithmetic; fxmul floor-truncation breaks exact pair cancellation by <= 1 LSB per fxmul stage
//       per pair per axis, so the drift is a small DETERMINISTIC nonzero bound (pinned honestly in
//       tests/cfl_test.cpp — the FL7 <=256-LSB-scale precedent), not exactly zero. Pinned verts are
//       momentum SINKS by design (infinite mass); the conservation statement is over all-dynamic pairs.
//   (3) CONTACT / BARRIER (the "cloth blocks fluid" half — the CL7 SolveSelfCollision projection mold
//       applied CROSS-POOL): a cloth vert and a fluid particle closer than rSum = clothRadius +
//       fluidRadius are pushed apart along the pair axis, inverse-mass split (fxdiv(invMass_self, wsum)
//       — pinned share 0), JACOBI (each side gathers its OWN correction from the OLD positions into a
//       scratch buffer; apply + ground clamp after). A COINCIDENT cross pair separates by the
//       deterministic POOL-IDENTITY tie-break (cloth +Y, fluid -Y — the CL7 index tie-break adapted:
//       the two pools are distinct so pool identity is the stable order).
//       *** HONEST POROSITY LIMIT (documented, the make-or-break honesty): a VERT-SPHERE barrier is
//       POROUS — the cloth surface is sampled only at its verts, so fluid can slip BETWEEN verts
//       whenever the lattice gap exceeds the combined radii (for a square lattice the worst gap centre
//       is spacing*sqrt(2)/2 from the nearest vert; choose rSum ABOVE that so the barrier holds
//       visually, e.g. spacing 0.5 -> gap 0.354 -> rSum 0.5). Even then a single Jacobi projection per
//       step is NOT a hard barrier: a particle moving > ~rSum in one dt can tunnel (bounded velocities
//       in the pinned scenes keep it from happening; a face-based barrier + CCD is future CF work). ***
//   (4) THE COMPOSED STEP StepClothFluid (the GF4 StepCGF mold): host-driven multi-pass —
//       (a) fluid::StepFluid VERBATIM (FL4 predict + K Jacobi density iters + velocity + collide);
//       (b) cloth::StepClothSelf VERBATIM (CL3 constraints + CL4 collide + CL7 self-collision);
//       (c) IDENTITY-AT-ZERO: kDrag == 0 AND contact off (rSum <= 0) -> EARLY RETURN here, zero
//           coupling state touched -> BOTH pools BIT-IDENTICAL to their uncoupled steps (the off-switch
//           contract every HF lobe carries);
//       (d) cross lists from the post-step positions (BuildCFLNeighbors);
//       (e) cross-pool CONTACT projection (positional, Jacobi both pools);
//       (f) TWO-WAY DRAG: velocity derivation (pos-prev)/dt — which now INCLUDES the contact push —
//           then the symmetric exchange + the prev re-encode on both pools (the GF4 lesson: velocity
//           couplings run ONCE, POST the position work, so nothing is clobbered by the derivation).
//   (5) LOCKSTEP + ROLLBACK (the GF5 RunCGFLockstep twin over cloth+fluid): CFLCommand (cloth wind /
//       fluid jet), SimCFLTick, the TWO-POOL snapshot, RunCFLLockstep / RunCFLRollback — a peer fed the
//       INPUT stream alone re-derives BOTH pools bit-for-bit; a misprediction is rolled back exactly.
//
// THE GPU CONVENTION FOR CF1 (documented decision — simplest correct choice): CF1 ships the PURE-CPU
// coupling core + the pure-CPU showcase (the GF5 --cgf-lockstep precedent: NO GPU dispatch, NO new
// shader, NO new RHI). Both backends run THIS identical CPU code, so the integer showcase golden is
// bit-identical cross-backend BY CONSTRUCTION (the strict zero-differing-pixel bar). The coupling math
// is int64 (FxLength/FxNormalize/fxdiv/RadiusSq) so any future GPU kernels would be DXC/Vulkan-only +
// Metal CPU-reference (the established fluid_dp.comp/cloth_solve.comp convention); the int32 cross-query
// passes would MSL-generate natively (the GF1 precedent). Deferred to a later CF slice.
//
// HONEST CAVEATS (beyond the porosity limit above): the drag kernel radius is the FLUID kernel's h
// (kernel.h) — build the scene with the coupling radius world.h == kernel.h so the box candidate list
// covers the kernel support exactly (the FL2 "box over-inclusive, kernel culls" split); the drag is
// dissipative by design (it relaxes relative velocity); the contact is a single Jacobi projection per
// step (deterministic-but-nonzero residual, the CL7/FL4 caveat); TotalDynamicMomentum below assumes
// UNIT-MASS dynamic particles (invMass == kOne, the scene convention) — a mass-weighted variant is
// future work. The claim is DETERMINISM + cross-platform bit-identity + lockstep replay of a two-way
// cloth<->fluid exchange, NOT a validated fluid-structure-interaction continuum model.
//
// REUSE MAP (read-only, NOT modified; couple_cf is the additive sibling):
//   * engine/sim/cloth.h: ClothParticle/kFlagPinned (CL1), ClothGrid/BuildConstraints (CL2),
//     StepClothSelf + ClothAdjacency (CL4/CL7 — the composed cloth step called AS-IS), SphereCollider,
//     ClothDigest. cloth.h stays byte-unchanged.
//   * engine/sim/fluid.h: FluidParticle/kFlagStatic (FL1), FluidGrid/BuildCellTable (FL2 — reused for
//     the fluid cell table over the shared grid), FluidKernel/H2Of/RadiusSq/BinOf (FL3 — the drag
//     kernel), StepFluid (FL4 — called AS-IS), SphereCollider, FluidDigest. fluid.h stays byte-unchanged.
//   * engine/sim/fpx.h: fx/fxmul/fxdiv/FxVec3/FxAdd/FxSub/FxScale/FxLength/FxNormalize/FloorDiv/FxCell/
//     CellId — the Q16.16 toolbox verbatim, NO new fixed-point primitives.
//   * engine/sim/couple_gf.h (the MOLD, not included): CGFWorld/CGFGrid/BuildCGFNeighbors/StepCGF/
//     RunCGFLockstep — CF1 is its cloth sibling (cloth-LOCAL twins below; couple_cf does NOT pull in
//     grain.h just for grid helpers — the CL7 "cloth-LOCAL twin" precedent).

#include <cstdint>
#include <vector>

#include "sim/fpx.h"     // read-only: the Q16.16 toolbox (fx/fxmul/fxdiv/FxVec3/FxLength/FxNormalize/...)
#include "sim/cloth.h"   // read-only: ClothParticle / ClothGrid / Constraint / StepClothSelf (CL1-CL7)
#include "sim/fluid.h"   // read-only: FluidParticle / FluidKernel / BuildCellTable / StepFluid (FL1-FL7)

namespace hf::sim {
namespace cfl {

// Reuse the fpx Q16.16 scalar + vector toolbox verbatim (NO new fixed-point primitives).
using fpx::fx;
using fpx::FxVec3;
using fpx::FxCell;
using fpx::fxmul;
using fpx::fxdiv;
using fpx::FxAdd;
using fpx::FxSub;
using fpx::FxScale;
using fpx::FxLength;
using fpx::FxNormalize;
using fpx::FloorDiv;
using fpx::CellId;
inline constexpr int kFrac = fpx::kFrac;   // Q16.16 fractional bits
inline constexpr fx  kOne  = fpx::kOne;    // 1.0 in Q16.16 (65536)

// ----- The unified two-pool world (a cloth vert pool + a fluid pool in one Q16.16 frame) -------------------
// Both pools share the SAME world units (ClothParticle and FluidParticle are the identical 44-byte
// (pos, prev, vel, invMass, flags) packing). The STATIC per-scene data (the cloth constraint graph, the
// colliders, the fluid kernel, the iteration counts) lives in CFLScene below — the GF4 "kernel is a step
// parameter" discipline generalized (CGFWorld carried only the pools + the scalar params; so does CFLWorld).
struct CFLWorld {
    std::vector<cloth::ClothParticle> cloth;        // the cloth vert pool (constraint-connected, CL1-CL7)
    std::vector<fluid::FluidParticle> fluid;        // the fluid pool (FL1-FL7)
    FxVec3 gravity;                                 // shared gravity (both pools integrate under it)
    fx     dt = 0;                                  // the tick length (Q16.16 seconds)
    fx     groundY = 0;                             // the shared ground plane
    fx     h = 0;                                   // the coupling radius (== the shared-grid cell size;
                                                    //   build scenes with h == scene.kernel.h, see banner)
    fx     clothRadius = 0;                         // per-vert contact radius (the cloth "thickness" half)
    fx     fluidRadius = 0;                         // per-fluid-particle contact radius
};

// The STATIC per-scene data threaded through the composed step (built once by the caller, constant across
// ticks — the GF4/GF5 FluidKernel-threading precedent extended to the cloth's static graph).
struct CFLScene {
    cloth::ClothGrid                    grid;           // the cloth lattice dims (CL2 topology)
    std::vector<cloth::Constraint>      constraints;    // the CL2 distance-constraint graph
    cloth::ClothAdjacency               excl;           // the CL7 self-collision exclusion ring
    std::vector<cloth::SphereCollider>  clothSpheres;   // CL4 static colliders (may be empty)
    std::vector<fluid::SphereCollider>  fluidSpheres;   // FL4 static colliders (may be empty)
    fluid::FluidKernel                  kernel;         // the FL3 kernel LUT (drag weights + FL4 solve)
    int                                 iters = 0;      // the cloth Gauss-Seidel / fluid Jacobi iterations
    fx                                  selfThickness = 0;  // CL7 self-collision thickness (0 = off)
    int                                 selfIters = 0;      // CL7 self-collision Jacobi iterations
    int                                 contactIters = 1;   // cross-pool contact Jacobi iterations per step
                                                            //   (the CL7 selfIters discipline: each iteration
                                                            //   re-gathers from the CURRENT positions over the
                                                            //   per-step candidate lists; more iterations =
                                                            //   a firmer barrier, deterministic either way)
    fx                                  visc = 0;           // FL7 XSPH viscosity coefficient for the fluid's
                                                            //   own step (0 = off == StepFluid bit-identical
                                                            //   by the FL7 identity-at-zero; a wet-cloth scene
                                                            //   wants it ON — an undamped PBF pile's startup
                                                            //   burst skates off the sheet)
};

// ----- The shared grid over BOTH pools' union AABB (the GF1 CGFGrid cloth-local twin) ----------------------
struct CFLGrid {
    fx     h = 0;        // Q16.16 cell size (== the coupling search radius)
    FxCell cellMin;      // the integer cell coord of the grid's (0,0,0) corner (the union lower cell)
    FxCell gridDim;      // the grid extent in cells per axis (cellCount = x*y*z)
};

// CFLCellOf(pos, h): the integer grid cell a position falls in, FloorDiv per axis. Pure int32 (the SAME
// quantization for BOTH pools — what makes the shared grid total).
inline FxCell CFLCellOf(const FxVec3& pos, fx h) {
    return FxCell{FloorDiv(pos.x, h), FloorDiv(pos.y, h), FloorDiv(pos.z, h)};
}

// CFLCellCount(grid): the total number of cells in the shared dense grid.
inline uint32_t CFLCellCount(const CFLGrid& grid) {
    return (uint32_t)(grid.gridDim.x * grid.gridDim.y * grid.gridDim.z);
}

// MakeCFLGrid(world): the shared bounded dense grid covering the UNION of both pools' cell ranges at
// cell-size world.h. BOTH pools empty -> a 1x1x1 grid at origin (deterministic degenerate); either pool
// empty -> the grid covers the OTHER pool's bounds. Pure int32 (the GF1 MakeCGFGrid twin).
inline CFLGrid MakeCFLGrid(const CFLWorld& world) {
    CFLGrid grid;
    grid.h = world.h;
    if (world.cloth.empty() && world.fluid.empty()) {
        grid.cellMin = FxCell{0, 0, 0};
        grid.gridDim = FxCell{1, 1, 1};
        return grid;
    }
    bool seeded = false;
    FxCell lo{0, 0, 0}, hi{0, 0, 0};
    auto fold = [&](const FxVec3& pos) {
        const FxCell c = CFLCellOf(pos, world.h);
        if (!seeded) { lo = c; hi = c; seeded = true; return; }
        if (c.x < lo.x) lo.x = c.x; if (c.x > hi.x) hi.x = c.x;
        if (c.y < lo.y) lo.y = c.y; if (c.y > hi.y) hi.y = c.y;
        if (c.z < lo.z) lo.z = c.z; if (c.z > hi.z) hi.z = c.z;
    };
    for (const cloth::ClothParticle& p : world.cloth) fold(p.pos);
    for (const fluid::FluidParticle& f : world.fluid) fold(f.pos);
    grid.cellMin = lo;
    grid.gridDim = FxCell{hi.x - lo.x + 1, hi.y - lo.y + 1, hi.z - lo.z + 1};
    return grid;
}

// FlatCFLCellId(cell, grid): the flat id of an absolute cell coord into the shared dense grid (offset by
// cellMin into [0,gridDim), then the CellId linearization). The caller guarantees the cell is in range.
inline uint32_t FlatCFLCellId(const FxCell& cell, const CFLGrid& grid) {
    const FxCell local{cell.x - grid.cellMin.x, cell.y - grid.cellMin.y, cell.z - grid.cellMin.z};
    return CellId(local, grid.gridDim);
}

// FluidGridFromCFL(g): adapt the shared grid to the FL2 FluidGrid so fluid::BuildCellTable buckets the
// fluid pool in the SAME coordinate frame (the GF1 FluidGridFromCGF twin — exact reuse of FL2).
inline fluid::FluidGrid FluidGridFromCFL(const CFLGrid& g) {
    fluid::FluidGrid out; out.h = g.h; out.cellMin = g.cellMin; out.gridDim = g.gridDim; return out;
}

// ----- The cloth cell table over the shared grid (cloth.h has no cell-table type -> the local twin) --------
// The FL2 BuildCellTable count->scan->emit applied to cloth verts (ascending vert index within each cell,
// fully deterministic — the CL7 BuildSelfCandidates internal bucketing, factored to the shared grid).
struct CFLClothCellTable {
    std::vector<uint32_t> cellStart;    // cellCount+1 exclusive prefix-sum offsets (CSR)
    std::vector<uint32_t> cellVerts;    // cloth vert indices grouped by cell (size == vert count)
};

inline CFLClothCellTable BuildClothCellTable(const std::vector<cloth::ClothParticle>& verts,
                                             const CFLGrid& grid) {
    const uint32_t n = (uint32_t)verts.size();
    const uint32_t cells = CFLCellCount(grid);
    CFLClothCellTable table;
    std::vector<uint32_t> counts((size_t)cells, 0u);
    for (uint32_t i = 0; i < n; ++i)
        ++counts[FlatCFLCellId(CFLCellOf(verts[(size_t)i].pos, grid.h), grid)];
    table.cellStart.assign((size_t)cells + 1u, 0u);
    uint32_t running = 0;
    for (uint32_t c = 0; c < cells; ++c) { table.cellStart[c] = running; running += counts[c]; }
    table.cellStart[cells] = running;
    table.cellVerts.assign((size_t)n, 0u);
    std::vector<uint32_t> cursor((size_t)cells, 0u);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t c = FlatCFLCellId(CFLCellOf(verts[(size_t)i].pos, grid.h), grid);
        table.cellVerts[table.cellStart[c] + cursor[c]] = i;   // ascending i (the ascending loop)
        ++cursor[c];
    }
    return table;
}

// ----- The CSR cross-pool query result (the two cross lists — the GF1 CGFNeighbors twin) -------------------
// cfStart[i..] is the exclusive prefix-sum of per-CLOTH-VERT fluid-neighbour counts (vertCount+1 entries);
// cfNeighbors[] holds the gathered FLUID indices grouped by vert, ascending stencil-cell (dz,dy,dx) then
// ascending fluid index. fcStart/fcNeighbors are the mirror, per FLUID over the cloth cell table.
struct CFLNeighbors {
    std::vector<uint32_t> cfStart;      // vertCount+1 exclusive prefix-sum offsets (CSR, cloth -> fluid)
    std::vector<uint32_t> cfNeighbors;  // gathered FLUID indices grouped by cloth vert (stencil order)
    std::vector<uint32_t> fcStart;      // fluidCount+1 exclusive prefix-sum offsets (CSR, fluid -> cloth)
    std::vector<uint32_t> fcNeighbors;  // gathered CLOTH-VERT indices grouped by fluid (stencil order)
};

// CountCross / EmitCross: the GF1 templated 27-cell-stencil cross scan (per query particle over the TARGET
// pool's cell table on the SHARED grid, the per-axis |dx| < h box accept, NO self-skip — the pools are
// distinct). Emit writes each query's accepted targets into its DISJOINT CSR slice in the FIXED
// (dz,dy,dx ascending; ascending j within a cell) order -> race-free, fully deterministic. Pure int32.
template <typename QueryVec, typename TargetVec, typename QueryArr, typename TargetArr>
inline uint32_t CountCross(const QueryArr& query, const TargetArr& target, const CFLGrid& grid,
                           const std::vector<uint32_t>& cellStart, const std::vector<uint32_t>& cellTargets,
                           std::vector<uint32_t>& perQueryOut, QueryVec qpos, TargetVec tpos) {
    const uint32_t n = (uint32_t)query.size();
    const fx h = grid.h;
    perQueryOut.assign((size_t)n, 0u);
    uint32_t total = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const FxVec3 pi = qpos(query[(size_t)i]);
        const FxCell ci = CFLCellOf(pi, h);
        uint32_t c = 0;
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const FxCell nc{ci.x + dx, ci.y + dy, ci.z + dz};
            if (nc.x < grid.cellMin.x || nc.x >= grid.cellMin.x + grid.gridDim.x) continue;
            if (nc.y < grid.cellMin.y || nc.y >= grid.cellMin.y + grid.gridDim.y) continue;
            if (nc.z < grid.cellMin.z || nc.z >= grid.cellMin.z + grid.gridDim.z) continue;
            const uint32_t cell = FlatCFLCellId(nc, grid);
            for (uint32_t s = cellStart[cell]; s < cellStart[cell + 1u]; ++s) {
                const uint32_t j = cellTargets[s];
                const FxVec3 pj = tpos(target[(size_t)j]);
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

template <typename QueryVec, typename TargetVec, typename QueryArr, typename TargetArr>
inline void EmitCross(const QueryArr& query, const TargetArr& target, const CFLGrid& grid,
                      const std::vector<uint32_t>& cellStart, const std::vector<uint32_t>& cellTargets,
                      const std::vector<uint32_t>& start, std::vector<uint32_t>& neighbors,
                      QueryVec qpos, TargetVec tpos) {
    const uint32_t n = (uint32_t)query.size();
    const fx h = grid.h;
    for (uint32_t i = 0; i < n; ++i) {
        const FxVec3 pi = qpos(query[(size_t)i]);
        const FxCell ci = CFLCellOf(pi, h);
        uint32_t base  = start[(size_t)i];
        uint32_t local = 0u;
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const FxCell nc{ci.x + dx, ci.y + dy, ci.z + dz};
            if (nc.x < grid.cellMin.x || nc.x >= grid.cellMin.x + grid.gridDim.x) continue;
            if (nc.y < grid.cellMin.y || nc.y >= grid.cellMin.y + grid.gridDim.y) continue;
            if (nc.z < grid.cellMin.z || nc.z >= grid.cellMin.z + grid.gridDim.z) continue;
            const uint32_t cell = FlatCFLCellId(nc, grid);
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

// Position accessors (pure, return the particle's pos by value — the GF1 GrainPos/FluidPos twins).
inline FxVec3 ClothPos(const cloth::ClothParticle& p) { return p.pos; }
inline FxVec3 FluidPos(const fluid::FluidParticle& f) { return f.pos; }

// ----- BuildCFLNeighbors: the two cross-pool count->scan->emit lists (the CF1 cross-query) ----------------
// (1) the SHARED grid + the cloth cell table + the fluid cell table over it; (2) cf: per cloth vert, COUNT
// the fluid 27-cell-stencil candidates -> SCAN -> cfStart -> EMIT -> cfNeighbors; (3) fc: the mirror per
// fluid over the cloth cell table. Deterministic (fixed stencil + ascending-index order). Pure int32.
inline CFLNeighbors BuildCFLNeighbors(const CFLWorld& world) {
    const uint32_t vertCount  = (uint32_t)world.cloth.size();
    const uint32_t fluidCount = (uint32_t)world.fluid.size();
    CFLNeighbors out;

    const CFLGrid grid = MakeCFLGrid(world);
    const CFLClothCellTable cTable = BuildClothCellTable(world.cloth, grid);
    const fluid::FluidGrid fGrid = FluidGridFromCFL(grid);
    const fluid::FluidCellTable fTable = fluid::BuildCellTable(world.fluid, fGrid);

    // (2) cf (cloth -> fluid): per vert, the FLUID cell table 27-cell stencil, |dx|<h accept.
    {
        std::vector<uint32_t> counts;
        const uint32_t total = CountCross(world.cloth, world.fluid, grid,
            fTable.cellStart, fTable.cellParticles, counts, &ClothPos, &FluidPos);
        out.cfStart.assign((size_t)vertCount + 1u, 0u);
        uint32_t running = 0;
        for (uint32_t i = 0; i < vertCount; ++i) { out.cfStart[(size_t)i] = running; running += counts[(size_t)i]; }
        out.cfStart[(size_t)vertCount] = running;
        out.cfNeighbors.assign((size_t)total, 0u);
        EmitCross(world.cloth, world.fluid, grid, fTable.cellStart, fTable.cellParticles,
                  out.cfStart, out.cfNeighbors, &ClothPos, &FluidPos);
    }

    // (3) fc (fluid -> cloth): per fluid, the CLOTH cell table 27-cell stencil, |dx|<h accept.
    {
        std::vector<uint32_t> counts;
        const uint32_t total = CountCross(world.fluid, world.cloth, grid,
            cTable.cellStart, cTable.cellVerts, counts, &FluidPos, &ClothPos);
        out.fcStart.assign((size_t)fluidCount + 1u, 0u);
        uint32_t running = 0;
        for (uint32_t i = 0; i < fluidCount; ++i) { out.fcStart[(size_t)i] = running; running += counts[(size_t)i]; }
        out.fcStart[(size_t)fluidCount] = running;
        out.fcNeighbors.assign((size_t)total, 0u);
        EmitCross(world.fluid, world.cloth, grid, cTable.cellStart, cTable.cellVerts,
                  out.fcStart, out.fcNeighbors, &FluidPos, &ClothPos);
    }

    return out;
}

// CountCF / CountFC: the total cross pairs each direction (== the CSR sentinels; symmetric — every pair
// appears once per direction). Pure integer.
inline uint32_t CountCF(const CFLNeighbors& nbr) { return nbr.cfStart.empty() ? 0u : nbr.cfStart.back(); }
inline uint32_t CountFC(const CFLNeighbors& nbr) { return nbr.fcStart.empty() ? 0u : nbr.fcStart.back(); }

// ----- SolveClothFluidContact: the cross-pool sphere-sphere separation (the "cloth blocks fluid" half) ----
// The CL7 SolveSelfCollision projection mold applied CROSS-POOL. For each cross pair closer than
// rSum = clothRadius + fluidRadius: pen = rSum - dist; each side accumulates ITS OWN inverse-mass share of
// the push along the pair axis (pointing away from the other pool's particle) JACOBI into a scratch buffer
// (all reads from the OLD positions), then both pools apply + ground-clamp. PINNED cloth verts / STATIC
// fluid take share 0 and never move (their partner takes ITS share only — the CL3 wsum split; a pinned
// vert vs a dynamic fluid particle -> the fluid takes fxdiv(invMass_f, wsum) == the full unit share).
// A coincident pair separates by the deterministic POOL-IDENTITY tie-break (cloth +Y, fluid -Y).
// int64-backed (FxLength/FxNormalize/fxdiv). Returns the number of (vert, fluid) projections gathered on
// the cloth side (a deterministic coverage stat). Bounds-checked (CSR sizes + target indices).
inline int SolveClothFluidContact(CFLWorld& world, const CFLNeighbors& nbr) {
    const uint32_t nc = (uint32_t)world.cloth.size();
    const uint32_t nf = (uint32_t)world.fluid.size();
    const fx rSum = world.clothRadius + world.fluidRadius;
    if (rSum <= 0) return 0;                                             // contact off -> EXACT no-op
    if (nbr.cfStart.size() != (size_t)nc + 1u || nbr.fcStart.size() != (size_t)nf + 1u) return 0;
    std::vector<FxVec3> dpc((size_t)nc, FxVec3{0, 0, 0});                // the cloth Jacobi scratch
    std::vector<FxVec3> dpf((size_t)nf, FxVec3{0, 0, 0});                // the fluid Jacobi scratch
    int projections = 0;

    // PASS 1a (cloth side): each vert gathers its OWN share from the OLD positions.
    for (uint32_t i = 0; i < nc; ++i) {
        const cloth::ClothParticle& pi = world.cloth[(size_t)i];
        if (pi.flags & cloth::kFlagPinned) continue;                     // pinned share 0 -> dp stays 0
        FxVec3 acc{0, 0, 0};
        for (uint32_t s = nbr.cfStart[(size_t)i]; s < nbr.cfStart[(size_t)i + 1u]; ++s) {
            const uint32_t j = nbr.cfNeighbors[(size_t)s];
            if (j >= nf) continue;                                       // bounds-checked skip
            const fluid::FluidParticle& pj = world.fluid[(size_t)j];
            const fx wsum = pi.invMass + pj.invMass;
            if (wsum == 0) continue;
            const FxVec3 d = FxSub(pi.pos, pj.pos);
            const fx dist = FxLength(d);
            if (dist >= rSum) continue;                                  // the exact radial cull
            const fx pen = rSum - dist;
            const FxVec3 axis = (dist == 0) ? FxVec3{0, kOne, 0}         // pool-identity tie-break: cloth +Y
                                            : FxNormalize(d);
            const fx wi = fxdiv(pi.invMass, wsum);
            acc = FxAdd(acc, FxScale(axis, fxmul(pen, wi)));
            ++projections;
        }
        dpc[(size_t)i] = acc;
    }
    // PASS 1b (fluid side): the mirror gather (its OWN share; the pair set is the SAME by the box symmetry).
    for (uint32_t i = 0; i < nf; ++i) {
        const fluid::FluidParticle& pi = world.fluid[(size_t)i];
        if (pi.flags & fluid::kFlagStatic) continue;                     // static share 0 -> dp stays 0
        FxVec3 acc{0, 0, 0};
        for (uint32_t s = nbr.fcStart[(size_t)i]; s < nbr.fcStart[(size_t)i + 1u]; ++s) {
            const uint32_t j = nbr.fcNeighbors[(size_t)s];
            if (j >= nc) continue;                                       // bounds-checked skip
            const cloth::ClothParticle& pj = world.cloth[(size_t)j];
            const fx wsum = pi.invMass + pj.invMass;
            if (wsum == 0) continue;
            const FxVec3 d = FxSub(pi.pos, pj.pos);
            const fx dist = FxLength(d);
            if (dist >= rSum) continue;
            const fx pen = rSum - dist;
            const FxVec3 axis = (dist == 0) ? FxVec3{0, -kOne, 0}        // pool-identity tie-break: fluid -Y
                                            : FxNormalize(d);
            const fx wi = fxdiv(pi.invMass, wsum);
            acc = FxAdd(acc, FxScale(axis, fxmul(pen, wi)));
        }
        dpf[(size_t)i] = acc;
    }
    // PASS 2: apply + ground clamp (pinned / static untouched; the clamp is idempotent over the own-steps').
    for (uint32_t i = 0; i < nc; ++i) {
        cloth::ClothParticle& p = world.cloth[(size_t)i];
        if (p.flags & cloth::kFlagPinned) continue;
        p.pos = FxAdd(p.pos, dpc[(size_t)i]);
        if (p.pos.y < world.groundY) p.pos.y = world.groundY;
    }
    for (uint32_t i = 0; i < nf; ++i) {
        fluid::FluidParticle& p = world.fluid[(size_t)i];
        if (p.flags & fluid::kFlagStatic) continue;
        p.pos = FxAdd(p.pos, dpf[(size_t)i]);
        if (p.pos.y < world.groundY) p.pos.y = world.groundY;
    }
    return projections;
}

// ----- ApplyClothFluidDrag: the TWO-WAY momentum exchange (the FL7 XSPH discipline, CROSS-POOL) ------------
// PASS 1 (gather, Jacobi — all reads from the OLD (pos, prev) state): per cloth vert i (skip pinned)
//   v_i = (pos_i - prev_i)/dt (per-axis fxdiv); acc += fxmul(v_fluid_j - v_i, W[bin(r_ij^2)]) over its
//   cfNeighbors fluid list (the FL3 kernel LUT; bin >= bins -> zero, the radial cull); v'_i = v_i +
//   fxmul(invMass_i, fxmul(kDrag, acc)).
// The fluid side mirrors it over fcNeighbors with (v_cloth_j - v_i). PASS 2: re-encode BOTH pools:
// vel = v', prev = pos - v'*dt (per-axis fxmul) — pos untouched, the state stays (pos, prev, vel) so the
// lockstep snapshot machinery applies unchanged. kDrag == 0 or dt == 0 -> EXACT no-op (identity-at-zero:
// re-encoding at kDrag=0 would NOT be byte-neutral post-collide — the FL7 early-out lesson).
// MOMENTUM: pairwise antisymmetric up to fxmul floor-truncation (<= a few LSB per pair per axis — the
// bound is pinned in tests/cfl_test.cpp); pinned verts / static fluid are momentum sinks by design.
// NOTE (the fair-baseline lesson, documented): the re-encode replaces the cloth's integrate-accumulated
// velocity with the PBD-derived (pos-prev)/dt velocity — which also STABILIZES/DAMPS the cloth (vanilla
// cloth.h never re-derives vel from the constraint-corrected positions, so its vel integrates gravity
// unboundedly at equilibrium). A wet-vs-dry sag comparison must therefore run the DRY control through the
// SAME coupled entry (an EMPTY fluid pool — same re-encode discipline, zero fluid load) so the sag delta
// isolates the fluid's push; tests/cfl_test.cpp + the showcase do exactly that.
// int64-backed (fxdiv + RadiusSq/BinOf). Bounds-checked.
inline void ApplyClothFluidDrag(CFLWorld& world, const CFLNeighbors& nbr,
                                const fluid::FluidKernel& kernel, fx kDrag) {
    const fx dt = world.dt;
    if (kDrag == 0 || dt == 0) return;                                   // identity-at-zero: EXACT no-op
    const uint32_t nc = (uint32_t)world.cloth.size();
    const uint32_t nf = (uint32_t)world.fluid.size();
    if (nbr.cfStart.size() != (size_t)nc + 1u || nbr.fcStart.size() != (size_t)nf + 1u) return;
    const int64_t h2 = fluid::H2Of(kernel.h);
    std::vector<FxVec3> vc((size_t)nc, FxVec3{0, 0, 0});                 // the smoothed cloth velocities
    std::vector<FxVec3> vf((size_t)nf, FxVec3{0, 0, 0});                 // the smoothed fluid velocities

    // PASS 1a (cloth side): v' for every non-pinned vert from the OLD state (read-only).
    for (uint32_t i = 0; i < nc; ++i) {
        const cloth::ClothParticle& pi = world.cloth[(size_t)i];
        if (pi.flags & cloth::kFlagPinned) continue;                     // pinned -> never re-encoded
        const fx vix = fxdiv(pi.pos.x - pi.prev.x, dt);
        const fx viy = fxdiv(pi.pos.y - pi.prev.y, dt);
        const fx viz = fxdiv(pi.pos.z - pi.prev.z, dt);
        fx ax = 0, ay = 0, az = 0;                                       // the pre-kDrag accumulate (int32)
        for (uint32_t s = nbr.cfStart[(size_t)i]; s < nbr.cfStart[(size_t)i + 1u]; ++s) {
            const uint32_t j = nbr.cfNeighbors[(size_t)s];
            if (j >= nf) continue;                                       // bounds-checked skip
            const fluid::FluidParticle& pj = world.fluid[(size_t)j];
            const int bin = fluid::BinOf(fluid::RadiusSq(pi.pos, pj.pos), h2, kernel.bins);
            if (bin >= kernel.bins) continue;                            // r >= h -> zero kernel weight
            const fx w = kernel.W[(size_t)bin];
            const fx vjx = fxdiv(pj.pos.x - pj.prev.x, dt);              // v_fluid re-derived (pure fn)
            const fx vjy = fxdiv(pj.pos.y - pj.prev.y, dt);
            const fx vjz = fxdiv(pj.pos.z - pj.prev.z, dt);
            ax += fxmul(vjx - vix, w);
            ay += fxmul(vjy - viy, w);
            az += fxmul(vjz - viz, w);
        }
        vc[(size_t)i] = FxVec3{vix + fxmul(pi.invMass, fxmul(kDrag, ax)),
                               viy + fxmul(pi.invMass, fxmul(kDrag, ay)),
                               viz + fxmul(pi.invMass, fxmul(kDrag, az))};
    }
    // PASS 1b (fluid side): the mirror gather over fcNeighbors.
    for (uint32_t i = 0; i < nf; ++i) {
        const fluid::FluidParticle& pi = world.fluid[(size_t)i];
        if (pi.flags & fluid::kFlagStatic) continue;                     // static -> never re-encoded
        const fx vix = fxdiv(pi.pos.x - pi.prev.x, dt);
        const fx viy = fxdiv(pi.pos.y - pi.prev.y, dt);
        const fx viz = fxdiv(pi.pos.z - pi.prev.z, dt);
        fx ax = 0, ay = 0, az = 0;
        for (uint32_t s = nbr.fcStart[(size_t)i]; s < nbr.fcStart[(size_t)i + 1u]; ++s) {
            const uint32_t j = nbr.fcNeighbors[(size_t)s];
            if (j >= nc) continue;                                       // bounds-checked skip
            const cloth::ClothParticle& pj = world.cloth[(size_t)j];
            const int bin = fluid::BinOf(fluid::RadiusSq(pi.pos, pj.pos), h2, kernel.bins);
            if (bin >= kernel.bins) continue;
            const fx w = kernel.W[(size_t)bin];
            const fx vjx = fxdiv(pj.pos.x - pj.prev.x, dt);              // v_cloth re-derived (pure fn)
            const fx vjy = fxdiv(pj.pos.y - pj.prev.y, dt);
            const fx vjz = fxdiv(pj.pos.z - pj.prev.z, dt);
            ax += fxmul(vjx - vix, w);
            ay += fxmul(vjy - viy, w);
            az += fxmul(vjz - viz, w);
        }
        vf[(size_t)i] = FxVec3{vix + fxmul(pi.invMass, fxmul(kDrag, ax)),
                               viy + fxmul(pi.invMass, fxmul(kDrag, ay)),
                               viz + fxmul(pi.invMass, fxmul(kDrag, az))};
    }
    // PASS 2: re-encode BOTH pools (vel = v'; prev = pos - v'*dt). Pinned / static untouched.
    for (uint32_t i = 0; i < nc; ++i) {
        cloth::ClothParticle& p = world.cloth[(size_t)i];
        if (p.flags & cloth::kFlagPinned) continue;
        p.vel = vc[(size_t)i];
        p.prev = FxVec3{p.pos.x - fxmul(vc[(size_t)i].x, dt),
                        p.pos.y - fxmul(vc[(size_t)i].y, dt),
                        p.pos.z - fxmul(vc[(size_t)i].z, dt)};
    }
    for (uint32_t i = 0; i < nf; ++i) {
        fluid::FluidParticle& p = world.fluid[(size_t)i];
        if (p.flags & fluid::kFlagStatic) continue;
        p.vel = vf[(size_t)i];
        p.prev = FxVec3{p.pos.x - fxmul(vf[(size_t)i].x, dt),
                        p.pos.y - fxmul(vf[(size_t)i].y, dt),
                        p.pos.z - fxmul(vf[(size_t)i].z, dt)};
    }
}

// ----- StepClothFluid: THE COMPOSED COUPLED STEP (the GF4 StepCGF mold over cloth + fluid) -----------------
// (1) fluid::StepFluid VERBATIM (FL4/FL7-family own physics, UNCHANGED); (2) cloth::StepClothSelf VERBATIM
// (CL3+CL4+CL7 own physics, UNCHANGED); (3) IDENTITY-AT-ZERO: kDrag == 0 AND contact off -> EARLY RETURN
// (both pools bit-identical to their uncoupled steps — the off-switch contract, proven in cfl_test);
// (4) the cross lists from the post-step positions; (5) the cross-pool CONTACT projection (positional);
// (6) the TWO-WAY DRAG (derives v = (pos-prev)/dt — INCLUDING the step-(5) contact push — exchanges, and
// re-encodes prev on both pools; the GF4 lesson: velocity couplings run ONCE, POST the position work).
// Pure integer, fixed op order -> two runs bit-identical, cross-platform bit-identical.
inline void StepClothFluid(CFLWorld& world, const CFLScene& scene, fx kDrag) {
    // (1) the fluid's own step (FL4 + optional FL7 viscosity), called AS-IS. scene.visc == 0 ->
    //     StepFluidVisc == StepFluid BIT-IDENTICAL (the FL7 identity-at-zero), so the uncoupled
    //     pool-invariance proof holds against plain StepFluid.
    fluid::StepFluidVisc(world.fluid, scene.kernel, scene.fluidSpheres, world.gravity, world.dt,
                         world.groundY, scene.iters, scene.visc);
    // (2) the cloth's own step (CL3/CL4/CL7), called AS-IS.
    cloth::StepClothSelf(scene.grid, world.cloth, scene.constraints, scene.excl, scene.clothSpheres,
                         world.gravity, world.dt, world.groundY, scene.iters,
                         scene.selfThickness, scene.selfIters);
    // (3) identity-at-zero: coupling fully off -> EXACT uncoupled step (zero coupling state touched).
    const fx rSum = world.clothRadius + world.fluidRadius;
    if (kDrag == 0 && rSum <= 0) return;
    // (4) the cross lists from the post-step positions (re-built each step, the GF2/GF3 discipline).
    const CFLNeighbors nbr = BuildCFLNeighbors(world);
    // (5) the cross-pool contact projection, `contactIters` Jacobi iterations (each re-gathers from the
    //     CURRENT positions over the per-step candidate lists — the CL7 selfIters discipline; no-op when
    //     rSum <= 0).
    for (int k = 0; k < scene.contactIters; ++k)
        SolveClothFluidContact(world, nbr);
    // (6) the two-way drag + prev re-encode (no-op when kDrag == 0).
    ApplyClothFluidDrag(world, nbr, scene.kernel, kDrag);
}

// StepClothFluidSteps(world, scene, kDrag, steps): run K coupled ticks (the showcase / test driver).
inline void StepClothFluidSteps(CFLWorld& world, const CFLScene& scene, fx kDrag, int steps) {
    for (int s = 0; s < steps; ++s) StepClothFluid(world, scene, kDrag);
}

// ----- Measurement helpers (deterministic integer stats for the proofs) ------------------------------------
// CountFluidBelow(fluid, yPlane): fluid particles with pos.y < yPlane — the "passed through the cloth"
// count (the barrier proof compares it WITHOUT vs WITH coupling; porosity may leak a few — report honestly).
inline uint32_t CountFluidBelow(const std::vector<fluid::FluidParticle>& fluidP, fx yPlane) {
    uint32_t n = 0;
    for (const fluid::FluidParticle& f : fluidP)
        if (f.pos.y < yPlane) ++n;
    return n;
}

// ClothCenterY(grid, verts): the pos.y of the lattice-centre vert (r=H/2, c=W/2) — the sag stat (the
// two-way proof pins its drop under the fluid load vs the dry run). Out-of-range -> 0 (bounds-checked).
inline fx ClothCenterY(const cloth::ClothGrid& grid, const std::vector<cloth::ClothParticle>& verts) {
    const int idx = cloth::ParticleIndex(grid, grid.H / 2, grid.W / 2);
    if (idx < 0 || (size_t)idx >= verts.size()) return 0;
    return verts[(size_t)idx].pos.y;
}

// TotalDynamicMomentum(world): the summed velocity over all DYNAMIC particles of BOTH pools (int64 per
// axis). With the scene convention invMass == kOne (unit mass) for every dynamic particle this IS the
// total momentum (mass 1 each) — the drag-drift proof metric. Pinned/static are excluded (infinite mass,
// they hold). Documented limit: a mass-weighted variant for non-unit masses is future work.
struct CFLMomentum { int64_t x = 0, y = 0, z = 0; };
inline CFLMomentum TotalDynamicMomentum(const CFLWorld& world) {
    CFLMomentum m;
    for (const cloth::ClothParticle& p : world.cloth) {
        if (p.flags & cloth::kFlagPinned) continue;
        m.x += (int64_t)p.vel.x; m.y += (int64_t)p.vel.y; m.z += (int64_t)p.vel.z;
    }
    for (const fluid::FluidParticle& f : world.fluid) {
        if (f.flags & fluid::kFlagStatic) continue;
        m.x += (int64_t)f.vel.x; m.y += (int64_t)f.vel.y; m.z += (int64_t)f.vel.z;
    }
    return m;
}

// ===== LOCKSTEP + ROLLBACK over the coupled cloth+fluid world (the GF5 RunCGFLockstep twin) ===============
// A CFLCommand is the deterministic per-tick INPUT a netcode layer would put on the wire (NOT full state):
// kCmdClothWind adds arg (a delta-velocity) to a cloth vert (a gust — pinned verts hold); kCmdFluidJet adds
// arg to a fluid particle (a jet — static fluid holds). Out-of-range target / unknown kind -> a no-op
// (deterministic). Streams are processed in ARRAY ORDER per tick (the deterministic-order contract).

inline constexpr uint32_t kCmdClothWind = 0u;   // arg added to target CLOTH vert's velocity (a gust)
inline constexpr uint32_t kCmdFluidJet  = 1u;   // arg added to target FLUID particle's velocity (a jet)

struct CFLCommand {
    uint32_t tick   = 0;   // the tick this input applies on
    uint32_t kind   = 0;   // kCmdClothWind / kCmdFluidJet
    uint32_t target = 0;   // the target index (a cloth vert for wind; a fluid index for jet)
    FxVec3   arg;          // the Q16.16 payload (delta-velocity)
};

// ApplyCFLCommand(world, c): apply ONE input command (pure integer adds; pinned cloth / static fluid are
// never mutated; out-of-range / unknown kind -> no-op). The couple_gf.h::ApplyCGFCommand twin.
inline void ApplyCFLCommand(CFLWorld& world, const CFLCommand& c) {
    if (c.kind == kCmdClothWind) {
        if (c.target >= (uint32_t)world.cloth.size()) return;    // out-of-range -> no-op
        cloth::ClothParticle& p = world.cloth[(size_t)c.target];
        if (p.flags & cloth::kFlagPinned) return;                // pinned verts hold
        p.vel.x += c.arg.x; p.vel.y += c.arg.y; p.vel.z += c.arg.z;
    } else if (c.kind == kCmdFluidJet) {
        if (c.target >= (uint32_t)world.fluid.size()) return;    // out-of-range -> no-op
        fluid::FluidParticle& f = world.fluid[(size_t)c.target];
        if (f.flags & fluid::kFlagStatic) return;                // static fluid holds
        f.vel.x += c.arg.x; f.vel.y += c.arg.y; f.vel.z += c.arg.z;
    }
    // unknown kind -> a no-op (deterministic).
}

// SimCFLTick: (1) apply ALL commands whose .tick == tick, in ARRAY ORDER; (2) one StepClothFluid. Pure
// integer, fixed order -> bit-identical on every peer/platform. The couple_gf.h::SimCGFTick twin.
inline void SimCFLTick(CFLWorld& world, const CFLScene& scene, fx kDrag,
                       const std::vector<CFLCommand>& stream, uint32_t tick) {
    for (const CFLCommand& c : stream)
        if (c.tick == tick) ApplyCFLCommand(world, c);
    StepClothFluid(world, scene, kDrag);
}

// CFLSnapshot: the TWO-POOL snapshot — a deep copy of BOTH the cloth AND the fluid vectors (the rollback
// primitive across the coupled state). The couple_gf.h::CGFSnapshot twin.
struct CFLSnapshot {
    std::vector<cloth::ClothParticle> cloth;
    std::vector<fluid::FluidParticle> fluid;
};

inline CFLSnapshot SnapshotCFL(const CFLWorld& world) {
    CFLSnapshot s;
    s.cloth = world.cloth;   // value copy: deep-copies the cloth verts
    s.fluid = world.fluid;   // value copy: deep-copies the fluid particles
    return s;
}

inline void RestoreCFL(CFLWorld& world, const CFLSnapshot& snap) {
    world.cloth = snap.cloth;
    world.fluid = snap.fluid;
}

// RunCFLLockstep(init, scene, kDrag, stream, ticks): THE peer entry point. Run `ticks` SimCFLTicks from a
// COPY of init, applying the command stream -> the final coupled state. authority == replica fed the SAME
// init + stream (inputs ONLY) -> BIT-IDENTICAL by determinism (the proof memcmps BOTH pools). The
// couple_gf.h::RunCGFLockstep twin.
inline CFLWorld RunCFLLockstep(const CFLWorld& init, const CFLScene& scene, fx kDrag,
                               const std::vector<CFLCommand>& stream, int ticks) {
    CFLWorld world = init;
    for (int t = 0; t < ticks; ++t)
        SimCFLTick(world, scene, kDrag, stream, (uint32_t)t);
    return world;
}

// RunCFLRollback: (1) advance 0..mispredictTick with the auth stream; (2) SNAPSHOT (both pools);
// (2b) speculatively advance <=3 ticks with the MISPREDICTED stream (the diverging client prediction);
// (3) restore + re-simulate mispredictTick..ticks with the CORRECT stream -> the corrected final state
// (proven == RunCFLLockstep(auth) while the mispredicted state DIFFERED). The RunCGFRollback twin.
inline CFLWorld RunCFLRollback(const CFLWorld& init, const CFLScene& scene, fx kDrag,
                               const std::vector<CFLCommand>& authStream,
                               const std::vector<CFLCommand>& mispredictStream, int ticks,
                               int mispredictTick) {
    CFLWorld world = init;
    for (int t = 0; t < mispredictTick; ++t)
        SimCFLTick(world, scene, kDrag, authStream, (uint32_t)t);
    const CFLSnapshot snap = SnapshotCFL(world);
    int specTicks = ticks - mispredictTick;
    if (specTicks > 3) specTicks = 3;
    for (int s = 0; s < specTicks; ++s)
        SimCFLTick(world, scene, kDrag, mispredictStream, (uint32_t)(mispredictTick + s));
    RestoreCFL(world, snap);
    for (int t = mispredictTick; t < ticks; ++t)
        SimCFLTick(world, scene, kDrag, authStream, (uint32_t)t);
    return world;
}

}  // namespace cfl
}  // namespace hf::sim
