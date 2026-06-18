#pragma once
// Slice GR1 — Deterministic GPU Granular/Sand: Q16.16 fixed-point GRAIN POOL INTEGRATOR + dropped block
// (the BEACHHEAD of FLAGSHIP #10: DETERMINISTIC GPU GRANULAR / SAND via Position-Based granular dynamics —
// the 4th member of the deterministic-sim family (rigid FPX -> cloth -> fluid -> GRAIN), adding the one
// physics the trilogy never modeled: dry friction / shear (angle-of-repose). GR1 is ONLY the integer grain
// POOL + gravity integrate + radius-aware ground rest — the integrator beachhead. NO neighbors (GR2), NO
// frictionless contact solve (GR3), NO Coulomb friction (GR4 — the new physics), NO lockstep (GR5), NO float
// render (GR6), NO float on the bit-exact path. Pure CPU, header-only, NO device, NO backend symbols, NO
// <cmath>. Namespace hf::sim::grain. The whole flagship reuses the proven engine/sim/fpx.h Q16.16 toolbox +
// the FL1-FL6 / CL1-CL6 mold. The STRUCTURAL TWIN of the FL1/CL1 integer beachhead (engine/sim/fluid.h /
// engine/sim/cloth.h): a pure-integer per-particle update proven GPU==CPU BIT-EXACT, with a cross-backend
// BIT-IDENTICAL integer golden — over a 3D grain BLOCK, with a first-class `radius` field.
//
// TWO DELTAS vs the FluidParticle twin the GR1 spec locks (everything else is the IntegrateFluidParticle
// body verbatim): (1) a first-class `radius` field -> 48-byte std430 packing (carried for GR3 contact /
// GR4 friction; GR1 uses it ONLY for the ground rest); (2) a RADIUS-AWARE ground rest — the grain's SURFACE
// rests on the floor (pos.y < groundY + radius -> groundY + radius), not its center.
//
// The integrator shader shaders/grain_integrate.comp.hlsl copies IntegrateGrainParticle's per-particle math
// VERBATIM (the fxmul + integrate + prev-snap + radius-aware floor-clamp), so tests/grain_test.cpp + the GPU
// pass exercise the EXACT math — which is what makes the integrated grain array bit-identical GPU==CPU AND
// cross-backend.
//
// THE CROSS-BACKEND CRUX (the make-or-break for GPU==CPU, like fluid.h's host-snapped FluidParticle array):
// the GPU consumes host-snapped Q16.16 INTEGERS (the GrainParticle array) and does ZERO floating point —
// every step is `(int64)a*b >> kFrac` (an ARITHMETIC right shift on int64, deterministic + identical on
// every compiler/vendor) + integer add + integer compare. In GR1 each grain is INDEPENDENT (no inter-grain
// neighbours/contact/friction until GR2-GR4), so the GPU per-thread write is order-independent / race-free
// with NO atomics, and two runs are byte-identical.
//
// THE int32-vs-int64 DECISION (the FL1/CL1 lesson, documented): the integrate is `vel += gravity*dt;
// pos += vel*dt`, both componentwise fxmul — the SAME form as fluid_integrate.comp / cloth_integrate.comp /
// fpx_integrate.comp, which needed int64 because the (int64)a*b product before the >>kFrac shift exceeds
// int32 for Q16.16 gravity*dt (gravity ≈ -9.8*65536 = -642253; products of two Q16.16 world-scale values
// blow past 2^31). To stay bit-exact to this int64-intermediate reference WITHOUT any overflow fragility,
// shaders/grain_integrate.comp.hlsl uses int64 (like fluid_integrate.comp) and is therefore VULKAN-SPIR-V-
// ONLY (glslc — the Metal HLSL->SPIR-V->MSL frontend — cannot parse int64_t in HLSL), NOT in the Metal
// hf_gen_msl list; the Metal --grain-integrate showcase runs the CPU grain::IntegrateGrains (the SAME
// bit-exact reference the Vulkan GPU==CPU memcmp compares against) -> byte-identical to the Vulkan GPU
// result BY CONSTRUCTION. Same established convention as fluid_integrate.comp (fluid.h:28-37).
//
// REUSE MAP (file:line): the Q16.16 toolbox is engine/sim/fpx.h — fx (int32 Q16.16, fpx.h:46), fxmul
// (fpx.h:54, the int64-intermediate multiply), FxVec3 + FxAdd/FxSub/FxScale (fpx.h:59-72), kOne/kFrac
// (fpx.h:47-48). The per-particle integrate mirrors fluid.h::IntegrateFluidParticle (fluid.h:148-165) with
// the radius-aware ground rest — READ, NOT modified (grain is the additive sibling; fpx.h #included
// read-only + stays byte-unchanged, exactly as fluid.h:52 / cloth.h do).

#include <cstdint>
#include <vector>

#include "sim/fpx.h"   // read-only: fx / fxmul / FxVec3 / FxAdd / FxSub / FxScale / kOne / kFrac

namespace hf::sim {
namespace grain {

// Reuse the fpx Q16.16 scalar + vector toolbox verbatim (NO new fixed-point primitives).
using fpx::fx;
using fpx::FxVec3;
using fpx::fxmul;
using fpx::FxAdd;
using fpx::FxSub;
using fpx::FxScale;
inline constexpr int kFrac = fpx::kFrac;     // Q16.16 fractional bits (== fpx::kFrac, MUST match the shader)
inline constexpr fx  kOne  = fpx::kOne;      // 1.0 in Q16.16 (65536)

// ----- The grain particle (the std430 GPU mirror; the FluidParticle packing discipline + the radius delta) -
// A single grain. std430-packable as plain int32s (pos.xyz, prev.xyz, vel.xyz, invMass, radius, flags) =
// 12 x 4-byte = 48 bytes, NO padding holes (memcmp-able; the GPU GrainParticle mirror). IDENTICAL to
// fluid.h::FluidParticle (44 bytes) PLUS the first-class `radius` field (the ONE packing delta, +4 bytes ->
// 48). Treats FxVec3 as 3 plain int32s (NOT a 16-byte-aligned vec3), so array stride 48 is a multiple of the
// 4-byte scalar alignment.
//   * pos     : current Q16.16 world position.
//   * prev    : the PREVIOUS position (prev = pos BEFORE the position integrate each step) — GR2-GR4's
//     neighbour/contact pass needs it (Verlet/predicted-position anchor); GR1 maintains it so the buffer
//     layout is FINAL from the beachhead (no struct churn in GR2-GR6).
//   * vel     : Q16.16 velocity (world units / second).
//   * invMass : Q16.16 inverse mass (0 => infinite mass / STATIC, never integrates). Carried for GR3/GR4.
//   * radius  : the Q16.16 grain radius (carried for GR3 contact / GR4 friction; GR1 uses it ONLY for the
//     radius-aware ground rest — the grain's SURFACE rests on the floor).
//   * flags   : bit0 = STATIC (a future boundary/wall grain — invMass 0, never moves). Reserved in GR1.
struct GrainParticle {
    FxVec3   pos;             // Q16.16 current position
    FxVec3   prev;            // Q16.16 previous position (prev = pos before the position integrate)
    FxVec3   vel;             // Q16.16 velocity (world units / second)
    fx       invMass = 0;     // Q16.16 inverse mass (0 => static / infinite mass)
    fx       radius  = 0;     // Q16.16 grain radius (the GR1 ground-rest input; GR3/GR4 contact/friction)
    uint32_t flags   = 0;     // bit0 = STATIC (reserved for GR4 boundary grains)
};

inline constexpr uint32_t kFlagStatic = 1u;   // bit0: the grain is a fixed boundary (never integrates)

// ----- The dropped-block config: a fixed W x H x D lattice of grains above the ground --------------------
// A solid W x H x D block of grains at a spacing, with its corner at `origin` (above the ground), uniform
// `radius`. The float layout constants are host-snapped to Q16.16 once at build (NOT in the per-step integer
// sim). GR1 picks a modest count (10x10x10 = 1000) — well under any budget; keep the SAME scene so it
// survives every later GR slice. spacing >= 2*radius so the initial block is non-overlapping (clean hand-off
// to GR3's contact solve).
struct GrainBlock {
    int    W = 0, H = 0, D = 0;   // block dims (x columns, y rows, z layers)
    fx     spacing = 0;           // Q16.16 spacing between adjacent grains (>= 2*radius -> non-overlapping)
    fx     radius  = 0;           // Q16.16 uniform grain radius
    FxVec3 origin;                // Q16.16 world position of grain (0,0,0) — the block's corner
};

// Index of grain (ix, iy, iz) in the x-major / y-mid / z-minor grain array (the deterministic, fixed
// traversal order the shader's one-thread-per-grain dispatch + the CPU reference share).
inline int GrainIndex(const GrainBlock& block, int ix, int iy, int iz) {
    return (iz * block.H + iy) * block.W + ix;
}

// ----- InitGrainBlock: the deterministic W x H x D dropped block (all dynamic, uniform radius, at rest) ---
// Builds a solid block of grains: grain (ix,iy,iz) at origin + (ix*spacing, iy*spacing, iz*spacing). All
// grains start at rest (vel 0, prev == pos), DYNAMIC (invMass = kOne, flags 0), uniform radius — GR1 has no
// boundary/static grains yet (that is GR4). The block sits in a corner ABOVE the ground so it FALLS and piles
// at the ground. Pure integer (the spacing/origin are already host-snapped Q16.16; this only adds integer
// multiples of spacing). Returns the populated grain vector (size W*H*D), in the GrainIndex traversal order.
inline std::vector<GrainParticle> InitGrainBlock(const GrainBlock& block) {
    std::vector<GrainParticle> grains((size_t)(block.W * block.H * block.D));
    for (int iz = 0; iz < block.D; ++iz) {
        for (int iy = 0; iy < block.H; ++iy) {
            for (int ix = 0; ix < block.W; ++ix) {
                GrainParticle p;
                // Host-snapped layout: integer multiples of the (already Q16.16) spacing -> exact, no float.
                p.pos = FxVec3{block.origin.x + (fx)(ix * (int)block.spacing),
                               block.origin.y + (fx)(iy * (int)block.spacing),
                               block.origin.z + (fx)(iz * (int)block.spacing)};
                p.prev    = p.pos;
                p.vel     = FxVec3{0, 0, 0};
                p.invMass = kOne;          // unit mass dynamic (GR1 has no static/boundary grains)
                p.radius  = block.radius;  // the uniform grain radius
                p.flags   = 0;
                grains[(size_t)GrainIndex(block, ix, iy, iz)] = p;
            }
        }
    }
    return grains;
}

// ----- IntegrateGrainParticle: the deterministic per-grain semi-implicit-Euler step (SHADER math) --------
// For a single grain, if NOT static (!(flags & kFlagStatic)):
//   vel += gravity * dt   (component-wise fxmul)            — the FL1/CL1 velocity integrate
//   prev = pos                                              — snapshot before the position move (GR2-GR4)
//   pos += vel * dt        (component-wise fxmul)            — the FL1/CL1 position integrate
//   RADIUS-AWARE non-penetration ground rest: if (pos.y < groundY + radius) { pos.y = groundY + radius;
//     if (vel.y < 0) vel.y = 0; }   — the grain's SURFACE rests on the floor (groundY + radius is a trivial
//     Q16.16 integer add); sets up GR3's collider projection. radius 0 reduces to the FL1 plain ground clamp.
// Static grains are UNTOUCHED. Pure integer; fixed op order; no RNG, no clock. Each grain is INDEPENDENT of
// every other (no inter-grain coupling in GR1), so the order over grains does NOT matter -> two-run
// bit-identical AND the GPU per-thread write is race-free with NO atomics. The shader runs THIS exact
// per-grain body. The IntegrateFluidParticle (fluid.h:148-165) body verbatim, with the radius-aware clamp.
inline void IntegrateGrainParticle(GrainParticle& p, const FxVec3& gravity, fx groundY, fx dt) {
    if (p.flags & kFlagStatic) return;
    // (1) integrate velocity: vel += gravity * dt.
    p.vel.x += fxmul(gravity.x, dt);
    p.vel.y += fxmul(gravity.y, dt);
    p.vel.z += fxmul(gravity.z, dt);
    // (2) snapshot the previous position (the predicted-position anchor GR2-GR4 reads).
    p.prev = p.pos;
    // (3) integrate position: pos += vel * dt.
    p.pos.x += fxmul(p.vel.x, dt);
    p.pos.y += fxmul(p.vel.y, dt);
    p.pos.z += fxmul(p.vel.z, dt);
    // (4) RADIUS-AWARE ground rest (no restitution — GR1 is free-fall + ground only). The grain's surface
    // rests on the floor: clamp the CENTER to groundY + radius.
    const fx restY = groundY + p.radius;
    if (p.pos.y < restY) {
        p.pos.y = restY;
        if (p.vel.y < 0) p.vel.y = 0;
    }
}

// ----- IntegrateGrains: one integrate STEP over the whole grain pool ------------------------------------
// Apply IntegrateGrainParticle to every grain once. The make-or-break reference the GPU memcmp's against.
// Order-independent (grains are independent in GR1) -> bit-identical regardless of GPU scheduling.
inline void IntegrateGrains(std::vector<GrainParticle>& grains, const FxVec3& gravity, fx dt, fx groundY) {
    const size_t n = grains.size();
    for (size_t i = 0; i < n; ++i)
        IntegrateGrainParticle(grains[i], gravity, groundY, dt);
}

// ----- IntegrateGrainSteps: run K integrate steps (the showcase / GPU K-step driver) --------------------
// K successive IntegrateGrains steps over the pool. The GPU grain_integrate.comp runs THIS exact K-step loop
// per thread (one thread per grain, since grains are independent in GR1).
inline void IntegrateGrainSteps(std::vector<GrainParticle>& grains, const FxVec3& gravity, fx dt, fx groundY,
                                int steps) {
    for (int s = 0; s < steps; ++s)
        IntegrateGrains(grains, gravity, dt, groundY);
}

// CountAtGround(grains, groundY): the deterministic count of grains resting at the ground floor (the grain's
// surface on the floor: pos.y == groundY + radius) — a reporting/stat helper for the showcase coverage proof.
// Pure integer compare -> bit-exact CPU<->GPU.
inline int CountAtGround(const std::vector<GrainParticle>& grains, fx groundY) {
    int n = 0;
    for (const GrainParticle& p : grains)
        if (p.pos.y == groundY + p.radius) ++n;
    return n;
}

// ===== Slice GR2 — the GRID-HASH NEIGHBOR SEARCH (the candidate contact-pair list GR3's solve iterates) ==
// Builds the per-grain CANDIDATE NEIGHBOR LIST over the GR1 grain pool via a uniform spatial-hash grid, with
// the proven count->scan->emit compaction (the FL2 fluid_cell_*/fluid_neighbor_* / FPX2 / CL2 twin). PURE
// INT32 -> the GR2 shaders MSL-generate NATIVELY (a true GPU pass on BOTH backends, unlike GR1's int64
// integrate): the whole neighbor search is integer INDEX arithmetic. Cell ids are FloorDiv per axis (the cell
// size is hSearch); the cell bucketing is count->scan->emit of grain indices; the candidate reject is a
// per-axis |pos_i.axis - pos_j.axis| < hSearch compare (fx is int32 -> a PURE INT32 compare, NO squaring, NO
// products, NO int64, NO sqrt). The exact radial overlap cull (centre distance < r_i + r_j) is DEFERRED to
// GR3's contact solve (the contact projection is a no-op for non-overlapping candidate pairs, exactly FL2's
// "over-inclusive box candidate, exact cull deferred to FL3" discipline). NO float, NO sqrt, NO int64.
//
// THE CONTACT SEARCH RADIUS `hSearch` (the one parameter delta vs FL2): the range within which two grains are
// CANDIDATE contact pairs. For uniform radius r, two grains contact when centre distance < 2*r (the
// diameter), so the search radius MUST be >= the contact diameter (a smaller radius would MISS real contacts
// in GR3) -> the caller asserts hSearch >= 2*maxRadius. The GR1 showcase block has spacing 1.0 + radius 0.25
// (diameter 0.5); the showcase picks hSearch = 1.5 (== 1.5 * spacing, a host-snapped Q16.16 constant), which
// is >= the diameter AND makes the non-overlapping block a NON-DEGENERATE neighbor graph (each interior grain
// sees its 26 lattice neighbours -> a rich "interior dense / surface sparse" heat viz).
//
// REUSE: fpx::FloorDiv (the deterministic floor-division for negative coords, fpx.h:177), fpx::FxCell
// (fpx.h:183), fpx::CellId (fpx.h:196). The whole search is engine/sim/fluid.h::MakeGrid/BuildCellTable/
// NeighborAccept/BuildNeighborList with Grain types + hSearch for the radius + GrainParticle.pos as position.

using fpx::FloorDiv;     // read-only: the deterministic floor-division (correct for negative coords)
using fpx::FxCell;       // read-only: the int3 cell coordinate
using fpx::CellId;       // read-only: the flat cell linearization

// ----- The bounded dense grid over the grain AABB (the FL2 cell-linearization scheme verbatim) -----------
// CHOSEN SCHEME: a BOUNDED DENSE GRID (== fluid.h::FluidGrid). Cell-size = hSearch (the contact search
// radius). A grain's cell coord is FloorDiv(pos.axis, hSearch) per axis (monotone across 0 for negatives).
// The grid covers [cellMin, cellMin+gridDim) cells; a cell's flat id is fpx::CellId of (coord - cellMin) into
// gridDim. The caller sizes the grid to the grain AABB (every grain's cell in [0,gridDim)), so the
// linearization is total + collision-free + deterministic. The origin offset cellMin lets the grid sit at any
// world location (incl. negative coords) deterministically.
struct GrainGrid {
    fx     hSearch = 0;    // Q16.16 cell size (== the contact search radius)
    FxCell cellMin;        // the integer cell coord of the grid's (0,0,0) corner (the AABB lower cell)
    FxCell gridDim;        // the grid extent in cells per axis (cellCount = x*y*z)
};

// GrainCellOf(pos, hSearch): the integer grid cell a grain's position falls in, FloorDiv per axis. Pure int32.
inline FxCell GrainCellOf(const FxVec3& pos, fx hSearch) {
    return FxCell{FloorDiv(pos.x, hSearch), FloorDiv(pos.y, hSearch), FloorDiv(pos.z, hSearch)};
}

// FlatGrainCellId(cell, grid): the flat id of an absolute cell coord into the bounded dense grid (offset by
// cellMin into [0,gridDim), then fpx::CellId). The caller guarantees the cell is in range (the grid was sized
// to the grain AABB); returns the linear cell index in [0, gridDim.x*y*z).
inline uint32_t FlatGrainCellId(const FxCell& cell, const GrainGrid& grid) {
    const FxCell local{cell.x - grid.cellMin.x, cell.y - grid.cellMin.y, cell.z - grid.cellMin.z};
    return CellId(local, grid.gridDim);
}

// GrainCellCount(grid): the total number of cells in the dense grid (gridDim.x * y * z).
inline uint32_t GrainCellCount(const GrainGrid& grid) {
    return (uint32_t)(grid.gridDim.x * grid.gridDim.y * grid.gridDim.z);
}

// MakeGrainGrid(grains, hSearch): build the bounded dense grid that tightly covers the grain pool at cell-size
// hSearch. cellMin = the min cell coord over all grains; gridDim = (maxCell - minCell + 1) per axis. Empty
// pool -> a 1x1x1 grid at origin (deterministic degenerate). Pure int32 (== fluid.h::MakeGrid).
inline GrainGrid MakeGrainGrid(const std::vector<GrainParticle>& grains, fx hSearch) {
    GrainGrid grid;
    grid.hSearch = hSearch;
    if (grains.empty()) {
        grid.cellMin = FxCell{0, 0, 0};
        grid.gridDim = FxCell{1, 1, 1};
        return grid;
    }
    FxCell lo = GrainCellOf(grains[0].pos, hSearch);
    FxCell hi = lo;
    for (const GrainParticle& p : grains) {
        const FxCell c = GrainCellOf(p.pos, hSearch);
        if (c.x < lo.x) lo.x = c.x; if (c.x > hi.x) hi.x = c.x;
        if (c.y < lo.y) lo.y = c.y; if (c.y > hi.y) hi.y = c.y;
        if (c.z < lo.z) lo.z = c.z; if (c.z > hi.z) hi.z = c.z;
    }
    grid.cellMin = lo;
    grid.gridDim = FxCell{hi.x - lo.x + 1, hi.y - lo.y + 1, hi.z - lo.z + 1};
    return grid;
}

// ----- BuildGrainCellTable: bucket grain indices into cells (the count->scan->emit on grains) -------------
// The CSR-style cell table: cellStart[c..] is the exclusive prefix-sum of per-cell counts (cellStart has
// cellCount+1 entries; cellStart[c]..cellStart[c+1] is cell c's slice), and cellGrains[] holds the grain
// indices grouped by cell, ASCENDING grain index within each cell (deterministic). This is count->scan->emit:
// (1) count grains per cell; (2) exclusive prefix-sum -> cellStart; (3) scatter each grain index into its
// cell's slice (the emit, ascending-index order by construction since the grain loop is ascending). Pure
// int32 -> the GPU grain_cell_{count,scan,emit} mirror this byte-for-byte. (DET-CRUX, the FL2 lesson: the
// EMIT is the single-thread ascending-grain scatter — a parallel atomic cursor would make the within-cell
// order GPU-scheduling-dependent -> non-deterministic. The cell COUNT + the neighbor passes are
// per-grain-disjoint and race-free; only the cell-emit scatter is the ordered pass.)
struct GrainCellTable {
    std::vector<uint32_t> cellStart;    // cellCount+1 exclusive prefix-sum offsets (CSR row pointers)
    std::vector<uint32_t> cellGrains;   // grain indices grouped by cell (size == grain count)
};

inline GrainCellTable BuildGrainCellTable(const std::vector<GrainParticle>& grains, const GrainGrid& grid) {
    const uint32_t n = (uint32_t)grains.size();
    const uint32_t cells = GrainCellCount(grid);
    GrainCellTable table;
    // (1) COUNT: per-cell grain count.
    std::vector<uint32_t> counts((size_t)cells, 0u);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t c = FlatGrainCellId(GrainCellOf(grains[i].pos, grid.hSearch), grid);
        ++counts[c];
    }
    // (2) SCAN: exclusive prefix-sum -> cellStart (cellCount+1 entries; the last == n).
    table.cellStart.assign((size_t)cells + 1u, 0u);
    uint32_t running = 0;
    for (uint32_t c = 0; c < cells; ++c) {
        table.cellStart[c] = running;
        running += counts[c];
    }
    table.cellStart[cells] = running;   // == n (the total)
    // (3) EMIT: scatter each grain index into its cell's slice (ascending index by the ascending loop).
    table.cellGrains.assign((size_t)n, 0u);
    std::vector<uint32_t> cursor((size_t)cells, 0u);   // per-cell write cursor (local offset)
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t c = FlatGrainCellId(GrainCellOf(grains[i].pos, grid.hSearch), grid);
        table.cellGrains[table.cellStart[c] + cursor[c]] = i;
        ++cursor[c];
    }
    return table;
}

// ----- The neighbor reject (the PURE INT32 per-axis |dx| < hSearch candidate test) -----------------------
// GrainNeighborAccept(a, b, hSearch): accept b as a candidate neighbor of a iff |a.axis - b.axis| < hSearch on
// EVERY axis (a box, NOT a sphere — the over-inclusive candidate set GR3's contact solve culls). PURE INT32:
// an integer subtract + abs + compare per axis, NO products, NO int64, NO sqrt. The shader copies THIS
// verbatim. (== fluid.h::NeighborAccept with hSearch for h.)
inline bool GrainNeighborAccept(const FxVec3& a, const FxVec3& b, fx hSearch) {
    fx dx = a.x - b.x; if (dx < 0) dx = -dx;
    fx dy = a.y - b.y; if (dy < 0) dy = -dy;
    fx dz = a.z - b.z; if (dz < 0) dz = -dz;
    return dx < hSearch && dy < hSearch && dz < hSearch;
}

// ----- BuildGrainNeighborList: per-grain candidates over the 27-cell stencil (count->scan->emit) ----------
// For each grain i, scan the 27 cells of its 3x3x3 stencil (the cell + its 26 neighbors); for each grain
// j != i in those cells, accept iff GrainNeighborAccept(pos_i, pos_j, hSearch). Emit the accepted j into
// neighbors[] at i's offset, in a FIXED order: ascending stencil-cell (dz,dy,dx -1..+1), then within a cell
// ascending j (cellGrains is already ascending-index per cell) -> fully deterministic. The variable-length
// per-grain lists are laid out by count->scan->emit (neighborStart = exclusive prefix-sum; neighbors[]
// grouped by i). Stencil cells outside the grid are skipped (clamped). Pure int32 (== fluid.h::
// BuildNeighborList).
struct GrainNeighborList {
    std::vector<uint32_t> neighborStart;   // grainCount+1 exclusive prefix-sum offsets (CSR)
    std::vector<uint32_t> neighbors;       // candidate neighbor j indices grouped by i (in stencil order)
};

// CountGrainNeighbors(grains, grid, table, hSearch, perGrainOut): the count pass. perGrainOut[i] = #candidate
// neighbors of i (j!=i in the 27-cell stencil passing GrainNeighborAccept); returns the total. The GPU
// grain_neighbor_count mirrors THIS per-thread (one thread per grain i).
inline uint32_t CountGrainNeighbors(const std::vector<GrainParticle>& grains, const GrainGrid& grid,
                                    const GrainCellTable& table, fx hSearch,
                                    std::vector<uint32_t>& perGrainOut) {
    const uint32_t n = (uint32_t)grains.size();
    perGrainOut.assign((size_t)n, 0u);
    uint32_t total = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const FxCell ci = GrainCellOf(grains[i].pos, grid.hSearch);
        uint32_t c = 0;
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const FxCell nc{ci.x + dx, ci.y + dy, ci.z + dz};
            // Skip stencil cells outside the bounded grid (clamp).
            if (nc.x < grid.cellMin.x || nc.x >= grid.cellMin.x + grid.gridDim.x) continue;
            if (nc.y < grid.cellMin.y || nc.y >= grid.cellMin.y + grid.gridDim.y) continue;
            if (nc.z < grid.cellMin.z || nc.z >= grid.cellMin.z + grid.gridDim.z) continue;
            const uint32_t cell = FlatGrainCellId(nc, grid);
            for (uint32_t s = table.cellStart[cell]; s < table.cellStart[cell + 1u]; ++s) {
                const uint32_t j = table.cellGrains[s];
                if (j == i) continue;                                      // NO self-neighbor
                if (GrainNeighborAccept(grains[i].pos, grains[j].pos, hSearch)) ++c;
            }
        }
        perGrainOut[i] = c;
        total += c;
    }
    return total;
}

// BuildGrainNeighborList(grains, grid, table, hSearch): the full mesher (count->scan->emit). (1)
// CountGrainNeighbors -> per-grain counts; (2) exclusive prefix-sum -> neighborStart; (3) emit each accepted j
// into i's disjoint slice in the FIXED stencil order. The list is grouped by i (ascending), then stencil-cell
// (dz,dy,dx ascending), then j (ascending within a cell) -> fully deterministic. The GPU does the SAME three
// passes (count/scan/emit) -> the GPU neighbors[]+neighborStart memcmp's against this byte-for-byte.
inline GrainNeighborList BuildGrainNeighborList(const std::vector<GrainParticle>& grains,
                                                const GrainGrid& grid, const GrainCellTable& table,
                                                fx hSearch) {
    const uint32_t n = (uint32_t)grains.size();
    GrainNeighborList list;
    std::vector<uint32_t> counts;
    const uint32_t total = CountGrainNeighbors(grains, grid, table, hSearch, counts);
    // (2) SCAN: exclusive prefix-sum -> neighborStart (grainCount+1 entries; the last == total).
    list.neighborStart.assign((size_t)n + 1u, 0u);
    uint32_t running = 0;
    for (uint32_t i = 0; i < n; ++i) {
        list.neighborStart[i] = running;
        running += counts[i];
    }
    list.neighborStart[n] = running;   // == total
    // (3) EMIT: each grain writes its candidates into its disjoint [neighborStart[i], ..) slice.
    list.neighbors.assign((size_t)total, 0u);
    for (uint32_t i = 0; i < n; ++i) {
        const FxCell ci = GrainCellOf(grains[i].pos, grid.hSearch);
        uint32_t local = 0;
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const FxCell nc{ci.x + dx, ci.y + dy, ci.z + dz};
            if (nc.x < grid.cellMin.x || nc.x >= grid.cellMin.x + grid.gridDim.x) continue;
            if (nc.y < grid.cellMin.y || nc.y >= grid.cellMin.y + grid.gridDim.y) continue;
            if (nc.z < grid.cellMin.z || nc.z >= grid.cellMin.z + grid.gridDim.z) continue;
            const uint32_t cell = FlatGrainCellId(nc, grid);
            for (uint32_t s = table.cellStart[cell]; s < table.cellStart[cell + 1u]; ++s) {
                const uint32_t j = table.cellGrains[s];
                if (j == i) continue;
                if (GrainNeighborAccept(grains[i].pos, grains[j].pos, hSearch)) {
                    list.neighbors[list.neighborStart[i] + local] = j;
                    ++local;
                }
            }
        }
    }
    return list;
}

}  // namespace grain
}  // namespace hf::sim
