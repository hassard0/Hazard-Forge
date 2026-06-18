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

// ===== Slice GR3 — the FRICTIONLESS CONTACT PROJECTION (the FL4 Jacobi-solve twin) ======================
// Resolve grain-grain OVERLAP (non-penetration) + ground/fpx::FxBody colliders by an inverse-mass-weighted
// NORMAL push-apart over the GR2 candidate list — the FL4 JACOBI multi-thread position-based solve. The
// contact correction is per-grain INDEPENDENT in the Jacobi formulation: each grain i reads the iteration-
// START positions (read-only) and accumulates its OWN Δp_i into a SEPARATE dp[] double-buffer, then ALL
// grains apply pos_i += dp_i (the FL4 SolveDensityConstraint + apply pattern). So the GPU solve is
// [numthreads(64,1,1)] MULTI-THREAD — NO single-thread serial dispatch, NO TDR ceiling (the cloth CL3 limit
// does NOT apply; the FL4 win). The math is int64 (FxLength/FxNormalize via FxISqrt, fxmul, fxdiv) ->
// grain_contact_dp.comp + grain_collide.comp are VULKAN-SPIR-V-ONLY (glslc can't parse int64), NOT in
// hf_gen_msl; the Metal --grain-contact showcase runs THIS CPU StepGrainContact (the SAME bit-exact
// reference — byte-identical by construction, the FL4/GR1 convention). grain_contact_apply is a plain
// int32 pos+=dp add (it COULD MSL-generate) but, like fluid_apply.comp, the FL4 host-driven solve keeps it
// Vulkan-only alongside the other two solve passes.
//
// THE CONTACT CONSTRAINT (the one new bit of math — the cloth-edge / FL4-density hybrid): for an overlapping
// grain PAIR (i, j) with centre distance d = |p_i − p_j| and pen = (r_i + r_j) − d > 0, push them apart along
// n_ij = unit(p_i − p_j) by their inverse-mass-weighted share. In the per-grain Jacobi accumulate over the
// SYMMETRIC GR2 neighbor list (j in i's list AND i in j's list), each grain i handles its OWN half:
//   Δp_i += ( w_i / (w_i + w_j) ) · pen · n_ij          // w = invMass; w_i+w_j==0 (both static) -> skip
// (grain j independently accumulates Δp_j += (w_j/(w_i+w_j))·pen·n_ji from the same pair — no double-apply,
// the FL4 "each i sums over its neighbours" structure). d == 0 (coincident) -> the FxNormalize +Y fallback
// (deterministic). Only pairs with pen > 0 contribute (the EXACT radial overlap cull GR2 deferred). Static
// grains (flags bit0 / invMass 0) -> Δp = 0 (the cloth pinned / FL4 static case). int64 throughout.
//
// HONEST CAVEAT (the FL4/CL3 discipline): K Jacobi iterations leave a DETERMINISTIC-but-nonzero contact
// residual (stiffness ∝ iterations); the fxdiv truncation makes it bit-REPRODUCIBLE, NOT analytically
// non-penetrating. The claim is DETERMINISM + cross-platform bit-identity; the showcase proof reports
// peak/summed penetration RELIEVED (penAfter < penBefore), NOT zero.

using fpx::fxdiv;       // read-only: the int64 Q16.16 divide
using fpx::FxLength;    // read-only: the int64 length (FxISqrt of the sum of squares)
using fpx::FxNormalize; // read-only: the int64 normalize (+Y fallback on length 0)

// kGrainCollideEps: the surface-snap tolerance (the kFluidCollideEps twin). FxNormalize+FxScale truncate
// toward zero, so the snapped FxLength lands a few LSBs short of radius — the assert band for "no grain
// inside a collider / below the floor".
inline constexpr fx kGrainCollideEps = 16;   // ~16 Q16.16 LSBs (the FL4 snap tolerance)

// ----- GrainSphereCollider + GrainSphereFromBody: the CL4/FL4 collider bridge (radius-aware) --------------
// A static sphere collider in the SAME Q16.16 world as the grains (the grain/rigid share — the CL4 deformable-
// meets-rigid precedent). GrainSphereFromBody bridges an fpx::FxBody (its pos + radius) to a collider so the
// sand piles AROUND the SAME rigid sphere the FPX sim integrates. Pure read of fpx::FxBody.
struct GrainSphereCollider {
    FxVec3 center;        // Q16.16 world-space sphere center (== fpx::FxBody::pos)
    fx     radius = 0;    // Q16.16 sphere radius (== fpx::FxBody::radius)
};
inline GrainSphereCollider GrainSphereFromBody(const fpx::FxBody& b) {
    return GrainSphereCollider{b.pos, b.radius};
}

// ----- SolveGrainContact: the Jacobi per-grain Δp accumulate (the FL4 SolveDensityConstraint twin) --------
// Given the grain array + the GR2 neighbor list, compute Δp_i for ALL grains into a SEPARATE dp[] buffer
// (the Jacobi double-buffer — reads the iteration-start positions, NOT the in-progress dp). For each grain i
// (if NOT static): accumulate over its GR2 neighbours j where pen = (r_i+r_j) − |p_i−p_j| > 0,
//   Δp_i += ( w_i / (w_i + w_j) ) · pen · unit(p_i − p_j)
// where w = invMass (both-static w_i+w_j==0 -> skip the pair), d==0 -> the +Y FxNormalize fallback. STATIC
// grains (flags & kFlagStatic, invMass 0) -> Δp_i = 0 (the cloth pinned case). int64 (FxLength/FxNormalize/
// fxmul/fxdiv). The shader grain_contact_dp.comp copies THIS body VERBATIM. Deterministic (the fixed GR2
// neighbour order). dpOut is sized to the grain count (one Δp per grain).
inline void SolveGrainContact(const std::vector<GrainParticle>& grains, const GrainNeighborList& list,
                              std::vector<FxVec3>& dpOut) {
    const uint32_t n = (uint32_t)grains.size();
    dpOut.assign((size_t)n, FxVec3{0, 0, 0});
    for (uint32_t i = 0; i < n; ++i) {
        if (grains[(size_t)i].flags & kFlagStatic) continue;   // static -> Δp = 0 (the pinned case)
        const fx wi = grains[(size_t)i].invMass;
        FxVec3 accum{0, 0, 0};
        const uint32_t s0 = list.neighborStart[i], s1 = list.neighborStart[i + 1u];
        for (uint32_t s = s0; s < s1; ++s) {
            const uint32_t j = list.neighbors[s];
            const fx wj = grains[(size_t)j].invMass;
            const fx wsum = wi + wj;
            if (wsum == 0) continue;                           // both static -> no push
            // d = p_i − p_j ; pen = (r_i + r_j) − |d| ; only pen > 0 contributes (the exact radial cull).
            const FxVec3 d = FxSub(grains[(size_t)i].pos, grains[(size_t)j].pos);
            const fx dist = FxLength(d);
            const fx pen = (grains[(size_t)i].radius + grains[(size_t)j].radius) - dist;
            if (pen <= 0) continue;                            // non-overlapping candidate -> no-op
            // share = w_i / (w_i + w_j) ; n = unit(p_i − p_j) (d==0 -> the +Y fallback) ; scale = share·pen.
            const fx share = fxdiv(wi, wsum);
            const fx scale = fxmul(share, pen);
            const FxVec3 nrm = FxNormalize(d);
            accum.x += fxmul(scale, nrm.x);
            accum.y += fxmul(scale, nrm.y);
            accum.z += fxmul(scale, nrm.z);
        }
        dpOut[(size_t)i] = accum;
    }
}

// ----- The radius-aware grain colliders (the CL4/FL4 mold, made radius-aware) ----------------------------
// CollideGrainPlane clamps the grain's SURFACE to the floor (pos.y >= groundY + radius — the GR1 rest);
// static grains ARE clamped (a fallen boundary grain is raised). CollideGrainSphere projects the grain
// CENTRE out to sphereRadius + grainRadius (the surfaces touch); static grains are plane-clamped but NOT
// sphere-projected (they hold). int32 AABB reject (against sphereR + grainR) then the int64 FxLength/
// FxNormalize snap (d==0 -> the +Y fallback). Copied VERBATIM into grain_collide.comp (int64 -> Vulkan-only).

// CollideGrainPlane: clamp every grain's surface to the ground plane (pos.y >= groundY + radius). The GR1
// radius-aware rest, factored out as the plane collider (static grains ARE clamped). Pure integer.
inline void CollideGrainPlane(std::vector<GrainParticle>& grains, fx groundY) {
    const size_t n = grains.size();
    for (size_t i = 0; i < n; ++i) {
        const fx restY = groundY + grains[i].radius;
        if (grains[i].pos.y < restY) grains[i].pos.y = restY;
    }
}

// CollideGrainSphere: project ONE grain out of ONE static sphere — its CENTRE snaps to sphereR + grainR (the
// surfaces touch). If static -> untouched. Else an int32 AABB reject (against sphereR + grainR) first; then
// if centre-distance < sphereR + grainR, snap along the outward normal (dist==0 -> the +Y fallback). Returns
// true iff projected (a contact). The CollideParticleSphere twin, made radius-aware.
inline bool CollideGrainSphere(GrainParticle& p, const GrainSphereCollider& s) {
    if (p.flags & kFlagStatic) return false;
    const fx surf = s.radius + p.radius;                 // the surfaces-touch distance
    const fx dx = p.pos.x - s.center.x;
    const fx dy = p.pos.y - s.center.y;
    const fx dz = p.pos.z - s.center.z;
    const fx ax = dx < 0 ? -dx : dx;
    const fx ay = dy < 0 ? -dy : dy;
    const fx az = dz < 0 ? -dz : dz;
    if (ax > surf || ay > surf || az > surf) return false;   // outside the AABB -> no overlap
    const FxVec3 d = FxVec3{dx, dy, dz};
    const fx dist = FxLength(d);
    if (dist >= surf) return false;                          // outside the (expanded) sphere -> untouched
    const FxVec3 nrm = FxNormalize(d);                       // dist==0 -> {0,kOne,0} fallback
    p.pos = FxAdd(s.center, FxScale(nrm, surf));             // snap the centre to sphereR + grainR
    return true;
}

// CollideGrainSpheres: project a grain array out of a STATIC sphere set (the CollideSpheres twin). Per grain
// (index order), per sphere (fixed order), CollideGrainSphere. Returns the contact count.
inline int CollideGrainSpheres(std::vector<GrainParticle>& grains,
                               const std::vector<GrainSphereCollider>& spheres) {
    int contacts = 0;
    const size_t n = grains.size();
    for (size_t i = 0; i < n; ++i)
        for (size_t s = 0; s < spheres.size(); ++s)
            if (CollideGrainSphere(grains[i], spheres[s])) ++contacts;
    return contacts;
}

// ----- StepGrainContact: one full JACOBI contact step (predict -> neighbors -> K iters -> velocity ->
// collide) — the make-or-break reference the GPU multi-pass driver memcmp's against -------------------
//   (1) IntegrateGrains predict (GR1 — vel += g*dt; prev = pos; pos += vel*dt; radius-aware ground rest).
//   (2) MakeGrainGrid + BuildGrainCellTable + BuildGrainNeighborList from the PREDICTED positions (GR2;
//       built ONCE per step — the neighbour set is fixed across the `iters` contact iterations; cell-size
//       = the GR2 hSearch, which is >= the contact diameter).
//   (3) `iters` JACOBI contact iterations, EACH: SolveGrainContact (ALL, Δp_i into a SEPARATE dp[] buffer)
//       -> apply pos_i += dp_i (ALL). Each sub-pass per-grain independent (reads iteration-start state).
//   (4) derive velocity from the NET position change: vel = (pos − prev) / dt (the FL4 PBF velocity update).
//   (5) CollideGrainPlane + CollideGrainSpheres (project out of the ground + the static FxBody spheres —
//       AFTER the solve + the velocity update). Returns the contact count (a coverage stat).
// Pure integer, fixed op order -> two-run bit-identical AND bit-exact GPU==CPU. `hSearch` is the GR2 search
// radius (caller asserts >= the max contact diameter).
inline int StepGrainContact(std::vector<GrainParticle>& grains,
                            const std::vector<GrainSphereCollider>& spheres, const FxVec3& gravity, fx dt,
                            fx groundY, fx hSearch, int iters) {
    const size_t n = grains.size();
    // (1) predict (GR1). prev = pos is snapshotted INSIDE IntegrateGrains (the predicted-position anchor).
    IntegrateGrains(grains, gravity, dt, groundY);
    // (2) neighbour list from the predicted positions (GR2; built once for this step).
    const GrainGrid grid = MakeGrainGrid(grains, hSearch);
    const GrainCellTable table = BuildGrainCellTable(grains, grid);
    const GrainNeighborList list = BuildGrainNeighborList(grains, grid, table, hSearch);
    // (3) `iters` JACOBI contact iterations.
    std::vector<FxVec3> dp;
    for (int it = 0; it < iters; ++it) {
        SolveGrainContact(grains, list, dp);                 // Δp_i for ALL (separate dp buffer)
        for (size_t i = 0; i < n; ++i) {                     // apply pos_i += Δp_i for ALL (Jacobi)
            if (grains[i].flags & kFlagStatic) continue;
            grains[i].pos = FxAdd(grains[i].pos, dp[i]);
        }
    }
    // (4) derive velocity from the net position change: vel = (pos − prev) / dt.
    if (dt != 0) {
        for (size_t i = 0; i < n; ++i) {
            if (grains[i].flags & kFlagStatic) continue;
            const FxVec3 dpos = FxSub(grains[i].pos, grains[i].prev);
            grains[i].vel = FxVec3{fxdiv(dpos.x, dt), fxdiv(dpos.y, dt), fxdiv(dpos.z, dt)};
        }
    }
    // (5) collision: clamp to the ground plane THEN project out of every static sphere (deterministic).
    CollideGrainPlane(grains, groundY);
    return CollideGrainSpheres(grains, spheres);
}

// ----- StepGrainContactSteps: run K full JACOBI contact steps (the showcase / GPU K-step driver) ---------
// K successive StepGrainContact steps. The GPU runs THIS exact K-step loop as a HOST-driven sequence of
// MULTI-THREAD dispatches (predict -> neighbours[GPU passes] -> {dp -> apply}×iters -> velocity -> collide
// per step, a ComputeToComputeBarrier between sub-passes). Returns the FINAL step's contact count.
inline int StepGrainContactSteps(std::vector<GrainParticle>& grains,
                                 const std::vector<GrainSphereCollider>& spheres, const FxVec3& gravity,
                                 fx dt, fx groundY, fx hSearch, int iters, int steps) {
    int contacts = 0;
    for (int s = 0; s < steps; ++s)
        contacts = StepGrainContact(grains, spheres, gravity, dt, groundY, hSearch, iters);
    return contacts;
}

// ----- GrainPenetration: a deterministic integer overlap metric (peak + summed pair penetration) ---------
// Over the GR2 neighbor list, sum/max the pair penetration pen = (r_i+r_j) − |p_i−p_j| > 0 (each unordered
// pair counted ONCE, i<j). Returns {peak, summed} in Q16.16 (int64 accumulator). DETERMINISTIC + bit-exact
// CPU<->GPU. The showcase's "overlap relieved" proof compares this BEFORE vs AFTER the solve (penAfter <
// penBefore — the FL4 honesty: relieved, NOT zero). Rebuilds the neighbor list internally from the grains.
struct GrainPenetration { int64_t peak = 0; int64_t summed = 0; };
inline GrainPenetration MeasureGrainPenetration(const std::vector<GrainParticle>& grains, fx hSearch) {
    GrainPenetration out;
    const GrainGrid grid = MakeGrainGrid(grains, hSearch);
    const GrainCellTable table = BuildGrainCellTable(grains, grid);
    const GrainNeighborList list = BuildGrainNeighborList(grains, grid, table, hSearch);
    const uint32_t n = (uint32_t)grains.size();
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t s = list.neighborStart[i]; s < list.neighborStart[i + 1u]; ++s) {
            const uint32_t j = list.neighbors[s];
            if (j <= i) continue;                            // count each unordered pair ONCE
            const FxVec3 d = FxSub(grains[(size_t)i].pos, grains[(size_t)j].pos);
            const fx pen = (grains[(size_t)i].radius + grains[(size_t)j].radius) - FxLength(d);
            if (pen > 0) { out.summed += (int64_t)pen; if ((int64_t)pen > out.peak) out.peak = (int64_t)pen; }
        }
    return out;
}

// ===== Slice GR4 — the TANGENTIAL COULOMB FRICTION (the angle-of-repose; the SIGNATURE of FLAGSHIP #10) ===
// The ONE physics the deterministic-sim trilogy (rigid->cloth->fluid) never modeled: dry tangential Coulomb
// friction / shear. AFTER the GR3 normal (non-penetration) push, project out the inverse-mass-weighted
// TANGENTIAL relative displacement of each overlapping pair, clamped to μ·pen (the Coulomb cone) — so a poured
// pile HOLDS A SLOPE (a self-supporting cone at its angle of repose, with NO container), visibly ≠ GR3's
// frictionless flat spread. The standard Unified-Particle / PBD friction, in Q16.16. Like the GR3 contact
// solve, friction is per-grain INDEPENDENT in the JACOBI formulation (each grain accumulates its OWN
// tangential Δp from the iteration-start positions into a SEPARATE dp[], then ALL apply) -> the GPU pass is
// [numthreads(64,1,1)] MULTI-THREAD, NO single-thread/TDR (the FL4/GR3 win). int64 throughout ->
// grain_friction.comp is VULKAN-SPIR-V-ONLY (NOT in hf_gen_msl); the Metal --grain-friction showcase runs THIS
// CPU StepGrainFriction (byte-identical by construction, the GR3/FL4 convention).
//
// THE FRICTION CONSTRAINT (the new math — standard Unified-Particle/PBD friction). For an overlapping pair
// (i, j) with n = unit(p_i − p_j), pen = (r_i+r_j) − |p_i−p_j| > 0, compute the TANGENTIAL part of the
// RELATIVE displacement this step (positions vs the step-start `prev`, snapshotted by IntegrateGrains):
//   Δx_rel = (p_i − prev_i) − (p_j − prev_j)              // relative displacement this step
//   Δx_t   = Δx_rel − (Δx_rel · n)·n                       // tangential component (subtract the normal part)
//   t      = |Δx_t| ; fmax = fxmul(μ, pen)                 // the Coulomb cone (pen the normal-pen proxy)
//   if t <= eps:        no friction
//   else if t <= fmax:  corr = Δx_t                        // STATIC: cancel ALL tangential slip
//   else:               corr = FxScale(Δx_t, fxdiv(fmax, t))   // KINETIC: clamp to the cone
//   Δp_i += −share·corr  (share = fxdiv(w_i, w_i+w_j))     // inverse-mass-weighted, grain i's half
// (grain j independently accumulates its symmetric half from the same pair — the Jacobi structure, no
// double-apply.) STATIC grains (flags bit0 / invMass 0) -> Δp 0. The `Δx_rel·n` dot is formed componentwise
// (fxmul + add — no FxDot in fpx.h). int64 throughout. The shader grain_friction.comp copies THIS VERBATIM.

// kGrainFrictionEps: the slip dead-band (a few Q16.16 LSBs). Below it the tangential displacement is treated
// as zero — fxdiv truncation noise, NOT real slip — so a settled grain accumulates no spurious friction Δp.
inline constexpr fx kGrainFrictionEps = 16;   // ~16 Q16.16 LSBs (the kGrainCollideEps twin)

// μ (the dry Coulomb friction coefficient) — a host-snapped Q16.16 constant. ~0.8 gives a recognizable
// self-supporting cone (an emergent repose slope clearly > GR3's flat spread, within a μ-implied band). The
// showcase uses THIS exact value so the CPU reference + the GPU + the golden share one constant.
inline constexpr fx kGrainMu = (fx)(0.8 * (double)kOne + 0.5);   // 0.8 in Q16.16 (52429)

// ----- SolveGrainFriction: the Jacobi per-grain TANGENTIAL Δp accumulate (the SolveGrainContact twin) ------
// Given the grain array + the GR2 neighbor list + μ, compute the tangential-friction Δp_i for ALL grains into
// a SEPARATE dp[] buffer (the Jacobi double-buffer — reads the iteration-start positions/prev, NOT the
// in-progress dp). For each grain i (if NOT static): over its GR2 neighbours j with pen = (r_i+r_j) −
// |p_i−p_j| > 0, compute the tangential relative displacement, clamp to fxmul(μ, pen) (static cancels all,
// kinetic scales by fxdiv(fmax, t)), accumulate Δp_i += −share·corr (share = fxdiv(w_i, w_i+w_j)). int64
// (FxLength/FxNormalize/fxmul/fxdiv). The shader grain_friction.comp copies THIS body VERBATIM. Deterministic
// (the fixed GR2 neighbour order, fixed op order). dpOut is sized to the grain count (one Δp per grain).
inline void SolveGrainFriction(const std::vector<GrainParticle>& grains, const GrainNeighborList& list, fx mu,
                               std::vector<FxVec3>& dpOut) {
    const uint32_t n = (uint32_t)grains.size();
    dpOut.assign((size_t)n, FxVec3{0, 0, 0});
    for (uint32_t i = 0; i < n; ++i) {
        if (grains[(size_t)i].flags & kFlagStatic) continue;   // static -> Δp = 0 (the pinned case)
        const fx wi = grains[(size_t)i].invMass;
        FxVec3 accum{0, 0, 0};
        const uint32_t s0 = list.neighborStart[i], s1 = list.neighborStart[i + 1u];
        for (uint32_t s = s0; s < s1; ++s) {
            const uint32_t j = list.neighbors[s];
            const fx wj = grains[(size_t)j].invMass;
            const fx wsum = wi + wj;
            if (wsum == 0) continue;                           // both static -> no friction
            // d = p_i − p_j ; pen = (r_i + r_j) − |d| ; only pen > 0 has a contact (the Coulomb cone needs it).
            const FxVec3 d = FxSub(grains[(size_t)i].pos, grains[(size_t)j].pos);
            const fx dist = FxLength(d);
            const fx pen = (grains[(size_t)i].radius + grains[(size_t)j].radius) - dist;
            if (pen <= 0) continue;                            // non-overlapping candidate -> no friction
            const FxVec3 nrm = FxNormalize(d);                 // n = unit(p_i − p_j) (d==0 -> +Y fallback)
            // Δx_rel = (p_i − prev_i) − (p_j − prev_j) — the relative displacement this step.
            const FxVec3 dxi = FxSub(grains[(size_t)i].pos, grains[(size_t)i].prev);
            const FxVec3 dxj = FxSub(grains[(size_t)j].pos, grains[(size_t)j].prev);
            const FxVec3 dxRel = FxSub(dxi, dxj);
            // Δx_t = Δx_rel − (Δx_rel · n)·n (subtract the normal component). Dot componentwise (no FxDot).
            const fx dotN = fxmul(dxRel.x, nrm.x) + fxmul(dxRel.y, nrm.y) + fxmul(dxRel.z, nrm.z);
            const FxVec3 dxT = FxSub(dxRel, FxScale(nrm, dotN));
            const fx t = FxLength(dxT);
            if (t <= kGrainFrictionEps) continue;              // no real slip (dead-band) -> no friction
            const fx fmax = fxmul(mu, pen);                    // the Coulomb cone radius
            FxVec3 corr;
            if (t <= fmax) corr = dxT;                         // STATIC: cancel ALL tangential slip
            else           corr = FxScale(dxT, fxdiv(fmax, t));// KINETIC: clamp the slip to the cone
            const fx share = fxdiv(wi, wsum);
            // Δp_i += −share·corr (grain i's inverse-mass-weighted half).
            accum.x -= fxmul(share, corr.x);
            accum.y -= fxmul(share, corr.y);
            accum.z -= fxmul(share, corr.z);
        }
        dpOut[(size_t)i] = accum;
    }
}

// ----- StepGrainFriction: one full JACOBI contact+friction step (the StepGrainContact driver + friction) ----
// The GR3 StepGrainContact driver with a SECOND Jacobi sub-pass (friction) per iteration:
//   (1) IntegrateGrains predict (GR1 — vel += g*dt; prev = pos; pos += vel*dt; radius-aware ground rest). The
//       prev snapshot is the friction's step-start anchor.
//   (2) MakeGrainGrid + BuildGrainCellTable + BuildGrainNeighborList from the PREDICTED positions (GR2; built
//       ONCE per step — fixed across the `iters` iterations).
//   (3) `iters` JACOBI iterations, EACH: SolveGrainContact->apply (the GR3 NORMAL push, UNCHANGED) THEN
//       SolveGrainFriction->apply (the new TANGENTIAL friction, reading the POST-normal positions). TWO Jacobi
//       sub-passes per iteration, each per-grain independent (reads sub-pass-start state) -> bit-exact +
//       multi-thread, NO TDR.
//   (4) derive velocity from the NET position change: vel = (pos − prev) / dt.
//   (5) CollideGrainPlane + CollideGrainSpheres (project out of the ground + the static spheres). Returns the
//       collider contact count. μ=0 -> SolveGrainFriction is a pure no-op -> byte-identical to StepGrainContact
//       (the frictionless control). Pure integer, fixed op order -> two-run bit-identical AND bit-exact
//       GPU==CPU. `hSearch` is the GR2 search radius (caller asserts >= the max contact diameter).
inline int StepGrainFriction(std::vector<GrainParticle>& grains,
                             const std::vector<GrainSphereCollider>& spheres, const FxVec3& gravity, fx dt,
                             fx groundY, fx hSearch, fx mu, int iters) {
    const size_t n = grains.size();
    // (1) predict (GR1). prev = pos snapshotted INSIDE IntegrateGrains (the friction's step-start anchor).
    IntegrateGrains(grains, gravity, dt, groundY);
    // (2) neighbour list from the predicted positions (GR2; built once for this step).
    const GrainGrid grid = MakeGrainGrid(grains, hSearch);
    const GrainCellTable table = BuildGrainCellTable(grains, grid);
    const GrainNeighborList list = BuildGrainNeighborList(grains, grid, table, hSearch);
    // (3) `iters` JACOBI iterations: SolveGrainContact->apply (normal) THEN SolveGrainFriction->apply (tangent).
    std::vector<FxVec3> dp;
    for (int it = 0; it < iters; ++it) {
        // (3a) the GR3 NORMAL push (UNCHANGED).
        SolveGrainContact(grains, list, dp);
        for (size_t i = 0; i < n; ++i) {
            if (grains[i].flags & kFlagStatic) continue;
            grains[i].pos = FxAdd(grains[i].pos, dp[i]);
        }
        // (3b) the GR4 TANGENTIAL friction (reads the POST-normal positions).
        SolveGrainFriction(grains, list, mu, dp);
        for (size_t i = 0; i < n; ++i) {
            if (grains[i].flags & kFlagStatic) continue;
            grains[i].pos = FxAdd(grains[i].pos, dp[i]);
        }
    }
    // (4) derive velocity from the net position change: vel = (pos − prev) / dt.
    if (dt != 0) {
        for (size_t i = 0; i < n; ++i) {
            if (grains[i].flags & kFlagStatic) continue;
            const FxVec3 dpos = FxSub(grains[i].pos, grains[i].prev);
            grains[i].vel = FxVec3{fxdiv(dpos.x, dt), fxdiv(dpos.y, dt), fxdiv(dpos.z, dt)};
        }
    }
    // (5) collision: clamp to the ground plane THEN project out of every static sphere (deterministic).
    CollideGrainPlane(grains, groundY);
    return CollideGrainSpheres(grains, spheres);
}

// ----- StepGrainFrictionSteps: run K full JACOBI contact+friction steps (the showcase / GPU K-step driver) --
// K successive StepGrainFriction steps. The GPU runs THIS exact K-step loop as a HOST-driven sequence of
// MULTI-THREAD dispatches (predict -> neighbours[GPU passes] -> {normal dp -> apply -> friction dp -> apply}
// ×iters -> velocity -> collide per step, a ComputeToComputeBarrier between sub-passes). Returns the final
// step's contact count.
inline int StepGrainFrictionSteps(std::vector<GrainParticle>& grains,
                                  const std::vector<GrainSphereCollider>& spheres, const FxVec3& gravity,
                                  fx dt, fx groundY, fx hSearch, fx mu, int iters, int steps) {
    int contacts = 0;
    for (int s = 0; s < steps; ++s)
        contacts = StepGrainFriction(grains, spheres, gravity, dt, groundY, hSearch, mu, iters);
    return contacts;
}

// ----- MeasureGrainRepose: the deterministic angle-of-repose SLOPE metric (the HONEST headline metric) ------
// A self-supporting heap's slope = height / baseRadius (a deterministic Q16.16 ratio). Over the settled grain
// pool: height = (max pos.y) − groundY; baseRadius = the max horizontal distance from the pile's CENTROID
// AXIS (the mean x,z of the grains) over all grains, in the x-z plane — the half-width of the footprint.
// slope = fxdiv(height, baseRadius) (baseRadius 0 -> slope 0, the degenerate single-column case). Pure
// integer (the centroid is an int64-accumulated mean; the horizontal distance is FxLength of the x-z offset);
// DETERMINISTIC + bit-exact CPU<->GPU. The HONEST framing (the FL4/FPX3 caveat): the repose angle is EMERGENT
// + iterative — this asserts slope > 0 by a clear margin (a real heap, NOT GR3's ~flat spread), deterministic
// + two-run byte-identical, within a μ-implied BAND, NOT an exact degree.
struct GrainRepose { fx height = 0; fx baseRadius = 0; fx slope = 0; };
inline GrainRepose MeasureGrainRepose(const std::vector<GrainParticle>& grains, fx groundY) {
    GrainRepose out;
    const size_t n = grains.size();
    if (n == 0) return out;
    // Centroid axis: the mean (x, z) of the grains (int64 accumulate -> /n, the deterministic centroid).
    int64_t sx = 0, sz = 0;
    fx maxY = grains[0].pos.y;
    for (size_t i = 0; i < n; ++i) {
        sx += (int64_t)grains[i].pos.x;
        sz += (int64_t)grains[i].pos.z;
        if (grains[i].pos.y > maxY) maxY = grains[i].pos.y;
    }
    const fx cx = (fx)(sx / (int64_t)n);
    const fx cz = (fx)(sz / (int64_t)n);
    // height = (max pos.y) − groundY ; baseRadius = max horizontal distance from the centroid axis (x-z).
    out.height = maxY - groundY;
    fx maxR = 0;
    for (size_t i = 0; i < n; ++i) {
        const FxVec3 horiz{grains[i].pos.x - cx, 0, grains[i].pos.z - cz};
        const fx r = FxLength(horiz);
        if (r > maxR) maxR = r;
    }
    out.baseRadius = maxR;
    out.slope = (maxR != 0) ? fxdiv(out.height, out.baseRadius) : 0;
    return out;
}

}  // namespace grain
}  // namespace hf::sim
