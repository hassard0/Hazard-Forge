#pragma once
// Slice FL1 — Deterministic GPU Fluid: Q16.16 fixed-point PARTICLE POOL INTEGRATOR + dam-break block
// (the BEACHHEAD of FLAGSHIP #9: DETERMINISTIC GPU FLUID via Position-Based Fluids — a PBF density-
// constraint fluid solver that is BIT-IDENTICAL CPU<->Vulkan<->Metal AND frame/run reproducible, unlike
// UE5's float/non-deterministic Niagara). FL1 is ONLY the integer particle POOL + gravity integrate +
// ground floor-clamp — the integrator beachhead. NO neighbors (FL2), NO density/lambda kernel (FL3),
// NO PBF density-constraint solve (FL4), NO lockstep (FL5), NO float render (FL6), NO float on the
// bit-exact path. Pure CPU, header-only, NO device, NO backend symbols, NO <cmath>. Namespace
// hf::sim::fluid. PBF is the DENSITY-CONSTRAINT TWIN of the shipped cloth PBD solver — the whole flagship
// reuses the proven engine/sim/fpx.h Q16.16 toolbox + the cloth CL1-CL6 mold; this completes the
// deterministic-sim trilogy (rigid FPX -> cloth -> fluid). The STRUCTURAL TWIN of the cloth CL1 integer
// beachhead (engine/sim/cloth.h): a pure-integer per-particle update proven GPU==CPU BIT-EXACT, with a
// cross-backend BIT-IDENTICAL integer golden — the SAME shape over a 3D particle BLOCK instead of a
// flat 2D sheet.
//
// The integrator shader shaders/fluid_integrate.comp.hlsl copies IntegrateFluid's per-particle math
// VERBATIM (the fxmul + integrate + prev-snap + floor-clamp), so tests/fluid_test.cpp + the GPU pass
// exercise the EXACT math — which is what makes the integrated particle array bit-identical GPU==CPU AND
// cross-backend.
//
// THE CROSS-BACKEND CRUX (the make-or-break for GPU==CPU, like cloth.h's host-snapped ClothParticle
// array): the GPU consumes host-snapped Q16.16 INTEGERS (the FluidParticle array) and does ZERO floating
// point — every step is `(int64)a*b >> kFrac` (an ARITHMETIC right shift on int64, deterministic +
// identical on every compiler/vendor) + integer add + integer compare. In FL1 each particle is
// INDEPENDENT (no inter-particle neighbours/density/constraints until FL2-FL4), so the GPU per-particle
// write is trivially order-independent / race-free with NO atomics, and two runs are byte-identical.
//
// THE int32-vs-int64 DECISION (the CL1/FPX1 lesson, documented): the integrate is `vel += gravity*dt;
// pos += vel*dt`, both componentwise fxmul — the SAME form as cloth_integrate.comp / fpx_integrate.comp,
// which needed int64 because the (int64)a*b product before the >>kFrac shift exceeds int32 for Q16.16
// gravity*dt (gravity ≈ -9.8*65536 = -642253; products of two Q16.16 world-scale values blow past 2^31).
// To stay bit-exact to this int64-intermediate reference WITHOUT any overflow fragility,
// shaders/fluid_integrate.comp.hlsl uses int64 (like cloth_integrate.comp) and is therefore VULKAN-SPIR-V-
// ONLY (glslc — the Metal HLSL->SPIR-V->MSL frontend — cannot parse int64_t in HLSL), NOT in the Metal
// hf_gen_msl list; the Metal --fluid-integrate showcase runs the CPU fluid::IntegrateFluid (the SAME
// bit-exact reference the Vulkan GPU==CPU memcmp compares against) -> byte-identical to the Vulkan GPU
// result BY CONSTRUCTION. Same established convention as cloth_integrate.comp / fpx_integrate.comp.
//
// REUSE MAP (file:line): the Q16.16 toolbox is engine/sim/fpx.h — fx (int32 Q16.16, fpx.h:46), fxmul
// (fpx.h:54, the int64-intermediate multiply), FxVec3 + FxAdd/FxSub/FxScale (fpx.h:59-72), kOne/kFrac
// (fpx.h:47-48). The per-particle integrate mirrors fpx.h::IntegrateBody (fpx.h:149, the semi-implicit
// Euler + ground floor-clamp) + cloth.h::IntegrateParticle (cloth.h:134, plus the prev=pos snap) — READ,
// NOT modified (fluid is the additive sibling; fpx.h #included read-only + stays byte-unchanged).

#include <cstdint>
#include <cmath>     // FL3 ONLY: host-side BuildKernelTable poly6/spiky float math, snapped to Q16.16 ONCE
                     // at build (the cloth.h restLen / mc.h kTriTable host-snap precedent). The per-step
                     // bit-exact sim path (FL1 integrate, FL2 neighbors, FL3 ComputeDensity/ComputeLambda)
                     // uses NO <cmath>/float — only the integer LUT BuildKernelTable produces.
#include <vector>

#include "sim/fpx.h"   // read-only: fx / fxmul / FxVec3 / FxAdd / FxSub / FxScale / kOne / kFrac

namespace hf::sim {
namespace fluid {

// Reuse the fpx Q16.16 scalar + vector toolbox verbatim (NO new fixed-point primitives).
using fpx::fx;
using fpx::FxVec3;
using fpx::fxmul;
using fpx::FxAdd;
using fpx::FxSub;
using fpx::FxScale;
inline constexpr int kFrac = fpx::kFrac;     // Q16.16 fractional bits (== fpx::kFrac, MUST match the shader)
inline constexpr fx  kOne  = fpx::kOne;      // 1.0 in Q16.16 (65536)

// ----- The fluid particle (the std430 GPU mirror; the ClothParticle packing discipline) -------------
// A single fluid particle. std430-packable as plain int32s (pos.xyz, prev.xyz, vel.xyz, invMass, flags)
// = 11 x 4-byte = 44 bytes, NO padding holes (memcmp-able; the GPU FluidParticle mirror). IDENTICAL
// layout to cloth.h::ClothParticle (the proven beachhead packing).
//   * pos  : current Q16.16 world position.
//   * prev : the PREVIOUS position (prev = pos BEFORE the position integrate each step) — FL3/FL4's PBF
//     density/constraint pass needs it (Verlet/predicted-position anchor); FL1 maintains it so the buffer
//     layout is final from the beachhead.
//   * vel  : Q16.16 velocity (world units / second).
//   * invMass : Q16.16 inverse mass (0 => infinite mass / STATIC, never integrates). Carried for FL4.
//   * flags : bit0 = STATIC (a future boundary/wall particle — invMass 0, never moves). Reserved in FL1.
struct FluidParticle {
    FxVec3   pos;             // Q16.16 current position
    FxVec3   prev;            // Q16.16 previous position (prev = pos before the position integrate)
    FxVec3   vel;             // Q16.16 velocity (world units / second)
    fx       invMass = 0;     // Q16.16 inverse mass (0 => static / infinite mass)
    uint32_t flags   = 0;     // bit0 = STATIC (reserved for FL4 boundary particles)
};

inline constexpr uint32_t kFlagStatic = 1u;   // bit0: the particle is a fixed boundary (never integrates)

// ----- The dam-break block config: a fixed W x H x D lattice of fluid particles in a corner ----------
// The classic dam-break initial condition: a solid W x H x D block of fluid particles at a spacing, with
// its corner at `origin` (above the ground). The float layout constants are host-snapped to Q16.16 once
// at build (NOT in the per-step integer sim). FL1 picks a modest count (10x10x10 = 1000) — well under the
// FL4 TDR budget so the SAME scene survives the later slices.
struct FluidBlock {
    int    W = 0, H = 0, D = 0;   // block dims (x columns, y rows, z layers)
    fx     spacing = 0;           // Q16.16 spacing between adjacent particles
    FxVec3 origin;                // Q16.16 world position of particle (0,0,0) — the block's corner
};

// Index of particle (ix, iy, iz) in the x-major / y-mid / z-minor particle array (the deterministic,
// fixed traversal order the shader's one-thread-per-particle dispatch + the CPU reference share).
inline int ParticleIndex(const FluidBlock& block, int ix, int iy, int iz) {
    return (iz * block.H + iy) * block.W + ix;
}

// ----- InitBlock: the deterministic W x H x D dam-break block (all dynamic, at rest) -----------------
// Builds a solid block of fluid particles: particle (ix,iy,iz) at origin + (ix*spacing, iy*spacing,
// iz*spacing). All particles start at rest (vel 0, prev == pos), DYNAMIC (invMass = kOne, flags 0) — FL1
// has no boundary/static particles yet (that is FL4). The block sits in a corner ABOVE the ground so it
// FALLS and piles at the ground (the dam-break fall). Pure integer (the spacing/origin are already
// host-snapped Q16.16; this only adds integer multiples of spacing). Returns the populated particle
// vector (size W*H*D), in the ParticleIndex traversal order.
inline std::vector<FluidParticle> InitBlock(const FluidBlock& block) {
    std::vector<FluidParticle> particles((size_t)(block.W * block.H * block.D));
    for (int iz = 0; iz < block.D; ++iz) {
        for (int iy = 0; iy < block.H; ++iy) {
            for (int ix = 0; ix < block.W; ++ix) {
                FluidParticle p;
                // Host-snapped layout: integer multiples of the (already Q16.16) spacing -> exact, no float.
                p.pos = FxVec3{block.origin.x + (fx)(ix * (int)block.spacing),
                               block.origin.y + (fx)(iy * (int)block.spacing),
                               block.origin.z + (fx)(iz * (int)block.spacing)};
                p.prev    = p.pos;
                p.vel     = FxVec3{0, 0, 0};
                p.invMass = kOne;   // unit mass dynamic (FL1 has no static/boundary particles)
                p.flags   = 0;
                particles[(size_t)ParticleIndex(block, ix, iy, iz)] = p;
            }
        }
    }
    return particles;
}

// ----- IntegrateFluidParticle: the deterministic per-particle semi-implicit-Euler step (SHADER math) --
// For a single particle, if NOT static (!(flags & kFlagStatic)):
//   vel += gravity * dt   (component-wise fxmul)        — the FPX1/CL1 velocity integrate
//   prev = pos                                          — snapshot before the position move (for FL3/FL4)
//   pos += vel * dt        (component-wise fxmul)        — the FPX1/CL1 position integrate
//   single non-penetration FLOOR clamp: if (pos.y < groundY) { pos.y = groundY; if (vel.y < 0) vel.y = 0; }
// Static particles are UNTOUCHED. Pure integer; fixed op order; no RNG, no clock. Each particle is
// INDEPENDENT of every other (no inter-particle coupling in FL1), so the order over particles does NOT
// matter -> two-run bit-identical AND the GPU per-thread write is race-free with NO atomics. The shader
// runs THIS exact per-particle body. Copied verbatim from cloth.h::IntegrateParticle (= fpx.h::IntegrateBody
// plus the prev=pos snap).
inline void IntegrateFluidParticle(FluidParticle& p, const FxVec3& gravity, fx groundY, fx dt) {
    if (p.flags & kFlagStatic) return;
    // (1) integrate velocity: vel += gravity * dt.
    p.vel.x += fxmul(gravity.x, dt);
    p.vel.y += fxmul(gravity.y, dt);
    p.vel.z += fxmul(gravity.z, dt);
    // (2) snapshot the previous position (the PBF predicted-position anchor FL3/FL4 reads).
    p.prev = p.pos;
    // (3) integrate position: pos += vel * dt.
    p.pos.x += fxmul(p.vel.x, dt);
    p.pos.y += fxmul(p.vel.y, dt);
    p.pos.z += fxmul(p.vel.z, dt);
    // (4) ground floor clamp (no restitution — FL1 is free-fall + ground only).
    if (p.pos.y < groundY) {
        p.pos.y = groundY;
        if (p.vel.y < 0) p.vel.y = 0;
    }
}

// ----- IntegrateFluid: one integrate STEP over the whole particle pool ------------------------------
// Apply IntegrateFluidParticle to every particle once. The make-or-break reference the GPU memcmp's
// against. Order-independent (particles are independent in FL1) -> bit-identical regardless of GPU
// scheduling.
inline void IntegrateFluid(std::vector<FluidParticle>& particles, const FxVec3& gravity, fx dt,
                           fx groundY) {
    const size_t n = particles.size();
    for (size_t i = 0; i < n; ++i)
        IntegrateFluidParticle(particles[i], gravity, groundY, dt);
}

// ----- IntegrateFluidSteps: run K integrate steps (the showcase / GPU K-step driver) ----------------
// K successive IntegrateFluid steps over the pool. The GPU fluid_integrate.comp runs THIS exact K-step
// loop per thread (one thread per particle, since particles are independent in FL1).
inline void IntegrateFluidSteps(std::vector<FluidParticle>& particles, const FxVec3& gravity, fx dt,
                                fx groundY, int steps) {
    for (int s = 0; s < steps; ++s)
        IntegrateFluid(particles, gravity, dt, groundY);
}

// CountAtGround(particles, groundY): the deterministic count of particles resting at the ground floor
// (pos.y == groundY) — a reporting/stat helper for the showcase coverage proof. Pure integer compare ->
// bit-exact CPU<->GPU.
inline int CountAtGround(const std::vector<FluidParticle>& particles, fx groundY) {
    int n = 0;
    for (const FluidParticle& p : particles)
        if (p.pos.y == groundY) ++n;
    return n;
}

// ===== Slice FL2 — the GRID-HASH NEIGHBOR SEARCH (the candidate list FL3's density gather iterates) ===
// Builds the per-particle NEIGHBOR LIST over the FL1 particle pool via a uniform spatial-hash grid, with
// the proven count->scan->emit compaction (the FPX2 fpx_pair_* / CL2 cloth_edge_* twin). PURE INT32 ->
// the shaders MSL-generate NATIVELY (a true GPU pass on both backends, unlike FL1's int64 integrate): the
// whole neighbor search is integer INDEX arithmetic. Cell ids are FloorDiv per axis (fpx::BroadphaseCell);
// the cell bucketing is count->scan->emit of particle indices; the candidate reject is a per-axis
// |p_i.axis - p_j.axis| < h compare (fx is int32 -> a PURE INT32 compare, NO squaring, NO int64). The
// exact radial r<h cull is DEFERRED to FL3 (the kernel W(r,h) is 0 for r>=h, so an over-inclusive box-
// candidate list is correct). NO float, NO sqrt, NO int64.
//
// REUSE: fpx::FloorDiv (the deterministic floor-division for negative coords, fpx.h:177), fpx::FxCell
// (fpx.h:183), fpx::CellId (fpx.h:196). The count->scan->emit shape is fpx.h::CountPairs/BuildPairs.

using fpx::FloorDiv;     // read-only: the deterministic floor-division (correct for negative coords)
using fpx::FxCell;       // read-only: the int3 cell coordinate
using fpx::CellId;       // read-only: the flat cell linearization
using fpx::fxdiv;        // read-only (FL3): the int64 Q16.16 truncating divide (C_i, λ_i)
using fpx::FxISqrt;      // read-only (FL3): the BUILD-TIME-ONLY integer sqrt (BuildKernelTable's |∇W| r)
using fpx::FxLength;     // read-only (FL3): the int64 Q16.16 vector length (the λ gradient direction)
using fpx::FxNormalize;  // read-only (FL3): the int64 Q16.16 unit vector (the λ gradient direction)

// ----- The bounded dense grid over a fixed AABB (the deterministic cell-linearization scheme) ---------
// CHOSEN SCHEME: a BOUNDED DENSE GRID. Cell-size = `h` (the PBF smoothing radius). A particle's cell coord
// is FloorDiv(p.axis, h) per axis (fpx::BroadphaseCell at cell-size h — monotone across 0 for negatives).
// The grid covers [cellMin, cellMin+gridDim) cells per axis; a cell's flat id is the fpx::CellId of
// (coord - cellMin) into gridDim. The caller sizes the grid to cover the whole particle AABB (every
// particle's cell lands in [0,gridDim)), so the linearization is total + collision-free + fully
// deterministic (NOT a hash mod table-size — a dense grid is simplest + has no hash collisions). The
// origin offset `cellMin` lets the grid sit at any world location (incl. negative coords) deterministically.
struct FluidGrid {
    fx     h = 0;          // Q16.16 cell size (== the PBF smoothing radius)
    FxCell cellMin;        // the integer cell coord of the grid's (0,0,0) corner (the AABB lower cell)
    FxCell gridDim;        // the grid extent in cells per axis (cellCount = x*y*z)
};

// CellOf(p, h): the integer grid cell a particle's position falls in, FloorDiv per axis (== the
// fpx::BroadphaseCell quantize at cell-size h). Pure int32, deterministic for negative coords.
inline FxCell CellOf(const FxVec3& p, fx h) {
    return FxCell{FloorDiv(p.x, h), FloorDiv(p.y, h), FloorDiv(p.z, h)};
}

// FlatCellId(cell, grid): the flat id of an absolute cell coord into the bounded dense grid (offset by
// cellMin into [0,gridDim), then fpx::CellId). The caller guarantees the cell is in range (the grid was
// sized to the particle AABB); returns the linear cell index in [0, gridDim.x*y*z).
inline uint32_t FlatCellId(const FxCell& cell, const FluidGrid& grid) {
    const FxCell local{cell.x - grid.cellMin.x, cell.y - grid.cellMin.y, cell.z - grid.cellMin.z};
    return CellId(local, grid.gridDim);
}

// CellCount(grid): the total number of cells in the dense grid (gridDim.x * y * z).
inline uint32_t CellCount(const FluidGrid& grid) {
    return (uint32_t)(grid.gridDim.x * grid.gridDim.y * grid.gridDim.z);
}

// MakeGrid(particles, h): build the bounded dense grid that tightly covers the particle pool at cell-size
// h. cellMin = the min cell coord over all particles; gridDim = (maxCell - minCell + 1) per axis. A pad of
// +1 cell on the high side is NOT needed (the 27-cell stencil clamps), but the min/max give exact bounds.
// Empty pool -> a 1x1x1 grid at origin (deterministic degenerate). Pure int32.
inline FluidGrid MakeGrid(const std::vector<FluidParticle>& particles, fx h) {
    FluidGrid grid;
    grid.h = h;
    if (particles.empty()) {
        grid.cellMin = FxCell{0, 0, 0};
        grid.gridDim = FxCell{1, 1, 1};
        return grid;
    }
    FxCell lo = CellOf(particles[0].pos, h);
    FxCell hi = lo;
    for (const FluidParticle& p : particles) {
        const FxCell c = CellOf(p.pos, h);
        if (c.x < lo.x) lo.x = c.x; if (c.x > hi.x) hi.x = c.x;
        if (c.y < lo.y) lo.y = c.y; if (c.y > hi.y) hi.y = c.y;
        if (c.z < lo.z) lo.z = c.z; if (c.z > hi.z) hi.z = c.z;
    }
    grid.cellMin = lo;
    grid.gridDim = FxCell{hi.x - lo.x + 1, hi.y - lo.y + 1, hi.z - lo.z + 1};
    return grid;
}

// ----- BuildCellTable: bucket particle indices into cells (the count->scan->emit on particles) ---------
// The CSR-style cell table: cellStart[c..] is the exclusive prefix-sum of per-cell counts (cellStart has
// cellCount+1 entries; cellStart[c]..cellStart[c+1] is cell c's slice), and cellParticles[] holds the
// particle indices grouped by cell, ASCENDING particle index within each cell (deterministic). This is the
// count->scan->emit: (1) count particles per cell; (2) exclusive prefix-sum -> cellStart; (3) scatter each
// particle index into its cell's slice (the emit, ascending-index order by construction since the particle
// loop is ascending). Pure int32 -> the GPU fluid_cell_{count,scan,emit} mirror this byte-for-byte.
struct FluidCellTable {
    std::vector<uint32_t> cellStart;       // cellCount+1 exclusive prefix-sum offsets (CSR row pointers)
    std::vector<uint32_t> cellParticles;   // particle indices grouped by cell (size == particle count)
};

inline FluidCellTable BuildCellTable(const std::vector<FluidParticle>& particles, const FluidGrid& grid) {
    const uint32_t n = (uint32_t)particles.size();
    const uint32_t cells = CellCount(grid);
    FluidCellTable table;
    // (1) COUNT: per-cell particle count.
    std::vector<uint32_t> counts((size_t)cells, 0u);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t c = FlatCellId(CellOf(particles[i].pos, grid.h), grid);
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
    // (3) EMIT: scatter each particle index into its cell's slice (ascending index by the ascending loop).
    table.cellParticles.assign((size_t)n, 0u);
    std::vector<uint32_t> cursor((size_t)cells, 0u);   // per-cell write cursor (local offset)
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t c = FlatCellId(CellOf(particles[i].pos, grid.h), grid);
        table.cellParticles[table.cellStart[c] + cursor[c]] = i;
        ++cursor[c];
    }
    return table;
}

// ----- The neighbor reject (the PURE INT32 per-axis |dx| < h candidate test) ---------------------------
// NeighborAccept(a, b, h): accept b as a candidate neighbor of a iff |a.axis - b.axis| < h on EVERY axis
// (a box, NOT a sphere — the over-inclusive candidate set FL3's kernel culls). PURE INT32: an integer
// subtract + abs + compare per axis, NO products, NO int64, NO sqrt. The shader copies THIS verbatim.
inline bool NeighborAccept(const FxVec3& a, const FxVec3& b, fx h) {
    fx dx = a.x - b.x; if (dx < 0) dx = -dx;
    fx dy = a.y - b.y; if (dy < 0) dy = -dy;
    fx dz = a.z - b.z; if (dz < 0) dz = -dz;
    return dx < h && dy < h && dz < h;
}

// ----- BuildNeighborList: per-particle candidate neighbors over the 27-cell stencil (count->scan->emit) -
// For each particle i, scan the 27 cells of its 3x3x3 stencil (the cell + its 26 neighbors); for each
// particle j != i in those cells, accept iff NeighborAccept(p_i, p_j, h). Emit the accepted j into
// neighbors[] at i's offset, in a FIXED order: ascending stencil-cell (dz,dy,dx -1..+1), then within a
// cell ascending j (cellParticles is already ascending-index per cell) -> fully deterministic. The
// variable-length per-particle lists are laid out by count->scan->emit (neighborStart = exclusive
// prefix-sum; neighbors[] grouped by i). Stencil cells outside the grid are skipped (clamped). Pure int32.
struct FluidNeighborList {
    std::vector<uint32_t> neighborStart;   // particleCount+1 exclusive prefix-sum offsets (CSR)
    std::vector<uint32_t> neighbors;       // candidate neighbor j indices grouped by i (in stencil order)
};

// CountNeighbors(particles, grid, table, h, perParticleOut): the count pass. perParticleOut[i] = #candidate
// neighbors of i (j!=i in the 27-cell stencil passing NeighborAccept); returns the total. The GPU
// fluid_neighbor_count mirrors THIS per-thread (one thread per particle i).
inline uint32_t CountNeighbors(const std::vector<FluidParticle>& particles, const FluidGrid& grid,
                               const FluidCellTable& table, fx h, std::vector<uint32_t>& perParticleOut) {
    const uint32_t n = (uint32_t)particles.size();
    perParticleOut.assign((size_t)n, 0u);
    uint32_t total = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const FxCell ci = CellOf(particles[i].pos, grid.h);
        uint32_t c = 0;
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const FxCell nc{ci.x + dx, ci.y + dy, ci.z + dz};
            // Skip stencil cells outside the bounded grid (clamp).
            if (nc.x < grid.cellMin.x || nc.x >= grid.cellMin.x + grid.gridDim.x) continue;
            if (nc.y < grid.cellMin.y || nc.y >= grid.cellMin.y + grid.gridDim.y) continue;
            if (nc.z < grid.cellMin.z || nc.z >= grid.cellMin.z + grid.gridDim.z) continue;
            const uint32_t cell = FlatCellId(nc, grid);
            for (uint32_t s = table.cellStart[cell]; s < table.cellStart[cell + 1u]; ++s) {
                const uint32_t j = table.cellParticles[s];
                if (j == i) continue;                                      // NO self-neighbor
                if (NeighborAccept(particles[i].pos, particles[j].pos, h)) ++c;
            }
        }
        perParticleOut[i] = c;
        total += c;
    }
    return total;
}

// BuildNeighborList(particles, grid, table, h): the full mesher (count->scan->emit). (1) CountNeighbors ->
// per-particle counts; (2) exclusive prefix-sum -> neighborStart; (3) emit each accepted j into i's
// disjoint slice in the FIXED stencil order. The list is grouped by i (ascending), then stencil-cell
// (dz,dy,dx ascending), then j (ascending within a cell) -> fully deterministic. The GPU does the SAME
// three passes (count/scan/emit) -> the GPU neighbors[]+neighborStart memcmp's against this byte-for-byte.
inline FluidNeighborList BuildNeighborList(const std::vector<FluidParticle>& particles,
                                           const FluidGrid& grid, const FluidCellTable& table, fx h) {
    const uint32_t n = (uint32_t)particles.size();
    FluidNeighborList list;
    std::vector<uint32_t> counts;
    const uint32_t total = CountNeighbors(particles, grid, table, h, counts);
    // (2) SCAN: exclusive prefix-sum -> neighborStart (particleCount+1 entries; the last == total).
    list.neighborStart.assign((size_t)n + 1u, 0u);
    uint32_t running = 0;
    for (uint32_t i = 0; i < n; ++i) {
        list.neighborStart[i] = running;
        running += counts[i];
    }
    list.neighborStart[n] = running;   // == total
    // (3) EMIT: each particle writes its candidates into its disjoint [neighborStart[i], ..) slice.
    list.neighbors.assign((size_t)total, 0u);
    for (uint32_t i = 0; i < n; ++i) {
        const FxCell ci = CellOf(particles[i].pos, grid.h);
        uint32_t local = 0;
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const FxCell nc{ci.x + dx, ci.y + dy, ci.z + dz};
            if (nc.x < grid.cellMin.x || nc.x >= grid.cellMin.x + grid.gridDim.x) continue;
            if (nc.y < grid.cellMin.y || nc.y >= grid.cellMin.y + grid.gridDim.y) continue;
            if (nc.z < grid.cellMin.z || nc.z >= grid.cellMin.z + grid.gridDim.z) continue;
            const uint32_t cell = FlatCellId(nc, grid);
            for (uint32_t s = table.cellStart[cell]; s < table.cellStart[cell + 1u]; ++s) {
                const uint32_t j = table.cellParticles[s];
                if (j == i) continue;
                if (NeighborAccept(particles[i].pos, particles[j].pos, h)) {
                    list.neighbors[list.neighborStart[i] + local] = j;
                    ++local;
                }
            }
        }
    }
    return list;
}

// ===== Slice FL3 — the PBF DENSITY + λ (host-snapped kernel LUT, the MAKE-OR-BREAK) ==================
// The only genuinely fluid-specific computation: per particle gather the SPH DENSITY ρ_i over the FL2
// neighbours via a HOST-SNAPPED Q16.16 kernel LUT, then the PBF scaling factor λ_i = −C_i/(Σ|∇C_i|²+ε)
// with the unilateral density constraint C_i = ρ_i/ρ0 − 1. Per-particle INDEPENDENT (a gather over the
// fixed FL2 neighbor list) -> multi-thread, NO atomics, NO single-thread/TDR. int64 (the squared
// distance r² + the λ divide) -> fluid_density.comp/fluid_lambda.comp are Vulkan-only + Metal runs the
// CPU reference (the FL1/CL3 convention). Strict zero-diff cross-backend (the integer bar).
//
// THE HOST-SNAPPED KERNEL LUT (the make-or-break move, the mc.h kTriTable / cloth.h restLen precedent):
// raw SPH would evaluate poly6 W(r,h)=(315/64πh⁹)(h²−r²)³ and spiky ∇W symbolically — divide-heavy +
// hard to keep monotone/non-negative in Q16.16. PBF + a TABULATED kernel dodges this: precompute W and
// |∇W| as HOST-snapped Q16.16 values over B bins of r² ∈ [0, h²), computed ONCE at build (host float ->
// host-snapped int, the ONLY FxISqrt/sqrt use, in BuildKernelTable). Shared C++ run on both backends ->
// identical. r ≥ h (bin ≥ B) -> W = 0 (the FL2 over-inclusive box-candidate list is culled here). The
// kernel is monotone-non-increasing + non-negative BY CONSTRUCTION (the table); there is NO in-shader
// divide in the density (the ONE divide is λ's fxdiv). HONEST: the BINNED LUT is a FIDELITY
// SIMPLIFICATION (a deterministic step-function approximation of the analytic continuous kernel — the
// FPX3/CL3 caveat shape); the claim is DETERMINISM + cross-platform bit-identity, NOT "more physically
// correct than Niagara".

// FluidKernel: the host-snapped Q16.16 kernel LUT. W[bin] = poly6 density-kernel value (>= 0,
// monotone-non-increasing in bin); gradW[bin] = the spiky-gradient MAGNITUDE |∇W| (>= 0). Both are
// indexed by BinOf(r²) in [0, B); a neighbour with bin >= B (r >= h) contributes ZERO. h/restDensity/
// epsilon are carried so ComputeDensity/ComputeLambda read them from the same struct.
struct FluidKernel {
    fx                h = 0;           // Q16.16 smoothing radius (== the FL2 cell-size / neighbour radius)
    fx                restDensity = 0; // Q16.16 ρ0 (the target rest density; C_i = ρ_i/ρ0 − 1)
    fx                epsilon = 0;     // Q16.16 CFM relaxation ε (the λ denominator regularizer)
    int               bins = 0;        // B (the number of r² bins over [0, h²))
    std::vector<fx>   W;               // [B] poly6 density-kernel, Q16.16, host-snapped, monotone-non-incr
    std::vector<fx>   gradW;           // [B] spiky |∇W| magnitude, Q16.16, host-snapped
};

// kKernelBins: the LUT resolution. B = 64 r² bins over [0, h²). A step-function approximation of the
// analytic kernel (the documented fidelity simplification); 64 bins is a deterministic, golden-stable
// choice (NOT tuned for physical accuracy — that is out of scope for the deterministic flagship).
inline constexpr int kKernelBins = 64;

// H2Of(h): the squared Q16.16 magnitude h*h as an int64 (Q32.32-scaled — the FxLength sum-of-squares
// convention). r² and h² are compared/binned in this SAME int64 squared-magnitude space, NO shift, so
// the binning is a pure int64 integer op identical CPU<->shader.
inline int64_t H2Of(fx h) { return (int64_t)h * (int64_t)h; }

// RadiusSq(a, b): the squared Q16.16 distance |a − b|² as an int64 (each axis (a.axis−b.axis)² is an
// int64 product — bounded positions STILL overflow int32 for dx², so int64 is REQUIRED; the FxLength
// sum-of-squares discipline). The make-or-break int64 op the density gather + λ run. Identical
// CPU<->shader (the shader copies THIS body VERBATIM).
inline int64_t RadiusSq(const FxVec3& a, const FxVec3& b) {
    const int64_t dx = (int64_t)a.x - (int64_t)b.x;
    const int64_t dy = (int64_t)a.y - (int64_t)b.y;
    const int64_t dz = (int64_t)a.z - (int64_t)b.z;
    return dx * dx + dy * dy + dz * dz;
}

// BinOf(r2, h2, bins): the LUT bin for a squared distance r2 (int64) given h2 (int64) over `bins` bins.
// r2 >= h2 -> `bins` (OUT OF RANGE: r >= h -> W = 0, the radial cull the FL2 box-candidate list deferred).
// else floor((r2 * bins) / h2) in [0, bins). Pure int64 integer arithmetic (a multiply + a truncating
// divide) -> identical CPU<->shader. h2 > 0 is guaranteed (h > 0).
inline int BinOf(int64_t r2, int64_t h2, int bins) {
    if (r2 >= h2) return bins;                 // r >= h -> out of range -> zero contribution
    return (int)((r2 * (int64_t)bins) / h2);   // floor; in [0, bins)
}

// BuildKernelTable(h, restDensity, bins, epsilon): the HOST-SNAPPED Q16.16 kernel LUT (the make-or-break,
// the ONLY place float / FxISqrt is used). Over `bins` bins of r² ∈ [0, h²), evaluate the analytic kernels
// at each bin's LOWER r² edge (r²_bin = bin * h² / bins; r = sqrt(r²_bin)):
//   * W[bin]   = poly6  (315/(64 π h⁹)) (h² − r²)³                (the density kernel; (h²−r²)³ is
//                monotone-non-increasing in r² so W[bin] is monotone-non-increasing in bin, BY CONSTRUCTION).
//   * gradW[bin] = spiky  (45/(π h⁶)) (h − r)²                   (the |∇W| MAGNITUDE the constraint
//                gradient uses; (h−r)² monotone-non-increasing in r).
// All in HOST DOUBLE, then ROUND-TO-NEAREST snapped to Q16.16 (the cloth.h restLen snap). Computed ONCE,
// shared C++ -> identical both backends. NO per-step float (the integer ComputeDensity/ComputeLambda just
// index this table). Document: B = bins (default kKernelBins = 64); h/ρ0/ε in Q16.16 world units.
inline FluidKernel BuildKernelTable(fx h, fx restDensity, int bins, fx epsilon) {
    FluidKernel k;
    k.h = h;
    k.restDensity = restDensity;
    k.epsilon = epsilon;
    k.bins = bins;
    k.W.assign((size_t)bins, 0);
    k.gradW.assign((size_t)bins, 0);
    if (bins <= 0 || h <= 0) return k;

    const double kPi = 3.14159265358979323846;
    const double hf  = (double)h / (double)kOne;           // h in world units (float, build-time only)
    const double h2f = hf * hf;                            // h² in world units
    // The analytic 3D normalizations (poly6 + spiky-gradient magnitude). Standard SPH constants.
    const double poly6Norm = 315.0 / (64.0 * kPi * std::pow(hf, 9.0));   // W(r,h) = poly6Norm (h²−r²)³
    const double spikyNorm = 45.0  / (kPi * std::pow(hf, 6.0));          // |∇W(r,h)| = spikyNorm (h−r)²
    auto snap = [](double v) -> fx {
        // round-to-nearest, ties away from zero (the cloth.h / fpx.h BuildPileWorld snap).
        const double s = v * (double)kOne;
        return (fx)(s + (s < 0 ? -0.5 : 0.5));
    };
    for (int b = 0; b < bins; ++b) {
        // The bin's LOWER r² edge in world units: r²_bin = (b / bins) * h² (the conservative, monotone
        // representative — bin 0 is r=0 -> the peak self-density W[0]).
        const double r2 = ((double)b / (double)bins) * h2f;
        const double r  = std::sqrt(r2);                  // build-time sqrt (host float; the ONLY sqrt)
        const double w  = poly6Norm * std::pow(h2f - r2, 3.0);     // >= 0 for r2 <= h2
        const double gw = spikyNorm * (hf - r) * (hf - r);         // |∇W| magnitude, >= 0
        k.W[(size_t)b]     = snap(w > 0.0 ? w : 0.0);
        k.gradW[(size_t)b] = snap(gw > 0.0 ? gw : 0.0);
    }
    return k;
}

// ----- ComputeDensity: per particle ρ_i = W[0] (self) + Σ_{j∈neighbors(i)} W[bin(r_ij²)] -------------
// The bit-exact core. For each particle i: start from the self-contribution W[0] (the r=0 peak), then for
// each FL2 neighbour j gather W[BinOf(|p_i−p_j|², h², bins)] (a neighbour with bin >= bins, i.e. r >= h,
// adds nothing — the deferred FL2 radial cull lands HERE). r_ij² is the int64 RadiusSq (bounded positions
// STILL overflow int32 for dx², so int64 is REQUIRED). Deterministic (the fixed FL2 neighbour order). The
// shader fluid_density.comp copies THIS body VERBATIM -> the host GPU==CPU memcmp catches any divergence.
inline void ComputeDensity(const std::vector<FluidParticle>& particles, const FluidNeighborList& list,
                           const FluidKernel& k, std::vector<fx>& densityOut) {
    const uint32_t n = (uint32_t)particles.size();
    const int64_t h2 = H2Of(k.h);
    const fx wSelf = (k.bins > 0) ? k.W[0] : 0;
    densityOut.assign((size_t)n, 0);
    for (uint32_t i = 0; i < n; ++i) {
        fx rho = wSelf;                                   // the self-density term (r = 0 -> W[0])
        const uint32_t s0 = list.neighborStart[i], s1 = list.neighborStart[i + 1u];
        for (uint32_t s = s0; s < s1; ++s) {
            const uint32_t j = list.neighbors[s];
            const int bin = BinOf(RadiusSq(particles[i].pos, particles[j].pos), h2, k.bins);
            if (bin < k.bins) rho += k.W[(size_t)bin];    // r >= h -> bin == k.bins -> no contribution
        }
        densityOut[(size_t)i] = rho;
    }
}

// ----- ComputeLambda: per particle λ_i = −C_i / (Σ_k |∇_k C_i|² + ε), unilateral C_i<0 -> λ_i=0 --------
// The PBF scaling factor. For each particle i:
//   C_i      = fxdiv(ρ_i, ρ0) − kOne                                    (the density constraint, Q16.16)
//   ∇_j C_i  = (gradW[bin] / ρ0) * dir(p_i − p_j)   for each neighbour j (the per-neighbour gradient)
//   ∇_i C_i  = −Σ_j ∇_j C_i  (the self gradient; |∇_i C_i|² included in the sum)
//   Σgrad²   = Σ_j |∇_j C_i|²  +  |∇_i C_i|²                            (the sum of squared gradients)
//   λ_i      = fxdiv(−C_i, Σgrad² + ε)
// The UNILATERAL clamp: C_i < 0 (under-dense surface — fluid doesn't PULL together) -> λ_i = 0 (a fixed
// integer compare). int64 throughout (fxdiv + the grad² fxmul accumulate + FxNormalize). The gradient
// magnitude per axis is fxmul(gradWScaled, dir.axis) where gradWScaled = fxdiv(gradW[bin], ρ0) and dir =
// FxNormalize(p_i − p_j) (the int64 FxLength/FxISqrt direction). The shader fluid_lambda.comp copies THIS
// body VERBATIM. Deterministic (fixed FL2 neighbour order). densityOut from ComputeDensity is the input.
inline void ComputeLambda(const std::vector<FluidParticle>& particles, const FluidNeighborList& list,
                          const FluidKernel& k, const std::vector<fx>& density, std::vector<fx>& lambdaOut) {
    const uint32_t n = (uint32_t)particles.size();
    const int64_t h2 = H2Of(k.h);
    lambdaOut.assign((size_t)n, 0);
    for (uint32_t i = 0; i < n; ++i) {
        // C_i = ρ_i / ρ0 − 1 (Q16.16). fxdiv handles ρ0 != 0 (BuildKernelTable's restDensity > 0).
        const fx Ci = fxdiv(density[(size_t)i], k.restDensity) - kOne;
        // Unilateral: an UNDER-dense particle (C_i < 0) gets λ = 0 (no pulling-together correction).
        if (Ci < 0) { lambdaOut[(size_t)i] = 0; continue; }

        FxVec3 gradSelf{0, 0, 0};   // ∇_i C_i = −Σ_j ∇_j C_i (accumulated; its |·|² joins the sum)
        fx sumGrad2 = 0;            // Σ_j |∇_j C_i|² (the neighbour terms; the self term added after)
        const uint32_t s0 = list.neighborStart[i], s1 = list.neighborStart[i + 1u];
        for (uint32_t s = s0; s < s1; ++s) {
            const uint32_t j = list.neighbors[s];
            const int bin = BinOf(RadiusSq(particles[i].pos, particles[j].pos), h2, k.bins);
            if (bin >= k.bins) continue;                      // r >= h -> zero gradient
            // gradWScaled = |∇W| / ρ0; dir = unit(p_i − p_j); ∇_j C_i = gradWScaled * dir (per axis fxmul).
            const fx gradWScaled = fxdiv(k.gradW[(size_t)bin], k.restDensity);
            const FxVec3 dir = FxNormalize(FxSub(particles[i].pos, particles[j].pos));
            const FxVec3 g{fxmul(gradWScaled, dir.x), fxmul(gradWScaled, dir.y), fxmul(gradWScaled, dir.z)};
            // |∇_j C_i|² = g·g (each fxmul is Q16.16; the sum is Q16.16).
            sumGrad2 += fxmul(g.x, g.x) + fxmul(g.y, g.y) + fxmul(g.z, g.z);
            // accumulate the self gradient ∇_i C_i = −Σ_j ∇_j C_i.
            gradSelf.x -= g.x; gradSelf.y -= g.y; gradSelf.z -= g.z;
        }
        // |∇_i C_i|² joins the sum (the standard PBF Σ_k includes k = i).
        sumGrad2 += fxmul(gradSelf.x, gradSelf.x) + fxmul(gradSelf.y, gradSelf.y) +
                    fxmul(gradSelf.z, gradSelf.z);
        // λ_i = −C_i / (Σgrad² + ε). The CFM ε regularizes the denominator (PBF's stability term).
        lambdaOut[(size_t)i] = fxdiv(-Ci, sumGrad2 + k.epsilon);
    }
}

// MeanDensity(density): the deterministic integer mean ρ̄ of the density array (a reporting/coherence
// stat — Σρ / N, pure integer). Empty -> 0. Bit-exact CPU<->GPU (the GPU read-back feeds the same sum).
inline fx MeanDensity(const std::vector<fx>& density) {
    if (density.empty()) return 0;
    int64_t sum = 0;
    for (fx d : density) sum += (int64_t)d;
    return (fx)(sum / (int64_t)density.size());
}

}  // namespace fluid
}  // namespace hf::sim
