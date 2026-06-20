#pragma once
// Slice BP1 — Deterministic Integer Broadphase: THE BODY GRID + CELL TABLE (the BEACHHEAD of FLAGSHIP #23:
// DETERMINISTIC INTEGER BROADPHASE, hf::sim::broad). Every rigid solver (convex/fric/persist/gjk) iterates
// ALL i<j body pairs — the O(n²) "all-pairs small scene" caveat documented across the suite. This flagship
// builds a deterministic integer spatial-hash broadphase that produces a bit-identical candidate-pair set so
// the rigid solvers scale. BP1 is the STRUCTURAL beachhead: a uniform BODY GRID + its CSR CELL TABLE (bodies
// bucketed into cells via count->scan->emit), bit-exact CPU<->Vulkan<->Metal.
//
// This is the engine/sim/grain.h GrainGrid/GrainCellTable machinery (GR2), CLONED byte-for-byte and keyed on
// fpx::FxBody instead of grain::GrainParticle. The count->scan->emit grid-hash is reproduced PER-BODY: the
// CELL COUNT pass is per-body race-free (an order-independent sum); the EMIT scatter is the SINGLE-THREAD
// ASCENDING-body pass (THE DET-CRUX, the grain.h:286-295 / FL2 lesson — a parallel atomic cursor would make
// the within-cell order GPU-scheduling-dependent -> non-deterministic). The cell math is PURE INT32 (FloorDiv
// per axis + integer compares + an ascending-index scatter; NO fxmul, NO int64, NO sqrt), so the shaders
// (broad_cell_{count,scan,emit}.comp.hlsl) MSL-GENERATE NATIVELY — a TRUE GPU pass on BOTH backends (the
// strongest cross-vendor proof tier, the grain_cell_*/boids_cell_* precedent).
//
// REUSE MAP (file:line — grounded, all read-only, all BYTE-FROZEN):
//   fpx.h: FloorDiv (:177), FxCell (:183), BroadphaseCell (:190), CellId (:196), FxBody (:116 — pos/radius
//   :124/flags :133). The cell quantization is reused VERBATIM. grain.h (read-only — the STRUCTURE to mirror
//   per-body, NOT its grain-typed fns): GrainCellOf/FlatGrainCellId/GrainCellCount/MakeGrainGrid/
//   GrainCellTable/BuildGrainCellTable (:245-328) — BP1 reproduces these shapes keyed on FxBody.
//
// SCOPE (BP1 only): the body grid + the CSR cell table + the BodyGridMeasure summary, the integer golden, the
// three proofs. OUT OF SCOPE (later slices): the 27-cell stencil + AABB pair cull + grid-pairs==all-pairs
// equivalence (BP2), the broadphase-driven world steps (BP3/BP4), lockstep (BP5), lit render (BP6). The
// large-body AABB-cell-span insert is a BP2 concern — BP1's bodies fit <=1 cell by the cellSize bound
// (cellSize >= 2*maxRadius), documented below. NOTE: a BOUNDED DENSE GRID (a sparse 2-cluster scene allocates
// the bounding-box cell volume — fine for the canonical pile scenes, the MakeGrainGrid precedent; a hashed
// grid is a future refinement).

#include <cstdint>
#include <vector>

#include "sim/gjk.h"   // read-only: transitively gives convex/fric/persist/fpx/grain. We use fpx:: only:
                       // FloorDiv / FxCell / CellId (the cell quantization) + FxBody (the bucketed primitive).
                       // gjk.h/convex.h/fric.h/persist.h/fpx.h/grain.h are BYTE-FROZEN — broad is the additive
                       // sibling.

namespace hf::sim {
namespace broad {

// Reuse the fpx Q16.16 scalar + the integer cell toolbox verbatim (NO new fixed-point primitives).
using fx = fpx::fx;

// ----- The bounded dense grid over the body AABB (the grain.h::GrainGrid scheme, keyed on bodies) ----------
// CHOSEN SCHEME: a BOUNDED DENSE GRID (== grain.h::GrainGrid). Cell-size = cellSize (a Q16.16 length; for the
// broadphase it is sized to the bodies' AABB extent — for BP1 it is an input, e.g. 2*maxRadius or a passed
// value). A body's cell coord is FloorDiv(pos.axis, cellSize) per axis (monotone across 0 for negatives). The
// grid covers [cellMin, cellMin+gridDim) cells; a cell's flat id is fpx::CellId of (coord - cellMin) into
// gridDim. MakeBodyGrid sizes the grid to the body AABB (every body's cell in [0,gridDim)), so the
// linearization is total + collision-free + deterministic. The origin offset cellMin lets the grid sit at any
// world location (incl. negative coords) deterministically.
//
// THE cellSize BOUND (the BP2 contract, documented now): the BP2 27-cell stencil assumes a body's AABB spans
// <=1 cell, so cellSize >= 2*maxRadius. BP1 does NOT enforce this (it only buckets body CENTRES), but the
// showcase/tests pick cellSize >= 2*maxRadius so BP2 inherits a valid grid. A body larger than one cell is a
// BP2 AABB-cell-span concern (deferred).
struct BodyGrid {
    fpx::FxCell cellMin;          // the integer cell coord of the grid's (0,0,0) corner (the AABB lower cell)
    fpx::FxCell gridDim{1, 1, 1}; // the grid extent in cells per axis (cellCount = x*y*z)
    fx          cellSize = 0;     // Q16.16 cell edge length (sized to the body AABB extent)
};

// BodyCellOf(pos, cellSize): the integer grid cell a body's position falls in, FloorDiv per axis. Pure int32.
// The GrainCellOf twin.
inline fpx::FxCell BodyCellOf(const fpx::FxVec3& pos, fx cellSize) {
    return fpx::FxCell{fpx::FloorDiv(pos.x, cellSize), fpx::FloorDiv(pos.y, cellSize),
                       fpx::FloorDiv(pos.z, cellSize)};
}

// FlatBodyCellId(cell, grid): the flat id of an absolute cell coord into the bounded dense grid (offset by
// cellMin into [0,gridDim), then fpx::CellId). The caller guarantees the cell is in range (the grid was sized
// to the body AABB); returns the linear cell index in [0, gridDim.x*y*z). The FlatGrainCellId twin.
inline uint32_t FlatBodyCellId(const fpx::FxCell& cell, const BodyGrid& grid) {
    const fpx::FxCell local{cell.x - grid.cellMin.x, cell.y - grid.cellMin.y, cell.z - grid.cellMin.z};
    return fpx::CellId(local, grid.gridDim);
}

// BodyCellCount(grid): the total number of cells in the dense grid (gridDim.x * y * z). The GrainCellCount twin.
inline uint32_t BodyCellCount(const BodyGrid& grid) {
    return (uint32_t)(grid.gridDim.x * grid.gridDim.y * grid.gridDim.z);
}

// MakeBodyGrid(bodies, cellSize): build the bounded dense grid that tightly covers the body set at cell-size
// cellSize. cellMin = the min cell coord over all body centres; gridDim = (maxCell - minCell + 1) per axis.
// Empty -> a 1x1x1 grid at origin (the deterministic degenerate). Pure int32 (== grain.h::MakeGrainGrid).
inline BodyGrid MakeBodyGrid(const std::vector<fpx::FxBody>& bodies, fx cellSize) {
    BodyGrid grid;
    grid.cellSize = cellSize;
    if (bodies.empty()) {
        grid.cellMin = fpx::FxCell{0, 0, 0};
        grid.gridDim = fpx::FxCell{1, 1, 1};
        return grid;
    }
    fpx::FxCell lo = BodyCellOf(bodies[0].pos, cellSize);
    fpx::FxCell hi = lo;
    for (const fpx::FxBody& b : bodies) {
        const fpx::FxCell c = BodyCellOf(b.pos, cellSize);
        if (c.x < lo.x) lo.x = c.x; if (c.x > hi.x) hi.x = c.x;
        if (c.y < lo.y) lo.y = c.y; if (c.y > hi.y) hi.y = c.y;
        if (c.z < lo.z) lo.z = c.z; if (c.z > hi.z) hi.z = c.z;
    }
    grid.cellMin = lo;
    grid.gridDim = fpx::FxCell{hi.x - lo.x + 1, hi.y - lo.y + 1, hi.z - lo.z + 1};
    return grid;
}

// ----- BuildBodyCellTable: bucket body indices into cells (the count->scan->emit on bodies) ----------------
// The CSR-style cell table: cellStart[c..] is the exclusive prefix-sum of per-cell counts (cellStart has
// cellCount+1 entries; cellStart[c]..cellStart[c+1] is cell c's slice; the last == body count), and
// cellBodies[] holds the body indices grouped by cell, ASCENDING body index within each cell (deterministic).
// count->scan->emit: (1) count bodies per cell; (2) exclusive prefix-sum -> cellStart; (3) scatter each body
// index into its cell's slice (the emit, ascending-index order by construction since the body loop is
// ascending). Pure int32 -> the GPU broad_cell_{count,scan,emit} mirror this byte-for-byte.
//
// (THE DET-CRUX, the grain.h:293 / FL2 lesson: the EMIT is the single-thread ascending-body scatter — a
// parallel atomic cursor would make the within-cell order GPU-scheduling-dependent -> non-deterministic. The
// cell COUNT pass is per-body-disjoint and race-free; only the cell-emit scatter is the ordered pass.)
struct BodyCellTable {
    std::vector<uint32_t> cellStart;    // cellCount+1 exclusive prefix-sum offsets (CSR row pointers)
    std::vector<uint32_t> cellBodies;   // body indices grouped by cell (size == body count)
};

inline BodyCellTable BuildBodyCellTable(const std::vector<fpx::FxBody>& bodies, const BodyGrid& grid) {
    const uint32_t n = (uint32_t)bodies.size();
    const uint32_t cells = BodyCellCount(grid);
    BodyCellTable table;
    // (1) COUNT: per-cell body count.
    std::vector<uint32_t> counts((size_t)cells, 0u);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t c = FlatBodyCellId(BodyCellOf(bodies[i].pos, grid.cellSize), grid);
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
    // (3) EMIT: scatter each body index into its cell's slice (ascending index by the ascending loop).
    table.cellBodies.assign((size_t)n, 0u);
    std::vector<uint32_t> cursor((size_t)cells, 0u);   // per-cell write cursor (local offset)
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t c = FlatBodyCellId(BodyCellOf(bodies[i].pos, grid.cellSize), grid);
        table.cellBodies[table.cellStart[c] + cursor[c]] = i;
        ++cursor[c];
    }
    return table;
}

// ----- BodyGridMeasure: a deterministic summary the showcase/test asserts (a pure function of the inputs) --
struct BodyGridMeasure {
    uint32_t bodies = 0;            // the body count
    uint32_t cells = 0;            // the total dense-grid cell count
    uint32_t occupiedCells = 0;    // the number of cells holding >=1 body
    uint32_t maxCellOccupancy = 0; // the peak per-cell body count
};

inline BodyGridMeasure MeasureBodyGrid(const std::vector<fpx::FxBody>& bodies, const BodyGrid& grid,
                                       const BodyCellTable& table) {
    BodyGridMeasure m;
    m.bodies = (uint32_t)bodies.size();
    m.cells = BodyCellCount(grid);
    for (uint32_t c = 0; c < m.cells; ++c) {
        const uint32_t cnt = table.cellStart[c + 1u] - table.cellStart[c];
        if (cnt > 0u) ++m.occupiedCells;
        if (cnt > m.maxCellOccupancy) m.maxCellOccupancy = cnt;
    }
    return m;
}

}  // namespace broad
}  // namespace hf::sim
