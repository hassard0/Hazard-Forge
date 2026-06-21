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

#include <algorithm>   // BP2: std::sort (the canonical pair-set sort for the equivalence proof)
#include <cstdint>
#include <cstring>     // BP2: std::memcmp (the byte-exact equivalence compare)
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

// ===== Slice BP2 — THE CANDIDATE-PAIR GENERATOR (the crux) ============================================
// The 27-cell-stencil candidate-pair generator over the BP1 body grid + CSR cell table: for each body i
// (ascending), scan the 3x3x3 = 27-cell stencil around i's cell (FIXED order, clamped to the bounded
// grid) and emit the canonical pair fpx::FxPair{i, j} for each body j in those cells with j > i AND
// fpx::AabbOverlap(BodyAabb(i), BodyAabb(j)). The result is the SAME candidate-pair set the O(n^2)
// fpx::BuildPairs all-pairs scan produces — PROVED by PairSetEquivalentToAllPairs (sort both lists by
// (i,j); byte-compare). Pure int32 (stencil iteration + the six-compare AABB predicate + an ascending
// scatter; NO fxmul, NO int64, NO sqrt) -> the broad_pair_{count,scan,emit}.comp shaders MSL-generate
// NATIVELY (a TRUE GPU pass on BOTH backends).
//
// REUSE (read-only, byte-frozen): fpx::FxAabb / fpx::BodyAabb / fpx::AabbOverlap / fpx::FxPair (the AABB
// predicate + the canonical pair shape) + fpx::BuildPairs (THE all-pairs reference). The 27-cell stencil
// STRUCTURE mirrors grain.h::BuildGrainNeighborList, but with the j>i de-dup + the AABB predicate (NOT
// the radius test) — a rigid pair list is canonical i<j ONCE (the fpx.h:283 discipline).
//
// THE DET-CRUX (the grain/BP1 lesson): the COUNT is per-body-disjoint (race-free); the EMIT is the
// single-thread ascending-body scatter (a parallel atomic cursor would make the within-list order
// GPU-schedule-dependent -> nondeterministic). Within body i the stencil cells are visited in a FIXED
// (dz,dy,dx) order, bodies within a cell ascending -> the per-i emit order is deterministic (NOT
// necessarily ascending-j across cells; the equivalence proof compares SETS via a canonical sort).
//
// THE cellSize BOUND: the ±1 stencil is EXACT only when cellSize >= the max body AABB diameter
// (2*maxRadius), so two overlapping AABBs are always within ±1 cell. The BP2 scene/tests honor this
// (uniform-radius dynamic bodies, cellSize >= 2*radius); a body larger than one cell is a BP3 concern.

// BroadphaseAccept(a, b): the candidate-pair predicate — fpx::AabbOverlap over the two body AABBs (the
// six-compare separating-axis test). Pure int32, copied into broad_pair_count/emit VERBATIM.
inline bool BroadphaseAccept(const fpx::FxBody& a, const fpx::FxBody& b) {
    return fpx::AabbOverlap(fpx::BodyAabb(a), fpx::BodyAabb(b));
}

// CountBroadphasePairs(bodies, grid, table, perBodyOut): per body i, count j>i in the 27-cell stencil
// with overlapping AABB (per-body-disjoint, race-free). perBodyOut[i] = that count; returns the total.
// The shader broad_pair_count.comp computes THIS per thread (one thread per body i). FIXED stencil order.
inline uint32_t CountBroadphasePairs(const std::vector<fpx::FxBody>& bodies, const BodyGrid& grid,
                                     const BodyCellTable& table, std::vector<uint32_t>& perBodyOut) {
    const uint32_t n = (uint32_t)bodies.size();
    perBodyOut.assign((size_t)n, 0u);
    uint32_t total = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const fpx::FxCell ci = BodyCellOf(bodies[i].pos, grid.cellSize);
        uint32_t c = 0;
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const fpx::FxCell nc{ci.x + dx, ci.y + dy, ci.z + dz};
            // Skip stencil cells outside the bounded grid (clamp).
            if (nc.x < grid.cellMin.x || nc.x >= grid.cellMin.x + grid.gridDim.x) continue;
            if (nc.y < grid.cellMin.y || nc.y >= grid.cellMin.y + grid.gridDim.y) continue;
            if (nc.z < grid.cellMin.z || nc.z >= grid.cellMin.z + grid.gridDim.z) continue;
            const uint32_t cell = FlatBodyCellId(nc, grid);
            for (uint32_t s = table.cellStart[cell]; s < table.cellStart[cell + 1u]; ++s) {
                const uint32_t j = table.cellBodies[s];
                if (j <= i) continue;                                  // canonical de-dup: emit (i,j) once
                if (BroadphaseAccept(bodies[i], bodies[j])) ++c;
            }
        }
        perBodyOut[i] = c;
        total += c;
    }
    return total;
}

// BuildBroadphasePairs(bodies, grid, table, perBodyOffset, pairsOut): the full count->scan->emit mesher.
// (1) CountBroadphasePairs -> per-body counts; (2) exclusive prefix-sum -> perBodyOffset (the serial
// scan); (3) single-thread ascending body i scatter each accepted (i,j>i) into i's disjoint slice in the
// FIXED stencil order. The pair list is grouped by i (ascending), then stencil-cell (dz,dy,dx ascending),
// then j (ascending within a cell) -> fully deterministic. The GPU broad_pair_{count,scan,emit} mirror
// this byte-for-byte.
inline void BuildBroadphasePairs(const std::vector<fpx::FxBody>& bodies, const BodyGrid& grid,
                                 const BodyCellTable& table, std::vector<uint32_t>& perBodyOffset,
                                 std::vector<fpx::FxPair>& pairsOut) {
    const uint32_t n = (uint32_t)bodies.size();
    std::vector<uint32_t> counts;
    const uint32_t total = CountBroadphasePairs(bodies, grid, table, counts);
    // (2) SCAN: exclusive prefix-sum -> perBodyOffset (each body's disjoint write base).
    perBodyOffset.assign((size_t)n, 0u);
    uint32_t running = 0;
    for (uint32_t i = 0; i < n; ++i) {
        perBodyOffset[i] = running;
        running += counts[i];
    }
    // (3) EMIT: single-thread ascending body i scatter into [perBodyOffset[i], ..) in the FIXED order.
    pairsOut.assign((size_t)total, fpx::FxPair{0u, 0u});
    for (uint32_t i = 0; i < n; ++i) {
        const fpx::FxCell ci = BodyCellOf(bodies[i].pos, grid.cellSize);
        uint32_t local = 0;
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const fpx::FxCell nc{ci.x + dx, ci.y + dy, ci.z + dz};
            if (nc.x < grid.cellMin.x || nc.x >= grid.cellMin.x + grid.gridDim.x) continue;
            if (nc.y < grid.cellMin.y || nc.y >= grid.cellMin.y + grid.gridDim.y) continue;
            if (nc.z < grid.cellMin.z || nc.z >= grid.cellMin.z + grid.gridDim.z) continue;
            const uint32_t cell = FlatBodyCellId(nc, grid);
            for (uint32_t s = table.cellStart[cell]; s < table.cellStart[cell + 1u]; ++s) {
                const uint32_t j = table.cellBodies[s];
                if (j <= i) continue;
                if (BroadphaseAccept(bodies[i], bodies[j])) {
                    pairsOut[perBodyOffset[i] + local] = fpx::FxPair{i, j};
                    ++local;
                }
            }
        }
    }
}

// SortPairsCanonical(pairs): sort a pair list by (i, j) ascending (a canonical total order). Used by the
// equivalence proof to compare the grid-pair SET against fpx::BuildPairs as SETS (sort-then-memcmp). Pure
// integer compares.
inline void SortPairsCanonical(std::vector<fpx::FxPair>& pairs) {
    std::sort(pairs.begin(), pairs.end(), [](const fpx::FxPair& a, const fpx::FxPair& b) {
        return a.i != b.i ? a.i < b.i : a.j < b.j;
    });
}

// PairSetEquivalentToAllPairs(bodies, gridPairs): THE EQUIVALENCE PROOF. Build the all-pairs reference via
// fpx::BuildPairs over the SAME bodies, sort BOTH lists by (i,j), and compare as SETS (same count,
// byte-identical after sort). Returns true iff the grid-emitted candidate set is BYTE-IDENTICAL to the
// O(n^2) reference — the broadphase is provably bit-transparent (no pair missed, none duplicated). EXACT
// (a byte memcmp of sorted lists), NOT within-band. The make-or-break falsifiable claim.
inline bool PairSetEquivalentToAllPairs(const std::vector<fpx::FxBody>& bodies,
                                        const std::vector<fpx::FxPair>& gridPairs) {
    fpx::FxWorld world;
    world.bodies = bodies;
    std::vector<uint32_t> refOffset;
    std::vector<fpx::FxPair> refPairs;
    fpx::BuildPairs(world, refOffset, refPairs);   // the all-pairs reference (already i-then-j ordered)
    if (refPairs.size() != gridPairs.size()) return false;
    std::vector<fpx::FxPair> a = gridPairs;        // copy: the grid list is stencil-ordered, sort it
    std::vector<fpx::FxPair> b = refPairs;
    SortPairsCanonical(a);
    SortPairsCanonical(b);
    if (a.empty()) return true;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(fpx::FxPair)) == 0;
}

// ----- BroadphasePairMeasure: a deterministic summary the showcase/test asserts (a pure function) -------
struct BroadphasePairMeasure {
    uint32_t bodies = 0;       // the body count
    uint32_t pairs = 0;        // the total candidate-pair count
    uint32_t maxPerBody = 0;   // the peak per-body emitted-pair count
    bool     allCanonical = false;  // every emitted pair has i < j AND each unordered pair appears once
};

inline BroadphasePairMeasure MeasureBroadphasePairs(const std::vector<fpx::FxBody>& bodies,
                                                    const BodyGrid& grid, const BodyCellTable& table) {
    BroadphasePairMeasure m;
    m.bodies = (uint32_t)bodies.size();
    std::vector<uint32_t> offset;
    std::vector<fpx::FxPair> pairs;
    BuildBroadphasePairs(bodies, grid, table, offset, pairs);
    m.pairs = (uint32_t)pairs.size();
    // peak per-body count (from the offsets + total).
    const uint32_t n = m.bodies;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t hi = (i + 1u < n) ? offset[i + 1u] : m.pairs;
        const uint32_t cnt = hi - offset[i];
        if (cnt > m.maxPerBody) m.maxPerBody = cnt;
    }
    // canonical: every pair i<j AND no unordered duplicate (sort + adjacent-equal check).
    bool canonical = true;
    for (const fpx::FxPair& p : pairs) if (p.i >= p.j) canonical = false;
    std::vector<fpx::FxPair> sorted = pairs;
    SortPairsCanonical(sorted);
    for (size_t s = 1; s < sorted.size(); ++s)
        if (sorted[s].i == sorted[s - 1].i && sorted[s].j == sorted[s - 1].j) canonical = false;
    m.allCanonical = canonical;
    return m;
}

// ===== Slice BP3 — THE BROADPHASE-DRIVEN BOX WORLD STEP (the scaling beat) =============================
// BP1 built the body grid, BP2 the candidate-pair generator (proven equivalent to all-pairs). BP3 puts it
// to WORK: a broadphase-driven box world step (StepConvexWorldBP) that reproduces convex::StepConvexWorld's
// 5-pass tick with the all-pairs O(n^2) impulse-solve + position de-penetration loops REPLACED by iteration
// over the BP2 candidate-pair list — so a LARGE box scene (256+ bodies) settles deterministically. The
// make-or-break is the SCALE PROOF: the broadphase-driven step settles a large scene BYTE-IDENTICAL to the
// all-pairs reference on the SAME scene (the broadphase changes ONLY performance, not a single bit — provable
// bit-transparency, extended from BP2's pair-set proof to the full dynamics).
//
// THE GAUSS-SEIDEL ORDER CRUX (the make-or-break). GS is ORDER-DEPENDENT (a pair sees earlier pairs'
// mutations). For StepConvexWorldBP to be byte-identical to the all-pairs StepConvexWorld, it MUST visit the
// candidate pairs in the SAME order the all-pairs loop visits the box-overlapping subset: (i,j) ASCENDING.
// BP2's BuildBroadphasePairs emits i-ascending but j-in-stencil-order, so BP3 SORTS the broadphase pair list
// by (i,j) (SortPairsCanonical) before the solve. Then: the all-pairs loop visits every i<j pair, skipping
// non-overlapping ones (a box overlap => AABB overlap => in the candidate set); StepConvexWorldBP visits
// exactly the AABB-candidate pairs in (i,j) order, processing the same box-overlapping subset in the same
// order -> byte-identical GS result. The AABB-non-overlapping pairs the all-pairs loop visits are no-ops
// (BoxSat finds no overlap), so omitting them changes nothing. This is WHY the scale proof holds.
//
// THE STATIC / LARGE-BODY HANDLING (deferred from BP2). A static floor is a LARGE box spanning many cells;
// the BP2 27-cell stencil over center-cells would MISS its pairs. The fix: BuildBroadphasePairsWithStatics =
// the BP2 grid stencil over the DYNAMIC bodies (statics excluded from the grid) UNION a dynamic-vs-static
// all-pairs pass (each dynamic body x each static body, fpx::AabbOverlap-tested — O(n*k), k = #statics, small;
// the floor is one body). The combined list is canonicalized i<j and the equivalence proof (carried into BP3)
// self-polices it: the combined broadphase pair set (sorted) == fpx::BuildPairs all-pairs (sorted) over ALL
// bodies — if the static handling misses a pair, the proof FAILS LOUDLY. This is the resolution of BP2's
// deferred large-body caveat.
//
// REUSE (read-only, byte-frozen): convex::StepConvexWorld (the 5-pass shell mirrored line-for-line with ONLY
// the (3)+(4) pair-loops swapped), convex::BoxSatStable / BuildManifold / SolveManifoldImpulse /
// FxBoxInvInertiaBody / WorldInvInertia / IsDynamic / ConvexWorld / ConvexStepConfig, fpx::IntegrateBodyFull /
// AabbOverlap / BodyAabb / FxPair. broad.h APPENDS only (BP1/BP2 byte-frozen).

// BroadStepConfig = convex::ConvexStepConfig (reused) + the grid cell size (>= 2*maxDynamicRadius so the BP2
// ±1 stencil is EXACT for the dynamic bodies). The static-vs-dynamic pass is all-pairs so statics need no
// cellSize bound. cellSize is a Q16.16 length.
struct BroadStepConfig {
    hf::sim::convex::ConvexStepConfig cfg;   // the convex 5-pass parameters (gravity/dt/iters/slop/beta/damp)
    fx                                cellSize = fpx::kOne * 2;   // the broadphase grid cell edge (Q16.16)
};

// BuildBroadphasePairsWithStatics(world, cellSize, pairsOut): the STATIC-AWARE broadphase pair list over a
// convex::ConvexWorld. (a) the BP2 grid stencil over the DYNAMIC bodies only (statics excluded from the grid,
// the body indices REMAPPED back to world indices); (b) a dynamic-vs-static all-pairs pass (each dynamic x
// each static, fpx::AabbOverlap-tested). The combined list is canonicalized i<j (lower world index first) and
// appended; the caller SORTS it by (i,j). NOT sorted here — the caller controls the canonical order. Statics
// are bodies with invMass==0 (the convex.h convention; matches IsDynamic's invMass!=0 gate). The dynamic-only
// grid keeps the bounded dense grid tight (a single far-flung static floor would NOT bloat the grid).
inline void BuildBroadphasePairsWithStatics(const hf::sim::convex::ConvexWorld& world, fx cellSize,
                                            std::vector<fpx::FxPair>& pairsOut) {
    namespace convex = hf::sim::convex;
    const uint32_t n = (uint32_t)world.bodies.size();
    pairsOut.clear();

    // Partition into dynamic vs static world indices (a static is invMass==0, the convex.h convention; a body
    // with the dynamic flag cleared but invMass!=0 is treated as a movable participant — but the canonical
    // scenes set invMass==0 <=> static). We use invMass==0 as the grid-exclusion + static-pass predicate so
    // that ALL non-static bodies (the grid's stencil set) match exactly the i<j pairs fpx::BuildPairs visits
    // that are NOT static-static. (The static-static pairs the all-pairs solve SKIPS; we never emit them.)
    std::vector<fpx::FxBody> dynBodies;       // the dynamic-only body set (for the BP2 grid)
    std::vector<uint32_t>    dynToWorld;      // dynBodies[d] -> world body index
    std::vector<uint32_t>    statics;         // the world indices of static bodies
    dynBodies.reserve(n);
    dynToWorld.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        if (world.bodies[i].invMass != 0) { dynBodies.push_back(world.bodies[i]); dynToWorld.push_back(i); }
        else                              { statics.push_back(i); }
    }

    // (a) the BP2 grid stencil over the DYNAMIC bodies; remap each emitted (di,dj) back to world indices,
    // canonicalized i<j (dynToWorld is ASCENDING so di<dj => world i<j, but canonicalize defensively).
    if (!dynBodies.empty()) {
        const BodyGrid grid = MakeBodyGrid(dynBodies, cellSize);
        const BodyCellTable table = BuildBodyCellTable(dynBodies, grid);
        std::vector<uint32_t> off;
        std::vector<fpx::FxPair> dynPairs;
        BuildBroadphasePairs(dynBodies, grid, table, off, dynPairs);
        for (const fpx::FxPair& p : dynPairs) {
            uint32_t wi = dynToWorld[p.i], wj = dynToWorld[p.j];
            if (wi > wj) { uint32_t t = wi; wi = wj; wj = t; }
            pairsOut.push_back(fpx::FxPair{wi, wj});
        }
    }

    // (b) the dynamic-vs-static all-pairs pass: each dynamic x each static, AABB-tested. O(n*k), k small.
    // Canonical i<j over world indices.
    for (uint32_t d = 0; d < (uint32_t)dynToWorld.size(); ++d) {
        const uint32_t wd = dynToWorld[d];
        const fpx::FxAabb ad = fpx::BodyAabb(world.bodies[wd]);
        for (uint32_t s = 0; s < (uint32_t)statics.size(); ++s) {
            const uint32_t ws = statics[s];
            if (!fpx::AabbOverlap(ad, fpx::BodyAabb(world.bodies[ws]))) continue;
            uint32_t wi = wd, wj = ws;
            if (wi > wj) { uint32_t t = wi; wi = wj; wj = t; }
            pairsOut.push_back(fpx::FxPair{wi, wj});
        }
    }
}

// BroadphaseStepPairsEquivalentToAllPairs(world, cellSize): the BP3 carry of BP2's equivalence proof onto the
// STATIC-AWARE broadphase. Build the combined static-aware pair set, sort by (i,j); build the all-pairs
// reference (fpx::BuildPairs over ALL bodies), DROP its static-static pairs (the solve skips them, the
// static-aware broadphase never emits them), sort; byte-compare as SETS. Returns true iff byte-identical — the
// static handling is provably complete (no pair missed, none duplicated). EXACT (a byte memcmp), the falsifiable
// self-policing claim the spec requires.
inline bool BroadphaseStepPairsEquivalentToAllPairs(const hf::sim::convex::ConvexWorld& world, fx cellSize) {
    std::vector<fpx::FxPair> bp;
    BuildBroadphasePairsWithStatics(world, cellSize, bp);
    SortPairsCanonical(bp);

    fpx::FxWorld fw;
    fw.bodies = world.bodies;
    std::vector<uint32_t> refOff;
    std::vector<fpx::FxPair> refAll;
    fpx::BuildPairs(fw, refOff, refAll);
    std::vector<fpx::FxPair> ref;
    ref.reserve(refAll.size());
    for (const fpx::FxPair& p : refAll) {
        if (world.bodies[p.i].invMass == 0 && world.bodies[p.j].invMass == 0) continue;  // static-static skip
        ref.push_back(p);
    }
    SortPairsCanonical(ref);

    if (bp.size() != ref.size()) return false;
    if (bp.empty()) return true;
    return std::memcmp(bp.data(), ref.data(), bp.size() * sizeof(fpx::FxPair)) == 0;
}

// StepConvexWorldBP(world, cfg): ONE broadphase-driven deterministic tick — convex::StepConvexWorld's 5-pass
// shell with the all-pairs (3) impulse-solve + (4) de-penetration loops REPLACED by iteration over the SORTED
// (i,j) broadphase pair list (re-built each tick from the CURRENT positions). Mirrors StepConvexWorld
// LINE-FOR-LINE; ONLY the pair source differs (the sorted candidate list vs the nested for i:for j>i). Pure
// integer, FIXED order. See convex.h:856-934 (the reference 5-pass body).
inline void StepConvexWorldBP(hf::sim::convex::ConvexWorld& world, const BroadStepConfig& bcfg) {
    namespace convex = hf::sim::convex;
    const hf::sim::convex::ConvexStepConfig& cfg = bcfg.cfg;
    const size_t n = world.bodies.size();

    // (1) Predict-integrate every dynamic body + per-tick velocity retention (VERBATIM StepConvexWorld:1).
    for (size_t i = 0; i < n; ++i) {
        if (convex::IsDynamic(world.bodies[i])) {
            fpx::IntegrateBodyFull(world.bodies[i], cfg.gravity, cfg.dt);
            if (cfg.linDamp != fpx::kOne) world.bodies[i].vel = convex::FxScale(world.bodies[i].vel, cfg.linDamp);
            if (cfg.angDamp != fpx::kOne) world.bodies[i].angVel = convex::FxScale(world.bodies[i].angVel, cfg.angDamp);
        }
    }

    // (2) The world inverse inertias, once per tick from the post-integrate orient (VERBATIM:2).
    std::vector<convex::FxMat3> invIW(n);
    for (size_t i = 0; i < n; ++i) {
        const convex::FxVec3 invIbody = convex::FxBoxInvInertiaBody(world.boxes[i], world.bodies[i].invMass);
        invIW[i] = convex::WorldInvInertia(world.bodies[i], invIbody);
    }

    // (2.5) RE-BUILD the broadphase pair list from the CURRENT positions + SORT it by (i,j) — the GS-order
    // crux (positions changed this tick, so the candidate set is re-derived; the fpx.h SimTick realism).
    std::vector<fpx::FxPair> pairs;
    BuildBroadphasePairsWithStatics(world, bcfg.cellSize, pairs);
    SortPairsCanonical(pairs);

    // (3) Impulse solve — solveIters world Gauss-Seidel sweeps over the SORTED pair list (VERBATIM the
    // StepConvexWorld:3 per-pair body, only the pair source swapped). Skip static-static (never emitted, but
    // keep the guard so the loop body is byte-identical to the all-pairs one). Mutation in place.
    for (uint32_t sweep = 0; sweep < cfg.solveIters; ++sweep) {
        for (const fpx::FxPair& p : pairs) {
            const size_t i = p.i, j = p.j;
            if (world.bodies[i].invMass == 0 && world.bodies[j].invMass == 0) continue;  // static-static
            const convex::SatResult sat = convex::BoxSatStable(world.bodies[i], world.boxes[i],
                                                               world.bodies[j], world.boxes[j]);
            if (!sat.overlap) continue;
            const convex::ContactManifold m = convex::BuildManifold(world.bodies[i], world.boxes[i],
                                                                    world.bodies[j], world.boxes[j], sat);
            if (m.count == 0) continue;
            convex::SolveManifoldImpulse(world.bodies[i], world.bodies[j], invIW[i], invIW[j], m,
                                         cfg.restitution, 1);   // ONE inner sweep — the outer loop is the GS
        }
    }

    // (4) Position de-penetration — posIters sweeps over the SORTED pair list (VERBATIM StepConvexWorld:4).
    for (uint32_t pit = 0; pit < cfg.posIters; ++pit) {
        for (const fpx::FxPair& p : pairs) {
            const size_t i = p.i, j = p.j;
            const fx invSum = world.bodies[i].invMass + world.bodies[j].invMass;
            if (invSum == 0) continue;   // both static -> skip
            const convex::SatResult sat = convex::BoxSatStable(world.bodies[i], world.boxes[i],
                                                               world.bodies[j], world.boxes[j]);
            if (!sat.overlap) continue;
            convex::FxVec3 nrm = sat.axis;
            if (convex::FxDot(nrm, fpx::FxSub(world.bodies[j].pos, world.bodies[i].pos)) < 0)
                nrm = convex::FxVec3{-nrm.x, -nrm.y, -nrm.z};
            const fx excess = sat.penetration - cfg.slop;
            if (excess <= 0) continue;
            const fx corrected = convex::fxmul(excess, cfg.beta);
            const fx wi = convex::fxdiv(world.bodies[i].invMass, invSum);
            const fx wj = fpx::kOne - wi;
            const convex::FxVec3 ci = convex::FxScale(nrm, convex::fxmul(corrected, wi));
            const convex::FxVec3 cj = convex::FxScale(nrm, convex::fxmul(corrected, wj));
            world.bodies[i].pos = fpx::FxSub(world.bodies[i].pos, ci);
            world.bodies[j].pos = convex::FxAdd(world.bodies[j].pos, cj);
        }
    }
    // (5) Orientation was already integrated in step (1).
}

// StepConvexWorldBPN(world, cfg, ticks): run `ticks` broadphase-driven steps -> a LARGE scene settles.
inline void StepConvexWorldBPN(hf::sim::convex::ConvexWorld& world, const BroadStepConfig& bcfg, uint32_t ticks) {
    for (uint32_t t = 0; t < ticks; ++t) StepConvexWorldBP(world, bcfg);
}

}  // namespace broad
}  // namespace hf::sim
