// Slice BP1 — Deterministic Integer Broadphase: THE BODY GRID + CSR CELL TABLE core (engine/sim/broad.h) that
// the GPU shaders/broad_cell_{count,scan,emit}.comp.hlsl copy VERBATIM + prove bit-identical. Pure CPU
// (header-only, no device, no backend symbols). Namespace hf::sim::broad. The grain.h GrainGrid/GrainCellTable
// integer-beachhead twin, keyed on fpx::FxBody.
//
// What this test PINS (the contracts the GPU broad_cell_* + the GPU==CPU proof build on):
//   * BuildBodyCellTable is a TOTAL PARTITION: each body index appears EXACTLY once; the within-cell indices
//     are ASCENDING; cellStart is monotone non-decreasing; cellStart[cellCount] == N; Σ per-cell counts == N.
//   * MakeBodyGrid tightly bounds the bodies: cellMin == the min body cell, gridDim == (maxCell-minCell+1) per
//     axis; the empty pool -> a 1x1x1 grid at origin (the deterministic degenerate).
//   * FlatBodyCellId is in [0, cellCount) for every body (the grid is sized to the body AABB).
//   * MeasureBodyGrid is a pure function (occupiedCells / maxCellOccupancy match a direct recount; two calls
//     identical).
//   * determinism: two builds over the same bodies/grid -> byte-identical tables.
//
// Pure C++ (hf_core), ASan-eligible like the other sim/render-math tests.
#include "sim/broad.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace broad = hf::sim::broad;
namespace fpx = hf::sim::fpx;
using broad::fx;
using fpx::kOne;
using fpx::kFrac;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Q16.16 of an integer (exact).
static fx FromInt(int v) { return (fx)(v << kFrac); }

// A dynamic body at an integer world position (radius defaulted; BP1 buckets centres).
static fpx::FxBody MakeBody(int x, int y, int z, fx radius) {
    fpx::FxBody b;
    b.pos = fpx::FxVec3{FromInt(x), FromInt(y), FromInt(z)};
    b.radius = radius;
    b.invMass = kOne;
    b.flags = fpx::kFlagDynamic;
    return b;
}

int main() {
    HF_TEST_MAIN_INIT();

    const fx kCellSize = kOne * 2;   // 2.0 world-unit cells (>= 2*maxRadius for the BP2 bound)
    const fx kRadius = kOne / 2;     // 0.5 (diameter 1.0 <= cellSize)

    // ================= MakeBodyGrid: tight bound + the empty degenerate =================
    {
        // Bodies at world x in {0,1, 5, 9}, y/z 0 -> with cellSize 2.0 cell x in {0,0, 2, 4}.
        std::vector<fpx::FxBody> bodies;
        bodies.push_back(MakeBody(0, 0, 0, kRadius));
        bodies.push_back(MakeBody(1, 0, 0, kRadius));   // same cell as body 0 (cell x 0)
        bodies.push_back(MakeBody(5, 0, 0, kRadius));   // cell x 2
        bodies.push_back(MakeBody(9, 0, 0, kRadius));   // cell x 4
        broad::BodyGrid grid = broad::MakeBodyGrid(bodies, kCellSize);
        check(grid.cellMin.x == 0 && grid.cellMin.y == 0 && grid.cellMin.z == 0,
              "MakeBodyGrid cellMin == the min body cell (0,0,0)");
        check(grid.gridDim.x == 5 && grid.gridDim.y == 1 && grid.gridDim.z == 1,
              "MakeBodyGrid gridDim == (maxCell-minCell+1) per axis (5,1,1)");
        check(grid.cellSize == kCellSize, "MakeBodyGrid carries cellSize");
        check(broad::BodyCellCount(grid) == 5u, "BodyCellCount == gridDim.x*y*z");

        // Negative coords: FloorDiv is monotone across 0 (cell of x=-1 is -1, not 0).
        std::vector<fpx::FxBody> neg;
        neg.push_back(MakeBody(-1, 0, 0, kRadius));   // cell x = floor(-1/2) = -1
        neg.push_back(MakeBody(3, 0, 0, kRadius));    // cell x = 1
        broad::BodyGrid ng = broad::MakeBodyGrid(neg, kCellSize);
        check(ng.cellMin.x == -1, "MakeBodyGrid cellMin spans negative coords (FloorDiv monotone)");
        check(ng.gridDim.x == 3, "MakeBodyGrid gridDim covers [-1..1] = 3 cells");
    }
    {
        // Empty -> 1x1x1 grid at origin (the deterministic degenerate).
        std::vector<fpx::FxBody> empty;
        broad::BodyGrid g = broad::MakeBodyGrid(empty, kCellSize);
        check(g.cellMin.x == 0 && g.cellMin.y == 0 && g.cellMin.z == 0, "empty grid cellMin == origin");
        check(g.gridDim.x == 1 && g.gridDim.y == 1 && g.gridDim.z == 1, "empty grid is 1x1x1");
        check(broad::BodyCellCount(g) == 1u, "empty grid has 1 cell");
        broad::BodyCellTable t = broad::BuildBodyCellTable(empty, g);
        check(t.cellStart.size() == 2u && t.cellStart[0] == 0u && t.cellStart[1] == 0u,
              "empty table cellStart == {0,0}");
        check(t.cellBodies.empty(), "empty table has no bodies");
    }

    // ================= FlatBodyCellId in [0, cellCount) for every body =================
    {
        // A 3x3x3 lattice (spacing 2 == cellSize so one body per cell) plus a few clustered.
        std::vector<fpx::FxBody> bodies;
        for (int z = 0; z < 3; ++z)
            for (int y = 0; y < 3; ++y)
                for (int x = 0; x < 3; ++x)
                    bodies.push_back(MakeBody(x * 2, y * 2, z * 2, kRadius));
        // Cluster: three extra bodies near the origin cell.
        bodies.push_back(MakeBody(0, 0, 1, kRadius));
        bodies.push_back(MakeBody(1, 1, 0, kRadius));
        bodies.push_back(MakeBody(1, 0, 1, kRadius));
        broad::BodyGrid grid = broad::MakeBodyGrid(bodies, kCellSize);
        const uint32_t cells = broad::BodyCellCount(grid);
        bool allInRange = true;
        for (const fpx::FxBody& b : bodies) {
            uint32_t c = broad::FlatBodyCellId(broad::BodyCellOf(b.pos, grid.cellSize), grid);
            if (c >= cells) allInRange = false;
        }
        check(allInRange, "FlatBodyCellId in [0, cellCount) for every body");

        // ============ BuildBodyCellTable: a TOTAL PARTITION ============
        broad::BodyCellTable table = broad::BuildBodyCellTable(bodies, grid);
        const uint32_t n = (uint32_t)bodies.size();
        check(table.cellStart.size() == (size_t)cells + 1u, "cellStart has cellCount+1 entries");
        check(table.cellBodies.size() == (size_t)n, "cellBodies has N entries");
        check(table.cellStart[0] == 0u, "cellStart[0] == 0");
        check(table.cellStart[cells] == n, "cellStart[cellCount] == N (the sentinel total)");

        // cellStart monotone non-decreasing.
        bool monotone = true;
        for (uint32_t c = 0; c < cells; ++c)
            if (table.cellStart[c] > table.cellStart[c + 1u]) monotone = false;
        check(monotone, "cellStart is monotone non-decreasing");

        // Σ per-cell counts == N.
        uint32_t sumCounts = 0;
        for (uint32_t c = 0; c < cells; ++c) sumCounts += table.cellStart[c + 1u] - table.cellStart[c];
        check(sumCounts == n, "sum of per-cell counts == N");

        // Every body index appears EXACTLY once.
        std::vector<int> seen((size_t)n, 0);
        bool inRangeIdx = true;
        for (uint32_t idx : table.cellBodies) {
            if (idx >= n) { inRangeIdx = false; continue; }
            ++seen[idx];
        }
        check(inRangeIdx, "every cellBodies entry is a valid body index");
        bool eachOnce = true;
        for (uint32_t i = 0; i < n; ++i) if (seen[i] != 1) eachOnce = false;
        check(eachOnce, "every body index appears EXACTLY once in cellBodies");

        // Within each cell the indices are ASCENDING, and each body sits in its own cell's slice.
        bool ascending = true, inOwnCell = true;
        for (uint32_t c = 0; c < cells; ++c) {
            for (uint32_t s = table.cellStart[c]; s < table.cellStart[c + 1u]; ++s) {
                uint32_t idx = table.cellBodies[s];
                if (s + 1u < table.cellStart[c + 1u] && table.cellBodies[s] >= table.cellBodies[s + 1u])
                    ascending = false;
                uint32_t bodyCell = broad::FlatBodyCellId(broad::BodyCellOf(bodies[idx].pos, grid.cellSize), grid);
                if (bodyCell != c) inOwnCell = false;
            }
        }
        check(ascending, "within each cell the body indices are ASCENDING");
        check(inOwnCell, "each body sits in its own FlatBodyCellId slice");

        // ============ MeasureBodyGrid: a pure function ============
        broad::BodyGridMeasure m = broad::MeasureBodyGrid(bodies, grid, table);
        check(m.bodies == n, "MeasureBodyGrid.bodies == N");
        check(m.cells == cells, "MeasureBodyGrid.cells == cellCount");
        // Direct recount.
        uint32_t occ = 0, maxOcc = 0;
        for (uint32_t c = 0; c < cells; ++c) {
            uint32_t cnt = table.cellStart[c + 1u] - table.cellStart[c];
            if (cnt > 0u) ++occ;
            if (cnt > maxOcc) maxOcc = cnt;
        }
        check(m.occupiedCells == occ, "MeasureBodyGrid.occupiedCells matches a direct recount");
        check(m.maxCellOccupancy == maxOcc, "MeasureBodyGrid.maxCellOccupancy matches a direct recount");
        check(occ >= 1u && occ <= cells, "occupiedCells in [1, cellCount]");
        check(maxOcc >= 1u, "the origin cluster -> maxCellOccupancy >= 1");
        // Pure: a second call is identical.
        broad::BodyGridMeasure m2 = broad::MeasureBodyGrid(bodies, grid, table);
        check(std::memcmp(&m, &m2, sizeof(m)) == 0, "MeasureBodyGrid is a pure function (two calls equal)");

        // ============ determinism: two builds byte-identical ============
        broad::BodyCellTable table2 = broad::BuildBodyCellTable(bodies, grid);
        check(table2.cellStart == table.cellStart && table2.cellBodies == table.cellBodies,
              "BuildBodyCellTable is deterministic (two builds byte-identical)");
    }

    // ================= a clustered scene: all bodies in ONE cell -> ascending {0..k-1} =================
    {
        std::vector<fpx::FxBody> cluster;
        for (int i = 0; i < 6; ++i) cluster.push_back(MakeBody(0, 0, 0, kRadius));  // all cell (0,0,0)
        broad::BodyGrid grid = broad::MakeBodyGrid(cluster, kCellSize);
        check(broad::BodyCellCount(grid) == 1u, "single-cluster grid is 1 cell");
        broad::BodyCellTable t = broad::BuildBodyCellTable(cluster, grid);
        check(t.cellStart.size() == 2u && t.cellStart[0] == 0u && t.cellStart[1] == 6u,
              "single-cluster cellStart == {0,6}");
        bool seq = (t.cellBodies.size() == 6u);
        for (uint32_t i = 0; i < t.cellBodies.size(); ++i) if (t.cellBodies[i] != i) seq = false;
        check(seq, "single-cluster cellBodies == {0,1,2,3,4,5} (ascending body index)");
    }

    if (g_fail == 0) std::printf("broad_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
