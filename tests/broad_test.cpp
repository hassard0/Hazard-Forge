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
#include <string>      // BP2: std::string for the per-scene test labels
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

// A dynamic body at a Q16.16 world position (for clustered/overlapping BP2 scenes with sub-unit offsets).
static fpx::FxBody MakeBodyFx(fx x, fx y, fx z, fx radius) {
    fpx::FxBody b;
    b.pos = fpx::FxVec3{x, y, z};
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

    // ======================= Slice BP2 — THE CANDIDATE-PAIR GENERATOR =======================
    // Pins the BP2 contracts the GPU broad_pair_* + the equivalence proof build on:
    //   * BuildBroadphasePairs emits the canonical i<j set; every emitted pair has i<j, each unordered pair
    //     appears EXACTLY once, the per-body offsets are an exclusive prefix-sum of the counts.
    //   * PairSetEquivalentToAllPairs is TRUE (grid-pairs == fpx::BuildPairs all-pairs, byte-identical after
    //     sort) for several scenes (sparse / clustered / all-overlapping) — THE CRUX.
    //   * determinism: two builds byte-identical.
    //   * degenerate single-body scene -> zero pairs.
    const fx kHalf = kOne / 2;   // 0.5 (a Q16.16-exact sub-unit offset)

    // Helper: build the grid + cell table for a body set at the BP2 cell-size and run the pair generator.
    auto buildPairs = [&](const std::vector<fpx::FxBody>& bodies, std::vector<uint32_t>& off,
                          std::vector<fpx::FxPair>& pairs) {
        broad::BodyGrid grid = broad::MakeBodyGrid(bodies, kCellSize);
        broad::BodyCellTable table = broad::BuildBodyCellTable(bodies, grid);
        broad::BuildBroadphasePairs(bodies, grid, table, off, pairs);
    };

    // ----- BroadphaseAccept == fpx::AabbOverlap over BodyAabb (the predicate identity) -----
    {
        fpx::FxBody a = MakeBodyFx(0, 0, 0, kRadius);
        fpx::FxBody b = MakeBodyFx(kHalf, 0, 0, kRadius);          // 0.5 apart, diam 1.0 -> overlap
        fpx::FxBody c = MakeBodyFx(kOne * 3, 0, 0, kRadius);       // 3 apart -> no overlap
        check(broad::BroadphaseAccept(a, b), "BroadphaseAccept: near pair overlaps");
        check(!broad::BroadphaseAccept(a, c), "BroadphaseAccept: distant pair does not overlap");
        check(broad::BroadphaseAccept(a, b) ==
              fpx::AabbOverlap(fpx::BodyAabb(a), fpx::BodyAabb(b)),
              "BroadphaseAccept == fpx::AabbOverlap(BodyAabb,BodyAabb)");
    }

    // ----- canonical i<j set + offsets + equivalence, over THREE scenes -----
    auto pinScene = [&](const std::vector<fpx::FxBody>& bodies, const char* tag) {
        std::vector<uint32_t> off;
        std::vector<fpx::FxPair> pairs;
        buildPairs(bodies, off, pairs);
        const uint32_t n = (uint32_t)bodies.size();
        // every emitted pair has i<j AND i,j in range.
        bool canonical = true;
        for (const fpx::FxPair& p : pairs)
            if (!(p.i < p.j && p.j < n)) canonical = false;
        check(canonical, (std::string(tag) + ": every emitted pair is canonical i<j in range").c_str());
        // each unordered pair appears EXACTLY once (sort + adjacent-dup check).
        std::vector<fpx::FxPair> sorted = pairs;
        broad::SortPairsCanonical(sorted);
        bool unique = true;
        for (size_t s = 1; s < sorted.size(); ++s)
            if (sorted[s].i == sorted[s - 1].i && sorted[s].j == sorted[s - 1].j) unique = false;
        check(unique, (std::string(tag) + ": each unordered pair appears once").c_str());
        // offsets are an exclusive prefix-sum: off.size()==n, monotone non-decreasing, sum == pair count
        // (each body i's slice [off[i], off[i+1]) — with off[n] implied == pairs.size()).
        bool offOk = (off.size() == (size_t)n);
        for (uint32_t i = 1; i < n; ++i) if (off[i] < off[i - 1]) offOk = false;
        if (n > 0) offOk = offOk && (off[0] == 0u);
        check(offOk, (std::string(tag) + ": perBodyOffset is a monotone exclusive prefix-sum").c_str());
        // THE EQUIVALENCE PROOF: grid-pairs == fpx::BuildPairs all-pairs (byte-identical after sort).
        check(broad::PairSetEquivalentToAllPairs(bodies, pairs),
              (std::string(tag) + ": PairSetEquivalentToAllPairs (grid-pairs == all-pairs)").c_str());
        // determinism: a second build is byte-identical.
        std::vector<uint32_t> off2;
        std::vector<fpx::FxPair> pairs2;
        buildPairs(bodies, off2, pairs2);
        bool det = (off2 == off) && (pairs2.size() == pairs.size()) &&
                   (pairs.empty() ||
                    std::memcmp(pairs2.data(), pairs.data(), pairs.size() * sizeof(fpx::FxPair)) == 0);
        check(det, (std::string(tag) + ": BuildBroadphasePairs is deterministic (two builds byte-equal)").c_str());
        return (uint32_t)pairs.size();
    };

    // Scene A — SPARSE: a 3x1x3 grid at spacing 2 (no two bodies overlap -> zero pairs).
    {
        std::vector<fpx::FxBody> bodies;
        for (int z = 0; z < 3; ++z)
            for (int x = 0; x < 3; ++x)
                bodies.push_back(MakeBody(x * 2, 0, z * 2, kRadius));
        uint32_t p = pinScene(bodies, "sparse");
        check(p == 0u, "sparse scene -> zero candidate pairs");
    }

    // Scene B — CLUSTERED: a sparse backbone + a tight 3-body cluster (a triangle of pairs) + a tight pair.
    {
        std::vector<fpx::FxBody> bodies;
        for (int z = 0; z < 3; ++z)
            for (int x = 0; x < 3; ++x)
                bodies.push_back(MakeBody(x * 2, 0, z * 2, kRadius));
        bodies.push_back(MakeBodyFx(kHalf, 0, kHalf, kRadius));   // near body 0 (0,0,0) -> overlaps
        bodies.push_back(MakeBodyFx(kHalf, 0, 0, kRadius));
        bodies.push_back(MakeBodyFx(0, 0, kHalf, kRadius));
        uint32_t p = pinScene(bodies, "clustered");
        check(p >= 3u, "clustered scene -> a non-trivial pair set (>=3 from the cluster triangle)");
    }

    // Scene C — ALL-OVERLAPPING: k bodies all within one cell, pairwise overlapping -> k*(k-1)/2 pairs
    // (the dense limit where grid-pairs == all-pairs must STILL hold exactly).
    {
        std::vector<fpx::FxBody> bodies;
        const int k = 5;
        for (int i = 0; i < k; ++i) bodies.push_back(MakeBodyFx((fx)(i * (int)(kOne / 8)), 0, 0, kRadius));
        uint32_t p = pinScene(bodies, "all-overlap");
        check(p == (uint32_t)(k * (k - 1) / 2), "all-overlapping scene -> k*(k-1)/2 pairs (the dense limit)");
    }

    // ----- a body straddling a cell boundary: the ±1 stencil still finds the overlap -----
    {
        // Two bodies at x=1.5 and x=2.5 (cells 0 and 1 at cellSize 2): 1.0 apart, diam 1.0 -> AABBs touch.
        std::vector<fpx::FxBody> bodies;
        bodies.push_back(MakeBodyFx(kOne + kHalf, 0, 0, kRadius));      // x=1.5, cell 0
        bodies.push_back(MakeBodyFx(kOne * 2 + kHalf, 0, 0, kRadius));  // x=2.5, cell 1
        std::vector<uint32_t> off;
        std::vector<fpx::FxPair> pairs;
        buildPairs(bodies, off, pairs);
        check(pairs.size() == 1u && pairs[0].i == 0u && pairs[0].j == 1u,
              "cross-cell-boundary overlap found by the ±1 stencil (canonical {0,1})");
        check(broad::PairSetEquivalentToAllPairs(bodies, pairs),
              "cross-cell-boundary: grid-pairs == all-pairs");
    }

    // ----- degenerate: a single body -> zero pairs -----
    {
        std::vector<fpx::FxBody> bodies;
        bodies.push_back(MakeBody(0, 0, 0, kRadius));
        std::vector<uint32_t> off;
        std::vector<fpx::FxPair> pairs;
        buildPairs(bodies, off, pairs);
        check(pairs.empty(), "single-body scene -> zero pairs");
        check(off.size() == 1u && off[0] == 0u, "single-body offsets == {0}");
        check(broad::PairSetEquivalentToAllPairs(bodies, pairs),
              "single-body: grid-pairs == all-pairs (both empty)");
    }

    // ----- empty: zero bodies -> zero pairs, equivalence holds -----
    {
        std::vector<fpx::FxBody> bodies;
        std::vector<uint32_t> off;
        std::vector<fpx::FxPair> pairs;
        buildPairs(bodies, off, pairs);
        check(pairs.empty() && off.empty(), "empty scene -> zero pairs, empty offsets");
        check(broad::PairSetEquivalentToAllPairs(bodies, pairs), "empty: grid-pairs == all-pairs");
    }

    // ----- MeasureBroadphasePairs is a pure summary (pairs match a direct build, canonical flag true) -----
    {
        std::vector<fpx::FxBody> bodies;
        bodies.push_back(MakeBodyFx(0, 0, 0, kRadius));
        bodies.push_back(MakeBodyFx(kHalf, 0, 0, kRadius));
        bodies.push_back(MakeBodyFx(0, 0, kHalf, kRadius));
        broad::BodyGrid grid = broad::MakeBodyGrid(bodies, kCellSize);
        broad::BodyCellTable table = broad::BuildBodyCellTable(bodies, grid);
        broad::BroadphasePairMeasure m = broad::MeasureBroadphasePairs(bodies, grid, table);
        std::vector<uint32_t> off; std::vector<fpx::FxPair> pairs;
        broad::BuildBroadphasePairs(bodies, grid, table, off, pairs);
        check(m.bodies == 3u && m.pairs == (uint32_t)pairs.size(), "MeasureBroadphasePairs.{bodies,pairs} match");
        check(m.allCanonical, "MeasureBroadphasePairs.allCanonical == true");
        broad::BroadphasePairMeasure m2 = broad::MeasureBroadphasePairs(bodies, grid, table);
        check(std::memcmp(&m, &m2, sizeof(m)) == 0, "MeasureBroadphasePairs is a pure function (two calls equal)");
    }

    // ======================= Slice BP3 — THE BROADPHASE-DRIVEN BOX WORLD STEP =======================
    // Pins the BP3 contracts the GPU broad_convex_step + the scale proof build on:
    //   * BuildBroadphasePairsWithStatics is equivalent to all-pairs over a scene WITH a large static floor
    //     (the static-vs-dynamic pass self-policed by BroadphaseStepPairsEquivalentToAllPairs).
    //   * StepConvexWorldBPN == convex::StepConvexWorldN BYTE-IDENTICAL on a moderate scene (the
    //     scale/bit-transparency proof, CPU-only — the broadphase changes ONLY performance, not a bit).
    //   * the broadphase step brings the pile to REST (MeasureStack); deterministic (two runs byte-equal).
    namespace convex = hf::sim::convex;
    {
        // The deterministic step config (== the --broad-convex-shot / convex-stack config).
        const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
        broad::BroadStepConfig bcfg;
        bcfg.cfg.gravity     = convex::FxVec3{0, kGravY, 0};
        bcfg.cfg.dt          = kOne / 60;
        bcfg.cfg.solveIters  = 12;
        bcfg.cfg.restitution = 0;
        bcfg.cfg.slop        = kOne / 64;
        bcfg.cfg.beta        = (fx)((int64_t)4 * kOne / 10);    // 0.4
        bcfg.cfg.linDamp     = (fx)((int64_t)98 * kOne / 100);  // 0.98
        bcfg.cfg.angDamp     = (fx)((int64_t)50 * kOne / 100);  // 0.5
        bcfg.cfg.posIters    = 4;
        bcfg.cellSize        = kOne * 2;   // >= 2*dynamic-radius (slabs fit; see below)

        auto makeBody = [&](fx x, fx y, fx z, bool dyn, fx radius) {
            fpx::FxBody b;
            b.pos = {x, y, z};
            b.orient = fpx::FxQuat{0, 0, 0, kOne};
            b.invMass = dyn ? kOne : 0;
            b.flags   = dyn ? fpx::kFlagDynamic : 0u;
            b.vel = {0, 0, 0};
            b.angVel = {0, 0, 0};
            b.radius = radius;
            return b;
        };
        // A moderate scene: a wide static FLOOR (a large box, radius spans many cells -> exercises the
        // static-vs-dynamic all-pairs pass) + a small grid of dynamic unit boxes dropped above it. The dynamic
        // boxes are small (half-extent 0.5, radius ~0.87) so cellSize=2 keeps the ±1 stencil exact for them.
        const convex::FxBox kFloorBox{convex::FxVec3{FromInt(6), kOne, FromInt(6)}};
        const convex::FxBox kUnit{convex::FxVec3{kOne / 2, kOne / 2, kOne / 2}};
        const fx kUnitRad = (fx)(0.87 * (double)kOne);   // > sqrt(3)/2 box diagonal, the broadphase AABB pad
        const fx kFloorRad = FromInt(6);                      // the floor's broadphase half-extent (spans cells)
        auto buildModerate = [&]() {
            convex::ConvexWorld w;
            w.bodies.push_back(makeBody(0, 0, 0, false, kFloorRad)); w.boxes.push_back(kFloorBox);  // 0 floor
            // a 3x3 grid of dynamic boxes at two stacked layers (18 boxes), spaced 1 unit, dropped low so they
            // settle quickly within the tick budget.
            for (int layer = 0; layer < 2; ++layer)
                for (int gz = -1; gz <= 1; ++gz)
                    for (int gx = -1; gx <= 1; ++gx)
                        w.bodies.push_back(makeBody(FromInt(gx), FromInt(1) + kOne / 2 + FromInt(layer), FromInt(gz), true, kUnitRad)),
                        w.boxes.push_back(kUnit);
            return w;
        };

        // ----- the static-aware broadphase is equivalent to all-pairs (the floor's pairs found) -----
        {
            convex::ConvexWorld w = buildModerate();
            check(broad::BroadphaseStepPairsEquivalentToAllPairs(w, bcfg.cellSize),
                  "BP3: static-aware broadphase == all-pairs (the large static floor's pairs are found)");
            // Re-check after a few ticks (positions moved -> the candidate set is re-derived each tick).
            broad::StepConvexWorldBPN(w, bcfg, 20);
            check(broad::BroadphaseStepPairsEquivalentToAllPairs(w, bcfg.cellSize),
                  "BP3: static-aware broadphase == all-pairs after settling (re-broadphased positions)");
        }

        // ----- THE SCALE PROOF (CPU-only, exhaustive): StepConvexWorldBPN == convex::StepConvexWorldN -----
        {
            const uint32_t kTicks = 120;
            convex::ConvexWorld bp = buildModerate();
            broad::StepConvexWorldBPN(bp, bcfg, kTicks);
            convex::ConvexWorld ap = buildModerate();
            convex::StepConvexWorldN(ap, bcfg.cfg, kTicks);   // the all-pairs reference, SAME scene
            bool byteEqual = (bp.bodies.size() == ap.bodies.size());
            for (size_t i = 0; i < bp.bodies.size() && byteEqual; ++i)
                if (std::memcmp(&bp.bodies[i], &ap.bodies[i], sizeof(fpx::FxBody)) != 0) byteEqual = false;
            check(byteEqual,
                  "BP3: StepConvexWorldBPN == convex::StepConvexWorldN BYTE-IDENTICAL (the scale/bit-transparency proof)");

            // ----- the broadphase step brings the pile to REST -----
            const convex::StackMeasure ms = convex::MeasureStack(bp);
            check(ms.maxSpeed < kOne, "BP3: the broadphase-stepped pile came to REST (maxSpeed below band)");
            check(ms.maxPenetration < kOne / 4, "BP3: the broadphase-stepped pile is HELD (maxPen within slop+band)");

            // ----- determinism: two BP runs byte-identical -----
            convex::ConvexWorld bp2 = buildModerate();
            broad::StepConvexWorldBPN(bp2, bcfg, kTicks);
            bool det = (bp.bodies.size() == bp2.bodies.size());
            for (size_t i = 0; i < bp.bodies.size() && det; ++i)
                if (std::memcmp(&bp.bodies[i], &bp2.bodies[i], sizeof(fpx::FxBody)) != 0) det = false;
            check(det, "BP3: StepConvexWorldBPN is deterministic (two runs byte-identical)");
        }

        // ----- a single dynamic box + the static floor: equivalence + rest (the minimal static case) -----
        {
            convex::ConvexWorld w;
            w.bodies.push_back(makeBody(0, 0, 0, false, kFloorRad)); w.boxes.push_back(kFloorBox);
            w.bodies.push_back(makeBody(0, FromInt(3), 0, true, kUnitRad)); w.boxes.push_back(kUnit);
            check(broad::BroadphaseStepPairsEquivalentToAllPairs(w, bcfg.cellSize),
                  "BP3: single dynamic + floor -> static-aware broadphase == all-pairs");
            convex::ConvexWorld bp = w;
            broad::StepConvexWorldBPN(bp, bcfg, 120);
            convex::ConvexWorld ap = w;
            convex::StepConvexWorldN(ap, bcfg.cfg, 120);
            check(std::memcmp(&bp.bodies[1], &ap.bodies[1], sizeof(fpx::FxBody)) == 0,
                  "BP3: single dynamic + floor -> StepConvexWorldBPN == all-pairs byte-identical");
        }
    }

    // ======================= Slice BP4 — THE BROADPHASE-DRIVEN HULL WORLD STEP =======================
    // Pins the BP4 contracts the GPU broad_hull_step + the scale proof build on:
    //   * BuildHullAabb bounds every world vertex of the body-placed hull (every vert inside the AABB).
    //   * the hull static-aware broadphase pair set is equivalent to the hull-AABB all-pairs set over a
    //     MIXED-hull scene WITH a static floor (self-policed by HullBroadphaseStepPairsEquivalentToAllPairs).
    //   * StepHullWorldBPN == gjk::StepHullWorldN BYTE-IDENTICAL on a moderate mixed-hull scene (the
    //     scale/bit-transparency proof, CPU-only); the pile rests; deterministic (two runs byte-equal).
    namespace gjk = hf::sim::gjk;
    {
        // The deterministic step config (== the --broad-hull-shot / gjk-settle config).
        const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
        broad::BroadStepConfig bcfg;
        bcfg.cfg.gravity     = convex::FxVec3{0, kGravY, 0};
        bcfg.cfg.dt          = kOne / 60;
        bcfg.cfg.solveIters  = 20;
        bcfg.cfg.restitution = 0;
        bcfg.cfg.slop        = kOne / 64;
        bcfg.cfg.beta        = (fx)((int64_t)4 * kOne / 10);    // 0.4
        bcfg.cfg.linDamp     = (fx)((int64_t)90 * kOne / 100);  // 0.90
        bcfg.cfg.angDamp     = (fx)((int64_t)5 * kOne / 100);   // 0.05
        bcfg.cfg.posIters    = 4;
        bcfg.cellSize        = kOne * 4;   // >= 2 * (max dynamic hull-AABB cube-proxy radius + the margin)

        auto fi = [&](int v) { return (fx)((int64_t)v * (int64_t)kOne); };
        auto fd = [&](double v) { return (fx)(v * (double)kOne); };
        auto makeBody = [&](fx x, fx y, fx z, bool dyn) {
            fpx::FxBody b;
            b.pos = {x, y, z};
            b.orient = fpx::FxQuat{0, 0, 0, kOne};
            b.invMass = dyn ? kOne : 0;
            b.flags   = dyn ? fpx::kFlagDynamic : 0u;
            b.vel = {0, 0, 0};
            b.angVel = {0, 0, 0};
            b.radius = 0;   // hulls drive the broadphase via BuildHullAabb (radius unused for the hull step)
            return b;
        };

        // ----- BuildHullAabb bounds every world vertex of a (rotated) hull -----
        {
            gjk::FxHull h = gjk::MakeWedge(kOne, kOne, kOne);
            fpx::FxBody b = makeBody(fi(2), fi(3), fd(-1.0), true);
            // a non-identity orientation (a small spin about Y) so the AABB exercises FxRotate.
            b.angVel = {0, fi(1), 0};
            fpx::IntegrateBodyFull(b, convex::FxVec3{0, 0, 0}, kOne / 30);
            const fpx::FxAabb a = broad::BuildHullAabb(h, b);
            bool allInside = true;
            for (uint32_t v = 0; v < h.count; ++v) {
                const fpx::FxVec3 wv = fpx::FxAdd(fpx::FxRotate(b.orient, h.verts[v]), b.pos);
                if (wv.x < a.lo.x || wv.x > a.hi.x || wv.y < a.lo.y || wv.y > a.hi.y ||
                    wv.z < a.lo.z || wv.z > a.hi.z) allInside = false;
            }
            check(allInside, "BP4: BuildHullAabb bounds every world vertex of the body-placed hull");
        }

        // A moderate MIXED-hull scene: a wide static FLOOR (box-hull half-extent 4) + a small grid of dynamic
        // mixed hulls (tetra/octa/wedge/box) dropped just above it on a 1.4 spacing so they settle.
        auto buildModerate = [&]() {
            gjk::HullWorld w;
            w.bodies.push_back(makeBody(0, 0, 0, false)); w.hulls.push_back(gjk::MakeBox(fi(4), kOne, fi(4)));  // 0 floor
            int idx = 0;
            for (int gz = -1; gz <= 1; ++gz)
                for (int gx = -1; gx <= 1; ++gx) {
                    const fx x = (fx)((int64_t)gx * 14 * kOne / 10);   // 1.4 spacing
                    const fx z = (fx)((int64_t)gz * 14 * kOne / 10);
                    w.bodies.push_back(makeBody(x, fd(1.8), z, true));
                    switch (idx & 3) {
                        case 0: w.hulls.push_back(gjk::MakeTetra(fd(0.7))); break;
                        case 1: w.hulls.push_back(gjk::MakeOcta(fd(0.7))); break;
                        case 2: w.hulls.push_back(gjk::MakeWedge(fd(0.7), fd(0.7), fd(0.7))); break;
                        default: w.hulls.push_back(gjk::MakeBox(fd(0.6), fd(0.6), fd(0.6))); break;
                    }
                    ++idx;
                }
            return w;
        };

        // ----- the hull static-aware broadphase is equivalent to all-pairs (the floor's pairs found) -----
        {
            gjk::HullWorld w = buildModerate();
            check(broad::HullBroadphaseStepPairsEquivalentToAllPairs(w, bcfg.cellSize),
                  "BP4: hull static-aware broadphase == all-pairs (the static floor's pairs are found)");
            broad::StepHullWorldBPN(w, bcfg, 20);
            check(broad::HullBroadphaseStepPairsEquivalentToAllPairs(w, bcfg.cellSize),
                  "BP4: hull static-aware broadphase == all-pairs after settling (re-broadphased positions)");
        }

        // ----- THE SCALE PROOF (CPU-only): StepHullWorldBPN == gjk::StepHullWorldN byte-for-byte -----
        {
            const uint32_t kTicks = 120;
            gjk::HullWorld bp = buildModerate();
            broad::StepHullWorldBPN(bp, bcfg, kTicks);
            gjk::HullWorld ap = buildModerate();
            gjk::StepHullWorldN(ap, bcfg.cfg, kTicks);   // the all-pairs reference, SAME scene
            bool byteEqual = (bp.bodies.size() == ap.bodies.size());
            for (size_t i = 0; i < bp.bodies.size() && byteEqual; ++i)
                if (std::memcmp(&bp.bodies[i], &ap.bodies[i], sizeof(fpx::FxBody)) != 0) byteEqual = false;
            check(byteEqual,
                  "BP4: StepHullWorldBPN == gjk::StepHullWorldN BYTE-IDENTICAL (the scale/bit-transparency proof)");

            // ----- the broadphase step brings the mixed-hull pile to REST (kOne*2: the documented single-point-
            // manifold rock band — the pile is COHERENT + HELD, a small residual rock is the GJ4 stability limit) -----
            const gjk::HullStackMeasure ms = gjk::MeasureHullStack(bp);
            check(ms.maxSpeed < kOne * 2, "BP4: the broadphase-stepped mixed-hull pile came to REST (maxSpeed below band)");
            check(ms.maxPenetration < kOne / 2, "BP4: the broadphase-stepped pile is HELD (maxPen within band)");

            // ----- determinism: two BP runs byte-identical -----
            gjk::HullWorld bp2 = buildModerate();
            broad::StepHullWorldBPN(bp2, bcfg, kTicks);
            bool det = (bp.bodies.size() == bp2.bodies.size());
            for (size_t i = 0; i < bp.bodies.size() && det; ++i)
                if (std::memcmp(&bp.bodies[i], &bp2.bodies[i], sizeof(fpx::FxBody)) != 0) det = false;
            check(det, "BP4: StepHullWorldBPN is deterministic (two runs byte-identical)");
        }

        // ----- a single dynamic hull + the static floor: equivalence + scale (the minimal static case) -----
        {
            gjk::HullWorld w;
            w.bodies.push_back(makeBody(0, 0, 0, false)); w.hulls.push_back(gjk::MakeBox(fi(4), kOne, fi(4)));
            w.bodies.push_back(makeBody(0, fd(3.0), 0, true)); w.hulls.push_back(gjk::MakeOcta(kOne));
            check(broad::HullBroadphaseStepPairsEquivalentToAllPairs(w, bcfg.cellSize),
                  "BP4: single dynamic hull + floor -> static-aware broadphase == all-pairs");
            gjk::HullWorld bp = w;
            broad::StepHullWorldBPN(bp, bcfg, 120);
            gjk::HullWorld ap = w;
            gjk::StepHullWorldN(ap, bcfg.cfg, 120);
            check(std::memcmp(&bp.bodies[1], &ap.bodies[1], sizeof(fpx::FxBody)) == 0,
                  "BP4: single dynamic hull + floor -> StepHullWorldBPN == all-pairs byte-identical");
        }
    }

    if (g_fail == 0) std::printf("broad_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
