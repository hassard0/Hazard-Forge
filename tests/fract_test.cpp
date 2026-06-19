// Slice FR1 — Deterministic Rigid-Body Fracture/Destruction Slice 1: the per-sample NEAREST-SEED CELL
// classification integer core (engine/sim/fract.h) that the GPU shaders/fract_classify.comp.hlsl copies
// VERBATIM + proves bit-identical. Pure CPU (header-only, no device, no backend symbols). Namespace
// hf::sim::fract.
//
// What this test PINS (the contracts the GPU fract_classify.comp + the GPU==CPU proof build on):
//   * NearestSeed/ClassifyFractCells: a 1-seed field -> every sample cellId 0.
//   * a 2-seed field -> the split is the integer bisector; equidistant samples go to the LOWER seed
//     index (STRICTLY-less tie-break).
//   * a sample exactly on a seed -> that seed's index (d2 == 0).
//   * partition completeness: every sample assigned exactly one cellId, all in [0, M); the per-cell
//     sample-count sum == sampleCount.
//   * SampleIndex/SampleCount round-trip (the (z*ny+y)*nx+x layout).
//   * determinism: two classifies of the SAME field+seeds -> byte-identical cellId arrays.
//
// Pure C++ (hf_core), ASan-eligible like the other sim-math tests.
#include "sim/fract.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace fract = hf::sim::fract;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();

    // ================= SampleIndex / SampleCount layout =================
    {
        fract::FractField f; f.nx = 4; f.ny = 3; f.nz = 2;
        check(fract::SampleCount(f) == 24, "SampleCount == nx*ny*nz");
        check(f.sampleCount() == 24, "field.sampleCount() == nx*ny*nz");
        // (z*ny + y)*nx + x
        check(fract::SampleIndex(f, 0, 0, 0) == 0, "SampleIndex origin == 0");
        check(fract::SampleIndex(f, 3, 0, 0) == 3, "SampleIndex (3,0,0) == 3");
        check(fract::SampleIndex(f, 0, 1, 0) == 4, "SampleIndex (0,1,0) == nx");
        check(fract::SampleIndex(f, 0, 0, 1) == 12, "SampleIndex (0,0,1) == nx*ny");
        check(fract::SampleIndex(f, 3, 2, 1) == 23, "SampleIndex max == sampleCount-1");
        // Round-trip: enumerate in (z,y,x) order and confirm the index is dense + monotone.
        int expect = 0; bool dense = true;
        for (int z = 0; z < f.nz; ++z)
            for (int y = 0; y < f.ny; ++y)
                for (int x = 0; x < f.nx; ++x)
                    if (fract::SampleIndex(f, x, y, z) != expect++) dense = false;
        check(dense, "SampleIndex dense + monotone over (z,y,x)");
    }

    // ================= NearestSeed primitive (pure int32 squared distance) =================
    {
        std::vector<fract::FractSeed> seeds = {{0, 0, 0}, {10, 0, 0}};
        // On a seed -> that seed.
        check(fract::NearestSeed(0, 0, 0, seeds.data(), 2) == 0u, "NearestSeed on seed0 -> 0");
        check(fract::NearestSeed(10, 0, 0, seeds.data(), 2) == 1u, "NearestSeed on seed1 -> 1");
        // Clearly closer to one or the other.
        check(fract::NearestSeed(2, 0, 0, seeds.data(), 2) == 0u, "NearestSeed near seed0 -> 0");
        check(fract::NearestSeed(8, 0, 0, seeds.data(), 2) == 1u, "NearestSeed near seed1 -> 1");
        // Exact bisector (x=5 equidistant: 25 == 25) -> lowest index wins (STRICTLY less tie-break).
        check(fract::NearestSeed(5, 0, 0, seeds.data(), 2) == 0u,
              "NearestSeed equidistant -> lower index (tie-break)");
        // 3D squared distance (no sqrt): a point closer in 3D.
        std::vector<fract::FractSeed> s3 = {{0, 0, 0}, {0, 0, 10}};
        check(fract::NearestSeed(0, 0, 4, s3.data(), 2) == 0u, "NearestSeed 3D closer to seed0");
        check(fract::NearestSeed(0, 0, 6, s3.data(), 2) == 1u, "NearestSeed 3D closer to seed1");
        check(fract::NearestSeed(0, 0, 5, s3.data(), 2) == 0u, "NearestSeed 3D bisector -> lower index");
    }

    // ================= 1-seed field -> all samples cellId 0 =================
    {
        fract::FractField f; f.nx = 6; f.ny = 5; f.nz = 4;
        std::vector<fract::FractSeed> seeds = {{3, 2, 1}};
        fract::FractCells cells;
        fract::ClassifyFractCells(f, seeds, cells);
        check((int)cells.cellId.size() == f.sampleCount(), "1-seed: cellId sized sampleCount");
        bool allZero = true;
        for (uint32_t id : cells.cellId) if (id != 0u) allZero = false;
        check(allZero, "1-seed: every sample cellId == 0");
        auto counts = fract::CellSampleCounts(cells, 1);
        check(counts.size() == 1 && (int)counts[0] == f.sampleCount(),
              "1-seed: the single cell owns ALL samples");
        check(fract::DistinctCellCount(counts) == 1, "1-seed: 1 distinct cell");
    }

    // ================= 2-seed field -> integer bisector + lower-index ties =================
    {
        // A 1D-ish split along x on an 11x1x1 lattice with seeds at x=0 and x=10. The bisector is x=5
        // (exactly equidistant, 25==25 -> seed 0). x<5 -> seed 0, x>5 -> seed 1, x==5 -> seed 0.
        fract::FractField f; f.nx = 11; f.ny = 1; f.nz = 1;
        std::vector<fract::FractSeed> seeds = {{0, 0, 0}, {10, 0, 0}};
        fract::FractCells cells;
        fract::ClassifyFractCells(f, seeds, cells);
        bool ok = true;
        for (int x = 0; x <= 10; ++x) {
            uint32_t want = (x <= 5) ? 0u : 1u;  // x==5 equidistant -> lower index 0
            if (cells.cellId[(size_t)fract::SampleIndex(f, x, 0, 0)] != want) ok = false;
        }
        check(ok, "2-seed: split at integer bisector, equidistant -> lower index");
        // Both seeds own >= 1 sample here.
        auto counts = fract::CellSampleCounts(cells, 2);
        check(counts[0] == 6u && counts[1] == 5u, "2-seed: 6 vs 5 split (bisector to lower index)");
        check(fract::DistinctCellCount(counts) == 2, "2-seed: 2 distinct cells");
    }

    // ================= sample exactly on a seed -> that seed (d2 == 0) =================
    {
        fract::FractField f; f.nx = 8; f.ny = 8; f.nz = 8;
        std::vector<fract::FractSeed> seeds = {{1, 1, 1}, {6, 6, 6}, {1, 6, 1}};
        fract::FractCells cells;
        fract::ClassifyFractCells(f, seeds, cells);
        for (int k = 0; k < (int)seeds.size(); ++k) {
            const auto& s = seeds[(size_t)k];
            uint32_t got = cells.cellId[(size_t)fract::SampleIndex(f, s.x, s.y, s.z)];
            check(got == (uint32_t)k, "sample on a seed -> that seed's index");
        }
    }

    // ================= partition completeness: all assigned, all in [0,M), sum == sampleCount ====
    {
        fract::FractField f; f.nx = 16; f.ny = 12; f.nz = 8;
        std::vector<fract::FractSeed> seeds = {
            {2, 2, 2}, {13, 2, 2}, {2, 9, 2}, {13, 9, 2}, {7, 5, 5},
            {2, 2, 5}, {13, 9, 5}, {8, 1, 1}, {1, 11, 7}, {14, 6, 6},
        };
        const int M = (int)seeds.size();
        fract::FractCells cells;
        fract::ClassifyFractCells(f, seeds, cells);
        check((int)cells.cellId.size() == f.sampleCount(), "partition: cellId sized sampleCount");
        bool inRange = true;
        for (uint32_t id : cells.cellId) if ((int)id >= M) inRange = false;
        check(inRange, "partition: every cellId in [0, M)");
        auto counts = fract::CellSampleCounts(cells, M);
        uint64_t sum = 0; for (uint32_t c : counts) sum += c;
        check(sum == (uint64_t)f.sampleCount(), "partition: per-cell counts sum == sampleCount");
        // With well-spread interior seeds every seed should own at least one sample (the showcase bar);
        // FR1 does NOT require this in general (a dominated seed may own 0), but pin it for this config.
        check(fract::DistinctCellCount(counts) == M, "partition: all 10 seeds own >=1 sample (spread)");
    }

    // ================= determinism: two classifies byte-identical =================
    {
        fract::FractField f; f.nx = 9; f.ny = 7; f.nz = 5;
        std::vector<fract::FractSeed> seeds = {{0, 0, 0}, {8, 6, 4}, {4, 3, 2}, {1, 5, 0}};
        fract::FractCells a, b;
        fract::ClassifyFractCells(f, seeds, a);
        fract::ClassifyFractCells(f, seeds, b);
        check(a.cellId.size() == b.cellId.size() &&
                  std::memcmp(a.cellId.data(), b.cellId.data(),
                              a.cellId.size() * sizeof(uint32_t)) == 0,
              "determinism: two classifies BYTE-IDENTICAL");
    }

    if (g_fail == 0) std::printf("fract_test: ALL PASS\n");
    else std::printf("fract_test: %d FAIL\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
