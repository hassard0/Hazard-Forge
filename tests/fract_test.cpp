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

    // ================= FR2: SampleCoord round-trips SampleIndex =================
    {
        fract::FractField f; f.nx = 7; f.ny = 5; f.nz = 3;
        bool roundTrip = true;
        for (int idx = 0; idx < f.sampleCount(); ++idx) {
            fract::FractCoord c = fract::SampleCoord(f, idx);
            if (fract::SampleIndex(f, c.x, c.y, c.z) != idx) roundTrip = false;
            if (c.x < 0 || c.x >= f.nx || c.y < 0 || c.y >= f.ny || c.z < 0 || c.z >= f.nz)
                roundTrip = false;
        }
        check(roundTrip, "SampleCoord round-trips SampleIndex over the whole lattice");
        fract::FractCoord c0 = fract::SampleCoord(f, 0);
        check(c0.x == 0 && c0.y == 0 && c0.z == 0, "SampleCoord(0) == origin");
    }

    // ================= FR2: ISqrt32 floor integer sqrt =================
    {
        check(fract::ISqrt32(0) == 0, "ISqrt32(0)==0");
        check(fract::ISqrt32(1) == 1, "ISqrt32(1)==1");
        check(fract::ISqrt32(3) == 1, "ISqrt32(3)==1 (floor)");
        check(fract::ISqrt32(4) == 2, "ISqrt32(4)==2");
        check(fract::ISqrt32(8) == 2, "ISqrt32(8)==2 (floor)");
        check(fract::ISqrt32(9) == 3, "ISqrt32(9)==3");
        check(fract::ISqrt32(3145728) == 1773, "ISqrt32 near the FR1 max squared distance");
    }

    // ================= FR2: 1-seed field -> 1 fragment, centroid the lattice center =================
    {
        fract::FractField f; f.nx = 5; f.ny = 5; f.nz = 5;
        std::vector<fract::FractSeed> seeds = {{2, 2, 2}};
        fract::FractCells cells; fract::ClassifyFractCells(f, seeds, cells);
        fract::FractFragments frags; fract::ExtractFragments(f, cells, 1, frags);
        check(frags.fragments.size() == 1, "1-seed: 1 fragment");
        const auto& fr = frags.fragments[0];
        // every sample -> cell 0; centroid = mean of [0,4]^3 = (2,2,2).
        check(fr.cx == 2 && fr.cy == 2 && fr.cz == 2, "1-seed: centroid == lattice center (2,2,2)");
        check(fr.minx == 0 && fr.miny == 0 && fr.minz == 0, "1-seed: AABB min == (0,0,0)");
        check(fr.maxx == 4 && fr.maxy == 4 && fr.maxz == 4, "1-seed: AABB max == (4,4,4)");
        check(fr.volume == (uint32_t)f.sampleCount(), "1-seed: volume == sampleCount (125)");
        check(fr.cellId == 0u, "1-seed: cellId == 0");
        check(frags.cellToFragment[0] == 0u, "1-seed: cellToFragment[0] == 0");
        check(frags.fragmentToCell[0] == 0u, "1-seed: fragmentToCell[0] == 0");
        // boundRadiusSq = max corner dist² = 2²+2²+2² = 12.
        check(fr.boundRadiusSq == 12, "1-seed: boundRadiusSq == 12 (corner)");
        check(fr.boundRadius == fract::ISqrt32(12), "1-seed: boundRadius == ISqrt32(12)");
        check(fr.invMass == fract::FractInvMass(fr.volume), "1-seed: invMass == kOne/volume");
    }

    // ================= FR2: 2-seed split -> 2 fragments, complementary volumes summing to N =====
    {
        fract::FractField f; f.nx = 10; f.ny = 1; f.nz = 1;
        std::vector<fract::FractSeed> seeds = {{0, 0, 0}, {9, 0, 0}};
        fract::FractCells cells; fract::ClassifyFractCells(f, seeds, cells);
        fract::FractFragments frags; fract::ExtractFragments(f, cells, 2, frags);
        check(frags.fragments.size() == 2, "2-seed: 2 fragments");
        check(frags.fragments[0].volume + frags.fragments[1].volume == (uint32_t)f.sampleCount(),
              "2-seed: complementary volumes sum to N");
        // x in [0,4] -> seed0 (5), x in [5,9] -> seed1 (5); bisector x=4.5 has no integer sample,
        // x=4 (d=4 vs 5) -> seed0, x=5 (d=5 vs 4) -> seed1.
        check(frags.fragments[0].volume == 5 && frags.fragments[1].volume == 5,
              "2-seed: even 5/5 split");
        check(frags.fragments[0].cellId == 0u && frags.fragments[1].cellId == 1u,
              "2-seed: fragments ordered by ascending cell index");
    }

    // ================= FR2: a dominated seed (0 samples) -> NO fragment + sentinel remap =========
    {
        // seed 1 sits AT seed 0 but with higher index -> the STRICTLY-less tie-break gives it 0 samples.
        fract::FractField f; f.nx = 4; f.ny = 4; f.nz = 4;
        std::vector<fract::FractSeed> seeds = {{1, 1, 1}, {1, 1, 1}, {3, 3, 3}};
        fract::FractCells cells; fract::ClassifyFractCells(f, seeds, cells);
        auto counts = fract::CellSampleCounts(cells, 3);
        check(counts[1] == 0u, "dominated: seed 1 owns 0 samples (verify the scene)");
        fract::FractFragments frags; fract::ExtractFragments(f, cells, 3, frags);
        check(frags.fragments.size() == 2, "dominated: 2 fragments (seed 1 dropped)");
        check(frags.cellToFragment[1] == fract::kNoFragment,
              "dominated: cellToFragment[1] == kNoFragment sentinel");
        check(frags.cellToFragment[0] == 0u && frags.cellToFragment[2] == 1u,
              "dominated: surviving cells compacted to ascending fragment indices 0,1");
        check(frags.fragments[0].cellId == 0u && frags.fragments[1].cellId == 2u,
              "dominated: fragmentToCell skips the empty cell");
        // CSR sentinel == sampleCount.
        check(frags.fragStart[3] == (uint32_t)f.sampleCount(), "dominated: CSR sentinel == sampleCount");
    }

    // ================= FR2: hand-computed centroid/AABB on a known small field =================
    {
        // A 3x1x1 field with seeds so cell 0 = {x=0,1}, cell 1 = {x=2}. Centroid of {0,1} = 0 (trunc),
        // of {2} = 2.
        fract::FractField f; f.nx = 3; f.ny = 1; f.nz = 1;
        std::vector<fract::FractSeed> seeds = {{0, 0, 0}, {2, 0, 0}};
        fract::FractCells cells; fract::ClassifyFractCells(f, seeds, cells);
        fract::FractFragments frags; fract::ExtractFragments(f, cells, 2, frags);
        check(frags.fragments.size() == 2, "hand: 2 fragments");
        // x=0->seed0, x=1->seed0 (d=1 vs 1, tie -> lower idx 0), x=2->seed1.
        check(frags.fragments[0].volume == 2 && frags.fragments[1].volume == 1, "hand: volumes 2,1");
        check(frags.fragments[0].cx == 0, "hand: centroid of {0,1} == 0 (truncating divide)");
        check(frags.fragments[0].minx == 0 && frags.fragments[0].maxx == 1, "hand: AABB x [0,1]");
        check(frags.fragments[1].cx == 2, "hand: centroid of {2} == 2");
        // boundRadiusSq of {0,1} about centroid 0: max(0, 1) = 1.
        check(frags.fragments[0].boundRadiusSq == 1, "hand: boundRadiusSq of {0,1} == 1");
        check(frags.fragments[1].boundRadiusSq == 0, "hand: singleton boundRadiusSq == 0");
    }

    // ================= FR2: Svol == N + every member inside its AABB+sphere (multi-seed) =========
    {
        fract::FractField f; f.nx = 16; f.ny = 12; f.nz = 8;
        std::vector<fract::FractSeed> seeds = {
            {2, 2, 1}, {13, 3, 2}, {3, 9, 5}, {12, 10, 6}, {8, 6, 3},
            {5, 1, 6}, {10, 7, 1}, {1, 5, 3},
        };
        const int M = (int)seeds.size();
        fract::FractCells cells; fract::ClassifyFractCells(f, seeds, cells);
        fract::FractFragments frags; fract::ExtractFragments(f, cells, M, frags);

        uint64_t sumVol = 0;
        for (const auto& fr : frags.fragments) sumVol += fr.volume;
        check(sumVol == (uint64_t)f.sampleCount(), "multi: Svol == sampleCount (mass partition)");

        bool allInside = true; int membersChecked = 0;
        for (size_t fi = 0; fi < frags.fragments.size(); ++fi) {
            const auto& fr = frags.fragments[fi];
            const uint32_t cell = frags.fragmentToCell[fi];
            for (uint32_t k = frags.fragStart[cell]; k < frags.fragStart[cell + 1u]; ++k) {
                fract::FractCoord p = fract::SampleCoord(f, (int)frags.fragSamples[k]);
                if (p.x < fr.minx || p.x > fr.maxx || p.y < fr.miny || p.y > fr.maxy ||
                    p.z < fr.minz || p.z > fr.maxz) allInside = false;
                const int dx = p.x - fr.cx, dy = p.y - fr.cy, dz = p.z - fr.cz;
                if (dx * dx + dy * dy + dz * dz > fr.boundRadiusSq) allInside = false;
                ++membersChecked;
            }
        }
        check(allInside, "multi: every member inside its fragment AABB + bounding sphere");
        check(membersChecked == f.sampleCount(), "multi: CSR covers every sample exactly once");
    }

    // ================= FR2: ExtractFragments determinism (two runs byte-identical) =================
    {
        fract::FractField f; f.nx = 12; f.ny = 9; f.nz = 6;
        std::vector<fract::FractSeed> seeds = {{1, 1, 1}, {10, 7, 4}, {5, 4, 2}, {2, 8, 5}, {9, 2, 1}};
        fract::FractCells cells; fract::ClassifyFractCells(f, seeds, cells);
        fract::FractFragments a, b;
        fract::ExtractFragments(f, cells, 5, a);
        fract::ExtractFragments(f, cells, 5, b);
        bool same = a.fragments.size() == b.fragments.size()
            && a.fragStart == b.fragStart && a.fragSamples == b.fragSamples
            && a.cellToFragment == b.cellToFragment && a.fragmentToCell == b.fragmentToCell
            && (a.fragments.empty() || std::memcmp(a.fragments.data(), b.fragments.data(),
                    a.fragments.size() * sizeof(fract::FractFragment)) == 0);
        check(same, "FR2 determinism: two ExtractFragments BYTE-IDENTICAL");
    }

    if (g_fail == 0) std::printf("fract_test: ALL PASS\n");
    else std::printf("fract_test: %d FAIL\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
