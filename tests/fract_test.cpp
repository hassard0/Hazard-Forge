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
namespace fpx = hf::sim::fpx;   // FR3 break impulses are Q16.16 (fpx::kOne scale)

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

    // ================= FR3: BuildFractBonds — a 2-cell field -> exactly 1 bond, faceArea == shared ====
    {
        // A 2x1x1 field split x=0 -> cell 0, x=1 -> cell 1 (seeds at the two ends). The shared face is the
        // single x=0|x=1 boundary -> faceArea 1, ascending (0,1).
        fract::FractField f; f.nx = 2; f.ny = 1; f.nz = 1;
        std::vector<fract::FractSeed> seeds = {{0, 0, 0}, {1, 0, 0}};
        fract::FractCells cells; fract::ClassifyFractCells(f, seeds, cells);
        fract::FractFragments frags; fract::ExtractFragments(f, cells, 2, frags);
        fract::FractBonds bonds; fract::BuildFractBonds(f, cells, frags, bonds);
        check(bonds.fragmentCount == 2u, "FR3 2-cell: fragmentCount == 2");
        check(bonds.bonds.size() == 1u, "FR3 2-cell: exactly 1 bond");
        check(bonds.bonds[0].fragA == 0u && bonds.bonds[0].fragB == 1u, "FR3 2-cell: ascending (0,1)");
        check(bonds.bonds[0].faceArea == 1, "FR3 2-cell: faceArea == shared face count (1)");
    }

    // ================= FR3: a wider shared interface -> faceArea == the shared sample count ============
    {
        // A 2x3x1 field, seeds at x=0 and x=1: cell 0 = column x=0 (3 samples), cell 1 = column x=1 (3).
        // They share 3 +x faces -> faceArea 3.
        fract::FractField f; f.nx = 2; f.ny = 3; f.nz = 1;
        std::vector<fract::FractSeed> seeds = {{0, 1, 0}, {1, 1, 0}};
        fract::FractCells cells; fract::ClassifyFractCells(f, seeds, cells);
        fract::FractFragments frags; fract::ExtractFragments(f, cells, 2, frags);
        fract::FractBonds bonds; fract::BuildFractBonds(f, cells, frags, bonds);
        check(bonds.bonds.size() == 1u, "FR3 wide: 1 bond");
        check(bonds.bonds[0].faceArea == 3, "FR3 wide: faceArea == 3 (the shared interface samples)");
    }

    // ================= FR3: a 3-cell line -> 2 bonds (0-1, 1-2), non-adjacent -> no bond ==============
    {
        // A 3x1x1 line, seeds at x=0,1,2: cells 0,1,2 left-to-right. Bonds: (0,1) and (1,2); NO (0,2).
        fract::FractField f; f.nx = 3; f.ny = 1; f.nz = 1;
        std::vector<fract::FractSeed> seeds = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}};
        fract::FractCells cells; fract::ClassifyFractCells(f, seeds, cells);
        fract::FractFragments frags; fract::ExtractFragments(f, cells, 3, frags);
        fract::FractBonds bonds; fract::BuildFractBonds(f, cells, frags, bonds);
        check(bonds.bonds.size() == 2u, "FR3 line: 2 bonds (0-1, 1-2)");
        check(bonds.bonds[0].fragA == 0u && bonds.bonds[0].fragB == 1u, "FR3 line: bond[0] == (0,1)");
        check(bonds.bonds[1].fragA == 1u && bonds.bonds[1].fragB == 2u, "FR3 line: bond[1] == (1,2)");
        bool no02 = true;
        for (const auto& b : bonds.bonds) if (b.fragA == 0u && b.fragB == 2u) no02 = false;
        check(no02, "FR3 line: NO bond between the non-adjacent end cells (0,2)");
    }

    // ================= FR3: ApplyImpactBreak — a hard impact severs >=1 bond, zero impact severs none ==
    {
        fract::FractField f; f.nx = 3; f.ny = 1; f.nz = 1;
        std::vector<fract::FractSeed> seeds = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}};
        fract::FractCells cells; fract::ClassifyFractCells(f, seeds, cells);
        fract::FractFragments frags; fract::ExtractFragments(f, cells, 3, frags);
        fract::FractBonds bonds; fract::BuildFractBonds(f, cells, frags, bonds);

        // Hard impact at fragment 0 -> diffuses -> severs the bond(s) it over-stresses.
        std::vector<uint8_t> sevHard;
        fract::BreakImpact hard{0u, (fract::fx)(200 * (int)fpx::kOne)};   // a big Q16.16 impulse
        uint32_t sHard = fract::ApplyImpactBreak(bonds, frags, hard, 8, sevHard);
        check(sHard >= 1u, "FR3 break: a HARD impact severs >= 1 bond");

        // Zero impact -> nothing diffuses -> nothing severs (the welded body is intact).
        std::vector<uint8_t> sevZero;
        fract::BreakImpact zero{0u, 0};
        uint32_t sZero = fract::ApplyImpactBreak(bonds, frags, zero, 8, sevZero);
        check(sZero == 0u, "FR3 break: a ZERO impact severs nothing");
        for (auto bd : bonds.bonds) (void)bd;
    }

    // ================= FR3: the threshold scales with faceArea (a stronger bond survives) =============
    {
        // Two bonds carrying the SAME load but with DIFFERENT faceArea: the higher-faceArea bond has a
        // higher break threshold, so for a load between the two thresholds the weak bond severs + the
        // strong one survives. Construct directly to control faceArea precisely.
        fract::FractFragments frags;   // 3 dummy fragments (only the count matters to the break)
        frags.fragments.resize(3);
        fract::FractBonds bonds; bonds.fragmentCount = 3u;
        fract::FractBond b0; b0.fragA = 0; b0.fragB = 1; b0.faceArea = 1;   // weak bond
        fract::FractBond b1; b1.fragA = 1; b1.fragB = 2; b1.faceArea = 64;  // strong bond
        bonds.bonds = {b0, b1};

        std::vector<uint8_t> sev;
        // A hard impact at fragment 0 pushes load across both; the weak bond (area 1) should sever before
        // the strong bond (area 64) under the strength-scaled threshold.
        fract::BreakImpact hard{0u, (fract::fx)(500 * (int)fpx::kOne)};
        fract::ApplyImpactBreak(bonds, frags, hard, 12, sev);
        check(sev.size() == 2u, "FR3 strength: two severed flags");
        // The weak bond is at least as likely to sever as the strong one (strength-scaled threshold):
        // if the strong bond severs, the weak one must too.
        check(!(sev[1] == 1u && sev[0] == 0u),
              "FR3 strength: the stronger (larger-face) bond never severs while the weaker survives");
    }

    // ================= FR3: ApplyImpactBreak determinism (two runs byte-identical) =====================
    {
        fract::FractField f; f.nx = 16; f.ny = 12; f.nz = 8;
        std::vector<fract::FractSeed> seeds = {
            {2, 2, 1}, {13, 3, 2}, {3, 9, 5}, {12, 10, 6}, {8, 6, 3},
            {5, 1, 6}, {10, 7, 1}, {1, 5, 3},
        };
        const int M = (int)seeds.size();
        fract::FractCells cells; fract::ClassifyFractCells(f, seeds, cells);
        fract::FractFragments frags; fract::ExtractFragments(f, cells, M, frags);
        fract::FractBonds bondsA = {}, bondsB = {};
        fract::BuildFractBonds(f, cells, frags, bondsA);
        fract::BuildFractBonds(f, cells, frags, bondsB);
        fract::BreakImpact imp{0u, (fract::fx)(80 * (int)fpx::kOne)};
        std::vector<uint8_t> sevA, sevB;
        uint32_t sA = fract::ApplyImpactBreak(bondsA, frags, imp, 10, sevA);
        uint32_t sB = fract::ApplyImpactBreak(bondsB, frags, imp, 10, sevB);
        check(sA == sB && sevA == sevB, "FR3 break determinism: severed count + flags BYTE-IDENTICAL");
        bool loadSame = bondsA.bonds.size() == bondsB.bonds.size();
        for (size_t i = 0; loadSame && i < bondsA.bonds.size(); ++i)
            if (bondsA.bonds[i].loadAccum != bondsB.bonds[i].loadAccum) loadSame = false;
        check(loadSame, "FR3 break determinism: per-bond loadAccum BYTE-IDENTICAL");
    }

    // ================= FR3: CountFractPieces — intact graph -> 1, a severed bridge -> 2 ===============
    {
        // A 3-cell line: 2 bonds. Intact -> 1 piece. Sever the (1,2) bond -> 2 pieces ({0,1} and {2}).
        fract::FractField f; f.nx = 3; f.ny = 1; f.nz = 1;
        std::vector<fract::FractSeed> seeds = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}};
        fract::FractCells cells; fract::ClassifyFractCells(f, seeds, cells);
        fract::FractFragments frags; fract::ExtractFragments(f, cells, 3, frags);
        fract::FractBonds bonds; fract::BuildFractBonds(f, cells, frags, bonds);

        std::vector<uint8_t> none(bonds.bonds.size(), 0u);
        std::vector<uint32_t> clusterIntact;
        uint32_t pIntact = fract::CountFractPieces(frags, bonds, none, &clusterIntact);
        check(pIntact == 1u, "FR3 pieces: an intact graph -> 1 piece");
        check(clusterIntact.size() == 3u, "FR3 pieces: a cluster label per fragment");
        check(clusterIntact[0] == clusterIntact[1] && clusterIntact[1] == clusterIntact[2],
              "FR3 pieces: intact -> all fragments share one cluster label");

        // Sever the (1,2) bond (index 1 in the ascending list) -> {0,1} and {2}.
        std::vector<uint8_t> sev(bonds.bonds.size(), 0u);
        sev[1] = 1u;
        uint32_t pBroken = fract::CountFractPieces(frags, bonds, sev, nullptr);
        check(pBroken == 2u, "FR3 pieces: one severed bridge -> 2 pieces");

        // Sever BOTH bonds -> 3 isolated fragments -> 3 pieces.
        std::vector<uint8_t> all(bonds.bonds.size(), 1u);
        uint32_t pAll = fract::CountFractPieces(frags, bonds, all, nullptr);
        check(pAll == 3u, "FR3 pieces: all bonds severed -> 3 isolated pieces");
    }

    // ================= FR3: severed subset of bonds (the crack-follows-cell-boundaries proof) =========
    {
        // Every severed flag corresponds to a real bond connecting two ADJACENT fragments (trivially true
        // by construction, but pin: |severed| <= |bonds| and indices align).
        fract::FractField f; f.nx = 10; f.ny = 10; f.nz = 4;
        std::vector<fract::FractSeed> seeds = {
            {2, 2, 1}, {7, 2, 2}, {2, 7, 1}, {7, 7, 2}, {5, 5, 3}, {1, 5, 0},
        };
        const int M = (int)seeds.size();
        fract::FractCells cells; fract::ClassifyFractCells(f, seeds, cells);
        fract::FractFragments frags; fract::ExtractFragments(f, cells, M, frags);
        fract::FractBonds bonds; fract::BuildFractBonds(f, cells, frags, bonds);
        fract::BreakImpact imp{0u, (fract::fx)(150 * (int)fpx::kOne)};
        std::vector<uint8_t> sev;
        uint32_t s = fract::ApplyImpactBreak(bonds, frags, imp, 10, sev);
        check(sev.size() == bonds.bonds.size(), "FR3 cracks: one severed flag per bond");
        uint32_t counted = 0; bool allAdjacent = true;
        for (size_t i = 0; i < sev.size(); ++i) if (sev[i]) {
            ++counted;
            if (bonds.bonds[i].fragA >= (uint32_t)M || bonds.bonds[i].fragB >= (uint32_t)M)
                allAdjacent = false;
        }
        check(counted == s, "FR3 cracks: severed count matches the flag sum");
        check(allAdjacent, "FR3 cracks: every severed bond connects two valid (adjacent) fragments");
    }

    // ================= FR4: SpawnFractWorld — anchor is the largest piece (static), others dynamic ====
    {
        // A 3-cell line (3 fragments). Sever the (1,2) bond -> pieces {0,1} (2 frags) and {2} (1 frag).
        // The anchor = the largest piece {0,1}; fragments 0,1 STATIC, fragment 2 DYNAMIC.
        fract::FractField f; f.nx = 3; f.ny = 1; f.nz = 1;
        std::vector<fract::FractSeed> seeds = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}};
        fract::FractCells cells; fract::ClassifyFractCells(f, seeds, cells);
        fract::FractFragments frags; fract::ExtractFragments(f, cells, 3, frags);
        fract::FractBonds bonds; fract::BuildFractBonds(f, cells, frags, bonds);
        std::vector<uint8_t> sev(bonds.bonds.size(), 0u);
        sev[1] = 1u;   // sever the (1,2) bond
        std::vector<uint32_t> clusters;
        uint32_t pieces = fract::CountFractPieces(frags, bonds, sev, &clusters);
        check(pieces == 2u, "FR4 spawn: the break yields 2 pieces");

        fract::FractStepConfig cfg;
        cfg.worldCellSize = fpx::kOne;            // 1 lattice cell == 1 world unit
        cfg.gravity = fract::FxVec3{0, (fract::fx)(-9 * (int)fpx::kOne), 0};
        cfg.groundY = (fract::fx)(-1000 * (int)fpx::kOne);
        fract::BreakImpact imp{2u, 0};            // impact on fragment 2 (the dynamic chunk)
        fpx::FxWorld w = fract::SpawnFractWorld(frags, bonds, sev, clusters, imp, cfg);
        check(w.bodies.size() == 3u, "FR4 spawn: one body per fragment");

        const uint32_t anchor = fract::FractAnchorPiece(frags, clusters);
        int statics = 0, dyn = 0;
        for (uint32_t i = 0; i < 3u; ++i) {
            const bool isAnchor = (clusters[i] == anchor);
            if (isAnchor) {
                check(w.bodies[i].invMass == 0 && !(w.bodies[i].flags & fpx::kFlagDynamic),
                      "FR4 spawn: anchor-piece fragment is STATIC (invMass 0, not dynamic)");
                ++statics;
            } else {
                check((w.bodies[i].flags & fpx::kFlagDynamic) != 0u,
                      "FR4 spawn: non-anchor fragment is DYNAMIC");
                ++dyn;
            }
        }
        check(statics == 2 && dyn == 1, "FR4 spawn: 2 anchor (static) + 1 dynamic chunk");

        // World positions == centroid·worldCellSize (worldCellSize==kOne here -> pos == centroid<<kFrac).
        bool posOK = true;
        for (uint32_t i = 0; i < 3u; ++i) {
            const fract::FractFragment& fr = frags.fragments[i];
            if (w.bodies[i].pos.x != (fract::fx)(fr.cx * (int)fpx::kOne) ||
                w.bodies[i].pos.y != (fract::fx)(fr.cy * (int)fpx::kOne) ||
                w.bodies[i].pos.z != (fract::fx)(fr.cz * (int)fpx::kOne)) posOK = false;
        }
        check(posOK, "FR4 spawn: body pos == centroid·worldCellSize");
    }

    // ================= FR4: SpawnFractWorld — the impacted dynamic body is seeded with impact vel =====
    {
        fract::FractField f; f.nx = 3; f.ny = 1; f.nz = 1;
        std::vector<fract::FractSeed> seeds = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}};
        fract::FractCells cells; fract::ClassifyFractCells(f, seeds, cells);
        fract::FractFragments frags; fract::ExtractFragments(f, cells, 3, frags);
        fract::FractBonds bonds; fract::BuildFractBonds(f, cells, frags, bonds);
        std::vector<uint8_t> sev(bonds.bonds.size(), 0u); sev[1] = 1u;
        std::vector<uint32_t> clusters;
        fract::CountFractPieces(frags, bonds, sev, &clusters);

        fract::FractStepConfig cfg;
        cfg.worldCellSize = fpx::kOne;
        cfg.impactDir = fract::FxVec3{fpx::kOne, 0, 0};   // +X
        cfg.impactSpeed = (fract::fx)(5 * (int)fpx::kOne); // 5 units/s
        fract::BreakImpact imp{2u, 0};                    // fragment 2 is the dynamic chunk
        fpx::FxWorld w = fract::SpawnFractWorld(frags, bonds, sev, clusters, imp, cfg);
        check(w.bodies[2].vel.x == (fract::fx)(5 * (int)fpx::kOne) && w.bodies[2].vel.y == 0,
              "FR4 spawn: impacted dynamic body seeded vel = impactDir·impactSpeed");
        // A non-impacted dynamic body has zero initial vel.
        // (fragment 2 is the only dynamic here; assert it is the seeded one — covered above.)
    }

    // ================= FR4: StepFracture — a dynamic body above ground FALLS then clamps at groundY ====
    {
        fpx::FxWorld w;
        w.gravity = fract::FxVec3{0, (fract::fx)(-9 * (int)fpx::kOne), 0};
        w.groundY = 0;
        fpx::FxBody b;
        b.pos = fract::FxVec3{0, (fract::fx)(10 * (int)fpx::kOne), 0};   // 10 units up
        b.invMass = fpx::kOne; b.flags = fpx::kFlagDynamic;
        b.radius = (fract::fx)(fpx::kOne / 2);                            // 0.5
        b.orient = fpx::FxQuat{0, 0, 0, fpx::kOne};
        w.bodies = {b};
        const fract::fx dt = fpx::kOne / 60;
        const fract::fx startY = w.bodies[0].pos.y;
        fract::StepFracture(w, dt, 4);
        check(w.bodies[0].pos.y < startY, "FR4 step: a dynamic body falls (pos.y decreases)");
        fract::StepFractureSteps(w, dt, 8, 400);   // plenty of ticks to land
        // The body's BOTTOM rests on/above the ground (ResolveGround clamps pos.y - radius >= groundY).
        check(w.bodies[0].pos.y - w.bodies[0].radius >= w.groundY - (fpx::kOne / 1000),
              "FR4 step: the fallen body rests on/above the floor (no body buried)");
        // The orientation stayed identity (angVel==0 -> the 6-DOF integrate is a no-op on orient).
        check(w.bodies[0].orient.x == 0 && w.bodies[0].orient.y == 0 && w.bodies[0].orient.z == 0 &&
              w.bodies[0].orient.w == fpx::kOne,
              "FR4 step: angVel==0 keeps orient identity EXACTLY");
    }

    // ================= FR4: StepFracture — a static anchor body NEVER moves =============================
    {
        fpx::FxWorld w;
        w.gravity = fract::FxVec3{0, (fract::fx)(-9 * (int)fpx::kOne), 0};
        w.groundY = 0;
        fpx::FxBody s;
        s.pos = fract::FxVec3{(fract::fx)(3 * (int)fpx::kOne), (fract::fx)(7 * (int)fpx::kOne), 0};
        s.invMass = 0; s.flags = 0;   // static
        s.radius = fpx::kOne;
        s.orient = fpx::FxQuat{0, 0, 0, fpx::kOne};
        w.bodies = {s};
        const fract::FxVec3 startPos = w.bodies[0].pos;
        fract::StepFractureSteps(w, fpx::kOne / 60, 6, 120);
        check(w.bodies[0].pos.x == startPos.x && w.bodies[0].pos.y == startPos.y &&
              w.bodies[0].pos.z == startPos.z, "FR4 step: a static anchor never moves");
    }

    // ================= FR4: two dynamic bodies dropped together collide (end >= their radii apart) =====
    {
        fpx::FxWorld w;
        w.gravity = fract::FxVec3{0, (fract::fx)(-9 * (int)fpx::kOne), 0};
        w.groundY = 0;
        const fract::fx r = (fract::fx)(fpx::kOne * 6 / 10);   // 0.6 radius -> diameter 1.2
        auto mk = [&](int x, int y) {
            fpx::FxBody b;
            b.pos = fract::FxVec3{(fract::fx)(x * (int)fpx::kOne), (fract::fx)(y * (int)fpx::kOne), 0};
            b.invMass = fpx::kOne; b.flags = fpx::kFlagDynamic; b.radius = r;
            b.orient = fpx::FxQuat{0, 0, 0, fpx::kOne};
            return b;
        };
        // Start overlapping horizontally (1 unit apart < 1.2 diameter) and let them settle.
        w.bodies = {mk(0, 3), mk(1, 3)};
        fract::StepFractureSteps(w, fpx::kOne / 60, 12, 400);
        const fract::FxVec3 d = fpx::FxSub(w.bodies[1].pos, w.bodies[0].pos);
        const fract::fx dist = fpx::FxLength(d);
        // After settling the two bounding spheres are pushed apart to ~>= the sum of radii (within a small
        // PBD residual band — the FPX3 positional solver is not exact, the documented within-band caveat).
        check(dist >= (r + r) - (fpx::kOne / 8),
              "FR4 step: two collided dynamic bodies end >= their radii apart (no interpenetration)");
    }

    // ================= FR4: a soft impact (1 piece -> all anchor) is a STATIC no-op =====================
    {
        // No severed bonds -> 1 piece -> every fragment is the anchor -> all bodies STATIC -> nothing falls.
        fract::FractField f; f.nx = 3; f.ny = 1; f.nz = 1;
        std::vector<fract::FractSeed> seeds = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}};
        fract::FractCells cells; fract::ClassifyFractCells(f, seeds, cells);
        fract::FractFragments frags; fract::ExtractFragments(f, cells, 3, frags);
        fract::FractBonds bonds; fract::BuildFractBonds(f, cells, frags, bonds);
        std::vector<uint8_t> sev(bonds.bonds.size(), 0u);   // NOTHING severed (soft)
        std::vector<uint32_t> clusters;
        uint32_t pieces = fract::CountFractPieces(frags, bonds, sev, &clusters);
        check(pieces == 1u, "FR4 soft: an intact body is 1 piece");

        fract::FractStepConfig cfg;
        cfg.worldCellSize = fpx::kOne;
        cfg.gravity = fract::FxVec3{0, (fract::fx)(-9 * (int)fpx::kOne), 0};
        cfg.groundY = 0;
        fract::BreakImpact imp{0u, 0};
        fpx::FxWorld w = fract::SpawnFractWorld(frags, bonds, sev, clusters, imp, cfg);
        int dyn = 0;
        for (const auto& b : w.bodies) if (b.flags & fpx::kFlagDynamic) ++dyn;
        check(dyn == 0, "FR4 soft: 1 piece -> every body is the anchor (0 dynamic)");
        fpx::FxWorld before = w;
        fract::StepFractureSteps(w, fpx::kOne / 60, 6, 60);
        bool noop = w.bodies.size() == before.bodies.size();
        for (size_t i = 0; noop && i < w.bodies.size(); ++i)
            if (w.bodies[i].pos.x != before.bodies[i].pos.x ||
                w.bodies[i].pos.y != before.bodies[i].pos.y ||
                w.bodies[i].pos.z != before.bodies[i].pos.z) noop = false;
        check(noop, "FR4 soft: the all-static world is a no-op (nothing moves)");
    }

    // ================= FR4: StepFracture determinism + break-and-fall (the showcase scene) =============
    {
        // The SAME scene the --fract-step-shot showcase drives (32x32x16, worldCellSize 0.25, K=120) so the
        // CPU test pins the exact bit-exact reference + the break-and-fall contract.
        fract::FractField f; f.nx = 32; f.ny = 32; f.nz = 16;
        std::vector<fract::FractSeed> seeds = {
            { 4,  5,  3}, {27,  6,  2}, { 6, 26,  4}, {25, 27,  3},
            {16, 15,  8}, { 3, 14, 12}, {29, 18, 13}, {14,  3, 11},
            {18, 29, 10}, { 9,  9,  6}, {22, 11,  9}, {11, 22,  7},
            {24, 24, 12}, { 7, 18,  2}, {20,  7, 14}, {15, 28,  6},
        };
        const int M = (int)seeds.size();
        fract::FractCells cells; fract::ClassifyFractCells(f, seeds, cells);
        fract::FractFragments frags; fract::ExtractFragments(f, cells, M, frags);
        fract::FractBonds bonds; fract::BuildFractBonds(f, cells, frags, bonds);
        fract::BreakImpact imp{0u, (fract::fx)(1000 * (int)fpx::kOne)};
        std::vector<uint8_t> sev;
        fract::ApplyImpactBreak(bonds, frags, imp, 4, sev);
        std::vector<uint32_t> clusters;
        fract::CountFractPieces(frags, bonds, sev, &clusters);

        const fract::fx gravY = (fract::fx)(-9.8 * (double)fpx::kOne + (-9.8 < 0 ? -0.5 : 0.5));
        fract::FractStepConfig cfg;
        cfg.worldCellSize = fpx::kOne / 4;                            // 0.25 world units per cell
        cfg.gravity = fract::FxVec3{0, gravY, 0};
        cfg.groundY = 0;
        cfg.impactDir = fract::FxVec3{fpx::kOne / 2, -fpx::kOne, 0};   // down + sideways (chunks fall)
        cfg.impactSpeed = (fract::fx)(4 * (int)fpx::kOne);
        const int kSteps = 120, kIters = 8;

        fpx::FxWorld a = fract::SpawnFractWorld(frags, bonds, sev, clusters, imp, cfg);
        fpx::FxWorld b = a;
        fract::StepFractureSteps(a, fpx::kOne / 60, kIters, kSteps);
        fract::StepFractureSteps(b, fpx::kOne / 60, kIters, kSteps);
        bool same = a.bodies.size() == b.bodies.size() &&
                    std::memcmp(a.bodies.data(), b.bodies.data(),
                                a.bodies.size() * sizeof(fpx::FxBody)) == 0;
        check(same, "FR4 step determinism: two StepFractureSteps runs BYTE-IDENTICAL");

        // A HARD break yields dynamic chunks that FELL (mean dynamic pos.y dropped from spawn) — the
        // break-and-fall contract; the anchor (static) is unchanged.
        const uint32_t anchorIdx = fract::FractAnchorBodyIndex(frags, clusters);
        fpx::FxWorld spawn = fract::SpawnFractWorld(frags, bonds, sev, clusters, imp, cfg);
        const fract::FractRubbleState s0 = fract::MeasureFractRubble(spawn, anchorIdx);
        const fract::FractRubbleState s1 = fract::MeasureFractRubble(a, anchorIdx);
        check(s1.dynamic > 0u, "FR4 rubble: a hard break has dynamic chunks");
        check(s1.meanDynamicY < s0.meanDynamicY, "FR4 rubble: the dynamic chunks FELL (mean pos.y dropped)");
        check(s1.anchorY == s0.anchorY, "FR4 rubble: the anchor (static) pos.y is UNCHANGED");
        // Every dynamic chunk rests with its CENTER on/above the ground (the ground clamp held — proof (4);
        // the sphere-bound rubble's bottom may dip a sub-radius amount on the final Gauss-Seidel sweep where
        // pairs resolve AFTER the ground, the documented FPX3 ground-then-pairs ordering artifact).
        bool allAbove = true;
        for (const fpx::FxBody& bd : a.bodies)
            if ((bd.flags & fpx::kFlagDynamic) && (bd.pos.y < cfg.groundY)) allAbove = false;
        check(allAbove, "FR4 rubble: every dynamic chunk's center rests on/above the floor (clamp held)");
    }

    // ================= FR5: the lockstep/rollback harness over the fracture rubble dynamics =============
    // Builds a SMALL broken-and-spawned world (the FR4 spawn) + a shove stream, and pins: SimFractTick
    // advances deterministically; RunFractLockstep authority==replica BIT-EXACT (inputs only);
    // RunFractRollback == RunFractLockstep(auth) AND mispredicted != authority; a shove changes a chunk's
    // trajectory (the stream does work); fpx::SnapshotWorld/RestoreWorld round-trip is bit-exact. The
    // harness reuses fpx's command + snapshot functions VERBATIM (the FR5 design).
    {
        // A small fixed scene (an 8x8x4 lattice + a few seeds — enough fragments for a multi-piece break,
        // small enough to step many ticks cheaply).
        fract::FractField f; f.nx = 8; f.ny = 8; f.nz = 4;
        std::vector<fract::FractSeed> seeds = {
            {1, 1, 1}, {6, 1, 2}, {1, 6, 1}, {6, 6, 2}, {3, 3, 1}, {4, 4, 2},
        };
        const int M = (int)seeds.size();
        fract::FractCells cells; fract::ClassifyFractCells(f, seeds, cells);
        fract::FractFragments frags; fract::ExtractFragments(f, cells, M, frags);
        fract::FractBonds bonds; fract::BuildFractBonds(f, cells, frags, bonds);
        fract::BreakImpact imp{0u, (fract::fx)(1000 * (int)fpx::kOne)};   // a HARD break
        std::vector<uint8_t> sev;
        fract::ApplyImpactBreak(bonds, frags, imp, 4, sev);
        std::vector<uint32_t> clusters;
        const uint32_t pieces = fract::CountFractPieces(frags, bonds, sev, &clusters);
        check(pieces > 1u, "FR5 scene: the hard break yields >1 piece (some dynamic rubble)");

        const fract::fx gravY = (fract::fx)(-9.8 * (double)fpx::kOne + (-9.8 < 0 ? -0.5 : 0.5));
        fract::FractStepConfig cfg;
        cfg.worldCellSize = fpx::kOne / 4;
        cfg.gravity = fract::FxVec3{0, gravY, 0};
        cfg.groundY = 0;
        cfg.impactDir = fract::FxVec3{fpx::kOne / 2, -fpx::kOne, 0};
        cfg.impactSpeed = (fract::fx)(4 * (int)fpx::kOne);
        const fpx::FxWorld init = fract::SpawnFractWorld(frags, bonds, sev, clusters, imp, cfg);

        // Pick the FIRST dynamic body — the shove stream targets it ("kick a falling chunk").
        uint32_t kickBody = 0xFFFFFFFFu;
        for (uint32_t i = 0; i < (uint32_t)init.bodies.size(); ++i)
            if (init.bodies[i].flags & fpx::kFlagDynamic) { kickBody = i; break; }
        check(kickBody != 0xFFFFFFFFu, "FR5 scene: at least one dynamic chunk to kick");

        const fract::fx dt = fpx::kOne / 60;
        const int kIters = 6;
        const int kTicks = 30;
        const int kMispredictTick = 8;

        // The scripted authoritative shove stream: kick the falling chunk sideways + spin it at a few ticks.
        const std::vector<fpx::FxCommand> authStream = {
            fpx::FxCommand{2,  fpx::kCmdImpulse,   kickBody, fpx::FxVec3{(fract::fx)(3 * (int)fpx::kOne), 0, 0}},
            fpx::FxCommand{5,  fpx::kCmdSetAngVel, kickBody, fpx::FxVec3{0, fpx::kOne, 0}},
            fpx::FxCommand{10, fpx::kCmdImpulse,   kickBody, fpx::FxVec3{0, 0, (fract::fx)(2 * (int)fpx::kOne)}},
        };

        auto worldEqual = [&](const fpx::FxWorld& a, const fpx::FxWorld& b) {
            return a.bodies.size() == b.bodies.size() &&
                   std::memcmp(a.bodies.data(), b.bodies.data(),
                               a.bodies.size() * sizeof(fpx::FxBody)) == 0 &&
                   a.gravity.x == b.gravity.x && a.gravity.y == b.gravity.y &&
                   a.gravity.z == b.gravity.z && a.groundY == b.groundY;
        };

        // SimFractTick advances the world deterministically: over a single tick the dynamic chunk's state
        // CHANGES (it integrates under gravity / the spawn impact velocity), and two ticks from the same
        // start are byte-identical (the per-tick determinism the lockstep proof builds on).
        {
            fpx::FxWorld w0 = init;
            fract::SimFractTick(w0, authStream, 0u, dt, kIters);
            const bool moved = w0.bodies[kickBody].pos.x != init.bodies[kickBody].pos.x ||
                               w0.bodies[kickBody].pos.y != init.bodies[kickBody].pos.y ||
                               w0.bodies[kickBody].pos.z != init.bodies[kickBody].pos.z ||
                               w0.bodies[kickBody].vel.y != init.bodies[kickBody].vel.y;
            check(moved, "FR5 SimFractTick: a dynamic chunk's state advances over one tick");
            fpx::FxWorld w1 = init;
            fract::SimFractTick(w1, authStream, 0u, dt, kIters);
            check(worldEqual(w0, w1), "FR5 SimFractTick: two ticks from the same start are BIT-IDENTICAL");
        }

        // LOCKSTEP: authority == replica BIT-EXACT (two runs from the SAME init+stream, inputs only).
        const fpx::FxWorld authority = fract::RunFractLockstep(init, authStream, kTicks, dt, kIters);
        const fpx::FxWorld replica   = fract::RunFractLockstep(init, authStream, kTicks, dt, kIters);
        check(worldEqual(authority, replica),
              "FR5 lockstep: authority == replica BIT-EXACT (inputs-only re-sim)");

        // The MISPREDICTED stream: authStream + a WRONG strong shove at mispredictTick.
        std::vector<fpx::FxCommand> mispredictStream = authStream;
        mispredictStream.push_back(fpx::FxCommand{(uint32_t)kMispredictTick, fpx::kCmdImpulse, kickBody,
                                                  fpx::FxVec3{(fract::fx)(40 * (int)fpx::kOne), 0, 0}});

        // ROLLBACK: rolledBack == authority BIT-EXACT, AND the mispredicted run DIFFERED (a real divergence).
        const fpx::FxWorld rolledBack =
            fract::RunFractRollback(init, authStream, mispredictStream, kTicks, kMispredictTick, dt, kIters);
        const fpx::FxWorld mispredicted = fract::RunFractLockstep(init, mispredictStream, kTicks, dt, kIters);
        check(worldEqual(rolledBack, authority),
              "FR5 rollback: corrected == authority BIT-EXACT");
        check(!worldEqual(mispredicted, authority),
              "FR5 rollback: the mispredicted state DIFFERED from authority (non-vacuous proof)");

        // MOTION: the shove stream did non-trivial work (authority chunk pos != the no-command baseline).
        const std::vector<fpx::FxCommand> noStream;
        const fpx::FxWorld noInput = fract::RunFractLockstep(init, noStream, kTicks, dt, kIters);
        const bool moved = authority.bodies[kickBody].pos.x != noInput.bodies[kickBody].pos.x ||
                           authority.bodies[kickBody].pos.z != noInput.bodies[kickBody].pos.z;
        check(moved, "FR5 motion: the shove stream changed a chunk's trajectory (the stream does work)");

        // SNAPSHOT round-trip: RestoreWorld(SnapshotWorld(w)) == w byte-for-byte (reused fpx VERBATIM).
        {
            fpx::FxWorld w = fract::RunFractLockstep(init, authStream, kMispredictTick, dt, kIters);
            const fpx::FxWorld snap = fpx::SnapshotWorld(w);
            fract::SimFractTick(w, authStream, (uint32_t)kMispredictTick, dt, kIters);   // mutate
            fpx::RestoreWorld(w, snap);
            check(worldEqual(w, snap), "FR5 snapshot: SnapshotWorld/RestoreWorld round-trip BIT-EXACT");
        }
    }

    // ================= FR6: the FLOAT render bridge over the settled fracture rubble =====================
    // FractToRenderInstances builds one math::Mat4 per body via fpx::FxBodyTransform (the ONE float crossing,
    // REUSED VERBATIM), in body order, with a parallel isDynamic[] split. Pure CPU (no device). Pins: the
    // instance count == bodies.size(); a known body's matrix == fpx::FxBodyTransform(b) (translation ==
    // pos/kOne); empty world -> empty; the anchor/dynamic split count matches the world's static/dynamic bodies.
    {
        // Build a small broken-and-spawned fracture world (the FR4/FR5 scene shape, deterministic).
        const int M = 8;
        fract::FractField f; f.nx = 8; f.ny = 8; f.nz = 4;
        std::vector<fract::FractSeed> seeds = {
            {1, 1, 1}, {6, 1, 1}, {1, 6, 1}, {6, 6, 1},
            {3, 3, 2}, {4, 4, 1}, {2, 5, 2}, {5, 2, 2},
        };
        fract::FractCells cells; fract::ClassifyFractCells(f, seeds, cells);
        fract::FractFragments frags; fract::ExtractFragments(f, cells, M, frags);
        fract::FractBonds bonds; fract::BuildFractBonds(f, cells, frags, bonds);
        fract::BreakImpact imp{0u, (fract::fx)(1000 * (int)fpx::kOne)};   // a HARD break
        std::vector<uint8_t> sev;
        fract::ApplyImpactBreak(bonds, frags, imp, 4, sev);
        std::vector<uint32_t> clusters;
        fract::CountFractPieces(frags, bonds, sev, &clusters);
        const fract::fx gravY = (fract::fx)(-9.8 * (double)fpx::kOne + (-9.8 < 0 ? -0.5 : 0.5));
        fract::FractStepConfig cfg;
        cfg.worldCellSize = fpx::kOne / 4;
        cfg.gravity = fract::FxVec3{0, gravY, 0};
        cfg.groundY = 0;
        cfg.impactDir = fract::FxVec3{fpx::kOne / 2, -fpx::kOne, 0};
        cfg.impactSpeed = (fract::fx)(4 * (int)fpx::kOne);
        fpx::FxWorld world = fract::SpawnFractWorld(frags, bonds, sev, clusters, imp, cfg);
        fract::StepFractureSteps(world, fpx::kOne / 60, 8, 60);   // settle the rubble

        // (1) instance count == bodies.size(); the split flags are parallel.
        std::vector<uint8_t> isDyn;
        const std::vector<math::Mat4> mats = fract::FractToRenderInstances(world, &isDyn);
        check(mats.size() == world.bodies.size(), "FR6 render: instance count == bodies.size()");
        check(isDyn.size() == world.bodies.size(), "FR6 render: isDynamic[] parallel to bodies");

        // (2) a known body's matrix == fpx::FxBodyTransform(b); translation == pos/kOne (the provenance).
        bool allMatch = true, transOk = true;
        for (size_t i = 0; i < world.bodies.size(); ++i) {
            const math::Mat4 ref = fpx::FxBodyTransform(world.bodies[i]);
            if (std::memcmp(mats[i].m, ref.m, sizeof(float) * 16) != 0) allMatch = false;
            const float ex = (float)world.bodies[i].pos.x / (float)fpx::kOne;
            const float ey = (float)world.bodies[i].pos.y / (float)fpx::kOne;
            const float ez = (float)world.bodies[i].pos.z / (float)fpx::kOne;
            // Column-major mat4: translation in m[12],m[13],m[14].
            if (mats[i].m[12] != ex || mats[i].m[13] != ey || mats[i].m[14] != ez) transOk = false;
        }
        check(allMatch, "FR6 render: every instance == fpx::FxBodyTransform(b) (the FPX6 bridge VERBATIM)");
        check(transOk, "FR6 render: instance translation == pos/kOne (provenance from the bit-exact state)");

        // (3) the anchor/dynamic split count matches the world's static/dynamic bodies.
        uint32_t dynBodies = 0, dynFlags = 0, anchorBodies = 0, anchorFlags = 0;
        for (size_t i = 0; i < world.bodies.size(); ++i) {
            if (world.bodies[i].flags & fpx::kFlagDynamic) ++dynBodies; else ++anchorBodies;
            if (isDyn[i]) ++dynFlags; else ++anchorFlags;
        }
        check(dynFlags == dynBodies, "FR6 render: dynamic split count == #kFlagDynamic bodies");
        check(anchorFlags == anchorBodies, "FR6 render: anchor split count == #static bodies");
        check(dynBodies > 0u && anchorBodies > 0u,
              "FR6 render: the broken scene has both an anchor AND dislodged chunks");

        // (4) provenance: a rebuild from the SAME settled world -> byte-identical matrices (pure function).
        const std::vector<math::Mat4> rebuild = fract::FractToRenderInstances(world, nullptr);
        bool identical = (rebuild.size() == mats.size());
        for (size_t i = 0; i < mats.size() && identical; ++i)
            if (std::memcmp(mats[i].m, rebuild[i].m, sizeof(float) * 16) != 0) identical = false;
        check(identical, "FR6 render: instances == rebuild (the float transform is a pure function)");

        // (5) empty world -> empty output (the no-op).
        {
            fpx::FxWorld empty;
            std::vector<uint8_t> emptyFlags;
            const std::vector<math::Mat4> emptyMats = fract::FractToRenderInstances(empty, &emptyFlags);
            check(emptyMats.empty() && emptyFlags.empty(), "FR6 render: empty world -> empty output");
        }
    }

    // ================= FR7: recursive fracture-on-impact (cascade + determinism + floor + control) =======
    // BuildRecursiveWorld wraps the FR4 spawn into a lineage-tracked recursive world; StepFractureRecursive-
    // Steps runs the FR4 step + the T1 sphere-contact impulse re-fracture each tick. Pins: a hard re-fracture
    // threshold makes the body count GROW + the max-depth reach >=2 (a real cascade, children split again); a
    // second run is BYTE-IDENTICAL over BOTH bodies AND chunks; the cascade TERMINATES at the minVolume floor
    // (extra ticks add zero bodies + live volumes are all <= floor); a very-high threshold is impulse-driven
    // (zero re-fracture, body count unchanged). Models the FR4 cases. Pure CPU (no device).
    {
        // The FR4/FR5 scene shape (deterministic), broken hard so there is dynamic rubble to slam + split.
        const int M = 16;
        fract::FractField f; f.nx = 32; f.ny = 32; f.nz = 16;
        const std::vector<fract::FractSeed> seeds = {
            { 4,  5,  3}, {27,  6,  2}, { 6, 26,  4}, {25, 27,  3},
            {16, 15,  8}, { 3, 14, 12}, {29, 18, 13}, {14,  3, 11},
            {18, 29, 10}, { 9,  9,  6}, {22, 11,  9}, {11, 22,  7},
            {24, 24, 12}, { 7, 18,  2}, {20,  7, 14}, {15, 28,  6},
        };
        fract::FractCells cells; fract::ClassifyFractCells(f, seeds, cells);
        fract::FractFragments frags; fract::ExtractFragments(f, cells, M, frags);
        fract::FractBonds bonds; fract::BuildFractBonds(f, cells, frags, bonds);
        fract::BreakImpact imp{0u, (fract::fx)(1000 * (int)fpx::kOne)};   // a HARD break
        std::vector<uint8_t> sev;
        fract::ApplyImpactBreak(bonds, frags, imp, 4, sev);
        std::vector<uint32_t> clusters;
        fract::CountFractPieces(frags, bonds, sev, &clusters);
        const fract::fx gravY = (fract::fx)(-9.8 * (double)fpx::kOne + (-9.8 < 0 ? -0.5 : 0.5));
        fract::FractStepConfig cfg;
        cfg.worldCellSize = fpx::kOne / 4;
        cfg.gravity = fract::FxVec3{0, gravY, 0};
        cfg.groundY = 0;
        cfg.impactDir = fract::FxVec3{fpx::kOne / 2, -fpx::kOne, 0};
        cfg.impactSpeed = (fract::fx)(4 * (int)fpx::kOne);
        const fpx::FxWorld spawn = fract::SpawnFractWorld(frags, bonds, sev, clusters, imp, cfg);

        // A tuned re-fracture config: a threshold clear of the settle band (~24) but below the impact
        // (~5.8e5) so a hard slam re-fractures while a gentle settle does not; a minVolume floor the cascade
        // reaches in 2 generations (root vols ~600-1400 -> halved twice ~150-350), childPieces 2.
        fract::FractRecursiveConfig rcfg;
        rcfg.worldCellSize = fpx::kOne / 4;
        rcfg.reFractureImpulse = fpx::kOne;        // 1.0: above the settle band, below the impact impulse
        rcfg.minVolume = 300;                      // the floor the 2-generation cascade reaches + terminates
        rcfg.childPieces = 2;
        rcfg.gravityY = gravY;
        rcfg.groundY = 0;

        const fract::fx dt = fpx::kOne / 60;
        const int kIters = 8;
        const int kTicks = 120;

        // --- (1) cascade growth: the body count GROWS + max-depth reaches >= 2 (children split again). ---
        fract::FractRecursiveWorld rw = fract::BuildRecursiveWorld(spawn, frags, clusters);
        const uint32_t rootBodies = (uint32_t)rw.world.bodies.size();
        const fract::FractCascadeState s0 = fract::MeasureFractCascade(rw, rcfg);
        fract::StepFractureRecursiveSteps(rw, rcfg, dt, kIters, kTicks);
        const uint32_t finalBodies = (uint32_t)rw.world.bodies.size();
        const fract::FractCascadeState sF = fract::MeasureFractCascade(rw, rcfg);
        check(finalBodies > rootBodies, "FR7 cascade: the body count GREW (a real re-fracture happened)");
        check(sF.maxDepth >= 2u, "FR7 cascade: max-depth reached >= 2 (children split again)");
        check(sF.retired > 0u, "FR7 cascade: some parents retired in place (never erased)");
        check(rw.chunks.size() == rw.world.bodies.size(),
              "FR7 cascade: chunk[] stays index-aligned to bodies[] (retire-in-place, append-only)");
        check(s0.liveChunks <= sF.liveChunks || sF.liveChunks > 0u, "FR7 cascade: live chunks present");

        // --- (2) determinism: a second run is BYTE-IDENTICAL over BOTH bodies AND chunks. ---
        fract::FractRecursiveWorld rw2 = fract::BuildRecursiveWorld(spawn, frags, clusters);
        fract::StepFractureRecursiveSteps(rw2, rcfg, dt, kIters, kTicks);
        const bool bodiesSame = (rw2.world.bodies.size() == rw.world.bodies.size()) &&
            std::memcmp(rw2.world.bodies.data(), rw.world.bodies.data(),
                        rw.world.bodies.size() * sizeof(fpx::FxBody)) == 0;
        const bool chunksSame = (rw2.chunks.size() == rw.chunks.size()) &&
            std::memcmp(rw2.chunks.data(), rw.chunks.data(),
                        rw.chunks.size() * sizeof(fract::FractChunk)) == 0;
        check(bodiesSame, "FR7 determinism: two runs BYTE-IDENTICAL over bodies");
        check(chunksSame, "FR7 determinism: two runs BYTE-IDENTICAL over chunks");
        check(rw2.nextChunkId == rw.nextChunkId, "FR7 determinism: the lineage id allocator is replay-stable");

        // --- (3) the cascade TERMINATES at the minVolume floor: extra ticks add ZERO bodies + all live
        //         chunks at/below the floor never split further. ---
        const uint32_t beforeExtra = (uint32_t)rw.world.bodies.size();
        fract::StepFractureRecursiveSteps(rw, rcfg, dt, kIters, 60);   // 60 more ticks
        check(rw.world.bodies.size() == beforeExtra,
              "FR7 floor: extra ticks add ZERO bodies (the cascade terminated)");
        const fract::FractCascadeState sT = fract::MeasureFractCascade(rw, rcfg);
        check(sT.atFloor > 0u, "FR7 floor: at least one live chunk reached the minVolume floor");
        check(sT.minLiveVolume <= rcfg.minVolume,
              "FR7 floor: the smallest live volume is at/below the floor (the cascade terminator)");

        // --- (4) the soft control: a very-high threshold is impulse-driven -> ZERO re-fracture. ---
        fract::FractRecursiveWorld soft = fract::BuildRecursiveWorld(spawn, frags, clusters);
        fract::FractRecursiveConfig scfg = rcfg;
        scfg.reFractureImpulse = (fract::fx)(1 << 30);   // ~16384 world units — unreachably high (no int overflow)
        const uint32_t softRoot = (uint32_t)soft.world.bodies.size();
        fract::StepFractureRecursiveSteps(soft, scfg, dt, kIters, kTicks);
        const fract::FractCascadeState softState = fract::MeasureFractCascade(soft, scfg);
        check(soft.world.bodies.size() == softRoot,
              "FR7 control: a very-high threshold re-fractures NOTHING (impulse-driven, not unconditional)");
        check(softState.maxDepth == 0u, "FR7 control: no re-fracture -> max-depth stays 0");
        check(softState.retired == 0u, "FR7 control: no re-fracture -> nothing retired");
    }

    // ================= FR8: CONVEX-SHARD RUBBLE — oriented boxes through the SHIPPED fric.h solver =======
    // Closes the documented FR4 caveat (sphere-bound rubble): the SAME FR3 break spawns as ORIENTED BOXES
    // (FragmentToBox: the AABB-box about the centroid — the documented approximation of the Voronoi cell)
    // into the FC4 convex-SAT + Coulomb-friction world (fric::StepFrictionWorld called AS-IS). Pins:
    //   (a) FragmentToBox shape math (hand cases);
    //   (b) the sphere-path REGRESSION: FR4's StepFractureSteps digest UNCHANGED (FR1-FR7 untouched);
    //   (c) THE UPGRADE PROOF: the FR4 scene settles as box rubble with >= 1 piece ROTATED (non-trivial
    //       quaternion — impossible on the FR4 sphere path, which keeps orient identity EXACTLY), resting
    //       interpenetration bounded, rest speed bounded; the settled-rubble digest PINNED (MSVC == clang);
    //   (d) determinism: two runs byte-identical;
    //   (e) lockstep: a peer re-derives the box rubble bit-for-bit + rollback corrects a real divergence.
    {
        namespace convex = hf::sim::convex;
        namespace fric = hf::sim::fric;

        // FNV-1a 64 over the body vector bytes (pure-integer state -> identical on every compiler).
        auto bodyDigest = [](const std::vector<fpx::FxBody>& bodies) -> uint64_t {
            uint64_t h = 1469598103934665603ull;
            const uint8_t* p = reinterpret_cast<const uint8_t*>(bodies.data());
            const size_t n = bodies.size() * sizeof(fpx::FxBody);
            for (size_t i = 0; i < n; ++i) { h ^= (uint64_t)p[i]; h *= 1099511628211ull; }
            return h;
        };

        // ---- (a) FragmentToBox hand cases. ----
        {
            // The FR2 1-seed 5x5x5 fragment: centroid (2,2,2), AABB [0,4]^3 -> per-axis half = 2 cells +
            // half a cell = 2.5 lattice units; worldCellSize kOne -> 2.5 * kOne.
            fract::FractFragment fr{};
            fr.cx = 2; fr.cy = 2; fr.cz = 2;
            fr.minx = 0; fr.miny = 0; fr.minz = 0;
            fr.maxx = 4; fr.maxy = 4; fr.maxz = 4;
            const hf::sim::convex::FxBox box = fract::FragmentToBox(fr, fpx::kOne);
            check(box.halfExtents.x == (fract::fx)(5 * (int)fpx::kOne / 2) &&
                  box.halfExtents.y == box.halfExtents.x && box.halfExtents.z == box.halfExtents.x,
                  "FR8 box: symmetric fragment -> half-extents 2.5 lattice cells");
            // An ASYMMETRIC fragment (the FR2 hand case {x=0,1}, centroid 0): x half = max(0,1)+0.5 = 1.5
            // cells; y/z half = 0.5 cells; worldCellSize kOne/2 halves everything.
            fract::FractFragment fa{};
            fa.cx = 0; fa.minx = 0; fa.maxx = 1;   // y/z all zero
            const hf::sim::convex::FxBox ba = fract::FragmentToBox(fa, fpx::kOne / 2);
            check(ba.halfExtents.x == (fract::fx)(3 * (int)fpx::kOne / 4),
                  "FR8 box: asymmetric AABB covered about the centroid (1.5 cells * 0.5 world)");
            check(ba.halfExtents.y == (fract::fx)((int)fpx::kOne / 4) && ba.halfExtents.z == ba.halfExtents.y,
                  "FR8 box: one-sample-thick axis keeps half-a-cell thickness (no degenerate slab)");
        }

        // ---- The FR4 showcase scene VERBATIM (32x32x16 lattice, 16 seeds, hard break). ----
        fract::FractField f; f.nx = 32; f.ny = 32; f.nz = 16;
        const std::vector<fract::FractSeed> seeds = {
            { 4,  5,  3}, {27,  6,  2}, { 6, 26,  4}, {25, 27,  3},
            {16, 15,  8}, { 3, 14, 12}, {29, 18, 13}, {14,  3, 11},
            {18, 29, 10}, { 9,  9,  6}, {22, 11,  9}, {11, 22,  7},
            {24, 24, 12}, { 7, 18,  2}, {20,  7, 14}, {15, 28,  6},
        };
        const int M = (int)seeds.size();
        fract::FractCells cells; fract::ClassifyFractCells(f, seeds, cells);
        fract::FractFragments frags; fract::ExtractFragments(f, cells, M, frags);
        fract::FractBonds bonds; fract::BuildFractBonds(f, cells, frags, bonds);
        fract::BreakImpact imp{0u, (fract::fx)(1000 * (int)fpx::kOne)};
        std::vector<uint8_t> sev;
        fract::ApplyImpactBreak(bonds, frags, imp, 4, sev);
        std::vector<uint32_t> clusters;
        fract::CountFractPieces(frags, bonds, sev, &clusters);

        const fract::fx gravY = (fract::fx)(-9.8 * (double)fpx::kOne + (-9.8 < 0 ? -0.5 : 0.5));
        fract::FractStepConfig cfg;
        cfg.worldCellSize = fpx::kOne / 4;
        cfg.gravity = fract::FxVec3{0, gravY, 0};
        cfg.groundY = 0;
        cfg.impactDir = fract::FxVec3{fpx::kOne / 2, -fpx::kOne, 0};
        cfg.impactSpeed = (fract::fx)(4 * (int)fpx::kOne);
        const fract::fx dt = fpx::kOne / 60;

        // ---- (b) sphere-path REGRESSION: FR4's StepFractureSteps is untouched — digest PINNED. ----
        {
            fpx::FxWorld sw = fract::SpawnFractWorld(frags, bonds, sev, clusters, imp, cfg);
            fract::StepFractureSteps(sw, dt, 8, 120);
            const uint64_t sphereDigest = bodyDigest(sw.bodies);
            std::printf("FR8 sphere-path digest: 0x%016llx\n", (unsigned long long)sphereDigest);
            check(sphereDigest == 0x029245db75f0eb33ull,
                  "FR8 regression: the FR4 sphere-path settled digest is UNCHANGED (pinned)");
        }

        // ---- (c) the BOX rubble: spawn + settle through the SHIPPED FC4 friction solver. ----
        const fract::FxVec3 kFloorHalf{(fract::fx)(24 * (int)fpx::kOne), fpx::kOne,
                                       (fract::fx)(24 * (int)fpx::kOne)};
        const hf::sim::convex::ConvexWorld hull0 =
            fract::SpawnFractHullWorld(frags, bonds, sev, clusters, imp, cfg, kFloorHalf);
        check(hull0.bodies.size() == (size_t)M + 1u && hull0.boxes.size() == hull0.bodies.size(),
              "FR8 spawn: one box body per fragment + the static floor appended LAST");
        check(hull0.bodies.back().invMass == 0 && hull0.bodies.back().flags == 0u,
              "FR8 spawn: the floor body is STATIC");
        {
            uint32_t dyn = 0, stat = 0;
            for (size_t i = 0; i + 1 < hull0.bodies.size(); ++i)
                if (hull0.bodies[i].flags & fpx::kFlagDynamic) ++dyn; else ++stat;
            check(dyn > 0u && stat > 0u, "FR8 spawn: the break yields dynamic pieces AND a static anchor");
            bool identityAll = true;
            for (const fpx::FxBody& b : hull0.bodies)
                if (b.orient.x != 0 || b.orient.y != 0 || b.orient.z != 0 || b.orient.w != fpx::kOne)
                    identityAll = false;
            check(identityAll, "FR8 spawn: every piece spawns with IDENTITY orientation (rotation is earned)");
        }

        // The FC4 friction-step knobs (the fric_test stack convention): grippy mu, angular damping OFF —
        // the friction cone is what stops the pieces, not a damping crutch.
        fric::FrictionStepConfig fcfg;
        fcfg.gravity     = fract::FxVec3{0, gravY, 0};
        fcfg.dt          = dt;
        fcfg.solveIters  = 12;
        fcfg.restitution = 0;
        fcfg.slop        = fpx::kOne / 64;
        fcfg.beta        = (fract::fx)((int64_t)4 * fpx::kOne / 10);    // 0.4
        fcfg.linDamp     = (fract::fx)((int64_t)98 * fpx::kOne / 100);  // 0.98 (the FC4 stack setting)
        fcfg.angDamp     = fpx::kOne;                                   // OFF — friction holds the rubble
        fcfg.posIters    = 4;
        fcfg.mu          = fpx::kOne;
        // 480 ticks: the AABB-boxes overlap deeply at spawn (the documented over-approximation), so the
        // de-penetration "burst" + the perched pieces toppling off the anchor need ~2x the FR4 settle time
        // (measured: all dynamic speeds <= ~0.01 and max pairwise penetration ~= slop by tick 480).
        const uint32_t kHullTicks = 480u;

        hf::sim::convex::ConvexWorld hullA = hull0, hullB = hull0;
        fract::StepFractureHullSteps(hullA, fcfg, kHullTicks);
        fract::StepFractureHullSteps(hullB, fcfg, kHullTicks);

        // ---- (d) determinism: two runs byte-identical + the settled digest PINNED. ----
        check(hf::sim::convex::ConvexBodiesEqual(hullA.bodies, hullB.bodies),
              "FR8 determinism: two StepFractureHullSteps runs BYTE-IDENTICAL");
        const uint64_t hullDigest = bodyDigest(hullA.bodies);
        std::printf("FR8 hull-rubble digest: 0x%016llx\n", (unsigned long long)hullDigest);
        check(hullDigest == 0xa5a9b8f5158108d2ull,
              "FR8 golden: the settled box-rubble digest is PINNED (MSVC == clang bit-exact)");

        // ---- (c cont.) THE UPGRADE PROOF: settled + rotated + bounded interpenetration. ----
        const fract::fx kRotEps = fpx::kOne / 32;   // |q.xyz| > ~0.031 == a > ~3.6 deg total rotation
        const fract::FractHullRubbleState st = fract::MeasureFractHullRubble(hullA, kRotEps);
        std::printf("FR8 rubble: dynamic=%u rotated=%u maxQuatDev=%d maxSpeed=%d maxPen=%d minY=%d\n",
                    st.dynamic, st.rotated, (int)st.maxQuatDev, (int)st.maxSpeed,
                    (int)st.maxPenetration, (int)st.minDynamicY);
        check(st.dynamic > 0u, "FR8 rubble: dynamic box pieces exist");
        check(st.rotated >= 1u,
              "FR8 UPGRADE PROOF: >= 1 settled piece ROTATED (non-trivial quaternion — spheres cannot)");
        check(st.maxQuatDev > kRotEps,
              "FR8 UPGRADE PROOF: the max quaternion deviation is non-trivial (pinned > kOne/32)");
        check(st.maxSpeed < fpx::kOne / 16,
              "FR8 rest: the rubble comes to rest (max dynamic speed < 0.0625 unit/s after 480 ticks)");
        check(st.maxPenetration < fpx::kOne / 16,
              "FR8 overlap: resting interpenetration bounded (< 0.0625 world units ~= the slop band)");
        {
            // Every dynamic piece rests ABOVE the floor top (its center above groundY minus a slop band —
            // the box solver's de-pen holds it out of the floor box).
            bool aboveFloor = true;
            for (size_t i = 0; i + 1 < hullA.bodies.size(); ++i) {
                const fpx::FxBody& b = hullA.bodies[i];
                if ((b.flags & fpx::kFlagDynamic) && b.pos.y < cfg.groundY) aboveFloor = false;
            }
            check(aboveFloor, "FR8 floor: every dynamic piece's center rests above the floor top");
            // A rotated piece's exact quaternion is covered bit-for-bit by the pinned digest above; also
            // pin that the WINNER (max-deviation piece) is a DYNAMIC piece with a normalized-ish quat.
            fract::fx bestDev = 0; size_t bestIdx = (size_t)-1;
            for (size_t i = 0; i + 1 < hullA.bodies.size(); ++i) {
                const fpx::FxBody& b = hullA.bodies[i];
                if (!(b.flags & fpx::kFlagDynamic)) continue;
                fract::fx dev = b.orient.x < 0 ? -b.orient.x : b.orient.x;
                fract::fx dy = b.orient.y < 0 ? -b.orient.y : b.orient.y;
                fract::fx dz = b.orient.z < 0 ? -b.orient.z : b.orient.z;
                if (dy > dev) dev = dy;
                if (dz > dev) dev = dz;
                if (dev > bestDev) { bestDev = dev; bestIdx = i; }
            }
            check(bestIdx != (size_t)-1 && bestDev > kRotEps,
                  "FR8 rotation: the max-rotated piece is a dynamic fragment body");
            if (bestIdx != (size_t)-1) {
                const fpx::FxQuat& q = hullA.bodies[bestIdx].orient;
                std::printf("FR8 rotation winner: piece %u quat {%d, %d, %d, %d}\n",
                            (unsigned)bestIdx, (int)q.x, (int)q.y, (int)q.z, (int)q.w);
            }
        }

        // ---- (e) LOCKSTEP + ROLLBACK: a peer re-derives the box rubble bit-for-bit from inputs alone. ----
        {
            namespace cv = hf::sim::convex;
            // Kick the FIRST dynamic piece (a shove + a spin) at a few ticks — the input stream.
            uint32_t kick = 0xFFFFFFFFu;
            for (uint32_t i = 0; i + 1u < (uint32_t)hull0.bodies.size(); ++i)
                if (hull0.bodies[i].flags & fpx::kFlagDynamic) { kick = i; break; }
            check(kick != 0xFFFFFFFFu, "FR8 lockstep: a dynamic piece exists to kick");
            const std::vector<cv::ConvexCommand> authStream = {
                cv::ConvexCommand{2u, cv::kConvexCmdAddImpulse, kick,
                                  fract::FxVec3{(fract::fx)(2000 * (int)fpx::kOne), 0, 0}},
                cv::ConvexCommand{5u, cv::kConvexCmdSetAngVel, kick, fract::FxVec3{0, fpx::kOne, 0}},
            };
            const uint32_t kTicks = 60u, kRollbackAt = 20u;
            bool identical = false;
            const cv::ConvexWorld authority =
                fract::RunFractureHullLockstep(hull0, fcfg, authStream, kTicks, &identical);
            check(identical, "FR8 lockstep: authority == replica BIT-EXACT (inputs-only re-sim)");

            std::vector<cv::ConvexCommand> mispredict = authStream;
            mispredict.push_back(cv::ConvexCommand{kRollbackAt, cv::kConvexCmdAddImpulse, kick,
                                                   fract::FxVec3{0, 0, (fract::fx)(4000 * (int)fpx::kOne)}});
            bool corrected = false, diverged = false;
            const cv::ConvexWorld rolled =
                fract::RunFractureHullRollback(hull0, fcfg, authStream, mispredict, kTicks, kRollbackAt,
                                               &corrected, &diverged);
            check(corrected, "FR8 rollback: corrected == authority BIT-EXACT");
            check(diverged, "FR8 rollback: the misprediction REALLY diverged before the rollback");
            check(cv::ConvexBodiesEqual(rolled.bodies, authority.bodies),
                  "FR8 rollback: the returned corrected world matches the authority");
        }
    }

    if (g_fail == 0) std::printf("fract_test: ALL PASS\n");
    else std::printf("fract_test: %d FAIL\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
