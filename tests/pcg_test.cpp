// Slice PCG1 — Seeded hash-PRNG primitive (engine/pcg/pcg.h), the BEACHHEAD of FLAGSHIP #22 (DETERMINISTIC
// PCG). Pure CPU (header-only, no device, no backend symbols). Namespace hf::pcg. The deterministic seeded
// integer hash-PRNG in Q16.16, pure int32 (MSL-native): PcgHash (the ParticleHash avalanche shape verbatim),
// PcgRand01 ([0,kOne) from the top 16 hash bits — a pure shift, no float/div), PcgRandRange, PcgUnitDir (the
// EmitDir no-trig direction table), and PcgStream (a {seed,salt} so distinct layers hash to distinct streams).
//
// What this test PINS (the contracts the later PCG slices + the cross-backend integer golden build on):
//   * replay-stable — same (seed,index) -> identical PcgHash/PcgRand01/PcgUnitDir across calls.
//   * seed-sensitive — a different seed yields a different sequence over a swept index range.
//   * range bounds — PcgRand01 in [0,kOne) over a swept range; PcgRandRange in [lo,hi].
//   * salt separation — two PcgStreams with different salt give different sequences for the same index.
//   * uniformity sanity — bucket N=4096 PcgRand01 samples into K bins, every bin within a LOOSE INTEGER band
//     of N/K (a fixed integer assertion — NO float tolerance).
//
// Pure C++ (hf_core), ASan-eligible like the other sim/render-math tests. pcg.h #includes fpx.h + particles.h
// read-only.
#include "pcg/pcg.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace pcg = hf::pcg;
using pcg::fx;
using pcg::kOne;
using pcg::kFrac;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();

    // ================= replay-stable: same (seed,index) -> identical values across calls =================
    {
        const uint32_t seed = 12345u;
        bool stable = true;
        for (uint32_t i = 0; i < 64 && stable; ++i) {
            if (pcg::PcgHash(seed, i) != pcg::PcgHash(seed, i)) stable = false;
            if (pcg::PcgRand01(seed, i) != pcg::PcgRand01(seed, i)) stable = false;
            const pcg::FxVec3 a = pcg::PcgUnitDir(seed, i);
            const pcg::FxVec3 b = pcg::PcgUnitDir(seed, i);
            if (a.x != b.x || a.y != b.y || a.z != b.z) stable = false;
        }
        check(stable, "PCG1 replay-stable: same (seed,index) -> identical hash/rand/dir across calls");
        // A handful of distinct indices -> distinct hashes (no trivial collisions in a small sweep).
        bool distinct = true;
        for (uint32_t a = 0; a < 16 && distinct; ++a)
            for (uint32_t b = a + 1; b < 16; ++b)
                if (pcg::PcgHash(seed, a) == pcg::PcgHash(seed, b)) { distinct = false; break; }
        check(distinct, "PCG1 PcgHash: distinct indices -> distinct hashes (small sweep)");
    }

    // ================= seed-sensitive: a different seed yields a different sequence =======================
    {
        const uint32_t seedA = 1u, seedB = 2u;
        const uint32_t N = 256u;
        uint32_t differ = 0;
        for (uint32_t i = 0; i < N; ++i)
            if (pcg::PcgRand01(seedA, i) != pcg::PcgRand01(seedB, i)) ++differ;
        // The vast majority of samples must differ (a different seed is a genuinely different stream).
        check(differ > N * 3 / 4, "PCG1 seed-sensitive: a different seed yields a different sequence");
    }

    // ================= range bounds: PcgRand01 in [0,kOne); PcgRandRange in [lo,hi] =======================
    {
        const uint32_t seed = 777u;
        const uint32_t N = 4096u;
        bool inUnit = true;
        for (uint32_t i = 0; i < N && inUnit; ++i) {
            const fx r = pcg::PcgRand01(seed, i);
            if (r < 0 || r >= kOne) inUnit = false;
        }
        check(inUnit, "PCG1 range: PcgRand01 in [0, kOne) over a swept index range (strict < kOne)");

        const fx lo = -kOne * 3, hi = kOne * 5;       // [-3, 5] in Q16.16
        bool inRange = true;
        for (uint32_t i = 0; i < N && inRange; ++i) {
            const fx r = pcg::PcgRandRange(seed, i, lo, hi);
            if (r < lo || r > hi) inRange = false;
        }
        check(inRange, "PCG1 range: PcgRandRange in [lo, hi] over a swept index range");

        // PcgRandRange with lo==hi collapses to lo exactly (degenerate width -> the constant).
        bool degen = true;
        for (uint32_t i = 0; i < 64 && degen; ++i)
            if (pcg::PcgRandRange(seed, i, kOne, kOne) != kOne) degen = false;
        check(degen, "PCG1 range: PcgRandRange(lo==hi) == lo exactly (degenerate width)");
    }

    // ================= salt separation: two PcgStreams with different salt diverge ========================
    {
        const pcg::PcgStream sa{42u, 0u};
        const pcg::PcgStream sb{42u, 0xABCDEF01u};    // SAME seed, DIFFERENT salt
        const uint32_t N = 256u;
        uint32_t differ = 0;
        for (uint32_t i = 0; i < N; ++i)
            if (pcg::PcgRand01(sa, i) != pcg::PcgRand01(sb, i)) ++differ;
        check(differ > N * 3 / 4, "PCG1 salt separation: same seed, different salt -> different sequence");

        // A zero-salt stream matches the bare-seed overload (salt 0 is the identity fold).
        bool matchesBare = true;
        for (uint32_t i = 0; i < N && matchesBare; ++i)
            if (pcg::PcgRand01(sa, i) != pcg::PcgRand01(42u, i)) matchesBare = false;
        check(matchesBare, "PCG1 salt: zero-salt stream == bare-seed overload (salt 0 identity)");
    }

    // ================= uniformity sanity: bucket 4096 samples into K bins, each within a loose band =======
    {
        const uint32_t seed = 0xC0FFEEu;
        const uint32_t N = 4096u;
        const int K = 16;                             // 16 bins -> N/K == 256 expected
        std::vector<uint32_t> bins((size_t)K, 0u);
        for (uint32_t i = 0; i < N; ++i) {
            const fx r = pcg::PcgRand01(seed, i);     // in [0, kOne) == [0, 65536)
            // bin index = the top 4 of the 16 fraction bits -> 0..15 (pure integer shift, no float).
            const int b = (int)((uint32_t)r >> (kFrac - 4));
            check(b >= 0 && b < K, "PCG1 uniformity: bin index in [0,K)");
            ++bins[(size_t)b];
        }
        // Every bin within a LOOSE integer band of N/K (256): expect [128, 384] (a fixed +-50% integer band,
        // no float tolerance). A well-mixed avalanche hash easily satisfies this.
        const uint32_t expected = N / (uint32_t)K;    // 256
        const uint32_t loBand = expected / 2;         // 128
        const uint32_t hiBand = expected + expected / 2;  // 384
        bool uniform = true;
        uint32_t total = 0;
        for (int b = 0; b < K; ++b) {
            total += bins[(size_t)b];
            if (bins[(size_t)b] < loBand || bins[(size_t)b] > hiBand) uniform = false;
        }
        check(total == N, "PCG1 uniformity: all N samples counted (partition complete)");
        check(uniform, "PCG1 uniformity: every bin within a loose integer band of N/K (no float tolerance)");
    }

    // ================= PCG2: jittered-grid scatter (count / in-cell / replay / seed-sensitive / no-op) ====
    {
        // The scatter area: a fixed square XZ patch in Q16.16 world units, Y on the ground plane.
        const pcg::PcgArea area{ pcg::FxVec3{0, kOne * 2, 0}, pcg::FxVec3{kOne * 16, kOne * 2, kOne * 16} };
        const int cellsX = 8, cellsZ = 6;                  // 48 cells (non-square -> exercises both axes)
        const pcg::PcgStream stA{ 2024u, 0x5CA77E20u };
        const pcg::PcgStream stB{ 9173u, 0x5CA77E20u };    // DIFFERENT seed, same salt

        // (1) count — exactly cellsX*cellsZ points.
        const std::vector<pcg::FxVec3> ptsA = pcg::ScatterGrid(stA, area, cellsX, cellsZ);
        check(ptsA.size() == (size_t)cellsX * cellsZ, "PCG2 count: ScatterGrid returns cellsX*cellsZ points");

        // (2) in-cell containment — every point inside its OWN cell's integer AABB (pure integer compares).
        const fx cellW = (area.max.x - area.min.x) / cellsX;
        const fx cellD = (area.max.z - area.min.z) / cellsZ;
        bool inCell = (ptsA.size() == (size_t)cellsX * cellsZ);
        for (int cz = 0; cz < cellsZ && inCell; ++cz)
            for (int cx = 0; cx < cellsX && inCell; ++cx) {
                const size_t idx = (size_t)(cz * cellsX + cx);
                const fx cellMinX = area.min.x + cellW * cx;
                const fx cellMinZ = area.min.z + cellD * cz;
                const pcg::FxVec3& p = ptsA[idx];
                if (!(p.x >= cellMinX && p.x < cellMinX + cellW)) inCell = false;
                if (!(p.z >= cellMinZ && p.z < cellMinZ + cellD)) inCell = false;
                if (p.y != area.min.y) inCell = false;
            }
        check(inCell, "PCG2 in-cell: every point lies strictly inside its own cell's integer AABB (Y == min.y)");

        // (3) replay-stable — two calls with the same args are byte-equal (element-by-element).
        const std::vector<pcg::FxVec3> ptsA2 = pcg::ScatterGrid(stA, area, cellsX, cellsZ);
        bool replay = (ptsA.size() == ptsA2.size());
        for (size_t i = 0; i < ptsA.size() && replay; ++i)
            if (ptsA[i].x != ptsA2[i].x || ptsA[i].y != ptsA2[i].y || ptsA[i].z != ptsA2[i].z) replay = false;
        check(replay, "PCG2 replay-stable: two calls with the same args are byte-equal");

        // (4) seed-sensitive — a different seed -> a DIFFERENT layout but the SAME count.
        const std::vector<pcg::FxVec3> ptsB = pcg::ScatterGrid(stB, area, cellsX, cellsZ);
        check(ptsB.size() == ptsA.size(), "PCG2 seed-sensitive: a different seed keeps the same count");
        bool differ = false;
        for (size_t i = 0; i < ptsA.size() && !differ; ++i)
            if (ptsA[i].x != ptsB[i].x || ptsA[i].z != ptsB[i].z) differ = true;
        check(differ, "PCG2 seed-sensitive: a different seed yields a different layout");

        // (5) no-op — cellsX<=0 or a degenerate area -> empty vector.
        check(pcg::ScatterGrid(stA, area, 0, cellsZ).empty(), "PCG2 no-op: cellsX<=0 -> empty");
        check(pcg::ScatterGrid(stA, area, cellsX, -1).empty(), "PCG2 no-op: cellsZ<=0 -> empty");
        const pcg::PcgArea degenerate{ pcg::FxVec3{kOne, 0, kOne}, pcg::FxVec3{kOne, 0, kOne} };  // zero extent
        check(pcg::ScatterGrid(stA, degenerate, cellsX, cellsZ).empty(), "PCG2 no-op: degenerate area -> empty");
    }

    // ================= PCG3: density mask + importance rejection (no-op / zero / monotone / replay / contain) =
    {
        // The same fixed square XZ patch as PCG2's checks, a sub-area radius for a genuine subset.
        const pcg::PcgArea area{ pcg::FxVec3{0, kOne * 2, 0}, pcg::FxVec3{kOne * 16, kOne * 2, kOne * 16} };
        const int cellsX = 8, cellsZ = 6;                  // 48 candidate cells
        const pcg::PcgStream st{ 2024u, 0x5CA77E20u };

        // The PCG2 candidates (the importance-rejection input — the mask only REMOVES from THIS set).
        const std::vector<pcg::FxVec3> grid = pcg::ScatterGrid(st, area, cellsX, cellsZ);

        // (1) no-op == PCG2 — a Uniform mask + density == kOne returns BYTE-EQUAL to ScatterGrid (same
        // positions, same order). The make-or-break no-op proof.
        pcg::PcgMask uniform;                               // default type == Uniform
        const std::vector<pcg::FxVec3> kept = pcg::ScatterMasked(st, area, cellsX, cellsZ, uniform, kOne);
        bool noop = (kept.size() == grid.size());
        for (size_t i = 0; i < grid.size() && noop; ++i)
            if (kept[i].x != grid[i].x || kept[i].y != grid[i].y || kept[i].z != grid[i].z) noop = false;
        check(noop, "PCG3 no-op: Uniform mask + density==kOne == ScatterGrid byte-equal (positions+order)");

        // (2) zero mask — a Radial mask with radius <= 0 keeps 0 points; AND density==0 keeps 0 points.
        pcg::PcgMask zeroRadial;
        zeroRadial.type = pcg::PcgMaskType::Radial;
        zeroRadial.center = pcg::FxVec3{ kOne * 8, kOne * 2, kOne * 8 };
        zeroRadial.radius = 0;                             // degenerate -> 0 everywhere
        check(pcg::ScatterMasked(st, area, cellsX, cellsZ, zeroRadial, kOne).empty(),
              "PCG3 zero mask: Radial radius<=0 keeps 0 points");
        check(pcg::ScatterMasked(st, area, cellsX, cellsZ, uniform, 0).empty(),
              "PCG3 zero mask: density==0 keeps 0 points");

        // (3) radial monotone — kept count NON-DECREASING as density sweeps 0 -> kOne, AND a radial disc with a
        // radius smaller than the area keeps a STRICT subset (< full grid).
        pcg::PcgMask radial;
        radial.type = pcg::PcgMaskType::Radial;
        radial.center = pcg::FxVec3{ kOne * 8, kOne * 2, kOne * 8 };   // centre of the 16x16 area
        radial.radius = kOne * 5;                          // < the area extent -> a genuine subset
        size_t prev = 0;
        bool monotone = true;
        for (int s = 0; s <= 8 && monotone; ++s) {
            const fx density = (fx)((int64_t)kOne * s / 8);            // 0, kOne/8, ... kOne
            const size_t n = pcg::ScatterMasked(st, area, cellsX, cellsZ, radial, density).size();
            if (n < prev) monotone = false;
            prev = n;
        }
        check(monotone, "PCG3 radial monotone: kept count non-decreasing as density sweeps 0 -> kOne");
        const std::vector<pcg::FxVec3> disc = pcg::ScatterMasked(st, area, cellsX, cellsZ, radial, kOne);
        check(disc.size() < grid.size() && !disc.empty(),
              "PCG3 radial subset: radius < area keeps a strict non-empty subset (< full grid)");

        // (4) replay-stable — two calls with the same args are byte-equal.
        const std::vector<pcg::FxVec3> disc2 = pcg::ScatterMasked(st, area, cellsX, cellsZ, radial, kOne);
        bool replay = (disc.size() == disc2.size());
        for (size_t i = 0; i < disc.size() && replay; ++i)
            if (disc[i].x != disc2[i].x || disc[i].y != disc2[i].y || disc[i].z != disc2[i].z) replay = false;
        check(replay, "PCG3 replay-stable: two ScatterMasked calls with the same args are byte-equal");

        // (5) containment — every survivor is one of the ScatterGrid candidates (the mask only REMOVES, never
        // moves). Same fixed order, so each survivor matches some candidate exactly.
        bool contained = true;
        for (size_t i = 0; i < disc.size() && contained; ++i) {
            bool found = false;
            for (size_t j = 0; j < grid.size() && !found; ++j)
                if (disc[i].x == grid[j].x && disc[i].y == grid[j].y && disc[i].z == grid[j].z) found = true;
            if (!found) contained = false;
        }
        check(contained, "PCG3 containment: every survivor is one of the ScatterGrid candidates (mask removes only)");
    }

    // ================= PCG4: per-instance transform rules (count / unit-orient / scale-range / no-op /
    // replay / seed-sensitive) ===========================================================================
    {
        // The same fixed square XZ patch as PCG2/PCG3's checks; a small grid so the sweep is cheap.
        const pcg::PcgArea area{ pcg::FxVec3{0, kOne * 2, 0}, pcg::FxVec3{kOne * 16, kOne * 2, kOne * 16} };
        const int cellsX = 8, cellsZ = 6;                  // 48 points
        const pcg::PcgStream stA{ 2024u, 0x5CA77E20u };
        const pcg::PcgStream stB{ 9173u, 0x5CA77E20u };    // DIFFERENT seed, same salt

        const std::vector<pcg::FxVec3> pts = pcg::ScatterGrid(stA, area, cellsX, cellsZ);
        const pcg::fx scaleLo = kOne / 2, scaleHi = kOne * 3 / 2;     // [0.5, 1.5]
        pcg::PcgTransform rule;                            // randomYaw == true by default
        rule.scaleLo = scaleLo; rule.scaleHi = scaleHi;

        const std::vector<pcg::PcgInstance> insA = pcg::BuildInstances(pts, stA, rule);

        // (1) count — exactly points.size() instances.
        check(insA.size() == pts.size(), "PCG4 count: BuildInstances returns one instance per input point");

        // (2) unit orient — every orient satisfies |y^2 + w^2 - kOne^2| within a loose integer tolerance, and
        // x == z == 0 exactly (the table is a pure +Y yaw). Tolerance ~ kOne*kOne/1000.
        const int64_t kOne2 = (int64_t)kOne * (int64_t)kOne;
        const int64_t tol = kOne2 / 1000;
        bool unit = (insA.size() == pts.size());
        for (size_t i = 0; i < insA.size() && unit; ++i) {
            const pcg::PcgInstance& in = insA[i];
            if (in.orient.x != 0 || in.orient.z != 0) unit = false;
            const int64_t mag2 = (int64_t)in.orient.y * (int64_t)in.orient.y +
                                 (int64_t)in.orient.w * (int64_t)in.orient.w;
            int64_t diff = mag2 - kOne2; if (diff < 0) diff = -diff;
            if (diff > tol) unit = false;
        }
        check(unit, "PCG4 unit orient: every orient is unit-within-tol (y^2+w^2 ~= kOne^2; x==z==0 exactly)");

        // (3) scale in range — every scale in [scaleLo, scaleHi].
        bool inRange = (insA.size() == pts.size());
        for (size_t i = 0; i < insA.size() && inRange; ++i)
            if (insA[i].scale < scaleLo || insA[i].scale > scaleHi) inRange = false;
        check(inRange, "PCG4 scale range: every scale in [scaleLo, scaleHi]");

        // (4) no-op — rule{randomYaw=false, scaleLo=kOne, scaleHi=kOne} -> identity orient + scale==kOne +
        // pos == the input points (the transform only annotates, never moves).
        pcg::PcgTransform noop;
        noop.randomYaw = false; noop.scaleLo = kOne; noop.scaleHi = kOne;
        const std::vector<pcg::PcgInstance> insN = pcg::BuildInstances(pts, stA, noop);
        bool isNoop = (insN.size() == pts.size());
        for (size_t i = 0; i < insN.size() && isNoop; ++i) {
            const pcg::PcgInstance& in = insN[i];
            if (in.orient.x != 0 || in.orient.y != 0 || in.orient.z != 0 || in.orient.w != kOne) isNoop = false;
            if (in.scale != kOne) isNoop = false;
            if (in.pos.x != pts[i].x || in.pos.y != pts[i].y || in.pos.z != pts[i].z) isNoop = false;
        }
        check(isNoop, "PCG4 no-op: identity yaw + fixed scale -> identity orient + scale==kOne + pos==input");

        // (5) replay-stable — two calls byte-equal (element-by-element).
        const std::vector<pcg::PcgInstance> insA2 = pcg::BuildInstances(pts, stA, rule);
        bool replay = (insA.size() == insA2.size());
        for (size_t i = 0; i < insA.size() && replay; ++i) {
            const pcg::PcgInstance& a = insA[i]; const pcg::PcgInstance& b = insA2[i];
            if (a.pos.x != b.pos.x || a.pos.y != b.pos.y || a.pos.z != b.pos.z) replay = false;
            if (a.orient.x != b.orient.x || a.orient.y != b.orient.y ||
                a.orient.z != b.orient.z || a.orient.w != b.orient.w) replay = false;
            if (a.scale != b.scale) replay = false;
        }
        check(replay, "PCG4 replay-stable: two BuildInstances calls with the same args are byte-equal");

        // (6) seed-sensitive — a different seed -> a different yaw/scale sequence (same count).
        const std::vector<pcg::PcgInstance> insB = pcg::BuildInstances(pts, stB, rule);
        check(insB.size() == insA.size(), "PCG4 seed-sensitive: a different seed keeps the same count");
        bool differ = false;
        for (size_t i = 0; i < insA.size() && !differ; ++i)
            if (insA[i].scale != insB[i].scale || insA[i].orient.y != insB[i].orient.y) differ = true;
        check(differ, "PCG4 seed-sensitive: a different seed yields a different yaw/scale sequence");
    }

    // ================= PCG5: declarative layered pipeline + overlap-prune (THE DETERMINISM HEADLINE) =======
    // no-overlaps / prune-removes-some / shuffle-invariance / lockstep+purity / no-op / empty.
    {
        // A fixed graph: a denser scatter so the prune has real work to do (overlaps to remove).
        const pcg::PcgArea area{ pcg::FxVec3{0, 0, 0}, pcg::FxVec3{kOne * 32, 0, kOne * 32} };
        const int cellsX = 24, cellsZ = 24;                // 576 candidate cells (dense -> overlaps)
        const pcg::PcgStream st{ 1337u, 0x5CA77E20u };

        pcg::PcgGraph g;
        g.area = area; g.cellsX = cellsX; g.cellsZ = cellsZ;
        g.useMask = true;
        g.mask.type = pcg::PcgMaskType::Radial;
        g.mask.center = pcg::FxVec3{ kOne * 16, 0, kOne * 16 };
        g.mask.radius = kOne * 32 / 3;
        g.density = kOne;
        g.transform.randomYaw = true; g.transform.scaleLo = kOne / 2; g.transform.scaleHi = kOne * 3 / 2;
        g.prune = true;
        g.pruneRadius = kOne;                              // big enough to force removals in the dense field

        // The pre-prune instances (mask+transform, NO prune) — the PruneOverlaps input.
        pcg::PcgGraph gNoPrune = g; gNoPrune.prune = false;
        const std::vector<pcg::PcgInstance> beforePrune = pcg::Generate(gNoPrune, st);

        // The pruned output.
        const std::vector<pcg::PcgInstance> survivors = pcg::Generate(g, st);

        // (1) no overlaps — EVERY surviving pair satisfies FxLength(pos_i-pos_j) >= r_i+r_j (O(K^2)).
        bool noOverlap = true;
        for (size_t i = 0; i < survivors.size() && noOverlap; ++i)
            for (size_t j = i + 1; j < survivors.size() && noOverlap; ++j) {
                const pcg::FxVec3 d{ survivors[i].pos.x - survivors[j].pos.x, 0,
                                     survivors[i].pos.z - survivors[j].pos.z };
                const fx dist = hf::sim::fpx::FxLength(d);
                const fx ri = pcg::fxmul(g.pruneRadius, survivors[i].scale);
                const fx rj = pcg::fxmul(g.pruneRadius, survivors[j].scale);
                if (dist < ri + rj) noOverlap = false;
            }
        check(noOverlap, "PCG5 no overlaps: every surviving pair has FxLength >= r_i+r_j (prune removed all overlaps)");

        // (2) prune removes some — survivors < instances before prune (it did real work).
        check(survivors.size() < beforePrune.size() && !survivors.empty(),
              "PCG5 prune removes some: survivors < beforePrune (a non-empty strict subset)");

        // (3) shuffle-invariance — reverse the pre-prune vector (a FIXED deterministic permutation, NO rng),
        // run PruneOverlaps on BOTH, assert survivor sets BYTE-IDENTICAL (same positions + canonical order).
        std::vector<pcg::PcgInstance> shuffled(beforePrune.rbegin(), beforePrune.rend());
        const std::vector<pcg::PcgInstance> survOrig = pcg::PruneOverlaps(beforePrune, g.pruneRadius);
        const std::vector<pcg::PcgInstance> survShuf = pcg::PruneOverlaps(shuffled,    g.pruneRadius);
        bool shuffleMatch = (survOrig.size() == survShuf.size());
        for (size_t i = 0; i < survOrig.size() && shuffleMatch; ++i) {
            const pcg::PcgInstance& a = survOrig[i]; const pcg::PcgInstance& b = survShuf[i];
            if (a.pos.x != b.pos.x || a.pos.y != b.pos.y || a.pos.z != b.pos.z) shuffleMatch = false;
            if (a.orient.x != b.orient.x || a.orient.y != b.orient.y ||
                a.orient.z != b.orient.z || a.orient.w != b.orient.w) shuffleMatch = false;
            if (a.scale != b.scale) shuffleMatch = false;
        }
        check(shuffleMatch, "PCG5 shuffle-invariance: shuffled input -> SAME survivors (order-canonical prune)");

        // (4) lockstep / purity — two Generate(g,st) calls byte-equal; a different seed -> different but valid
        // (still no overlaps).
        const std::vector<pcg::PcgInstance> survB = pcg::Generate(g, st);
        bool lockstep = (survivors.size() == survB.size());
        for (size_t i = 0; i < survivors.size() && lockstep; ++i) {
            const pcg::PcgInstance& a = survivors[i]; const pcg::PcgInstance& b = survB[i];
            if (a.pos.x != b.pos.x || a.pos.y != b.pos.y || a.pos.z != b.pos.z) lockstep = false;
            if (a.orient.y != b.orient.y || a.orient.w != b.orient.w || a.scale != b.scale) lockstep = false;
        }
        check(lockstep, "PCG5 lockstep/purity: two Generate(g,stream) calls are byte-equal (peer determinism)");

        const pcg::PcgStream st2{ 9173u, 0x5CA77E20u };    // a DIFFERENT seed
        const std::vector<pcg::PcgInstance> survDiff = pcg::Generate(g, st2);
        bool differ = (survDiff.size() != survivors.size());
        for (size_t i = 0; i < survDiff.size() && i < survivors.size() && !differ; ++i)
            if (survDiff[i].pos.x != survivors[i].pos.x || survDiff[i].pos.z != survivors[i].pos.z) differ = true;
        check(differ, "PCG5 seed-sensitive: a different seed -> a different field");
        bool diffNoOverlap = true;
        for (size_t i = 0; i < survDiff.size() && diffNoOverlap; ++i)
            for (size_t j = i + 1; j < survDiff.size() && diffNoOverlap; ++j) {
                const pcg::FxVec3 d{ survDiff[i].pos.x - survDiff[j].pos.x, 0,
                                     survDiff[i].pos.z - survDiff[j].pos.z };
                const fx ri = pcg::fxmul(g.pruneRadius, survDiff[i].scale);
                const fx rj = pcg::fxmul(g.pruneRadius, survDiff[j].scale);
                if (hf::sim::fpx::FxLength(d) < ri + rj) diffNoOverlap = false;
            }
        check(diffNoOverlap, "PCG5 seed-sensitive: a different seed still yields a valid (no-overlap) field");

        // (5) no-op — g.prune=false -> Generate == BuildInstances over the same scatter; AND pruneRadius<=0 ->
        // all kept (canonical order, but same SET).
        const std::vector<pcg::FxVec3> pts = pcg::ScatterMasked(st, area, cellsX, cellsZ, g.mask, g.density);
        const std::vector<pcg::PcgInstance> directBuild = pcg::BuildInstances(pts, st, g.transform);
        bool noopMatch = (beforePrune.size() == directBuild.size());
        for (size_t i = 0; i < beforePrune.size() && noopMatch; ++i) {
            const pcg::PcgInstance& a = beforePrune[i]; const pcg::PcgInstance& b = directBuild[i];
            if (a.pos.x != b.pos.x || a.pos.z != b.pos.z || a.orient.y != b.orient.y || a.scale != b.scale)
                noopMatch = false;
        }
        check(noopMatch, "PCG5 no-op: g.prune=false -> Generate == BuildInstances over the same scatter");
        const std::vector<pcg::PcgInstance> allKept = pcg::PruneOverlaps(beforePrune, 0);
        check(allKept.size() == beforePrune.size(), "PCG5 no-op: pruneRadius<=0 keeps ALL instances");

        // (6) empty — cellsX<=0 -> empty.
        pcg::PcgGraph gEmpty = g; gEmpty.cellsX = 0;
        check(pcg::Generate(gEmpty, st).empty(), "PCG5 empty: cellsX<=0 -> empty graph");

        // ============= PCG6 render bridge: PcgToRenderInstances provenance + count + no-op =============
        // The render bridge is a PURE FUNCTION of the bit-exact Generate output: one Mat4 per instance,
        // count matches, deterministic across calls, and empty-in -> empty-out (the base-scene no-op).
        const float kBaseRadius = 0.35f;
        const std::vector<math::Mat4> rmats = pcg::PcgToRenderInstances(survivors, kBaseRadius);
        check(rmats.size() == survivors.size(),
              "PCG6 render bridge: one model matrix per instance (count matches Generate)");
        const std::vector<math::Mat4> rmats2 = pcg::PcgToRenderInstances(survivors, kBaseRadius);
        bool rstable = (rmats.size() == rmats2.size());
        for (size_t i = 0; i < rmats.size() && rstable; ++i)
            for (int k = 0; k < 16 && rstable; ++k)
                if (rmats[i].m[k] != rmats2[i].m[k]) rstable = false;
        check(rstable, "PCG6 render bridge: deterministic (two calls byte-identical)");
        check(pcg::PcgToRenderInstances({}, kBaseRadius).empty(),
              "PCG6 render bridge: empty instances -> empty output (base-scene no-op)");
        check(pcg::PcgToRenderInstances(pcg::Generate(gEmpty, st), kBaseRadius).empty(),
              "PCG6 render bridge: empty graph -> zero render instances (no-op)");
    }

    if (g_fail == 0) std::printf("pcg_test: ALL CHECKS PASSED\n");
    else std::printf("pcg_test: %d CHECK(S) FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
