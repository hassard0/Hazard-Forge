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

    if (g_fail == 0) std::printf("pcg_test: ALL CHECKS PASSED\n");
    else std::printf("pcg_test: %d CHECK(S) FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
