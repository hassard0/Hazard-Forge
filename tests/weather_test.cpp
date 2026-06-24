// Slice WE1 — Deterministic integer DRIFTING CLOUD-DENSITY field (engine/weather/weather.h), the
// BEACHHEAD of FLAGSHIP #27 (DETERMINISTIC DYNAMIC WEATHER). Pure CPU (header-only, no device, no backend
// symbols). Namespace hf::weather. IntCloudDensity(x,z,frame,seed,coverage,octaves) is a Q16.16 density in
// [0,kOne] from a drifted integer fBm value-noise carved by coverage, bit-identical CPU<->Vulkan<->Metal
// BY CONSTRUCTION — a pure integer noise basis + integer wind advection + integer coverage carve, NO
// runtime sin/frac(sin())/<cmath>.
//
// What this test PINS (the contracts the cross-backend integer golden builds on):
//   * replay-stable — the SAME (x,z,frame,seed,coverage,octaves) -> the IDENTICAL density; GenCloudSlice
//     reproducible across calls.
//   * drift — GenCloudSlice(...,frame=F) != GenCloudSlice(...,frame=F+1): the field advected (clouds move).
//   * bounds — every density in [0, kOne].
//   * zero-coverage no-op — coverage == 0 -> every density 0 (clear sky).
//   * coverage monotone — higher coverage -> a larger total density sum (more/denser cloud).
//
// Pure C++ (hf_core), ASan-eligible like the other sim/render-math tests. weather.h #includes sim/fpx.h +
// terrain/procterrain.h read-only.
#include "weather/weather.h"

#include <cstdint>
#include <cstdio>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace wx = hf::weather;
using wx::fx;
using wx::kOne;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();

    const int      kOct      = 5;
    const uint32_t kSeed     = 0xC10D5EEDu;
    const int      kN        = 64;
    const fx       kWorld    = kOne * 48;        // 48 world units across the field
    const uint32_t kFrame    = 30u;
    const fx       kCoverage = (kOne * 2) / 3;   // a real (~0.67) cloud coverage

    // ================= replay-stable: same inputs -> identical density / field across calls ============
    {
        bool stable = true;
        for (int i = 0; i < 256 && stable; ++i) {
            const fx x = static_cast<fx>(i) * (kOne / 4);
            const fx z = static_cast<fx>(255 - i) * (kOne / 4);
            if (wx::IntCloudDensity(x, z, kFrame, kSeed, kCoverage, kOct) !=
                wx::IntCloudDensity(x, z, kFrame, kSeed, kCoverage, kOct)) stable = false;
        }
        check(stable, "WE1 replay-stable: same args -> identical IntCloudDensity across calls");

        const std::vector<fx> f1 = wx::GenCloudSlice(kSeed, kN, kWorld, kFrame, kCoverage, kOct);
        const std::vector<fx> f2 = wx::GenCloudSlice(kSeed, kN, kWorld, kFrame, kCoverage, kOct);
        check(f1.size() == static_cast<size_t>(kN) * kN, "WE1 GenCloudSlice: n*n cells");
        check(f1 == f2, "WE1 replay-stable: GenCloudSlice reproducible (two builds identical)");
    }

    // ================= drift: frame F vs F+1 -> the field advected (the clouds moved) ==================
    {
        const std::vector<fx> a = wx::GenCloudSlice(kSeed, kN, kWorld, kFrame,     kCoverage, kOct);
        const std::vector<fx> b = wx::GenCloudSlice(kSeed, kN, kWorld, kFrame + 1, kCoverage, kOct);
        check(a != b, "WE1 drift: GenCloudSlice(frame=F) != GenCloudSlice(frame=F+1) (clouds drifted)");
    }

    // ================= bounds: every density in [0, kOne] =============================================
    {
        bool inRange = true;
        const std::vector<fx> field = wx::GenCloudSlice(kSeed, kN, kWorld, kFrame, kCoverage, kOct);
        for (fx d : field) { if (d < 0 || d > kOne) { inRange = false; break; } }
        check(inRange, "WE1 bounds: every density in [0, kOne]");
        // A real (coverage>0, octaves>0) field has cloud somewhere (non-flat / coherent).
        fx maxD = 0;
        for (fx d : field) if (d > maxD) maxD = d;
        check(maxD > 0, "WE1 bounds: a real field has cloud somewhere (coherent / non-flat)");
    }

    // ================= zero-coverage no-op: coverage == 0 -> every density 0 ==========================
    {
        // IntCloudDensity directly.
        bool clear = true;
        for (int i = 0; i < 256; ++i) {
            const fx x = static_cast<fx>(i) * (kOne / 4);
            const fx z = static_cast<fx>(i * 3) * (kOne / 4);
            if (wx::IntCloudDensity(x, z, kFrame, kSeed, 0, kOct) != 0) clear = false;
        }
        check(clear, "WE1 zero-coverage no-op: coverage == 0 -> IntCloudDensity == 0 everywhere");
        // GenCloudSlice.
        const std::vector<fx> clearField = wx::GenCloudSlice(kSeed, kN, kWorld, kFrame, 0, kOct);
        bool allZero = true;
        for (fx d : clearField) if (d != 0) allZero = false;
        check(allZero, "WE1 zero-coverage no-op: GenCloudSlice(coverage=0) -> every density 0");
    }

    // ================= coverage monotone: higher coverage -> a larger total density sum ===============
    {
        auto totalDensity = [&](fx coverage) {
            int64_t sum = 0;
            const std::vector<fx> field = wx::GenCloudSlice(kSeed, kN, kWorld, kFrame, coverage, kOct);
            for (fx d : field) sum += static_cast<int64_t>(d);
            return sum;
        };
        const int64_t sLow  = totalDensity(kOne / 4);    // sparse
        const int64_t sMid  = totalDensity(kOne / 2);    // moderate
        const int64_t sHigh = totalDensity(kOne);        // full
        check(sLow <= sMid && sMid <= sHigh,
              "WE1 coverage monotone: higher coverage -> a non-decreasing total density sum");
        check(sHigh > sLow,
              "WE1 coverage monotone: full coverage strictly denser than sparse (a real spread)");
    }

    if (g_fail == 0) std::printf("weather_test: ALL CHECKS PASSED\n");
    return g_fail == 0 ? 0 : 1;
}
