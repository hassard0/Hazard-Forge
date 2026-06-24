// Slice PT1 — Integer fBm heightfield generation (engine/terrain/procterrain.h), the BEACHHEAD of
// FLAGSHIP #26 (DETERMINISTIC PROCEDURAL TERRAIN). Pure CPU (header-only, no device, no backend
// symbols). Namespace hf::terrain. IntHeight(x,z,octaves,seed) is a Q16.16 height from a fractal sum
// (fBm) of integer value-noise octaves, bit-identical CPU<->Vulkan<->Metal BY CONSTRUCTION — a pure
// integer hash + smoothstep-bilinear blend, NO runtime sin/sqrt/floor/<cmath>.
//
// What this test PINS (the contracts the cross-backend integer golden builds on):
//   * replay-stable — the SAME (x,z,octaves,seed) -> the IDENTICAL height; GenHeightField reproducible.
//   * seed-sensitive — a different seed -> a different field (the grids differ).
//   * bounds — IntValueNoise in [0, kOne); the fBm height bounded well inside the +-32768 world bound.
//   * zero-octaves flat no-op — octaves <= 0 -> every cell 0.
//   * smooth — adjacent cells differ by a bounded amount (the smoothstep fade => no creased seams).
//
// Pure C++ (hf_core), ASan-eligible like the other sim/render-math tests. procterrain.h #includes
// sim/fpx.h read-only.
#include "terrain/procterrain.h"

#include <cstdint>
#include <cstdio>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace ter = hf::terrain;
using ter::fx;
using ter::kOne;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();

    const int      kOct  = 5;
    const uint32_t kSeed = 0xC0FFEE11u;
    const int      kN    = 64;
    const fx       kWorld = kOne * 32;   // 32 world units across the field

    // ================= replay-stable: same inputs -> identical height / field across calls ==============
    {
        bool stable = true;
        for (int i = 0; i < 256 && stable; ++i) {
            const fx x = static_cast<fx>(i) * (kOne / 4);
            const fx z = static_cast<fx>(255 - i) * (kOne / 4);
            if (ter::IntHeight(x, z, kOct, kSeed) != ter::IntHeight(x, z, kOct, kSeed)) stable = false;
        }
        check(stable, "PT1 replay-stable: same (x,z,octaves,seed) -> identical IntHeight across calls");

        const std::vector<fx> f1 = ter::GenHeightField(kSeed, kN, kWorld, kOct);
        const std::vector<fx> f2 = ter::GenHeightField(kSeed, kN, kWorld, kOct);
        check(f1.size() == static_cast<size_t>(kN) * kN, "PT1 GenHeightField: n*n cells");
        check(f1 == f2, "PT1 replay-stable: GenHeightField reproducible (two builds identical)");
    }

    // ================= seed-sensitive: a different seed -> a different field ============================
    {
        const std::vector<fx> a = ter::GenHeightField(kSeed,         kN, kWorld, kOct);
        const std::vector<fx> b = ter::GenHeightField(kSeed ^ 0x1u,  kN, kWorld, kOct);
        check(a != b, "PT1 seed-sensitive: a different seed -> a different field");
    }

    // ================= bounds: IntValueNoise in [0, kOne); fBm height bounded ===========================
    {
        bool noiseInRange = true;
        for (int i = 0; i < 64 && noiseInRange; ++i) {
            for (int j = 0; j < 64; ++j) {
                const fx x = static_cast<fx>(i) * (kOne / 3) + (kOne / 7);
                const fx z = static_cast<fx>(j) * (kOne / 5) + (kOne / 11);
                const fx v = ter::IntValueNoise(x, z, kSeed);
                if (v < 0 || v >= kOne) { noiseInRange = false; break; }
            }
        }
        check(noiseInRange, "PT1 bounds: IntValueNoise in [0, kOne) for every sampled position");

        // The fBm height is bounded well inside the +-32768 Q16.16 world bound. With base amp kOne/2 and
        // halving amplitude over <=5 octaves and value-noise in [0,kOne), |h| < kOne (== 65536). Assert a
        // sane max (4*kOne is a generous ceiling) so an overflow/runaway is caught.
        fx maxAbs = 0;
        const std::vector<fx> field = ter::GenHeightField(kSeed, kN, kWorld, kOct);
        for (fx h : field) { const fx a = h < 0 ? -h : h; if (a > maxAbs) maxAbs = a; }
        check(maxAbs < 4 * kOne, "PT1 bounds: fBm height bounded well inside the world bound");
        check(maxAbs > 0,        "PT1 bounds: a real field is non-flat (coherent)");
    }

    // ================= zero-octaves flat no-op: octaves <= 0 -> every cell 0 ============================
    {
        bool flat0 = true, flatNeg = true;
        // IntHeight directly.
        for (int i = 0; i < 256; ++i) {
            const fx x = static_cast<fx>(i) * (kOne / 4);
            const fx z = static_cast<fx>(i * 3) * (kOne / 4);
            if (ter::IntHeight(x, z, 0,  kSeed) != 0) flat0 = false;
            if (ter::IntHeight(x, z, -3, kSeed) != 0) flatNeg = false;
        }
        check(flat0,   "PT1 zero-octaves no-op: octaves == 0 -> IntHeight == 0 everywhere");
        check(flatNeg, "PT1 zero-octaves no-op: octaves < 0 -> IntHeight == 0 everywhere");
        // GenHeightField.
        const std::vector<fx> flatField = ter::GenHeightField(kSeed, kN, kWorld, 0);
        bool allZero = true;
        for (fx h : flatField) if (h != 0) allZero = false;
        check(allZero, "PT1 zero-octaves no-op: GenHeightField(octaves=0) -> every cell 0");
    }

    // ================= smooth: adjacent cells differ by a bounded amount (the fade works) ===============
    {
        // Sample the field on a finer grid (more cells per world unit) so adjacent samples are CLOSE in
        // world space and the smoothstep fade keeps them close in height. A loose per-step bound proves
        // there are no creased seams (a discontinuous lattice would jump ~kOne between neighbours).
        const int  fineN  = 256;
        const std::vector<fx> field = ter::GenHeightField(kSeed, fineN, kWorld, kOct);
        const fx kStepBound = kOne / 2;   // a loose bound: neighbours differ by < 0.5
        fx maxStep = 0;
        for (int gz = 0; gz < fineN; ++gz) {
            for (int gx = 0; gx + 1 < fineN; ++gx) {
                const fx a = field[static_cast<size_t>(gz) * fineN + gx];
                const fx b = field[static_cast<size_t>(gz) * fineN + gx + 1];
                const fx d = a > b ? a - b : b - a;
                if (d > maxStep) maxStep = d;
            }
        }
        check(maxStep < kStepBound, "PT1 smooth: adjacent cells differ by a bounded amount (no seams)");
    }

    if (g_fail == 0) std::printf("procterrain_test: ALL CHECKS PASSED\n");
    return g_fail == 0 ? 0 : 1;
}
