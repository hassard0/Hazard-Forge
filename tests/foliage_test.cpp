// Slice FO1 — Deterministic integer WIND FIELD (engine/foliage/foliage.h), the BEACHHEAD of FLAGSHIP #25
// (DETERMINISTIC FOLIAGE AT SCALE). Pure CPU (header-only, no device, no backend symbols). Namespace
// hf::foliage. WindBend(wind, pos, frame) is a Q16.16 bend angle from a sum of host-baked sine "gust"
// waves over (position, frame#), bit-identical CPU<->Vulkan<->Metal BY CONSTRUCTION — a committed int16
// LUT (kFoliageWind16) indexed by an integer phase accumulator, NO runtime sin/<cmath>.
//
// What this test PINS (the contracts the cross-backend integer golden builds on):
//   * determinism / replay-stable — the SAME (wind, pos, frame) -> the IDENTICAL bend across calls.
//   * zero-amplitude no-op — master=0 (or every amp=0) -> WindBend == 0 for every sampled position.
//   * frame-sensitivity — the field ANIMATES: a different frame changes the bend over a swept patch.
//   * LUT provenance — kFoliageWind16 matches round(32767*sin(2*pi*i/256)) at the cardinal points.
//
// Pure C++ (hf_core), ASan-eligible like the other sim/render-math tests. foliage.h #includes fpx.h
// read-only.
#include "foliage/foliage.h"

#include <cstdint>
#include <cstdio>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace fol = hf::foliage;
using fol::fx;
using fol::kOne;
using fol::FxVec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// A fixed multi-gust field shared by the assertions below.
static fol::WindField MakeField() {
    fol::WindField w;
    w.gustCount = 3;
    w.master    = kOne;
    w.gusts[0]  = fol::Gust{ 0x00030000, 0x00010000, 0x02000000u, kOne / 8 };
    w.gusts[1]  = fol::Gust{ 0x00010000, 0x00040000, 0x01000000u, kOne / 12 };
    w.gusts[2]  = fol::Gust{ 0x00020000, 0x00020000, 0x03000000u, kOne / 16 };
    return w;
}

int main() {
    HF_TEST_MAIN_INIT();

    const fol::WindField w = MakeField();

    // ================= determinism / replay-stable: same inputs -> identical bend across calls ===========
    {
        bool stable = true;
        for (int i = 0; i < 256 && stable; ++i) {
            const FxVec3 p{ (fx)(i * (kOne / 4)), 0, (fx)((255 - i) * (kOne / 4)) };
            const uint32_t frame = (uint32_t)(i * 7 + 3);
            if (fol::WindBend(w, p, frame) != fol::WindBend(w, p, frame)) stable = false;
        }
        check(stable, "FO1 determinism: same (wind,pos,frame) -> identical bend across calls");
    }

    // ================= zero-amplitude no-op: master=0 (and all-amp=0) -> bend == 0 everywhere ============
    {
        fol::WindField zeroMaster = w;
        zeroMaster.master = 0;
        fol::WindField zeroAmp = w;
        for (int g = 0; g < zeroAmp.gustCount; ++g) zeroAmp.gusts[g].amp = 0;

        bool masterZero = true, ampZero = true;
        for (int gx = 0; gx < 64 && (masterZero || ampZero); ++gx) {
            for (int gz = 0; gz < 64; ++gz) {
                const FxVec3 p{ (fx)(gx * kOne), 0, (fx)(gz * kOne) };
                if (fol::WindBend(zeroMaster, p, 17u) != 0) masterZero = false;
                if (fol::WindBend(zeroAmp,    p, 17u) != 0) ampZero    = false;
            }
        }
        check(masterZero, "FO1 zero-amplitude no-op: master=0 -> WindBend == 0 for every sampled position");
        check(ampZero,    "FO1 zero-amplitude no-op: all amp=0 -> WindBend == 0 for every sampled position");
    }

    // ================= frame-sensitivity: the field animates (a different frame changes the bend) ========
    {
        uint32_t differ = 0, total = 0;
        for (int gx = 0; gx < 48; ++gx) {
            for (int gz = 0; gz < 48; ++gz) {
                const FxVec3 p{ (fx)(gx * kOne / 2), 0, (fx)(gz * kOne / 2) };
                if (fol::WindBend(w, p, 0u) != fol::WindBend(w, p, 37u)) ++differ;
                ++total;
            }
        }
        // The wind genuinely advances with the frame — most sampled positions change.
        check(differ > total / 2, "FO1 frame-sensitivity: a different frame changes the bend over the patch");
    }

    // ================= LUT provenance: the cardinal points of round(32767*sin(2*pi*i/256)) ===============
    {
        check(fol::kFoliageWind16[0]   == 0,      "FO1 LUT: sin(0) == 0");
        check(fol::kFoliageWind16[64]  == 32767,  "FO1 LUT: sin(pi/2) == 32767 (peak)");
        check(fol::kFoliageWind16[128] == 0,      "FO1 LUT: sin(pi) == 0");
        check(fol::kFoliageWind16[192] == -32767, "FO1 LUT: sin(3pi/2) == -32767 (trough)");
    }

    if (g_fail == 0) std::printf("foliage_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
