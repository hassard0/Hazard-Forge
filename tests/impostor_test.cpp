// Slice FO1 (impostor arc) — DETERMINISTIC FOLIAGE IMPOSTORS + CROSS-FADE LOD (engine/foliage/impostor.h),
// the last FLAGSHIP #25 audit item (FO-B). Pure CPU (header-only, no device, no backend symbols). Namespace
// hf::foliage::impostor. The deterministic CORE of octahedral impostor addressing + dithered cross-fade LOD:
//
//   OctEncode/Decode — the integer octahedral unit-vector<->square map (Cigolle et al., Y-pole fold);
//   ImpostorCell    — the N×N atlas cell a view direction addresses (nearest cell in oct-space);
//   CrossFadeWeight — the integer 0->kOne cross-fade ramp over a transition band (replacing FO4's hard pop);
//   Bayer-4x4 + per-instance hash dither — the STABLE (shimmer-free) screen-door coverage of the fade;
//   ComputeFadeState / RunImpostorScene — the per-instance fade state over a camera fly-through.
//
// What this test PINS (the contracts the strict-zero cross-backend integer golden builds on):
//   (a) OCT: encode/decode round-trip within a pinned LSB band; cardinal dirs -> pinned cells; a pinned dir
//       set -> pinned N×N cells; the +Y pole -> center cell, the -Y pole -> the corner cell.
//   (b) CROSS-FADE: the weight ramp 0->kOne over the band pinned; outside the band == pure LOD; the dither
//       coverage at weights {0, 1/4, 1/2, 3/4, 1} pinned (fraction of the Bayer pattern that passes).
//   (c) DITHER STABILITY: the same weight -> the same mask every "frame" (no shimmer); the Bayer matrix +
//       its spatial distribution pinned; the per-instance screen-door is stable + ~wFar in aggregate.
//   (d) FADE TRACE: an instance the camera approaches -> the pinned per-tick fade state (far -> cross -> near).
//   (e) SCENE: a foliage field over a camera path -> the impostor-cell assignment + cross-fade full digest.
//
// Pure C++ (hf_core), ASan-eligible. impostor.h #includes foliage.h (which #includes fpx.h) read-only.
#include "foliage/impostor.h"

#include <cstdint>
#include <cstdio>
#include <algorithm>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace imp = hf::foliage::impostor;
using imp::fx;
using imp::kOne;
using imp::FxVec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static fx fabsfx(fx v) { return v < 0 ? -v : v; }

int main() {
    HF_TEST_MAIN_INIT();

    // ============================== (a) OCTAHEDRAL ENCODE / DECODE / CELL ==============================
    {
        // Cardinal cells (gridN=8): +Y pole -> center (4,4)=36; -Y pole -> corner (7,7)=63.
        check(imp::ImpostorCell(FxVec3{ 0,  kOne, 0 }, 8) == 36, "OCT: +Y pole -> center cell 36");
        check(imp::ImpostorCell(FxVec3{ 0, -kOne, 0 }, 8) == 63, "OCT: -Y pole -> corner cell 63");

        // +X and +Z map to opposite edge midpoints of the upper-hemisphere square (u or v == +kOne edge).
        // +X: (u,v)=(kOne,0) -> col 7, row 4 -> 4*8+7 = 39.  +Z: (0,kOne) -> col 4, row 7 -> 7*8+4 = 60.
        check(imp::ImpostorCell(FxVec3{ kOne, 0, 0 }, 8) == 39, "OCT: +X -> edge cell 39");
        check(imp::ImpostorCell(FxVec3{ 0, 0, kOne }, 8) == 60, "OCT: +Z -> edge cell 60");
        check(imp::ImpostorCell(FxVec3{ -kOne, 0, 0 }, 8) == 32, "OCT: -X -> edge cell 32 (col0,row4)");
        check(imp::ImpostorCell(FxVec3{ 0, 0, -kOne }, 8) == 4,  "OCT: -Z -> edge cell 4 (col4,row0)");

        // A pinned oblique direction set -> pinned cells (bit-exact addressing).
        check(imp::ImpostorCell(FxVec3{ kOne, kOne, 0 }, 8) == 38,  "OCT: (+X,+Y) oblique -> cell 38");
        check(imp::ImpostorCell(FxVec3{ kOne, kOne, kOne }, 8) == 45, "OCT: (+X,+Y,+Z) oblique -> cell 45");
        check(imp::ImpostorCell(FxVec3{ -kOne, -kOne, kOne }, 8) == 49, "OCT: (-X,-Y,+Z) lower-hemi -> cell 49");

        // Round-trip: OctDecode(OctEncode(dir)) recovers the UNIT dir within a pinned LSB band. Sweep a set of
        // pinned directions (both hemispheres, off the seams) and take the max per-axis absolute error.
        const FxVec3 dirs[] = {
            { kOne, 0, 0 }, { 0, kOne, 0 }, { 0, 0, kOne },
            { kOne, kOne, 0 }, { kOne, 0, kOne }, { 0, kOne, kOne },
            { kOne, kOne, kOne }, { -kOne, kOne, kOne }, { kOne, -kOne, kOne },
            { -kOne, -kOne, -kOne }, { kOne*2, kOne, -kOne }, { -kOne, kOne*3, kOne*2 },
        };
        fx maxErr = 0;
        for (const FxVec3& d : dirs) {
            // The reference UNIT vector (normalize d with the same fpx floor-sqrt path the decoder uses).
            const fx len = hf::sim::fpx::FxLength(d);
            const FxVec3 unit{ (fx)(((int64_t)d.x * kOne) / len),
                               (fx)(((int64_t)d.y * kOne) / len),
                               (fx)(((int64_t)d.z * kOne) / len) };
            const FxVec3 rt = imp::OctDecode(imp::OctEncode(d));
            maxErr = (fx)std::max<int64_t>(maxErr, fabsfx(rt.x - unit.x));
            maxErr = (fx)std::max<int64_t>(maxErr, fabsfx(rt.y - unit.y));
            maxErr = (fx)std::max<int64_t>(maxErr, fabsfx(rt.z - unit.z));
        }
        // PINNED BAND: the round-trip is within ~1/256 of a unit (two integer divides + a floor-sqrt
        // normalize; NOT bit-exact — the honest octahedral precision bound). kOne/256 == 256.
        check(maxErr < (kOne / 256), "OCT: decode(encode(dir)) round-trip within the pinned LSB band (<kOne/256)");

        // gridN scaling: a finer atlas addresses more cells for the same dir set (monotone resolution).
        check(imp::ImpostorCell(FxVec3{ 0, kOne, 0 }, 16) == 16 * 8 + 8, "OCT: +Y pole -> center cell at gridN=16");
    }

    // ============================== (b) CROSS-FADE WEIGHT + DITHER COVERAGE ==============================
    {
        const fx threshold = kOne * 16;
        const fx band      = kOne * 4;   // band [12, 20]

        // Outside the band -> pure LOD (the FO4 endpoints, now ramp limits).
        imp::FadeWeights below = imp::CrossFadeWeight(kOne * 5,  threshold, band);
        imp::FadeWeights above = imp::CrossFadeWeight(kOne * 30, threshold, band);
        check(below.wNear == kOne && below.wFar == 0, "CROSS-FADE: dist below band -> pure NEAR (wNear=kOne)");
        check(above.wNear == 0 && above.wFar == kOne, "CROSS-FADE: dist above band -> pure FAR (wFar=kOne)");

        // At the band CENTER (dist==threshold) -> half/half (wNear==wFar==kOne/2).
        imp::FadeWeights mid = imp::CrossFadeWeight(threshold, threshold, band);
        check(mid.wNear == kOne / 2 && mid.wFar == kOne / 2, "CROSS-FADE: dist==threshold -> wNear==wFar==kOne/2");

        // Partition of unity + monotone ramp across the band (wFar non-decreasing, wNear+wFar==kOne).
        bool partition = true, monotone = true;
        fx prevFar = -1;
        for (int d = 10; d <= 22; ++d) {
            imp::FadeWeights w = imp::CrossFadeWeight(kOne * d, threshold, band);
            if (w.wNear + w.wFar != kOne) partition = false;
            if (w.wFar < prevFar) monotone = false;
            prevFar = w.wFar;
        }
        check(partition, "CROSS-FADE: wNear + wFar == kOne across the band (partition of unity)");
        check(monotone,  "CROSS-FADE: wFar non-decreasing with distance (monotone ramp)");

        // band<=0 -> the FO4 hard swap (degenerate control).
        imp::FadeWeights hardLo = imp::CrossFadeWeight(kOne * 15, threshold, 0);
        imp::FadeWeights hardHi = imp::CrossFadeWeight(kOne * 17, threshold, 0);
        check(hardLo.wNear == kOne && hardHi.wFar == kOne, "CROSS-FADE: band==0 -> the FO4 hard swap");

        // Dither coverage at weights {0, 1/4, 1/2, 3/4, 1} -> {0,4,8,12,16} of the 16 Bayer cells (pinned).
        check(imp::DitherCoverage(0)          == 0,  "DITHER: wFar=0 -> 0/16 pass (all near)");
        check(imp::DitherCoverage(kOne / 4)   == 4,  "DITHER: wFar=1/4 -> 4/16 pass");
        check(imp::DitherCoverage(kOne / 2)   == 8,  "DITHER: wFar=1/2 -> 8/16 pass");
        check(imp::DitherCoverage(kOne*3 / 4) == 12, "DITHER: wFar=3/4 -> 12/16 pass");
        check(imp::DitherCoverage(kOne)       == 16, "DITHER: wFar=1 -> 16/16 pass (all far)");
    }

    // ============================== (c) DITHER STABILITY + SPATIAL DISTRIBUTION ==========================
    {
        // The Bayer matrix values pinned (the ordered-dither pattern itself).
        static const int kExpectBayer[16] = { 0,8,2,10, 12,4,14,6, 3,11,1,9, 15,7,13,5 };
        bool bayerOk = true;
        for (int i = 0; i < 16; ++i) if (imp::kBayer4[i] != kExpectBayer[i]) bayerOk = false;
        check(bayerOk, "DITHER: the Bayer-4x4 ordered matrix is pinned");

        // STABILITY (no shimmer): the same weight -> the SAME 4x4 pass-mask every "frame". Compute the mask
        // twice (simulating two frames at a fixed weight) and require bit-equality.
        const fx w = kOne * 3 / 8;   // an arbitrary mid weight
        uint32_t maskA = 0, maskB = 0;
        for (int y = 0; y < 4; ++y) for (int x = 0; x < 4; ++x) {
            if (imp::DitherPass(x, y, w)) maskA |= (1u << (y * 4 + x));
        }
        for (int y = 0; y < 4; ++y) for (int x = 0; x < 4; ++x) {
            if (imp::DitherPass(x, y, w)) maskB |= (1u << (y * 4 + x));
        }
        check(maskA == maskB, "DITHER STABILITY: same weight -> identical mask across frames (no shimmer)");
        // The mask at 3/8 -> scale = floor(3/8*16)=6 -> thresholds {0,1,2,3,4,5} pass = 6 cells; pinned count.
        int bits = 0; for (int i = 0; i < 16; ++i) if (maskA & (1u << i)) ++bits;
        check(bits == 6, "DITHER STABILITY: wFar=3/8 -> exactly 6/16 cells pass (pinned)");

        // SPATIAL DISTRIBUTION: the ordered pattern spreads coverage — at 1/2 the passing cells are the low
        // Bayer half {0..7}, an interleaved checkerboard-ish spread (not a contiguous block). Pin the exact set.
        uint32_t halfMask = 0;
        for (int y = 0; y < 4; ++y) for (int x = 0; x < 4; ++x)
            if (imp::DitherPass(x, y, kOne / 2)) halfMask |= (1u << (y * 4 + x));
        // Bayer<8 at positions: idx where kBayer4<8 = {0,2,4,6, ...} -> compute the reference.
        uint32_t refHalf = 0;
        for (int i = 0; i < 16; ++i) if (kExpectBayer[i] < 8) refHalf |= (1u << i);
        check(halfMask == refHalf, "DITHER: the 1/2-coverage spatial mask matches the Bayer low-half set (pinned)");

        // PER-INSTANCE screen-door: stable per instance (same instId+weight -> same decision), and in
        // aggregate ~wFar of the instances show far. Count over a fixed instance range at wFar=1/2.
        bool instStable = true;
        int showFar = 0, total = 0;
        for (uint32_t id = 0; id < 4096; ++id) {
            const bool a = imp::InstanceShowsFar(id, kOne / 2);
            const bool b = imp::InstanceShowsFar(id, kOne / 2);
            if (a != b) instStable = false;
            if (a) ++showFar;
            ++total;
        }
        check(instStable, "DITHER: per-instance screen-door is stable (same id+weight -> same decision)");
        // ~half show far — pin the EXACT count (a deterministic hash => an exact, reproducible tally).
        check(showFar == 2055, "DITHER: per-instance screen-door at wFar=1/2 -> pinned 2055/4096 show far (~half)");
        // Endpoints: wFar=0 -> none show far; wFar=kOne -> all show far.
        int far0 = 0, far1 = 0;
        for (uint32_t id = 0; id < 4096; ++id) {
            if (imp::InstanceShowsFar(id, 0)) ++far0;
            if (imp::InstanceShowsFar(id, kOne)) ++far1;
        }
        check(far0 == 0 && far1 == 4096, "DITHER: per-instance endpoints wFar=0->none, wFar=kOne->all show far");
    }

    // ============================== (d) FADE TRACE: camera approaches an instance ========================
    {
        // A single instance at z=+20; a camera flying from z=-30 to z=+30 (approaching then passing). The XZ
        // distance goes far -> through the band -> near -> and out again; pin the far->cross->near ordering.
        const FxVec3 inst{ 0, 0, kOne * 20 };
        const fx threshold = kOne * 16, band = kOne * 6;   // band [10, 22]
        uint32_t prevMode = imp::kFadeFarOnly;
        bool sawCross = false, sawNear = false, monotoneModeDown = true;
        // As the camera approaches (distance shrinking from >22 to <10), mode must go far(2)->cross(1)->near(0).
        for (int cz = -30; cz <= 20; ++cz) {   // stop at the instance (closest approach ~ z=20)
            const FxVec3 cam{ 0, kOne * 2, (fx)(kOne * cz) };
            const imp::FadeState st = imp::ComputeFadeState(inst, cam, threshold, band, 8, 7u);
            if (st.mode > prevMode) monotoneModeDown = false;   // mode never increases as we approach
            if (st.mode == imp::kFadeCross)    sawCross = true;
            if (st.mode == imp::kFadeNearOnly) sawNear  = true;
            prevMode = st.mode;
        }
        check(monotoneModeDown, "FADE TRACE: mode never increases as the camera approaches (far->cross->near)");
        check(sawCross, "FADE TRACE: the instance passes THROUGH the cross-fade band (no pop)");
        check(sawNear,  "FADE TRACE: the instance reaches near-only when close");

        // Pin the exact per-tick mode sequence at three sampled camera distances.
        auto modeAt = [&](int cz) {
            const FxVec3 cam{ 0, kOne * 2, (fx)(kOne * cz) };
            return imp::ComputeFadeState(inst, cam, threshold, band, 8, 7u).mode;
        };
        check(modeAt(-30) == imp::kFadeFarOnly, "FADE TRACE: cam z=-30 (dist 50) -> far-only");
        check(modeAt(2)   == imp::kFadeCross,   "FADE TRACE: cam z=2 (dist 18, in band) -> cross-fade");
        check(modeAt(19)  == imp::kFadeNearOnly,"FADE TRACE: cam z=19 (dist 1) -> near-only");
    }

    // ============================== (e) SCENE: field + camera fly-through digest =========================
    {
        // Determinism: two runs of the fixed scene -> identical digest (the strict-zero cross-backend golden).
        const imp::ImpostorSceneRun r1 = imp::RunImpostorScene();
        const imp::ImpostorSceneRun r2 = imp::RunImpostorScene();
        check(r1.digest == r2.digest, "SCENE determinism: two runs -> identical fade/addressing digest");

        // Coherence: the field spans all three modes at the reference tick (near, cross, far all present).
        check(r1.nearOnly > 0 && r1.inBand > 0 && r1.farOnly > 0,
              "SCENE coherence: the reference frame has near + cross-fade + impostor plants");
        // Every plant accounted for.
        check(r1.nearOnly + r1.inBand + r1.farOnly == r1.instances, "SCENE: mode counts partition the field");
        // Impostor cells addressed (far/cross plants) is >0 and <= gridN*gridN.
        check(r1.impostorCells > 0 && r1.impostorCells <= r1.gridN * r1.gridN,
              "SCENE: distinct impostor cells addressed within [1, gridN*gridN]");

        // The traced instance (near the +Z field edge) is APPROACHED and PASSED by the fly-through camera: it
        // starts far, dips through cross into near as the camera arrives, then back out to far — the "no pop"
        // proof at the scene level. One entry per tick; starts + ends far; reaches near AND cross en route.
        check((int)r1.trace.size() == r1.ticks, "SCENE trace: one entry per tick");
        check(r1.trace.front().mode == imp::kFadeFarOnly, "SCENE trace: starts far-only (camera distant)");
        check(r1.trace.back().mode  == imp::kFadeFarOnly, "SCENE trace: ends far-only (camera flew past)");
        bool traceNear = false, traceCross = false;
        for (const imp::ImpTraceEntry& e : r1.trace) {
            if (e.mode == imp::kFadeNearOnly) traceNear = true;
            if (e.mode == imp::kFadeCross)    traceCross = true;
        }
        check(traceNear && traceCross, "SCENE trace: reaches near-only AND passes through cross-fade (no pop)");

        // PIN the scene digest + the reference-frame stat line (the strict-zero integer golden fingerprint).
        std::printf("impostor SCENE: instances=%d gridN=%d ticks=%d refTick=%d near=%d inBand=%d far=%d cells=%d digest=0x%016llx\n",
                    r1.instances, r1.gridN, r1.ticks, r1.refTick, r1.nearOnly, r1.inBand, r1.farOnly, r1.impostorCells,
                    (unsigned long long)r1.digest);
        check(r1.instances == 144 && r1.gridN == 8 && r1.ticks == 24 && r1.refTick == 12,
              "SCENE: pinned field/grid/ticks (144 plants, 8x8 atlas, 24 ticks, ref tick 12)");
        check(r1.nearOnly == 36 && r1.inBand == 100 && r1.farOnly == 8 && r1.impostorCells == 28,
              "SCENE: pinned reference-frame mix (near=36, cross=100, far=8, cells=28)");
        check(r1.digest == 0x104198cb9fd5dfafull, "SCENE: pinned strict-zero fade/addressing digest");
    }

    if (g_fail == 0) std::printf("impostor_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
