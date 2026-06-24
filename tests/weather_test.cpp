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
using wx::FxVec3;  // Slice WE2: the Q16.16 drop position (re-exported from pcg.h via weather.h)

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

    // ===================================================================================================
    // ================= Slice WE2 — Deterministic integer PRECIPITATION (rain/snow) field ===============
    // Each drop's position is a PURE FUNCTION of (seed, frame); the drop falls fallSpeed/frame and WRAPS
    // into [0, columnH) with no accumulator, so two peers see byte-identical rain. PrecipDrop/GenPrecip are
    // pure integer (pcg::PcgRandRange scatter + an int64 positive-modulo fall). Checks: replay-stable; rain
    // falls (each drop's Y at F+1 == wrapped Y(F)-fallSpeed); bounds; the no-op; wrap continuity.
    // ===================================================================================================
    {
        wx::PrecipField p;
        p.seed      = 0xDEADBEEFu;
        p.count     = 1500;
        p.areaW     = kOne * 32;
        p.areaD     = kOne * 32;
        p.columnH   = kOne * 16;
        p.fallSpeed = kOne / 3;            // 0.333.. world units / frame
        const uint32_t F = 40u;

        // (1) replay-stable — same (p, frame) -> identical drops across calls.
        {
            const std::vector<wx::FxVec3> a = wx::GenPrecip(p, F);
            const std::vector<wx::FxVec3> b = wx::GenPrecip(p, F);
            check(a.size() == static_cast<size_t>(p.count), "WE2 GenPrecip: count drops");
            bool same = (a.size() == b.size());
            for (size_t i = 0; same && i < a.size(); ++i)
                if (a[i].x != b[i].x || a[i].y != b[i].y || a[i].z != b[i].z) same = false;
            check(same, "WE2 replay-stable: GenPrecip(p,F) reproducible across calls");
        }

        // (2) rain falls — GenPrecip(p,F) != GenPrecip(p,F+1); each drop's Y at F+1 == wrapped Y(F)-fallSpeed.
        {
            const std::vector<wx::FxVec3> a = wx::GenPrecip(p, F);
            const std::vector<wx::FxVec3> b = wx::GenPrecip(p, F + 1);
            bool differ = (a.size() != b.size());
            const int64_t H = static_cast<int64_t>(p.columnH);
            bool fellCorrectly = (a.size() == b.size());
            for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
                if (a[i].x != b[i].x || a[i].y != b[i].y || a[i].z != b[i].z) differ = true;
                // X/Z are fixed (the scatter does not move).
                if (a[i].x != b[i].x || a[i].z != b[i].z) fellCorrectly = false;
                // Y descended by fallSpeed, modulo-wrapped into [0, columnH).
                const int64_t expected = (((static_cast<int64_t>(a[i].y) - static_cast<int64_t>(p.fallSpeed)) % H) + H) % H;
                if (static_cast<int64_t>(b[i].y) != expected) fellCorrectly = false;
            }
            check(differ, "WE2 rain falls: GenPrecip(p,F) != GenPrecip(p,F+1) (the drops descended)");
            check(fellCorrectly, "WE2 rain falls: each drop Y(F+1) == wrapped (Y(F) - fallSpeed); X/Z fixed");
        }

        // (3) bounds — every drop x in [0,areaW), z in [0,areaD), y in [0,columnH).
        {
            bool inRange = true;
            for (uint32_t f = F; f < F + 8 && inRange; ++f) {
                const std::vector<wx::FxVec3> d = wx::GenPrecip(p, f);
                for (const wx::FxVec3& q : d) {
                    if (q.x < 0 || q.x >= p.areaW || q.z < 0 || q.z >= p.areaD ||
                        q.y < 0 || q.y >= p.columnH) { inRange = false; break; }
                }
            }
            check(inRange, "WE2 bounds: every drop x in [0,areaW), z in [0,areaD), y in [0,columnH)");
        }

        // (4) no-op — count <= 0 -> 0 drops.
        {
            wx::PrecipField z = p;
            z.count = 0;
            check(wx::GenPrecip(z, F).empty(), "WE2 no-op: count == 0 -> 0 drops");
            z.count = -5;
            check(wx::GenPrecip(z, F).empty(), "WE2 no-op: count < 0 -> 0 drops");
        }

        // (5) wrap continuity — a drop near the bottom wraps to near the top across the frame boundary
        //     (no negative Y, no escape past columnH). Find a drop whose Y(F) < fallSpeed (it will wrap).
        {
            const std::vector<wx::FxVec3> a = wx::GenPrecip(p, F);
            const std::vector<wx::FxVec3> b = wx::GenPrecip(p, F + 1);
            bool foundWrap = false, wrapClean = true;
            for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
                if (a[i].y < p.fallSpeed) {                  // near the bottom -> will underflow and wrap up
                    foundWrap = true;
                    // After the wrap, Y must be near the top (>= columnH - fallSpeed) and strictly in [0,columnH).
                    if (b[i].y < 0 || b[i].y >= p.columnH || b[i].y < p.columnH - p.fallSpeed) wrapClean = false;
                }
            }
            check(foundWrap, "WE2 wrap continuity: at least one drop near the bottom (will wrap)");
            check(wrapClean, "WE2 wrap continuity: a wrapping drop re-enters near the top, no negative/escape Y");
        }
    }

    // ===================================================================================================
    // ================= Slice WE3 — Deterministic integer TIME-OF-DAY sun + sky ramp ====================
    // SunSky(frame) returns a Q16.16 sun direction (the sun arcs over the day) + a sky-color ramp
    // (midnight->dawn->noon->dusk), a PURE FUNCTION of the frame, bit-identical cross-backend (the baked
    // sine LUT — NO runtime trig). Checks: replay-stable; the sun MOVES; midnight vs noon distinct (noon
    // higher elevation + the sky colors differ); every sky channel in [0,kOne]; the day is periodic.
    // ===================================================================================================
    {
        // (1) replay-stable — same frame -> identical SunSky across calls.
        {
            bool stable = true;
            for (uint32_t f = 0; f < 256 && stable; ++f) {
                const wx::SunSkyState a = wx::SunSky(f * 5u);
                const wx::SunSkyState b = wx::SunSky(f * 5u);
                if (a.sunDir.x != b.sunDir.x || a.sunDir.y != b.sunDir.y || a.sunDir.z != b.sunDir.z ||
                    a.skyColor.x != b.skyColor.x || a.skyColor.y != b.skyColor.y || a.skyColor.z != b.skyColor.z ||
                    a.sunElev != b.sunElev) stable = false;
            }
            check(stable, "WE3 replay-stable: same frame -> identical SunSky across calls");
        }

        // (2) sun moves — SunSky(F).sunDir != SunSky(F + kDayFrames/8).sunDir (the sun arcs as the day advances).
        {
            const uint32_t F = 100u;
            const wx::SunSkyState a = wx::SunSky(F);
            const wx::SunSkyState b = wx::SunSky(F + wx::kDayFrames / 8u);
            const bool moved = (a.sunDir.x != b.sunDir.x || a.sunDir.y != b.sunDir.y || a.sunDir.z != b.sunDir.z);
            check(moved, "WE3 sun moves: SunSky(F).sunDir != SunSky(F + kDayFrames/8).sunDir (the sun arcs)");
        }

        // (3) midnight vs noon distinct — noon sunElev > midnight sunElev AND the sky colors differ.
        {
            const wx::SunSkyState mid  = wx::SunSky(0);                       // midnight (low/negative elevation)
            const wx::SunSkyState noon = wx::SunSky(wx::kDayFrames / 2u);     // noon (high elevation)
            check(noon.sunElev > mid.sunElev,
                  "WE3 midnight vs noon: noon sunElev > midnight sunElev (the sun arcs up over the day)");
            const bool skyDiffers = (mid.skyColor.x != noon.skyColor.x ||
                                     mid.skyColor.y != noon.skyColor.y ||
                                     mid.skyColor.z != noon.skyColor.z);
            check(skyDiffers, "WE3 midnight vs noon: the midnight vs noon sky colors differ");
        }

        // (4) bounds — every skyColor channel in [0, kOne] across the whole day.
        {
            bool inRange = true;
            for (uint32_t f = 0; f < wx::kDayFrames && inRange; ++f) {
                const wx::SunSkyState s = wx::SunSky(f);
                if (s.skyColor.x < 0 || s.skyColor.x > kOne ||
                    s.skyColor.y < 0 || s.skyColor.y > kOne ||
                    s.skyColor.z < 0 || s.skyColor.z > kOne) inRange = false;
            }
            check(inRange, "WE3 bounds: every skyColor channel in [0, kOne] across the day");
        }

        // (5) periodic — SunSky(frame) == SunSky(frame + kDayFrames) (the day loops).
        {
            bool periodic = true;
            for (uint32_t f = 0; f < 64 && periodic; ++f) {
                const uint32_t frame = f * 17u + 3u;
                const wx::SunSkyState a = wx::SunSky(frame);
                const wx::SunSkyState b = wx::SunSky(frame + wx::kDayFrames);
                if (a.sunDir.x != b.sunDir.x || a.sunDir.y != b.sunDir.y || a.sunDir.z != b.sunDir.z ||
                    a.skyColor.x != b.skyColor.x || a.skyColor.y != b.skyColor.y || a.skyColor.z != b.skyColor.z ||
                    a.sunElev != b.sunElev) periodic = false;
            }
            check(periodic, "WE3 periodic: SunSky(frame) == SunSky(frame + kDayFrames) (the day loops)");
        }
    }

    if (g_fail == 0) std::printf("weather_test: ALL CHECKS PASSED\n");
    return g_fail == 0 ? 0 : 1;
}
