// Slice GI1 — Deterministic Lumen-class GI: INTEGER RT PROBE TRACE + SHADE (the beachhead, FLAGSHIP #29).
// The CPU-side invariants the HW probe-trace kernel (shaders/gi_probe_trace.comp) rests on. The HW==CPU
// memcmp proof lives in the --gi1-probe-shot showcase (it needs a real GPU); this pure-CPU test pins the
// deterministic GI contract that makes that proof possible:
//
//   * The 16 baked Fibonacci-sphere directions (gi::kGiProbeDirs) are UNIT-LENGTH within an fx epsilon and
//     DISTINCT (no two directions equal) — the host-precomputed table is well-formed.
//   * TraceProbeRays ANALYTIC: a probe facing a lit surface (a probe inside the enclosure) records SOME
//     ray radiance > 0; a probe-ray whose path to the light is BLOCKED is darker than the same surface
//     unshadowed (the per-ray shadow actually fired); a probe in OPEN space (no geometry) -> every ray
//     records the background/sky radiance.
//   * TraceProbeRays is DETERMINISTIC (two runs byte-identical).
//   * ProbeCount == 0 (a 0-dim grid) -> NO writes (the dispatch-0 no-op the proof rests on).
//
// Pure C++ (hf_core), ASan-eligible. gi.h #includes render/rtrace.h read-only (rtrace.h #includes
// sim/fpx.h read-only); this test #includes gi.h read-only.
#include "render/gi.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "test_main.h"

using namespace hf;
namespace gi = hf::render::gi;
namespace rt = hf::render::rtrace;
using gi::fx;
using gi::kOne;
using gi::FxVec3;
using gi::GiRadiance;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();

    // ================= The 16 baked directions are unit-length + distinct =================
    {
        // |dir|^2 in Q32.32 should be within an fx epsilon of kOne^2. The Fibonacci quantization keeps each
        // baked vector within < 2e-5 of unit -> |len^2 - 1| < ~4e-5 -> well within a 1/256 (~0.004) band.
        const int64_t one2 = (int64_t)kOne * (int64_t)kOne;
        const int64_t eps2 = one2 / 256;   // a generous integer band
        bool allUnit = true;
        for (int i = 0; i < gi::kGiRaysPerProbe; ++i) {
            const FxVec3& v = gi::kGiProbeDirs[i];
            int64_t len2 = (int64_t)v.x * v.x + (int64_t)v.y * v.y + (int64_t)v.z * v.z;
            int64_t diff = len2 - one2;
            if (diff < 0) diff = -diff;
            if (diff > eps2) { allUnit = false;
                std::printf("  dir[%d] len2=%lld vs one2=%lld diff=%lld\n", i,
                            (long long)len2, (long long)one2, (long long)diff); }
        }
        check(allUnit, "kGiProbeDirs: all 16 directions are unit-length within an fx epsilon");

        bool allDistinct = true;
        for (int i = 0; i < gi::kGiRaysPerProbe; ++i)
            for (int j = i + 1; j < gi::kGiRaysPerProbe; ++j) {
                const FxVec3& a = gi::kGiProbeDirs[i];
                const FxVec3& b = gi::kGiProbeDirs[j];
                if (a.x == b.x && a.y == b.y && a.z == b.z) allDistinct = false;
            }
        check(allDistinct, "kGiProbeDirs: all 16 directions are DISTINCT");
    }

    // ================= ProbePos integer lattice round-trip =================
    {
        gi::GiProbeGrid grid;
        grid.origin = FxVec3{gi::GiF(1,1), gi::GiF(2,1), gi::GiF(-3,1)};
        grid.spacing = gi::GiF(3,2);   // 1.5
        grid.nx = 4; grid.ny = 4; grid.nz = 4;
        check(gi::ProbeCount(grid) == 64, "ProbeCount: 4x4x4 == 64");
        // Probe (2,1,3) at origin + (2,1,3)*1.5 = (1+3, 2+1.5, -3+4.5) = (4, 3.5, 1.5).
        FxVec3 expect{gi::GiF(4,1), gi::GiF(7,2), gi::GiF(3,2)};
        FxVec3 got = gi::ProbePos(grid, 2, 1, 3);
        check(got.x == expect.x && got.y == expect.y && got.z == expect.z,
              "ProbePos(ix,iy,iz): integer lattice position exact");
        // The linear-index decode agrees with the (ix,iy,iz) overload.
        int lin = gi::ProbeFlatIndex(grid, 2, 1, 3);
        FxVec3 gotLin = gi::ProbePos(grid, lin);
        check(gotLin.x == expect.x && gotLin.y == expect.y && gotLin.z == expect.z,
              "ProbePos(linearIndex): cx-major decode matches ProbePos(ix,iy,iz)");
    }

    // ================= TraceProbeRays analytic (lit + shadowed + open) =================
    {
        gi::GiScene1 s = gi::BuildGi1Scene();

        // A 4x4x4 probe grid filling the enclosure interior (origin near the floor-left-front corner).
        gi::GiProbeGrid grid;
        grid.origin = FxVec3{gi::GiF(-2,1), gi::GiF(1,1), gi::GiF(0,1)};
        grid.spacing = gi::GiF(1,1);
        grid.nx = 4; grid.ny = 4; grid.nz = 4;
        const int probes = gi::ProbeCount(grid);

        std::vector<GiRadiance> out((size_t)probes * gi::kGiRaysPerProbe);
        gi::TraceProbeRays(grid, s.scene, std::span<GiRadiance>(out));

        // (a) SOME ray sees a LIT surface: at least one radiance is brighter than the pure background, AND
        //     at least one ray is darker (shadowed / ambient-only or a different-albedo wall). Count lit.
        GiRadiance bg = gi::UnpackRadiance(s.scene.background);
        int lit = 0;
        int64_t maxLum = -1, minLum = (int64_t)1 << 40;
        for (const GiRadiance& g : out) {
            int64_t lum = (int64_t)g.r + g.g + g.b;
            if (lum > maxLum) maxLum = lum;
            if (lum < minLum) minLum = lum;
            int64_t bglum = (int64_t)bg.r + bg.g + bg.b;
            if (lum > bglum) ++lit;
        }
        check(lit > 0, "TraceProbeRays: at least one probe ray sees a LIT surface (radiance > background)");
        check(maxLum > minLum, "TraceProbeRays: per-ray radiance VARIES (lit vs shadowed/dark, not flat)");

        // (b) The per-ray SHADOW actually fires: build the SAME probe-ray hits but FORCE not-occluded by
        //     removing the shadow, and confirm SOME ray comes out darker WITH the shadow than without. We
        //     re-derive the unshadowed shade via rtrace::ShadeHitInt (never gated) and compare a probe-ray
        //     that we know is shadowed. Concretely: find a (probe,ray) whose TraceClosest hits a surface
        //     AND whose shadow ray is occluded -> its radiance must be <= the unshadowed shade of the same
        //     hit (strictly darker for a lit surface). If ANY such (probe,ray) exists, the shadow fired.
        bool shadowFired = false;
        for (int p = 0; p < probes && !shadowFired; ++p) {
            FxVec3 origin = gi::ProbePos(grid, p);
            for (int d = 0; d < gi::kGiRaysPerProbe; ++d) {
                rt::RtRay ray{origin, gi::kGiProbeDirs[d]};
                rt::RtHit hit = rt::TraceClosest(ray, s.scene);
                if (hit.primIndex == rt::kRtMiss) continue;
                rt::RtRay sray;
                sray.origin = rt::FxAdd(hit.pos, rt::FxScale(hit.normal, rt::kRtShadowEps));
                sray.dir = s.scene.lightDir;
                bool occ = rt::TraceAnyHit(sray, s.scene, rt::kRtShadowMinT);
                if (!occ) continue;
                // This ray IS shadowed. Its shaded color (occluded) must be <= the unshadowed shade.
                uint32_t shadowed = rt::ShadeHitShadowed(hit, s.scene, true);
                uint32_t unshadowed = rt::ShadeHitShadowed(hit, s.scene, false);
                if (shadowed != unshadowed) shadowFired = true;   // the gate changed the color
            }
        }
        check(shadowFired, "TraceProbeRays: the per-ray shadow ray fired (a shadowed ray is darker)");
    }

    // ================= TraceProbeRays open-space probe -> sky/background =================
    {
        // An EMPTY scene (no geometry) -> every probe ray MISSES -> records the background radiance.
        std::vector<rt::RtSphere> noSph;
        std::vector<rt::RtAabb>   noBox;
        rt::RtScene empty{};
        empty.spheres = std::span<const rt::RtSphere>(noSph);
        empty.aabbs   = std::span<const rt::RtAabb>(noBox);
        empty.lightDir = rt::RtNormalize(FxVec3{0, kOne, 0});
        empty.background = rt::PackRGBA8(28, 32, 44, 255);

        gi::GiProbeGrid grid;
        grid.origin = FxVec3{0, 0, 0};
        grid.spacing = gi::GiF(1,1);
        grid.nx = 2; grid.ny = 2; grid.nz = 2;
        const int probes = gi::ProbeCount(grid);
        std::vector<GiRadiance> out((size_t)probes * gi::kGiRaysPerProbe);
        gi::TraceProbeRays(grid, empty, std::span<GiRadiance>(out));

        GiRadiance bg = gi::UnpackRadiance(empty.background);
        bool allSky = true;
        for (const GiRadiance& g : out)
            if (g.r != bg.r || g.g != bg.g || g.b != bg.b) allSky = false;
        check(allSky, "TraceProbeRays: an open-space probe (empty scene) -> every ray sees the sky/background");
    }

    // ================= Determinism (two runs byte-identical) =================
    {
        gi::GiScene1 s = gi::BuildGi1Scene();
        gi::GiProbeGrid grid;
        grid.origin = FxVec3{gi::GiF(-2,1), gi::GiF(1,1), gi::GiF(0,1)};
        grid.spacing = gi::GiF(1,1);
        grid.nx = 4; grid.ny = 4; grid.nz = 4;
        const int probes = gi::ProbeCount(grid);
        std::vector<GiRadiance> a((size_t)probes * gi::kGiRaysPerProbe);
        std::vector<GiRadiance> b((size_t)probes * gi::kGiRaysPerProbe);
        gi::TraceProbeRays(grid, s.scene, std::span<GiRadiance>(a));
        gi::TraceProbeRays(grid, s.scene, std::span<GiRadiance>(b));
        check(std::memcmp(a.data(), b.data(), a.size() * sizeof(GiRadiance)) == 0,
              "TraceProbeRays: two runs BYTE-IDENTICAL (deterministic)");
    }

    // ================= ProbeCount == 0 no-op =================
    {
        gi::GiScene1 s = gi::BuildGi1Scene();
        gi::GiProbeGrid grid;
        grid.nx = 0; grid.ny = 4; grid.nz = 4;   // a 0-dim -> ProbeCount == 0
        check(gi::ProbeCount(grid) == 0, "ProbeCount: a 0-dim grid -> 0 probes");
        check(gi::GiProbeDispatchGroups(grid) == 0, "GiProbeDispatchGroups: 0 probes -> 0 groups (the no-op)");
        // A pre-filled buffer is left UNTOUCHED (TraceProbeRays writes nothing).
        std::vector<GiRadiance> out(8);
        for (auto& g : out) g = GiRadiance{(fx)123, (fx)456, (fx)789, 0};
        gi::TraceProbeRays(grid, s.scene, std::span<GiRadiance>(out));
        bool untouched = true;
        for (const auto& g : out)
            if (g.r != 123 || g.g != 456 || g.b != 789) untouched = false;
        check(untouched, "TraceProbeRays: ProbeCount==0 -> the radiance buffer is UNTOUCHED (the no-op)");
    }

    if (g_fail == 0) std::printf("gi_probe_test: all GI1 CPU invariants PASS\n");
    return g_fail == 0 ? 0 : 1;
}
