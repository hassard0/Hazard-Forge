// Slice RT3 — Hardware Ray Tracing: the CPU-side invariants the HW hard-shadow kernel
// (shaders/rt_shadow.comp) rests on. The HW==CPU memcmp proof lives in the --rt3-shadow-shot showcase (it
// needs a real GPU); this pure-CPU test pins the deterministic shadow contract that makes that proof
// possible:
//
//   * TraceAnyHit ANALYTIC: a point directly under a sphere, shadow-ray toward an OVERHEAD light, is
//     OCCLUDED; an OPEN point (no occluder along the ray) is NOT occluded; and the boolean is
//     ORDER-INDEPENDENT — shuffling the scene's primitive storage yields the IDENTICAL boolean (the
//     determinism-of-any-hit guarantee — the HW path may early-out, the SW may scan all, same result).
//   * ShadeHitShadowed == ShadeHitInt when NOT occluded (the gate is transparent in light); and strictly
//     DARKER (ambient-only) when occluded — for a lit surface.
//   * RenderSceneShadowed is DETERMINISTIC (two runs byte-identical).
//   * 0 < shadowed < hits on the RT2 scene (the shadow ray actually darkened SOME but not ALL lit surface
//     — a real-occlusion sanity, not just determinism).
//
// Pure C++ (hf_core), ASan-eligible. rtrace.h #includes sim/fpx.h read-only; rtrace.h is BYTE-FROZEN (RT3
// is APPEND-ONLY); this test #includes it read-only and defines the RT2 scene locally (the showcase TU).
#include "render/rtrace.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "test_main.h"

using namespace hf;
namespace rt = hf::render::rtrace;
using rt::fx;
using rt::kOne;
using rt::FxVec3;
using rt::F;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Build the SAME RT2 scene the showcase defines (a ground AABB + a 4x4 sphere grid + two boxes).
struct Rt2Scene {
    std::vector<rt::RtSphere> spheres;
    std::vector<rt::RtAabb>   aabbs;
    rt::RtScene  scene;
    rt::RtCamera camera;
};

static Rt2Scene BuildRt2Scene() {
    Rt2Scene r;
    uint32_t nextPrim = 0;
    r.aabbs.push_back(rt::RtAabb{FxVec3{F(-20,1), F(-3,1), F(-20,1)},
                                 FxVec3{F(20,1),  F(-1,1), F(20,1)}, nextPrim++});
    for (int gz = 0; gz < 4; ++gz) {
        for (int gx = 0; gx < 4; ++gx) {
            fx cx = F(2 * gx - 3, 1);
            fx cz = F(2 + 2 * gz, 1);
            fx cy = F(0, 1);
            fx rad = (gz & 1) ? F(3, 4) : F(1, 1);
            r.spheres.push_back(rt::RtSphere{FxVec3{cx, cy, cz}, rad, nextPrim++});
        }
    }
    r.aabbs.push_back(rt::RtAabb{FxVec3{F(-9,2), F(-1,1), F(1,1)},
                                 FxVec3{F(-5,2), F(3,2),  F(5,2)}, nextPrim++});
    r.aabbs.push_back(rt::RtAabb{FxVec3{F(5,2),  F(-1,1), F(2,1)},
                                 FxVec3{F(9,2),  F(2,1),  F(7,2)}, nextPrim++});
    r.scene.spheres = std::span<const rt::RtSphere>(r.spheres);
    r.scene.aabbs   = std::span<const rt::RtAabb>(r.aabbs);
    r.scene.lightDir = rt::RtNormalize(FxVec3{F(4,10), F(8,10), F(-3,10)});
    r.scene.background = rt::PackRGBA8(34, 40, 56, 255);
    r.camera.eye     = FxVec3{F(0,1), F(2,1), F(-9,1)};
    r.camera.right   = FxVec3{kOne, 0, 0};
    r.camera.up      = FxVec3{0, kOne, 0};
    r.camera.forward = FxVec3{0, 0, kOne};
    r.camera.halfW   = F(7, 10);
    r.camera.halfH   = F(7, 10);
    return r;
}

int main() {
    HF_TEST_MAIN_INIT();

    // ================= TraceAnyHit analytic (occluded vs open + order-independence) =================
    {
        // A tiny analytic scene: ONE sphere of radius 1 centered at the origin, plus a directional light
        // straight up (+Y). A ground point just below the sphere (y = -2) fires a shadow ray toward +Y ->
        // it must pass through the sphere -> OCCLUDED. A point far to the side (x = 10, y = -2) fires the
        // same +Y ray -> nothing above it -> NOT occluded.
        std::vector<rt::RtSphere> sp;
        sp.push_back(rt::RtSphere{FxVec3{0, 0, 0}, kOne, /*primIndex*/ 7});
        std::vector<rt::RtAabb> bx;  // none
        rt::RtScene s{};
        s.spheres = std::span<const rt::RtSphere>(sp);
        s.aabbs   = std::span<const rt::RtAabb>(bx);
        s.lightDir = FxVec3{0, kOne, 0};   // straight up
        s.background = 0;

        rt::RtRay underRay;   // from just below the sphere, toward +Y
        underRay.origin = FxVec3{0, F(-2,1), 0};
        underRay.dir    = FxVec3{0, kOne, 0};
        bool underOcc = rt::TraceAnyHit(underRay, s, rt::kRtShadowMinT);
        check(underOcc, "TraceAnyHit: a point under the sphere toward an overhead light is OCCLUDED");

        rt::RtRay openRay;    // far to the side; nothing above
        openRay.origin = FxVec3{F(10,1), F(-2,1), 0};
        openRay.dir    = FxVec3{0, kOne, 0};
        bool openOcc = rt::TraceAnyHit(openRay, s, rt::kRtShadowMinT);
        check(!openOcc, "TraceAnyHit: an open point (no occluder) is NOT occluded");

        // Order-independence: a multi-primitive scene where SHUFFLING the storage order leaves the boolean
        // unchanged (the any-hit OR is order-invariant — the HW early-out and the SW full-scan agree).
        std::vector<rt::RtSphere> shA, shB;
        shA.push_back(rt::RtSphere{FxVec3{0, F(3,1), 0}, kOne, 1});  // occluder above
        shA.push_back(rt::RtSphere{FxVec3{F(20,1), 0, 0}, kOne, 2}); // irrelevant
        shA.push_back(rt::RtSphere{FxVec3{0, F(6,1), 0}, kOne, 3});  // another occluder above
        shB.push_back(shA[2]); shB.push_back(shA[0]); shB.push_back(shA[1]);  // shuffled
        rt::RtScene sa{}; sa.spheres = std::span<const rt::RtSphere>(shA);
        sa.aabbs = std::span<const rt::RtAabb>(bx); sa.lightDir = s.lightDir;
        rt::RtScene sb{}; sb.spheres = std::span<const rt::RtSphere>(shB);
        sb.aabbs = std::span<const rt::RtAabb>(bx); sb.lightDir = s.lightDir;
        rt::RtRay upRay; upRay.origin = FxVec3{0, 0, 0}; upRay.dir = FxVec3{0, kOne, 0};
        bool occA = rt::TraceAnyHit(upRay, sa, rt::kRtShadowMinT);
        bool occB = rt::TraceAnyHit(upRay, sb, rt::kRtShadowMinT);
        check(occA && occB, "TraceAnyHit: an occluded ray is occluded in BOTH storage orders");
        check(occA == occB, "TraceAnyHit: the boolean is ORDER-INDEPENDENT (shuffled prims agree)");
    }

    // ================= ShadeHitShadowed vs ShadeHitInt (gate transparent in light, dark in shadow) =====
    {
        rt::RtScene s{};
        std::vector<rt::RtSphere> sp; std::vector<rt::RtAabb> bx;
        s.spheres = std::span<const rt::RtSphere>(sp);
        s.aabbs   = std::span<const rt::RtAabb>(bx);
        s.lightDir = FxVec3{0, kOne, 0};
        s.background = rt::PackRGBA8(34, 40, 56, 255);

        // A lit hit: normal toward the light (max dot) -> a non-trivial diffuse.
        rt::RtHit hit{};
        hit.t = kOne; hit.primIndex = 0; hit.pos = FxVec3{0, 0, 0};
        hit.normal = FxVec3{0, kOne, 0};   // toward +Y == lightDir -> dot = 1

        uint32_t shadeInt  = rt::ShadeHitInt(hit, s);
        uint32_t shadeLit  = rt::ShadeHitShadowed(hit, s, /*occluded*/ false);
        uint32_t shadeDark = rt::ShadeHitShadowed(hit, s, /*occluded*/ true);
        check(shadeLit == shadeInt,
              "ShadeHitShadowed (not occluded) == ShadeHitInt (the gate is transparent in light)");
        // Strictly darker when occluded: each channel <= the lit channel, and at least one strictly less.
        auto ch = [](uint32_t c, int i) { return (int)((c >> (8 * i)) & 0xFF); };
        bool allLE = true, someLess = false;
        for (int i = 0; i < 3; ++i) {
            if (ch(shadeDark, i) > ch(shadeLit, i)) allLE = false;
            if (ch(shadeDark, i) < ch(shadeLit, i)) someLess = true;
        }
        check(allLE && someLess, "ShadeHitShadowed (occluded) is strictly DARKER than lit (ambient-only)");

        // A MISS shades to background regardless of occlusion (defensive).
        rt::RtHit miss = rt::RtMissHit();
        check(rt::ShadeHitShadowed(miss, s, true) == s.background &&
              rt::ShadeHitShadowed(miss, s, false) == s.background,
              "ShadeHitShadowed: a miss -> background (occlusion irrelevant)");
    }

    // ================= RenderSceneShadowed determinism + a real shadow count on the RT2 scene ==========
    {
        Rt2Scene s = BuildRt2Scene();
        const uint32_t W = 320, H = 240;
        const size_t kPixels = (size_t)W * H;

        std::vector<uint32_t> a(kPixels, 0), b(kPixels, 0);
        uint32_t shadA = rt::RenderSceneShadowed(s.scene, s.camera, W, H, std::span<uint32_t>(a));
        uint32_t shadB = rt::RenderSceneShadowed(s.scene, s.camera, W, H, std::span<uint32_t>(b));
        check(shadA == shadB, "RenderSceneShadowed: two-run shadowed count matches");
        check(std::memcmp(a.data(), b.data(), kPixels * sizeof(uint32_t)) == 0,
              "RenderSceneShadowed: two runs BYTE-IDENTICAL (deterministic)");

        // Primary-hit count (the lit denominator).
        uint32_t hits = 0;
        for (uint32_t py = 0; py < H; ++py) {
            for (uint32_t px = 0; px < W; ++px) {
                rt::RtRay ray = rt::PrimaryRay(s.camera, px, py, W, H);
                if (rt::TraceClosest(ray, s.scene).primIndex != rt::kRtMiss) ++hits;
            }
        }
        check(shadA > 0, "RenderSceneShadowed: SOME surface is shadowed (real occlusion)");
        check(shadA < hits, "RenderSceneShadowed: NOT every lit surface is shadowed (0 < shadowed < hits)");
        if (!(shadA > 0 && shadA < hits))
            std::printf("  (shadowed:%u hits:%u — expected 0 < shadowed < hits)\n", shadA, hits);
    }

    if (g_fail == 0) std::printf("rt_shadow_test: all RT3 CPU invariants PASS\n");
    return g_fail == 0 ? 0 : 1;
}
