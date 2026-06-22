// Slice RT6 — Hardware Ray Tracing: the CPU-side invariants the HW lit-hero kernel (shaders/rt_hero.comp)
// rests on. The HW==CPU memcmp proof lives in the --rt6-hero-shot showcase (it needs a real GPU); this
// pure-CPU test pins the deterministic hero contract that makes that proof possible:
//
//   * SkyGradient MONOTONIC in dir.y: a ray pointing straight UP (zenith) and one pointing straight DOWN
//     (horizon) yield DISTINCT colors, and the gradient is monotone in dir.y per channel (the integer lerp
//     between the two pinned endpoints). The straight-up color == the pinned zenith; straight-down == the
//     pinned horizon. Two distinct directions -> (in general) distinct colors.
//   * RenderSceneHero == RenderSceneReflected when NO ray ever misses (the sky is never sampled): if every
//     primary AND reflection ray hits a primitive, the hero render is BYTE-IDENTICAL to the RT4 reflected
//     render (RenderSceneHero only changes the MISS color). We construct an enclosed scene (a big box "room")
//     so no ray escapes, and assert byte-equality.
//   * The HERO scene (the SAME one --rt6-hero-shot builds) has shadowed>0 AND reflective>0 (the full feature
//     set contributes) AND the sky is GRADED (>=2 distinct colors among the primary-miss pixels).
//   * RenderSceneHero is DETERMINISTIC (two runs byte-identical, both image and counts).
//
// Pure C++ (hf_core), ASan-eligible. rtrace.h #includes sim/fpx.h read-only; rtrace.h's RT1+RT3+RT4 code is
// BYTE-FROZEN (RT6 is APPEND-ONLY); this test #includes it read-only and defines the hero scene locally.
#include "render/rtrace.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "test_main.h"

using namespace hf;
namespace rt = hf::render::rtrace;
using rt::fx;
using rt::kOne;
using rt::kFrac;
using rt::FxVec3;
using rt::F;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

static int chOf(uint32_t c, int i) { return (int)((c >> (8 * i)) & 0xFF); }

// Build the SAME curated hero scene the --rt6-hero-shot / --rt6-hero showcases define (a reflective ground
// AABB primIndex 0 + an arc of 7 spheres primIndex 1..7 with the mirror sphere at primIndex
// kRtMirrorSpherePrim == 6 + two flanking boxes primIndex 8,9), at the SAME 3/4 hero camera.
struct HeroScene {
    std::vector<rt::RtSphere> spheres;
    std::vector<rt::RtAabb>   aabbs;
    rt::RtScene  scene;
    rt::RtCamera camera;
};

static HeroScene BuildHeroScene() {
    HeroScene r;
    uint32_t nextPrim = 0;
    r.aabbs.push_back(rt::RtAabb{FxVec3{F(-20,1), F(-3,1), F(-20,1)},
                                 FxVec3{F(20,1),  F(-1,1), F(20,1)}, nextPrim++});
    struct SphSpec { int cxN, cxD, czN, czD, rN, rD; };
    const SphSpec arc[7] = {
        { -7,2,  5,1,  1,1 },   // primIndex 1
        { -9,4,  7,2,  3,4 },   // primIndex 2
        { -3,2,  4,1,  1,1 },   // primIndex 3
        {  0,1,  9,2,  5,4 },   // primIndex 4
        {  3,2,  4,1,  1,1 },   // primIndex 5
        {  9,4,  7,2,  1,1 },   // primIndex 6 : THE MIRROR SPHERE (kRtMirrorSpherePrim)
        {  7,2,  5,1,  3,4 },   // primIndex 7
    };
    for (int i = 0; i < 7; ++i) {
        fx cx = F(arc[i].cxN, arc[i].cxD);
        fx cz = F(arc[i].czN, arc[i].czD);
        fx rad = F(arc[i].rN, arc[i].rD);
        fx cy = (fx)(F(-1,1) + rad);
        r.spheres.push_back(rt::RtSphere{FxVec3{cx, cy, cz}, rad, nextPrim++});
    }
    r.aabbs.push_back(rt::RtAabb{FxVec3{F(-11,1), F(-1,1), F(3,1)},
                                 FxVec3{F(-9,1),  F(2,1),  F(5,1)}, nextPrim++});
    r.aabbs.push_back(rt::RtAabb{FxVec3{F(9,1),   F(-1,1), F(3,1)},
                                 FxVec3{F(11,1),  F(2,1),  F(5,1)}, nextPrim++});
    r.scene.spheres = std::span<const rt::RtSphere>(r.spheres);
    r.scene.aabbs   = std::span<const rt::RtAabb>(r.aabbs);
    r.scene.lightDir = rt::RtNormalize(FxVec3{F(4,10), F(8,10), F(-3,10)});
    r.scene.background = rt::PackRGBA8(34, 40, 56, 255);
    r.camera.eye     = FxVec3{F(0,1), F(3,1), F(-11,1)};
    r.camera.right   = FxVec3{kOne, 0, 0};
    r.camera.up      = FxVec3{0, kOne, 0};
    r.camera.forward = FxVec3{0, 0, kOne};
    r.camera.halfW   = F(8, 10);
    r.camera.halfH   = F(6, 10);
    return r;
}

int main() {
    HF_TEST_MAIN_INIT();

    // ================= SkyGradient: endpoints + monotonic in dir.y ====================================
    {
        // Straight UP (dir.y = +1) -> the pinned ZENITH color; straight DOWN (dir.y = -1) -> the pinned
        // HORIZON color (t = (n.y + 1)/2: up -> t=1 -> zenith; down -> t=0 -> horizon).
        uint32_t up   = rt::SkyGradient(FxVec3{0, kOne, 0});
        uint32_t down = rt::SkyGradient(FxVec3{0, -kOne, 0});
        check(chOf(up, 0) == rt::kRtSkyZenithR && chOf(up, 1) == rt::kRtSkyZenithG &&
              chOf(up, 2) == rt::kRtSkyZenithB,
              "SkyGradient: straight-up ray == the pinned zenith color");
        check(chOf(down, 0) == rt::kRtSkyHorizonR && chOf(down, 1) == rt::kRtSkyHorizonG &&
              chOf(down, 2) == rt::kRtSkyHorizonB,
              "SkyGradient: straight-down ray == the pinned horizon color");
        check(up != down, "SkyGradient: zenith and horizon colors are DISTINCT (the sky is non-flat)");

        // Monotonic in dir.y: sample a ladder of upward-tilting dirs; each channel marches monotonically from
        // the horizon value toward the zenith value as dir.y rises.
        const int kSteps = 9;
        std::vector<uint32_t> ladder(kSteps);
        for (int i = 0; i < kSteps; ++i) {
            // dir.y from -1..+1 (a fixed +Z forward component keeps the ray non-degenerate).
            fx y = (fx)(-kOne + (fx)((int64_t)2 * kOne * i / (kSteps - 1)));
            ladder[i] = rt::SkyGradient(FxVec3{0, y, kOne});
        }
        bool mono = true;
        for (int ch = 0; ch < 3; ++ch) {
            int horizonCh = chOf(ladder[0], ch);          // approx (dir not purely down, but lowest y)
            int zenithCh  = chOf(ladder[kSteps - 1], ch);
            int dir = (zenithCh >= horizonCh) ? +1 : -1;  // expected march direction for this channel
            for (int i = 1; i < kSteps; ++i) {
                int prev = chOf(ladder[i - 1], ch);
                int cur  = chOf(ladder[i], ch);
                if (dir > 0 ? (cur < prev) : (cur > prev)) mono = false;
            }
        }
        check(mono, "SkyGradient: each channel is MONOTONIC as dir.y rises (horizon -> zenith)");

        // Two distinct directions -> distinct colors (a level ray vs an up ray).
        uint32_t level = rt::SkyGradient(FxVec3{0, 0, kOne});
        check(level != up, "SkyGradient: a level ray and an up ray give DISTINCT colors");
    }

    // ================= RenderSceneHero == RenderSceneReflected when NO ray misses ======================
    {
        // An ENCLOSED "room": a big inward box AABB around the camera + a couple of spheres so no primary OR
        // reflection ray ever escapes to the sky. The ground (primIndex 0) is reflective; the mirror sphere
        // (primIndex 6) is reflective. With no misses, RenderSceneHero (sky on miss) must be byte-identical to
        // RenderSceneReflected (background on miss) — the sky is simply never sampled.
        std::vector<rt::RtSphere> sp;
        std::vector<rt::RtAabb> bx;
        // The room: a large box; rays from inside hit its inner faces. primIndex 0 (reflective floor-ish).
        bx.push_back(rt::RtAabb{FxVec3{F(-30,1), F(-30,1), F(-30,1)},
                                FxVec3{F(30,1),  F(30,1),  F(30,1)}, /*primIndex*/ 0});
        // A couple of spheres in front, including the mirror sphere (primIndex 6).
        sp.push_back(rt::RtSphere{FxVec3{F(0,1), F(0,1), F(4,1)}, F(1,1), /*primIndex*/ 6});
        sp.push_back(rt::RtSphere{FxVec3{F(-3,2), F(0,1), F(3,1)}, F(3,4), /*primIndex*/ 3});

        rt::RtScene s{};
        s.spheres = std::span<const rt::RtSphere>(sp);
        s.aabbs   = std::span<const rt::RtAabb>(bx);
        s.lightDir = rt::RtNormalize(FxVec3{F(4,10), F(8,10), F(-3,10)});
        s.background = rt::PackRGBA8(34, 40, 56, 255);
        rt::RtCamera cam{};
        cam.eye     = FxVec3{F(0,1), F(0,1), F(0,1)};   // inside the room
        cam.right   = FxVec3{kOne, 0, 0};
        cam.up      = FxVec3{0, kOne, 0};
        cam.forward = FxVec3{0, 0, kOne};
        cam.halfW   = F(7, 10);
        cam.halfH   = F(7, 10);

        const uint32_t W = 120, H = 96;
        const size_t kPixels = (size_t)W * H;
        std::vector<uint32_t> hero(kPixels, 0), refl(kPixels, 0);
        rt::RenderSceneHero(s, cam, W, H, std::span<uint32_t>(hero));
        rt::RenderSceneReflected(s, cam, W, H, std::span<uint32_t>(refl));

        // First confirm NO primary ray missed (the enclosure holds) — else the invariant is vacuous.
        bool anyMiss = false;
        for (uint32_t py = 0; py < H && !anyMiss; ++py)
            for (uint32_t px = 0; px < W; ++px) {
                rt::RtRay pr = rt::PrimaryRay(cam, px, py, W, H);
                if (rt::TraceClosest(pr, s).primIndex == rt::kRtMiss) { anyMiss = true; break; }
            }
        check(!anyMiss, "enclosure: no primary ray escapes the room (the no-miss invariant is non-vacuous)");
        check(std::memcmp(hero.data(), refl.data(), kPixels * sizeof(uint32_t)) == 0,
              "RenderSceneHero == RenderSceneReflected when no ray misses (the sky is never sampled)");
    }

    // ================= The HERO scene: shadowed>0, reflective>0, sky graded + determinism ==============
    {
        HeroScene h = BuildHeroScene();
        const uint32_t W = 480, H = 360;
        const size_t kPixels = (size_t)W * H;

        std::vector<uint32_t> a(kPixels, 0), b(kPixels, 0);
        rt::RtHeroCounts cA = rt::RenderSceneHero(h.scene, h.camera, W, H, std::span<uint32_t>(a));
        rt::RtHeroCounts cB = rt::RenderSceneHero(h.scene, h.camera, W, H, std::span<uint32_t>(b));

        check(cA.shadowed > 0, "hero scene: shadowed > 0 (hard shadows contribute)");
        check(cA.reflective > 0, "hero scene: reflective > 0 (mirror reflections contribute)");
        check(cA.shadowed == cB.shadowed && cA.reflective == cB.reflective,
              "RenderSceneHero: two-run counts match");
        check(std::memcmp(a.data(), b.data(), kPixels * sizeof(uint32_t)) == 0,
              "RenderSceneHero: two runs BYTE-IDENTICAL (deterministic)");

        // The sky is GRADED: >= 2 distinct colors among the PRIMARY-MISS pixels.
        uint32_t firstSky = 0; bool haveFirst = false; bool twoDistinct = false;
        for (uint32_t py = 0; py < H && !twoDistinct; ++py)
            for (uint32_t px = 0; px < W; ++px) {
                rt::RtRay pr = rt::PrimaryRay(h.camera, px, py, W, H);
                if (rt::TraceClosest(pr, h.scene).primIndex != rt::kRtMiss) continue;
                uint32_t c = a[(size_t)py * W + px];
                if (!haveFirst) { firstSky = c; haveFirst = true; }
                else if (c != firstSky) { twoDistinct = true; break; }
            }
        check(haveFirst, "hero scene: there ARE primary-miss (sky) pixels");
        check(twoDistinct, "hero scene: the sky is GRADED (>=2 distinct miss colors)");
    }

    if (g_fail == 0) std::printf("rt_hero_test: all RT6 CPU invariants PASS\n");
    return g_fail == 0 ? 0 : 1;
}
