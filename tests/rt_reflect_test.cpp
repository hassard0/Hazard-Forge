// Slice RT4 — Hardware Ray Tracing: the CPU-side invariants the HW mirror-reflection kernel
// (shaders/rt_reflect.comp) rests on. The HW==CPU memcmp proof lives in the --rt4-reflect-shot showcase (it
// needs a real GPU); this pure-CPU test pins the deterministic reflection contract that makes that proof
// possible:
//
//   * FxReflect ANALYTIC: reflecting a ray about an axis-aligned plane normal flips ONLY the component
//     along the normal (the mirror law r = d - 2*(d·n)*n); and reflecting TWICE returns the original
//     direction (the involution property of a mirror).
//   * BLEND boundaries: the per-channel integer blend c = (cS*(kOne-refl) + cR*refl) >> kFrac is EXACTLY
//     cS at refl=0 (so a non-reflective pixel is byte-identical to the RT3 shadowed shade) and EXACTLY cR
//     at refl=kOne (the reflected color fully replaces the surface).
//   * refl=0 SCENE: a scene whose primitives are ALL non-reflective makes RenderSceneReflected
//     BYTE-IDENTICAL to RenderSceneShadowed (the reflection path is a transparent no-op when nothing
//     reflects).
//   * REFLECTIVE pixel CONTRIBUTES: on the RT2 scene (reflective ground + mirror sphere) RenderSceneReflected
//     DIFFERS from the matte RenderSceneShadowed at >=1 pixel (the reflection actually changed the image —
//     not a no-op blend); and reflective:>0.
//   * RenderSceneReflected is DETERMINISTIC (two runs byte-identical).
//
// Pure C++ (hf_core), ASan-eligible. rtrace.h #includes sim/fpx.h read-only; rtrace.h's RT1+RT3 code is
// BYTE-FROZEN (RT4 is APPEND-ONLY); this test #includes it read-only and defines the RT2 scene locally.
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

// Build the SAME RT2 scene the showcase defines (a ground AABB + a 4x4 sphere grid + two boxes). The ground
// (primIndex 0) and the mirror sphere (primIndex kRtMirrorSpherePrim) are reflective.
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

// The EXACT per-channel integer blend rtrace::RenderSceneReflected pins (for the boundary-property tests).
static uint32_t BlendRGBA8(uint32_t cS, uint32_t cR, fx refl) {
    auto chOf = [](uint32_t c, int i) -> int32_t { return (int32_t)((c >> (8 * i)) & 0xFF); };
    auto blend = [&](int i) -> int32_t {
        int32_t s = chOf(cS, i);
        int32_t r = chOf(cR, i);
        int64_t mix = (int64_t)s * (int64_t)(kOne - refl) + (int64_t)r * (int64_t)refl;
        return (int32_t)(mix >> kFrac);
    };
    return rt::PackRGBA8(blend(0), blend(1), blend(2), 255);
}

int main() {
    HF_TEST_MAIN_INIT();

    // ================= FxReflect analytic (axis-plane mirror + involution) =============================
    {
        // Reflect a ray about the +Y plane normal (n = (0,1,0)). A downward-and-forward dir (1,-1,0) must
        // mirror to (1,+1,0): the Y component flips, X/Z unchanged (the mirror law about the Y axis).
        FxVec3 n{0, kOne, 0};
        FxVec3 d{kOne, -kOne, 0};
        FxVec3 r = rt::FxReflect(d, n);
        check(r.x == kOne && r.y == kOne && r.z == 0,
              "FxReflect: about +Y, the Y component flips and X/Z are unchanged (the mirror law)");

        // Involution: reflecting the reflected ray about the SAME normal returns the original direction.
        FxVec3 rr = rt::FxReflect(r, n);
        check(rr.x == d.x && rr.y == d.y && rr.z == d.z,
              "FxReflect: reflecting TWICE about the same normal returns the original direction");

        // A grazing-but-nonaxis case off the +X plane (n = (1,0,0)): dir (-1,0,-1) -> (+1,0,-1) (X flips).
        FxVec3 nx{kOne, 0, 0};
        FxVec3 dx{-kOne, 0, -kOne};
        FxVec3 rx = rt::FxReflect(dx, nx);
        check(rx.x == kOne && rx.y == 0 && rx.z == -kOne,
              "FxReflect: about +X, the X component flips and Y/Z are unchanged");
        FxVec3 rrx = rt::FxReflect(rx, nx);
        check(rrx.x == dx.x && rrx.y == dx.y && rrx.z == dx.z,
              "FxReflect: involution holds for the +X mirror too");
    }

    // ================= Blend boundaries (refl=0 -> cS exactly; refl=kOne -> cR exactly) =================
    {
        uint32_t cS = rt::PackRGBA8(200, 120, 40, 255);
        uint32_t cR = rt::PackRGBA8(10, 220, 130, 255);
        check(BlendRGBA8(cS, cR, 0) == cS,
              "blend: refl=0 -> the surface color cS EXACTLY (no reflection contribution)");
        check(BlendRGBA8(cS, cR, kOne) == cR,
              "blend: refl=kOne -> the reflected color cR EXACTLY (the surface fully replaced)");
        // A mid blend lands strictly between (each channel between the two endpoints, inclusive).
        uint32_t half = BlendRGBA8(cS, cR, kOne / 2);
        auto ch = [](uint32_t c, int i) { return (int)((c >> (8 * i)) & 0xFF); };
        bool between = true;
        for (int i = 0; i < 3; ++i) {
            int lo = std::min(ch(cS, i), ch(cR, i));
            int hi = std::max(ch(cS, i), ch(cR, i));
            if (ch(half, i) < lo || ch(half, i) > hi) between = false;
        }
        check(between, "blend: a 0.5 blend lands between cS and cR per channel");
    }

    // ================= refl=0 SCENE: no reflective prims -> reflected == shadowed BYTE-IDENTICAL ==========
    {
        // A scene whose primitives are ALL non-reflective (no primIndex 0, no kRtMirrorSpherePrim). Use a
        // ground-like AABB and two spheres at primIndices that ReflectivityFor returns 0 for.
        std::vector<rt::RtSphere> sp;
        std::vector<rt::RtAabb> bx;
        // A floor AABB at a NON-reflective primIndex (deliberately not 0).
        bx.push_back(rt::RtAabb{FxVec3{F(-20,1), F(-3,1), F(-20,1)},
                                FxVec3{F(20,1),  F(-1,1), F(20,1)}, /*primIndex*/ 10});
        sp.push_back(rt::RtSphere{FxVec3{F(0,1), F(0,1), F(2,1)}, F(1,1), /*primIndex*/ 11});
        sp.push_back(rt::RtSphere{FxVec3{F(-2,1), F(0,1), F(3,1)}, F(3,4), /*primIndex*/ 12});
        // Sanity: every primIndex here is non-reflective.
        check(rt::ReflectivityFor(10) == 0 && rt::ReflectivityFor(11) == 0 && rt::ReflectivityFor(12) == 0,
              "ReflectivityFor: the control-scene primitives are all non-reflective");

        rt::RtScene s{};
        s.spheres = std::span<const rt::RtSphere>(sp);
        s.aabbs   = std::span<const rt::RtAabb>(bx);
        s.lightDir = rt::RtNormalize(FxVec3{F(4,10), F(8,10), F(-3,10)});
        s.background = rt::PackRGBA8(34, 40, 56, 255);
        rt::RtCamera cam{};
        cam.eye     = FxVec3{F(0,1), F(2,1), F(-9,1)};
        cam.right   = FxVec3{kOne, 0, 0};
        cam.up      = FxVec3{0, kOne, 0};
        cam.forward = FxVec3{0, 0, kOne};
        cam.halfW   = F(7, 10);
        cam.halfH   = F(7, 10);

        const uint32_t W = 160, H = 120;
        const size_t kPixels = (size_t)W * H;
        std::vector<uint32_t> refl(kPixels, 0), shad(kPixels, 0);
        uint32_t reflective = rt::RenderSceneReflected(s, cam, W, H, std::span<uint32_t>(refl));
        rt::RenderSceneShadowed(s, cam, W, H, std::span<uint32_t>(shad));
        check(reflective == 0, "RenderSceneReflected: a no-reflective-prim scene has reflective:0");
        check(std::memcmp(refl.data(), shad.data(), kPixels * sizeof(uint32_t)) == 0,
              "RenderSceneReflected == RenderSceneShadowed when nothing reflects (byte-identical)");
    }

    // ================= REFLECTIVE pixel CONTRIBUTES + determinism on the RT2 scene =====================
    {
        Rt2Scene s = BuildRt2Scene();
        const uint32_t W = 320, H = 240;
        const size_t kPixels = (size_t)W * H;

        std::vector<uint32_t> a(kPixels, 0), b(kPixels, 0), shad(kPixels, 0);
        uint32_t reflA = rt::RenderSceneReflected(s.scene, s.camera, W, H, std::span<uint32_t>(a));
        uint32_t reflB = rt::RenderSceneReflected(s.scene, s.camera, W, H, std::span<uint32_t>(b));
        rt::RenderSceneShadowed(s.scene, s.camera, W, H, std::span<uint32_t>(shad));

        check(reflA == reflB, "RenderSceneReflected: two-run reflective count matches");
        check(std::memcmp(a.data(), b.data(), kPixels * sizeof(uint32_t)) == 0,
              "RenderSceneReflected: two runs BYTE-IDENTICAL (deterministic)");

        check(reflA > 0, "RenderSceneReflected: SOME surface is reflective (reflective > 0)");

        // The reflection actually contributed: the reflected image differs from the matte shadowed image at
        // >= 1 pixel (not a no-op blend).
        bool differs = false;
        for (size_t p = 0; p < kPixels; ++p) if (a[p] != shad[p]) { differs = true; break; }
        check(differs, "RenderSceneReflected: differs from the matte RenderSceneShadowed (reflection contributed)");
        if (!(reflA > 0 && differs))
            std::printf("  (reflective:%u differs:%d — expected reflective>0 AND >=1 changed pixel)\n",
                        reflA, (int)differs);
    }

    if (g_fail == 0) std::printf("rt_reflect_test: all RT4 CPU invariants PASS\n");
    return g_fail == 0 ? 0 : 1;
}
