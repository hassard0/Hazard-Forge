// Slice RT1 — Hardware Ray Tracing: the DETERMINISTIC Q16.16 SW reference ray tracer (engine/render/
// rtrace.h) that the GPU shaders/rt_trace.comp.hlsl copies VERBATIM + proves bit-identical. Pure CPU
// (header-only, no device, no backend symbols). Namespace hf::render::rtrace.
//
// What this test PINS (the contracts the GPU rt_trace.comp + the GPU==CPU proof build on):
//   * IntersectSphere: a ray fired at a sphere CENTER hits at the closed-form t == dist - radius (a
//     non-negative root), with the surface normal pointing back along the ray; a ray pointed away misses.
//   * IntersectAabb: a ray fired at a known AABB FACE hits at the slab-entry t with the correct face
//     normal; a parallel-outside ray misses.
//   * Miss handling: TraceClosest over an empty / non-intersecting scene returns kRtMiss + kRtNoHit.
//   * TraceClosest: picks the NEAREST primitive by the (t, primIndex) total order, INCLUDING an equal-t
//     tie-break (two coincident hits at the same t -> the smaller primIndex wins, order-independent).
//   * Degenerate ray rejection: a dir==0 ray hits nothing (FxDot(dir,dir)==0 guards every primitive).
//   * RenderScene: two runs byte-identical (determinism) + a non-trivial hit count (the scene is visible).
//   * ShadeHitInt: a fixed normal under a fixed light quantizes to a PINNED RGBA8 (integer-exactness); a
//     miss returns the background exactly.
//
// Pure C++ (hf_core), ASan-eligible like the other render-math tests.
#include "render/rtrace.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace rt = hf::render::rtrace;
using rt::fx;
using rt::kOne;
using rt::kFrac;
using rt::FxVec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Q16.16 of an integer (exact).
static fx FromInt(int v) { return (fx)(v << kFrac); }

int main() {
    HF_TEST_MAIN_INIT();

    // ================= IntersectSphere — analytic closed form =================
    {
        // A unit-dir ray along +Z from the origin at a sphere centered at (0,0,10) radius 2.
        // Closed form: the near hit t == dist - radius == 10 - 2 == 8 (in Q16.16).
        rt::RtRay ray{FxVec3{0, 0, 0}, FxVec3{0, 0, kOne}};
        rt::RtSphere s{FxVec3{0, 0, FromInt(10)}, FromInt(2), /*primIndex*/ 7};
        rt::RtHit h;
        bool hit = rt::IntersectSphere(ray, s, h);
        check(hit, "IntersectSphere: a ray fired at the sphere center HITS");
        check(h.t == FromInt(8), "IntersectSphere: near root t == dist - radius (8.0) closed form");
        check(h.primIndex == 7u, "IntersectSphere: writes the primitive index");
        // The hit position is at z==8 (origin + dir*t); the normal points back toward -Z (the near cap).
        check(h.pos.z == FromInt(8), "IntersectSphere: hit pos z == 8.0");
        check(h.normal.z == -kOne, "IntersectSphere: near-cap normal points back along the ray (-Z)");

        // A ray pointed AWAY from the sphere (along -Z) misses (both roots behind the origin).
        rt::RtRay away{FxVec3{0, 0, 0}, FxVec3{0, 0, -kOne}};
        rt::RtHit hm;
        check(!rt::IntersectSphere(away, s, hm), "IntersectSphere: ray pointed away MISSES");

        // A ray that passes well to the side misses (discriminant < 0).
        rt::RtRay side{FxVec3{FromInt(100), 0, 0}, FxVec3{0, 0, kOne}};
        rt::RtHit hs;
        check(!rt::IntersectSphere(side, s, hs), "IntersectSphere: ray far to the side MISSES (disc<0)");
    }

    // ================= IntersectAabb — slab-test face hit =================
    {
        // A box [-1,1]^2 in XY, spanning z in [4,6]. A +Z ray from the origin hits the -Z face at t==4.
        rt::RtAabb box{FxVec3{FromInt(-1), FromInt(-1), FromInt(4)},
                       FxVec3{FromInt(1), FromInt(1), FromInt(6)}, /*primIndex*/ 3};
        rt::RtRay ray{FxVec3{0, 0, 0}, FxVec3{0, 0, kOne}};
        rt::RtHit h;
        bool hit = rt::IntersectAabb(ray, box, h);
        check(hit, "IntersectAabb: a ray fired into a box face HITS");
        check(h.t == FromInt(4), "IntersectAabb: slab-entry t == 4.0 (the near -Z face)");
        check(h.primIndex == 3u, "IntersectAabb: writes the primitive index");
        check(h.normal.z == -kOne, "IntersectAabb: -Z face normal == (0,0,-1)");
        check(h.pos.z == FromInt(4), "IntersectAabb: hit pos z == 4.0");

        // A ray parallel to the box but OUTSIDE its XY slabs misses.
        rt::RtRay parallel{FxVec3{FromInt(10), 0, 0}, FxVec3{0, 0, kOne}};
        rt::RtHit hm;
        check(!rt::IntersectAabb(parallel, box, hm), "IntersectAabb: parallel ray outside the slab MISSES");

        // An X-face hit: a +X ray from (-10,0,5) hits the box's -X face (x==-1) at t==9, normal (-1,0,0).
        rt::RtRay xray{FxVec3{FromInt(-10), 0, FromInt(5)}, FxVec3{kOne, 0, 0}};
        rt::RtHit hx;
        check(rt::IntersectAabb(xray, box, hx), "IntersectAabb: +X ray into the -X face HITS");
        check(hx.t == FromInt(9), "IntersectAabb: -X face slab-entry t == 9.0");
        check(hx.normal.x == -kOne, "IntersectAabb: -X face normal == (-1,0,0)");
    }

    // ================= Miss handling + kRtMiss =================
    {
        rt::RtScene empty{};   // no spheres, no aabbs
        rt::RtRay ray{FxVec3{0, 0, 0}, FxVec3{0, 0, kOne}};
        rt::RtHit h = rt::TraceClosest(ray, empty);
        check(h.primIndex == rt::kRtMiss, "TraceClosest: empty scene -> kRtMiss");
        check(h.t == rt::kRtNoHit, "TraceClosest: empty scene -> kRtNoHit (t sentinel)");

        // A scene with one sphere the ray points AWAY from -> still a miss.
        std::vector<rt::RtSphere> spheres{rt::RtSphere{FxVec3{0, 0, FromInt(10)}, FromInt(2), 0}};
        rt::RtScene sc{};
        sc.spheres = std::span<const rt::RtSphere>(spheres);
        rt::RtRay away{FxVec3{0, 0, 0}, FxVec3{0, 0, -kOne}};
        rt::RtHit hm = rt::TraceClosest(away, sc);
        check(hm.primIndex == rt::kRtMiss, "TraceClosest: ray away from the only prim -> kRtMiss");
    }

    // ================= TraceClosest — nearest by (t, primIndex) =================
    {
        // Two spheres on the +Z axis at z=10 (r=1) and z=5 (r=1). A +Z ray hits the NEARER (z=5) first.
        std::vector<rt::RtSphere> spheres{
            rt::RtSphere{FxVec3{0, 0, FromInt(10)}, FromInt(1), /*primIndex*/ 0},
            rt::RtSphere{FxVec3{0, 0, FromInt(5)},  FromInt(1), /*primIndex*/ 1},
        };
        rt::RtScene sc{};
        sc.spheres = std::span<const rt::RtSphere>(spheres);
        rt::RtRay ray{FxVec3{0, 0, 0}, FxVec3{0, 0, kOne}};
        rt::RtHit h = rt::TraceClosest(ray, sc);
        check(h.primIndex == 1u, "TraceClosest: picks the NEARER sphere (primIndex 1 at z=5)");
        check(h.t == FromInt(4), "TraceClosest: nearer hit t == 5 - 1 == 4.0");

        // Storage-order independence: reverse the array, same nearest result.
        std::vector<rt::RtSphere> rev{spheres[1], spheres[0]};
        rt::RtScene sc2{};
        sc2.spheres = std::span<const rt::RtSphere>(rev);
        rt::RtHit h2 = rt::TraceClosest(ray, sc2);
        check(h2.primIndex == 1u && h2.t == h.t,
              "TraceClosest: result is storage-order-independent (reversed array -> same hit)");
    }

    // ================= TraceClosest — EQUAL-t tie-break (smaller primIndex wins) =================
    {
        // Two identical spheres at the SAME center -> the same t. The tie-break must pick the SMALLER
        // primIndex deterministically regardless of storage order.
        std::vector<rt::RtSphere> spheres{
            rt::RtSphere{FxVec3{0, 0, FromInt(10)}, FromInt(2), /*primIndex*/ 9},
            rt::RtSphere{FxVec3{0, 0, FromInt(10)}, FromInt(2), /*primIndex*/ 4},
        };
        rt::RtScene sc{};
        sc.spheres = std::span<const rt::RtSphere>(spheres);
        rt::RtRay ray{FxVec3{0, 0, 0}, FxVec3{0, 0, kOne}};
        rt::RtHit h = rt::TraceClosest(ray, sc);
        check(h.primIndex == 4u, "TraceClosest: equal-t tie-break picks the smaller primIndex (4)");

        // Reverse storage order -> SAME winner (the tie-break is order-independent).
        std::vector<rt::RtSphere> rev{spheres[1], spheres[0]};
        rt::RtScene sc2{};
        sc2.spheres = std::span<const rt::RtSphere>(rev);
        rt::RtHit h2 = rt::TraceClosest(ray, sc2);
        check(h2.primIndex == 4u, "TraceClosest: equal-t tie-break is storage-order-independent");
    }

    // ================= Degenerate-ray rejection =================
    {
        // A dir==0 ray: FxDot(dir,dir)==0 -> every IntersectSphere returns false; the AABB slab test sees
        // d[ax]==0 on all axes and (for a box the origin is outside) misses. TraceClosest -> kRtMiss.
        std::vector<rt::RtSphere> spheres{rt::RtSphere{FxVec3{0, 0, FromInt(10)}, FromInt(2), 0}};
        std::vector<rt::RtAabb> aabbs{
            rt::RtAabb{FxVec3{FromInt(-1), FromInt(-1), FromInt(4)},
                       FxVec3{FromInt(1), FromInt(1), FromInt(6)}, 1}};
        rt::RtScene sc{};
        sc.spheres = std::span<const rt::RtSphere>(spheres);
        sc.aabbs   = std::span<const rt::RtAabb>(aabbs);
        rt::RtRay degenerate{FxVec3{0, 0, 0}, FxVec3{0, 0, 0}};
        rt::RtSphere s0 = spheres[0];
        rt::RtHit hs;
        check(!rt::IntersectSphere(degenerate, s0, hs), "IntersectSphere: degenerate dir==0 -> no hit");
        rt::RtHit h = rt::TraceClosest(degenerate, sc);
        check(h.primIndex == rt::kRtMiss, "TraceClosest: degenerate dir==0 ray -> kRtMiss");
    }

    // ================= ShadeHitInt — integer-exact RGBA8 =================
    {
        // A hit on primIndex 0 (warm-red albedo) with the normal facing the light directly (ndl == 1).
        // Light straight back along -Z, normal +Z toward it -> dot == 1 -> diffuse == 1.0.
        rt::RtScene sc{};
        sc.lightDir = FxVec3{0, 0, kOne};
        sc.background = rt::PackRGBA8(34, 40, 56, 255);
        rt::RtHit hit;
        hit.primIndex = 0;
        hit.normal = FxVec3{0, 0, kOne};   // ndl == dot(+Z, +Z) == 1.0
        uint32_t rgba = rt::ShadeHitInt(hit, sc);
        // albedo[0] = (0.78, 0.30, 0.26); diffuse == 1.0; quantized: round-toward-zero of v*255.
        // r = floor(0.78*255) ... compute the SAME way the header does to PIN it.
        auto q = [&](int pct) -> int {
            fx ch = (fx)(((int64_t)kOne * pct) / 100);
            // diffuse == 1.0 -> lit == ch; (ch*255)>>kFrac.
            return (int)(((int64_t)ch * 255) >> kFrac);
        };
        uint32_t expect = rt::PackRGBA8(q(78), q(30), q(26), 255);
        check(rgba == expect, "ShadeHitInt: full-light warm-red albedo quantizes to the PINNED RGBA8");
        check((rgba >> 24) == 255u, "ShadeHitInt: alpha == 255");

        // A miss returns the background EXACTLY.
        rt::RtHit miss = rt::RtMissHit();
        check(rt::ShadeHitInt(miss, sc) == sc.background, "ShadeHitInt: a miss returns the background");

        // The ambient floor: a normal facing AWAY from the light (ndl clamped to 0) -> diffuse == ambient.
        rt::RtHit dark;
        dark.primIndex = 0;
        dark.normal = FxVec3{0, 0, -kOne};   // dot(-Z, +Z) == -1 -> clamped to 0
        uint32_t darkRgba = rt::ShadeHitInt(dark, sc);
        fx ambient = (fx)(((int64_t)kOne * 18) / 100);
        auto qa = [&](int pct) -> int {
            fx ch = (fx)(((int64_t)kOne * pct) / 100);
            fx lit = rt::fxmul(ch, ambient);
            return (int)(((int64_t)lit * 255) >> kFrac);
        };
        uint32_t expectDark = rt::PackRGBA8(qa(78), qa(30), qa(26), 255);
        check(darkRgba == expectDark, "ShadeHitInt: a back-facing normal lit only by the ambient floor");
    }

    // ================= RenderScene — two-run determinism + a visible scene =================
    {
        rt::RtScene1 r = rt::BuildRt1Scene();
        const uint32_t W = 64, H = 48;
        std::vector<uint32_t> a((size_t)W * H, 0), b((size_t)W * H, 0);
        uint32_t hitsA = rt::RenderScene(r.scene, r.camera, W, H, std::span<uint32_t>(a));
        uint32_t hitsB = rt::RenderScene(r.scene, r.camera, W, H, std::span<uint32_t>(b));
        check(hitsA == hitsB, "RenderScene: two runs report the same hit count");
        check(std::memcmp(a.data(), b.data(), a.size() * sizeof(uint32_t)) == 0,
              "RenderScene: two runs are byte-identical (determinism)");
        check(hitsA > 0, "RenderScene: the scene is visible (some pixels hit a primitive)");
        check(hitsA < W * H, "RenderScene: the scene has background (not every pixel is a hit)");

        // Provenance: rebuilding the scene + rendering again is also byte-identical (pure function).
        rt::RtScene1 r2 = rt::BuildRt1Scene();
        std::vector<uint32_t> c((size_t)W * H, 0);
        rt::RenderScene(r2.scene, r2.camera, W, H, std::span<uint32_t>(c));
        check(std::memcmp(a.data(), c.data(), a.size() * sizeof(uint32_t)) == 0,
              "RenderScene: a fresh scene build renders byte-identically (pure function of the scene)");
    }

    if (g_fail == 0) { std::printf("rt_test OK\n"); return 0; }
    std::printf("rt_test: %d failures\n", g_fail);
    return 1;
}
