// Slice RT2 — Hardware Ray Tracing: the CPU-side invariants the HW inline-ray-query kernel
// (shaders/rt_query.comp) rests on. The HW==SW-GPU==CPU memcmp proof lives in the --rt2-query-shot
// showcase (it needs a real GPU); this pure-CPU test pins the deterministic-reconciliation contract that
// makes that proof possible:
//
//   * The RT2 scene's rtrace::RenderScene is DETERMINISTIC (two runs byte-identical).
//   * The AABB-MARGIN INVARIANT (the candidate-completeness guarantee): for every primitive, every true fx
//     hit point produced by the frozen rtrace:: intersection lies INSIDE that primitive's float bounding
//     AABB inflated by kRtAabbMargin. This is exactly what guarantees the driver's float BVH overlap (a
//     superset of the inflated AABB) never SKIPS a true fx hit -> the HW path can't miss a hit the no-cull
//     SW reference finds.
//   * The per-pixel HIT MASK of the full scene matches a BRUTE-FORCE reference over the same primitives
//     (the no-cull oracle the HW path is reconciled against).
//
// Pure C++ (hf_core), ASan-eligible. rtrace.h #includes sim/fpx.h read-only; rtrace.h is BYTE-FROZEN
// (this test #includes it read-only and defines the RT2 scene locally, mirroring the showcase TU).
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

// The fixed Q16.16 AABB-margin (kRtAabbMargin = 1/64 world unit) the backend adds to each BLAS AABB
// half-extent. Mirrors engine/rhi_vulkan/vulkan_accel.h::kRtAabbMargin (1.0f/64.0f) in fx.
static const fx kRtAabbMarginFx = kOne / 64;

// Build the SAME RT2 scene the showcase defines (a ground AABB + a 4x4 sphere grid + two boxes). The
// owning storage lives in the returned struct.
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

    Rt2Scene s = BuildRt2Scene();
    const uint32_t W = 320, H = 240;
    const size_t kPixels = (size_t)W * H;

    // ================= RenderScene determinism (two runs byte-identical) =================
    {
        std::vector<uint32_t> a(kPixels, 0), b(kPixels, 0);
        uint32_t hitsA = rt::RenderScene(s.scene, s.camera, W, H, std::span<uint32_t>(a));
        uint32_t hitsB = rt::RenderScene(s.scene, s.camera, W, H, std::span<uint32_t>(b));
        check(hitsA == hitsB, "RT2 RenderScene: two-run hit count matches");
        check(std::memcmp(a.data(), b.data(), kPixels * sizeof(uint32_t)) == 0,
              "RT2 RenderScene: two runs BYTE-IDENTICAL (deterministic)");
        check(hitsA > 0 && hitsA < kPixels, "RT2 RenderScene: a non-trivial, partial set of pixels HIT");
    }

    // ================= AABB-margin invariant (candidate completeness) =================
    // For every primitive, fire a dense bundle of rays from the scene camera; for each TRUE fx hit on
    // that primitive, the hit point must lie INSIDE the primitive's float bounding AABB inflated by
    // kRtAabbMargin. (If a real hit lay OUTSIDE the inflated AABB, the driver's float BVH could cull the
    // leaf and the HW path would skip the hit -> the completeness oracle would fail. The margin guarantees
    // it never does.) We test the actual fx hit POSITIONS the frozen intersection produces.
    {
        int sphereChecked = 0, aabbChecked = 0, outside = 0;

        // The inflated bound for a sphere: center +- (radius + margin).
        auto insideSphereBound = [&](const rt::RtSphere& sp, const FxVec3& pos) -> bool {
            fx m = sp.radius + kRtAabbMarginFx;
            return pos.x >= sp.center.x - m && pos.x <= sp.center.x + m &&
                   pos.y >= sp.center.y - m && pos.y <= sp.center.y + m &&
                   pos.z >= sp.center.z - m && pos.z <= sp.center.z + m;
        };
        // The inflated bound for an AABB: lo - margin .. hi + margin.
        auto insideAabbBound = [&](const rt::RtAabb& bx, const FxVec3& pos) -> bool {
            return pos.x >= bx.lo.x - kRtAabbMarginFx && pos.x <= bx.hi.x + kRtAabbMarginFx &&
                   pos.y >= bx.lo.y - kRtAabbMarginFx && pos.y <= bx.hi.y + kRtAabbMarginFx &&
                   pos.z >= bx.lo.z - kRtAabbMarginFx && pos.z <= bx.hi.z + kRtAabbMarginFx;
        };

        for (uint32_t py = 0; py < H; ++py) {
            for (uint32_t px = 0; px < W; ++px) {
                rt::RtRay ray = rt::PrimaryRay(s.camera, px, py, W, H);
                // Test each sphere independently (the per-primitive intersection the HW candidate runs).
                for (const rt::RtSphere& sp : s.spheres) {
                    rt::RtHit h;
                    if (rt::IntersectSphere(ray, sp, h)) {
                        ++sphereChecked;
                        if (!insideSphereBound(sp, h.pos)) ++outside;
                    }
                }
                for (const rt::RtAabb& bx : s.aabbs) {
                    rt::RtHit h;
                    if (rt::IntersectAabb(ray, bx, h)) {
                        ++aabbChecked;
                        if (!insideAabbBound(bx, h.pos)) ++outside;
                    }
                }
            }
        }
        check(sphereChecked > 0, "RT2 margin: sphere hits were exercised");
        check(aabbChecked > 0, "RT2 margin: aabb hits were exercised");
        check(outside == 0, "RT2 margin invariant: EVERY true fx hit point lies inside its inflated AABB");
        if (outside != 0)
            std::printf("  (%d hit points fell outside their inflated AABB — margin insufficient)\n", outside);
    }

    // ================= Hit mask matches a brute-force reference =================
    // The full-scene closest-hit mask (hit vs miss, by background color) must equal an INDEPENDENT
    // brute-force reference that re-traces every primitive per pixel. This is the no-cull oracle the HW
    // path is reconciled against (HW image == this brute-force image, proven in the showcase memcmp).
    {
        std::vector<uint32_t> img(kPixels, 0);
        rt::RenderScene(s.scene, s.camera, W, H, std::span<uint32_t>(img));
        int maskMismatch = 0, primMismatch = 0;
        for (uint32_t py = 0; py < H; ++py) {
            for (uint32_t px = 0; px < W; ++px) {
                rt::RtRay ray = rt::PrimaryRay(s.camera, px, py, W, H);
                // Independent brute-force closest-hit (NOT via TraceClosest's lambda) — same total order.
                rt::RtHit best = rt::RtMissHit();
                for (const rt::RtSphere& sp : s.spheres) {
                    rt::RtHit h;
                    if (rt::IntersectSphere(ray, sp, h)) {
                        if (best.primIndex == rt::kRtMiss || h.t < best.t ||
                            (h.t == best.t && h.primIndex < best.primIndex)) best = h;
                    }
                }
                for (const rt::RtAabb& bx : s.aabbs) {
                    rt::RtHit h;
                    if (rt::IntersectAabb(ray, bx, h)) {
                        if (best.primIndex == rt::kRtMiss || h.t < best.t ||
                            (h.t == best.t && h.primIndex < best.primIndex)) best = h;
                    }
                }
                bool refHit = best.primIndex != rt::kRtMiss;
                bool imgHit = img[(size_t)py * W + px] != s.scene.background;
                if (refHit != imgHit) ++maskMismatch;
                // The shaded color of a ref-hit pixel must match RenderScene's (closest primIndex agrees).
                if (refHit && imgHit) {
                    uint32_t refShade = rt::ShadeHitInt(best, s.scene);
                    if (refShade != img[(size_t)py * W + px]) ++primMismatch;
                }
            }
        }
        check(maskMismatch == 0, "RT2 hit mask matches the brute-force no-cull reference");
        check(primMismatch == 0, "RT2 closest-prim shade matches the brute-force reference per pixel");
    }

    if (g_fail == 0) std::printf("rt_query_test: all RT2 CPU invariants PASS\n");
    return g_fail == 0 ? 0 : 1;
}
