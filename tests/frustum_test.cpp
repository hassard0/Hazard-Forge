// Slice AQ — frustum culling. Pure CPU math (header-only, no device, no backend symbols).
// Mirrors engine/render/frustum.h, the SAME extraction + tests the --cull-shot showcase uses to
// decide which renderables to skip. Namespace hf::render::frustum.
//
// The crux this test PINS: Gribb-Hartmann plane extraction is convention-sensitive. engine/math
// Mat4 is COLUMN-MAJOR (element(row,col) == m[col*4 + row]) and the engine's clip space is the
// Vulkan one (x,y in [-w,w], z in [0,w], built by Mat4::Perspective with a baked Y-flip). A
// transpose or a depth-range bug here silently culls visible geometry, so we:
//   * hand-check the 6 extracted planes of a KNOWN symmetric perspective (the inward normals point
//     where geometry actually is),
//   * hand-check SphereOutside / inside on points placed by hand relative to those planes,
//   * hand-check AabbOutside on the positive/negative-vertex cases,
//   * and cross-check the whole partition against an INDEPENDENT brute-force reference (project each
//     test point through the matrix and clip in NDC) over many random spheres — if our cheap plane
//     test ever culls something the projection keeps on-screen, that is a false-cull (a visible bug)
//     and the test fails.
//
// Pure C++ (hf_core), ASan-eligible like the other render-math tests.
#include "render/frustum.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include <random>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::math;
namespace fr = hf::render::frustum;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }

// Independent reference: a point is inside the frustum iff its CLIP coords satisfy the Vulkan clip
// inequalities -w<=x<=w, -w<=y<=w, 0<=z<=w (this is what the rasterizer actually keeps). Done by
// raw matrix*vector with NO reuse of frustum.h, so it cannot share a bug with the code under test.
static bool RefPointInside(const Mat4& m, const Vec3& p) {
    float x = m.m[0]*p.x + m.m[4]*p.y + m.m[8] *p.z + m.m[12];
    float y = m.m[1]*p.x + m.m[5]*p.y + m.m[9] *p.z + m.m[13];
    float z = m.m[2]*p.x + m.m[6]*p.y + m.m[10]*p.z + m.m[14];
    float w = m.m[3]*p.x + m.m[7]*p.y + m.m[11]*p.z + m.m[15];
    return x >= -w && x <= w && y >= -w && y <= w && z >= 0.0f && z <= w;
}

int main() {
    HF_TEST_MAIN_INIT();
    // A known view-proj: camera at +Z=5 looking at origin (down -Z), 90deg vertical FOV, square
    // aspect, near 1, far 11. Symmetric, so the side planes are tilted at 45deg.
    const float fovY = 1.57079633f;  // 90 degrees
    const float aspect = 1.0f;
    const float zn = 1.0f, zf = 11.0f;
    Mat4 view = Mat4::LookAt(Vec3{0, 0, 5}, Vec3{0, 0, 0}, Vec3{0, 1, 0});
    Mat4 proj = Mat4::Perspective(fovY, aspect, zn, zf);
    Mat4 vp = proj * view;

    fr::Frustum f = fr::FromViewProj(vp);

    // ---- All extracted planes are normalized (|n| == 1). ----
    {
        bool allNorm = true;
        for (int i = 0; i < 6; ++i)
            if (!approx(length(f.planes[i].n), 1.0f)) allNorm = false;
        check(allNorm, "all 6 frustum planes are normalized");
    }

    // ---- Hand-checked near / far planes. Camera at z=5 looking down -Z: the near plane sits at
    // z = 5-near = 4 with an inward normal pointing toward -Z (0,0,-1); far at z = 5-far = -6 with
    // an inward normal pointing toward +Z (0,0,+1). signedDistance(inside point) must be >= 0. ----
    {
        // Order in frustum.h: 0=L 1=R 2=B 3=T 4=N 5=F.
        const fr::Plane& nearP = f.planes[4];
        const fr::Plane& farP  = f.planes[5];
        check(approx(nearP.n.x, 0) && approx(nearP.n.y, 0) && approx(nearP.n.z, -1),
              "near-plane inward normal is (0,0,-1)");
        // dist(n,p)+d at the near plane z=4 is 0: d = -(n . pNear) = -((-1)*4) = 4.
        check(approx(nearP.d, 4.0f), "near-plane d == 4 (plane at z=4)");
        check(approx(farP.n.x, 0) && approx(farP.n.y, 0) && approx(farP.n.z, 1),
              "far-plane inward normal is (0,0,+1)");
        // far plane at z=-6: d = -(1 * -6) = 6.
        check(approx(farP.d, 6.0f), "far-plane d == 6 (plane at z=-6)");
        // A point at the camera-facing center (z=0) is inside both: dist near = -0 + 4 = 4 > 0,
        // dist far = 0 + 6 = 6 > 0.
        check(fr::SignedDistance(nearP, Vec3{0, 0, 0}) > 0, "origin in front of near plane");
        check(fr::SignedDistance(farP, Vec3{0, 0, 0}) > 0, "origin in front of far plane");
    }

    // ---- Hand-checked side planes. 90deg square frustum -> the 4 side planes are 45deg. The left
    // plane (toward -X) has inward normal pointing +X-ish; through the camera apex (0,0,5). For a
    // point on the -X edge at z=0 it should be ~on the plane. ----
    {
        const fr::Plane& L = f.planes[0];
        const fr::Plane& R = f.planes[1];
        const fr::Plane& B = f.planes[2];
        const fr::Plane& T = f.planes[3];
        // Inward normals point toward the frustum interior. X is NOT flipped by the projection, so
        // left points +X / right points -X. The Vulkan projection BAKES A Y-FLIP (Mat4::Perspective
        // m[5] = -1/tan), so the clip-space "bottom" plane (R3+R1) maps to the WORLD +Y half and the
        // clip "top" (R3-R1) to world -Y — i.e. the Y inward normals are flipped vs a no-flip proj.
        // (This is exactly the convention subtlety the test exists to pin; the functional checks
        // below — apex pass-through, origin-inside, and the brute-force partition — are what gate
        // correctness, and they confirm 0 false-culls.)
        check(L.n.x > 0, "left-plane inward normal points +X");
        check(R.n.x < 0, "right-plane inward normal points -X");
        check(B.n.y < 0, "bottom-clip-plane inward normal points -Y in world (Y-flip baked in proj)");
        check(T.n.y > 0, "top-clip-plane inward normal points +Y in world (Y-flip baked in proj)");
        // All four side planes pass through the apex (camera eye) (0,0,5): signedDistance ~ 0.
        check(approx(fr::SignedDistance(L, Vec3{0, 0, 5}), 0.0f, 1e-3f), "left plane passes through apex");
        check(approx(fr::SignedDistance(R, Vec3{0, 0, 5}), 0.0f, 1e-3f), "right plane passes through apex");
        check(approx(fr::SignedDistance(B, Vec3{0, 0, 5}), 0.0f, 1e-3f), "bottom plane passes through apex");
        check(approx(fr::SignedDistance(T, Vec3{0, 0, 5}), 0.0f, 1e-3f), "top plane passes through apex");
        // The origin (on the center axis at z=0) is inside all four side planes.
        check(fr::SignedDistance(L, Vec3{0, 0, 0}) > 0 && fr::SignedDistance(R, Vec3{0, 0, 0}) > 0 &&
              fr::SignedDistance(B, Vec3{0, 0, 0}) > 0 && fr::SignedDistance(T, Vec3{0, 0, 0}) > 0,
              "origin inside all four side planes");
    }

    // ---- SphereOutside hand cases. ----
    {
        // A small sphere at the origin (center of the frustum) is NOT outside.
        check(!fr::SphereOutside(f, Vec3{0, 0, 0}, 0.5f), "centered sphere is kept (inside)");
        // A sphere far to the left (-X = -100) is fully outside the left plane -> culled.
        check(fr::SphereOutside(f, Vec3{-100, 0, 0}, 1.0f), "far-left sphere is culled");
        // A sphere behind the camera (z = +50, behind near plane at z=4) is culled.
        check(fr::SphereOutside(f, Vec3{0, 0, 50}, 1.0f), "behind-camera sphere is culled");
        // A sphere beyond the far plane (z = -100) is culled.
        check(fr::SphereOutside(f, Vec3{0, 0, -100}, 1.0f), "beyond-far sphere is culled");
        // A sphere centered just OUTSIDE the left plane but with a radius large enough to poke back
        // in is CONSERVATIVELY kept (it straddles the plane). Place center 1 unit past where a
        // radius-2 sphere still reaches the interior.
        const fr::Plane& L = f.planes[0];
        // Pick a center 1 unit outside the left plane; radius 2 reaches across -> NOT culled.
        Vec3 onPlane{-4.0f, 0, 0};  // roughly near the left edge at z~? use plane math instead:
        (void)onPlane;
        // Construct a center exactly at signedDistance = -1 from L by stepping from origin along -n.
        Vec3 c = Vec3{0, 0, 0} + (L.n * (-(fr::SignedDistance(L, Vec3{0, 0, 0}) + 1.0f)));
        check(approx(fr::SignedDistance(L, c), -1.0f, 1e-3f), "constructed center is 1 outside L");
        check(!fr::SphereOutside(f, c, 2.0f), "straddling sphere (r>dist) is conservatively kept");
        check(fr::SphereOutside(f, c, 0.5f),  "sphere fully past L (r<dist) is culled");
    }

    // ---- AabbOutside p/n-vertex cases. ----
    {
        // Box at the origin is kept.
        check(!fr::AabbOutside(f, Vec3{-0.5f, -0.5f, -0.5f}, Vec3{0.5f, 0.5f, 0.5f}),
              "centered AABB is kept");
        // Box far to the left is culled (its positive vertex toward +X is still left of the plane).
        check(fr::AabbOutside(f, Vec3{-101, -1, -1}, Vec3{-99, 1, 1}), "far-left AABB is culled");
        // Box beyond the far plane is culled.
        check(fr::AabbOutside(f, Vec3{-1, -1, -101}, Vec3{1, 1, -99}), "beyond-far AABB is culled");
        // Box behind the camera is culled.
        check(fr::AabbOutside(f, Vec3{-1, -1, 49}, Vec3{1, 1, 51}), "behind-camera AABB is culled");
        // A LARGE box that straddles the left plane (spans from deep outside to inside) is kept:
        // its p-vertex toward +X is inside even though the n-vertex is far outside.
        check(!fr::AabbOutside(f, Vec3{-100, -1, -1}, Vec3{1, 1, 1}), "straddling AABB is kept");
    }

    // ---- Cull-partition vs the independent brute-force projection reference. ----
    // For many random sphere centers: if the projection reference says the CENTER is inside the
    // clip volume, our SphereOutside (with any radius>=0) MUST return false (keeping it). That is
    // the render-invariance contract: never false-cull a visible point. We also confirm we DO cull
    // points that project comfortably outside on all sides (so the test isn't trivially passing by
    // keeping everything).
    {
        std::mt19937 rng(0xC0FFEEu);
        std::uniform_real_distribution<float> ux(-30.0f, 30.0f);
        std::uniform_real_distribution<float> uz(-15.0f, 8.0f);
        int falseCulls = 0, culledCount = 0, total = 0;
        for (int i = 0; i < 20000; ++i) {
            Vec3 c{ux(rng), ux(rng), uz(rng)};
            bool refInside = RefPointInside(vp, c);
            // Treat the sphere as a point (radius 0) for the strict invariance check: a kept clip
            // point must never be culled.
            bool culledPoint = fr::SphereOutside(f, c, 0.0f);
            ++total;
            if (culledPoint) ++culledCount;
            if (refInside && culledPoint) ++falseCulls;
        }
        check(falseCulls == 0, "frustum never false-culls a point inside the clip volume");
        check(culledCount > 0, "frustum does cull genuinely-outside points (test is non-trivial)");
        std::printf("partition: %d total, %d culled, %d false-culls\n", total, culledCount, falseCulls);
    }

    // ---- Sphere variant of the partition: a sphere whose center+radius keeps it inside the clip
    // volume (sampled corner check) must never be culled. ----
    {
        std::mt19937 rng(0x1234u);
        std::uniform_real_distribution<float> ux(-20.0f, 20.0f);
        std::uniform_real_distribution<float> uz(-12.0f, 4.0f);
        std::uniform_real_distribution<float> ur(0.0f, 3.0f);
        int falseCulls = 0;
        for (int i = 0; i < 20000; ++i) {
            Vec3 c{ux(rng), ux(rng), uz(rng)};
            float r = ur(rng);
            // Reference "sphere overlaps frustum": ANY of the 6 axis-extreme points or the center
            // is inside the clip volume -> the sphere reaches on-screen, must be kept.
            bool refReaches = RefPointInside(vp, c);
            const Vec3 offs[6] = {{r,0,0},{-r,0,0},{0,r,0},{0,-r,0},{0,0,r},{0,0,-r}};
            for (const auto& o : offs) if (RefPointInside(vp, c + o)) refReaches = true;
            if (refReaches && fr::SphereOutside(f, c, r)) ++falseCulls;
        }
        check(falseCulls == 0, "frustum never culls a sphere that reaches the clip volume");
    }

    if (g_fail == 0) { std::printf("frustum_test: all checks passed\n"); return 0; }
    std::printf("frustum_test: %d FAILURES\n", g_fail);
    return 1;
}
