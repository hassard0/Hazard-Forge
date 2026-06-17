// Slice BH — screen-space projected decals. Pure CPU math: the decal-box local<->world transform
// round-trip, the inside-unit-box test, the top-down UV projection, and the edge fade. No device,
// ASan-eligible (links hf_core). Mirrors the math the --decal-shot showcase and decal.frag use
// (engine/render/decal.h), exactly as ssr_test pins engine/render/ssr.h for the SSR pass.
#include "render/decal.h"
#include "math/math.h"
#include <cmath>
#include <cstdio>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::math;
namespace decal = hf::render::decal;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-3f) { return std::fabs(a - b) < eps; }

int main() {
    HF_TEST_MAIN_INIT();
    // ---- BuildDecalTransform + WorldToDecalLocal round-trip (axis-aligned box). ----
    {
        Vec3 center{2.0f, 0.0f, -1.5f};
        Vec3 half{1.5f, 0.5f, 2.0f};
        Vec3 rot{0.0f, 0.0f, 0.0f};
        Mat4 localToWorld = decal::BuildDecalTransform(center, half, rot);
        Mat4 worldToDecal = localToWorld.Inverse();

        // Box center -> local origin.
        Vec3 lc = decal::WorldToDecalLocal(center, worldToDecal);
        check(approx(lc.x, 0.0f) && approx(lc.y, 0.0f) && approx(lc.z, 0.0f),
              "center maps to local origin");

        // center + R*halfExtents (R == identity here) -> local corner (0.5,0.5,0.5).
        Vec3 cornerW = center + half;   // R = I
        Vec3 lcorner = decal::WorldToDecalLocal(cornerW, worldToDecal);
        check(approx(lcorner.x, 0.5f) && approx(lcorner.y, 0.5f) && approx(lcorner.z, 0.5f),
              "center+halfExtents maps to local corner (0.5,0.5,0.5)");

        // Opposite corner -> (-0.5,-0.5,-0.5).
        Vec3 ncorner = decal::WorldToDecalLocal(center - half, worldToDecal);
        check(approx(ncorner.x, -0.5f) && approx(ncorner.y, -0.5f) && approx(ncorner.z, -0.5f),
              "center-halfExtents maps to local (-0.5,-0.5,-0.5)");
    }

    // ---- InsideUnitBox: center true, just-outside-face false, corner boundary-inclusive true. ----
    {
        check(decal::InsideUnitBox(Vec3{0.0f, 0.0f, 0.0f}), "center is inside the unit box");
        check(!decal::InsideUnitBox(Vec3{0.5001f, 0.0f, 0.0f}), "just outside +x face is outside");
        check(!decal::InsideUnitBox(Vec3{0.0f, -0.5001f, 0.0f}), "just outside -y face is outside");
        check(decal::InsideUnitBox(Vec3{0.5f, 0.5f, 0.5f}), "corner (0.5,0.5,0.5) is boundary-inclusive");
        check(decal::InsideUnitBox(Vec3{-0.5f, -0.5f, -0.5f}), "corner (-0.5,-0.5,-0.5) is inclusive");
    }

    // ---- DecalUV: top-down projection along local -Y => uv = local.xz + 0.5. ----
    {
        Vec2 a = decal::DecalUV(Vec3{-0.5f, 0.3f, -0.5f});
        check(approx(a.x, 0.0f) && approx(a.y, 0.0f), "(-0.5,*,-0.5) -> uv (0,0)");
        Vec2 b = decal::DecalUV(Vec3{0.5f, -0.1f, 0.5f});
        check(approx(b.x, 1.0f) && approx(b.y, 1.0f), "(0.5,*,0.5) -> uv (1,1)");
        Vec2 c = decal::DecalUV(Vec3{0.0f, 0.49f, 0.0f});
        check(approx(c.x, 0.5f) && approx(c.y, 0.5f), "origin -> uv (0.5,0.5)");
        // UV is independent of the projection (local Y) axis (top-down).
        Vec2 d0 = decal::DecalUV(Vec3{0.2f, -0.4f, -0.1f});
        Vec2 d1 = decal::DecalUV(Vec3{0.2f,  0.4f, -0.1f});
        check(approx(d0.x, d1.x) && approx(d0.y, d1.y), "UV independent of projection (Y) axis");
    }

    // ---- EdgeFade: 1 at center, ->0 approaching a face, monotonic along an axis. ----
    {
        const float fade = 0.2f;
        check(approx(decal::EdgeFade(Vec3{0.0f, 0.0f, 0.0f}, fade), 1.0f), "edge fade is 1 at center");
        check(approx(decal::EdgeFade(Vec3{0.5f, 0.0f, 0.0f}, fade), 0.0f), "edge fade is 0 at +x face");
        check(approx(decal::EdgeFade(Vec3{0.0f, 0.0f, 0.5f}, fade), 0.0f), "edge fade is 0 at +z face");
        // Monotonic non-increasing as we approach the +x face along x.
        float prev = 1.0f;
        bool mono = true;
        for (int s = 0; s <= 10; ++s) {
            float x = (float)s / 10.0f * 0.5f;       // 0 .. 0.5
            float f = decal::EdgeFade(Vec3{x, 0.0f, 0.0f}, fade);
            if (f > prev + 1e-4f) mono = false;
            prev = f;
        }
        check(mono, "edge fade is monotonic non-increasing toward a face");
    }

    // ---- A ROTATED decal: a point on the rotated face maps inside; an outside point maps outside. ----
    {
        Vec3 center{0.0f, 0.0f, 0.0f};
        Vec3 half{1.0f, 0.25f, 1.0f};
        Vec3 rot{0.0f, 0.7853981634f, 0.0f};        // 45 deg about Y
        Mat4 localToWorld = decal::BuildDecalTransform(center, half, rot);
        Mat4 worldToDecal = localToWorld.Inverse();

        // World point on the rotated +x face: rotate (half.x,0,0) by RotateY(45deg).
        Mat4 R = Mat4::RotateY(rot.y);
        Vec3 faceW = center + MulPoint(R, Vec3{half.x, 0.0f, 0.0f});
        Vec3 faceL = decal::WorldToDecalLocal(faceW, worldToDecal);
        check(decal::InsideUnitBox(faceL), "rotated: point on the rotated +x face is inside");
        check(approx(faceL.x, 0.5f), "rotated: that face point maps to local x = 0.5");

        // A world point 0.5 past that same rotated face is outside.
        Vec3 outW = center + MulPoint(R, Vec3{half.x * 1.6f, 0.0f, 0.0f});
        Vec3 outL = decal::WorldToDecalLocal(outW, worldToDecal);
        check(!decal::InsideUnitBox(outL), "rotated: a point past the rotated face is outside");

        // The corner along the rotated diagonal still maps to the local corner.
        Vec3 cW = center + MulPoint(R, Vec3{half.x, half.y, half.z});
        Vec3 cL = decal::WorldToDecalLocal(cW, worldToDecal);
        check(approx(cL.x, 0.5f) && approx(cL.y, 0.5f) && approx(cL.z, 0.5f),
              "rotated: rotated corner maps to local (0.5,0.5,0.5)");
    }

    if (g_fail == 0) std::printf("decal_test: all checks passed\n");
    else std::printf("decal_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
