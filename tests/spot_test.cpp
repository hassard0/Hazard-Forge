// Slice AE — spot light + perspective spot-light shadow. Pure CPU math: spotViewProj projection +
// cone attenuation. No device, ASan-eligible (links hf_core). Mirrors the math the --spot-shot
// showcase and lit_spot.frag use.
#include "render/spot.h"
#include "math/math.h"
#include <cmath>
#include <cstdio>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::math;
namespace spot = hf::render::spot;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-3f) { return std::fabs(a - b) < eps; }

// Project a world point through a clip-space matrix and divide by w.
static Vec3 project(const Mat4& m, const Vec3& p, float& outW) {
    return MulPointDivide(m, p, outW);
}

int main() {
    HF_TEST_MAIN_INIT();
    // Spot mounted at (0,10,0) aiming straight down -Y, 20deg outer half-angle.
    const Vec3  pos{0.0f, 10.0f, 0.0f};
    const Vec3  dir{0.0f, -1.0f, 0.0f};
    const float inner = 0.20f;          // ~11.5deg
    const float outer = 0.349066f;      // 20deg
    const float nearZ = 0.5f;
    const float range = 30.0f;

    Mat4 vp = spot::SpotViewProj(pos, dir, outer, nearZ, range);

    // ---- A point at cone center + mid-range projects near clip center, depth in [0,1]. ----
    {
        // 12 units down the axis from the light (well within range, on-axis).
        Vec3 center = pos + dir * 12.0f;   // (0,-2,0)
        float w = 0.0f;
        Vec3 ndc = project(vp, center, w);
        check(w > 0.0f, "on-axis point in front of light (w>0)");
        check(approx(ndc.x, 0.0f, 2e-3f), "on-axis projects to clip center x~0");
        check(approx(ndc.y, 0.0f, 2e-3f), "on-axis projects to clip center y~0");
        check(ndc.z >= 0.0f - 1e-3f && ndc.z <= 1.0f + 1e-3f, "on-axis depth in [0,1]");
    }

    // ---- A point well OUTSIDE the cone projects outside [-1,1] (off the shadow map). ----
    {
        // Same depth down the axis, but pushed far sideways (~45deg off axis >> 20deg outer).
        Vec3 outside = pos + dir * 12.0f + Vec3{12.0f, 0.0f, 0.0f};
        float w = 0.0f;
        Vec3 ndc = project(vp, outside, w);
        bool outOfFrustum = (ndc.x < -1.0f || ndc.x > 1.0f || ndc.y < -1.0f || ndc.y > 1.0f);
        check(outOfFrustum, "far-off-axis point projects outside the [-1,1] shadow frustum");
    }

    // ---- A point at the cone EDGE (== outer half-angle) lands ~at the frustum edge (|ndc|~1). ----
    {
        // At distance d down the axis, the outer cone edge is at lateral offset d*tan(outer).
        float d = 12.0f;
        float lateral = d * std::tan(outer);
        Vec3 edge = pos + dir * d + Vec3{lateral, 0.0f, 0.0f};
        float w = 0.0f;
        Vec3 ndc = project(vp, edge, w);
        check(approx(std::fabs(ndc.x), 1.0f, 5e-3f), "cone-edge point lands at frustum edge |ndc.x|~1");
        check(std::fabs(ndc.y) < 1e-2f, "cone-edge point on the X meridian has ndc.y~0");
    }

    // ---- Cone attenuation: 1 at center, smooth to 0 past the outer angle. ----
    {
        // dirToFrag straight down the axis -> on-axis -> attenuation 1.
        check(approx(spot::ConeAttenuation(dir, dir, inner, outer), 1.0f), "cone att == 1 on-axis");
        // A direction just inside the inner cone is still full brightness.
        Vec3 nearAxis = normalize(Vec3{std::sin(inner * 0.5f), -std::cos(inner * 0.5f), 0.0f});
        check(approx(spot::ConeAttenuation(dir, nearAxis, inner, outer), 1.0f),
              "cone att == 1 inside inner cone");
        // A direction past the outer cone is fully dark.
        Vec3 pastOuter = normalize(Vec3{std::sin(outer + 0.1f), -std::cos(outer + 0.1f), 0.0f});
        check(approx(spot::ConeAttenuation(dir, pastOuter, inner, outer), 0.0f),
              "cone att == 0 past outer cone");
        // A direction exactly between inner and outer is partial (0,1) and monotone.
        float mid = 0.5f * (inner + outer);
        Vec3 between = normalize(Vec3{std::sin(mid), -std::cos(mid), 0.0f});
        float a = spot::ConeAttenuation(dir, between, inner, outer);
        check(a > 0.0f && a < 1.0f, "cone att in (0,1) between inner and outer");
    }

    if (g_fail == 0) std::printf("spot_test: all checks passed\n");
    else std::printf("spot_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
