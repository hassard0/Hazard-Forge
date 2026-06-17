// Slice AJ — volumetric fog / light shafts. Pure CPU math: the Henyey-Greenstein phase function +
// the camera-basis world-ray reconstruction + Beer-Lambert transmittance. No device, ASan-eligible
// (links hf_core). Mirrors the math the --volumetric-shot showcase and volumetric.frag use
// (engine/render/volumetric.h).
#include "render/volumetric.h"
#include "math/math.h"
#include <cmath>
#include <cstdio>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::math;
namespace vol = hf::render::volumetric;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }

int main() {
    HF_TEST_MAIN_INIT();
    // ---- HG: at g=0 the phase is isotropic = 1/(4*pi) for every angle. ----
    {
        const float iso = 1.0f / (4.0f * vol::kPi);
        check(approx(vol::HenyeyGreenstein(1.0f, 0.0f), iso), "HG g=0 forward isotropic 1/4pi");
        check(approx(vol::HenyeyGreenstein(0.0f, 0.0f), iso), "HG g=0 side isotropic 1/4pi");
        check(approx(vol::HenyeyGreenstein(-1.0f, 0.0f), iso), "HG g=0 back isotropic 1/4pi");
    }

    // ---- HG: forward-scatter (g>0) peaks toward cosTheta=+1 and is weakest at cosTheta=-1. ----
    {
        const float g = 0.6f;
        float fwd  = vol::HenyeyGreenstein(1.0f, g);
        float side = vol::HenyeyGreenstein(0.0f, g);
        float back = vol::HenyeyGreenstein(-1.0f, g);
        check(fwd > side, "HG g=0.6 forward > side");
        check(side > back, "HG g=0.6 side > back");
        check(fwd > back, "HG g=0.6 forward peak > back");
        // Back-scatter (g<0) is the mirror image.
        check(vol::HenyeyGreenstein(-1.0f, -g) > vol::HenyeyGreenstein(1.0f, -g),
              "HG g=-0.6 back > forward");
    }

    // ---- HG integrates over the sphere to ~1 (normalised). Numerically integrate over solid angle:
    // integral of HG(cosTheta,g) dOmega = 2*pi * integral_{-1}^{1} HG(mu,g) dmu == 1. ----
    {
        const float g = 0.6f;
        const int N = 20000;
        double integral = 0.0;
        for (int k = 0; k < N; ++k) {
            float mu = -1.0f + (2.0f * (k + 0.5f) / (float)N);   // midpoint on [-1,1]
            integral += vol::HenyeyGreenstein(mu, g);
        }
        integral *= (2.0 / (double)N);          // dmu = 2/N
        integral *= 2.0 * (double)vol::kPi;     // azimuthal 2*pi
        check(std::fabs(integral - 1.0) < 1e-2, "HG integrates over the sphere to ~1");
    }

    // ---- World ray: a centered pixel (uv = 0.5) reconstructs to the camera forward direction. ----
    {
        Vec3 fwd{0.0f, 0.0f, -1.0f};
        Vec3 right{1.0f, 0.0f, 0.0f};
        Vec3 up{0.0f, 1.0f, 0.0f};
        const float tanHalf = std::tan(0.5f * 1.04719755f);
        const float aspect = 1280.0f / 720.0f;
        Vec3 r = vol::WorldRayUnnormalized(0.5f, 0.5f, fwd, right, up, tanHalf, aspect);
        check(approx(r.x, 0.0f) && approx(r.y, 0.0f) && approx(r.z, -1.0f),
              "centered pixel ray == camFwd");
        // The forward (camFwd) projection of rayU is unit, so a sample at view depth t sits at
        // viewPos + rayU*t with -z == t for this axis-aligned basis.
        Vec3 r2 = vol::WorldRayUnnormalized(0.5f, 0.5f, fwd, right, up, tanHalf, aspect);
        check(approx(-r2.z, 1.0f), "rayU has unit camFwd projection");
        // An off-center pixel tilts the ray toward +X (right of screen) / +Y (top of screen).
        Vec3 rr = vol::WorldRayUnnormalized(1.0f, 0.5f, fwd, right, up, tanHalf, aspect);
        check(rr.x > 0.0f, "right-edge pixel ray tilts +X");
        Vec3 rt = vol::WorldRayUnnormalized(0.5f, 0.0f, fwd, right, up, tanHalf, aspect);
        check(rt.y > 0.0f, "top-edge pixel (v=0) ray tilts +Y");
    }

    // ---- Beer-Lambert transmittance: 1 at distance 0, monotonically decreasing, in (0,1]. ----
    {
        check(approx(vol::Transmittance(0.0f, 0.04f), 1.0f), "transmittance(0) == 1");
        float a = vol::Transmittance(5.0f, 0.04f);
        float b = vol::Transmittance(10.0f, 0.04f);
        check(a < 1.0f && a > 0.0f, "transmittance(5) in (0,1)");
        check(b < a, "transmittance decreases with distance");
        check(approx(vol::Transmittance(10.0f, 0.0f), 1.0f), "zero extinction -> no attenuation");
    }

    if (g_fail == 0) std::printf("volumetric_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
