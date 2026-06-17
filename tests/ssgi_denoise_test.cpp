// Slice BR — SSGI spatial denoise. Pure CPU math (engine/render/ssgi.h, header-only): the
// edge-stopping BilateralWeight tap weight, a CPU bilateral-blur mini-model (variance reduction +
// mean preservation on a uniform-surface field, and EDGE PRESERVATION across a depth step), and
// determinism. No device, ASan-eligible (links hf_core). Mirrors the math the --ssgi-denoise-shot
// showcase and ssgi_denoise.frag use.
#include "render/ssgi.h"
#include "math/math.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::math;
namespace ssgi = hf::render::ssgi;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }

static float Mean(const std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    double s = 0.0; for (float x : v) s += x;
    return (float)(s / (double)v.size());
}
static float Variance(const std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    float m = Mean(v);
    double s = 0.0; for (float x : v) { double d = x - m; s += d * d; }
    return (float)(s / (double)v.size());
}

int main() {
    HF_TEST_MAIN_INIT();
    const float spatialSigma = 2.0f;
    const float depthSigma   = 0.5f;
    const float normalPower  = 16.0f;
    const Vec3  up{0, 1, 0};

    // ---- BilateralWeight at the CENTER: zero spatial dist, equal depth, equal normal -> 1. ----
    {
        float w = ssgi::BilateralWeight(0.0f, spatialSigma, 1.0f, 1.0f, depthSigma, up, up, normalPower);
        check(approx(w, 1.0f), "BilateralWeight == 1 at the center (no spatial/depth/normal diff)");
    }

    // ---- Decays with spatial distance (depth+normal held equal). Strictly decreasing. ----
    {
        float prev = 2.0f;
        for (float d2 : {0.0f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f}) {
            float w = ssgi::BilateralWeight(d2, spatialSigma, 1.0f, 1.0f, depthSigma, up, up, normalPower);
            check(w <= prev + 1e-7f, "BilateralWeight is non-increasing in spatial distance");
            check(w > 0.0f, "BilateralWeight stays positive for equal depth/normal");
            prev = w;
        }
        // A clearly smaller weight far from the center than at it.
        float wc = ssgi::BilateralWeight(0.0f, spatialSigma, 1.0f, 1.0f, depthSigma, up, up, normalPower);
        float wf = ssgi::BilateralWeight(16.0f, spatialSigma, 1.0f, 1.0f, depthSigma, up, up, normalPower);
        check(wf < wc, "BilateralWeight: far tap weighs strictly less than the center");
    }

    // ---- ~0 across a LARGE depth difference (edge-stop). ----
    {
        // depthDiff = 5 view units, depthSigma 0.5 -> exp(-25/(2*0.25)) ~ exp(-50) ~ 0.
        float w = ssgi::BilateralWeight(0.0f, spatialSigma, 1.0f, 6.0f, depthSigma, up, up, normalPower);
        check(w < 1e-6f, "BilateralWeight ~0 across a large depth difference (depth edge-stop)");
        // Monotone in |depth diff|: bigger diff -> smaller weight.
        float prev = 2.0f;
        for (float dd : {0.0f, 0.25f, 0.5f, 1.0f, 2.0f}) {
            float w2 = ssgi::BilateralWeight(0.0f, spatialSigma, 1.0f, 1.0f + dd, depthSigma, up, up, normalPower);
            check(w2 <= prev + 1e-7f, "BilateralWeight is non-increasing in |depth diff|");
            prev = w2;
        }
    }

    // ---- ~0 across OPPOSING normals (dot <= 0 -> the pow term is 0). ----
    {
        Vec3 down{0, -1, 0};
        float w = ssgi::BilateralWeight(0.0f, spatialSigma, 1.0f, 1.0f, depthSigma, up, down, normalPower);
        check(approx(w, 0.0f), "BilateralWeight == 0 across opposing normals (dot <= 0)");
        Vec3 ortho{1, 0, 0};
        float wo = ssgi::BilateralWeight(0.0f, spatialSigma, 1.0f, 1.0f, depthSigma, up, ortho, normalPower);
        check(approx(wo, 0.0f), "BilateralWeight == 0 across orthogonal normals (dot == 0)");
        // Monotone in normal divergence: as the tap normal tilts away from center, weight drops.
        float prev = 2.0f;
        for (float ang : {0.0f, 0.2f, 0.4f, 0.8f, 1.2f}) {
            Vec3 n = normalize(Vec3{std::sin(ang), std::cos(ang), 0.0f});  // tilt away from +Y
            float w2 = ssgi::BilateralWeight(0.0f, spatialSigma, 1.0f, 1.0f, depthSigma, up, n, normalPower);
            check(w2 <= prev + 1e-7f, "BilateralWeight is non-increasing in normal divergence");
            prev = w2;
        }
    }

    // ---- Blur sanity: a noisy UNIFORM-depth/normal field -> reduced variance + preserved mean. ----
    {
        const int W = 16, H = 16;
        std::vector<float> field(W * H), depth(W * H, 1.0f);
        std::vector<Vec3>  normals(W * H, up);
        // Deterministic high-frequency "noise" around 0.5 (a fixed checkerboard-ish pattern, no RNG).
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float n = (((x * 7 + y * 13) % 5) - 2) * 0.15f;  // in [-0.30, +0.30]
                field[y * W + x] = 0.5f + n;
            }
        ssgi::SsgiDenoiseParams p;  // defaults (radius 2 etc.)
        std::vector<float> blurred = ssgi::BilateralDenoiseScalar(field, depth, normals, W, H, p);

        check(Variance(blurred) < Variance(field) * 0.6f,
              "bilateral blur reduces variance on a uniform-surface noisy field (smoother)");
        check(approx(Mean(blurred), Mean(field), 5e-3f),
              "bilateral blur preserves the mean on a uniform-surface field");
    }

    // ---- Edge preservation: a depth EDGE down the middle -> blur does NOT mix across it; both
    // sides keep their distinct values. Left side value 0.2 @ depth 1; right side 0.9 @ depth 10. ----
    {
        const int W = 16, H = 8;
        std::vector<float> field(W * H), depth(W * H);
        std::vector<Vec3>  normals(W * H, up);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                bool left = (x < W / 2);
                field[y * W + x] = left ? 0.2f : 0.9f;
                depth[y * W + x] = left ? 1.0f : 10.0f;   // a big depth step at the seam
            }
        ssgi::SsgiDenoiseParams p;
        std::vector<float> blurred = ssgi::BilateralDenoiseScalar(field, depth, normals, W, H, p);

        // Pixels adjacent to the seam on each side keep their own side's value (no cross-mixing).
        int yMid = H / 2;
        float leftEdge  = blurred[yMid * W + (W / 2 - 1)];
        float rightEdge = blurred[yMid * W + (W / 2)];
        check(approx(leftEdge,  0.2f, 1e-3f), "depth-edge: left side stays 0.2 at the seam (no cross-blur)");
        check(approx(rightEdge, 0.9f, 1e-3f), "depth-edge: right side stays 0.9 at the seam (no cross-blur)");
        // The two sides remain clearly distinct everywhere (blur didn't average them to ~0.55).
        for (int y = 0; y < H; ++y) {
            check(blurred[y * W + 0] < 0.5f, "depth-edge: far-left column stays low");
            check(blurred[y * W + (W - 1)] > 0.5f, "depth-edge: far-right column stays high");
        }
    }

    // ---- Normal edge preservation: a NORMAL seam (left faces +Y, right faces +X, uniform depth)
    // also blocks cross-blur even though depth is continuous. ----
    {
        const int W = 16, H = 8;
        std::vector<float> field(W * H), depth(W * H, 1.0f);
        std::vector<Vec3>  normals(W * H);
        Vec3 nx{1, 0, 0};
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                bool left = (x < W / 2);
                field[y * W + x]   = left ? 0.2f : 0.9f;
                normals[y * W + x] = left ? up : nx;   // 90-degree normal seam, dot == 0
            }
        ssgi::SsgiDenoiseParams p;
        std::vector<float> blurred = ssgi::BilateralDenoiseScalar(field, depth, normals, W, H, p);
        int yMid = H / 2;
        check(approx(blurred[yMid * W + (W / 2 - 1)], 0.2f, 1e-3f),
              "normal-edge: left side stays 0.2 at the seam (orthogonal normals block blur)");
        check(approx(blurred[yMid * W + (W / 2)], 0.9f, 1e-3f),
              "normal-edge: right side stays 0.9 at the seam");
    }

    // ---- Determinism: same input -> byte-identical output. ----
    {
        const int W = 12, H = 12;
        std::vector<float> field(W * H), depth(W * H, 1.0f);
        std::vector<Vec3>  normals(W * H, up);
        for (int i = 0; i < W * H; ++i) field[i] = 0.5f + (((i * 11) % 7) - 3) * 0.07f;
        ssgi::SsgiDenoiseParams p;
        std::vector<float> a = ssgi::BilateralDenoiseScalar(field, depth, normals, W, H, p);
        std::vector<float> b = ssgi::BilateralDenoiseScalar(field, depth, normals, W, H, p);
        bool identical = (a.size() == b.size());
        for (size_t i = 0; i < a.size() && identical; ++i) identical = (a[i] == b[i]);
        check(identical, "BilateralDenoiseScalar is deterministic (byte-identical across runs)");
    }

    // ---- Default params are the documented showcase values. ----
    {
        ssgi::SsgiDenoiseParams p;
        check(p.radius == 2, "default denoise radius == 2 (5x5 window)");
        check(approx(p.spatialSigma, 2.0f), "default spatialSigma == 2.0");
        check(approx(p.depthSigma, 0.5f), "default depthSigma == 0.5");
        check(approx(p.normalPower, 16.0f), "default normalPower == 16.0");
    }

    if (g_fail == 0) { std::printf("ssgi_denoise_test: all checks passed\n"); return 0; }
    std::printf("ssgi_denoise_test: %d FAILURES\n", g_fail);
    return 1;
}
