// Slice AP — temporal anti-aliasing. Pure CPU math (header-only, no device, no backend symbols).
// Mirrors the math the --taa-shot showcase and taa_resolve.frag use (engine/render/taa.h):
//   * Halton(base,index) low-discrepancy radical inverse (deterministic jitter source),
//   * Jitter(frameIndex,w,h) sub-pixel NDC offset (bounded, deterministic, run-to-run identical),
//   * ClipHistoryToNeighborhood = per-channel AABB clamp (exactly what the shader does),
//   * ResolveBlend = lerp(clampedHistory, current, alpha) (alpha=1 first frame, ~0.1 steady state).
// Pure C++ (hf_core), ASan-eligible like the other render-math tests.
#include "render/taa.h"
#include "math/math.h"
#include <cmath>
#include <cstdio>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::math;
namespace taa = hf::render::taa;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-5f) { return std::fabs(a - b) < eps; }

int main() {
    HF_TEST_MAIN_INIT();
    // ---- Halton base-2 radical inverse: the canonical van der Corput sequence. ----
    // index 1..6 of base 2 = 1/2, 1/4, 3/4, 1/8, 5/8, 3/8.
    {
        check(approx(taa::Halton(2, 1), 0.5f),   "Halton(2,1) == 1/2");
        check(approx(taa::Halton(2, 2), 0.25f),  "Halton(2,2) == 1/4");
        check(approx(taa::Halton(2, 3), 0.75f),  "Halton(2,3) == 3/4");
        check(approx(taa::Halton(2, 4), 0.125f), "Halton(2,4) == 1/8");
        check(approx(taa::Halton(2, 5), 0.625f), "Halton(2,5) == 5/8");
        check(approx(taa::Halton(2, 6), 0.375f), "Halton(2,6) == 3/8");
    }
    // ---- Halton base-3 radical inverse: index 1..4 of base 3 = 1/3, 2/3, 1/9, 4/9. ----
    {
        check(approx(taa::Halton(3, 1), 1.0f / 3.0f), "Halton(3,1) == 1/3");
        check(approx(taa::Halton(3, 2), 2.0f / 3.0f), "Halton(3,2) == 2/3");
        check(approx(taa::Halton(3, 3), 1.0f / 9.0f), "Halton(3,3) == 1/9");
        check(approx(taa::Halton(3, 4), 4.0f / 9.0f), "Halton(3,4) == 4/9");
    }
    // ---- Halton stays in [0,1) and is deterministic (same args -> same value, byte-equal). ----
    {
        bool inRange = true, deterministic = true;
        for (int i = 1; i <= 64; ++i) {
            float h2 = taa::Halton(2, i), h3 = taa::Halton(3, i);
            if (h2 < 0.0f || h2 >= 1.0f || h3 < 0.0f || h3 >= 1.0f) inRange = false;
            if (taa::Halton(2, i) != h2 || taa::Halton(3, i) != h3) deterministic = false;
        }
        check(inRange, "Halton(2/3, 1..64) stays in [0,1)");
        check(deterministic, "Halton is a pure deterministic function of its args");
    }

    const int W = 1280, H = 720;

    // ---- Jitter is a sub-pixel NDC offset: |offset.x| <= 1/W, |offset.y| <= 1/H. ----
    // Halton-0.5 is in [-0.5,0.5); *2/dim gives an NDC offset spanning at most one pixel.
    {
        bool bounded = true;
        for (int f = 0; f < 8; ++f) {
            taa::Vec2 j = taa::Jitter(f, W, H);
            // NDC half-pixel is 1/dim (NDC spans [-1,1] => 2/dim per pixel => +/- 1/dim sub-pixel).
            if (std::fabs(j.x) > 1.0f / (float)W + 1e-6f) bounded = false;
            if (std::fabs(j.y) > 1.0f / (float)H + 1e-6f) bounded = false;
        }
        check(bounded, "Jitter offset is within one sub-pixel (|x|<=1/W, |y|<=1/H)");
    }

    // ---- Jitter matches the documented formula exactly for frame 0 (Halton(.,1)). ----
    {
        taa::Vec2 j = taa::Jitter(0, W, H);
        float ex = (taa::Halton(2, 1) - 0.5f) * 2.0f / (float)W;   // (0.5-0.5)*2/W == 0
        float ey = (taa::Halton(3, 1) - 0.5f) * 2.0f / (float)H;   // (1/3-0.5)*2/H
        check(approx(j.x, ex), "Jitter(0).x matches (Halton(2,1)-0.5)*2/W");
        check(approx(j.y, ey), "Jitter(0).y matches (Halton(3,1)-0.5)*2/H");
        check(approx(j.x, 0.0f), "Jitter(0).x == 0 (Halton(2,1)==0.5)");
    }

    // ---- Jitter is deterministic + frame-dependent (distinct frames -> distinct offsets). ----
    {
        taa::Vec2 a = taa::Jitter(1, W, H);
        taa::Vec2 b = taa::Jitter(2, W, H);
        check(a.x != b.x || a.y != b.y, "different frame indices give different jitter");
        taa::Vec2 a2 = taa::Jitter(1, W, H);
        check(a.x == a2.x && a.y == a2.y, "same frame index gives identical jitter (deterministic)");
    }

    // ---- ClipHistoryToNeighborhood = per-channel clamp into [boxMin, boxMax]. ----
    {
        Vec3 boxMin{0.1f, 0.2f, 0.3f};
        Vec3 boxMax{0.6f, 0.7f, 0.8f};
        // A history sample entirely inside the box is unchanged.
        Vec3 inside{0.3f, 0.4f, 0.5f};
        Vec3 ci = taa::ClipHistoryToNeighborhood(inside, boxMin, boxMax);
        check(approx(ci.x, inside.x) && approx(ci.y, inside.y) && approx(ci.z, inside.z),
              "history inside the AABB is unchanged");
        // A history sample outside the box clamps per channel to the nearest face.
        Vec3 outside{-0.5f, 1.2f, 0.45f};
        Vec3 co = taa::ClipHistoryToNeighborhood(outside, boxMin, boxMax);
        check(approx(co.x, boxMin.x), "history below min clamps up to boxMin.x");
        check(approx(co.y, boxMax.y), "history above max clamps down to boxMax.y");
        check(approx(co.z, 0.45f),    "history inside on a channel is left alone (z)");
        // Clamped result is always within the AABB.
        bool within = co.x >= boxMin.x - 1e-6f && co.x <= boxMax.x + 1e-6f &&
                      co.y >= boxMin.y - 1e-6f && co.y <= boxMax.y + 1e-6f &&
                      co.z >= boxMin.z - 1e-6f && co.z <= boxMax.z + 1e-6f;
        check(within, "clamped history lands inside the neighborhood AABB");
    }

    // ---- ResolveBlend = lerp(clampedHistory, current, alpha). ----
    {
        Vec3 cur{1.0f, 0.0f, 0.5f};
        Vec3 hist{0.0f, 1.0f, 0.5f};
        // alpha == 1.0 (first frame): output is the current frame, history ignored.
        Vec3 r1 = taa::ResolveBlend(cur, hist, 1.0f);
        check(approx(r1.x, cur.x) && approx(r1.y, cur.y) && approx(r1.z, cur.z),
              "alpha=1 returns the current frame (first-frame start)");
        // alpha == 0.0: output is the (clamped) history unchanged.
        Vec3 r0 = taa::ResolveBlend(cur, hist, 0.0f);
        check(approx(r0.x, hist.x) && approx(r0.y, hist.y) && approx(r0.z, hist.z),
              "alpha=0 returns the history");
        // alpha == 0.1 steady state: 90% history + 10% current, per channel.
        Vec3 rs = taa::ResolveBlend(cur, hist, 0.1f);
        check(approx(rs.x, 0.1f * cur.x + 0.9f * hist.x), "alpha=0.1 lerps x (10% current)");
        check(approx(rs.y, 0.1f * cur.y + 0.9f * hist.y), "alpha=0.1 lerps y");
        check(approx(rs.z, 0.5f), "alpha=0.1 leaves an equal channel at 0.5");
    }

    if (g_fail == 0) std::printf("taa_test: all checks passed\n");
    else std::printf("taa_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
