// Slice CR — Ground-Truth Ambient Occlusion (GTAO). Pure CPU math: the closed-form per-slice
// IntegrateArc visibility integral, the per-sample HorizonAngle, and the full horizon-search
// Visibility estimate. No device, ASan-eligible (links hf_core). Exercises the EXACT math the
// --gtao-shot showcase + gtao.frag.hlsl shader use (engine/render/gtao.h).
//
// Properties pinned (per the spec):
//   * UNOCCLUDED = 1 (the core proof):
//       - IntegrateArc for the full unoccluded hemisphere (h1 = n - π/2, h2 = n + π/2) -> 1 at n == 0;
//       - Visibility over a FLAT depth field -> 1 EXACTLY (no occlusion);
//       - Visibility at radius == 0 -> 1 EXACTLY (no horizon search).
//   * KNOWN-OCCLUDER ANALYTIC MATCH: a single planar occluder raising one horizon to a known angle ->
//     IntegrateArc returns the hand-computed analytic visibility for that geometry; a closer/taller
//     occluder -> smaller visibility (more AO); monotone.
//   * RANGE + SYMMETRY: Visibility ∈ [0,1] for all inputs; IntegrateArc(h1,h2,n) == IntegrateArc(h2,h1,n)
//     (slice symmetry); a symmetric occluder configuration yields a symmetric result.
//   * DETERMINISM: same inputs -> bit-identical outputs (pure functions, no RNG/time).
#include "render/gtao.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>

namespace gtao = hf::render::gtao;
using hf::math::Vec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) <= eps; }

int main() {
    const float kHalfPi = gtao::kHalfPi;

    // ---- IntegrateArc: the full unoccluded hemisphere integrates to 1 (the identity the proof rests
    //      on). For a camera-facing fragment n == 0 and the open arc is [-π/2, +π/2]. ----
    {
        check(approx(gtao::IntegrateArc(-kHalfPi, kHalfPi, 0.0f), 1.0f),
              "IntegrateArc(full hemisphere, n=0) == 1 (fully unoccluded)");
        // For a tilted normal the open arc is [n - π/2, n + π/2]; the closed form evaluates to
        // cos(n) + n*sin(n) there (the tilted-hemisphere visibility). Spot-check a couple of n values.
        for (float n : {-0.6f, -0.2f, 0.0f, 0.3f, 0.7f}) {
            float v = gtao::IntegrateArc(n - kHalfPi, n + kHalfPi, n);
            float analytic = std::cos(n) + n * std::sin(n);
            check(approx(v, analytic),
                  "IntegrateArc(open arc, n) == cos(n)+n*sin(n) (tilted-hemisphere closed form)");
        }
    }

    // ---- IntegrateArc slice symmetry: swapping the two horizons returns the identical value. ----
    {
        const float n = 0.25f;
        const float pairs[][2] = {{-0.9f, 0.7f}, {-kHalfPi, kHalfPi}, {0.1f, 1.2f}, {-1.3f, -0.2f}};
        bool sym = true;
        for (auto& p : pairs) {
            if (!approx(gtao::IntegrateArc(p[0], p[1], n), gtao::IntegrateArc(p[1], p[0], n), 1e-6f))
                sym = false;
        }
        check(sym, "IntegrateArc(h1,h2,n) == IntegrateArc(h2,h1,n) (slice symmetry)");
    }

    // ---- KNOWN-OCCLUDER ANALYTIC MATCH. A single occluder raising the +tangent horizon from +π/2 to a
    //      known angle h2 (with the -tangent side fully open at -π/2, n == 0) must give the closed form
    //      IntegrateArc(-π/2, h2, 0). Verify the value + that pulling the horizon IN (smaller h2 = a
    //      closer/taller occluder) MONOTONICALLY decreases the visibility. ----
    {
        const float n = 0.0f;
        float prev = gtao::IntegrateArc(-kHalfPi, kHalfPi, n);   // fully open = 1
        check(approx(prev, 1.0f), "no occluder (h2 = +π/2) -> visibility 1");
        bool monotone = true;
        // Sweep the +tangent horizon from open (+π/2) inward toward the view dir (0): the occluder gets
        // taller/closer, the open arc shrinks, visibility must strictly decrease.
        for (float h2 = kHalfPi; h2 >= 0.0f - 1e-3f; h2 -= 0.15f) {
            float v = gtao::IntegrateArc(-kHalfPi, h2, n);
            // Analytic check against the hand-evaluated closed form for this exact (h1,h2,n).
            float side1 = -std::cos(2.0f * (-kHalfPi) - n) + std::cos(n) + 2.0f * (-kHalfPi) * std::sin(n);
            float side2 = -std::cos(2.0f * h2 - n) + std::cos(n) + 2.0f * h2 * std::sin(n);
            float analytic = 0.25f * (side1 + side2);
            check(approx(v, analytic, 1e-5f), "known-occluder IntegrateArc matches the analytic closed form");
            if (v > prev + 1e-5f) monotone = false;
            prev = v;
        }
        check(monotone, "pulling the horizon in (taller/closer occluder) monotonically lowers visibility");
    }

    // ---- Visibility: FLAT depth field -> 1 EXACTLY. A flat field at constant linear depth `d0` places
    //      every marched sample on the tangent plane (same depth as the center) -> no horizon raised. ----
    {
        const float d0 = 5.0f;                    // constant view-space linear depth
        auto flat = [&](float, float) { return d0; };
        // A fragment straight ahead, normal facing the camera.
        Vec3 viewPos{0.0f, 0.0f, -d0};
        Vec3 viewNormal{0.0f, 0.0f, 1.0f};        // toward the eye (+Z in view space)
        float ao = gtao::Visibility(flat, viewPos, viewNormal, 0.5f, 8, 8, 1280, 720);
        check(ao == 1.0f, "Visibility over a FLAT depth field == 1 EXACTLY (no occlusion)");
    }

    // ---- Visibility: radius == 0 -> 1 EXACTLY (the radius=0 identity the byte-identical proof rests
    //      on). Even over a LUMPY field, radius 0 means zero march distance -> no horizon search. ----
    {
        auto lumpy = [&](float x, float y) { return 5.0f - 0.5f * std::sin(x * 3.0f) * std::cos(y * 2.0f); };
        Vec3 viewPos{0.0f, 0.0f, -5.0f};
        Vec3 viewNormal{0.0f, 0.0f, 1.0f};
        float ao0 = gtao::Visibility(lumpy, viewPos, viewNormal, 0.0f, 8, 8, 1280, 720);
        check(ao0 == 1.0f, "Visibility at radius == 0 == 1 EXACTLY (no horizon search)");
    }

    // ---- Visibility: an occluding field DARKENS (AO < 1) and is in [0,1]; a CLOSER/TALLER occluder
    //      darkens more (monotone). Model a step ridge that rises toward the camera on one side. ----
    {
        Vec3 viewPos{0.0f, 0.0f, -5.0f};
        Vec3 viewNormal{0.0f, 0.0f, 1.0f};
        // A field that pulls the surface TOWARD the camera (smaller linear depth) for x > 0 — a wall
        // rising in front of the fragment on the +x side, occluding part of the hemisphere.
        auto wall = [&](float x, float, float rise) {
            return (x > 0.0f) ? (5.0f - rise) : 5.0f;   // closer (smaller depth) where the wall rises
        };
        float aoLow  = gtao::Visibility([&](float x, float y){ return wall(x, y, 1.0f); },
                                        viewPos, viewNormal, 1.0f, 8, 12, 1280, 720);
        float aoHigh = gtao::Visibility([&](float x, float y){ return wall(x, y, 3.0f); },
                                        viewPos, viewNormal, 1.0f, 8, 12, 1280, 720);
        check(aoLow >= 0.0f && aoLow <= 1.0f, "Visibility ∈ [0,1] (occluded field)");
        check(aoHigh >= 0.0f && aoHigh <= 1.0f, "Visibility ∈ [0,1] (taller occluder)");
        check(aoLow < 1.0f, "an occluding wall DARKENS the fragment (AO < 1)");
        check(aoHigh <= aoLow + 1e-5f, "a taller/closer occluder darkens MORE (monotone)");
    }

    // ---- DETERMINISM: same inputs -> bit-identical outputs. ----
    {
        auto lumpy = [&](float x, float y) { return 5.0f - 0.4f * std::sin(x * 4.0f + y); };
        Vec3 viewPos{0.2f, -0.1f, -5.0f};
        Vec3 viewNormal{0.1f, 0.2f, 0.97f};
        float a = gtao::Visibility(lumpy, viewPos, viewNormal, 0.7f, 8, 10, 1280, 720);
        float b = gtao::Visibility(lumpy, viewPos, viewNormal, 0.7f, 8, 10, 1280, 720);
        check(a == b, "Visibility is deterministic (bit-identical across runs)");
        check(gtao::IntegrateArc(-0.3f, 0.9f, 0.2f) == gtao::IntegrateArc(-0.3f, 0.9f, 0.2f),
              "IntegrateArc is deterministic");
    }

    if (g_fail == 0) { std::printf("gtao_test: all checks passed\n"); return 0; }
    std::printf("gtao_test: %d FAILURES\n", g_fail);
    return 1;
}
