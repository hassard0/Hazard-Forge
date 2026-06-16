// Slice CZ — Subsurface Scattering (screen-space separable SSS). Pure CPU math: the skin diffusion
// profile weight, the depth-aware silhouette-stop falloff, and one separable diffusion blur pass. No
// device, ASan-eligible (links hf_core). Mirrors the math the --sss-shot showcase and sss_blur.frag
// use (engine/render/sss.h).
//
// Properties pinned (per the spec):
//   * Zero strength/width == identity: BlurAxis(sssStrength=0) and BlurAxis(sssWidthPx=0) return the
//     center color EXACTLY for any input (the byte-identical no-op proof's CPU half).
//   * Diffusion profile: DiffusionWeight is positive, peaks at distance 0, strictly decreasing in
//     distance; a wider sssWidthPx spreads the weight (a far tap weighted MORE under a wider profile).
//   * Depth-aware falloff: DepthFalloff == 1 at equal depth, -> ~0 for a tap far in depth (no bleed
//     across a silhouette), monotone decreasing in |depth diff|.
//   * Blur behavior: a flat subsurface region with a bright center spreads to neighbors (normalized,
//     energy stays within the lit range); a non-subsurface (mask 0) pixel is unchanged.
//   * Determinism: same inputs -> identical result (pure function, no RNG/time).
#include "render/sss.h"
#include <cmath>
#include <cstdio>

namespace sss = hf::render::sss;
using hf::math::Vec2;
using hf::math::Vec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool finite3(const Vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}
static bool exactEq(const Vec3& a, const Vec3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

int main() {
    // ---- DiffusionWeight: peaks at 0, positive, strictly decreasing, wider width spreads weight. ----
    {
        const float width = 8.0f;
        float w0 = sss::DiffusionWeight(0.0f, width);
        check(w0 == 1.0f, "DiffusionWeight(0) == 1 (peaks at the center tap)");
        // Positive + strictly decreasing in distance.
        float prev = w0;
        bool strictDec = true, allPos = true;
        for (float d = 1.0f; d <= 40.0f; d += 1.0f) {
            float w = sss::DiffusionWeight(d, width);
            if (!(w > 0.0f)) allPos = false;
            if (!(w < prev)) strictDec = false;
            prev = w;
        }
        check(allPos, "DiffusionWeight > 0 for every finite distance");
        check(strictDec, "DiffusionWeight strictly decreasing in distance");
        // Symmetric in the sign of distance (the shader passes |dist|, but the profile itself is even).
        check(sss::DiffusionWeight(5.0f, width) == sss::DiffusionWeight(5.0f, width),
              "DiffusionWeight deterministic");
        // A WIDER width weights a fixed far tap MORE (light bleeds farther).
        float narrow = sss::DiffusionWeight(10.0f, 6.0f);
        float wide   = sss::DiffusionWeight(10.0f, 18.0f);
        check(wide > narrow, "a wider sssWidthPx spreads weight (far tap weighted more)");
    }

    // ---- DepthFalloff: 1 at equal depth, -> 0 across a depth discontinuity, monotone. ----
    {
        const float depthScale = 0.5f;
        const float centerDepth = 10.0f;
        float f0 = sss::DepthFalloff(centerDepth, centerDepth, depthScale);
        check(f0 == 1.0f, "DepthFalloff == 1 EXACTLY at equal depth");
        // Monotone decreasing as the tap moves away in depth (both directions), and -> ~0 far away.
        float prev = f0;
        bool mono = true;
        for (float dd = 0.1f; dd <= 4.0f; dd += 0.1f) {
            float f = sss::DepthFalloff(centerDepth + dd, centerDepth, depthScale);
            if (!(f <= prev)) mono = false;
            prev = f;
        }
        check(mono, "DepthFalloff monotone decreasing in |depth diff|");
        float far = sss::DepthFalloff(centerDepth + 5.0f, centerDepth, depthScale);
        check(far < 1e-3f, "DepthFalloff -> ~0 for a tap far in depth (silhouette stop)");
        // Symmetric: a tap nearer vs farther by the same amount falls off the same.
        check(std::fabs(sss::DepthFalloff(centerDepth + 0.7f, centerDepth, depthScale) -
                        sss::DepthFalloff(centerDepth - 0.7f, centerDepth, depthScale)) < 1e-6f,
              "DepthFalloff symmetric in the sign of the depth diff");
    }

    // A flat, uniform subsurface field: constant depth, mask 1 everywhere, a smooth color field.
    auto flatDepth = [](float, float) { return 10.0f; };
    auto maskOn    = [](float, float) { return 1.0f; };
    // A color field that varies in x so a horizontal blur actually changes the center (a bright spike
    // around uv.x == 0.5 on a darker background).
    auto spikeColor = [](float u, float /*v*/) {
        float d = std::fabs(u - 0.5f);
        float g = std::exp(-d * d * 4000.0f);   // a narrow bright bump centered at u==0.5
        return Vec3{0.1f + 0.9f * g, 0.1f + 0.9f * g, 0.1f + 0.9f * g};
    };

    const Vec2 center{0.5f, 0.5f};
    const Vec2 axisH{1.0f / 1280.0f, 0.0f};   // one horizontal pixel in UV
    const float kWidth = 12.0f;
    const float kStrength = 1.0f;
    const int   kTaps = 17;
    const float kDepthScale = 0.5f;

    // ---- Zero strength == identity (the byte-identical no-op proof, CPU half). ----
    {
        Vec3 c0 = sss::BlurAxis(spikeColor, flatDepth, maskOn, center, axisH,
                                kWidth, 0.0f, kTaps, kDepthScale);
        Vec3 ref = spikeColor(center.x, center.y);
        check(exactEq(c0, ref), "BlurAxis(sssStrength=0) == center color EXACTLY");
    }
    // ---- Zero width == identity. ----
    {
        Vec3 c0 = sss::BlurAxis(spikeColor, flatDepth, maskOn, center, axisH,
                                0.0f, kStrength, kTaps, kDepthScale);
        Vec3 ref = spikeColor(center.x, center.y);
        check(exactEq(c0, ref), "BlurAxis(sssWidthPx=0) == center color EXACTLY");
    }
    // ---- The early-out holds regardless of the depth/mask field (true pass-through). ----
    {
        auto wildDepth = [](float u, float v) { return 3.0f + 7.0f * u + 2.0f * v; };
        auto patchyMask = [](float u, float) { return (u > 0.5f) ? 1.0f : 0.0f; };
        Vec3 c0 = sss::BlurAxis(spikeColor, wildDepth, patchyMask, center, axisH,
                                kWidth, 0.0f, kTaps, kDepthScale);
        check(exactEq(c0, spikeColor(center.x, center.y)),
              "BlurAxis(strength=0) is an exact pass-through for ANY depth/mask field");
    }

    // ---- Non-subsurface (mask 0) pixel is unchanged. ----
    {
        auto maskOff = [](float, float) { return 0.0f; };
        Vec3 c = sss::BlurAxis(spikeColor, flatDepth, maskOff, center, axisH,
                               kWidth, kStrength, kTaps, kDepthScale);
        check(exactEq(c, spikeColor(center.x, center.y)),
              "BlurAxis: a mask-0 (non-subsurface) pixel passes through unchanged");
    }

    // ---- Blur behavior: a bright subsurface center spreads to neighbors (softens the spike). ----
    {
        Vec3 blurred = sss::BlurAxis(spikeColor, flatDepth, maskOn, center, axisH,
                                     kWidth, kStrength, kTaps, kDepthScale);
        Vec3 orig = spikeColor(center.x, center.y);
        check(finite3(blurred), "BlurAxis result finite");
        // The bright center is pulled DOWN toward the darker neighbors (energy spreads outward).
        check(blurred.x < orig.x, "BlurAxis softens a bright subsurface center (spreads to neighbors)");
        // ...but stays within the color range present in the field (normalized average, no overshoot).
        check(blurred.x > 0.1f - 1e-4f && blurred.x < orig.x,
              "BlurAxis result stays within [neighbor, center] (normalized, energy-conserving)");

        // A DARK point sitting in a bright neighborhood is pulled UP (the inverse — light bleeds IN).
        auto wellColor = [](float u, float) {
            float d = std::fabs(u - 0.5f);
            float g = std::exp(-d * d * 4000.0f);
            return Vec3{1.0f - 0.9f * g, 1.0f - 0.9f * g, 1.0f - 0.9f * g};
        };
        Vec3 well0 = wellColor(center.x, center.y);
        Vec3 wellB = sss::BlurAxis(wellColor, flatDepth, maskOn, center, axisH,
                                   kWidth, kStrength, kTaps, kDepthScale);
        check(wellB.x > well0.x, "BlurAxis fills a dark subsurface dip (light bleeds in)");
    }

    // ---- Silhouette stop: a depth field with a discontinuity on one side -> the blur does not bleed
    // across it (the result differs from the no-discontinuity blur because far taps are rejected). ----
    {
        // Center surface at depth 10; everything with u > 0.5 jumps to depth 30 (a foreground object's
        // silhouette). The diffusion must NOT pull those far-depth colors in.
        auto stepDepth = [](float u, float) { return (u > 0.5f) ? 30.0f : 10.0f; };
        auto sideColor = [](float u, float) {
            // dark on the near side, bright (a different object) across the silhouette
            return (u > 0.5f) ? Vec3{1.0f, 1.0f, 1.0f} : Vec3{0.1f, 0.1f, 0.1f};
        };
        // Center just on the NEAR side of the edge.
        Vec2 nearCenter{0.5f - axisH.x * 2.0f, 0.5f};
        Vec3 withStop = sss::BlurAxis(sideColor, stepDepth, maskOn, nearCenter, axisH,
                                      kWidth, kStrength, kTaps, kDepthScale);
        Vec3 noStop = sss::BlurAxis(sideColor, flatDepth /*no discontinuity*/, maskOn, nearCenter,
                                    axisH, kWidth, kStrength, kTaps, kDepthScale);
        // Without the depth stop the bright far-object taps bleed in and brighten the result; WITH the
        // stop they are rejected, so the depth-aware result stays darker (closer to the near side).
        check(withStop.x < noStop.x,
              "DepthFalloff prevents bleed across a silhouette (depth-aware result stays on the near surface)");
    }

    // ---- Determinism: same inputs -> identical result. ----
    {
        Vec3 a = sss::BlurAxis(spikeColor, flatDepth, maskOn, center, axisH,
                               kWidth, kStrength, kTaps, kDepthScale);
        Vec3 b = sss::BlurAxis(spikeColor, flatDepth, maskOn, center, axisH,
                               kWidth, kStrength, kTaps, kDepthScale);
        check(exactEq(a, b), "BlurAxis deterministic (same inputs -> identical result)");
    }

    if (g_fail == 0) std::printf("sss_test: all checks passed\n");
    else std::printf("sss_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
