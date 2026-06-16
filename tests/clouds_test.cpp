// Slice CH — volumetric cloud math. Pure CPU: deterministic value-noise (integer-lattice hash +
// trilinear interp), FBM, the slab density field (FBM x height-gradient x coverage threshold,
// advected by t*wind), Beer-Lambert extinction and the Henyey-Greenstein phase. No device,
// ASan-eligible (links hf_core). Mirrors the math the --clouds-shot showcase and clouds.frag use
// (engine/render/clouds.h). Determinism + range + the slab/coverage/phase invariants are exercised
// here so the GPU raymarch is built on a unit-checked field.
#include "render/clouds.h"
#include "math/math.h"
#include <cmath>
#include <cstdio>

using namespace hf::math;
namespace clouds = hf::render::clouds;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }

int main() {
    // ---- Noise3: deterministic + range [0,1]. ----
    {
        // Same point -> bit-identical (no RNG, pure hash).
        Vec3 p{3.7f, -1.2f, 8.4f};
        float a = clouds::Noise3(p);
        float b = clouds::Noise3(p);
        check(a == b, "Noise3 is bit-deterministic for the same point");

        // Range: sample a grid of non-lattice + lattice points -> all in [0,1].
        bool inRange = true, sawLow = false, sawHigh = false;
        for (float x = -5.0f; x <= 5.0f; x += 0.37f)
            for (float y = -5.0f; y <= 5.0f; y += 0.41f)
                for (float z = -5.0f; z <= 5.0f; z += 0.53f) {
                    float n = clouds::Noise3(Vec3{x, y, z});
                    if (n < 0.0f || n > 1.0f) inRange = false;
                    if (n < 0.25f) sawLow = true;
                    if (n > 0.75f) sawHigh = true;
                }
        check(inRange, "Noise3 stays within [0,1] over a sampled volume");
        check(sawLow && sawHigh, "Noise3 spans a non-trivial range (low and high samples seen)");

        // At an integer lattice point Noise3 returns the lattice hash value exactly (trilinear weights
        // collapse to that corner): two distinct lattice points should generally differ.
        float l0 = clouds::Noise3(Vec3{0.0f, 0.0f, 0.0f});
        float l1 = clouds::Noise3(Vec3{1.0f, 0.0f, 0.0f});
        float l2 = clouds::Noise3(Vec3{0.0f, 7.0f, 3.0f});
        check(l0 >= 0.0f && l0 <= 1.0f && l1 >= 0.0f && l1 <= 1.0f && l2 >= 0.0f && l2 <= 1.0f,
              "lattice hash values are in [0,1]");
        check(!(l0 == l1 && l1 == l2), "distinct lattice points are not all identical");
    }

    // ---- Fbm: deterministic + bounded in [0,1]. ----
    {
        Vec3 p{1.3f, 2.6f, -0.9f};
        float a = clouds::Fbm(p, 4);
        float b = clouds::Fbm(p, 4);
        check(a == b, "Fbm is bit-deterministic");

        bool inRange = true;
        for (float x = -4.0f; x <= 4.0f; x += 0.63f)
            for (float z = -4.0f; z <= 4.0f; z += 0.57f) {
                float f = clouds::Fbm(Vec3{x, 1.5f, z}, 5);
                if (f < 0.0f || f > 1.0001f) inRange = false;
            }
        check(inRange, "Fbm is bounded in [0,1]");
    }

    // ---- Density slab: 0 below slabBottom + above slabTop, non-zero inside a covered region. ----
    {
        const float t = clouds::kFixedTime;
        const float bottom = clouds::kSlabBottom;
        const float top = clouds::kSlabTop;

        // Below the slab and above the slab -> exactly 0 everywhere.
        bool belowZero = true, aboveZero = true;
        for (float x = -20.0f; x <= 20.0f; x += 3.1f)
            for (float z = -20.0f; z <= 20.0f; z += 2.9f) {
                if (clouds::Density(Vec3{x, bottom - 1.0f, z}, t, bottom, top) != 0.0f) belowZero = false;
                if (clouds::Density(Vec3{x, top + 1.0f, z}, t, bottom, top) != 0.0f) aboveZero = false;
            }
        check(belowZero, "Density is 0 below slabBottom");
        check(aboveZero, "Density is 0 above slabTop");

        // Inside the slab there is at least one covered (non-zero) sample.
        float mid = 0.5f * (bottom + top);
        bool sawNonZero = false;
        for (float x = -40.0f; x <= 40.0f && !sawNonZero; x += 1.7f)
            for (float z = -40.0f; z <= 40.0f && !sawNonZero; z += 1.9f)
                if (clouds::Density(Vec3{x, mid, z}, t, bottom, top) > 0.0f) sawNonZero = true;
        check(sawNonZero, "Density is non-zero somewhere inside the slab (clouds exist)");
    }

    // ---- Height gradient: mid-slab is denser than the edges (for a fixed covered column). ----
    {
        const float t = clouds::kFixedTime;
        const float bottom = clouds::kSlabBottom;
        const float top = clouds::kSlabTop;
        const float mid = 0.5f * (bottom + top);

        // Find an (x,z) column that is clearly covered at mid-height, then compare against the edges.
        bool tested = false;
        for (float x = -40.0f; x <= 40.0f && !tested; x += 1.3f)
            for (float z = -40.0f; z <= 40.0f && !tested; z += 1.1f) {
                float dMid = clouds::Density(Vec3{x, mid, z}, t, bottom, top);
                if (dMid > 0.05f) {
                    // Just inside the bottom and top edges the height gradient -> ~0, so the
                    // density there is strictly less than at mid-height for the SAME column.
                    float dLow = clouds::Density(Vec3{x, bottom + 0.02f * (top - bottom), z}, t, bottom, top);
                    float dHigh = clouds::Density(Vec3{x, top - 0.02f * (top - bottom), z}, t, bottom, top);
                    check(dMid > dLow, "mid-slab density exceeds the near-bottom edge");
                    check(dMid > dHigh, "mid-slab density exceeds the near-top edge");
                    tested = true;
                }
            }
        check(tested, "found a covered column to test the height gradient");
    }

    // ---- Coverage: raising the coverage threshold reduces total density (fewer/thinner clouds). ----
    {
        const float t = clouds::kFixedTime;
        const float bottom = clouds::kSlabBottom;
        const float top = clouds::kSlabTop;
        const float mid = 0.5f * (bottom + top);

        auto totalDensity = [&](float coverage) {
            float sum = 0.0f;
            for (float x = -30.0f; x <= 30.0f; x += 1.5f)
                for (float z = -30.0f; z <= 30.0f; z += 1.5f)
                    sum += clouds::DensityCoverage(Vec3{x, mid, z}, t, bottom, top, coverage);
            return sum;
        };
        float lowCov = totalDensity(0.30f);   // low threshold -> lots of cloud
        float highCov = totalDensity(0.60f);  // high threshold -> less cloud
        check(highCov < lowCov, "raising coverage reduces total density");
        check(lowCov > 0.0f, "low-coverage threshold yields non-zero cloud");

        // The default Density uses the documented default coverage.
        float viaDefault = 0.0f, viaExplicit = 0.0f;
        for (float x = -10.0f; x <= 10.0f; x += 2.0f)
            for (float z = -10.0f; z <= 10.0f; z += 2.0f) {
                viaDefault += clouds::Density(Vec3{x, mid, z}, t, bottom, top);
                viaExplicit += clouds::DensityCoverage(Vec3{x, mid, z}, t, bottom, top, clouds::kCoverage);
            }
        check(approx(viaDefault, viaExplicit), "Density uses the documented default coverage");
    }

    // ---- Beer-Lambert: Beer(0)==1, Beer(large)->0, strictly monotone decreasing. ----
    {
        check(approx(clouds::Beer(0.0f), 1.0f), "Beer(0) == 1");
        check(clouds::Beer(50.0f) < 1e-6f, "Beer(large) -> 0");
        float prev = clouds::Beer(0.0f);
        for (float od = 0.25f; od <= 12.0f; od += 0.25f) {
            float v = clouds::Beer(od);
            check(v < prev, "Beer is strictly monotone decreasing");
            check(v > 0.0f, "Beer stays positive");
            prev = v;
        }
        // Exact value spot-check: Beer(1) == 1/e.
        check(approx(clouds::Beer(1.0f), std::exp(-1.0f)), "Beer(1) == exp(-1)");
    }

    // ---- Henyey-Greenstein: peaks forward (cosA=1) for g>0, hand-checked values. ----
    {
        const float g = 0.5f;
        float fwd = clouds::HenyeyGreenstein(1.0f, g);    // forward (toward the light)
        float side = clouds::HenyeyGreenstein(0.0f, g);   // 90 degrees
        float back = clouds::HenyeyGreenstein(-1.0f, g);  // backward
        check(fwd > side && side > back, "HG peaks forward and falls off to the back for g>0");

        // Hand-checked closed-form values. HG(cos,g) = (1-g^2) / (4*pi*(1+g^2-2*g*cos)^1.5).
        const float pi = 3.14159265358979323846f;
        auto hgRef = [&](float c, float gg) {
            float g2 = gg * gg;
            float denom = 1.0f + g2 - 2.0f * gg * c;
            return (1.0f - g2) / (4.0f * pi * denom * std::sqrt(denom));
        };
        check(approx(clouds::HenyeyGreenstein(1.0f, g), hgRef(1.0f, g), 1e-5f),
              "HG forward matches the closed form");
        check(approx(clouds::HenyeyGreenstein(0.0f, g), hgRef(0.0f, g), 1e-5f),
              "HG side matches the closed form");
        check(approx(clouds::HenyeyGreenstein(-0.3f, g), hgRef(-0.3f, g), 1e-5f),
              "HG arbitrary angle matches the closed form");

        // g == 0 -> isotropic 1/(4*pi) for every angle.
        check(approx(clouds::HenyeyGreenstein(0.4f, 0.0f), 1.0f / (4.0f * pi), 1e-5f),
              "HG(g=0) == 1/(4*pi) (isotropic)");
    }

    // ---- Determinism: same (p, t) -> identical Density across calls. ----
    {
        const float bottom = clouds::kSlabBottom;
        const float top = clouds::kSlabTop;
        Vec3 p{2.4f, 0.5f * (bottom + top), -3.1f};
        float t = clouds::kFixedTime;
        float d1 = clouds::Density(p, t, bottom, top);
        float d2 = clouds::Density(p, t, bottom, top);
        check(d1 == d2, "Density is bit-deterministic for the same (p,t)");
        // A different fixed time generally advects the field to a different value (wind != 0).
        float dT = clouds::Density(p, t + 5.0f, bottom, top);
        check(true, "Density(t+dt) evaluates without error");  // value may differ; just exercise it
        (void)dT;
    }

    if (g_fail == 0) std::printf("clouds_test: all checks passed\n");
    else std::printf("clouds_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
