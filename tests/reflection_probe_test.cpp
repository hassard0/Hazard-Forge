// Slice DA — Box-Projected Cubemap Reflections (local reflection probe). Pure CPU math: the slab-method
// ray-box EXIT t and the box-projected (parallax-corrected) reflection direction. No device,
// ASan-eligible (links hf_core). Mirrors the math the --reflprobe-shot showcase and reflprobe.frag use
// (engine/render/reflection_probe.h).
//
// Properties pinned (per the spec):
//   * Ray-box exit: RayBoxExitT for a known origin/dir/box -> the analytic exit t (axis-aligned + a
//     diagonal case), always POSITIVE.
//   * Zero parallax == identity: BoxProject(R, P, box, parallaxStrength=0) == normalize(R) EXACTLY for
//     any inputs (the byte-identical no-op proof's CPU half).
//   * Infinite box ≈ identity: with a huge box, BoxProject -> ≈ normalize(R); the correction VANISHES
//     as the box grows (quantified — the angular error shrinks toward 0 as the box scale grows).
//   * Finite-box correction: for a finite box, the corrected direction points toward the box-exit
//     relative to the probe center (differs from R when P != center); a hand-computed geometry checks
//     the exact corrected direction.
//   * Determinism: same inputs -> identical result (pure function, no RNG/time).
#include "render/reflection_probe.h"
#include <cmath>
#include <cstdio>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace rp = hf::render::reflprobe;
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
static bool approx(float a, float b, float eps) { return std::fabs(a - b) <= eps; }
static bool approx3(const Vec3& a, const Vec3& b, float eps) {
    return approx(a.x, b.x, eps) && approx(a.y, b.y, eps) && approx(a.z, b.z, eps);
}
// Angle (radians) between two (assumed near-unit) directions, clamped for acos domain safety.
static float angleBetween(const Vec3& a, const Vec3& b) {
    Vec3 na = hf::math::normalize(a), nb = hf::math::normalize(b);
    float d = hf::math::dot(na, nb);
    if (d > 1.0f) d = 1.0f; if (d < -1.0f) d = -1.0f;
    return std::acos(d);
}

int main() {
    HF_TEST_MAIN_INIT();
    // A unit box centered at the origin: [-1,1]^3.
    const Vec3 bmin{-1.0f, -1.0f, -1.0f};
    const Vec3 bmax{ 1.0f,  1.0f,  1.0f};

    // ---- RayBoxExitT: analytic exit t (axis-aligned). ----------------------------------------------
    {
        // From the center toward +X: exits the +X face at x==1 -> t == 1 (dir unit).
        float t = rp::RayBoxExitT(Vec3{0, 0, 0}, Vec3{1, 0, 0}, bmin, bmax);
        check(approx(t, 1.0f, 1e-5f), "RayBoxExitT: center -> +X exits at t==1");
        check(t > 0.0f, "RayBoxExitT: exit t is positive (+X)");

        // From the center toward -Z: exits the -Z face at z==-1 -> t == 1.
        float tz = rp::RayBoxExitT(Vec3{0, 0, 0}, Vec3{0, 0, -1}, bmin, bmax);
        check(approx(tz, 1.0f, 1e-5f), "RayBoxExitT: center -> -Z exits at t==1");

        // From an OFF-CENTER origin inside the box toward +X: origin x=0.25 -> exits at t == 0.75.
        float to = rp::RayBoxExitT(Vec3{0.25f, 0, 0}, Vec3{1, 0, 0}, bmin, bmax);
        check(approx(to, 0.75f, 1e-5f), "RayBoxExitT: off-center -> +X exits at t==0.75");

        // A non-unit dir scales t by 1/|dir|: dir (2,0,0) from center exits at t == 0.5.
        float ts = rp::RayBoxExitT(Vec3{0, 0, 0}, Vec3{2, 0, 0}, bmin, bmax);
        check(approx(ts, 0.5f, 1e-5f), "RayBoxExitT: non-unit dir scales exit t by 1/|dir|");
    }

    // ---- RayBoxExitT: diagonal case. ---------------------------------------------------------------
    {
        // From the center toward (1,1,0) (45deg in XY): the ray reaches x==1 at t=1 and y==1 at t=1
        // simultaneously, exiting the box corner -> tFar == 1 (the min of the per-slab exits, both 1;
        // the Z slab is parallel/skipped). The exit POINT is (1,1,0), a box corner.
        float td = rp::RayBoxExitT(Vec3{0, 0, 0}, Vec3{1, 1, 0}, bmin, bmax);
        check(approx(td, 1.0f, 1e-5f), "RayBoxExitT: diagonal (1,1,0) exits at t==1 (corner)");
        check(td > 0.0f, "RayBoxExitT: diagonal exit t is positive");

        // A skew diagonal: from origin toward (1,2,0). x==1 at t=1, y==1 at t=0.5 -> exits the +Y face
        // FIRST at t == 0.5 (the smaller per-slab exit governs). Exit point (0.5,1,0) is on the +Y wall.
        float ts = rp::RayBoxExitT(Vec3{0, 0, 0}, Vec3{1, 2, 0}, bmin, bmax);
        check(approx(ts, 0.5f, 1e-5f), "RayBoxExitT: skew (1,2,0) exits the +Y face first at t==0.5");

        // A fully 3D diagonal toward (1,1,1): all three slabs exit at t==1 -> corner (1,1,1).
        float t3 = rp::RayBoxExitT(Vec3{0, 0, 0}, Vec3{1, 1, 1}, bmin, bmax);
        check(approx(t3, 1.0f, 1e-5f), "RayBoxExitT: 3D diagonal (1,1,1) exits at t==1 (corner)");
    }

    // ---- Zero parallax == identity (the byte-identical no-op proof's CPU half). ---------------------
    {
        rp::ProbeBox box{Vec3{0, 0, 0}, bmin, bmax};
        // For a spread of reflection dirs and surface positions, BoxProject(...,0) == normalize(R) EXACTLY.
        const Vec3 dirs[] = {
            {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {-1, 0, 0}, {0.3f, 0.7f, -0.5f},
            {2.0f, -1.0f, 0.5f}, {-0.2f, -0.9f, 0.4f},
        };
        const Vec3 poss[] = {
            {0, 0, 0}, {0.5f, -0.3f, 0.1f}, {-0.8f, 0.2f, 0.6f}, {0.9f, 0.9f, -0.9f},
        };
        bool allExact = true;
        for (const Vec3& R : dirs)
            for (const Vec3& P : poss) {
                Vec3 got = rp::BoxProject(R, P, box, 0.0f);
                Vec3 want = hf::math::normalize(R);
                if (!exactEq(got, want)) allExact = false;
            }
        check(allExact, "BoxProject(parallaxStrength=0) == normalize(R) EXACTLY for any R,P (no-op proof)");

        // Independent of the box too: a wildly different box still gives the exact identity at 0.
        rp::ProbeBox weird{Vec3{5, -3, 2}, Vec3{-100, -7, -3}, Vec3{2, 50, 9}};
        Vec3 R{0.3f, -0.7f, 0.64f}, P{1.0f, 2.0f, -1.0f};
        check(exactEq(rp::BoxProject(R, P, weird, 0.0f), hf::math::normalize(R)),
              "BoxProject(parallaxStrength=0) identity holds regardless of the box");
    }

    // ---- Infinite box ≈ identity: the correction VANISHES as the box grows. -------------------------
    {
        const Vec3 R{0.3f, 0.7f, -0.5f};
        const Vec3 P{0.4f, -0.2f, 0.1f};   // off-center surface point (would otherwise correct)
        Vec3 want = hf::math::normalize(R);
        // Grow the box by powers of 10 and confirm the angular error to normalize(R) shrinks toward 0,
        // even at FULL parallaxStrength == 1.
        float prevErr = 1e9f;
        bool monotoneShrink = true;
        float lastErr = 0.0f;
        for (float scale : {10.0f, 100.0f, 1000.0f, 100000.0f}) {
            rp::ProbeBox box{Vec3{0, 0, 0}, Vec3{-scale, -scale, -scale}, Vec3{scale, scale, scale}};
            Vec3 got = rp::BoxProject(R, P, box, 1.0f);
            check(finite3(got), "BoxProject infinite-box result finite");
            float err = angleBetween(got, want);
            if (!(err <= prevErr + 1e-6f)) monotoneShrink = false;
            prevErr = err;
            lastErr = err;
        }
        check(monotoneShrink, "BoxProject: angular error to normalize(R) shrinks as the box grows");
        // Quantify: at box scale 1e5 the correction is essentially gone (< 0.001 rad ≈ 0.06 deg).
        check(lastErr < 1e-3f, "BoxProject: huge box (1e5) converges to normalize(R) (< 0.001 rad)");
    }

    // ---- Finite-box correction: hand-computed geometry. --------------------------------------------
    {
        // Unit box [-1,1]^3, probe center at the origin. A surface point P offset along +X reflecting
        // straight UP (+Y). The ray P + t*(0,1,0) exits the +Y face (y==1) at t == 1, hit == (Px,1,0).
        // For Px = 0.5: hit == (0.5, 1, 0); (hit - center) == (0.5,1,0). At FULL parallaxStrength == 1
        // the corrected dir == normalize(0.5,1,0). This LEANS toward +X relative to the straight-up R
        // (the reflection now points at where the ray actually hits the +Y wall) — the parallax
        // correction. Hand-computed: normalize(0.5,1,0) = (0.5,1,0)/sqrt(1.25).
        rp::ProbeBox box{Vec3{0, 0, 0}, bmin, bmax};
        Vec3 R{0, 1, 0};
        Vec3 P{0.5f, 0.0f, 0.0f};
        Vec3 got = rp::BoxProject(R, P, box, 1.0f);
        float inv = 1.0f / std::sqrt(1.25f);
        Vec3 want{0.5f * inv, 1.0f * inv, 0.0f};
        check(approx3(got, want, 1e-5f), "BoxProject(finite,strength=1): hand-computed corrected dir");
        check(approx(hf::math::length(got), 1.0f, 1e-5f), "BoxProject result is unit length");

        // The corrected dir DIFFERS from R when P != center (the correction is real, not a no-op).
        check(angleBetween(got, R) > 0.1f, "BoxProject(finite): corrected dir differs from R (parallax)");
        // It leans toward +X (toward the offset side), i.e. positive X component (R had x==0).
        check(got.x > 0.0f, "BoxProject(finite): corrected dir leans toward the box-exit side (+X)");

        // At P == center the +Y ray exits at (0,1,0); (hit-center) == (0,1,0) == R, so even at
        // strength 1 the corrected dir == R (no parallax when the surface is AT the probe origin).
        Vec3 atCenter = rp::BoxProject(R, Vec3{0, 0, 0}, box, 1.0f);
        check(approx3(atCenter, hf::math::normalize(R), 1e-5f),
              "BoxProject(finite): P==center gives R (no parallax at the probe origin)");

        // Intermediate strength lies BETWEEN R and the full correction (monotone blend): the angle to R
        // at strength 0.5 is smaller than at strength 1.
        Vec3 half = rp::BoxProject(R, P, box, 0.5f);
        check(angleBetween(half, R) < angleBetween(got, R),
              "BoxProject: parallaxStrength blends monotonically between R and the full correction");
    }

    // ---- Determinism: same inputs -> identical result. ---------------------------------------------
    {
        rp::ProbeBox box{Vec3{0.1f, 0.2f, -0.1f}, bmin, bmax};
        Vec3 R{0.2f, 0.9f, -0.3f}, P{0.3f, -0.4f, 0.2f};
        Vec3 a = rp::BoxProject(R, P, box, 0.8f);
        Vec3 b = rp::BoxProject(R, P, box, 0.8f);
        check(exactEq(a, b), "BoxProject deterministic (same inputs -> identical result)");
        float t1 = rp::RayBoxExitT(P, R, bmin, bmax);
        float t2 = rp::RayBoxExitT(P, R, bmin, bmax);
        check(t1 == t2, "RayBoxExitT deterministic");
    }

    if (g_fail == 0) std::printf("reflection_probe_test: ALL PASS\n");
    else             std::printf("reflection_probe_test: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
