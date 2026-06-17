// Slice DB — Color Grading (analytic lift/gamma/gain + ASC-CDL slope/offset/power + saturation, post-
// tonemap). Pure CPU math: no device, ASan-eligible (links hf_core). Mirrors the math the
// --colorgrade-shot showcase and color_grade.frag.hlsl use (engine/render/color_grade.h).
//
// Properties pinned (per the spec):
//   * Identity = no-op: Grade(c, Identity) == c EXACTLY for a set of colors (0, 1, mid-grey, a
//     saturated color); each sub-op (LiftGammaGain / ASC_CDL / Saturate) at its identity -> c EXACTLY
//     (the byte-identical identity-grade == ungraded render proof's CPU half).
//   * Monotone controls: gain brightens; lift raises blacks MORE than brights; gamma > 1 brightens
//     mids; slope/offset/power match the ASC-CDL SOP definition (hand-checked); saturation 0 ->
//     greyscale (== Rec.709 luma), saturation > 1 -> more saturated.
//   * Range/robustness: no NaN at c == 0 with the power/gamma terms; the ASC-CDL clamp floors a
//     driven-negative channel at >= 0.
//   * Determinism: same inputs -> identical result (pure function, no RNG/time).
#include "render/color_grade.h"

#include <cmath>
#include <cstdio>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace grade = hf::render::grade;
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
static bool approx(float a, float b, float eps = 1e-5f) { return std::fabs(a - b) <= eps; }
static bool approx3(const Vec3& a, const Vec3& b, float eps = 1e-5f) {
    return approx(a.x, b.x, eps) && approx(a.y, b.y, eps) && approx(a.z, b.z, eps);
}

int main() {
    HF_TEST_MAIN_INIT();
    const grade::GradeParams kId = grade::GradeParams::Identity();

    // A representative set of LDR colors (post-tonemap, nominally [0,1]).
    const Vec3 colors[] = {
        {0.0f, 0.0f, 0.0f},     // black
        {1.0f, 1.0f, 1.0f},     // white
        {0.5f, 0.5f, 0.5f},     // mid-grey
        {0.8f, 0.2f, 0.35f},    // a saturated color
        {0.15f, 0.62f, 0.9f},   // a cool color
        {0.42f, 0.42f, 0.42f},  // an off mid-grey
    };

    // ---- IDENTITY = no-op: Grade(c, Identity) == c EXACTLY (the byte-identical proof's core). ----
    {
        for (const Vec3& c : colors)
            check(exactEq(grade::Grade(c, kId), c), "Grade(c, identity) == c EXACTLY");
    }

    // ---- Each sub-op at its identity -> c EXACTLY. ----
    {
        const Vec3 zero{0, 0, 0}, one{1, 1, 1};
        for (const Vec3& c : colors) {
            check(exactEq(grade::LiftGammaGain(c, zero, one, one), c),
                  "LiftGammaGain(c, lift0, gamma1, gain1) == c EXACTLY");
            check(exactEq(grade::ASC_CDL(c, one, zero, one), c),
                  "ASC_CDL(c, slope1, offset0, power1) == c EXACTLY");
            check(exactEq(grade::Saturate(c, 1.0f), c),
                  "Saturate(c, saturation1) == c EXACTLY");
        }
    }

    // ---- PowCh: exact at exponent 1, finite/non-negative at base 0. ----
    {
        check(grade::PowCh(0.37f, 1.0f) == 0.37f, "PowCh(x, 1) == x EXACTLY");
        check(grade::PowCh(0.0f, 0.5f) == 0.0f, "PowCh(0, 0.5) == 0 (no NaN)");
        check(grade::PowCh(0.0f, 2.2f) == 0.0f, "PowCh(0, 2.2) == 0 (no NaN)");
        check(std::isfinite(grade::PowCh(-0.3f, 0.4f)),
              "PowCh(negative, fractional) is finite (clamped base)");
        check(approx(grade::PowCh(0.25f, 0.5f), 0.5f), "PowCh(0.25, 0.5) == 0.5");
    }

    // ---- GAIN brightens uniformly (a multiply). ----
    {
        const Vec3 c{0.4f, 0.4f, 0.4f};
        Vec3 g = grade::LiftGammaGain(c, {0, 0, 0}, {1, 1, 1}, {1.5f, 1.5f, 1.5f});
        check(approx3(g, Vec3{0.6f, 0.6f, 0.6f}), "gain 1.5 multiplies (0.4 -> 0.6)");
        check(g.x > c.x && g.y > c.y && g.z > c.z, "gain > 1 brightens");
    }

    // ---- LIFT raises blacks MORE than brights (the (1-c) weight); never lowers. ----
    {
        const Vec3 lift{0.2f, 0.2f, 0.2f};
        const Vec3 one{1, 1, 1}, zeroLift{0, 0, 0};
        Vec3 dark  = grade::LiftGammaGain({0.1f, 0.1f, 0.1f}, lift, one, one);
        Vec3 bright = grade::LiftGammaGain({0.9f, 0.9f, 0.9f}, lift, one, one);
        float dDark   = dark.x   - 0.1f;   // how much the dark value rose
        float dBright = bright.x - 0.9f;   // how much the bright value rose
        check(dDark > 0.0f, "lift raises a dark value");
        check(dBright >= 0.0f, "lift never lowers a value");
        check(dDark > dBright, "lift raises blacks MORE than brights (the (1-c) weight)");
        // lift 0 adds nothing.
        check(exactEq(grade::LiftGammaGain({0.3f, 0.6f, 0.9f}, zeroLift, one, one),
                      Vec3{0.3f, 0.6f, 0.9f}), "lift 0 adds nothing");
        // Hand-check: c=0.1, lift 0.2 -> 0.1 + 0.2*(1-0.1) = 0.1 + 0.18 = 0.28.
        check(approx(dark.x, 0.28f), "lift hand-check: 0.1 + 0.2*(0.9) == 0.28");
    }

    // ---- GAMMA > 1 brightens the midtones (^(1/gamma), exponent < 1 bows the curve up). ----
    {
        const Vec3 mid{0.5f, 0.5f, 0.5f};
        const Vec3 one{1, 1, 1}, zero{0, 0, 0};
        Vec3 brighter = grade::LiftGammaGain(mid, zero, {2.0f, 2.0f, 2.0f}, one);
        Vec3 darker   = grade::LiftGammaGain(mid, zero, {0.5f, 0.5f, 0.5f}, one);
        check(brighter.x > 0.5f, "gamma > 1 brightens the midtones");
        check(darker.x < 0.5f, "gamma < 1 darkens the midtones");
        // Hand-check: 0.5 ^ (1/2) = sqrt(0.5) ~ 0.7071.
        check(approx(brighter.x, std::sqrt(0.5f)), "gamma 2 hand-check: 0.5^(1/2) == sqrt(0.5)");
        // Gamma does not move the endpoints 0 and 1.
        Vec3 ends = grade::LiftGammaGain({0.0f, 1.0f, 0.5f}, zero, {2.0f, 2.0f, 2.0f}, one);
        check(approx(ends.x, 0.0f) && approx(ends.y, 1.0f), "gamma fixes the 0 and 1 endpoints");
    }

    // ---- ASC-CDL slope/offset/power: hand-checked against out = (in*slope + offset)^power, clamp>=0. ----
    {
        const Vec3 one{1, 1, 1}, zero{0, 0, 0};
        // slope 2 (offset 0, power 1): 0.3 -> 0.6.
        check(approx3(grade::ASC_CDL({0.3f, 0.3f, 0.3f}, {2, 2, 2}, zero, one),
                      Vec3{0.6f, 0.6f, 0.6f}), "ASC-CDL slope 2: 0.3 -> 0.6");
        // offset 0.1 (slope 1, power 1): 0.4 -> 0.5.
        check(approx3(grade::ASC_CDL({0.4f, 0.4f, 0.4f}, one, {0.1f, 0.1f, 0.1f}, one),
                      Vec3{0.5f, 0.5f, 0.5f}), "ASC-CDL offset 0.1: 0.4 -> 0.5");
        // power 2 (slope 1, offset 0): 0.5 -> 0.25 (darkens the [0,1] midtones).
        check(approx3(grade::ASC_CDL({0.5f, 0.5f, 0.5f}, one, zero, {2, 2, 2}),
                      Vec3{0.25f, 0.25f, 0.25f}), "ASC-CDL power 2: 0.5 -> 0.25");
        // Combined: in=0.5, slope=2, offset=-0.2, power=2 -> (1.0 - 0.2)^2 = 0.8^2 = 0.64.
        check(approx(grade::ASC_CDL({0.5f, 0.5f, 0.5f}, {2, 2, 2}, {-0.2f, -0.2f, -0.2f}, {2, 2, 2}).x,
                     0.64f), "ASC-CDL combined hand-check: (0.5*2 - 0.2)^2 == 0.64");
        // The clamp floors a driven-negative affine at 0 (no NaN from a fractional power of a negative).
        Vec3 neg = grade::ASC_CDL({0.1f, 0.1f, 0.1f}, one, {-0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, 0.5f});
        check(neg.x == 0.0f, "ASC-CDL clamps a driven-negative channel to >= 0");
        check(finite3(neg), "ASC-CDL no NaN at a negative affine with a fractional power");
    }

    // ---- SATURATION: 0 -> greyscale (Rec.709 luma); > 1 -> more saturated; preserves luma. ----
    {
        const Vec3 c{0.8f, 0.2f, 0.35f};
        float luma = c.x * grade::kLumaR + c.y * grade::kLumaG + c.z * grade::kLumaB;
        Vec3 grey = grade::Saturate(c, 0.0f);
        check(approx3(grey, Vec3{luma, luma, luma}),
              "saturation 0 -> greyscale at the Rec.709 luma");
        check(approx(grey.x, grey.y) && approx(grey.y, grey.z),
              "saturation 0 -> all channels equal (achromatic)");
        // saturation > 1 spreads the channels FARTHER from the luma than the original.
        Vec3 punchy = grade::Saturate(c, 1.5f);
        float origSpread   = std::fabs(c.x - luma) + std::fabs(c.y - luma) + std::fabs(c.z - luma);
        float punchySpread = std::fabs(punchy.x - luma) + std::fabs(punchy.y - luma) +
                             std::fabs(punchy.z - luma);
        check(punchySpread > origSpread, "saturation > 1 increases the spread from grey (more saturated)");
        // A neutral grey stays grey under any saturation (it IS its own luma).
        check(approx3(grade::Saturate({0.5f, 0.5f, 0.5f}, 0.0f), Vec3{0.5f, 0.5f, 0.5f}),
              "a neutral grey is unchanged by desaturation");
        // Hand-check at saturation 0.5: out = grey + (c-grey)*0.5 (halfway to grey).
        Vec3 half = grade::Saturate(c, 0.5f);
        check(approx(half.x, luma + (c.x - luma) * 0.5f), "saturation 0.5 lerps halfway to grey");
    }

    // ---- The composed grade differs from the input under a non-trivial grade (it actually grades). ----
    {
        grade::GradeParams p;
        p.lift   = {0.04f, 0.06f, 0.08f};   // teal-ish black lift
        p.gain   = {1.10f, 1.02f, 0.92f};   // warm highlights
        p.gamma  = {1.05f, 1.0f, 0.98f};
        p.saturation = 1.15f;
        const Vec3 c{0.4f, 0.45f, 0.5f};
        Vec3 g = grade::Grade(c, p);
        check(!exactEq(g, c), "a non-trivial grade changes the color");
        check(finite3(g), "a non-trivial grade stays finite");
    }

    // ---- Range/no-NaN: every sub-op + the composition finite at the extremes (0 and 1). ----
    {
        grade::GradeParams p;
        p.gamma  = {2.2f, 2.2f, 2.2f};
        p.power  = {2.0f, 2.0f, 2.0f};
        p.offset = {-0.3f, -0.3f, -0.3f};   // can drive the affine negative -> exercise the clamp
        p.saturation = 1.4f;
        for (const Vec3& c : colors) {
            Vec3 g = grade::Grade(c, p);
            check(finite3(g), "Grade finite (no NaN) at the [0,1] extremes with power/gamma/clamp terms");
        }
    }

    // ---- Determinism: same inputs -> identical result (pure function). ----
    {
        grade::GradeParams p;
        p.lift = {0.03f, 0.05f, 0.07f}; p.gain = {1.1f, 1.0f, 0.9f}; p.saturation = 1.2f;
        const Vec3 c{0.37f, 0.51f, 0.62f};
        check(exactEq(grade::Grade(c, p), grade::Grade(c, p)), "Grade deterministic (bit-identical)");
    }

    if (g_fail == 0) std::printf("color_grade_test: ALL PASS\n");
    else std::printf("color_grade_test: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
