#pragma once
// Slice DB — Color Grading (analytic lift/gamma/gain + ASC-CDL slope/offset/power + saturation,
// applied POST-tonemap) math — pure CPU (header-only, no device, no backend symbols). Namespace
// hf::render::grade. Mirrors dof.h / gtao.h / sss.h: a tiny shared-math header ABOVE the RHI seam
// (ZERO vk*/MTL*/mtl::/Backend::Metal CODE symbols — the only mentions of "vk"/"MTL" anywhere in
// this slice's above-seam files are seam-discipline doc comments + the [[vk::binding]] HLSL
// decorations). The color-grade fragment shader (shaders/color_grade.frag.hlsl) copies LiftGammaGain
// + ASC_CDL + Saturate + Grade VERBATIM, so tests/color_grade_test.cpp exercises the EXACT per-pixel
// grade the GPU pass runs — which is what makes the IDENTITY-grade render BYTE-IDENTICAL to the
// ungraded render AND bit-identical cross-backend.
//
// THE TECHNIQUE (analytic color grade): after the scene is tonemapped to a displayed LDR color (the
// engine's standard post.frag ACES + gamma result), an artist "grades" it for a cinematic LOOK by a
// composition of three classic analytic controls applied per pixel:
//   1. LIFT / GAMMA / GAIN — the lift/gamma/gain "color wheels": GAIN scales the highlights (a
//      multiply), LIFT raises the shadows (an offset that fades out as the input brightens), GAMMA
//      bends the midtones (a power). Per channel, so each can tint shadows/mids/highlights toward a
//      hue (e.g. lift toward teal, gain toward warm — the teal-shadows / warm-highlights split-tone).
//   2. ASC-CDL slope/offset/power — the American Society of Cinematographers Color Decision List
//      primary grade, the standard interchange transfer: out = (in*slope + offset)^power, clamped >=0.
//      Slope is a per-channel gain, offset a per-channel lift, power a per-channel gamma — a second,
//      industry-standard primary on top of the lift/gamma/gain wheels (both are offered because they
//      are the two grades every DI/colour pipeline exposes; composing them gives a rich primary).
//   3. SATURATION — a luma-preserving saturation control: lerp between the pixel's Rec.709 luma (a
//      neutral grey of the same brightness) and the pixel color. saturation 1 = unchanged, 0 =
//      greyscale, >1 = more saturated.
//
// THE IDENTITY-GRADE NO-OP PROOF (what makes this golden-safe — like GTAO radius=0==no-AO, SSS
// strength=0==no-SSS, POM heightScale=0==plain): at the IDENTITY params (lift 0, gamma 1, gain 1,
// slope 1, offset 0, power 1, saturation 1) EACH sub-op returns its input EXACTLY:
//   * LiftGammaGain: gain 1 * (c + lift 0 * (1-c)) = c, then ^(1/gamma) with gamma 1 -> ^1 = c. The
//     1/gamma exponent is guarded to skip the pow when gamma == 1 (pow(x,1.0f) is exact on IEEE in
//     principle, but the guard makes the identity UNCONDITIONALLY exact + branch-clean — no library
//     rounding can sneak in), and lift == 0 is folded away (c + 0 = c) by construction.
//   * ASC_CDL: c*slope 1 + offset 0 = c, then ^power with power 1 -> c (the power == 1 exponent is
//     likewise guarded to skip the pow). The clamp >= 0 only touches negatives, and a graded LDR
//     color in [0,1] composed from identity stays >= 0, so the clamp is a no-op at identity.
//   * Saturate: lerp(luma, c, 1) = luma + (c - luma)*1 = c EXACTLY (the lerp at t==1 returns the
//     second endpoint with no rounding — c - luma + luma telescopes; we ALSO short-circuit
//     saturation == 1 to return c directly so it is exact + branch-clean regardless of the luma).
// So Grade(c, IDENTITY) == c EXACTLY (no constant bias, no clamp drift, no pow rounding). The showcase
// renders the color_grade pass at the IDENTITY params and asserts it is BYTE-IDENTICAL (SHA) to the
// engine's standard ungraded (tonemap-only) render of the same scene — then renders the real cinematic
// grade as the golden. The proof is the SAME color_grade shader at identity vs the ungraded render, so
// it is backend-portable. The unit test additionally pins each sub-op at its identity + the documented
// monotone effect of each control.
//
// CONVENTIONS:
//   * The grade operates on the DISPLAYED LDR color (post-tonemap, post-gamma, nominally in [0,1]).
//     It does NOT tonemap — it grades the tonemapped result (grading-before-tonemap is explicit YAGNI
//     in the spec; this grades the SDR result, documented).
//   * lift/gamma/gain/slope/offset/power are per-CHANNEL (math::Vec3); saturation is a single scalar.
//   * Rec.709 luma weights (0.2126, 0.7152, 0.0722) — the standard SDR/Rec.709 relative luminance.
//
// Pure, deterministic functions: no RNG, no time.

#include "math/math.h"

#include <algorithm>
#include <cmath>

namespace hf::render::grade {

// Rec.709 relative-luminance weights (the standard SDR luma coefficients).
inline constexpr float kLumaR = 0.2126f;
inline constexpr float kLumaG = 0.7152f;
inline constexpr float kLumaB = 0.0722f;

// --- The grade controls --------------------------------------------------------------------------
// GradeParams bundles the three analytic primaries. The IDENTITY params (the documented no-op):
//   lift = (0,0,0), gamma = (1,1,1), gain = (1,1,1), slope = (1,1,1), offset = (0,0,0),
//   power = (1,1,1), saturation = 1. Grade(c, Identity()) == c EXACTLY for every c.
struct GradeParams {
    math::Vec3 lift   {0.0f, 0.0f, 0.0f};  // raises the shadows (fades out toward white); identity 0
    math::Vec3 gamma  {1.0f, 1.0f, 1.0f};  // bends the midtones (applied as ^(1/gamma)); identity 1
    math::Vec3 gain   {1.0f, 1.0f, 1.0f};  // scales the highlights (a multiply); identity 1
    math::Vec3 slope  {1.0f, 1.0f, 1.0f};  // ASC-CDL per-channel gain; identity 1
    math::Vec3 offset {0.0f, 0.0f, 0.0f};  // ASC-CDL per-channel lift; identity 0
    math::Vec3 power  {1.0f, 1.0f, 1.0f};  // ASC-CDL per-channel gamma; identity 1
    float      saturation {1.0f};          // luma-preserving saturation; identity 1 (0 grey, >1 punchy)

    // The canonical IDENTITY (the default-constructed params already ARE identity; this is the named,
    // self-documenting form the showcase + tests use).
    static GradeParams Identity() { return GradeParams{}; }
};

// --- Per-channel exact power (guarded at exponent 1) ---------------------------------------------
// PowCh(base, exp) = base^exp, but returns `base` EXACTLY when exp == 1 (so the gamma == 1 / power == 1
// identity is unconditionally exact, with no library pow rounding) and treats a tiny negative base as 0
// (a fractional power of a negative is NaN; the grade never wants a negative channel). For exp != 1 it
// is the standard std::pow on a clamped-non-negative base.
inline float PowCh(float base, float exp) {
    if (exp == 1.0f) return base;             // exact identity at exponent 1 (guarded, branch-clean)
    float b = (base > 0.0f) ? base : 0.0f;    // a fractional power of a negative would be NaN -> clamp
    return std::pow(b, exp);
}

// --- Lift / gamma / gain ("color wheels") --------------------------------------------------------
// LiftGammaGain(c, lift, gamma, gain) -> the lifted/gamma'd/gained color, per channel:
//
//     out = ( gain * (c + lift*(1 - c)) ) ^ (1 / gamma)
//
// Reading it term by term (per channel):
//   * lift*(1 - c)  — LIFT adds a shadow offset that is full at c == 0 and FADES to 0 at c == 1, so it
//     raises the blacks without washing out the whites (a "lift" of the toe). lift 0 -> adds nothing.
//   * + c           — the original value.
//   * gain *        — GAIN scales the whole thing (a highlight gain / overall exposure of the graded
//     color). gain 1 -> no scale.
//   * ^(1/gamma)    — GAMMA bends the midtones: gamma > 1 -> exponent < 1 -> the curve bows UP ->
//     midtones BRIGHTEN; gamma < 1 -> midtones darken. gamma 1 -> exponent 1 -> unchanged (guarded
//     exact by PowCh).
//
// IDENTITY (lift 0, gamma 1, gain 1): 1 * (c + 0) = c, then ^(1/1) = c EXACTLY.
//   Properties the unit test pins:
//   * lift raises dark values MORE than bright values (the (1-c) weight), and a positive lift never
//     lowers a value; * gain > 1 brightens uniformly (a multiply); * gamma > 1 brightens the midtones.
inline math::Vec3 LiftGammaGain(const math::Vec3& c, const math::Vec3& lift,
                                const math::Vec3& gamma, const math::Vec3& gain) {
    auto channel = [](float v, float lf, float gm, float gn) {
        float lifted = v + lf * (1.0f - v);   // raise the shadows (fades out toward white)
        float gained = gn * lifted;           // highlight gain
        return PowCh(gained, 1.0f / gm);      // midtone gamma (^(1/gamma); exact when gamma == 1)
    };
    return math::Vec3{channel(c.x, lift.x, gamma.x, gain.x),
                      channel(c.y, lift.y, gamma.y, gain.y),
                      channel(c.z, lift.z, gamma.z, gain.z)};
}

// --- ASC-CDL slope/offset/power ------------------------------------------------------------------
// ASC_CDL(c, slope, offset, power) -> the ASC Color Decision List primary grade, per channel:
//
//     out = max( c*slope + offset, 0 ) ^ power
//
// The published ASC-CDL transfer (SOP — Slope/Offset/Power): a per-channel affine (slope*in + offset)
// followed by a per-channel power, with the affine clamped to >= 0 before the power (a fractional power
// of a negative is undefined; the CDL spec clamps). slope is a gain, offset a lift, power a gamma.
//
// IDENTITY (slope 1, offset 0, power 1): max(c*1 + 0, 0) = max(c,0) = c for c >= 0, then ^1 = c EXACTLY
// (a graded LDR color composed from identity is >= 0, so the clamp is a no-op; PowCh makes ^1 exact).
//   Properties the unit test pins (hand-checked against the SOP definition):
//   * slope scales (slope 2 doubles before the power); * offset shifts up/down (and the clamp floors a
//     driven-negative channel at 0); * power applies a gamma (power 2 darkens the [0,1] midtones).
inline math::Vec3 ASC_CDL(const math::Vec3& c, const math::Vec3& slope,
                          const math::Vec3& offset, const math::Vec3& power) {
    auto channel = [](float v, float sl, float of, float pw) {
        float sop = v * sl + of;              // slope * in + offset (the affine)
        if (sop < 0.0f) sop = 0.0f;           // CDL clamp >= 0 before the power
        return PowCh(sop, pw);                // per-channel power (exact when power == 1)
    };
    return math::Vec3{channel(c.x, slope.x, offset.x, power.x),
                      channel(c.y, slope.y, offset.y, power.y),
                      channel(c.z, slope.z, offset.z, power.z)};
}

// --- Saturation (luma-preserving) ----------------------------------------------------------------
// Saturate(c, saturation) -> lerp(luma709(c), c, saturation), per channel:
//
//     out = grey + (c - grey) * saturation,   grey = dot(c, Rec.709 luma)
//
// SATURATION pulls the color toward (or past) a neutral grey of the SAME Rec.709 luminance: saturation
// 0 collapses every channel to that grey (greyscale at the pixel's luma), 1 leaves the color unchanged,
// > 1 pushes the channels further from grey (more saturated / punchier).
//
// IDENTITY (saturation 1): grey + (c - grey)*1 = c EXACTLY (we also short-circuit saturation == 1 to
// return c directly, so it is exact regardless of the luma rounding).
//   Properties the unit test pins:
//   * saturation 0 -> every channel == the Rec.709 luma of c (greyscale); * saturation > 1 spreads the
//     channels FARTHER from the luma (a saturated input gets more saturated); * the result preserves
//     the luma direction (it moves along the c - grey axis).
inline math::Vec3 Saturate(const math::Vec3& c, float saturation) {
    if (saturation == 1.0f) return c;         // exact identity at saturation 1 (guarded, branch-clean)
    float grey = c.x * kLumaR + c.y * kLumaG + c.z * kLumaB;  // Rec.709 luma (a neutral of equal luma)
    return math::Vec3{grey + (c.x - grey) * saturation,
                      grey + (c.y - grey) * saturation,
                      grey + (c.z - grey) * saturation};
}

// --- The composed grade --------------------------------------------------------------------------
// Grade(c, p) -> the full analytic grade, composing the three primaries IN THIS ORDER:
//
//     c1 = LiftGammaGain(c,  p.lift, p.gamma, p.gain)   // the lift/gamma/gain wheels first
//     c2 = ASC_CDL      (c1, p.slope, p.offset, p.power) // then the ASC-CDL primary on top
//     out = Saturate    (c2, p.saturation)               // then the saturation control last
//
// ORDER RATIONALE (documented): the two tonal primaries (the lift/gamma/gain wheels, then the ASC-CDL
// SOP) shape the per-channel transfer first; saturation — a chroma operation around the resulting luma
// — is applied LAST so it acts on the final graded color (the standard DI order: primary balance, then
// saturation trim). Each stage is identity at its identity params, so the WHOLE composition is identity
// at the IDENTITY params:
//   Grade(c, GradeParams::Identity()) == c  EXACTLY  (the byte-identical no-op proof's per-pixel core).
inline math::Vec3 Grade(const math::Vec3& c, const GradeParams& p) {
    math::Vec3 r = LiftGammaGain(c, p.lift, p.gamma, p.gain);
    r = ASC_CDL(r, p.slope, p.offset, p.power);
    r = Saturate(r, p.saturation);
    return r;
}

}  // namespace hf::render::grade
