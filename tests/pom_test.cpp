// Slice CP — Parallax Occlusion Mapping (POM). Pure CPU math: the steep-parallax + binary-refine
// ParallaxUV march and the SelfShadow soft self-occlusion. No device, ASan-eligible (links hf_core).
// Exercises the EXACT math the --pom-shot showcase + the pom.frag.hlsl shader use (engine/render/pom.h).
//
// Properties pinned (per the spec):
//   * ZERO HEIGHT = IDENTITY (the core proof): ParallaxUV(uv, viewDir, heightScale=0, ...) == uv EXACTLY
//     for any uv/view; a STRAIGHT-ON view (viewDirTangent.xy == 0) -> uv EXACTLY. This proves the march/
//     refine introduce no UV drift at zero amplitude (hence the heightScale=0 render is byte-identical
//     to plain normal mapping).
//   * OFFSET DIRECTION + MAGNITUDE: heightScale>0 + a grazing view -> the returned UV is offset from
//     baseUV in the tangent-space view direction; a deeper local height -> larger offset; a steeper
//     grazing angle -> larger offset; monotone.
//   * MARCH INTERSECTION CORRECTNESS: the refined point lies ON the height field (rayDepth ≈ 1 - h)
//     within the march tolerance, no overshoot past the surface.
//   * SELF-SHADOW: a crevice texel with the light occluded by a ridge -> shadow < 1; an exposed texel
//     -> 1 (and the degenerate exits return 1 exactly); monotone in heightScale.
//   * DETERMINISM: same inputs -> same UV/shadow (pure functions, no RNG/time).
#include "render/pom.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace pom = hf::render::pom;
using hf::math::Vec2;
using hf::math::Vec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool eqExact2(const Vec2& a, const Vec2& b) { return a.x == b.x && a.y == b.y; }
static float len2(const Vec2& a, const Vec2& b) {
    float dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

// --- Deterministic procedural height field (mirrors the showcase's in-engine brick/groove field). A
// smooth "raised brick" valued in [0,1]: a sinusoidal mortar groove in u and v dips toward 0 at the
// cell borders and rises toward 1 in the brick centers. Smooth (no discontinuity) so the march/refine
// converge cleanly and the intersection-on-surface test is well-defined. ---
static float BrickHeight(Vec2 uv) {
    float fu = uv.x - std::floor(uv.x);
    float fv = uv.y - std::floor(uv.y);
    // Smooth bumps centered in each cell: sin(pi*f)^k peaks at the center, dips to 0 at the borders.
    float bu = std::sin(3.14159265358979323846f * fu);
    float bv = std::sin(3.14159265358979323846f * fv);
    float h = bu * bu * bv * bv;            // in [0,1], 1 at cell center, 0 at the mortar lines
    return std::min(std::max(h, 0.0f), 1.0f);
}

// A single recessed step: h == 1 (flat top) for u < 0.5, dropping to a low plateau for u > 0.5. Used to
// build an explicit ridge for the self-shadow test (a ridge to the light's side casts into the recess).
static float StepHeight(Vec2 uv) {
    float fu = uv.x - std::floor(uv.x);
    return (fu < 0.5f) ? 1.0f : 0.15f;
}

int main() {
    HF_TEST_MAIN_INIT();
    const int kSteps = 32;

    // ---- ZERO HEIGHT = IDENTITY (the core equivalence proof). ----
    {
        const Vec2 uvs[] = {{0.0f, 0.0f}, {0.13f, 0.78f}, {0.5f, 0.5f}, {0.91f, 0.22f}};
        const Vec3 views[] = {
            {0.3f, 0.1f, 0.9f},        // grazing
            {-0.6f, 0.4f, 0.5f},       // steep grazing
            {0.0f, 0.0f, 1.0f},        // straight on
            {0.05f, -0.2f, 0.7f},
        };
        bool allIdentity = true;
        for (const Vec2& uv : uvs)
            for (const Vec3& v : views) {
                Vec2 r = pom::ParallaxUV(uv, v, 0.0f, kSteps, BrickHeight);
                if (!eqExact2(r, uv)) allIdentity = false;
            }
        check(allIdentity, "ZERO HEIGHT: ParallaxUV(heightScale=0) == baseUV EXACTLY for all uv/view");

        // A STRAIGHT-ON view (viewDirTangent.xy == 0) -> baseUV EXACTLY even with heightScale > 0.
        bool straightOn = true;
        for (const Vec2& uv : uvs) {
            Vec2 r = pom::ParallaxUV(uv, Vec3{0.0f, 0.0f, 1.0f}, 0.08f, kSteps, BrickHeight);
            if (!eqExact2(r, uv)) straightOn = false;
        }
        check(straightOn, "STRAIGHT-ON view (viewDir.xy==0) -> baseUV EXACTLY (no lateral march)");
    }

    // ---- OFFSET DIRECTION + MAGNITUDE. ----
    {
        // A grazing view leaning in +u,+v. With heightScale>0 the returned UV must shift OFF baseUV, and
        // (since the texel slides opposite the view lean) in the -maxOffset direction, i.e. -u,-v here.
        // A point on the brick SLOPE (height strictly in (0,1)) so the ray actually descends into the
        // field and the march produces an offset (a brick-center top texel h==1 has surfDepth 0 and
        // correctly returns baseUV — that's the flat-top fast path, not a parallax point).
        const Vec2 baseUV{0.3f, 0.3f};
        const Vec3 graze{0.5f, 0.5f, 0.5f};            // 45-deg lean in u and v
        Vec2 r = pom::ParallaxUV(baseUV, graze, 0.06f, kSteps, BrickHeight);
        check(len2(r, baseUV) > 1e-5f, "heightScale>0 + grazing view -> UV is offset from baseUV");
        check(r.x <= baseUV.x && r.y <= baseUV.y,
              "the parallax offset is in the tangent-space view-lean direction (-u,-v here)");

        // STEEPER grazing angle (smaller .z, larger xy/z) -> LARGER offset (monotone in the lean).
        Vec2 rShallow = pom::ParallaxUV(baseUV, Vec3{0.3f, 0.0f, 0.95f}, 0.06f, kSteps, BrickHeight);
        Vec2 rSteep   = pom::ParallaxUV(baseUV, Vec3{0.3f, 0.0f, 0.45f}, 0.06f, kSteps, BrickHeight);
        check(len2(rSteep, baseUV) > len2(rShallow, baseUV),
              "a steeper grazing angle yields a larger parallax offset (monotone)");

        // DEEPER height amplitude (larger heightScale) -> LARGER offset (more parallax travel).
        Vec2 rLow  = pom::ParallaxUV(baseUV, graze, 0.03f, kSteps, BrickHeight);
        Vec2 rHigh = pom::ParallaxUV(baseUV, graze, 0.10f, kSteps, BrickHeight);
        check(len2(rHigh, baseUV) > len2(rLow, baseUV),
              "a larger heightScale yields a larger parallax offset (deeper field, more travel)");
    }

    // ---- MARCH INTERSECTION CORRECTNESS: the refined UV lies ON the height field, no overshoot. ----
    {
        // For a set of grazing views, the returned UV must satisfy rayDepth ≈ (1 - h(uv)) at the refined
        // depth. We recompute the rayDepth implied by the offset and compare to the surface depth there.
        const Vec2 baseUV{0.35f, 0.6f};
        const float heightScale = 0.07f;
        const Vec3 views[] = {{0.5f, 0.2f, 0.6f}, {-0.4f, 0.5f, 0.55f}, {0.6f, -0.3f, 0.5f}};
        bool onSurface = true;
        for (const Vec3& v : views) {
            Vec2 uv = pom::ParallaxUV(baseUV, v, heightScale, kSteps, BrickHeight);
            // Recover the rayDepth from the UV offset: offset = maxOffset * rayDepth, with
            // maxOffset = (v.xy/v.z)*heightScale. Use the larger component to avoid /0.
            float mox = (v.x / v.z) * heightScale;
            float moy = (v.y / v.z) * heightScale;
            float rayDepth = (std::fabs(mox) > std::fabs(moy))
                                 ? (baseUV.x - uv.x) / mox
                                 : (baseUV.y - uv.y) / moy;
            float surfDepth = 1.0f - BrickHeight(uv);
            // The refine converges to rayDepth == surfDepth; allow one march-layer of tolerance.
            float tol = 1.5f / (float)kSteps;
            if (std::fabs(rayDepth - surfDepth) > tol) onSurface = false;
            // No overshoot past the bottom of the field.
            if (rayDepth > 1.0f + 1e-3f) onSurface = false;
        }
        check(onSurface, "refined intersection lies ON the height field (rayDepth ≈ 1-h), no overshoot");
    }

    // ---- SELF-SHADOW. ----
    {
        // A texel just INSIDE the recess (u just past the step) with the light coming from the +u side
        // (toward the ridge at u<0.5): the ridge blocks the light -> shadow < 1.
        const Vec2 recess{0.55f, 0.5f};               // on the low plateau, near the ridge
        const Vec3 lightToward{-1.0f, 0.0f, 0.6f};    // toward -u (over the ridge) and up
        float sCrevice = pom::SelfShadow(recess, lightToward, 0.5f, kSteps, StepHeight);
        check(sCrevice < 1.0f, "a recessed texel with a ridge between it and the light is shadowed (<1)");
        check(sCrevice >= 0.0f, "self-shadow factor stays >= 0");

        // An EXPOSED texel on the flat top, light overhead-ish away from any ridge -> fully lit.
        const Vec2 exposed{0.2f, 0.5f};               // on the flat top (h==1)
        const Vec3 lightUp{0.2f, 0.0f, 1.0f};
        float sExposed = pom::SelfShadow(exposed, lightUp, 0.5f, kSteps, StepHeight);
        check(sExposed == 1.0f, "an exposed flat-top texel is fully lit (shadow == 1 exactly)");

        // DEGENERATE exits return 1.0 EXACTLY.
        check(pom::SelfShadow(recess, lightToward, 0.0f, kSteps, StepHeight) == 1.0f,
              "heightScale==0 -> no self-occlusion (shadow == 1 exactly)");
        check(pom::SelfShadow(recess, Vec3{0.0f, 0.0f, 1.0f}, 0.5f, kSteps, StepHeight) == 1.0f,
              "light straight overhead (lightDir.xy==0) -> shadow == 1 exactly");
        check(pom::SelfShadow(recess, Vec3{-1.0f, 0.0f, -0.2f}, 0.5f, kSteps, StepHeight) == 1.0f,
              "light below the tangent plane (lightDir.z<=0) -> shadow == 1 exactly");

        // MONOTONE in heightScale: a deeper field can only add occlusion (shadow non-increasing).
        float sShallow = pom::SelfShadow(recess, lightToward, 0.25f, kSteps, StepHeight);
        float sDeep    = pom::SelfShadow(recess, lightToward, 0.75f, kSteps, StepHeight);
        check(sDeep <= sShallow + 1e-6f, "self-shadow is non-increasing in heightScale (monotone)");
    }

    // ---- DETERMINISM: same inputs -> bit-identical outputs. ----
    {
        const Vec2 baseUV{0.42f, 0.31f};
        const Vec3 v{0.4f, 0.2f, 0.6f};
        Vec2 a = pom::ParallaxUV(baseUV, v, 0.06f, kSteps, BrickHeight);
        Vec2 b = pom::ParallaxUV(baseUV, v, 0.06f, kSteps, BrickHeight);
        check(eqExact2(a, b), "ParallaxUV is deterministic (bit-identical across runs)");
        float s1 = pom::SelfShadow(Vec2{0.55f, 0.5f}, Vec3{-1.0f, 0.0f, 0.6f}, 0.5f, kSteps, StepHeight);
        float s2 = pom::SelfShadow(Vec2{0.55f, 0.5f}, Vec3{-1.0f, 0.0f, 0.6f}, 0.5f, kSteps, StepHeight);
        check(s1 == s2, "SelfShadow is deterministic (bit-identical across runs)");
    }

    if (g_fail == 0) { std::printf("pom_test: all checks passed\n"); return 0; }
    std::printf("pom_test: %d FAILURES\n", g_fail);
    return 1;
}
