// Slice CO — Order-Independent Transparency (Weighted Blended OIT). Pure CPU math: the McGuire depth
// Weight, the per-fragment Accumulate (additive accum + multiplicative revealage), and the ResolveOver
// composite. No device, ASan-eligible (links hf_core). Mirrors the math the --oit-shot showcase + the
// oit_accum/oit_resolve shaders use (engine/render/oit.h).
//
// Properties pinned (per the spec):
//   * Weight: POSITIVE + FINITE for all inputs; DECREASING in depth (a nearer fragment weighted >= a
//     farther one at equal alpha); scales with alpha; handles alpha 0..1.
//   * ORDER INDEPENDENCE (the core proof): accumulate a fragment set in order A vs a PERMUTED order ->
//     the resolved color is BIT-IDENTICAL (accum is a SUM, revealage a PRODUCT — both commutative, and
//     the chosen fragment set sums bit-stably across orderings). Multiple permutations.
//   * Resolve correctness: a single opaque-ish fragment (alpha~1) resolves ~ its own color; a single
//     near-transparent fragment (alpha~0) resolves ~ the background; revealage 1 (no fragments) ->
//     background EXACTLY.
//   * Determinism: same fragments -> same resolve (pure functions, no RNG/time).
#include "render/oit.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace oit = hf::render::oit;
using hf::math::Vec3;
using hf::math::Vec4;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool finite3(const Vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}
static bool eqExact3(const Vec3& a, const Vec3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;   // BIT-identical (== on float)
}

// One transparent fragment: straight color + straight alpha + view-space linear depth. The showcase
// stores its overlapping glass layers exactly this way.
struct Frag { Vec3 color; float alpha; float viewDepth; };

// Accumulate a fragment list in the GIVEN index order, then resolve over `bg`. This is the EXACT CPU
// mirror of what the GPU does: weight each fragment (oit::Weight), premultiply, Accumulate (additive
// accum + multiplicative revealage), then ResolveOver. Returned color must be order-independent.
static Vec3 AccumulateAndResolve(const std::vector<Frag>& frags, const std::vector<int>& order,
                                 const Vec3& bg) {
    Vec4 accum{0, 0, 0, 0};
    float revealage = 1.0f;
    for (int idx : order) {
        const Frag& f = frags[(size_t)idx];
        float w = oit::Weight(f.viewDepth, f.alpha);
        Vec4 premult{f.color.x * f.alpha, f.color.y * f.alpha, f.color.z * f.alpha, f.alpha};
        oit::Accumulate(premult, w, accum, revealage);
    }
    return oit::ResolveOver(accum, revealage, bg);
}

int main() {
    HF_TEST_MAIN_INIT();
    // ---- Weight: positive + finite for all inputs. ----
    {
        const float depths[] = {-5.0f, 0.0f, 0.001f, 1.0f, 10.0f, 90.0f, 500.0f, 5000.0f};
        const float alphas[] = {0.0f, 0.01f, 0.25f, 0.5f, 0.9f, 1.0f};
        bool allPosFinite = true;
        for (float d : depths)
            for (float a : alphas) {
                float w = oit::Weight(d, a);
                if (!std::isfinite(w) || w < 0.0f) allPosFinite = false;
            }
        check(allPosFinite, "Weight is positive + finite for all (depth, alpha) inputs");
        // alpha 0 -> weight 0 (no contribution); alpha 1 -> the full depth term.
        check(oit::Weight(10.0f, 0.0f) == 0.0f, "Weight(depth, alpha=0) == 0");
        check(oit::Weight(10.0f, 1.0f) > 0.0f, "Weight(depth, alpha=1) > 0");
    }

    // ---- Weight: DECREASING in depth (nearer >= farther at equal alpha). ----
    {
        const float a = 0.6f;
        // A sweep of increasing depths: each weight must be >= the next (monotone non-increasing).
        const float depths[] = {1.0f, 5.0f, 20.0f, 50.0f, 100.0f, 200.0f, 400.0f};
        bool monotone = true;
        for (size_t k = 0; k + 1 < sizeof(depths) / sizeof(depths[0]); ++k) {
            float wNear = oit::Weight(depths[k], a);
            float wFar  = oit::Weight(depths[k + 1], a);
            if (!(wNear >= wFar)) monotone = false;
        }
        check(monotone, "Weight is non-increasing in depth (nearer fragment weighted >= farther)");
        // A strictly-nearer fragment in the un-saturated band is strictly heavier.
        check(oit::Weight(5.0f, a) > oit::Weight(120.0f, a),
              "a near fragment is strictly heavier than a far one (un-saturated band)");
        // Scales with alpha: more opaque -> more weight, at equal depth.
        check(oit::Weight(20.0f, 0.9f) > oit::Weight(20.0f, 0.2f),
              "Weight scales with alpha (more opaque carries more weight)");
    }

    // ---- ORDER INDEPENDENCE (the core proof): canonical vs PERMUTED -> BIT-identical resolve. ----
    {
        // A fixed transparent set (5 overlapping glass layers) chosen for BIT-EXACT order stability
        // (the spec's option (a): "choose object alphas/weights whose premultiplied contributions sum
        // without rounding divergence across orders"). Float addition is NOT associative, so an
        // arbitrary depth/color set diverges by ~1 ULP across orderings. To make the per-fragment
        // additive accum (accum += premultColor*weight) BIT-EXACT regardless of order we pin:
        //   * a COMMON view depth (here 0) + a COMMON alpha (0.5) -> oit::Weight returns the IDENTICAL
        //     constant for every fragment. At depth 0 the McGuire depth term saturates the clamp
        //     ceiling EXACTLY (0.03/1e-5 = 3000), so weight == 0.5*3000 == 1500.0 (a clean value).
        //   * DYADIC colors (multiples of 1/4) -> each contribution color*0.5*1500 == color*750 is an
        //     EXACT small value (e.g. 0.75*750 = 562.5, 0.5*750 = 375) and any partial sum of a few of
        //     them stays exactly representable in float (well within 24 mantissa bits) -> the running
        //     sum is identical for every permutation, BIT-for-BIT. revealage == (1-0.5)^5 = 0.03125 is
        //     a power of two (also order-exact under the product). This is the SAME bit-stable input
        //     discipline the showcase uses for its internal permuted==canonical SHA proof.
        std::vector<Frag> frags = {
            {Vec3{0.75f, 0.25f, 0.25f}, 0.50f, 0.0f},   // red-ish
            {Vec3{0.25f, 0.75f, 0.25f}, 0.50f, 0.0f},   // green-ish
            {Vec3{0.25f, 0.50f, 1.00f}, 0.50f, 0.0f},   // blue-ish
            {Vec3{1.00f, 0.75f, 0.25f}, 0.50f, 0.0f},   // yellow-ish
            {Vec3{0.75f, 0.25f, 1.00f}, 0.50f, 0.0f},   // magenta-ish
        };
        const Vec3 bg{0.25f, 0.25f, 0.25f};   // dyadic background -> the final composite stays exact too

        std::vector<int> canonical = {0, 1, 2, 3, 4};
        Vec3 ref = AccumulateAndResolve(frags, canonical, bg);
        check(finite3(ref), "canonical-order resolve is finite");

        // Several permutations -> each must be BIT-IDENTICAL to the canonical resolve.
        const std::vector<std::vector<int>> perms = {
            {3, 1, 0, 2, 4},
            {4, 3, 2, 1, 0},   // full reverse
            {2, 0, 4, 1, 3},
            {1, 4, 0, 3, 2},
            {0, 2, 4, 1, 3},
        };
        bool allIdentical = true;
        for (const std::vector<int>& p : perms) {
            Vec3 r = AccumulateAndResolve(frags, p, bg);
            if (!eqExact3(r, ref)) {
                allIdentical = false;
                std::printf("  order-dependence: perm resolve (%.9g,%.9g,%.9g) != ref (%.9g,%.9g,%.9g)\n",
                            r.x, r.y, r.z, ref.x, ref.y, ref.z);
            }
        }
        check(allIdentical,
              "ORDER INDEPENDENCE: every permuted draw order resolves BIT-IDENTICALLY (sum+product)");
    }

    // ---- Resolve correctness. ----
    {
        const Vec3 bg{0.10f, 0.20f, 0.30f};
        // revealage 1 (NO fragments) -> background EXACTLY (no transparent coverage).
        {
            Vec4 accum{0, 0, 0, 0};
            Vec3 r = oit::ResolveOver(accum, 1.0f, bg);
            check(eqExact3(r, bg), "revealage 1 (no fragments) resolves to the background EXACTLY");
        }
        // A single near-OPAQUE fragment (alpha ~ 1) resolves ~ its own color (background ~ hidden).
        {
            Frag f{Vec3{0.80f, 0.15f, 0.55f}, 0.999f, 8.0f};
            Vec3 r = AccumulateAndResolve({f}, {0}, bg);
            float dc = std::fabs(r.x - f.color.x) + std::fabs(r.y - f.color.y) + std::fabs(r.z - f.color.z);
            check(dc < 0.02f, "a single alpha~1 fragment resolves ~ its own color");
        }
        // A single near-TRANSPARENT fragment (alpha ~ 0) resolves ~ the background.
        {
            Frag f{Vec3{0.80f, 0.15f, 0.55f}, 0.001f, 8.0f};
            Vec3 r = AccumulateAndResolve({f}, {0}, bg);
            float db = std::fabs(r.x - bg.x) + std::fabs(r.y - bg.y) + std::fabs(r.z - bg.z);
            check(db < 0.02f, "a single alpha~0 fragment resolves ~ the background");
        }
        // The resolve is always finite (guarded divide by max(accum.a, eps)).
        {
            Vec4 accum{0.0f, 0.0f, 0.0f, 0.0f};   // zero accum, revealage < 1 (degenerate)
            Vec3 r = oit::ResolveOver(accum, 0.5f, bg);
            check(finite3(r), "ResolveOver is finite even for a zero-accum / guarded divisor");
        }
    }

    // ---- Determinism: same fragments -> bit-identical resolve. ----
    {
        std::vector<Frag> frags = {
            {Vec3{0.7f, 0.3f, 0.2f}, 0.4f, 7.0f},
            {Vec3{0.2f, 0.6f, 0.9f}, 0.6f, 11.0f},
            {Vec3{0.5f, 0.5f, 0.5f}, 0.5f, 14.0f},
        };
        const Vec3 bg{0.05f, 0.05f, 0.08f};
        std::vector<int> ord = {0, 1, 2};
        Vec3 a = AccumulateAndResolve(frags, ord, bg);
        Vec3 b = AccumulateAndResolve(frags, ord, bg);
        check(eqExact3(a, b), "resolve is deterministic (bit-identical across runs)");
        // Weight + Accumulate are deterministic too.
        check(oit::Weight(13.0f, 0.4f) == oit::Weight(13.0f, 0.4f), "Weight is deterministic");
    }

    if (g_fail == 0) { std::printf("oit_test: all checks passed\n"); return 0; }
    std::printf("oit_test: %d FAILURES\n", g_fail);
    return 1;
}
