// Slice BN — data-driven post-process stack. Pure CPU math: the JSON config loader (ordered effect
// list + params, empty=pass-through, order preserved, unknown kind rejected), the ColorGrade
// lift/gamma/gain curve (identity + a known grade), FilmGrain determinism (pure function of the integer
// pixel coord, bounded, zero-intensity no-op), ChromaticAberration (zero strength / center -> no
// aberration), and the stack-apply CPU mirror (== composing the per-effect evaluators in order). No
// device, ASan-eligible (links hf_core). Mirrors the math the --poststack-shot showcase + the
// post_stack.frag shader use (engine/render/post_stack.h), as ssr_test/decal_test pin ssr.h/decal.h.
#include "render/post_stack.h"
#include "math/math.h"

#include <cmath>
#include <cstdint>
#include <cstdio>

using namespace hf::math;
namespace post = hf::render::post;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }
static bool approx3(const Vec3& a, const Vec3& b, float eps = 1e-4f) {
    return approx(a.x, b.x, eps) && approx(a.y, b.y, eps) && approx(a.z, b.z, eps);
}

int main() {
    // ================= Config: LoadPostStack ==================================================
    {
        // Ordered effect list with params, order preserved.
        const char* js = R"({"effects":[
            {"kind":"tonemap","exposure":1.7},
            {"kind":"colorgrade","lift":[-0.02,0,0.03],"gamma":[1,1,1],"gain":[1.1,1.02,0.9]},
            {"kind":"chromatic","strength":2.0},
            {"kind":"vignette","outer":0.8,"inner":0.35},
            {"kind":"grain","intensity":0.05}
        ]})";
        post::LoadResult r = post::LoadPostStack(js);
        check(r.ok, "valid stack JSON parses");
        check(r.stack.size() == 5, "stack has 5 effects");
        check(r.stack.effects[0].kind == post::Kind::Tonemap, "effect 0 is tonemap");
        check(r.stack.effects[1].kind == post::Kind::ColorGrade, "effect 1 is colorgrade");
        check(r.stack.effects[2].kind == post::Kind::ChromaticAberration, "effect 2 is chromatic");
        check(r.stack.effects[3].kind == post::Kind::Vignette, "effect 3 is vignette");
        check(r.stack.effects[4].kind == post::Kind::FilmGrain, "effect 4 is grain (order preserved)");

        // Params parsed.
        check(approx(r.stack.effects[0].exposure, 1.7f), "tonemap exposure parsed");
        check(approx3(r.stack.effects[1].lift, Vec3{-0.02f, 0.0f, 0.03f}), "colorgrade lift parsed");
        check(approx3(r.stack.effects[1].gain, Vec3{1.1f, 1.02f, 0.9f}), "colorgrade gain parsed");
        check(approx(r.stack.effects[2].strength, 2.0f), "chromatic strength parsed");
        check(approx(r.stack.effects[3].vignetteInner, 0.35f), "vignette inner parsed");
        check(approx(r.stack.effects[4].intensity, 0.05f), "grain intensity parsed");
    }
    {
        // Empty / absent effects -> pass-through stack.
        post::LoadResult a = post::LoadPostStack(R"({"effects":[]})");
        check(a.ok && a.stack.empty(), "empty effects array -> pass-through stack");
        post::LoadResult b = post::LoadPostStack(R"({})");
        check(b.ok && b.stack.empty(), "absent effects -> pass-through stack");
    }
    {
        // Unknown effect kind is rejected.
        post::LoadResult r = post::LoadPostStack(R"({"effects":[{"kind":"bogus"}]})");
        check(!r.ok, "unknown effect kind fails the load");
    }

    // ================= ColorGrade math ========================================================
    {
        // Identity params (lift 0, gamma 1, gain 1) -> input unchanged.
        Vec3 in{0.2f, 0.5f, 0.8f};
        Vec3 out = post::ApplyColorGrade(in, {0, 0, 0}, {1, 1, 1}, {1, 1, 1});
        check(approx3(out, in), "ColorGrade identity leaves input unchanged");

        // Known lift/gamma/gain -> expected: out = gain * pow(max(in+lift,0), 1/gamma).
        Vec3 lift{0.1f, 0.0f, -0.1f}, gamma{2.0f, 1.0f, 0.5f}, gain{1.5f, 1.0f, 2.0f};
        Vec3 g = post::ApplyColorGrade(in, lift, gamma, gain);
        float ex = 1.5f * std::pow(std::max(0.2f + 0.1f, 0.0f), 1.0f / 2.0f);
        float ey = 1.0f * std::pow(std::max(0.5f + 0.0f, 0.0f), 1.0f / 1.0f);
        float ez = 2.0f * std::pow(std::max(0.8f - 0.1f, 0.0f), 1.0f / 0.5f);
        check(approx3(g, Vec3{ex, ey, ez}), "ColorGrade matches gain*pow(max(in+lift,0),1/gamma)");

        // Negative (in+lift) clamps to 0 before the power (no NaN).
        Vec3 nclamp = post::ApplyColorGrade(Vec3{0.0f, 0.0f, 0.0f}, {-0.5f, -0.5f, -0.5f},
                                            {2.0f, 2.0f, 2.0f}, {1, 1, 1});
        check(approx3(nclamp, Vec3{0, 0, 0}), "ColorGrade clamps negative base to 0");
    }

    // ================= FilmGrain determinism ==================================================
    {
        // grain(pixel) is a PURE function of the integer pixel coord: same coord -> same value.
        float a1 = post::Grain(37, 101, 1.0f);
        float a2 = post::Grain(37, 101, 1.0f);
        check(a1 == a2, "grain is deterministic for the same pixel coord");

        // Different coords generally differ (sample a handful).
        check(post::Grain(0, 0, 1.0f) != post::Grain(1, 0, 1.0f), "grain varies across x");
        check(post::Grain(0, 0, 1.0f) != post::Grain(0, 1, 1.0f), "grain varies across y");

        // Zero intensity -> no change.
        Vec3 c{0.4f, 0.5f, 0.6f};
        Vec3 z = post::ApplyFilmGrain(c, 12, 34, 0.0f);
        check(approx3(z, c), "zero-intensity grain leaves the color unchanged");

        // hash01 is bounded [0,1) and (hash-0.5) in [-0.5,0.5]; scan a grid.
        bool bounded = true;
        for (uint32_t y = 0; y < 64; ++y)
            for (uint32_t x = 0; x < 64; ++x) {
                float h = post::GrainHash01(x, y);
                if (h < 0.0f || h >= 1.0f) bounded = false;
                float s = (h - 0.5f);
                if (s < -0.5f || s > 0.5f) bounded = false;
            }
        check(bounded, "grain hash in [0,1), signed grain in [-0.5,0.5]");

        // The additive grain at intensity I is bounded by +/-0.5*I.
        bool ampBounded = true;
        for (uint32_t y = 0; y < 64; ++y)
            for (uint32_t x = 0; x < 64; ++x) {
                float gg = post::Grain(x, y, 0.05f);
                if (std::fabs(gg) > 0.5f * 0.05f + 1e-7f) ampBounded = false;
            }
        check(ampBounded, "grain amplitude bounded by 0.5*intensity");
    }

    // ================= ChromaticAberration =====================================================
    {
        Vec2 texel{1.0f / 256.0f, 1.0f / 256.0f};
        // A sampler that returns the UV encoded as color so we can detect offset sampling.
        auto sampler = [](Vec2 uv) -> Vec3 { return {uv.x, uv.y, uv.x + uv.y}; };

        // Zero strength -> no offset: R/G/B all sample uv -> color == sample(uv).
        Vec2 uv{0.75f, 0.6f};
        Vec3 zero = post::ApplyChromaticAberration(uv, 0.0f, texel, sampler);
        Vec3 base = sampler(uv);
        check(approx3(zero, base), "zero strength -> no aberration");

        // Center pixel (uv == 0.5) -> radial direction 0 -> no offset.
        Vec2 off = post::ChromaticOffset(Vec2{0.5f, 0.5f}, 10.0f, texel);
        check(approx(off.x, 0.0f) && approx(off.y, 0.0f), "center pixel has zero radial offset");
        Vec3 ctr = post::ApplyChromaticAberration(Vec2{0.5f, 0.5f}, 10.0f, texel, sampler);
        Vec3 ctrBase = sampler(Vec2{0.5f, 0.5f});
        check(approx3(ctr, ctrBase), "center pixel -> ~no aberration");

        // Off-center with strength > 0 -> the R/B channels sample shifted UVs along the radial dir.
        Vec2 offN = post::ChromaticOffset(uv, 4.0f, texel);
        check(!(approx(offN.x, 0.0f) && approx(offN.y, 0.0f)), "off-center has a nonzero offset");
        // Offset is along the radial direction (uv-0.5), so its sign matches (uv-0.5).
        check((offN.x > 0.0f) == (uv.x - 0.5f > 0.0f), "offset x along radial direction");
        check((offN.y > 0.0f) == (uv.y - 0.5f > 0.0f), "offset y along radial direction");
    }

    // ================= Vignette ================================================================
    {
        // Center is bright (factor ~1), corner is dark (factor ~0).
        float ctr = post::VignetteFactor(Vec2{0.5f, 0.5f}, 0.8f, 0.35f);
        float corner = post::VignetteFactor(Vec2{1.0f, 1.0f}, 0.8f, 0.35f);
        check(approx(ctr, 1.0f), "vignette factor ~1 at center");
        check(corner < ctr, "vignette darkens toward the corner");
    }

    // ================= Stack apply (CPU mirror == composing per-effect in order) ================
    {
        Vec2 texel{1.0f / 256.0f, 1.0f / 256.0f};
        Vec2 uv{0.7f, 0.55f};
        uint32_t px = 179, py = 140;
        auto sampler = [](Vec2 q) -> Vec3 {
            // A smooth synthetic scene color in [0,1]-ish HDR.
            return {0.3f + 0.2f * q.x, 0.4f + 0.1f * q.y, 0.5f};
        };

        post::PostStack stack = post::DefaultShowcaseStack();
        Vec3 got = post::ApplyStack(stack, uv, texel, px, py, sampler);

        // Compose the same effects BY HAND in the same order.
        Vec3 c = sampler(uv);
        c = post::ApplyTonemap(c, 1.7f);
        c = post::ApplyColorGrade(c, Vec3{-0.02f, 0.0f, 0.03f}, Vec3{1, 1, 1}, Vec3{1.10f, 1.02f, 0.90f});
        c = post::ApplyChromaticAberration(uv, 2.0f, texel, sampler);
        c = post::ApplyVignette(c, uv, 0.8f, 0.35f);
        c = post::ApplyFilmGrain(c, px, py, 0.05f);
        check(approx3(got, c, 1e-5f), "ApplyStack == hand-composed per-effect chain in order");

        // The stack is deterministic: two evaluations match exactly.
        Vec3 got2 = post::ApplyStack(stack, uv, texel, px, py, sampler);
        check(got.x == got2.x && got.y == got2.y && got.z == got2.z, "ApplyStack is deterministic");

        // An empty stack is a pass-through (returns sample(uv)).
        post::PostStack empty;
        Vec3 pass = post::ApplyStack(empty, uv, texel, px, py, sampler);
        check(approx3(pass, sampler(uv)), "empty stack is a pass-through");
    }

    if (g_fail == 0) std::printf("post_stack_test: all checks passed\n");
    else std::printf("post_stack_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
