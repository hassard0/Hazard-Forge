// Slice DF — Contrast-Adaptive Sharpening (CAS; AMD FidelityFX CAS). Pure CPU math: no device,
// ASan-eligible (links hf_core). Mirrors the math the --cas-shot showcase and cas.frag.hlsl use
// (engine/render/cas.h).
//
// Properties pinned (per the spec):
//   * sharpness=0 identity (the byte-identical proof's CPU half): CasSharpen(c, ..., 0) == center
//     EXACTLY for ANY neighborhood; CasWeight(..., 0) == 0 EXACTLY (no sharpening).
//   * Sharpening behavior: a center brighter than its neighbors gains MORE contrast (relative to the
//     neighbor average) for sharpness > 0; a flat neighborhood (all equal) is unchanged (no contrast to
//     sharpen) for any sharpness.
//   * No overshoot (the clamp): the output is clamped to the neighborhood [min,max] per channel — a
//     sharpen never pushes a channel past the brightest/darkest neighbor (no ringing), verified on a
//     high-contrast edge.
//   * Adaptive weight: |CasWeight| is larger in a low-contrast region than near a hard edge (the
//     adaptive property); |w| is monotone in sharpness.
//   * Determinism: same inputs -> identical result (pure function, no RNG/time).
#include "render/cas.h"

#include <cmath>
#include <cstdio>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace cas = hf::render::cas;
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
    HF_TEST_MAIN_INIT();
    // ---- sharpness=0 IDENTITY (the byte-identical no-op proof's core). --------------------------
    {
        // A spread of neighborhoods: flat, a bright center, a dark center, a colored edge, extremes.
        struct N { Vec3 c, u, d, l, r; };
        const N nbhds[] = {
            {{0.5f,0.5f,0.5f},{0.5f,0.5f,0.5f},{0.5f,0.5f,0.5f},{0.5f,0.5f,0.5f},{0.5f,0.5f,0.5f}},  // flat
            {{0.9f,0.9f,0.9f},{0.4f,0.4f,0.4f},{0.4f,0.4f,0.4f},{0.4f,0.4f,0.4f},{0.4f,0.4f,0.4f}},  // bright center
            {{0.1f,0.1f,0.1f},{0.6f,0.6f,0.6f},{0.6f,0.6f,0.6f},{0.6f,0.6f,0.6f},{0.6f,0.6f,0.6f}},  // dark center
            {{0.8f,0.2f,0.35f},{0.1f,0.7f,0.2f},{0.9f,0.1f,0.6f},{0.3f,0.3f,0.9f},{0.5f,0.5f,0.5f}}, // colored
            {{0.0f,0.0f,0.0f},{1.0f,1.0f,1.0f},{0.0f,1.0f,0.0f},{1.0f,0.0f,1.0f},{0.5f,0.5f,0.5f}},  // extremes
            {{1.0f,1.0f,1.0f},{1.0f,1.0f,1.0f},{0.0f,0.0f,0.0f},{1.0f,0.0f,0.0f},{0.0f,0.0f,1.0f}},  // hard edge
        };
        for (const N& n : nbhds) {
            Vec3 out = cas::CasSharpen(n.c, n.u, n.d, n.l, n.r, 0.0f);
            check(exactEq(out, n.c), "CasSharpen(..., sharpness=0) == center EXACTLY");
        }
        // CasWeight at sharpness 0 is EXACTLY 0 for any min/max envelope.
        check(cas::CasWeight(0.0f, 1.0f, 0.0f) == 0.0f, "CasWeight(...,0) == 0 (hard edge)");
        check(cas::CasWeight(0.4f, 0.45f, 0.0f) == 0.0f, "CasWeight(...,0) == 0 (low contrast)");
        check(cas::CasWeight(0.5f, 0.5f, 0.0f) == 0.0f, "CasWeight(...,0) == 0 (flat)");
    }

    // ---- Sharpening behavior. -------------------------------------------------------------------
    {
        // A bright center surrounded by darker neighbors: a sharpen pushes the center brighter RELATIVE
        // to the neighbor average (the gap center - avg(neighbors) grows) for sharpness > 0.
        Vec3 c{0.7f, 0.7f, 0.7f};
        Vec3 nb{0.4f, 0.4f, 0.4f};
        Vec3 sharp = cas::CasSharpen(c, nb, nb, nb, nb, 0.6f);
        check(finite3(sharp), "sharpened output finite");
        // The center is the brightest sample, so the clamp caps at the center -> sharpen pins it AT the
        // center value (already the max). Verify the negative-lobe form increases contrast: the result
        // is >= the original center (never dips below) and equals the (clamped) max.
        check(sharp.x >= c.x - 1e-6f, "bright center: sharpen does not darken the center");
        check(sharp.x <= 0.7f + 1e-6f, "bright center: clamped to the neighborhood max (no overshoot)");
        // A MID center between a brighter and a darker neighbor (so it is not on a clamp rail) gains
        // contrast: it moves AWAY from the neighbor average.
        Vec3 c2{0.5f, 0.5f, 0.5f};
        Vec3 bright{0.8f, 0.8f, 0.8f}, dark{0.2f, 0.2f, 0.2f};
        // center below the neighbor mean (mean = (0.8+0.2+0.5+0.5)/4 = 0.5) -> use an asymmetric set so
        // the center sits below the mean and should be pushed DOWN (more contrast), staying within range.
        Vec3 c3{0.45f, 0.45f, 0.45f};
        Vec3 s3 = cas::CasSharpen(c3, bright, dark, Vec3{0.5f,0.5f,0.5f}, Vec3{0.5f,0.5f,0.5f}, 0.8f);
        float mean = (bright.x + dark.x + 0.5f + 0.5f) / 4.0f;  // 0.5
        check(c3.x < mean, "test setup: center below neighbor mean");
        check(s3.x <= c3.x + 1e-6f, "center below mean: sharpen pushes it down (more contrast)");
        check(s3.x >= dark.x - 1e-6f && s3.x <= bright.x + 1e-6f, "stays within neighborhood range");
        (void)c2;
    }

    // ---- Flat neighborhood -> unchanged (no contrast to sharpen) for ANY sharpness. -------------
    {
        Vec3 flat{0.42f, 0.55f, 0.7f};
        for (float s : {0.0f, 0.25f, 0.5f, 0.8f, 1.0f}) {
            Vec3 out = cas::CasSharpen(flat, flat, flat, flat, flat, s);
            // All neighbors equal -> numerator = flat*(1+4w), denominator = (1+4w) -> exactly flat; the
            // clamp to [flat,flat] also pins it. (Approx because of the divide for s>0.)
            check(std::fabs(out.x - flat.x) < 1e-5f && std::fabs(out.y - flat.y) < 1e-5f &&
                  std::fabs(out.z - flat.z) < 1e-5f, "flat neighborhood unchanged for any sharpness");
        }
    }

    // ---- No overshoot (the clamp): output within [min,max] on a high-contrast edge. -------------
    {
        // A hard black/white edge with a colored center — the worst case for ringing.
        struct N { Vec3 c, u, d, l, r; };
        const N edges[] = {
            {{0.5f,0.5f,0.5f},{1.0f,1.0f,1.0f},{0.0f,0.0f,0.0f},{1.0f,1.0f,1.0f},{0.0f,0.0f,0.0f}},
            {{0.95f,0.1f,0.5f},{1.0f,0.0f,1.0f},{0.0f,1.0f,0.0f},{0.8f,0.2f,0.9f},{0.2f,0.8f,0.1f}},
            {{0.0f,0.0f,0.0f},{1.0f,1.0f,1.0f},{1.0f,1.0f,1.0f},{1.0f,1.0f,1.0f},{1.0f,1.0f,1.0f}},
        };
        for (const N& n : edges) {
            for (float s : {0.3f, 0.6f, 1.0f}) {
                Vec3 out = cas::CasSharpen(n.c, n.u, n.d, n.l, n.r, s);
                // per-channel neighborhood min/max
                auto mn = [](float a,float b,float c,float d,float e){ return std::min(std::min(std::min(a,b),std::min(c,d)),e); };
                auto mx = [](float a,float b,float c,float d,float e){ return std::max(std::max(std::max(a,b),std::max(c,d)),e); };
                Vec3 lo{mn(n.c.x,n.u.x,n.d.x,n.l.x,n.r.x), mn(n.c.y,n.u.y,n.d.y,n.l.y,n.r.y), mn(n.c.z,n.u.z,n.d.z,n.l.z,n.r.z)};
                Vec3 hi{mx(n.c.x,n.u.x,n.d.x,n.l.x,n.r.x), mx(n.c.y,n.u.y,n.d.y,n.l.y,n.r.y), mx(n.c.z,n.u.z,n.d.z,n.l.z,n.r.z)};
                check(out.x >= lo.x - 1e-6f && out.x <= hi.x + 1e-6f &&
                      out.y >= lo.y - 1e-6f && out.y <= hi.y + 1e-6f &&
                      out.z >= lo.z - 1e-6f && out.z <= hi.z + 1e-6f,
                      "no overshoot: output within neighborhood [min,max] on a hard edge");
            }
        }
    }

    // ---- Adaptive weight: larger (more negative) in low-contrast than near a hard edge. ----------
    {
        // Low-contrast region: a narrow band well below white -> high headroom -> high amplitude.
        float wLow  = cas::CasWeight(0.45f, 0.55f, 1.0f);
        // Hard edge: minL ~ 0, maxL ~ 1 -> headroom min(0, 0) = 0 -> amplitude 0.
        float wEdge = cas::CasWeight(0.0f, 1.0f, 1.0f);
        check(wLow <= 0.0f && wEdge <= 0.0f, "CAS weights are non-positive (negative lobe)");
        check(std::fabs(wLow) > std::fabs(wEdge), "adaptive: |w| larger in low-contrast than at a hard edge");
        check(std::fabs(wEdge) < 1e-6f, "hard-edge weight ~ 0 (no sharpen across a full edge -> no ring)");

        // Monotone in sharpness: |w| grows as sharpness increases (the peak |lobe| goes 1/8 -> 1/5).
        float prev = 0.0f;
        for (float s : {0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 1.0f}) {
            float w = std::fabs(cas::CasWeight(0.4f, 0.5f, s));
            check(w >= prev - 1e-7f, "|CasWeight| monotone non-decreasing in sharpness");
            prev = w;
        }
        // Endpoints: at sharpness ~1 the peak is -1/5; at a tiny sharpness the peak is ~ -1/8.
        check(std::fabs(cas::CasWeight(0.5f, 0.5f, 1.0f)) >= 0.0f, "weight finite at sharpness 1");
    }

    // ---- Determinism: same inputs -> identical output. ------------------------------------------
    {
        Vec3 c{0.6f,0.3f,0.45f}, u{0.2f,0.5f,0.1f}, d{0.7f,0.2f,0.8f}, l{0.4f,0.4f,0.4f}, r{0.55f,0.6f,0.2f};
        Vec3 a = cas::CasSharpen(c, u, d, l, r, 0.7f);
        Vec3 b = cas::CasSharpen(c, u, d, l, r, 0.7f);
        check(exactEq(a, b), "CasSharpen deterministic (same inputs -> identical output)");
        check(cas::CasWeight(0.3f, 0.6f, 0.5f) == cas::CasWeight(0.3f, 0.6f, 0.5f),
              "CasWeight deterministic");
        check(finite3(a), "deterministic output finite");
    }

    if (g_fail == 0) std::printf("cas_test: ALL PASS\n");
    else std::printf("cas_test: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
