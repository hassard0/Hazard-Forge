// Slice CG — depth of field. Pure CPU math: the thin-lens circle-of-confusion (CoC) in pixels +
// the scatter-as-gather blur weight. No device, ASan-eligible (links hf_core). Mirrors the math the
// --dof-shot showcase and dof.frag use (engine/render/dof.h).
//
// Properties pinned (per the spec):
//   * CoC == 0 exactly at the focal plane (depth == focalDist) -> the focal subject stays sharp.
//   * CoC grows with |depth - focalDist|, monotonically on EACH side (near depth<focalDist and far
//     depth>focalDist), and is clamped to [0, maxCoCpx].
//   * Both a NEAR object (depth < focalDist) and a FAR object (depth > focalDist) have CoC > 0, and
//     the formula is robust (no NaN/Inf) at depth -> 0 and depth -> focalLength.
//   * BlurWeight gather: a tap whose CoC COVERS the center (tapDistPx <= tapCoCpx) contributes; a tap
//     outside its CoC contributes ~0; a focal (CoC ~ 0) neighbor does NOT blur the center.
//   * Determinism: same inputs -> identical CoC (pure function, no RNG/time).
#include "render/dof.h"
#include <cmath>
#include <cstdio>

namespace dof = hf::render::dof;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool finite(float x) { return std::isfinite(x); }

int main() {
    // Fixed lens params (matching the showcase family): focal distance 6 view-units, an f-stop-ish
    // aperture, a 50mm-ish focal length expressed in the same view units the CoC formula uses, and a
    // max CoC of 16 px. The exact units are documented in dof.h; the test only relies on the relations.
    // `aperture` here folds in the view-units -> screen-pixels scale (as the showcase passes it), so the
    // saturated far field (depth -> inf) reaches ~aperture*focalLen/(focalDist-focalLen) which exceeds
    // maxCoCpx and clamps. The exact scaling is documented in dof.h; the test only needs the relations.
    const float focalDist  = 6.0f;
    const float aperture   = 120.0f;
    const float focalLen   = 1.2f;   // < focalDist (a real lens: object distance > focal length)
    const float maxCoCpx   = 16.0f;

    // ---- CoC == 0 at the focal plane (sharp). ----
    {
        float c = dof::CircleOfConfusion(focalDist, focalDist, aperture, focalLen, maxCoCpx);
        check(c == 0.0f, "CoC is exactly 0 at depth == focalDist");
    }

    // ---- CoC grows with |depth - focalDist|, monotone on the FAR side (depth > focalDist). ----
    {
        float prev = -1.0f;
        bool monotone = true, allPos = true;
        for (float depth = focalDist + 0.25f; depth <= focalDist + 12.0f; depth += 0.25f) {
            float c = dof::CircleOfConfusion(depth, focalDist, aperture, focalLen, maxCoCpx);
            if (!(c > 0.0f)) allPos = false;
            if (c + 1e-6f < prev) monotone = false;   // non-decreasing as depth recedes
            prev = c;
        }
        check(allPos, "far CoC (depth > focalDist) is > 0");
        check(monotone, "far CoC is monotonically non-decreasing with depth");
    }

    // ---- CoC grows with |depth - focalDist|, monotone on the NEAR side (depth < focalDist). ----
    // As depth moves from focalDist down toward the camera, defocus increases -> CoC non-decreasing.
    {
        float prev = -1.0f;
        bool monotone = true, allPos = true;
        for (float depth = focalDist - 0.25f; depth >= 0.5f; depth -= 0.25f) {
            float c = dof::CircleOfConfusion(depth, focalDist, aperture, focalLen, maxCoCpx);
            if (!(c > 0.0f)) allPos = false;
            if (c + 1e-6f < prev) monotone = false;   // non-decreasing as depth approaches the camera
            prev = c;
        }
        check(allPos, "near CoC (depth < focalDist) is > 0");
        check(monotone, "near CoC is monotonically non-decreasing toward the camera");
    }

    // ---- A larger defocus magnitude gives a larger CoC than a smaller one (each side). ----
    {
        float farNear = dof::CircleOfConfusion(focalDist + 1.0f, focalDist, aperture, focalLen, maxCoCpx);
        float farFar  = dof::CircleOfConfusion(focalDist + 4.0f, focalDist, aperture, focalLen, maxCoCpx);
        check(farFar > farNear, "farther-from-focus -> larger CoC (far side)");
        float nearClose = dof::CircleOfConfusion(focalDist - 1.0f, focalDist, aperture, focalLen, maxCoCpx);
        float nearFar   = dof::CircleOfConfusion(focalDist - 3.0f, focalDist, aperture, focalLen, maxCoCpx);
        check(nearFar > nearClose, "closer-to-camera -> larger CoC (near side)");
    }

    // ---- CoC is clamped at maxCoCpx (a hugely-defocused far object saturates, never exceeds). ----
    {
        float c = dof::CircleOfConfusion(1000.0f, focalDist, aperture, focalLen, maxCoCpx);
        check(c <= maxCoCpx + 1e-4f, "CoC never exceeds maxCoCpx");
        check(c >= maxCoCpx - 1e-3f, "a far-defocused object saturates near maxCoCpx");
        // A small aperture clamps to a smaller-or-equal max too (no overflow with a tiny maxCoC).
        float c2 = dof::CircleOfConfusion(1000.0f, focalDist, aperture, focalLen, 4.0f);
        check(c2 <= 4.0f + 1e-4f, "CoC clamps to a smaller maxCoCpx");
    }

    // ---- Robustness: no NaN/Inf at the singular inputs the spec calls out. ----
    {
        float cZero = dof::CircleOfConfusion(0.0f, focalDist, aperture, focalLen, maxCoCpx);
        check(finite(cZero), "CoC is finite at depth -> 0 (no divide-by-zero NaN)");
        check(cZero >= 0.0f && cZero <= maxCoCpx + 1e-4f, "CoC at depth 0 stays in [0,maxCoC]");
        float cFL = dof::CircleOfConfusion(focalLen, focalDist, aperture, focalLen, maxCoCpx);
        check(finite(cFL), "CoC is finite at depth == focalLength");
        // depth exactly negative / behind camera also must not blow up.
        float cNeg = dof::CircleOfConfusion(-2.0f, focalDist, aperture, focalLen, maxCoCpx);
        check(finite(cNeg) && cNeg >= 0.0f, "CoC is finite + non-negative for a behind-camera depth");
    }

    // ---- Determinism: the pure function returns the identical bits for identical inputs. ----
    {
        float a = dof::CircleOfConfusion(8.7f, focalDist, aperture, focalLen, maxCoCpx);
        float b = dof::CircleOfConfusion(8.7f, focalDist, aperture, focalLen, maxCoCpx);
        check(a == b, "CoC is deterministic (identical inputs -> identical bits)");
    }

    // ---- BlurWeight: scatter-as-gather. A neighbor tap contributes to the center iff its CoC disk
    //      covers the center pixel (tapDistPx <= tapCoCpx). ----
    {
        // A tap 3px away with a CoC of 8px COVERS the center -> contributes.
        float inside = dof::BlurWeight(8.0f, 3.0f);
        check(inside > 0.0f, "a tap whose CoC covers the center contributes (>0)");
        // A tap 10px away with a CoC of 4px does NOT reach the center -> ~0.
        float outside = dof::BlurWeight(4.0f, 10.0f);
        check(outside <= 1e-4f, "a tap outside its CoC contributes ~0");
        // The center tap itself (dist 0) with any positive CoC contributes.
        float self = dof::BlurWeight(8.0f, 0.0f);
        check(self > 0.0f, "the center tap (dist 0) contributes");
        check(self >= inside - 1e-4f, "weight is non-increasing with distance within the CoC");
    }

    // ---- A focal (CoC ~ 0) NEIGHBOR does not blur the center: a sharp tap can't spread. ----
    {
        // A neighbor exactly at the focal plane has ~0 CoC; even at distance 1px it can't cover center.
        float focalNeighbor = dof::BlurWeight(0.0f, 1.0f);
        check(focalNeighbor <= 1e-4f, "a focal (CoC~0) neighbor does not blur the center");
        // But that same focal neighbor DOES contribute to ITS OWN pixel (dist 0) -> stays sharp.
        float selfFocal = dof::BlurWeight(0.0f, 0.0f);
        check(selfFocal > 0.0f, "a focal pixel still gathers itself (dist 0) so it stays sharp");
    }

    // ---- BlurWeight determinism. ----
    {
        float a = dof::BlurWeight(6.5f, 2.5f);
        float b = dof::BlurWeight(6.5f, 2.5f);
        check(a == b, "BlurWeight is deterministic");
    }

    if (g_fail == 0) { std::printf("dof_test: all checks passed\n"); return 0; }
    std::printf("dof_test: %d FAILURES\n", g_fail);
    return 1;
}
