// Slice DN — DDGI Slice 5: the GI COMPOSITE. Pure CPU math: the composite-specific properties of the
// single-bounce DDGI indirect-diffuse term the lit_ddgi.frag.hlsl shader adds as its LAST contribution —
//   indirect = InterpolateIrradianceSH(wpos, N) * (1 - metallic)
//   rgb     += indirect * albedo * giStrength
// where InterpolateIrradianceSH = probesh::InterpolateIrradiance (the VERBATIM in-shader copy of
// probegi::NearestProbes + probesh::InterpolateSH + probesh::SHEvaluate). This test exercises the EXACT CPU
// mirror of the composite the shader runs; the GPU==CPU bit-exact blend itself is proven (memcmp) by the DL
// --probeinterp-shot showcase, and the giStrength=0==no-GI BYTE-IDENTICAL render proof by the DN --ddgi-shot
// showcase. Here we pin the COMPOSITE-LEVEL invariants (per the DN spec §5):
//   * giStrength=0 identity: indirect*albedo*0 == 0.0 EXACTLY (the no-op / 77-goldens-untouched guarantee).
//   * Bleed direction: a red-dominant ProbeSH toward a facing normal -> a red-dominant indirect color.
//   * probeCount=0 (dimX=0): the disabled-path InterpolateIrradiance -> {0,0,0} -> zero indirect.
//   * metallic=1: the (1-metallic) factor zeroes the indirect term (metals take no indirect diffuse).
//   * Determinism: same inputs -> bit-identical indirect color (two evaluations memcmp-equal).
// No device, ASan-eligible (links hf_core).
#include "render/probe_gi.h"
#include "render/probe_sh.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace probegi = hf::render::probegi;
namespace probesh = hf::render::probesh;
using hf::math::Vec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// The EXACT composite the lit_ddgi.frag.hlsl shader computes per pixel (CPU mirror): the indirect-diffuse
// contribution added to rgb. Returns indirect*albedo*giStrength (the value `rgb +=` accumulates).
static Vec3 CompositeContribution(const Vec3& wpos, const Vec3& N, const probegi::ProbeGrid& grid,
                                  const probesh::ProbeSH* probes, int probeCount, const Vec3& albedo,
                                  float metallic, float giStrength) {
    Vec3 irr = probesh::InterpolateIrradiance(wpos, grid, probes, probeCount, N);
    Vec3 indirect = irr * (1.0f - metallic);
    return Vec3{indirect.x * albedo.x * giStrength,
                indirect.y * albedo.y * giStrength,
                indirect.z * albedo.z * giStrength};
}

int main() {
    HF_TEST_MAIN_INIT();
    probegi::ProbeGrid grid;
    grid.origin = Vec3{-2.0f, 0.0f, -2.0f};
    grid.dimX = 2; grid.dimY = 2; grid.dimZ = 2; grid.spacing = 4.0f;
    const int probeCount = grid.probeCount();   // 8

    // A RED-dominant set of probes: a strong, equal positive band-0 (DC/ambient) term per probe with the
    // red channel dominating. Band-0 with cosineLobe scale kA0=pi reconstructs to a positive, direction-
    // independent irradiance that is red-dominant -> the canonical color-bleed source.
    std::vector<probesh::ProbeSH> redProbes((size_t)probeCount);
    std::memset(redProbes.data(), 0, redProbes.size() * sizeof(probesh::ProbeSH));
    for (int p = 0; p < probeCount; ++p) {
        redProbes[p].coeff[0][0] = 0.8f;   // band-0 red
        redProbes[p].coeff[0][1] = 0.1f;   // band-0 green
        redProbes[p].coeff[0][2] = 0.1f;   // band-0 blue
    }

    const Vec3 wpos{0.0f, 2.0f, 0.0f};                 // interior point
    const Vec3 N = hf::math::normalize(Vec3{1.0f, 0.2f, 0.0f});  // a facing normal
    const Vec3 albedo{0.8f, 0.8f, 0.8f};               // neutral floor

    // ====================================================================================
    // (1) giStrength=0 identity: the composite contribution is EXACTLY {0,0,0}.
    // ====================================================================================
    {
        Vec3 c = CompositeContribution(wpos, N, grid, redProbes.data(), probeCount, albedo,
                                       /*metallic=*/0.0f, /*giStrength=*/0.0f);
        // Bit-exact zero (a multiply by the literal float 0): the byte-identical no-op the 77-goldens-
        // untouched guarantee rests on. Check the exact float bits, not an epsilon.
        check(c.x == 0.0f && c.y == 0.0f && c.z == 0.0f, "giStrength=0 -> indirect contribution == 0.0 exactly");
    }

    // ====================================================================================
    // (2) Bleed direction: a red-dominant ProbeSH toward a facing normal -> a red-dominant indirect color
    //     (and a STRICTLY POSITIVE one with giStrength>0, so it actually lightens the floor).
    // ====================================================================================
    {
        Vec3 c = CompositeContribution(wpos, N, grid, redProbes.data(), probeCount, albedo,
                                       /*metallic=*/0.0f, /*giStrength=*/1.0f);
        check(c.x > 0.0f, "red-dominant probe -> positive indirect (the bounce lightens the floor)");
        check(c.x > c.y && c.x > c.z, "red-dominant probe -> red-dominant indirect color (bleed direction)");
        // The raw irradiance (pre-albedo/strength) is also red-dominant.
        Vec3 irr = probesh::InterpolateIrradiance(wpos, grid, redProbes.data(), probeCount, N);
        check(irr.x > irr.y && irr.x > irr.z, "InterpolateIrradiance from a red probe is red-dominant");
    }

    // ====================================================================================
    // (3) probeCount=0 (dimX=0): the disabled-path InterpolateIrradiance -> {0,0,0} -> zero indirect even at
    //     giStrength>0 (the DK fallback re-asserted in the composite context).
    // ====================================================================================
    {
        probegi::ProbeGrid empty = grid; empty.dimX = 0;
        Vec3 c = CompositeContribution(wpos, N, empty, redProbes.data(), /*probeCount=*/empty.probeCount(),
                                       albedo, /*metallic=*/0.0f, /*giStrength=*/1.0f);
        check(c.x == 0.0f && c.y == 0.0f && c.z == 0.0f, "probeCount=0 (dimX=0) -> zero indirect");
        // And the probeCount<=0 argument guard alone (valid grid, but count claimed 0) also zeros it.
        Vec3 c2 = CompositeContribution(wpos, N, grid, redProbes.data(), /*probeCount=*/0, albedo,
                                        0.0f, 1.0f);
        check(c2.x == 0.0f && c2.y == 0.0f && c2.z == 0.0f, "probeCount<=0 arg -> zero indirect");
    }

    // ====================================================================================
    // (4) metallic=1: the (1-metallic) factor zeroes the indirect term exactly (metals take no indirect
    //     diffuse), regardless of giStrength.
    // ====================================================================================
    {
        Vec3 c = CompositeContribution(wpos, N, grid, redProbes.data(), probeCount, albedo,
                                       /*metallic=*/1.0f, /*giStrength=*/1.0f);
        check(c.x == 0.0f && c.y == 0.0f && c.z == 0.0f, "metallic=1 -> (1-metallic) zeroes the indirect term");
    }

    // ====================================================================================
    // (5) Determinism: the same inputs produce a bit-identical indirect color (the GI term is world-anchored
    //     probe data, deterministic; the shader's mad path mirrors std::fma).
    // ====================================================================================
    {
        Vec3 a = CompositeContribution(wpos, N, grid, redProbes.data(), probeCount, albedo, 0.0f, 0.75f);
        Vec3 b = CompositeContribution(wpos, N, grid, redProbes.data(), probeCount, albedo, 0.0f, 0.75f);
        check(std::memcmp(&a, &b, sizeof(Vec3)) == 0, "composite contribution is deterministic (bit-identical)");
    }

    if (g_fail == 0) std::printf("ddgi_test: OK\n");
    else             std::printf("ddgi_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
