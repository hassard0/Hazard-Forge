// Slice DR — DDGI MULTI-BOUNCE (2nd light bounce). Pure CPU logic/math for the 2nd-bounce feed the NEW
// probe_bake_gi.frag adds and the SH-buffer selection the --ddgimb-shot composite uses, mirrored in
// render/probe_multibounce.h. Pins (per the DR spec §5):
//   * SelectBounceSH(1) -> SH0 (the bounce-1 capture is SKIPPED -> the DN single-bounce path; this is the
//     make-or-break byte-identical no-op selector); SelectBounceSH(2) -> SH1 (the 2nd bounce).
//   * ClampBounceCount: <1 -> 1, >2 -> 2 (exactly 2 captures max — YAGNI).
//   * 2nd-bounce MONOTONICITY (the CPU mirror): the bounce-1 capture ADDS a non-negative indirect
//     (SH0 irradiance * albedo * giStrength, all >= 0) to a non-negative direct radiance, so an SH1 built
//     from (direct + added) reconstructs to an irradiance >= the SH0 (direct-only) irradiance for a
//     positive-albedo scene — the bounce only ADDS light, never darkens.
//   * probeCount=0 (dimX=0) -> zero indirect feed (the DN zero-SH fallback) -> SH1 == SH0 (no 2nd bounce).
//   * Determinism: same inputs -> bit-identical added indirect (two evaluations memcmp-equal).
// No device, ASan-eligible (links hf_core; reuses render/probe_multibounce.h + probe_gi.h + probe_sh.h).
#include "render/probe_multibounce.h"
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
namespace probemb = hf::render::probemb;
using hf::math::Vec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();
    probegi::ProbeGrid grid;
    grid.origin = Vec3{-2.0f, 0.0f, -2.0f};
    grid.dimX = 2; grid.dimY = 2; grid.dimZ = 2; grid.spacing = 4.0f;
    const int probeCount = grid.probeCount();   // 8

    // SH0: a positive-DC (band-0) direct capture — a slightly red-dominant ambient, the canonical color-
    // bleed source. SH1 (the bounce-1 SH) is distinct (it carries MORE light) so the pointer select is
    // unambiguous.
    std::vector<probesh::ProbeSH> sh0((size_t)probeCount);
    std::vector<probesh::ProbeSH> sh1((size_t)probeCount);
    std::memset(sh0.data(), 0, sh0.size() * sizeof(probesh::ProbeSH));
    std::memset(sh1.data(), 0, sh1.size() * sizeof(probesh::ProbeSH));
    for (int p = 0; p < probeCount; ++p) {
        sh0[p].coeff[0][0] = 0.6f; sh0[p].coeff[0][1] = 0.2f; sh0[p].coeff[0][2] = 0.2f;
        // SH1 = SH0 plus the (non-negative) 2nd-bounce contribution -> strictly more DC light.
        sh1[p].coeff[0][0] = 0.9f; sh1[p].coeff[0][1] = 0.3f; sh1[p].coeff[0][2] = 0.3f;
    }

    // ====================================================================================
    // (1) SelectBounceSH: bounceCount=1 -> SH0 (the no-op selector); bounceCount>=2 -> SH1.
    // ====================================================================================
    {
        check(probemb::SelectBounceSH(1, sh0.data(), sh1.data()) == sh0.data(),
              "SelectBounceSH(1) -> SH0 (bounce-1 skipped -> the DN single-bounce no-op)");
        check(probemb::SelectBounceSH(2, sh0.data(), sh1.data()) == sh1.data(),
              "SelectBounceSH(2) -> SH1 (the 2nd bounce)");
        // bounceCount<1 clamps to 1 -> SH0; bounceCount>2 clamps to 2 -> SH1.
        check(probemb::SelectBounceSH(0, sh0.data(), sh1.data()) == sh0.data(),
              "SelectBounceSH(0) clamps to 1 -> SH0");
        check(probemb::SelectBounceSH(5, sh0.data(), sh1.data()) == sh1.data(),
              "SelectBounceSH(5) clamps to 2 -> SH1");
        // The no-op MUST select SH0 even when sh1 is null (it is never dereferenced at bounceCount=1).
        check(probemb::SelectBounceSH<probesh::ProbeSH>(1, sh0.data(), nullptr) == sh0.data(),
              "SelectBounceSH(1) -> SH0 with a null SH1 (no deref on the no-op path)");
    }

    // ====================================================================================
    // (2) ClampBounceCount: <1 -> 1, >kMaxBounces -> kMaxBounces, in-range unchanged.
    // ====================================================================================
    {
        check(probemb::ClampBounceCount(-3) == 1, "ClampBounceCount(-3) -> 1");
        check(probemb::ClampBounceCount(0) == 1, "ClampBounceCount(0) -> 1");
        check(probemb::ClampBounceCount(1) == 1, "ClampBounceCount(1) -> 1");
        check(probemb::ClampBounceCount(2) == 2, "ClampBounceCount(2) -> 2");
        check(probemb::ClampBounceCount(99) == probemb::kMaxBounces, "ClampBounceCount(99) -> kMaxBounces");
    }

    const Vec3 wpos{0.0f, 2.0f, 0.0f};
    const Vec3 N = hf::math::normalize(Vec3{1.0f, 0.2f, 0.0f});
    const Vec3 albedo{0.8f, 0.8f, 0.8f};
    const float giStrength = 3.0f;

    // ====================================================================================
    // (3) 2nd-bounce MONOTONICITY (the CPU mirror): the bounce-1 added indirect is component-wise >= 0
    //     (the bounce only ADDS light) and STRICTLY positive for the positive-albedo red-dominant SH0.
    // ====================================================================================
    {
        Vec3 added = probemb::BounceIndirect(wpos, N, grid, sh0.data(), probeCount, albedo, giStrength);
        check(probemb::BounceAddsLight(added), "bounce-1 added indirect is component-wise >= 0 (only adds light)");
        check(added.x > 0.0f && added.y > 0.0f && added.z > 0.0f,
              "positive-albedo red-dominant SH0 -> strictly positive 2nd-bounce add (brighter)");
        check(added.x > added.y && added.x > added.z, "the 2nd-bounce add inherits SH0's red-dominant bleed");

        // The SH1 irradiance (direct + the non-negative add, mirrored as a band-0 boost) is >= the SH0
        // irradiance toward N — the brighter-never-darker guarantee the golden's visible 2nd bounce rests on.
        Vec3 irr0 = probesh::InterpolateIrradiance(wpos, grid, sh0.data(), probeCount, N);
        Vec3 irr1 = probesh::InterpolateIrradiance(wpos, grid, sh1.data(), probeCount, N);
        check(irr1.x >= irr0.x && irr1.y >= irr0.y && irr1.z >= irr0.z,
              "SH1 irradiance >= SH0 irradiance (2nd-bounce monotonicity: brighter, never darker)");
        check(irr1.x > irr0.x, "SH1 is STRICTLY brighter than SH0 for the positive-albedo scene");
    }

    // ====================================================================================
    // (4) probeCount=0 (dimX=0): the disabled-path bounce feed -> {0,0,0} (no 2nd bounce added).
    // ====================================================================================
    {
        probegi::ProbeGrid empty = grid; empty.dimX = 0;
        Vec3 added = probemb::BounceIndirect(wpos, N, empty, sh0.data(), empty.probeCount(), albedo, giStrength);
        check(added.x == 0.0f && added.y == 0.0f && added.z == 0.0f,
              "probeCount=0 (dimX=0) -> zero 2nd-bounce indirect (the DN zero-SH fallback)");
        // The probeCount<=0 argument guard alone also zeros it.
        Vec3 added2 = probemb::BounceIndirect(wpos, N, grid, sh0.data(), /*probeCount=*/0, albedo, giStrength);
        check(added2.x == 0.0f && added2.y == 0.0f && added2.z == 0.0f,
              "probeCount<=0 arg -> zero 2nd-bounce indirect");
    }

    // ====================================================================================
    // (5) Determinism: the same inputs produce a bit-identical added indirect (world-anchored probe data;
    //     the shader's mad path mirrors std::fma).
    // ====================================================================================
    {
        Vec3 a = probemb::BounceIndirect(wpos, N, grid, sh0.data(), probeCount, albedo, giStrength);
        Vec3 b = probemb::BounceIndirect(wpos, N, grid, sh0.data(), probeCount, albedo, giStrength);
        check(std::memcmp(&a, &b, sizeof(Vec3)) == 0, "the 2nd-bounce add is deterministic (bit-identical)");
    }

    if (g_fail == 0) std::printf("ddgi_multibounce_test: OK\n");
    else             std::printf("ddgi_multibounce_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
