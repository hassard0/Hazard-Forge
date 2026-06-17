// Slice CX — Volumetric Shadows (sun light shafts through the fog). Pure CPU math: the per-froxel
// sun-shadow VISIBILITY gate (render::froxel::SunVisibility) the froxel inject (froxel_inject.comp.hlsl
// behind the volumetricShadows flag) multiplies into the sun in-scatter, plus the CSM cascade selection
// (render::froxel::SelectCascade) that mirrors lit_csm.frag's split pick. No device, ASan-eligible
// (links hf_core). Exercises the EXACT shadow-sampling convention the lit pass + the inject shader use
// (project worldPos by the cascade lightViewProj -> smUV/curDepth -> depth-compare-with-bias), against a
// PROCEDURAL shadow field, so the math is pinned before any GPU work.
//
// Properties pinned (per the spec):
//   * LIT froxel -> visibility 1: a froxel CLOSER to the sun than the stored occluder -> 1; a degenerate
//     / out-of-cascade froxel (off the shadow map) -> 1 (the lit pass's shadow==1 default).
//   * SHADOWED froxel -> visibility 0: a froxel BEHIND a closer occluder (beyond the bias) -> 0.
//   * BIAS -> no self-shadow acne: a froxel AT the occluder depth (within the bias) -> 1 (no acne).
//   * CASCADE selection: a froxel at a known view depth selects the expected cascade; the light-space
//     projection round-trips (a point IN the cascade frustum projects to in-[0,1] UV + depth).
//   * DETERMINISM: same inputs -> bit-identical visibility (pure functions, no RNG/time).
//   * The existing CS/CV froxel cases are unaffected (this file only ADDS the CX gate; froxel_test.cpp
//     still pins the CS/CV math).
#include "render/froxel.h"
#include "render/csm.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace froxel = hf::render::froxel;
namespace csm = hf::render::csm;
using hf::math::Vec3;
using hf::math::Mat4;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();
    // A fixed directional sun shadow projection: ortho fit around the origin, looking along the sun's
    // travel direction (the SAME Ortho*LookAt the --volshadows-shot / lit pass builds). Vulkan clip:
    // depth [0,1], Y-flip baked.
    const Vec3 sunTravel = hf::math::normalize(Vec3{-0.3f, -0.9f, -0.25f});
    const Vec3 sceneCenter{0.0f, 0.5f, -2.0f};
    const Vec3 lightEye = sceneCenter - sunTravel * 30.0f;
    Mat4 lightView = Mat4::LookAt(lightEye, sceneCenter, {0, 1, 0});
    Mat4 lightOrtho = Mat4::Ortho(-22.0f, 22.0f, -22.0f, 22.0f, 1.0f, 60.0f);
    Mat4 sunVP = lightOrtho * lightView;

    const float kBias = 0.0025f;   // mirrors lit.frag's shadow bias

    // Helper: the light-space depth a world point projects to (curDepth), for building the procedural
    // occluder field self-consistently with SunVisibility's own projection.
    auto lightDepthOf = [&](const Vec3& wp) -> float {
        float w = 1.0f;
        Vec3 p = hf::math::MulPointDivide(sunVP, wp, w);
        return p.z;
    };
    auto lightUVOf = [&](const Vec3& wp, float& u, float& v) {
        float w = 1.0f;
        Vec3 p = hf::math::MulPointDivide(sunVP, wp, w);
        u = p.x * 0.5f + 0.5f; v = p.y * 0.5f + 0.5f;
    };

    // ---- LIT: a froxel CLOSER to the sun than the stored occluder -> visibility 1. The occluder field
    //      stores depth 1.0 everywhere (the far plane == nothing occludes), so every in-range froxel is
    //      lit. ----
    {
        auto farField = [](float, float) { return 1.0f; };
        const Vec3 froxelW{0.0f, 1.0f, -2.0f};
        float vis = froxel::SunVisibility(froxelW, sunVP, farField, kBias);
        check(vis == 1.0f, "LIT: froxel closer to the sun than the occluder -> visibility 1");
    }

    // ---- DEGENERATE / out-of-cascade -> 1 (forced visibility). A world point projecting OUTSIDE the
    //      ortho frustum (far to the side) lands off the [0,1] shadow UV / depth -> treated as LIT. ----
    {
        auto anyField = [](float, float) { return 0.0f; };   // even a "fully occluding" field
        const Vec3 wayOff{500.0f, 500.0f, 500.0f};
        float vis = froxel::SunVisibility(wayOff, sunVP, anyField, kBias);
        check(vis == 1.0f, "DEGENERATE: out-of-cascade froxel -> forced visibility 1 (no off-map shadow)");
    }

    // ---- SHADOWED: a froxel BEHIND a closer occluder -> visibility 0. Place an occluder closer to the
    //      sun (smaller light-space depth) than the froxel, beyond the bias. ----
    {
        const Vec3 froxelW{0.0f, 1.0f, -2.0f};
        float froxelDepth = lightDepthOf(froxelW);
        // The occluder sits clearly in front (toward the sun): a smaller stored depth, well beyond bias.
        float occluderDepth = froxelDepth - 0.05f;
        auto occField = [&](float, float) { return occluderDepth; };
        float vis = froxel::SunVisibility(froxelW, sunVP, occField, kBias);
        check(vis == 0.0f, "SHADOWED: froxel behind a closer occluder (beyond bias) -> visibility 0");
        // Monotone in the gap: pulling the occluder even further toward the sun stays shadowed.
        auto occField2 = [&](float, float) { return froxelDepth - 0.2f; };
        check(froxel::SunVisibility(froxelW, sunVP, occField2, kBias) == 0.0f,
              "SHADOWED: a farther-in-front occluder stays shadowed");
    }

    // ---- BIAS: a froxel AT the occluder depth (within the bias) -> 1 (no self-shadow acne). The stored
    //      occluder == the froxel's own depth (a surface shadow-mapping itself); the bias prevents acne. ----
    {
        const Vec3 froxelW{0.0f, 1.0f, -2.0f};
        float froxelDepth = lightDepthOf(froxelW);
        auto selfField = [&](float, float) { return froxelDepth; };  // exactly at the occluder
        check(froxel::SunVisibility(froxelW, sunVP, selfField, kBias) == 1.0f,
              "BIAS: froxel AT the occluder depth -> visibility 1 (no self-shadow acne)");
        // Just inside the bias window (occluder slightly in front, less than bias) -> still lit.
        auto nearField = [&](float, float) { return froxelDepth - kBias * 0.5f; };
        check(froxel::SunVisibility(froxelW, sunVP, nearField, kBias) == 1.0f,
              "BIAS: occluder within the bias window -> visibility 1 (no acne)");
        // Just past the bias window -> shadowed (the bias is a hard threshold, not a no-op).
        auto pastField = [&](float, float) { return froxelDepth - kBias * 2.0f; };
        check(froxel::SunVisibility(froxelW, sunVP, pastField, kBias) == 0.0f,
              "BIAS: occluder just past the bias window -> visibility 0 (bias is a finite threshold)");
    }

    // ---- The occluder field is sampled at the froxel's projected smUV (self-consistent projection):
    //      build a field that occludes ONLY at the froxel's projected UV and is clear elsewhere; the
    //      froxel must read its OWN tile -> shadowed. ----
    {
        const Vec3 froxelW{3.0f, 1.0f, -4.0f};
        float fu, fv; lightUVOf(froxelW, fu, fv);
        float froxelDepth = lightDepthOf(froxelW);
        auto tileField = [&](float u, float v) {
            // Occlude (depth in front) only near the froxel's own UV; clear (far) elsewhere.
            return (std::fabs(u - fu) < 0.02f && std::fabs(v - fv) < 0.02f)
                       ? (froxelDepth - 0.05f) : 1.0f;
        };
        check(froxel::SunVisibility(froxelW, sunVP, tileField, kBias) == 0.0f,
              "PROJECTION: froxel samples its OWN projected smUV tile -> shadowed there");
    }

    // ---- CASCADE selection mirrors lit_csm.frag: smallest cascade whose split far-distance contains the
    //      view depth; last cascade is the catch-all; single-cascade always 0. ----
    {
        // 4 increasing splits (view-space far distances).
        float splits[4] = {5.0f, 15.0f, 40.0f, 90.0f};
        check(froxel::SelectCascade(2.0f, splits, 4) == 0, "CASCADE: depth in cascade 0 -> 0");
        check(froxel::SelectCascade(5.0f, splits, 4) == 0, "CASCADE: depth == split0 (inclusive) -> 0");
        check(froxel::SelectCascade(10.0f, splits, 4) == 1, "CASCADE: depth in cascade 1 -> 1");
        check(froxel::SelectCascade(30.0f, splits, 4) == 2, "CASCADE: depth in cascade 2 -> 2");
        check(froxel::SelectCascade(80.0f, splits, 4) == 3, "CASCADE: depth in cascade 3 -> 3");
        check(froxel::SelectCascade(1000.0f, splits, 4) == 3, "CASCADE: beyond last split -> last (catch-all)");
        // Single-cascade (the --volshadows-shot showcase) always 0.
        float one[1] = {90.0f};
        check(froxel::SelectCascade(1.0f, one, 1) == 0, "CASCADE: single cascade, near -> 0");
        check(froxel::SelectCascade(89.0f, one, 1) == 0, "CASCADE: single cascade, far -> 0");
        // Matches the csm::CsmSplits ordering (strictly increasing) the lit pass builds.
        auto cs = csm::CsmSplits(0.5f, 90.0f, 4, 0.5f);
        check(cs[0] < cs[1] && cs[1] < cs[2] && cs[2] < cs[3],
              "CASCADE: csm::CsmSplits strictly increasing (the splits SelectCascade consumes)");
    }

    // ---- LIGHT-SPACE ROUND-TRIP: a point inside the ortho frustum projects to in-[0,1] UV + depth (so
    //      SunVisibility actually samples the map, not the degenerate early-out). ----
    {
        const Vec3 inFrustum{0.0f, 0.5f, -2.0f};   // the scene center -> near the cascade center
        float u, v; lightUVOf(inFrustum, u, v);
        float d = lightDepthOf(inFrustum);
        check(u >= 0.0f && u <= 1.0f && v >= 0.0f && v <= 1.0f && d >= 0.0f && d <= 1.0f,
              "ROUND-TRIP: a point in the cascade frustum -> in-[0,1] light-space UV + depth");
        // And SunVisibility there is NOT the degenerate early-out: with a clear field it's lit, with a
        // closer occluder it's shadowed (both reachable -> the sample path ran).
        auto clear = [](float, float) { return 1.0f; };
        auto occ = [&](float, float) { return d - 0.05f; };
        check(froxel::SunVisibility(inFrustum, sunVP, clear, kBias) == 1.0f &&
              froxel::SunVisibility(inFrustum, sunVP, occ, kBias) == 0.0f,
              "ROUND-TRIP: in-frustum froxel reaches the sample path (lit clear / shadowed occluded)");
    }

    // ---- DETERMINISM: same inputs -> bit-identical visibility. ----
    {
        const Vec3 froxelW{1.0f, 2.0f, -3.0f};
        float d = lightDepthOf(froxelW);
        auto field = [&](float, float) { return d - 0.03f; };
        float a = froxel::SunVisibility(froxelW, sunVP, field, kBias);
        float b = froxel::SunVisibility(froxelW, sunVP, field, kBias);
        check(a == b, "DETERMINISM: SunVisibility same inputs -> bit-identical");
        float dsplits[4] = {5, 15, 40, 90};
        check(froxel::SelectCascade(12.3f, dsplits, 4) == froxel::SelectCascade(12.3f, dsplits, 4),
              "DETERMINISM: SelectCascade deterministic");
    }

    // ---- The volumetricShadows == false IDENTITY (documented contract): the shader skips the gate
    //      entirely (sun visibility forced to 1) -> sunScatter * 1 == sunScatter EXACTLY. We assert the
    //      MULTIPLICATIVE identity the inject pass relies on for the byte-identical shadows-off == CV
    //      proof: any sun scatter * SunVisibility(forced 1) == the sun scatter unchanged. ----
    {
        auto farField = [](float, float) { return 1.0f; };
        const Vec3 froxelW{0.0f, 1.0f, -2.0f};
        float vis = froxel::SunVisibility(froxelW, sunVP, farField, kBias);
        Vec3 sunScatter{0.37f, 0.21f, 0.55f};
        Vec3 gated = sunScatter * vis;
        check(gated.x == sunScatter.x && gated.y == sunScatter.y && gated.z == sunScatter.z,
              "IDENTITY: visibility 1 is a clean multiplicative identity on the sun scatter (CV proof)");
    }

    if (g_fail == 0) { std::printf("vol_shadows_test: all checks passed\n"); return 0; }
    std::printf("vol_shadows_test: %d FAILURES\n", g_fail);
    return 1;
}
