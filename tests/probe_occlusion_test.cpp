// Slice DP — DDGI Visibility Slice 2: Chebyshev Occlusion Weighting. Pure CPU math: the direction->
// (face,u,v) cube-query (DistDirToFaceUV, matching DO's FaceView convention), the per-direction moment
// readback (SampleProbeMoments), the Chebyshev (variance-shadow) visibility closed form
// (ChebyshevVisibility), the occlusionStrength=0 identity (the all-vis==1 weighted blend == the unweighted
// probesh::InterpolateSH, byte-identical — the make-or-break no-op the shader's occlusionStrength==0->DN
// branch guarantees), the all-occluded->zero-indirect fallback (no NaN), and determinism. No device,
// ASan-eligible (links hf_core). Exercises the EXACT math the --ddgiocc-shot showcase + the Metal --ddgiocc
// showcase consume (engine/render/probe_dist.h + lit_ddgi_occ.frag.hlsl), so the in-shader Chebyshev weight
// agrees with this CPU reference.
//
// Properties pinned (per the spec §5):
//   * DistDirToFaceUV: the 6 axis dirs -> the 6 face centres; a dir round-trips to an in-range (face,u,v).
//   * SampleProbeMoments: a store with a known moment at a texel -> the matching dir reads it back.
//   * Chebyshev closed form: dist<=mean -> vis==1; dist>>mean small var -> vis≈0; var==0 -> a hard step.
//   * occlusionStrength=0 identity: the all-vis==1 weighted blend == probesh::InterpolateSH byte-identical.
//   * all-occluded -> zero indirect (no NaN); determinism.
#include "render/probe_dist.h"
#include "render/probe_gi.h"
#include "render/probe_sh.h"
#include "render/cubemap.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace probedist = hf::render::probedist;
namespace probesh   = hf::render::probesh;
namespace probegi   = hf::render::probegi;
namespace cubemap   = hf::render::cubemap;
using hf::math::Vec3;
using hf::math::Vec2;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();

    // ====================================================================================
    // DistDirToFaceUV — the 6 axis dirs map to the 6 face centres; round-trips in range.
    // ====================================================================================
    {
        // The 6 axis directions select faces +X,-X,+Y,-Y,+Z,-Z (the cubemap face order). Each maps to the
        // FACE CENTRE: cubemap::DirToFaceUV gives UV (0.5,0.5) on-axis, and 0.5*kDistFace=4 -> texel (4,4).
        const Vec3 axes[6] = {
            {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}};
        bool faceOk = true, centreOk = true;
        for (int expectFace = 0; expectFace < 6; ++expectFace) {
            int face, u, v;
            probedist::DistDirToFaceUV(axes[expectFace], face, u, v);
            if (face != expectFace) faceOk = false;
            // On-axis -> face centre. With kDistFace=8, floor(0.5*8)=4.
            if (u != 4 || v != 4) centreOk = false;
        }
        check(faceOk, "DistDirToFaceUV: the 6 axis dirs select faces +X,-X,+Y,-Y,+Z,-Z in order");
        check(centreOk, "DistDirToFaceUV: an on-axis dir maps to the face-centre texel (4,4)");

        // Consistency with the capture convention: DistDirToFaceUV's face == cubemap::DirToFaceUV's face,
        // and the quantised texel is the floor of (face UV * kDistFace) — exactly DO's capture sub-sampling.
        bool consistent = true, inRange = true;
        const Vec3 dirs[] = {
            {0.9f, 0.1f, -0.2f}, {-0.3f, 0.8f, 0.5f}, {0.2f, -0.1f, 0.95f},
            {-0.7f, -0.6f, 0.1f}, {0.4f, 0.5f, -0.77f}, {0.33f, 0.33f, 0.33f}};
        for (const Vec3& d : dirs) {
            int face, u, v;
            probedist::DistDirToFaceUV(d, face, u, v);
            int cmFace; Vec2 uv;
            cubemap::DirToFaceUV(d, cmFace, uv);
            if (face != cmFace) consistent = false;
            int eu = (int)std::floor(uv.x * (float)probedist::kDistFace);
            int ev = (int)std::floor(uv.y * (float)probedist::kDistFace);
            if (eu < 0) eu = 0; if (eu > probedist::kDistFace - 1) eu = probedist::kDistFace - 1;
            if (ev < 0) ev = 0; if (ev > probedist::kDistFace - 1) ev = probedist::kDistFace - 1;
            if (u != eu || v != ev) consistent = false;
            if (face < 0 || face >= 6 || u < 0 || u >= probedist::kDistFace ||
                v < 0 || v >= probedist::kDistFace) inRange = false;
        }
        check(consistent, "DistDirToFaceUV: face + quantised texel match cubemap::DirToFaceUV (DO convention)");
        check(inRange, "DistDirToFaceUV: every dir maps to an in-range (face, u, v)");
    }

    // ====================================================================================
    // SampleProbeMoments — a store with a known moment at a texel reads back via the matching dir.
    // ====================================================================================
    {
        const int kProbes = 8;
        std::vector<probedist::ProbeDistMoments> store(
            (size_t)kProbes * probedist::kProbeTexels, probedist::ProbeDistMoments{{0.0f, 0.0f}});
        // Plant a distinct moment at probe 3, the face-centre texel (4,4) of each face. Sampling the axis
        // dir of that face reads it back.
        const Vec3 axes[6] = {
            {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}};
        bool readback = true;
        for (int f = 0; f < 6; ++f) {
            float d = 2.0f + (float)f;   // distinct per face
            store[probedist::ProbeDistTexelIndex(3, f, 4, 4)] = probedist::MomentsFromDistance(d);
        }
        for (int f = 0; f < 6; ++f) {
            float d = 2.0f + (float)f;
            Vec2 mom = probedist::SampleProbeMoments(store.data(), 3, axes[f]);
            if (mom.x != d || mom.y != d * d) readback = false;
        }
        check(readback, "SampleProbeMoments: an axis dir reads back the moment planted at that face centre");

        // null store -> {0,0} (no UB, the documented no-data fallback).
        Vec2 z = probedist::SampleProbeMoments(nullptr, 0, Vec3{1, 0, 0});
        check(z.x == 0.0f && z.y == 0.0f, "SampleProbeMoments: null store -> {0,0}");

        // Determinism: byte-identical across calls.
        Vec2 a = probedist::SampleProbeMoments(store.data(), 3, axes[2]);
        Vec2 b = probedist::SampleProbeMoments(store.data(), 3, axes[2]);
        check(std::memcmp(&a, &b, sizeof(Vec2)) == 0, "SampleProbeMoments: deterministic (byte-identical)");
    }

    // ====================================================================================
    // Chebyshev closed form — dist<=mean -> 1; dist>>mean small var -> ≈0; var==0 -> hard step.
    // ====================================================================================
    {
        // A moderately-certain occluder at mean=5, with some variance.
        float mean = 5.0f, var = 0.25f;
        Vec2 mom{mean, mean * mean + var};   // m[1] = mean^2 + var

        // dist <= mean -> EXACTLY 1 (the early-out, fully visible).
        check(probedist::ChebyshevVisibility(mom, 0.0f) == 1.0f, "Chebyshev: dist=0 (<=mean) -> vis==1");
        check(probedist::ChebyshevVisibility(mom, mean) == 1.0f, "Chebyshev: dist==mean -> vis==1 (boundary)");
        check(probedist::ChebyshevVisibility(mom, mean - 1.0f) == 1.0f, "Chebyshev: dist<mean -> vis==1");

        // dist >> mean with small var -> ≈0 (strongly attenuated, occluded).
        float visFar = probedist::ChebyshevVisibility(mom, mean + 20.0f);
        check(visFar > 0.0f && visFar < 0.01f, "Chebyshev: dist>>mean (small var) -> vis≈0 (occluded)");

        // The closed form at a known point: dist = mean + dd -> var/(var+dd^2).
        float dd = 2.0f;
        float expect = var / (var + dd * dd);
        float got = probedist::ChebyshevVisibility(mom, mean + dd);
        check(std::fabs(got - expect) <= 1e-6f, "Chebyshev: dist=mean+dd -> var/(var+dd^2) (closed form)");

        // var==0 -> a HARD step: 1 at dist<=mean, 0 strictly past it (no NaN from 0/0 — the dist<=mean
        // branch returns 1 first; past mean it is 0/(0+dd^2)==0).
        Vec2 hard{mean, mean * mean};   // var == 0 exactly
        check(probedist::ChebyshevVisibility(hard, mean) == 1.0f, "Chebyshev: var==0, dist==mean -> 1 (no NaN)");
        check(probedist::ChebyshevVisibility(hard, mean + 0.001f) == 0.0f,
              "Chebyshev: var==0, dist>mean -> 0 (hard step, no 0/0 NaN)");
        float hv = probedist::ChebyshevVisibility(hard, mean + 5.0f);
        check(hv == 0.0f && !std::isnan(hv), "Chebyshev: var==0 far -> 0, finite (no NaN)");

        // Determinism.
        float x = probedist::ChebyshevVisibility(mom, mean + dd);
        float y = probedist::ChebyshevVisibility(mom, mean + dd);
        check(std::memcmp(&x, &y, sizeof(float)) == 0, "Chebyshev: deterministic (byte-identical)");
    }

    // ====================================================================================
    // occlusionStrength=0 identity — the all-vis==1 renormalised weighted blend == the unweighted
    // probesh::InterpolateSH, BYTE-IDENTICAL. This is the CPU mirror of the no-op the shader guarantees by
    // branching to the verbatim DN path at occlusionStrength==0. Two facts make it exact:
    //   (1) the trilinear weights are a partition of unity computed with std::fma -> they sum to EXACTLY 1.
    //   (2) with all vis==1, vw[c]==tri.w[c], wsum==1, and vw[c]/wsum == tri.w[c]/1.0 == tri.w[c] exactly,
    //       so the renormalised mad-blend reproduces InterpolateSH's per-corner fma accumulation to the bit.
    // ====================================================================================
    {
        probegi::ProbeGrid grid;
        grid.origin = Vec3{-2.0f, 0.0f, -2.0f};
        grid.dimX = 2; grid.dimY = 2; grid.dimZ = 2; grid.spacing = 4.0f;
        const int probeN = grid.probeCount();

        // A deterministic set of distinct per-probe SH records.
        std::vector<probesh::ProbeSH> probes((size_t)probeN);
        for (int p = 0; p < probeN; ++p)
            for (int i = 0; i < 9; ++i)
                for (int c = 0; c < 3; ++c)
                    probes[p].coeff[i][c] = 0.013f * (float)(p * 27 + i * 3 + c) - 0.2f;

        // A query point strictly inside the cell (a non-trivial trilinear mix).
        Vec3 wpos{0.7f, 1.3f, -0.4f};

        // Reference: the unweighted trilinear blend (== the shader's DN no-op path).
        probesh::ProbeSH ref = probesh::InterpolateSH(wpos, grid, probes.data(), probeN);

        // The all-vis==1 renormalised weighted blend (the shader's occlusionStrength>0 arithmetic with
        // every vis forced to 1). First confirm wsum (== Σ tri.w[c]) is EXACTLY 1.0.
        probegi::ProbeTrilinear tri = probegi::NearestProbes(wpos, grid);
        float wsum = 0.0f;
        for (int c = 0; c < 8; ++c) wsum += tri.w[c];
        check(wsum == 1.0f, "occ=0 identity: the trilinear weights are a partition of unity (Σw == 1.0 exact)");

        probesh::ProbeSH weighted{};
        for (int i = 0; i < 9; ++i) { weighted.coeff[i][0] = 0; weighted.coeff[i][1] = 0; weighted.coeff[i][2] = 0; }
        float invSum = 1.0f / wsum;   // == 1.0 exactly
        for (int c = 0; c < 8; ++c) {
            float nw = tri.w[c] * invSum;   // == tri.w[c]
            const probesh::ProbeSH& src = probes[tri.idx[c]];
            for (int i = 0; i < 9; ++i) {
                weighted.coeff[i][0] = std::fma(nw, src.coeff[i][0], weighted.coeff[i][0]);
                weighted.coeff[i][1] = std::fma(nw, src.coeff[i][1], weighted.coeff[i][1]);
                weighted.coeff[i][2] = std::fma(nw, src.coeff[i][2], weighted.coeff[i][2]);
            }
        }
        check(std::memcmp(&weighted, &ref, sizeof(probesh::ProbeSH)) == 0,
              "occ=0 identity: all-vis==1 weighted blend == probesh::InterpolateSH BYTE-IDENTICAL");
    }

    // ====================================================================================
    // all-occluded -> zero indirect (no NaN). With every probe's vis==0, wsum<=eps -> the blend stays the
    // zeroed SH -> SHEvaluate -> {0,0,0}, finite (no /wsum NaN). Mirrors the shader's wsum<=1e-6 fallback.
    // ====================================================================================
    {
        probegi::ProbeGrid grid;
        grid.origin = Vec3{-2.0f, 0.0f, -2.0f};
        grid.dimX = 2; grid.dimY = 2; grid.dimZ = 2; grid.spacing = 4.0f;
        const int probeN = grid.probeCount();
        std::vector<probesh::ProbeSH> probes((size_t)probeN);
        for (int p = 0; p < probeN; ++p)
            for (int i = 0; i < 9; ++i)
                for (int c = 0; c < 3; ++c) probes[p].coeff[i][c] = 1.0f;

        Vec3 wpos{0.7f, 1.3f, -0.4f};
        probegi::ProbeTrilinear tri = probegi::NearestProbes(wpos, grid);

        // All probes fully occluded: vis==0 -> vw all 0 -> wsum==0 (<= 1e-6) -> the zeroed SH blend.
        float wsum = 0.0f;
        float vw[8];
        for (int c = 0; c < 8; ++c) { vw[c] = tri.w[c] * 0.0f; wsum += vw[c]; }
        probesh::ProbeSH blended{};
        for (int i = 0; i < 9; ++i) { blended.coeff[i][0] = 0; blended.coeff[i][1] = 0; blended.coeff[i][2] = 0; }
        if (wsum > 1e-6f) {
            float invSum = 1.0f / wsum;
            for (int c = 0; c < 8; ++c) {
                float nw = vw[c] * invSum;
                const probesh::ProbeSH& src = probes[tri.idx[c]];
                for (int i = 0; i < 9; ++i) {
                    blended.coeff[i][0] = std::fma(nw, src.coeff[i][0], blended.coeff[i][0]);
                    blended.coeff[i][1] = std::fma(nw, src.coeff[i][1], blended.coeff[i][1]);
                    blended.coeff[i][2] = std::fma(nw, src.coeff[i][2], blended.coeff[i][2]);
                }
            }
        }
        Vec3 indirect = probesh::SHEvaluate(blended, hf::math::normalize(Vec3{0, 1, 0}));
        check(indirect.x == 0.0f && indirect.y == 0.0f && indirect.z == 0.0f,
              "all-occluded: wsum<=eps -> zeroed SH -> indirect {0,0,0}");
        check(!std::isnan(indirect.x) && !std::isnan(indirect.y) && !std::isnan(indirect.z),
              "all-occluded: no NaN from the /wsum fallback");
    }

    if (g_fail == 0) { std::printf("probe_occlusion_test OK\n"); return 0; }
    std::printf("probe_occlusion_test: %d failures\n", g_fail);
    return 1;
}
