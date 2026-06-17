// Slice DO — DDGI Visibility Slice 1: Per-Probe Distance-Moment Capture. Pure CPU math: the moment-from-
// distance reference (the GPU==CPU bit-exact step), the distance-texel count (the moment-store size + the
// probeCount=0 no-op), the per-probe×face×texel flat-slot indexing (bijection + round-trip + full
// coverage), and the deterministic per-probe mean-distance viz. No device, ASan-eligible (links hf_core).
// Exercises the EXACT math the --probedist-shot showcase (samples/hello_triangle/main.cpp) + the Metal
// --probedist showcase consume (engine/render/probe_dist.h), so the GPU capture loop's probe×face×texel
// slot layout + the moment-from-distance step agree with this CPU reference.
//
// Properties pinned (per the spec §5):
//   * MomentsFromDistance(d) == {d, d*d} EXACTLY for a fixed set + at 0 (the bare-multiply second moment).
//   * DistTexelCount == probeCount*384; EXACTLY 0 at dimX/dimY/dimZ == 0 (the probeCount=0 no-op).
//   * ProbeDistFaceIndex(p,f) == the texel-block start p*384 + f*64; ProbeDistTexelIndex round-trips
//     (ProbeDistTexelFromIndex) + the layout p*384+f*64+v*8+u is a bijection over [0, probeCount*384)
//     (full coverage, no overlap, no gap).
//   * Determinism: MomentsFromDistance + ProbeMeanDistance are byte-identical across calls.
#include "render/probe_dist.h"
#include "render/cubemap.h"
#include "render/probe_gi.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace probedist = hf::render::probedist;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();

    // ====================================================================================
    // ProbeDistMoments — std430-tight (the SSBO contract)
    // ====================================================================================
    {
        static_assert(sizeof(probedist::ProbeDistMoments) == 8,
                      "ProbeDistMoments must be 8 bytes (two float, std430-tight)");
        check(sizeof(probedist::ProbeDistMoments) == 8, "ProbeDistMoments sizeof == 8 (std430-tight)");
        check(probedist::kFaces == 6, "kFaces == 6 (the cube faces)");
        check(probedist::kDistFace == 8, "kDistFace == 8 (per-face edge)");
        check(probedist::kFaceTexels == 64, "kFaceTexels == 64 (8x8)");
        check(probedist::kProbeTexels == 384, "kProbeTexels == 384 (8x8x6)");
    }

    // ====================================================================================
    // MomentsFromDistance(d) == {d, d*d} EXACTLY (the GPU==CPU bit-exact reference step)
    // ====================================================================================
    {
        // A fixed set spanning tiny, unit, and far distances. The match is EXACT (==, not approx): the
        // second moment is a single bare multiply d*d, one IEEE-754 round, identical to the shader.
        const float ds[] = {0.0f, 0.5f, 1.0f, 2.0f, 3.5f, 7.0f, 12.5f, 60.0f, 123.75f, 1.0e4f};
        bool exact = true;
        for (float d : ds) {
            probedist::ProbeDistMoments m = probedist::MomentsFromDistance(d);
            // The reference second moment computed the SAME way (bare multiply) — must be bit-identical.
            float dd = d * d;
            if (m.m[0] != d || m.m[1] != dd) exact = false;
            // And the second moment must equal a BARE multiply (no contraction / no fma divergence).
            if (std::memcmp(&m.m[1], &dd, sizeof(float)) != 0) exact = false;
        }
        check(exact, "MomentsFromDistance(d) == {d, d*d} EXACTLY over a fixed distance set (bare multiply)");

        probedist::ProbeDistMoments z = probedist::MomentsFromDistance(0.0f);
        check(z.m[0] == 0.0f && z.m[1] == 0.0f, "MomentsFromDistance(0) == {0, 0}");

        // Determinism: a second call is byte-identical.
        probedist::ProbeDistMoments a = probedist::MomentsFromDistance(3.5f);
        probedist::ProbeDistMoments b = probedist::MomentsFromDistance(3.5f);
        check(std::memcmp(&a, &b, sizeof(a)) == 0, "MomentsFromDistance: deterministic (byte-identical)");
    }

    // ====================================================================================
    // DistTexelCount == probeCount*384; 0 at any zero dim / probeCount 0
    // ====================================================================================
    {
        probedist::ProbeGrid grid;          // the SMALL DO capture grid: 2x2x2 = 8 probes
        grid.dimX = 2; grid.dimY = 2; grid.dimZ = 2; grid.spacing = 1.0f;
        check(grid.probeCount() == 8, "DO capture grid: 2x2x2 == 8 probes");
        check(probedist::DistTexelCount(grid) == 8 * 384,
              "DistTexelCount: 8 probes -> 8*384 distance texels (probeCount*384)");

        probedist::ProbeGrid g111; g111.dimX = 1; g111.dimY = 1; g111.dimZ = 1;
        check(probedist::DistTexelCount(g111) == 384, "DistTexelCount: 1 probe -> 384 texels");
        probedist::ProbeGrid g321; g321.dimX = 3; g321.dimY = 2; g321.dimZ = 1;
        check(probedist::DistTexelCount(g321) == 6 * 384, "DistTexelCount: 6 probes -> 6*384 texels");

        // The probeCount=0 no-op: each zero dim -> probeCount 0 -> DistTexelCount 0 (capture loop skipped).
        probedist::ProbeGrid gx; gx.dimX = 0; gx.dimY = 2; gx.dimZ = 2;
        probedist::ProbeGrid gy; gy.dimX = 2; gy.dimY = 0; gy.dimZ = 2;
        probedist::ProbeGrid gz; gz.dimX = 2; gz.dimY = 2; gz.dimZ = 0;
        check(gx.probeCount() == 0 && probedist::DistTexelCount(gx) == 0,
              "Disabled: dimX==0 -> probeCount 0 -> DistTexelCount 0 (capture loop skipped)");
        check(gy.probeCount() == 0 && probedist::DistTexelCount(gy) == 0,
              "Disabled: dimY==0 -> probeCount 0 -> DistTexelCount 0");
        check(gz.probeCount() == 0 && probedist::DistTexelCount(gz) == 0,
              "Disabled: dimZ==0 -> probeCount 0 -> DistTexelCount 0");
    }

    // ====================================================================================
    // ProbeDistFaceIndex / ProbeDistTexelIndex — bijection, round-trip, full coverage
    // ====================================================================================
    {
        const int kProbes = 8;
        const int kSlots = kProbes * probedist::kProbeTexels;   // 8 * 384 = 3072

        // ProbeDistFaceIndex is the per-probe×face TEXEL-BLOCK start: p*384 + f*64.
        check(probedist::ProbeDistFaceIndex(0, 0) == 0 &&
                  probedist::ProbeDistFaceIndex(0, 1) == 64 &&
                  probedist::ProbeDistFaceIndex(0, 5) == 320 &&
                  probedist::ProbeDistFaceIndex(1, 0) == 384 &&
                  probedist::ProbeDistFaceIndex(7, 5) == 7 * 384 + 320,
              "ProbeDistFaceIndex: probe-major face block (p*384 + f*64)");

        // ProbeDistTexelIndex(p,f,u,v) round-trips via ProbeDistTexelFromIndex, and the layout
        // p*384+f*64+v*8+u is a BIJECTION over [0, probeCount*384) (no overlap, no gap, all in range).
        std::vector<int> hit(kSlots, 0);
        bool roundTrip = true, inRange = true;
        for (int p = 0; p < kProbes; ++p)
            for (int f = 0; f < probedist::kFaces; ++f)
                for (int v = 0; v < probedist::kDistFace; ++v)
                    for (int u = 0; u < probedist::kDistFace; ++u) {
                        int slot = probedist::ProbeDistTexelIndex(p, f, u, v);
                        // Explicit layout formula must match.
                        if (slot != p * 384 + f * 64 + v * 8 + u) inRange = false;
                        if (slot < 0 || slot >= kSlots) { inRange = false; continue; }
                        hit[slot]++;
                        int rp, rf, ru, rv;
                        probedist::ProbeDistTexelFromIndex(slot, rp, rf, ru, rv);
                        if (rp != p || rf != f || ru != u || rv != v) roundTrip = false;
                    }
        check(inRange, "ProbeDistTexelIndex: every (p,f,u,v) maps to [0, probeCount*384) via p*384+f*64+v*8+u");
        check(roundTrip, "ProbeDistTexelIndex: round-trips (p,f,u,v) via ProbeDistTexelFromIndex");
        bool noOverlap = true, fullCover = true;
        for (int s = 0; s < kSlots; ++s) {
            if (hit[s] > 1) noOverlap = false;
            if (hit[s] == 0) fullCover = false;
        }
        check(noOverlap, "ProbeDistTexelIndex: no two texels share a slot (no overlap)");
        check(fullCover, "ProbeDistTexelIndex: covers every slot exactly once (no gaps) — bijection");
    }

    // ====================================================================================
    // ProbeFaceViewProj reuse — exact reuse of cubemap::FaceViewProj at the probe center
    // ====================================================================================
    {
        const hf::math::Vec3 probeCenter{2.0f, -1.0f, 3.0f};
        const float zN = 0.05f, zF = 60.0f;
        bool vpOk = true;
        for (int f = 0; f < probedist::kFaces; ++f) {
            hf::math::Mat4 a = probedist::ProbeFaceViewProj(f, probeCenter, zN, zF);
            hf::math::Mat4 b = hf::render::cubemap::FaceViewProj(f, probeCenter, zN, zF);
            if (std::memcmp(&a, &b, sizeof(hf::math::Mat4)) != 0) vpOk = false;
        }
        check(vpOk, "ProbeFaceViewProj == cubemap::FaceViewProj at the probe center (exact reuse)");
    }

    // ====================================================================================
    // ProbeMeanDistance — hand cases + determinism (the swatch-viz value)
    // ====================================================================================
    {
        // A constant-distance probe (all texels d=4) means to 4.
        std::vector<probedist::ProbeDistMoments> cf(probedist::kProbeTexels);
        for (auto& m : cf) m = probedist::MomentsFromDistance(4.0f);
        float mean = probedist::ProbeMeanDistance(cf.data(), (int)cf.size());
        check(std::fabs(mean - 4.0f) <= 1e-4f, "ProbeMeanDistance: constant-distance probe means to that distance");

        // A mixed probe: two texels at d=2 and d=6 -> mean 4.
        probedist::ProbeDistMoments mixed[2] = {
            probedist::MomentsFromDistance(2.0f), probedist::MomentsFromDistance(6.0f)};
        float mm = probedist::ProbeMeanDistance(mixed, 2);
        check(std::fabs(mm - 4.0f) <= 1e-4f, "ProbeMeanDistance: mixed probe means to the per-texel mean");

        // Empty / null -> 0.
        check(probedist::ProbeMeanDistance(mixed, 0) == 0.0f, "ProbeMeanDistance: texelCount 0 -> 0");
        check(probedist::ProbeMeanDistance(nullptr, 4) == 0.0f, "ProbeMeanDistance: null data -> 0 (no UB)");

        // Determinism: byte-identical across calls.
        float a = probedist::ProbeMeanDistance(cf.data(), (int)cf.size());
        float b = probedist::ProbeMeanDistance(cf.data(), (int)cf.size());
        check(std::memcmp(&a, &b, sizeof(float)) == 0, "ProbeMeanDistance: deterministic (byte-identical)");
    }

    if (g_fail == 0) { std::printf("probe_dist_test OK\n"); return 0; }
    std::printf("probe_dist_test: %d failures\n", g_fail);
    return 1;
}
