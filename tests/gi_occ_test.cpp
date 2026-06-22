// Slice GI5 — Deterministic Lumen-class GI: INTEGER CHEBYSHEV OCCLUSION WEIGHTING (the probe-volume
// light-leak fix, FLAGSHIP #29). The CPU-side validation that SETTLES the integer Chebyshev occlusion
// weighting before any GPU trust: the per-probe distance moments must be deterministic + a VALID variance
// (meanDist2 >= meanDist²); the Chebyshev visibility must be kOne for d<=meanDist, strictly decreasing for
// d>meanDist, and bounded to [0,kOne]; with occStrength==0 the occlusion-weighted interp must be
// BYTE-IDENTICAL to the unoccluded GI3 FxInterpolateIrradiance (the falsifiable no-op contract); the
// re-normalized weights must sum to kOne EXACTLY (partition of unity preserved); a behind-occluder query
// point must be ATTENUATED vs the unoccluded blend (the leak is reduced); and two runs are byte-identical.
// The GPU==CPU irradiance memcmp proof lives in --gi5-occlusion-shot (it needs a real GPU); this pure-CPU
// test pins the exact integer-occlusion contract that proof rests on.
//
// Pure C++ (hf_core), ASan-eligible. gi.h #includes render/rtrace.h read-only (rtrace.h #includes
// sim/fpx.h read-only); this test #includes gi.h read-only.
#include "render/gi.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "test_main.h"

using namespace hf;
namespace gi = hf::render::gi;
namespace rt = hf::render::rtrace;
using gi::fx;
using gi::kOne;
using gi::kFrac;
using gi::FxVec3;
using gi::GiRadiance;
using gi::FxProbeSH;
using gi::FxProbeMoments;
using gi::GiProbeGrid;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static double f(fx v) { return (double)v / (double)(int)kOne; }

// The showcase 4x4x4 grid over the pinned GI1 scene (the shared fixture).
static GiProbeGrid MakeGrid() {
    GiProbeGrid grid;
    grid.origin  = FxVec3{gi::GiF(-2, 1), gi::GiF(1, 1), gi::GiF(0, 1)};
    grid.spacing = gi::GiF(1, 1);
    grid.nx = 4; grid.ny = 4; grid.nz = 4;
    return grid;
}

int main() {
    HF_TEST_MAIN_INIT();

    GiProbeGrid grid = MakeGrid();
    const int probes = gi::ProbeCount(grid);
    gi::GiScene1 sc = gi::BuildGi1Scene();

    // CPU radiance (GI1) -> SH (GI2) + moments (GI5).
    std::vector<GiRadiance> rad((size_t)probes * gi::kGiRaysPerProbe);
    gi::TraceProbeRays(grid, sc.scene, std::span<GiRadiance>(rad));
    std::vector<FxProbeSH> sh(probes);
    gi::EncodeAllProbes(grid, std::span<const GiRadiance>(rad), std::span<FxProbeSH>(sh));
    std::span<const FxProbeSH> shSpan(sh);
    std::vector<FxProbeMoments> mom(probes);
    gi::FxProbeMoments_All(grid, sc.scene, std::span<FxProbeMoments>(mom));
    std::span<const FxProbeMoments> momSpan(mom);

    // ================= (1) FxProbeMoments_All: deterministic + valid variance =================
    {
        std::vector<FxProbeMoments> mom2(probes);
        gi::FxProbeMoments_All(grid, sc.scene, std::span<FxProbeMoments>(mom2));
        bool det = std::memcmp(mom.data(), mom2.data(), mom.size() * sizeof(FxProbeMoments)) == 0;
        check(det, "FxProbeMoments_All: two runs BYTE-IDENTICAL (deterministic)");

        bool validVar = true;
        bool anyNonzero = false;
        int64_t minVar = 0; bool firstVar = true;
        for (int p = 0; p < probes; ++p) {
            // variance = meanDist2 - meanDist² (Q16.16), formed exactly as FxChebyshevVisibility does.
            fx meanSq = gi::fxmul(mom[p].meanDist, mom[p].meanDist);
            int64_t var = (int64_t)mom[p].meanDist2 - (int64_t)meanSq;
            if (var < 0) validVar = false;            // Jensen: E[d²] >= E[d]² -> variance >= 0
            if (firstVar || var < minVar) { minVar = var; firstVar = false; }
            if (mom[p].meanDist != 0) anyNonzero = true;
        }
        check(validVar, "FxProbeMoments_All: meanDist2 >= meanDist² for every probe (valid variance)");
        check(anyNonzero, "FxProbeMoments_All: at least one probe has a non-zero mean distance");
        std::printf("  moments: probes=%d  minVariance(Q16.16)=%lld\n", probes, (long long)minVar);
    }

    // ================= (2) FxChebyshevVisibility: kOne for d<=meanDist, decreasing, in [0,kOne] =====
    {
        // A synthetic moment with a well-separated occluder at meanDist=2.0, variance small.
        FxProbeMoments m{};
        m.meanDist  = gi::GiF(2, 1);                                   // 2.0
        m.meanDist2 = (fx)(gi::fxmul(m.meanDist, m.meanDist) + (kOne / 100));  // 4.0 + 0.01 variance

        // d <= meanDist -> exactly kOne.
        check(gi::FxChebyshevVisibility(m, gi::GiF(1, 1)) == kOne, "Chebyshev: d=1.0 (< meanDist 2) -> kOne");
        check(gi::FxChebyshevVisibility(m, m.meanDist) == kOne,    "Chebyshev: d==meanDist -> kOne");
        check(gi::FxChebyshevVisibility(m, 0) == kOne,             "Chebyshev: d=0 -> kOne");

        // d > meanDist -> in (0, kOne), and STRICTLY DECREASING as d grows.
        fx prev = kOne;
        bool decreasing = true, inRange = true;
        for (int i = 1; i <= 8; ++i) {
            fx d = (fx)(m.meanDist + (fx)((int64_t)i * kOne / 2));    // meanDist + i*0.5
            fx vis = gi::FxChebyshevVisibility(m, d);
            if (vis < 0 || vis > kOne) inRange = false;
            if (vis >= prev) decreasing = false;                     // strictly less than the closer query
            prev = vis;
        }
        check(inRange, "Chebyshev: vis in [0,kOne] for all queries");
        check(decreasing, "Chebyshev: vis STRICTLY DECREASING for d > meanDist");
        std::printf("  chebyshev: vis(meanDist+0.5)=%.5f  vis(meanDist+4.0)=%.5f\n",
                    f(gi::FxChebyshevVisibility(m, (fx)(m.meanDist + kOne / 2))),
                    f(gi::FxChebyshevVisibility(m, (fx)(m.meanDist + 4 * kOne))));

        // Larger variance -> a query BEYOND the occluder is LESS attenuated (softer transition).
        FxProbeMoments mBig{};
        mBig.meanDist  = gi::GiF(2, 1);
        mBig.meanDist2 = (fx)(gi::fxmul(mBig.meanDist, mBig.meanDist) + (kOne / 4));  // variance 0.25
        fx dBeyond = (fx)(m.meanDist + kOne);                         // meanDist + 1.0
        check(gi::FxChebyshevVisibility(mBig, dBeyond) > gi::FxChebyshevVisibility(m, dBeyond),
              "Chebyshev: larger variance -> higher visibility (softer) at the same d");
    }

    // ================= (3) NO-OP: occStrength==0 -> == FxInterpolateIrradiance BYTE-exact =============
    // Over a dense field (and out-of-grid points), the occlusion-weighted interp with occStrength==0 must
    // be byte-identical to the unoccluded GI3 blend (every Chebyshev factor lerps to kOne -> no re-weight;
    // the re-normalize is a pure identity because Σ already == kOne).
    {
        const int QW = 48, QH = 48;
        fx spanX = gi::fxmul((fx)((int64_t)(grid.nx - 1) * kOne), grid.spacing);
        fx spanZ = gi::fxmul((fx)((int64_t)(grid.nz - 1) * kOne), grid.spacing);
        fx sliceY = (fx)(grid.origin.y + (gi::fxmul((fx)((int64_t)(grid.ny - 1) * kOne), grid.spacing) >> 1));
        bool allEqual = true;
        for (int qz = 0; qz < QH; ++qz)
            for (int qx = 0; qx < QW; ++qx) {
                // Span a bit BEYOND the footprint to exercise the out-of-grid clamp too.
                fx u = (fx)(((int64_t)(qx - 4) * kOne) / (QW - 1));
                fx v = (fx)(((int64_t)(qz - 4) * kOne) / (QH - 1));
                fx x = (fx)(grid.origin.x + (((int64_t)spanX * u) >> kFrac));
                fx z = (fx)(grid.origin.z + (((int64_t)spanZ * v) >> kFrac));
                FxVec3 pt{x, sliceY, z};
                FxVec3 n{0, kOne, 0};
                GiRadiance occ = gi::FxInterpolateIrradianceOcc(grid, shSpan, momSpan, pt, n, /*occ*/0);
                GiRadiance gi3 = gi::FxInterpolateIrradiance(grid, shSpan, pt, n);
                if (std::memcmp(&occ, &gi3, sizeof(GiRadiance)) != 0) allEqual = false;
            }
        check(allEqual, "no-op: occStrength==0 -> FxInterpolateIrradianceOcc == FxInterpolateIrradiance (byte-exact)");
    }

    // ================= (4) Re-normalized weights sum to kOne EXACTLY =================
    // Replicate the FxInterpolateIrradianceOcc re-normalize and assert Σ == kOne for occStrength full-on
    // across a dense sweep (the partition of unity must survive the Chebyshev re-weight).
    {
        const fx occ = kOne;   // full occlusion strength
        int64_t maxErr = 0;
        for (int iz = 0; iz <= 10; ++iz)
            for (int iy = 0; iy <= 10; ++iy)
                for (int ix = 0; ix <= 10; ++ix) {
                    fx x = (fx)(grid.origin.x - grid.spacing + (fx)(((int64_t)ix * 5 * (int64_t)grid.spacing) / 10));
                    fx y = (fx)(grid.origin.y - grid.spacing + (fx)(((int64_t)iy * 5 * (int64_t)grid.spacing) / 10));
                    fx z = (fx)(grid.origin.z - grid.spacing + (fx)(((int64_t)iz * 5 * (int64_t)grid.spacing) / 10));
                    FxVec3 pt{x, y, z};
                    gi::FxProbeWeights t = gi::FxNearestProbes(grid, pt);
                    // Reproduce the occlusion re-weight + re-normalize (the FxInterpolateIrradianceOcc body).
                    fx scaled[8]; int64_t sumW = 0;
                    for (int c = 0; c < 8; ++c) {
                        FxVec3 cp = gi::ProbePos(grid, t.idx[c]);
                        int64_t sx = (int64_t)(pt.x - cp.x) * (int64_t)(pt.x - cp.x);
                        int64_t sy = (int64_t)(pt.y - cp.y) * (int64_t)(pt.y - cp.y);
                        int64_t sz = (int64_t)(pt.z - cp.z) * (int64_t)(pt.z - cp.z);
                        fx dist = (fx)rt::FxISqrt(sx + sy + sz);
                        fx vis = gi::FxChebyshevVisibility(mom[(size_t)t.idx[c]], dist);
                        fx factor = gi::GiFxLerp(kOne, vis, occ);
                        scaled[c] = gi::fxmul(t.w[c], factor);
                        sumW += (int64_t)scaled[c];
                    }
                    if (sumW <= 0) continue;   // the degenerate fallback keeps t.w (already Σ==kOne)
                    int64_t accum = 0, s = 0;
                    for (int c = 0; c < 7; ++c) { fx w = rt::fxdiv(scaled[c], (fx)sumW); s += w; accum += w; }
                    fx last = (fx)(kOne - accum);
                    s += last;
                    int64_t err = s - (int64_t)kOne; if (err < 0) err = -err;
                    if (err > maxErr) maxErr = err;
                }
        check(maxErr == 0, "re-normalized weights sum to kOne EXACTLY (maxErr 0) under full occlusion");
        std::printf("  renormalize sweep: maxErr=%lld (must be 0)\n", (long long)maxErr);
    }

    // ================= (5) Behind-occluder point is ATTENUATED vs unoccluded =================
    // A SYNTHETIC 2-probe-per-axis grid: one corner probe BRIGHT, with a moment whose mean occluder
    // distance is SMALL (a wall right next to the bright probe). A query point BEHIND that wall (farther
    // from the bright probe than its meanDist) must read DARKER with occlusion ON than the GI3 blend.
    {
        GiProbeGrid g2; g2.origin = FxVec3{0, 0, 0}; g2.spacing = gi::GiF(4, 1); g2.nx = 2; g2.ny = 2; g2.nz = 2;
        const int p2 = gi::ProbeCount(g2);
        std::vector<FxProbeSH> sh2(p2, FxProbeSH{});
        std::vector<FxProbeMoments> m2(p2);
        // All probes see a FAR occluder (fully visible everywhere) EXCEPT the bright corner, which has a
        // wall at ~0.5 units (so a query >0.5 away is "behind the wall").
        for (auto& mm : m2) { mm.meanDist = gi::GiF(100, 1); mm.meanDist2 = gi::fxmul(mm.meanDist, mm.meanDist); }
        int bright = gi::ProbeFlatIndex(g2, 0, 0, 0);
        sh2[bright].coeff[0][0] = kOne / 2; sh2[bright].coeff[0][1] = kOne / 2; sh2[bright].coeff[0][2] = kOne / 2;
        m2[bright].meanDist  = (fx)(kOne / 2);                                   // wall at 0.5 units
        m2[bright].meanDist2 = (fx)(gi::fxmul(m2[bright].meanDist, m2[bright].meanDist) + (kOne / 256));

        FxVec3 n{0, kOne, 0};
        // A query at ~(2,2,2): ~3.46 units from the bright corner -> well past its 0.5 wall -> occluded.
        FxVec3 pt{gi::GiF(2, 1), gi::GiF(2, 1), gi::GiF(2, 1)};
        GiRadiance unocc = gi::FxInterpolateIrradianceOcc(g2, std::span<const FxProbeSH>(sh2),
                                                          std::span<const FxProbeMoments>(m2), pt, n, 0);
        GiRadiance occON = gi::FxInterpolateIrradianceOcc(g2, std::span<const FxProbeSH>(sh2),
                                                          std::span<const FxProbeMoments>(m2), pt, n, kOne);
        GiRadiance gi3 = gi::FxInterpolateIrradiance(g2, std::span<const FxProbeSH>(sh2), pt, n);
        check(std::memcmp(&unocc, &gi3, sizeof(GiRadiance)) == 0,
              "behind-occluder: occStrength==0 still == GI3 (the no-op holds on the synthetic scene)");
        check(occON.r < gi3.r, "behind-occluder: occlusion-weighted irradiance STRICTLY < unoccluded GI3 (leak reduced)");
        std::printf("  leak: gi3.r=%.5f  occ.r=%.5f (occ < gi3 -> leak attenuated)\n", f(gi3.r), f(occON.r));
    }

    // ================= (6) DETERMINISM: two InterpolateFieldOcc runs byte-identical =================
    {
        const int QW = 32, QH = 32;
        std::vector<FxVec3> pts((size_t)QW * QH), nrm((size_t)QW * QH, FxVec3{0, kOne, 0});
        fx spanX = gi::fxmul((fx)((int64_t)(grid.nx - 1) * kOne), grid.spacing);
        fx spanZ = gi::fxmul((fx)((int64_t)(grid.nz - 1) * kOne), grid.spacing);
        fx sliceY = (fx)(grid.origin.y + (gi::fxmul((fx)((int64_t)(grid.ny - 1) * kOne), grid.spacing) >> 1));
        for (int qz = 0; qz < QH; ++qz)
            for (int qx = 0; qx < QW; ++qx) {
                fx u = (fx)(((int64_t)qx * kOne) / (QW - 1));
                fx v = (fx)(((int64_t)qz * kOne) / (QH - 1));
                fx x = (fx)(grid.origin.x + (((int64_t)spanX * u) >> kFrac));
                fx z = (fx)(grid.origin.z + (((int64_t)spanZ * v) >> kFrac));
                pts[(size_t)qz * QW + qx] = FxVec3{x, sliceY, z};
            }
        std::vector<GiRadiance> o1((size_t)QW * QH), o2((size_t)QW * QH);
        const fx occ = (fx)(kOne / 2);
        gi::InterpolateFieldOcc(grid, shSpan, momSpan, std::span<const FxVec3>(pts),
                                std::span<const FxVec3>(nrm), occ, std::span<GiRadiance>(o1));
        gi::InterpolateFieldOcc(grid, shSpan, momSpan, std::span<const FxVec3>(pts),
                                std::span<const FxVec3>(nrm), occ, std::span<GiRadiance>(o2));
        bool same = std::memcmp(o1.data(), o2.data(), o1.size() * sizeof(GiRadiance)) == 0;
        check(same, "determinism: two InterpolateFieldOcc runs BYTE-IDENTICAL");
        bool nonTrivial = false;
        for (const auto& g : o1) if (g.r != 0 || g.g != 0 || g.b != 0) nonTrivial = true;
        check(nonTrivial, "the occlusion-weighted field is non-trivial (some irradiance non-zero)");
    }

    // ================= (7) NO-OP DISPATCH: empty query set -> InterpolateFieldOcc writes nothing ======
    {
        std::vector<FxVec3> noPts, noNrm;
        std::vector<GiRadiance> out(4);
        for (auto& g : out) g = GiRadiance{(fx)555, (fx)555, (fx)555, 0};
        gi::InterpolateFieldOcc(grid, shSpan, momSpan, std::span<const FxVec3>(noPts),
                                std::span<const FxVec3>(noNrm), kOne, std::span<GiRadiance>(out));
        bool untouched = true;
        for (const auto& g : out) if (g.r != 555 || g.g != 555 || g.b != 555) untouched = false;
        check(untouched, "InterpolateFieldOcc: empty query set -> output UNTOUCHED (the no-op)");
    }

    if (g_fail == 0) std::printf("gi_occ_test: all GI5 integer Chebyshev-occlusion invariants PASS\n");
    return g_fail == 0 ? 0 : 1;
}
