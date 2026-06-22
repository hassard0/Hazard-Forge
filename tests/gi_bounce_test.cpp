// Slice GI4 — Deterministic Lumen-class GI: INTEGER MULTI-BOUNCE FEEDBACK (light that bounces, FLAGSHIP #29).
// The CPU-side validation that SETTLES the integer multi-bounce feedback before any GPU trust — the second
// make-or-break beat (after GI2's crux). It pins the two falsifiable guards the --gi4-bounce-shot HW==CPU
// proof rests on: (a) the NO-OP CONTRACT — ShadeHitGI with a zero indirect is BYTE-IDENTICAL to the unpacked
// ShadeHitShadowed, so K==1 BounceProbes == the GI1+GI2 single-bounce SH EXACTLY; (b) the MONOTONICITY guard
// — SH_2's reconstructed irradiance >= SH_1's component-wise (the bounce only ADDS light) AND strictly
// greater for >=1 probe (the integer indirect did NOT truncate to zero — the underflow guard, the GI2 crux
// re-verified under feedback). Plus: convergence/in-range (maxIrr across K<=3 stays < 2.0, no overflow); a
// colored-wall scene shows the bounce TINT (an indirect channel imbalance); and two runs are byte-identical.
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
using gi::GiProbeGrid;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static double f(fx v) { return (double)v / (double)(int)kOne; }

// The < 2.0 Q16.16 headroom bound (2.0 in Q16.16 == 2*kOne).
static const int64_t kHeadroom = 2 * (int64_t)kOne;

// Build the showcase 4x4x4 grid (the same fixture the GI1/GI2/GI3 showcases pin).
static GiProbeGrid MakeGrid() {
    GiProbeGrid grid;
    grid.origin  = FxVec3{gi::GiF(-2, 1), gi::GiF(1, 1), gi::GiF(0, 1)};
    grid.spacing = gi::GiF(1, 1);
    grid.nx = 4; grid.ny = 4; grid.nz = 4;
    return grid;
}

// The single-bounce SH (the GI1 trace + GI2 encode) — the no-op oracle K==1 BounceProbes must equal.
static std::vector<FxProbeSH> SingleBounceSH(const GiProbeGrid& grid, const rt::RtScene& scene) {
    const int N = gi::kGiRaysPerProbe;
    const int probes = gi::ProbeCount(grid);
    std::vector<GiRadiance> rad((size_t)probes * N);
    gi::TraceProbeRays(grid, scene, std::span<GiRadiance>(rad));
    std::vector<FxProbeSH> sh(probes);
    gi::EncodeAllProbes(grid, std::span<const GiRadiance>(rad), std::span<FxProbeSH>(sh));
    return sh;
}

int main() {
    HF_TEST_MAIN_INIT();

    gi::GiScene1 sc = gi::BuildGi1Scene();
    const rt::RtScene& scene = sc.scene;
    GiProbeGrid grid = MakeGrid();
    const int probes = gi::ProbeCount(grid);

    // ================= (1) ClampBounces — the fixed-K clamp (1..kGiMaxBounces) =================
    {
        check(gi::ClampBounces(0) == 1, "ClampBounces(0) == 1 (clamp up to single-bounce)");
        check(gi::ClampBounces(1) == 1, "ClampBounces(1) == 1");
        check(gi::ClampBounces(3) == 3, "ClampBounces(3) == 3 (kGiMaxBounces)");
        check(gi::ClampBounces(99) == gi::kGiMaxBounces, "ClampBounces(99) == kGiMaxBounces (clamp down)");
        check(gi::kGiMaxBounces == 3, "kGiMaxBounces == 3");
    }

    // ================= (2) THE NO-OP CONTRACT: ShadeHitGI(indirect==0) == ShadeHitShadowed =================
    // For every probe-ray hit in the scene, ShadeHitGI with a zero indirect must be BYTE-IDENTICAL to
    // UnpackRadiance(ShadeHitShadowed(...)) — the indirect term is a literal integer +0.
    {
        bool allMatch = true;
        int sampled = 0;
        for (int p = 0; p < probes; ++p) {
            FxVec3 origin = gi::ProbePos(grid, p);
            for (int d = 0; d < gi::kGiRaysPerProbe; ++d) {
                rt::RtRay ray{origin, gi::kGiProbeDirs[d]};
                rt::RtHit hit = rt::TraceClosest(ray, scene);
                bool occluded = false;
                if (hit.primIndex != rt::kRtMiss) {
                    rt::RtRay sray;
                    sray.origin = rt::FxAdd(hit.pos, rt::FxScale(hit.normal, rt::kRtShadowEps));
                    sray.dir = scene.lightDir;
                    occluded = rt::TraceAnyHit(sray, scene, rt::kRtShadowMinT);
                }
                GiRadiance gi4 = gi::ShadeHitGI(hit, scene, occluded, GiRadiance{});
                GiRadiance ref = gi::UnpackRadiance(rt::ShadeHitShadowed(hit, scene, occluded));
                if (std::memcmp(&gi4, &ref, sizeof(GiRadiance)) != 0) allMatch = false;
                ++sampled;
            }
        }
        check(allMatch, "no-op: ShadeHitGI(indirectIrr==0) == UnpackRadiance(ShadeHitShadowed), BYTE-exact");
        std::printf("  no-op sampled %d probe-ray shades (all byte-identical)\n", sampled);
    }

    // ================= (3) K==1 BounceProbes == GI1+GI2 single-bounce SH EXACTLY =================
    {
        std::vector<FxProbeSH> bounceK1(probes);
        gi::BounceProbes(scene, grid, 1, std::span<FxProbeSH>(bounceK1));
        std::vector<FxProbeSH> single = SingleBounceSH(grid, scene);
        bool same = std::memcmp(bounceK1.data(), single.data(), (size_t)probes * sizeof(FxProbeSH)) == 0;
        check(same, "K==1 BounceProbes == GI1-trace + GI2-encode single-bounce SH, BYTE-IDENTICAL");
    }

    // ================= (4) MONOTONICITY: SH_2 irradiance >= SH_1 component-wise, strictly > for >=1 probe ===
    // The make-or-break underflow guard: the 2nd bounce only ADDS non-negative indirect light, so the
    // reconstructed +Y irradiance is >= the single-bounce one everywhere, AND is measurably greater for at
    // least one probe (so the integer indirect did NOT truncate to zero).
    {
        std::vector<FxProbeSH> sh1(probes), sh2(probes);
        gi::BounceProbes(scene, grid, 1, std::span<FxProbeSH>(sh1));
        gi::BounceProbes(scene, grid, 2, std::span<FxProbeSH>(sh2));

        const FxVec3 nrm{0, kOne, 0};
        bool allGe = true;
        int brighter = 0;
        for (int p = 0; p < probes; ++p) {
            GiRadiance i1 = gi::FxSHEvaluate(sh1[p], nrm);
            GiRadiance i2 = gi::FxSHEvaluate(sh2[p], nrm);
            if (i2.r < i1.r || i2.g < i1.g || i2.b < i1.b) allGe = false;
            if (i2.r > i1.r || i2.g > i1.g || i2.b > i1.b) ++brighter;
        }
        check(allGe, "monotonicity: SH_2 +Y irradiance >= SH_1 component-wise (bounce only ADDS light)");
        check(brighter >= 1, "monotonicity: SH_2 strictly brighter for >=1 probe (no integer underflow)");
        std::printf("  2nd-bounce brighter probes: %d of %d\n", brighter, probes);
    }

    // ================= (5) CONVERGENCE / IN-RANGE: maxIrr across K<=3 stays < 2.0 (no overflow) =================
    // The GI2 [-2,2] headroom must hold under the bounded feedback. Reconstruct the max +Y irradiance over a
    // few normals at every K and assert it never reaches the 2.0 bound.
    {
        const FxVec3 normals[3] = { FxVec3{0, kOne, 0}, FxVec3{kOne, 0, 0}, FxVec3{0, 0, kOne} };
        bool inRange = true;
        fx worst = 0;
        for (int K = 1; K <= gi::kGiMaxBounces; ++K) {
            std::vector<FxProbeSH> shK(probes);
            gi::BounceProbes(scene, grid, K, std::span<FxProbeSH>(shK));
            // Also assert every stored coeff is within the headroom (the GI2 envelope under feedback).
            for (int p = 0; p < probes; ++p)
                for (int i = 0; i < 9; ++i)
                    for (int c = 0; c < 3; ++c) {
                        int64_t a = shK[p].coeff[i][c]; if (a < 0) a = -a;
                        if (a >= kHeadroom) inRange = false;
                    }
            for (const FxVec3& n : normals) {
                fx m = gi::GiMaxIrradiance(std::span<const FxProbeSH>(shK), n);
                if (m > worst) worst = m;
                if ((int64_t)m >= kHeadroom) inRange = false;
            }
        }
        check(inRange, "in-range: every coeff AND maxIrr across K<=3 stays < 2.0 (GI2 headroom holds)");
        std::printf("  worst irradiance across K<=3: %.5f (must be < 2.0)\n", f(worst));
    }

    // ================= (6) COLORED-WALL TINT: the bounce shows an indirect channel imbalance =================
    // The GI1 scene has a warm-RED left wall (primIndex 0, albedo ~0.78,0.30,0.26) and a GREEN right wall
    // (primIndex 2). The 2nd bounce injects each wall's albedo-tinted radiance into its neighbors, so the
    // indirect contribution is NOT grey: summed across probes, the RED-channel gain of SH_2 over SH_1 differs
    // from the BLUE-channel gain (a color bleed, not a uniform brightening). This proves the bounce CARRIES
    // COLOR (light bounces off the red wall and tints — the headline visual claim), deterministically.
    {
        std::vector<FxProbeSH> sh1(probes), sh2(probes);
        gi::BounceProbes(scene, grid, 1, std::span<FxProbeSH>(sh1));
        gi::BounceProbes(scene, grid, 2, std::span<FxProbeSH>(sh2));
        const FxVec3 nrm{0, kOne, 0};
        int64_t gainR = 0, gainG = 0, gainB = 0;
        for (int p = 0; p < probes; ++p) {
            GiRadiance i1 = gi::FxSHEvaluate(sh1[p], nrm);
            GiRadiance i2 = gi::FxSHEvaluate(sh2[p], nrm);
            gainR += (int64_t)i2.r - i1.r;
            gainG += (int64_t)i2.g - i1.g;
            gainB += (int64_t)i2.b - i1.b;
        }
        // The bounce adds light (a positive total gain) and the per-channel gains are NOT all equal (a tint).
        bool addsLight = (gainR > 0) || (gainG > 0) || (gainB > 0);
        bool tinted = (gainR != gainG) || (gainR != gainB) || (gainG != gainB);
        check(addsLight, "colored-wall: the 2nd bounce adds light (positive summed irradiance gain)");
        check(tinted, "colored-wall: the bounce is TINTED (per-channel gain imbalance — color bleed)");
        std::printf("  bounce gain RGB: (%lld, %lld, %lld) [tinted -> not all equal]\n",
                    (long long)gainR, (long long)gainG, (long long)gainB);
    }

    // ================= (7) DETERMINISM: two BounceProbes runs byte-identical =================
    {
        std::vector<FxProbeSH> a(probes), b(probes);
        gi::BounceProbes(scene, grid, gi::kGiMaxBounces, std::span<FxProbeSH>(a));
        gi::BounceProbes(scene, grid, gi::kGiMaxBounces, std::span<FxProbeSH>(b));
        bool same = std::memcmp(a.data(), b.data(), (size_t)probes * sizeof(FxProbeSH)) == 0;
        check(same, "determinism: two BounceProbes(K=3) runs BYTE-IDENTICAL");
    }

    // ================= (8) NO-OP GRID: ProbeCount==0 -> outSH untouched =================
    {
        GiProbeGrid empty; empty.nx = 0; empty.ny = 4; empty.nz = 4;
        std::vector<FxProbeSH> out(4);
        for (auto& s : out) { for (int i=0;i<9;++i) for (int c=0;c<3;++c) s.coeff[i][c] = (fx)777; }
        gi::BounceProbes(scene, empty, 2, std::span<FxProbeSH>(out));
        bool untouched = true;
        for (const auto& s : out) for (int i=0;i<9;++i) for (int c=0;c<3;++c) if (s.coeff[i][c] != 777) untouched = false;
        check(untouched, "no-op grid: ProbeCount==0 -> outSH UNTOUCHED");
    }

    if (g_fail == 0) std::printf("gi_bounce_test: all GI4 integer multi-bounce invariants PASS\n");
    return g_fail == 0 ? 0 : 1;
}
