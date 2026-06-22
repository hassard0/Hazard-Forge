// Slice GI3 — Deterministic Lumen-class GI: INTEGER TRILINEAR SH INTERPOLATION (the continuous irradiance
// field, FLAGSHIP #29). The CPU-side validation that SETTLES the integer trilinear blend before any GPU
// trust: the 8 corner trilinear weights must sum to kOne EXACTLY (partition of unity — a HARD invariant);
// the 8 corner indices must be the correct cell corners; a query exactly AT a probe position must reproduce
// that probe's FxSHEvaluate irradiance BYTE-for-byte (lattice-point identity — the falsifiable proof the
// index/weight math is right); a query at a cell CENTER blends all 8 equally (w == kOne/8 each); out-of-grid
// queries clamp to the boundary; a point nearer a brighter probe reads brighter (monotone sense); and two
// runs are byte-identical (determinism). The GPU==CPU irradiance memcmp proof lives in --gi3-interp-shot (it
// needs a real GPU); this pure-CPU test pins the exact integer-interpolation contract that proof rests on.
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
using gi::FxProbeWeights;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static double f(fx v) { return (double)v / (double)(int)kOne; }

// Build the showcase 4x4x4 grid + a CPU-encoded SH buffer over the pinned GI1 scene (the shared fixture).
static GiProbeGrid MakeGrid() {
    GiProbeGrid grid;
    grid.origin  = FxVec3{gi::GiF(-2, 1), gi::GiF(1, 1), gi::GiF(0, 1)};
    grid.spacing = gi::GiF(1, 1);
    grid.nx = 4; grid.ny = 4; grid.nz = 4;
    return grid;
}
static std::vector<FxProbeSH> MakeSH(const GiProbeGrid& grid) {
    gi::GiScene1 sc = gi::BuildGi1Scene();
    const int N = gi::kGiRaysPerProbe;
    const int probes = gi::ProbeCount(grid);
    std::vector<GiRadiance> rad((size_t)probes * N);
    gi::TraceProbeRays(grid, sc.scene, std::span<GiRadiance>(rad));
    std::vector<FxProbeSH> sh(probes);
    gi::EncodeAllProbes(grid, std::span<const GiRadiance>(rad), std::span<FxProbeSH>(sh));
    return sh;
}

int main() {
    HF_TEST_MAIN_INIT();

    GiProbeGrid grid = MakeGrid();
    const int probes = gi::ProbeCount(grid);
    std::vector<FxProbeSH> sh = MakeSH(grid);
    std::span<const FxProbeSH> shSpan(sh);

    // ================= (1) PARTITION OF UNITY: Σ w == kOne EXACTLY across many points =================
    // Sweep a dense set of query points across (and beyond) the lattice footprint; at EVERY point the 8
    // trilinear weights must sum to EXACTLY kOne (integer, maxErr 0).
    {
        int64_t maxErr = 0;
        // 11x11x11 points spanning [origin - spacing, origin + (dim)*spacing] (deliberately overshooting
        // the grid bounds to exercise the out-of-grid clamp too).
        for (int iz = 0; iz <= 10; ++iz)
            for (int iy = 0; iy <= 10; ++iy)
                for (int ix = 0; ix <= 10; ++ix) {
                    fx x = (fx)(grid.origin.x - grid.spacing + (fx)(((int64_t)ix * 5 * (int64_t)grid.spacing) / 10));
                    fx y = (fx)(grid.origin.y - grid.spacing + (fx)(((int64_t)iy * 5 * (int64_t)grid.spacing) / 10));
                    fx z = (fx)(grid.origin.z - grid.spacing + (fx)(((int64_t)iz * 5 * (int64_t)grid.spacing) / 10));
                    FxProbeWeights w = gi::FxNearestProbes(grid, FxVec3{x, y, z});
                    int64_t s = 0;
                    for (int c = 0; c < 8; ++c) s += (int64_t)w.w[c];
                    int64_t err = s - (int64_t)kOne; if (err < 0) err = -err;
                    if (err > maxErr) maxErr = err;
                    // Every weight is non-negative (a valid convex blend).
                    for (int c = 0; c < 8; ++c) check(w.w[c] >= 0, "trilinear weight >= 0 (convex blend)");
                }
        check(maxErr == 0, "partition of unity: Σ w == kOne EXACTLY across the sweep (maxErr 0)");
        std::printf("  partition-of-unity sweep: maxErr=%lld (must be 0)\n", (long long)maxErr);
    }

    // ================= (2) THE 8 INDICES ARE THE CORRECT CELL CORNERS =================
    // A query strictly inside cell (1,1,1) -> the 8 corners are probes (1..2, 1..2, 1..2). Verify the
    // bit-pattern: corner c == ProbeFlatIndex(1+(c&1), 1+((c>>1)&1), 1+((c>>2)&1)).
    {
        // Point at the center of cell (1,1,1): origin + (1.5,1.5,1.5)*spacing.
        fx half = (fx)(grid.spacing / 2);
        fx x = (fx)(grid.origin.x + grid.spacing + half);
        fx y = (fx)(grid.origin.y + grid.spacing + half);
        fx z = (fx)(grid.origin.z + grid.spacing + half);
        FxProbeWeights w = gi::FxNearestProbes(grid, FxVec3{x, y, z});
        bool idxOk = true;
        for (int c = 0; c < 8; ++c) {
            int sx = (c & 1), sy = ((c >> 1) & 1), sz = ((c >> 2) & 1);
            int expect = gi::ProbeFlatIndex(grid, 1 + sx, 1 + sy, 1 + sz);
            if (w.idx[c] != expect) idxOk = false;
        }
        check(idxOk, "the 8 corner indices are the correct cell-(1,1,1) corners");
    }

    // ================= (3) CELL-CENTER -> w == kOne/8 each =================
    // A query at a cell center has frac == kOne/2 on every axis -> every corner weight is (1/2)^3 = 1/8.
    // kOne/2 = 32768, so (1/2)^3 in Q16.16 == kOne/8 == 8192. The leftover-at-corner-7 construction must
    // STILL land all 8 at exactly kOne/8 (the products are exact powers of two -> no truncation loss).
    {
        fx half = (fx)(grid.spacing / 2);
        fx x = (fx)(grid.origin.x + grid.spacing + half);
        fx y = (fx)(grid.origin.y + grid.spacing + half);
        fx z = (fx)(grid.origin.z + grid.spacing + half);
        FxProbeWeights w = gi::FxNearestProbes(grid, FxVec3{x, y, z});
        bool eighths = true;
        for (int c = 0; c < 8; ++c) if (w.w[c] != kOne / 8) eighths = false;
        check(eighths, "cell center -> w == kOne/8 each (8192) for all 8 corners");
        std::printf("  cell-center weights: w[0]=%d (kOne/8=%d)\n", w.w[0], kOne / 8);
    }

    // ================= (4) LATTICE-POINT IDENTITY (byte-exact) =================
    // A query EXACTLY at probe p's position -> the blend is EXACTLY probe p's SH -> FxInterpolateIrradiance
    // == FxSHEvaluate(sh[p], normal) BYTE-for-byte, for every probe and a set of normals.
    {
        const FxVec3 normals[3] = { FxVec3{0, kOne, 0}, FxVec3{kOne, 0, 0}, FxVec3{0, 0, kOne} };
        bool allExact = true;
        for (int p = 0; p < probes; ++p) {
            FxVec3 pos = gi::ProbePos(grid, p);
            // The weights at a probe pos: corner 0 (the lo,lo,lo corner of the cell this probe anchors) must
            // carry kOne. For the TOP-boundary probes the floor-cell clamp anchors them as the +offset
            // corner, so the SH blend still resolves to exactly this probe's SH (frac==0 zeros the rest).
            FxProbeWeights w = gi::FxNearestProbes(grid, pos);
            int64_t s = 0; for (int c = 0; c < 8; ++c) s += w.w[c];
            if (s != (int64_t)kOne) allExact = false;   // partition still holds at the lattice point
            for (const FxVec3& n : normals) {
                GiRadiance blended = gi::FxInterpolateIrradiance(grid, shSpan, pos, n);
                GiRadiance direct  = gi::FxSHEvaluate(sh[p], n);
                if (std::memcmp(&blended, &direct, sizeof(GiRadiance)) != 0) allExact = false;
            }
        }
        check(allExact, "lattice-point identity: query@probe == that probe's FxSHEvaluate, BYTE-exact");
    }

    // ================= (5) OUT-OF-GRID CLAMP =================
    // A query far below the grid clamps to the (0,0,0) corner cell with frac 0 -> identical to probe 0's SH;
    // far above clamps to the top corner -> identical to the last probe's SH.
    {
        FxVec3 farBelow{(fx)(grid.origin.x - grid.spacing * 10),
                        (fx)(grid.origin.y - grid.spacing * 10),
                        (fx)(grid.origin.z - grid.spacing * 10)};
        FxVec3 farAbove{(fx)(grid.origin.x + grid.spacing * 100),
                        (fx)(grid.origin.y + grid.spacing * 100),
                        (fx)(grid.origin.z + grid.spacing * 100)};
        FxVec3 n{0, kOne, 0};
        GiRadiance below = gi::FxInterpolateIrradiance(grid, shSpan, farBelow, n);
        GiRadiance probe0 = gi::FxSHEvaluate(sh[0], n);
        check(std::memcmp(&below, &probe0, sizeof(GiRadiance)) == 0,
              "out-of-grid (far below) clamps to probe 0 exactly");
        GiRadiance above = gi::FxInterpolateIrradiance(grid, shSpan, farAbove, n);
        GiRadiance lastP = gi::FxSHEvaluate(sh[probes - 1], n);
        check(std::memcmp(&above, &lastP, sizeof(GiRadiance)) == 0,
              "out-of-grid (far above) clamps to the last probe exactly");
    }

    // ================= (6) MONOTONE: a point nearer a brighter probe is brighter =================
    // Build a SYNTHETIC 2-probe-per-axis grid where one corner probe is bright (a large DC coeff) and the
    // opposite corner is dark; a query nearer the bright probe must read a brighter +Y irradiance.
    {
        GiProbeGrid g2; g2.origin = FxVec3{0, 0, 0}; g2.spacing = kOne; g2.nx = 2; g2.ny = 2; g2.nz = 2;
        const int p2 = gi::ProbeCount(g2);
        std::vector<FxProbeSH> sh2(p2);
        for (auto& s : sh2) s = FxProbeSH{};
        // Probe 0 (corner 0,0,0) BRIGHT (DC coeff large, all channels); probe 7 (1,1,1) dark; rest mid.
        int bright = gi::ProbeFlatIndex(g2, 0, 0, 0);
        sh2[bright].coeff[0][0] = kOne / 2; sh2[bright].coeff[0][1] = kOne / 2; sh2[bright].coeff[0][2] = kOne / 2;
        FxVec3 n{0, kOne, 0};
        // Near the bright corner (0.1,0.1,0.1) vs near the dark corner (0.9,0.9,0.9).
        FxVec3 near{(fx)(kOne / 10), (fx)(kOne / 10), (fx)(kOne / 10)};
        FxVec3 farp{(fx)(kOne * 9 / 10), (fx)(kOne * 9 / 10), (fx)(kOne * 9 / 10)};
        GiRadiance irrNear = gi::FxInterpolateIrradiance(g2, std::span<const FxProbeSH>(sh2), near, n);
        GiRadiance irrFar  = gi::FxInterpolateIrradiance(g2, std::span<const FxProbeSH>(sh2), farp, n);
        check(irrNear.r > irrFar.r, "monotone: a point nearer the bright probe reads BRIGHTER (+Y)");
        std::printf("  monotone: irr(near bright)=%.5f  irr(far)=%.5f\n", f(irrNear.r), f(irrFar.r));
    }

    // ================= (7) DETERMINISM: two InterpolateField runs byte-identical =================
    {
        // A dense query set: the GiFieldToImage slice points, gathered as a field.
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
        std::vector<GiRadiance> out1((size_t)QW * QH), out2((size_t)QW * QH);
        gi::InterpolateField(grid, shSpan, std::span<const FxVec3>(pts), std::span<const FxVec3>(nrm),
                             std::span<GiRadiance>(out1));
        gi::InterpolateField(grid, shSpan, std::span<const FxVec3>(pts), std::span<const FxVec3>(nrm),
                             std::span<GiRadiance>(out2));
        bool same = std::memcmp(out1.data(), out2.data(), out1.size() * sizeof(GiRadiance)) == 0;
        check(same, "determinism: two InterpolateField runs BYTE-IDENTICAL");
        // The field is non-trivial (some pixel non-zero -> the blend actually lit something).
        bool nonTrivial = false;
        for (const auto& g : out1) if (g.r != 0 || g.g != 0 || g.b != 0) nonTrivial = true;
        check(nonTrivial, "the interpolated field is non-trivial (some irradiance non-zero)");
    }

    // ================= (8) NO-OP: empty query set -> InterpolateField writes nothing =================
    {
        std::vector<FxVec3> noPts, noNrm;
        std::vector<GiRadiance> out(4);
        for (auto& g : out) g = GiRadiance{(fx)777, (fx)777, (fx)777, 0};
        gi::InterpolateField(grid, shSpan, std::span<const FxVec3>(noPts), std::span<const FxVec3>(noNrm),
                             std::span<GiRadiance>(out));
        bool untouched = true;
        for (const auto& g : out) if (g.r != 777 || g.g != 777 || g.b != 777) untouched = false;
        check(untouched, "InterpolateField: empty query set -> output UNTOUCHED (the no-op)");
        check(gi::GiInterpDispatchGroups(0) == 0, "GiInterpDispatchGroups(0) == 0 (the 0-dispatch no-op)");
        check(gi::GiInterpDispatchGroups(64) == 1 && gi::GiInterpDispatchGroups(65) == 2,
              "GiInterpDispatchGroups sizes correctly");
    }

    // ================= (9) DEGENERATE GRID -> disabled fallback =================
    {
        GiProbeGrid bad; bad.nx = 0; bad.ny = 4; bad.nz = 4;
        FxProbeWeights w = gi::FxNearestProbes(bad, FxVec3{0, 0, 0});
        check(w.w[0] == kOne, "degenerate grid -> w[0] == kOne (disabled fallback)");
        bool restZero = true; for (int c = 1; c < 8; ++c) if (w.w[c] != 0) restZero = false;
        check(restZero, "degenerate grid -> w[1..7] == 0");
    }

    if (g_fail == 0) std::printf("gi_interp_test: all GI3 integer trilinear-SH-interp invariants PASS\n");
    return g_fail == 0 ? 0 : 1;
}
