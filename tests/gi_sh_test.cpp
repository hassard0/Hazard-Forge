// Slice GI2 — Deterministic Lumen-class GI: INTEGER 3rd-order SH ENCODE (THE CRUX, FLAGSHIP #29). The
// CPU-side validation that SETTLES THE CRUX before any GPU trust: the integer SH encode/evaluate must
// round-trip a directional radiance, the dynamic-range envelope must hold (no coeff / no reconstructed
// irradiance exceeds the < 2.0 Q16.16 headroom, AND band-2 coeffs are NON-ZERO for a directional input —
// NO underflow-to-zero), zero radiance -> zero SH, FxSHEvaluate >= 0, a constant ambient radiance ->
// only the DC (l=0) coeff non-zero (SH orthogonality), and two encodes are bit-identical (determinism).
// The GPU SH == CPU SH memcmp proof lives in --gi2-shencode-shot (it needs a real GPU); this pure-CPU test
// pins the exact integer-SH contract that proof rests on.
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

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Q16.16 -> double, for the human-readable envelope prints (NOT on any bit-exact path).
static double f(fx v) { return (double)v / (double)(int)kOne; }

// The < 2.0 Q16.16 headroom bound (2.0 in Q16.16 == 2*kOne).
static const int64_t kHeadroom = 2 * (int64_t)kOne;

int main() {
    HF_TEST_MAIN_INIT();

    const int N = gi::kGiRaysPerProbe;   // 16

    // Helper: build a 16-ray radiance slice from a per-ray RGB callback.
    auto makeRays = [&](auto fn) {
        std::vector<GiRadiance> rays(N);
        for (int r = 0; r < N; ++r) rays[r] = fn(r);
        return rays;
    };

    // ================= (1) ZERO radiance -> ZERO SH =================
    {
        auto rays = makeRays([](int) { return GiRadiance{0, 0, 0, 0}; });
        FxProbeSH sh = gi::FxSHEncodeProbe(rays.data(), N);
        bool allZero = true;
        for (int i = 0; i < 9; ++i)
            for (int c = 0; c < 3; ++c)
                if (sh.coeff[i][c] != 0) allZero = false;
        check(allZero, "zero radiance -> zero SH (every coeff exactly 0)");
    }

    // ================= (2) CONSTANT ambient radiance -> ONLY the DC (l=0) coeff dominates =================
    // SH orthogonality: a uniform field projects almost entirely onto Y00; bands 1..2 are ~0 (the 16-ray
    // discretization leaves a tiny residual, well below the DC term). We assert the DC coeff is large and
    // strictly dominates every band-1/band-2 coeff.
    {
        const fx amb = kOne / 2;   // 0.5 radiance, all channels
        auto rays = makeRays([&](int) { return GiRadiance{amb, amb, amb, 0}; });
        FxProbeSH sh = gi::FxSHEncodeProbe(rays.data(), N);
        // DC: coeff[0] ~= Y00 * 0.5 ~= 0.141 -> ~9243 in Q16.16.
        fx dc = sh.coeff[0][0];
        bool dcBig = dc > kOne / 8;   // > 0.125
        check(dcBig, "constant ambient -> DC (l=0) coeff is large");
        bool dcDominates = true;
        for (int i = 1; i < 9; ++i) {
            for (int c = 0; c < 3; ++c) {
                fx v = sh.coeff[i][c];
                if (v < 0) v = -v;
                if (v >= dc / 4) dcDominates = false;   // every non-DC coeff << DC (orthogonality)
            }
        }
        check(dcDominates, "constant ambient -> only the DC coeff dominates (SH orthogonality: bands 1-2 ~0)");
    }

    // ================= (3) DIRECTIONAL input -> band-2 NON-ZERO (NO underflow), in-range =================
    // Light ONE ray (ray 6, a dir with strong band-2 projection) at full radiance; assert SOME band-2 coeff
    // is non-zero (the encode did NOT underflow the tiny band-2 fractions to the LSB) AND every coeff is
    // within the < 2.0 headroom.
    {
        const int litRay = 6;
        auto rays = makeRays([&](int r) {
            return (r == litRay) ? GiRadiance{kOne, kOne, kOne, 0} : GiRadiance{0, 0, 0, 0};
        });
        FxProbeSH sh = gi::FxSHEncodeProbe(rays.data(), N);
        // band-2 == coeff indices 4..8.
        bool band2NonZero = false;
        fx maxBand2 = 0;
        for (int i = 4; i < 9; ++i) {
            for (int c = 0; c < 3; ++c) {
                fx v = sh.coeff[i][c];
                fx a = v < 0 ? -v : v;
                if (v != 0) band2NonZero = true;
                if (a > maxBand2) maxBand2 = a;
            }
        }
        check(band2NonZero, "directional input -> band-2 coeff NON-ZERO (no underflow-to-zero — THE CRUX)");
        // headroom on the coeffs.
        bool inRange = true;
        for (int i = 0; i < 9; ++i)
            for (int c = 0; c < 3; ++c) {
                int64_t a = sh.coeff[i][c]; if (a < 0) a = -a;
                if (a >= kHeadroom) inRange = false;
            }
        check(inRange, "directional input -> every coeff within the < 2.0 Q16.16 headroom");
        std::printf("  directional(ray%d): maxBand2coeff=%.6f\n", litRay, f(maxBand2));
    }

    // ================= (4) ROUND-TRIP within a documented band =================
    // Encode a radiance that is bright along +Y (a few rays in the upper hemisphere), evaluate the SH
    // irradiance in +Y, and confirm it RECOVERS a positive irradiance correlated with the input. The
    // integer 3rd-order SH is a LOW-FREQUENCY reconstruction (9 coeffs, 16 rays), so the band is generous:
    // a +Y-lit probe must read BRIGHTER in +Y than a -Y-lit probe reads in +Y (directional sense recovered).
    {
        // +Y-lit: rays with z-component... actually our dirs use z as the "up" axis (kGiProbeDirs.z spans
        // -1..1). Light the rays whose dir has the LARGEST +Y component, then evaluate in +Y.
        auto litTowardPlusY = makeRays([&](int r) {
            const FxVec3& d = gi::kGiProbeDirs[r];
            return (d.y > 0) ? GiRadiance{kOne, kOne, kOne, 0} : GiRadiance{0, 0, 0, 0};
        });
        auto litTowardMinusY = makeRays([&](int r) {
            const FxVec3& d = gi::kGiProbeDirs[r];
            return (d.y < 0) ? GiRadiance{kOne, kOne, kOne, 0} : GiRadiance{0, 0, 0, 0};
        });
        FxProbeSH shPlus  = gi::FxSHEncodeProbe(litTowardPlusY.data(), N);
        FxProbeSH shMinus = gi::FxSHEncodeProbe(litTowardMinusY.data(), N);
        FxVec3 plusY{0, kOne, 0};
        GiRadiance irrPlus  = gi::FxSHEvaluate(shPlus, plusY);
        GiRadiance irrMinus = gi::FxSHEvaluate(shMinus, plusY);
        check(irrPlus.r > irrMinus.r,
              "round-trip: a +Y-lit probe reads BRIGHTER in +Y than a -Y-lit probe (directional sense)");
        // The recovered irradiance is positive and in-band.
        check(irrPlus.r > 0, "round-trip: +Y-lit probe evaluated in +Y -> positive irradiance");
        std::printf("  round-trip: irr(+Y-lit, +Y)=%.5f  irr(-Y-lit, +Y)=%.5f\n",
                    f(irrPlus.r), f(irrMinus.r));
    }

    // ================= (5) FxSHEvaluate >= 0 (clamped, no negative irradiance) =================
    {
        // A pathological single-ray directional input can produce negative SH lobes; FxSHEvaluate must clamp.
        auto rays = makeRays([&](int r) {
            return (r == 0) ? GiRadiance{kOne, kOne, kOne, 0} : GiRadiance{0, 0, 0, 0};
        });
        FxProbeSH sh = gi::FxSHEncodeProbe(rays.data(), N);
        bool allNonNeg = true;
        for (int d = 0; d < N; ++d) {
            GiRadiance irr = gi::FxSHEvaluate(sh, gi::kGiProbeDirs[d]);
            if (irr.r < 0 || irr.g < 0 || irr.b < 0) allNonNeg = false;
        }
        check(allNonNeg, "FxSHEvaluate >= 0 (clamped, no negative irradiance) in every direction");
    }

    // ================= (6) DYNAMIC-RANGE ENVELOPE across the GI scene (the crux bound) =================
    // Build the pinned GI1 scene + the showcase 4x4x4 grid, trace, encode, and assert: NO coeff and NO
    // reconstructed irradiance (over every probe x every probe-dir) exceeds the < 2.0 headroom, AND at
    // least one band-2 coeff somewhere is non-zero (the scene's directional lighting exercises band-2).
    {
        gi::GiScene1 sc = gi::BuildGi1Scene();
        gi::GiProbeGrid grid;
        grid.origin  = FxVec3{gi::GiF(-2, 1), gi::GiF(1, 1), gi::GiF(0, 1)};
        grid.spacing = gi::GiF(1, 1);
        grid.nx = 4; grid.ny = 4; grid.nz = 4;
        const int probes = gi::ProbeCount(grid);

        std::vector<GiRadiance> rad((size_t)probes * N);
        gi::TraceProbeRays(grid, sc.scene, std::span<GiRadiance>(rad));
        std::vector<FxProbeSH> sh(probes);
        gi::EncodeAllProbes(grid, std::span<const GiRadiance>(rad), std::span<FxProbeSH>(sh));

        int64_t maxAbsCoeff = 0;
        bool sceneBand2NonZero = false;
        for (int p = 0; p < probes; ++p) {
            for (int i = 0; i < 9; ++i)
                for (int c = 0; c < 3; ++c) {
                    int64_t a = sh[p].coeff[i][c]; if (a < 0) a = -a;
                    if (a > maxAbsCoeff) maxAbsCoeff = a;
                    if (i >= 4 && sh[p].coeff[i][c] != 0) sceneBand2NonZero = true;
                }
        }
        int64_t maxIrr = 0;
        for (int p = 0; p < probes; ++p) {
            for (int d = 0; d < N; ++d) {
                GiRadiance irr = gi::FxSHEvaluate(sh[p], gi::kGiProbeDirs[d]);
                int64_t vals[3] = {irr.r, irr.g, irr.b};
                for (int k = 0; k < 3; ++k) { int64_t a = vals[k] < 0 ? -vals[k] : vals[k];
                    if (a > maxIrr) maxIrr = a; }
            }
        }
        check(maxAbsCoeff < kHeadroom, "envelope: maxAbsCoeff < 2.0 (no overflow on the GI scene)");
        check(maxIrr < kHeadroom, "envelope: maxIrr < 2.0 (no overflow on the GI scene)");
        check(sceneBand2NonZero, "envelope: band-2 NON-ZERO on the GI scene (no underflow — directional lit)");
        std::printf("  envelope: maxAbsCoeff=%.5f (<2.0)  maxIrr=%.5f (<2.0)  band2NonZero=%s\n",
                    f((fx)maxAbsCoeff), f((fx)maxIrr), sceneBand2NonZero ? "true" : "false");
    }

    // ================= (7) DETERMINISM: two encodes byte-identical =================
    {
        gi::GiScene1 sc = gi::BuildGi1Scene();
        gi::GiProbeGrid grid;
        grid.origin  = FxVec3{gi::GiF(-2, 1), gi::GiF(1, 1), gi::GiF(0, 1)};
        grid.spacing = gi::GiF(1, 1);
        grid.nx = 4; grid.ny = 4; grid.nz = 4;
        const int probes = gi::ProbeCount(grid);
        std::vector<GiRadiance> rad((size_t)probes * N);
        gi::TraceProbeRays(grid, sc.scene, std::span<GiRadiance>(rad));
        std::vector<FxProbeSH> sh1(probes), sh2(probes);
        gi::EncodeAllProbes(grid, std::span<const GiRadiance>(rad), std::span<FxProbeSH>(sh1));
        gi::EncodeAllProbes(grid, std::span<const GiRadiance>(rad), std::span<FxProbeSH>(sh2));
        bool same = std::memcmp(sh1.data(), sh2.data(), (size_t)probes * sizeof(FxProbeSH)) == 0;
        check(same, "determinism: two EncodeAllProbes runs BYTE-IDENTICAL");
    }

    // ================= (8) NO-OP: ProbeCount==0 -> EncodeAllProbes writes nothing =================
    {
        gi::GiProbeGrid grid;
        grid.nx = 0; grid.ny = 4; grid.nz = 4;
        check(gi::ProbeCount(grid) == 0, "ProbeCount: a 0-dim grid -> 0 probes");
        std::vector<FxProbeSH> sh(8);
        for (auto& s : sh) for (int i = 0; i < 9; ++i) for (int c = 0; c < 3; ++c) s.coeff[i][c] = (fx)777;
        std::vector<GiRadiance> rad(8);
        gi::EncodeAllProbes(grid, std::span<const GiRadiance>(rad), std::span<FxProbeSH>(sh));
        bool untouched = true;
        for (const auto& s : sh)
            for (int i = 0; i < 9; ++i) for (int c = 0; c < 3; ++c) if (s.coeff[i][c] != 777) untouched = false;
        check(untouched, "EncodeAllProbes: ProbeCount==0 -> the SH buffer is UNTOUCHED (the no-op)");
    }

    if (g_fail == 0) std::printf("gi_sh_test: all GI2 integer-SH CRUX invariants PASS\n");
    return g_fail == 0 ? 0 : 1;
}
