// Slice DJ — DDGI Slice 3: Probe SH-Encode. Pure CPU math: the 3rd-order real-SH basis (SHBasis9), the
// per-sample fma accumulation (SHEncodeAccumulate), the solid-angle normalization (SHNormalize), and the
// cosine-lobe reconstruction (SHEvaluate) — the EXACT encode the --probesh-shot showcase + the Metal
// --probesh showcase run and the probe_sh_encode.comp shader copies VERBATIM, so the GPU per-probe SH
// SSBO is BIT-EXACT to this CPU reference over the same captured cube radiance. No device, ASan-eligible
// (links hf_core).
//
// Properties pinned (per the spec §5):
//   * SHBasis9: Y00 == 0.282095 const; the axis values (+X/+Y/+Z) hand-checked vs the documented
//     polynomial constants; no NaN; ProbeSH is 108 bytes.
//   * Uniform radiance over a full sphere -> band-0 only: coeff[0] == uniform * (DC normalization),
//     coeff[1..8] ~= 0 (the classic SH sanity check).
//   * Directional lobe: a bright sample in direction d -> SHEvaluate(sh,d) > SHEvaluate(sh,-d).
//   * Zero radiance -> zero SH exactly (the zero-radiance == zero-SH proof's CPU half).
//   * Determinism: the same samples -> bit-identical SH.
#include "render/probe_sh.h"
#include "render/probe_gi.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace probesh = hf::render::probesh;
using hf::math::Vec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-5f) { return std::fabs(a - b) <= eps; }

// A deterministic full-sphere sample set (Fibonacci-sphere, reused from probe_gi.h) + its SHBasis9
// weights + the uniform per-sample solid-angle weight (4*pi / N). This mirrors how the showcase builds
// the host-precomputed sample table the GPU + CPU both read.
struct Sample { Vec3 dir; float basis[9]; float saWeight; };
static std::vector<Sample> MakeSamples(int n) {
    std::vector<Sample> s((size_t)n);
    const float sa = (4.0f * 3.14159265358979324f) / (float)n;   // uniform per-sample solid angle
    for (int i = 0; i < n; ++i) {
        s[i].dir = hf::render::probegi::FibonacciSphere(i, n);   // unit-length, deterministic
        probesh::SHBasis9(s[i].dir, s[i].basis);
        s[i].saWeight = sa;
    }
    return s;
}

int main() {
    HF_TEST_MAIN_INIT();
    // ====================================================================================
    // ProbeSH layout + SHBasis9 axis values + Y00 const + no NaN
    // ====================================================================================
    {
        check(sizeof(probesh::ProbeSH) == 108, "ProbeSH is 108 bytes (9 coeffs x 3 channels, std430)");

        float b[9];
        // Y00 is the documented DC constant on every direction.
        probesh::SHBasis9(Vec3{1, 0, 0}, b);
        check(approx(b[0], 0.282095f, 1e-6f), "SHBasis9: Y00 == 0.282095 (band-0 DC constant)");

        // +X axis: band-1 x term (out[3]) == kY1; the y,z band-1 terms 0.
        check(approx(b[1], 0.0f) && approx(b[2], 0.0f) && approx(b[3], probesh::kY1),
              "SHBasis9(+X): band-1 == {0, 0, kY1} (x term)");
        // +X band-2: 3z^2-1 == -1 -> out[6] == -kY2b; x^2-y^2 == 1 -> out[8] == kY2c; xy/yz/xz == 0.
        check(approx(b[4], 0.0f) && approx(b[5], 0.0f) && approx(b[6], -probesh::kY2b) &&
                  approx(b[7], 0.0f) && approx(b[8], probesh::kY2c),
              "SHBasis9(+X): band-2 == {0, 0, -kY2b, 0, kY2c}");

        // +Y axis: band-1 y term (out[1]) == kY1; band-2 x^2-y^2 == -1 -> out[8] == -kY2c.
        probesh::SHBasis9(Vec3{0, 1, 0}, b);
        check(approx(b[1], probesh::kY1) && approx(b[2], 0.0f) && approx(b[3], 0.0f),
              "SHBasis9(+Y): band-1 == {kY1, 0, 0} (y term)");
        check(approx(b[6], -probesh::kY2b) && approx(b[8], -probesh::kY2c),
              "SHBasis9(+Y): band-2 (3z^2-1)==-kY2b, (x^2-y^2)==-kY2c");

        // +Z axis: band-1 z term (out[2]) == kY1; band-2 3z^2-1 == 2 -> out[6] == 2*kY2b.
        probesh::SHBasis9(Vec3{0, 0, 1}, b);
        check(approx(b[2], probesh::kY1) && approx(b[1], 0.0f) && approx(b[3], 0.0f),
              "SHBasis9(+Z): band-1 == {0, kY1, 0} (z term)");
        check(approx(b[6], 2.0f * probesh::kY2b),
              "SHBasis9(+Z): band-2 (3z^2-1) == 2*kY2b");

        // No NaN/inf anywhere.
        bool finite = true;
        for (int i = 0; i < 9; ++i) if (!std::isfinite(b[i])) finite = false;
        check(finite, "SHBasis9: all basis values finite (polynomial, no NaN)");
    }

    // ====================================================================================
    // Uniform radiance over a full sphere -> band-0 only (the classic SH sanity check)
    // ====================================================================================
    {
        const int N = 256;
        auto samples = MakeSamples(N);
        const Vec3 uniform{0.6f, 0.4f, 0.2f};   // a constant radiance over the whole sphere

        probesh::ProbeSH sh{};
        float total = 0.0f;
        for (const auto& s : samples) {
            probesh::SHEncodeAccumulate(sh, uniform, s.basis, s.saWeight);
            total += s.saWeight;
        }
        probesh::SHNormalize(sh, total);

        // Band 0: the DC coefficient of a constant function L is L * Y00 (projection of a constant onto
        // the orthonormal DC basis over the unit sphere == L * 4pi * Y00 / 4pi == L * Y00).
        check(approx(sh.coeff[0][0], uniform.x * probesh::kY00, 2e-3f) &&
                  approx(sh.coeff[0][1], uniform.y * probesh::kY00, 2e-3f) &&
                  approx(sh.coeff[0][2], uniform.z * probesh::kY00, 2e-3f),
              "Uniform sphere: coeff[0] == uniform * Y00 (band-0 DC)");

        // Bands 1..8: ~0 (a constant function has no directional content).
        bool higherBandsZero = true;
        for (int i = 1; i < 9; ++i)
            for (int c = 0; c < 3; ++c)
                if (std::fabs(sh.coeff[i][c]) > 5e-3f) higherBandsZero = false;
        check(higherBandsZero, "Uniform sphere: coeff[1..8] ~= 0 (no directional content)");

        // The reconstructed irradiance of a uniform radiance L is the same in every direction == pi*L
        // (the cosine-lobe DC term A0 * coeff[0] * Y00 == pi * L * Y00^2 ... actually == pi*L). Spot-check
        // two opposite directions agree (uniform -> isotropic reconstruction).
        Vec3 up = probesh::SHEvaluate(sh, Vec3{0, 1, 0});
        Vec3 dn = probesh::SHEvaluate(sh, Vec3{0, -1, 0});
        check(approx(up.x, dn.x, 3e-3f) && approx(up.y, dn.y, 3e-3f) && approx(up.z, dn.z, 3e-3f),
              "Uniform sphere: reconstruction is isotropic (up == down)");
    }

    // ====================================================================================
    // Directional lobe: a bright sample in direction d -> SHEvaluate(d) > SHEvaluate(-d)
    // ====================================================================================
    {
        const int N = 256;
        auto samples = MakeSamples(N);
        const Vec3 d = hf::math::normalize(Vec3{0.3f, 0.8f, 0.5f});

        // Bright radiance for samples in the hemisphere around d, dark elsewhere (a directional lobe).
        probesh::ProbeSH sh{};
        float total = 0.0f;
        for (const auto& s : samples) {
            float c = hf::math::dot(s.dir, d);
            Vec3 rad = (c > 0.0f) ? Vec3{c, c, c} : Vec3{0, 0, 0};   // bright toward d
            probesh::SHEncodeAccumulate(sh, rad, s.basis, s.saWeight);
            total += s.saWeight;
        }
        probesh::SHNormalize(sh, total);

        Vec3 towardD  = probesh::SHEvaluate(sh, d);
        Vec3 awayFromD = probesh::SHEvaluate(sh, -d);
        check(towardD.x > awayFromD.x && towardD.y > awayFromD.y && towardD.z > awayFromD.z,
              "Directional lobe: SHEvaluate(d) > SHEvaluate(-d) (captures the directional distribution)");
        // The raw (non-cosine) reconstruction toward d is positive (the lobe energy).
        Vec3 rawD = probesh::SHEvaluate(sh, d, /*cosineLobe=*/false);
        check(rawD.x > 0.0f, "Directional lobe: raw SHEvaluate(d) is positive");
    }

    // ====================================================================================
    // Zero radiance -> zero SH exactly (the zero-radiance == zero-SH proof's CPU half)
    // ====================================================================================
    {
        const int N = 128;
        auto samples = MakeSamples(N);
        probesh::ProbeSH sh{};
        float total = 0.0f;
        for (const auto& s : samples) {
            probesh::SHEncodeAccumulate(sh, Vec3{0, 0, 0}, s.basis, s.saWeight);
            total += s.saWeight;
        }
        probesh::SHNormalize(sh, total);
        probesh::ProbeSH zero{};   // value-initialized: all coeffs exactly 0
        check(std::memcmp(&sh, &zero, sizeof(probesh::ProbeSH)) == 0,
              "Zero radiance -> ProbeSH is EXACTLY all-zero (byte-identical to a cleared record)");

        // SHEvaluate of a zero SH is zero in every direction.
        Vec3 e = probesh::SHEvaluate(sh, Vec3{0, 1, 0});
        check(e.x == 0.0f && e.y == 0.0f && e.z == 0.0f, "Zero SH -> SHEvaluate == 0");
    }

    // ====================================================================================
    // SHNormalize degenerate guard: totalWeight <= 0 leaves coeffs untouched
    // ====================================================================================
    {
        probesh::ProbeSH sh{};
        sh.coeff[0][0] = 1.5f; sh.coeff[3][2] = -2.0f;
        probesh::ProbeSH before = sh;
        probesh::SHNormalize(sh, 0.0f);
        check(std::memcmp(&sh, &before, sizeof(probesh::ProbeSH)) == 0,
              "SHNormalize(totalWeight<=0): coeffs untouched (degenerate guard)");
    }

    // ====================================================================================
    // EncodeDispatchGroups: probeCount>0 -> ceil(n/64); probeCount==0 at any zero dim -> 0
    // ====================================================================================
    {
        probesh::ProbeGrid grid; grid.dimX = 2; grid.dimY = 2; grid.dimZ = 2;   // 8 probes
        check(grid.probeCount() == 8 && probesh::EncodeDispatchGroups(grid) == 1,
              "EncodeDispatchGroups: 8 probes -> 1 group");
        probesh::ProbeGrid big; big.dimX = 8; big.dimY = 4; big.dimZ = 8;       // 256 probes
        check(probesh::EncodeDispatchGroups(big) == 4, "EncodeDispatchGroups: 256 probes -> 4 groups");
        probesh::ProbeGrid gx = grid; gx.dimX = 0;
        probesh::ProbeGrid gy = grid; gy.dimY = 0;
        probesh::ProbeGrid gz = grid; gz.dimZ = 0;
        check(gx.probeCount() == 0 && probesh::EncodeDispatchGroups(gx) == 0,
              "Disabled: dimX==0 -> probeCount 0 -> EncodeDispatchGroups 0 (dispatch-0 no-op)");
        check(probesh::EncodeDispatchGroups(gy) == 0 && probesh::EncodeDispatchGroups(gz) == 0,
              "Disabled: dimY==0 / dimZ==0 -> EncodeDispatchGroups 0");
    }

    // ====================================================================================
    // Determinism: encoding the same samples twice -> bit-identical SH
    // ====================================================================================
    {
        const int N = 200;
        auto samples = MakeSamples(N);
        auto encode = [&](probesh::ProbeSH& out) {
            out = probesh::ProbeSH{};
            float total = 0.0f;
            for (int i = 0; i < N; ++i) {
                Vec3 rad{0.5f + 0.1f * (float)(i % 5), 0.3f, 0.7f - 0.05f * (float)(i % 3)};
                probesh::SHEncodeAccumulate(out, rad, samples[i].basis, samples[i].saWeight);
                total += samples[i].saWeight;
            }
            probesh::SHNormalize(out, total);
        };
        probesh::ProbeSH a{}, b{};
        encode(a); encode(b);
        check(std::memcmp(&a, &b, sizeof(probesh::ProbeSH)) == 0,
              "SH encode is deterministic (byte-identical across two runs)");
    }

    if (g_fail == 0) { std::printf("probe_sh_test OK\n"); return 0; }
    std::printf("probe_sh_test: %d failures\n", g_fail);
    return 1;
}
