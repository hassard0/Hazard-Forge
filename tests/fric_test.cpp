// Slice FC1 — Deterministic Contact Friction: THE TANGENT BASIS (the BEACHHEAD of FLAGSHIP #20:
// DETERMINISTIC TANGENTIAL CONTACT FRICTION, hf::sim::fric). The integer core (engine/sim/fric.h) that the
// GPU shaders/fric_basis.comp.hlsl copies VERBATIM + proves bit-identical. Pure CPU (header-only, hf_core),
// ASan-eligible. fric.h #includes sim/convex.h read-only (which transitively gives fpx).
//
// What this test PINS (the contracts the GPU fric_basis.comp + the GPU==CPU proof build on):
//   * MakeTangentBasis(+z): the two tangents lie in the xy-plane (z component ~0), are mutually orthogonal,
//     each orthogonal to z, and unit (within the integer epsilon).
//   * MakeTangentBasis(oblique): an arbitrary tilted unit normal gives an orthonormal (t1,t2): the three dot
//     products |n·t1|,|n·t2|,|t1·t2| are ~0 and FxLength(t1),FxLength(t2) ~ kOne (within epsilon).
//   * The least-aligned-axis choice is the FIXED argmin: for n ~ +x the chosen reference cardinal axis is
//     NOT x (e0); for n ~ +z it is NOT z (e2). Lowest-index tie-break.
//   * MeasureBasis: deterministic orthonormality summary over a set of normals (maxDotErr / lenErr bounded).
//   * Determinism: two runs of MakeTangentBasis / MeasureBasis byte-identical.
//
// Pure C++ (hf_core), ASan-eligible like the other sim/render-math tests.
#include "sim/fric.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace fric = hf::sim::fric;
namespace convex = hf::sim::convex;
namespace fpx = hf::sim::fpx;
using convex::fx;
using convex::kOne;
using convex::kFrac;
using convex::FxVec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

static fx Absfx(fx v) { return v < 0 ? -v : v; }

// A small integer epsilon for the orthonormality drift (fixed-point FxISqrt/fxdiv truncation). ~1/256 unit.
static const fx kEps = kOne / 256;   // ~256 Q16.16 ticks

int main() {
    HF_TEST_MAIN_INIT();

    // ================= MakeTangentBasis(+z): tangents in the xy-plane =================
    {
        const FxVec3 n{0, 0, kOne};
        const fric::TangentBasis b = fric::MakeTangentBasis(n);
        // Each tangent orthogonal to z (its z component dotted with n is the z component itself ~0).
        check(Absfx(convex::FxDot(n, b.t1)) < kEps, "tangent +z: n . t1 ~ 0");
        check(Absfx(convex::FxDot(n, b.t2)) < kEps, "tangent +z: n . t2 ~ 0");
        // Mutually orthogonal.
        check(Absfx(convex::FxDot(b.t1, b.t2)) < kEps, "tangent +z: t1 . t2 ~ 0");
        // Each unit length.
        check(Absfx(fpx::FxLength(b.t1) - kOne) < kEps, "tangent +z: |t1| ~ 1");
        check(Absfx(fpx::FxLength(b.t2) - kOne) < kEps, "tangent +z: |t2| ~ 1");
        // Both lie in the xy-plane (z component ~0) since n is +z.
        check(Absfx(b.t1.z) < kEps, "tangent +z: t1.z ~ 0 (in xy-plane)");
        check(Absfx(b.t2.z) < kEps, "tangent +z: t2.z ~ 0 (in xy-plane)");
    }

    // ================= MakeTangentBasis(oblique): an orthonormal basis =================
    {
        // An oblique unit normal = FxNormalize of a small integer vector (1,2,3).
        const FxVec3 raw{kOne, 2 * kOne, 3 * kOne};
        const FxVec3 n = fpx::FxNormalize(raw);
        const fric::TangentBasis b = fric::MakeTangentBasis(n);
        check(Absfx(convex::FxDot(n, b.t1)) < kEps, "tangent oblique: n . t1 ~ 0");
        check(Absfx(convex::FxDot(n, b.t2)) < kEps, "tangent oblique: n . t2 ~ 0");
        check(Absfx(convex::FxDot(b.t1, b.t2)) < kEps, "tangent oblique: t1 . t2 ~ 0");
        check(Absfx(fpx::FxLength(b.t1) - kOne) < kEps, "tangent oblique: |t1| ~ 1");
        check(Absfx(fpx::FxLength(b.t2) - kOne) < kEps, "tangent oblique: |t2| ~ 1");
    }

    // ================= the least-aligned-axis argmin is the fixed choice =================
    {
        // For n == +x (== e0), the chosen reference cardinal axis must NOT be e0 (|n . e0| == 1 is the MOST
        // aligned). The fixed lowest-index tie-break picks e1 (the first of the two equally-least-aligned).
        const FxVec3 nx{kOne, 0, 0};
        check(fric::LeastAlignedAxis(nx) == 1u, "argmin: n=+x -> ref axis e1 (NOT x)");
        // For n == +z (== e2), the chosen axis must NOT be e2; lowest-index tie-break picks e0.
        const FxVec3 nz{0, 0, kOne};
        check(fric::LeastAlignedAxis(nz) == 0u, "argmin: n=+z -> ref axis e0 (NOT z)");
        // For n == +y (== e1), lowest-index tie-break picks e0.
        const FxVec3 ny{0, kOne, 0};
        check(fric::LeastAlignedAxis(ny) == 0u, "argmin: n=+y -> ref axis e0 (NOT y)");
        // For a normal mostly along x (but not exactly), the least-aligned is still NOT x.
        const FxVec3 nxish = fpx::FxNormalize(FxVec3{10 * kOne, kOne, 0});
        check(fric::LeastAlignedAxis(nxish) != 0u, "argmin: n~+x -> ref axis != x");
    }

    // ================= the axis-aligned normals are all degeneracy-safe =================
    {
        const FxVec3 axisN[6] = {
            {kOne, 0, 0}, {-kOne, 0, 0}, {0, kOne, 0}, {0, -kOne, 0}, {0, 0, kOne}, {0, 0, -kOne},
        };
        for (int i = 0; i < 6; ++i) {
            const fric::TangentBasis b = fric::MakeTangentBasis(axisN[i]);
            const bool ortho = Absfx(convex::FxDot(axisN[i], b.t1)) < kEps &&
                               Absfx(convex::FxDot(axisN[i], b.t2)) < kEps &&
                               Absfx(convex::FxDot(b.t1, b.t2)) < kEps;
            const bool unit = Absfx(fpx::FxLength(b.t1) - kOne) < kEps &&
                              Absfx(fpx::FxLength(b.t2) - kOne) < kEps;
            check(ortho, "degeneracy: axis-aligned normal orthonormal (dots ~0)");
            check(unit, "degeneracy: axis-aligned normal orthonormal (lengths ~1)");
        }
    }

    // ================= MeasureBasis: deterministic orthonormality summary =================
    {
        std::vector<FxVec3> normals;
        normals.push_back(FxVec3{kOne, 0, 0});
        normals.push_back(FxVec3{0, kOne, 0});
        normals.push_back(FxVec3{0, 0, kOne});
        normals.push_back(fpx::FxNormalize(FxVec3{kOne, 2 * kOne, 3 * kOne}));
        normals.push_back(fpx::FxNormalize(FxVec3{-2 * kOne, kOne, kOne}));
        const fric::BasisMeasure m = fric::MeasureBasis(normals);
        check(m.normals == (uint32_t)normals.size(), "measure: counts every normal");
        check(m.maxDotErr < kEps, "measure: maxDotErr within epsilon");
        check(Absfx(m.minLen - kOne) < kEps, "measure: minLen ~ 1");
        check(Absfx(m.maxLen - kOne) < kEps, "measure: maxLen ~ 1");
        // determinism: a second measure is byte-identical.
        const fric::BasisMeasure m2 = fric::MeasureBasis(normals);
        check(std::memcmp(&m, &m2, sizeof(fric::BasisMeasure)) == 0, "measure: two runs byte-identical");
    }

    // ================= determinism: two MakeTangentBasis runs byte-identical =================
    {
        const FxVec3 n = fpx::FxNormalize(FxVec3{3 * kOne, -kOne, 2 * kOne});
        const fric::TangentBasis a = fric::MakeTangentBasis(n);
        const fric::TangentBasis b = fric::MakeTangentBasis(n);
        check(std::memcmp(&a, &b, sizeof(fric::TangentBasis)) == 0,
              "determinism: two MakeTangentBasis runs byte-identical");
    }

    if (g_fail == 0) std::printf("fric_test: ALL PASS\n");
    else std::printf("fric_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
