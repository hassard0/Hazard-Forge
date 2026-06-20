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

    // ================= FC2 — BuildFrictionPoints over the frozen CX2 manifold =================
    // Scene helpers (the CX2 box-pair conventions): a body at a pos+orient, a unit box, the fixed-point
    // helpers fi (integer->Q16.16) / fh (a fraction).
    {
        auto fi = [&](int v) { return (fx)(v * (int)kOne); };
        auto fh = [&](int num, int den) { return (fx)((int64_t)num * (int)kOne / den); };
        auto bodyAt = [&](fx x, fx y, fx z, fpx::FxQuat q) {
            fpx::FxBody b; b.pos = {x, y, z}; b.orient = q; return b;
        };
        const fpx::FxQuat qI{0, 0, 0, kOne};
        const fpx::FxQuat qX = fpx::FxQuatNormalize(fpx::FxQuat{(fx)25080, 0, 0, (fx)60547}); // 45° about X
        const fpx::FxQuat qZ = fpx::FxQuatNormalize(fpx::FxQuat{0, 0, (fx)25080, (fx)60547}); // 45° about Z
        const convex::FxBox kUnit{convex::FxVec3{kOne, kOne, kOne}};
        const convex::FxBox kBig{convex::FxVec3{fi(4), fi(4), fi(4)}};

        // ---- a DEEP FACE-FACE pair: count matches BuildManifold, basis orthonormal to the A->B normal,
        // all accumulators zero. (A big box with a unit box pushed deep inside on z -> a 4-point face patch.)
        {
            const fpx::FxBody A = bodyAt(fi(8), 0, fi(0), qI);
            const fpx::FxBody B = bodyAt(fi(8), 0, fh(3, 2), qI);
            const fric::FrictionManifold fm2 = fric::BuildFrictionPoints(A, kBig, B, kUnit);
            const convex::SatResult sat = convex::BoxSatStable(A, kBig, B, kUnit);
            const convex::ContactManifold m = convex::BuildManifold(A, kBig, B, kUnit, sat);
            check(fm2.count == m.count, "FC2 face: count matches BuildManifold");
            check(fm2.count >= 1u, "FC2 face: at least one contact point");
            // The A->B normal (the SolveManifoldImpulse rule applied to the manifold normal).
            convex::FxVec3 nAB = m.normal;
            if (convex::FxDot(nAB, fpx::FxSub(B.pos, A.pos)) < 0) nAB = convex::FxVec3{-nAB.x, -nAB.y, -nAB.z};
            for (uint32_t k = 0; k < fm2.count; ++k) {
                const fric::FrictionPoint& fp = fm2.pts[k];
                // normal stored == the A->B normal.
                check(std::memcmp(&fp.normal, &nAB, sizeof(convex::FxVec3)) == 0,
                      "FC2 face: stored normal == A->B normal");
                // (t1,t2) orthonormal to the normal.
                check(Absfx(convex::FxDot(fp.normal, fp.t1)) < kEps, "FC2 face: n . t1 ~ 0");
                check(Absfx(convex::FxDot(fp.normal, fp.t2)) < kEps, "FC2 face: n . t2 ~ 0");
                check(Absfx(convex::FxDot(fp.t1, fp.t2)) < kEps, "FC2 face: t1 . t2 ~ 0");
                check(Absfx(fpx::FxLength(fp.t1) - kOne) < kEps, "FC2 face: |t1| ~ 1");
                check(Absfx(fpx::FxLength(fp.t2) - kOne) < kEps, "FC2 face: |t2| ~ 1");
                // accumulators zeroed at build.
                check(fp.normalImpulse == 0, "FC2 face: normalImpulse zero");
                check(fp.tangentImpulse1 == 0, "FC2 face: tangentImpulse1 zero");
                check(fp.tangentImpulse2 == 0, "FC2 face: tangentImpulse2 zero");
                // the basis == the FC1 MakeTangentBasis of the stored normal (the exact reuse contract).
                const fric::TangentBasis tb = fric::MakeTangentBasis(fp.normal);
                check(std::memcmp(&tb.t1, &fp.t1, sizeof(convex::FxVec3)) == 0, "FC2 face: t1 == MakeTangentBasis");
                check(std::memcmp(&tb.t2, &fp.t2, sizeof(convex::FxVec3)) == 0, "FC2 face: t2 == MakeTangentBasis");
            }
        }

        // ---- a SEPARATED pair: count 0.
        {
            const fpx::FxBody A = bodyAt(fi(-7), 0, fi(6), qI);
            const fpx::FxBody B = bodyAt(fi(-2), 0, fi(6), qI);
            const fric::FrictionManifold fm = fric::BuildFrictionPoints(A, kUnit, B, kUnit);
            check(fm.count == 0u, "FC2 separated: count 0");
        }

        // ---- an EDGE-EDGE pair: count 1 with a valid orthonormal basis, the normal A->B.
        {
            const fpx::FxBody A = bodyAt(0, 0, fi(2), qX);
            const fpx::FxBody B = bodyAt(fi(1), fi(1), fi(3), qZ);
            const convex::SatResult sat = convex::BoxSatStable(A, kUnit, B, kUnit);
            const fric::FrictionManifold fm = fric::BuildFrictionPoints(A, kUnit, B, kUnit);
            check(sat.overlap, "FC2 edge: the pair overlaps (precondition)");
            check(fm.count == convex::BuildManifold(A, kUnit, B, kUnit, sat).count,
                  "FC2 edge: count matches BuildManifold");
            for (uint32_t k = 0; k < fm.count; ++k) {
                const fric::FrictionPoint& fp = fm.pts[k];
                check(Absfx(convex::FxDot(fp.normal, fp.t1)) < kEps, "FC2 edge: n . t1 ~ 0");
                check(Absfx(convex::FxDot(fp.normal, fp.t2)) < kEps, "FC2 edge: n . t2 ~ 0");
                check(Absfx(convex::FxDot(fp.t1, fp.t2)) < kEps, "FC2 edge: t1 . t2 ~ 0");
                check(Absfx(fpx::FxLength(fp.t1) - kOne) < kEps, "FC2 edge: |t1| ~ 1");
                check(Absfx(fpx::FxLength(fp.t2) - kOne) < kEps, "FC2 edge: |t2| ~ 1");
                check(fp.normalImpulse == 0 && fp.tangentImpulse1 == 0 && fp.tangentImpulse2 == 0,
                      "FC2 edge: accumulators zero");
            }
        }

        // ---- the normal points A->B for every overlapping pair (FxDot(normal, B.pos - A.pos) >= 0).
        {
            std::vector<convex::SatPair> pairs;
            const convex::FxBox U = kUnit;
            pairs.push_back({bodyAt(fi(5), 0, fi(6), qI), kBig, bodyAt(fi(6), 0, fi(6), qI), U});
            pairs.push_back({bodyAt(fi(-7), 0, fi(2), qI), U, bodyAt(fi(-5), 0, fi(2), qI), U});
            pairs.push_back({bodyAt(0, 0, fi(2), qX), U, bodyAt(fi(1), fi(1), fi(3), qZ), U});
            bool allAtoB = true;
            for (const convex::SatPair& p : pairs) {
                const fric::FrictionManifold fm =
                    fric::BuildFrictionPoints(p.bodyA, p.boxA, p.bodyB, p.boxB);
                const convex::FxVec3 ab = fpx::FxSub(p.bodyB.pos, p.bodyA.pos);
                for (uint32_t k = 0; k < fm.count; ++k)
                    if (convex::FxDot(fm.pts[k].normal, ab) < 0) allAtoB = false;
            }
            check(allAtoB, "FC2: every contact normal points A->B");
        }

        // ---- determinism: two BuildFrictionPoints runs byte-identical.
        {
            const fpx::FxBody A = bodyAt(fi(8), 0, fi(0), qI);
            const fpx::FxBody B = bodyAt(fi(8), 0, fh(3, 2), qI);
            const fric::FrictionManifold a = fric::BuildFrictionPoints(A, kBig, B, kUnit);
            const fric::FrictionManifold b = fric::BuildFrictionPoints(A, kBig, B, kUnit);
            check(std::memcmp(&a, &b, sizeof(fric::FrictionManifold)) == 0,
                  "FC2 determinism: two BuildFrictionPoints runs byte-identical");
        }

        // ---- MeasureFrictionPoints: the deterministic summary over a set of pairs.
        {
            std::vector<convex::SatPair> pairs;
            const convex::FxBox U = kUnit;
            pairs.push_back({bodyAt(fi(-7), 0, fi(6), qI), U, bodyAt(fi(-2), 0, fi(6), qI), U}); // separated
            pairs.push_back({bodyAt(fi(5), 0, fi(6), qI), kBig, bodyAt(fi(6), 0, fi(6), qI), U}); // face
            pairs.push_back({bodyAt(0, 0, fi(2), qX), U, bodyAt(fi(1), fi(1), fi(3), qZ), U});    // edge
            const fric::FrictionPointMeasure pm = fric::MeasureFrictionPoints(pairs);
            check(pm.pairs == (uint32_t)pairs.size(), "FC2 measure: counts every pair");
            check(pm.pairsWithContact >= 2u, "FC2 measure: the two overlapping pairs report contact");
            check(pm.totalPoints >= 2u, "FC2 measure: total points accumulated");
            check(pm.maxDotErr < kEps, "FC2 measure: maxDotErr within epsilon");
            const fric::FrictionPointMeasure pm2 = fric::MeasureFrictionPoints(pairs);
            check(std::memcmp(&pm, &pm2, sizeof(fric::FrictionPointMeasure)) == 0,
                  "FC2 measure: two runs byte-identical");
        }
    }

    if (g_fail == 0) std::printf("fric_test: ALL PASS\n");
    else std::printf("fric_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
