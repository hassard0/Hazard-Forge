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

    // ================= FC3 — SolveFrictionImpulse / ResolveContactFriction (the cone-clamped solver) =====
    // The SOLVER slice: a dynamic box pressed onto a static box with an incoming TANGENTIAL velocity. The
    // Coulomb cone clamps the tangent impulse to +-mu*jn, applied to BOTH linear AND angular velocity ->
    // the slide is REDUCED (high mu) and the contact drag imparts spin; mu=0 leaves the velocities exactly
    // the CX3 frictionless (normal-only) result AND == convex::ResolveContactPair.
    {
        auto fi = [&](int v) { return (fx)(v * (int)kOne); };
        auto fh = [&](int num, int den) { return (fx)((int64_t)num * (int)kOne / den); };
        const fpx::FxQuat qI{0, 0, 0, kOne};
        const convex::FxBox kUnit{convex::FxVec3{kOne, kOne, kOne}};
        const convex::FxBox kFloor{convex::FxVec3{fi(4), kOne, kOne}};
        // A dynamic body sliding on a static floor: static box A at origin (top face y=+1); dynamic unit box
        // B centered at y=1.5 so it overlaps the floor top by 0.5, descending (vel.y<0 -> approaching normal)
        // AND sliding in +x (the tangential velocity friction must arrest).
        auto makeStaticFloor = [&]() {
            fpx::FxBody A; A.pos = {0, 0, 0}; A.orient = qI; A.invMass = 0; A.flags = 0u; return A;
        };
        auto makeSlider = [&](fx velX) {
            fpx::FxBody B; B.pos = {0, fh(3, 2), 0}; B.orient = qI;
            B.invMass = kOne; B.flags = fpx::kFlagDynamic;
            B.vel = {velX, fi(-3), 0}; B.angVel = {0, 0, 0};
            return B;
        };

        // ---- HIGH-mu: the slide is REDUCED + spin imparted (THE NEW PHYSICS) ----
        {
            const fric::FricSolveConfig cfg{/*restitution*/0, /*mu*/kOne, /*iters*/8};
            fpx::FxBody A = makeStaticFloor();
            fpx::FxBody B = makeSlider(fi(4));   // sliding fast in +x
            const fx preTanSpeed = B.vel.x < 0 ? -B.vel.x : B.vel.x;
            const fpx::FxVec3 preAng = B.angVel;
            fric::ResolveContactFriction(A, kFloor, B, kUnit, cfg);
            const fx postTanSpeed = B.vel.x < 0 ? -B.vel.x : B.vel.x;
            check(postTanSpeed < preTanSpeed, "FC3 high-mu: tangential slide REDUCED");
            const bool spun = (B.angVel.x != preAng.x || B.angVel.y != preAng.y || B.angVel.z != preAng.z);
            check(spun, "FC3 high-mu: contact drag imparted spin (angVel changed)");
        }

        // ---- mu=0: the tangential velocity is UNCHANGED by friction AND the body == ResolveContactPair ----
        {
            const fric::FricSolveConfig cfgFric{/*restitution*/0, /*mu*/0, /*iters*/8};
            const convex::ContactSolveConfig cfgCx{/*restitution*/0, /*iters*/8};
            fpx::FxBody Af = makeStaticFloor(), Bf = makeSlider(fi(4));
            fpx::FxBody Ac = makeStaticFloor(), Bc = makeSlider(fi(4));
            const fx preTanX = Bf.vel.x;
            fric::ResolveContactFriction(Af, kFloor, Bf, kUnit, cfgFric);
            convex::ResolveContactPair(Ac, kFloor, Bc, kUnit, cfgCx);
            // mu=0 -> the friction (tangent) solve applies zero tangent impulse -> tangential x velocity is
            // exactly the pre-solve value (the normal impulse is along y, leaving x untouched).
            check(Bf.vel.x == preTanX, "FC3 mu=0: tangential x velocity UNCHANGED");
            // mu=0 -> the resolved body is BYTE-IDENTICAL to the CX3 frictionless ResolveContactPair.
            check(std::memcmp(&Af, &Ac, sizeof(fpx::FxBody)) == 0, "FC3 mu=0: body A == ResolveContactPair");
            check(std::memcmp(&Bf, &Bc, sizeof(fpx::FxBody)) == 0, "FC3 mu=0: body B == ResolveContactPair");
        }

        // ---- the friction impulse stays within the Coulomb cone (|jt| <= mu*jn per point) ----
        {
            const fx mu = fh(1, 2);   // mu = 0.5
            const fric::FricSolveConfig cfg{/*restitution*/0, mu, /*iters*/8};
            fpx::FxBody A = makeStaticFloor();
            fpx::FxBody B = makeSlider(fi(4));
            // Build the manifold + world inertias exactly as ResolveContactFriction does, then solve and
            // inspect the stored accumulators.
            const convex::SatResult sat = convex::BoxSatStable(A, kFloor, B, kUnit);
            check(sat.overlap, "FC3 cone: the pair overlaps (precondition)");
            fric::FrictionManifold fm = fric::BuildFrictionPoints(A, kFloor, B, kUnit);
            check(fm.count >= 1u, "FC3 cone: at least one friction point");
            const convex::FxVec3 invIaB = convex::FxBoxInvInertiaBody(kFloor, A.invMass);
            const convex::FxVec3 invIbB = convex::FxBoxInvInertiaBody(kUnit, B.invMass);
            const convex::FxMat3 invIaW = convex::WorldInvInertia(A, invIaB);
            const convex::FxMat3 invIbW = convex::WorldInvInertia(B, invIbB);
            fric::SolveFrictionImpulse(A, B, invIaW, invIbW, fm, cfg.restitution, cfg.mu, cfg.iters);
            bool withinCone = true;
            for (uint32_t k = 0; k < fm.count; ++k) {
                const fric::FrictionPoint& fp = fm.pts[k];
                const fx jn = fp.normalImpulse;
                const fx cone = fpx::fxmul(mu, jn);   // +-mu*jn
                if (Absfx(fp.tangentImpulse1) > cone || Absfx(fp.tangentImpulse2) > cone) withinCone = false;
            }
            check(withinCone, "FC3 cone: |jt| <= mu*jn for every friction point");
        }

        // ---- a SEPARATING contact (vn >= 0) applies NO impulse ----
        {
            const fric::FricSolveConfig cfg{/*restitution*/0, /*mu*/kOne, /*iters*/8};
            fpx::FxBody A = makeStaticFloor();
            fpx::FxBody B = makeSlider(fi(4));
            B.vel = {fi(4), fi(3), 0};   // moving AWAY (vel.y > 0 -> separating normal velocity)
            const fpx::FxBody before = B;
            fric::ResolveContactFriction(A, kFloor, B, kUnit, cfg);
            // vn >= 0 at every point -> the normal impulse is skipped -> jn==0 -> the cone collapses to 0 ->
            // NO tangent impulse either -> the body is UNCHANGED.
            check(std::memcmp(&B, &before, sizeof(fpx::FxBody)) == 0,
                  "FC3 separating: vn>=0 applies no impulse (body unchanged)");
        }

        // ---- determinism: two ResolveContactFriction runs byte-identical ----
        {
            const fric::FricSolveConfig cfg{/*restitution*/0, /*mu*/kOne, /*iters*/8};
            fpx::FxBody A1 = makeStaticFloor(), B1 = makeSlider(fi(4));
            fpx::FxBody A2 = makeStaticFloor(), B2 = makeSlider(fi(4));
            fric::ResolveContactFriction(A1, kFloor, B1, kUnit, cfg);
            fric::ResolveContactFriction(A2, kFloor, B2, kUnit, cfg);
            check(std::memcmp(&A1, &A2, sizeof(fpx::FxBody)) == 0 &&
                  std::memcmp(&B1, &B2, sizeof(fpx::FxBody)) == 0,
                  "FC3 determinism: two ResolveContactFriction runs byte-identical");
        }

        // ---- a separated pair (no overlap) -> ResolveContactFriction is a no-op ----
        {
            const fric::FricSolveConfig cfg{/*restitution*/0, /*mu*/kOne, /*iters*/8};
            fpx::FxBody A; A.pos = {fi(-7), 0, fi(6)}; A.orient = qI; A.invMass = 0; A.flags = 0u;
            fpx::FxBody B; B.pos = {fi(-2), 0, fi(6)}; B.orient = qI;
            B.invMass = kOne; B.flags = fpx::kFlagDynamic; B.vel = {fi(4), 0, 0};
            const fpx::FxBody before = B;
            fric::ResolveContactFriction(A, kUnit, B, kUnit, cfg);
            check(std::memcmp(&B, &before, sizeof(fpx::FxBody)) == 0,
                  "FC3 separated: no-overlap pair leaves the body unchanged");
        }
    }

    // ================= FC4 — StepFrictionWorld (the friction-locked world step) =====================
    // The MONEY-PHYSICS beat: a dynamic box released on a TILTED static box GRIPS (high mu) or SLIDES (low
    // mu, a strict inequality — friction is what holds it, not the geometry), and a settling box stack stands
    // at angDamp = kOne (friction physically holds the tower). Plus determinism + statics-frozen.
    {
        auto fi = [&](int v) { return (fx)(v * (int)kOne); };
        const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));   // host-snapped -9.8

        auto makeBody = [&](fx x, fx y, fx z, fpx::FxQuat q, bool dyn) {
            fpx::FxBody b;
            b.pos = {x, y, z};
            b.orient = q;
            b.invMass = dyn ? kOne : 0;
            b.flags   = dyn ? fpx::kFlagDynamic : 0u;
            b.vel = {0, 0, 0};
            b.angVel = {0, 0, 0};
            return b;
        };

        // ---------- THE RAMP scene: a TILTED static box + a dynamic box resting on its inclined top face.
        // 18 deg tilt about Z (tan ~ 0.325): mu=kOne (1.0) GRIPS, mu=kOne/16 (~0.06 < tan) SLIDES.
        const fpx::FxQuat qI{0, 0, 0, kOne};
        const fpx::FxQuat qRamp = fpx::FxQuatNormalize(fpx::FxQuat{0, 0, (fx)10252, (fx)64729});  // 18 deg / Z
        const convex::FxBox kRamp{convex::FxVec3{fi(6), kOne, fi(3)}};       // a wide flat slab (ramp)
        const convex::FxBox kBox{convex::FxVec3{kOne, kOne, kOne}};          // a unit box (the slider)

        // The dynamic box's rest center = the ramp's surface offset (0, rampHalfY + boxHalfY, 0) rotated into
        // the ramp's frame, so the box sits flush on the inclined top face. A tiny extra lift lets it settle.
        auto rampSliderStart = [&]() {
            const convex::FxVec3 localUp{0, kRamp.halfExtents.y + kBox.halfExtents.y, 0};
            convex::FxVec3 off = fpx::FxRotate(qRamp, localUp);
            return convex::FxVec3{off.x, off.y, off.z};
        };
        auto buildRamp = [&](bool dummy) {
            (void)dummy;
            convex::ConvexWorld w;
            w.bodies.push_back(makeBody(0, 0, 0, qRamp, false)); w.boxes.push_back(kRamp);   // static ramp
            const convex::FxVec3 s = rampSliderStart();
            w.bodies.push_back(makeBody(s.x, s.y, s.z, qRamp, true)); w.boxes.push_back(kBox);  // slider
            return w;
        };

        fric::FrictionStepConfig rampCfg;
        rampCfg.gravity     = convex::FxVec3{0, kGravY, 0};
        rampCfg.dt          = kOne / 60;
        rampCfg.solveIters  = 20;
        rampCfg.restitution = 0;
        rampCfg.slop        = kOne / 64;
        rampCfg.beta        = (fx)((int64_t)4 * kOne / 10);    // 0.4
        rampCfg.linDamp     = kOne;
        rampCfg.angDamp     = kOne;
        rampCfg.posIters    = 4;
        const uint32_t kRampTicks = 240u;

        // The down-ramp displacement = how far the slider's center moved along the ramp's local-X (the slope
        // direction). We measure it by projecting (finalPos - startPos) onto the ramp's world X axis.
        auto downRampDisp = [&](fx mu) {
            fric::FrictionStepConfig cfg = rampCfg; cfg.mu = mu;
            convex::ConvexWorld w = buildRamp(true);
            const convex::FxVec3 start = w.bodies[1].pos;
            fric::StepFrictionWorldN(w, cfg, kRampTicks);
            const convex::FxVec3 end = w.bodies[1].pos;
            convex::FxVec3 rampX[3];
            convex::BoxAxes(w.bodies[0], rampX);   // ramp world axes; [0] = local X (down-slope)
            const convex::FxVec3 d = fpx::FxSub(end, start);
            const fx along = convex::FxDot(d, rampX[0]);
            return along < 0 ? -along : along;   // magnitude of down-ramp travel
        };

        const fx gripDisp  = downRampDisp(kOne);          // high mu -> grips
        const fx slideDisp = downRampDisp(kOne / 16);     // low mu -> slides
        // (1) the HIGH-mu box GRIPS: its down-ramp displacement is below a small threshold (it did NOT slide).
        check(gripDisp < fi(1), "FC4 ramp: high-mu box GRIPS (down-ramp displacement small)");
        // (2) the LOW-mu box SLIDES FARTHER than the high-mu box by a clear margin (a strict inequality —
        // friction is what holds the high-mu box).
        check(slideDisp > gripDisp + kOne / 2, "FC4 ramp: low-mu box SLIDES farther (strict inequality)");

        // ---------- THE STACK scene at angDamp = kOne: friction holds the tower (the FC4 headline).
        const convex::FxBox kFloor{convex::FxVec3{fi(8), kOne, fi(8)}};
        const convex::FxBox kSlab{convex::FxVec3{fi(3) / 2, kOne / 2, fi(3) / 2}};   // 3 x 1 x 3
        auto buildStack = [&]() {
            convex::ConvexWorld w;
            w.bodies.push_back(makeBody(0, 0, 0, qI, false)); w.boxes.push_back(kFloor);
            w.bodies.push_back(makeBody(0, fi(1) + kOne * 5 / 8, 0, qI, true)); w.boxes.push_back(kSlab);
            w.bodies.push_back(makeBody(0, fi(2) + kOne * 5 / 8, 0, qI, true)); w.boxes.push_back(kSlab);
            w.bodies.push_back(makeBody(0, fi(3) + kOne * 5 / 8, 0, qI, true)); w.boxes.push_back(kSlab);
            return w;
        };
        fric::FrictionStepConfig stackCfg;
        stackCfg.gravity     = convex::FxVec3{0, kGravY, 0};
        stackCfg.dt          = kOne / 60;
        stackCfg.solveIters  = 20;
        stackCfg.restitution = 0;
        stackCfg.slop        = kOne / 64;
        stackCfg.beta        = (fx)((int64_t)4 * kOne / 10);    // 0.4
        stackCfg.linDamp     = (fx)((int64_t)98 * kOne / 100);  // 0.98 (LINEAR damping only — settles drop)
        stackCfg.angDamp     = kOne;                            // OFF — friction holds the tower (the headline)
        stackCfg.posIters    = 4;
        stackCfg.mu          = kOne;                            // grippy contacts
        const uint32_t kStackTicks = 240u;

        convex::ConvexWorld stack = buildStack();
        fric::StepFrictionWorldN(stack, stackCfg, kStackTicks);
        const fric::StackMeasure ms = fric::MeasureFrictionStack(stack);
        check(ms.maxSpeed < kOne / 2, "FC4 stack: comes to REST at angDamp=kOne (maxSpeed small)");
        check(ms.maxPenetration < kOne / 16, "FC4 stack: no interpenetration at angDamp=kOne");
        const fx y1 = stack.bodies[1].pos.y, y2 = stack.bodies[2].pos.y, y3 = stack.bodies[3].pos.y;
        const fx g01 = y2 - y1, g12 = y3 - y2;
        const fx loBand = fi(1) - kOne / 4, hiBand = fi(1) + kOne / 4;
        const bool ordered = (y1 < y2 && y2 < y3) &&
                             (g01 > loBand && g01 < hiBand) && (g12 > loBand && g12 < hiBand);
        check(ordered, "FC4 stack: stays ORDERED + box-height apart at angDamp=kOne (tower stands)");

        // ---------- determinism: two StepFrictionWorldN runs byte-identical (both scenes).
        {
            convex::ConvexWorld a = buildStack(), b = buildStack();
            fric::StepFrictionWorldN(a, stackCfg, kStackTicks);
            fric::StepFrictionWorldN(b, stackCfg, kStackTicks);
            bool same = true;
            for (size_t i = 0; i < a.bodies.size() && same; ++i)
                if (std::memcmp(&a.bodies[i], &b.bodies[i], sizeof(fpx::FxBody)) != 0) same = false;
            check(same, "FC4 determinism: two StepFrictionWorldN stack runs byte-identical");
        }
        {
            fric::FrictionStepConfig cfg = rampCfg; cfg.mu = kOne;
            convex::ConvexWorld a = buildRamp(true), b = buildRamp(true);
            fric::StepFrictionWorldN(a, cfg, kRampTicks);
            fric::StepFrictionWorldN(b, cfg, kRampTicks);
            bool same = true;
            for (size_t i = 0; i < a.bodies.size() && same; ++i)
                if (std::memcmp(&a.bodies[i], &b.bodies[i], sizeof(fpx::FxBody)) != 0) same = false;
            check(same, "FC4 determinism: two StepFrictionWorldN ramp runs byte-identical");
        }

        // ---------- statics never move: the floor (body 0) and the ramp (body 0) are byte-unchanged.
        {
            convex::ConvexWorld w = buildStack();
            const fpx::FxBody floorBefore = w.bodies[0];
            fric::StepFrictionWorldN(w, stackCfg, kStackTicks);
            check(std::memcmp(&w.bodies[0], &floorBefore, sizeof(fpx::FxBody)) == 0,
                  "FC4 statics: the floor never moves");
        }
        {
            fric::FrictionStepConfig cfg = rampCfg; cfg.mu = kOne;
            convex::ConvexWorld w = buildRamp(true);
            const fpx::FxBody rampBefore = w.bodies[0];
            fric::StepFrictionWorldN(w, cfg, kRampTicks);
            check(std::memcmp(&w.bodies[0], &rampBefore, sizeof(fpx::FxBody)) == 0,
                  "FC4 statics: the ramp never moves");
        }
    }

    // ========================================================================================
    // ============ Slice FC5 — LOCKSTEP + ROLLBACK (the netcode headline) =====================
    // ========================================================================================
    // SimFricTick (apply commands in fixed order + StepFrictionWorld) + RunFricLockstep
    // (authority==replica from inputs alone) + RunFricRollback (corrected==authority AND the mispredict
    // genuinely diverged) + determinism (two runs byte-identical). The friction world IS a convex::ConvexWorld,
    // so FC5 REUSES the frozen CX5 command/snapshot helpers (convex::ConvexCommand/ApplyConvexCommands/
    // SnapshotConvex/RestoreConvex/ConvexBodiesEqual) VERBATIM, swapping StepFrictionWorld for StepConvexWorld.
    // PURE CPU, all orders PINNED -> bit-identical.
    {
        const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
        const fpx::FxQuat qI{0, 0, 0, kOne};
        auto makeBody = [&](fx x, fx y, fx z, bool dyn) {
            fpx::FxBody b;
            b.pos = {x, y, z};
            b.orient = qI;
            b.invMass = dyn ? kOne : 0;
            b.flags   = dyn ? fpx::kFlagDynamic : 0u;
            b.vel = {0, 0, 0};
            b.angVel = {0, 0, 0};
            return b;
        };
        // THE LOCKSTEP SCENE: the FC4 stack — a static floor + 3 dynamic flat-slab boxes — at angDamp=kOne
        // (friction holds the tower). A small command stream knocks it; it re-settles under friction.
        const convex::FxBox kFloor{convex::FxVec3{(fx)(8 * (int)kOne), kOne, (fx)(8 * (int)kOne)}};
        const convex::FxBox kSlab{convex::FxVec3{(fx)(3 * (int)kOne) / 2, kOne / 2, (fx)(3 * (int)kOne) / 2}};
        auto buildStack = [&]() {
            convex::ConvexWorld w;
            w.bodies.push_back(makeBody(0, 0, 0, false)); w.boxes.push_back(kFloor);
            w.bodies.push_back(makeBody(0, (fx)(1 * (int)kOne) + kOne * 5 / 8, 0, true)); w.boxes.push_back(kSlab);
            w.bodies.push_back(makeBody(0, (fx)(2 * (int)kOne) + kOne * 5 / 8, 0, true)); w.boxes.push_back(kSlab);
            w.bodies.push_back(makeBody(0, (fx)(3 * (int)kOne) + kOne * 5 / 8, 0, true)); w.boxes.push_back(kSlab);
            return w;
        };
        fric::FrictionStepConfig cfg;
        cfg.gravity     = convex::FxVec3{0, kGravY, 0};
        cfg.dt          = kOne / 60;
        cfg.solveIters  = 20;
        cfg.restitution = 0;
        cfg.slop        = kOne / 64;
        cfg.beta        = (fx)((int64_t)4 * kOne / 10);    // 0.4
        cfg.linDamp     = (fx)((int64_t)98 * kOne / 100);  // 0.98 (LINEAR damping — settle the drop)
        cfg.angDamp     = kOne;                            // OFF — friction holds the tower (the FC4 headline)
        cfg.posIters    = 4;
        cfg.mu          = kOne;                            // grippy contacts

        // The scripted authoritative command stream: knock the stack with impulse/angVel perturbations at
        // fixed early ticks (impulse = arg·invMass; statics unaffected). Body 3 is the TOP slab.
        const std::vector<convex::ConvexCommand> authStream = {
            convex::ConvexCommand{4u,  convex::kConvexCmdAddImpulse, 3u, FxVec3{(fx)(2 * (int)kOne), 0, 0}},
            convex::ConvexCommand{8u,  convex::kConvexCmdSetAngVel,  2u, FxVec3{0, kOne, 0}},
            convex::ConvexCommand{12u, convex::kConvexCmdAddImpulse, 1u, FxVec3{0, 0, kOne}},
        };
        const uint32_t kLsTicks = 120u;
        const uint32_t kRollbackAt = 20u;

        // ---------- SimFricTick / ApplyConvexCommands: impulse on a dynamic body, no-op on a static body.
        {
            convex::ConvexWorld w = buildStack();
            const fpx::FxBody floorBefore = w.bodies[0];     // body 0 = static floor
            const fpx::FxBody dynBefore   = w.bodies[1];     // body 1 = dynamic slab
            std::vector<convex::ConvexCommand> cmds = {
                convex::ConvexCommand{0u, convex::kConvexCmdAddImpulse, 0u, FxVec3{(fx)(5 * (int)kOne), 0, 0}},
                convex::ConvexCommand{0u, convex::kConvexCmdAddImpulse, 1u, FxVec3{(fx)(3 * (int)kOne), 0, 0}},
            };
            convex::ApplyConvexCommands(w, cmds, 0u);   // (the helper SimFricTick calls, isolated here)
            check(std::memcmp(&w.bodies[0], &floorBefore, sizeof(fpx::FxBody)) == 0,
                  "FC5 ApplyConvexCommands -> impulse on a STATIC body is a no-op (invMass==0)");
            check(w.bodies[1].vel.x == dynBefore.vel.x + fpx::fxmul((fx)(3 * (int)kOne), w.bodies[1].invMass),
                  "FC5 ApplyConvexCommands -> impulse on a DYNAMIC body changes vel by arg*invMass");
        }

        // ---------- SnapshotConvex / RestoreConvex bit-exact round-trip (the rollback primitive).
        {
            convex::ConvexWorld w0 = buildStack();
            for (uint32_t t = 0; t < 30u; ++t) fric::SimFricTick(w0, cfg, authStream, t);
            const convex::ConvexSnapshot snap = convex::SnapshotConvex(w0, 30u);
            convex::ConvexWorld w = w0;
            fric::SimFricTick(w, cfg, authStream, 30u);   // diverge
            convex::RestoreConvex(w, snap);
            check(convex::ConvexBodiesEqual(w.bodies, w0.bodies),
                  "FC5 RestoreConvex(SnapshotConvex) round-trips the friction world byte-for-byte");
        }

        // ---------- RunFricLockstep: authority == replica BIT-IDENTICAL (inputs only).
        {
            convex::ConvexWorld w0 = buildStack();
            bool identical = false;
            convex::ConvexWorld authority = fric::RunFricLockstep(w0, cfg, authStream, kLsTicks, &identical);
            check(identical, "FC5 RunFricLockstep: authority == replica BIT-IDENTICAL (inputs only)");
            // The converged stack is still a coherent ordered tower (knocked + re-settled under friction).
            check(authority.bodies[1].pos.y < authority.bodies[2].pos.y &&
                  authority.bodies[2].pos.y < authority.bodies[3].pos.y,
                  "FC5 lockstep converged stack -> boxes still ordered ascending in y (re-settled)");
            const fric::StackMeasure ms = fric::MeasureFrictionStack(authority);
            check(ms.maxSpeed < kOne, "FC5 lockstep converged stack -> at rest after the perturbation");
        }

        // ---------- determinism: two RunFricLockstep runs byte-identical.
        {
            convex::ConvexWorld w0 = buildStack();
            convex::ConvexWorld a = fric::RunFricLockstep(w0, cfg, authStream, kLsTicks);
            convex::ConvexWorld b = fric::RunFricLockstep(w0, cfg, authStream, kLsTicks);
            check(convex::ConvexBodiesEqual(a.bodies, b.bodies),
                  "FC5 RunFricLockstep determinism: two runs BYTE-IDENTICAL");
        }

        // ---------- RunFricRollback: corrected==authority BIT-EXACT AND the mispredict genuinely diverged.
        {
            convex::ConvexWorld w0 = buildStack();
            // The MISPREDICTED stream: the auth stream + a WRONG strong impulse at rollbackAt on the top slab.
            std::vector<convex::ConvexCommand> mispredictStream = authStream;
            mispredictStream.push_back(convex::ConvexCommand{kRollbackAt, convex::kConvexCmdAddImpulse, 3u,
                                                             FxVec3{(fx)(30 * (int)kOne), 0, 0}});
            bool corrected = false, diverged = false;
            convex::ConvexWorld rolledBack =
                fric::RunFricRollback(w0, cfg, authStream, mispredictStream, kLsTicks, kRollbackAt,
                                      &corrected, &diverged);
            check(diverged, "FC5 RunFricRollback: the mispredicted intermediate genuinely DIVERGED (real)");
            check(corrected, "FC5 RunFricRollback: corrected == authority BIT-EXACT");
            // The corrected world equals the plain authority lockstep run byte-for-byte.
            convex::ConvexWorld authority = fric::RunFricLockstep(w0, cfg, authStream, kLsTicks);
            check(convex::ConvexBodiesEqual(rolledBack.bodies, authority.bodies),
                  "FC5 RunFricRollback: rolledBack == RunFricLockstep(authStream) byte-for-byte");
        }
    }

    // ============ Slice FC6 — THE LIT 3D RENDER CAPSTONE (provenance: the render is a pure function) =========
    // FrictionToRenderInstances of a SETTLED friction RAMP world (a tilted static ramp + a dynamic box gripped on
    // it) -> the instance count == the body count; the dynamic/static split is correct; each instance's
    // translation == convex::ConvexBoxShapeTransform / fpx::FxBodyTransform translation (the provenance proof);
    // two calls are byte-identical (the render is a pure function of the bit-exact world). Pure CPU, render-only
    // FLOAT delegate to the frozen convex helper.
    {
        auto fi = [&](int v) { return (fx)(v * (int)kOne); };
        const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
        auto makeBody = [&](fx x, fx y, fx z, fpx::FxQuat q, bool dyn) {
            fpx::FxBody b;
            b.pos = {x, y, z};
            b.orient = q;
            b.invMass = dyn ? kOne : 0;
            b.flags   = dyn ? fpx::kFlagDynamic : 0u;
            b.vel = {0, 0, 0};
            b.angVel = {0, 0, 0};
            return b;
        };
        // The FC4 ramp scene (== the --fric-render-shot scene): an 18-degree tilted static ramp + a dynamic box
        // placed on its top face, settled K ticks at mu=kOne so friction GRIPS the box on the incline.
        const fpx::FxQuat qRamp = fpx::FxQuatNormalize(fpx::FxQuat{0, 0, (fx)10252, (fx)64729}); // 18 deg / Z
        const convex::FxBox kRamp{convex::FxVec3{fi(6), kOne, fi(3)}};
        const convex::FxBox kSlider{convex::FxVec3{kOne, kOne, kOne}};
        fric::FrictionStepConfig cfg;
        cfg.gravity = FxVec3{0, kGravY, 0};
        cfg.dt = kOne / 60; cfg.solveIters = 20; cfg.restitution = 0;
        cfg.slop = kOne / 64; cfg.beta = (fx)((int64_t)4 * kOne / 10);
        cfg.linDamp = kOne; cfg.angDamp = kOne; cfg.posIters = 4; cfg.mu = kOne;

        convex::ConvexWorld world;
        world.bodies.push_back(makeBody(0, 0, 0, qRamp, false)); world.boxes.push_back(kRamp);   // static ramp
        const convex::FxVec3 off =
            fpx::FxRotate(qRamp, convex::FxVec3{0, kRamp.halfExtents.y + kSlider.halfExtents.y, 0});
        world.bodies.push_back(makeBody(off.x, off.y, off.z, qRamp, true)); world.boxes.push_back(kSlider);
        const uint32_t kBodies = (uint32_t)world.bodies.size();   // 2: 1 static ramp + 1 dynamic box
        fric::StepFrictionWorldN(world, cfg, 240u);

        const convex::ConvexRenderInstances ri = fric::FrictionToRenderInstances(world);
        const uint32_t kFloorN = (uint32_t)ri.floor.size();
        const uint32_t kBoxN   = (uint32_t)ri.boxes.size();

        // (1) the instance count == the body count.
        check(kFloorN + kBoxN == kBodies,
              "FC6 FrictionToRenderInstances: instance count == body count");
        // (2) the dynamic/static split is correct (1 static ramp -> floor, 1 dynamic box -> boxes).
        check(kFloorN == 1u, "FC6 split: the static ramp -> floor (1)");
        check(kBoxN == 1u, "FC6 split: the dynamic box -> boxes (1)");

        // (3) provenance — each instance's translation == the convex::ConvexBoxShapeTransform translation (the
        // delegate produces EXACTLY the frozen CX6 matrices; the translation is FxToFloat(pos) of each body).
        {
            const math::Mat4 floorM = convex::ConvexBoxShapeTransform(world.bodies[0], world.boxes[0]);
            const math::Mat4 boxM   = convex::ConvexBoxShapeTransform(world.bodies[1], world.boxes[1]);
            // column-major Mat4: the translation is m[12..14].
            check(std::memcmp(ri.floor[0].m, floorM.m, sizeof(float) * 16) == 0,
                  "FC6 provenance: floor instance == ConvexBoxShapeTransform(static ramp)");
            check(std::memcmp(ri.boxes[0].m, boxM.m, sizeof(float) * 16) == 0,
                  "FC6 provenance: box instance == ConvexBoxShapeTransform(dynamic box)");
            // the translation column equals FxToFloat(pos) of each body (the FxBodyTransform translation).
            check(ri.boxes[0].m[12] == fpx::FxToFloat(world.bodies[1].pos.x) &&
                  ri.boxes[0].m[13] == fpx::FxToFloat(world.bodies[1].pos.y) &&
                  ri.boxes[0].m[14] == fpx::FxToFloat(world.bodies[1].pos.z),
                  "FC6 provenance: box translation == FxToFloat(body.pos)");
        }

        // (4) the dynamic box GRIPPED on the ramp — it rests ABOVE the ramp top (did not slide off / fall
        // through). The settled box centre stays well above the floor plane (friction held it on the incline).
        check(world.bodies[1].pos.y > fi(1),
              "FC6 money-shot: the dynamic box is GRIPPED on the ramp (rests above the ground, not slid off)");

        // (5) the render is a PURE FUNCTION of the bit-exact world — two calls byte-identical (the provenance
        // contract the showcase asserts).
        {
            const convex::ConvexRenderInstances ri2 = fric::FrictionToRenderInstances(world);
            bool identical = (ri2.floor.size() == ri.floor.size()) && (ri2.boxes.size() == ri.boxes.size());
            for (size_t k = 0; k < ri.floor.size() && identical; ++k)
                if (std::memcmp(ri.floor[k].m, ri2.floor[k].m, sizeof(float) * 16) != 0) identical = false;
            for (size_t k = 0; k < ri.boxes.size() && identical; ++k)
                if (std::memcmp(ri.boxes[k].m, ri2.boxes[k].m, sizeof(float) * 16) != 0) identical = false;
            check(identical, "FC6 FrictionToRenderInstances: two calls BYTE-IDENTICAL (pure function)");
        }
    }

    if (g_fail == 0) std::printf("fric_test: ALL PASS\n");
    else std::printf("fric_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
