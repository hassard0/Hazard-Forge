#pragma once
// Slice FC1 — Deterministic Contact Friction: THE TANGENT BASIS (the integer BEACHHEAD of FLAGSHIP #20:
// DETERMINISTIC TANGENTIAL CONTACT FRICTION — the Coulomb friction cone on the convex box-box angular-impulse
// contacts, hf::sim::fric). Friction is the most numerically-sensitive part of a contact solver and the
// corner where mainstream float solvers diverge first machine-to-machine; locking it deterministically
// completes the convex arc. FC1 builds the primitive every later friction slice needs: a deterministic,
// bit-exact orthonormal TANGENT BASIS (t1, t2) at a contact, derived from the manifold normal n. INTEGER
// bit-exact, Q16.16.
//
// Header-only, namespace hf::sim::fric, #include "sim/convex.h" READ-ONLY (do NOT modify convex.h — fric.h is
// a NEW additive sibling). convex.h transitively gives the fpx Q16.16 toolbox: FxVec3/fx/kOne/kFrac, FxDot
// (convex.h), FxCross (convex.h), and from fpx.h FxSub/FxScale/FxNormalize/FxLength/FxISqrt.
//
// THE int64 REALITY (the FPX3/CX1 lesson, the honest proof-strength call): FxNormalize/FxISqrt + the
// FxDot/FxCross Q16.16 products are int64 (world-scale products overflow int32). DXC compiles int64 (the
// Vulkan path); glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL. So the GPU shader
// shaders/fric_basis.comp.hlsl is VULKAN-SPIR-V-ONLY (in the Vulkan compile list, NOT in the Metal hf_gen_msl
// list); the Metal --fric-basis runs the CPU MakeTangentBasis -> byte-identical to the Vulkan GPU result BY
// CONSTRUCTION (the convex_sat.comp / fpx_solve.comp convention), while the Vulkan side carries the GPU==CPU
// memcmp proof. fric_basis.comp copies MakeTangentBasis's body VERBATIM, so the GPU exercises the EXACT
// integer ops -> a divergence is exactly what the host GPU==CPU memcmp catches.
//
// THE FIXED INTEGER GRAM-SCHMIDT (the determinism crux): for a unit normal n, of the three cardinal axes
// e0=(1,0,0), e1=(0,1,0), e2=(0,0,1) pick the one LEAST aligned with n (smallest |FxDot(n, e_i)|), FIXED
// tie-break = lowest index i wins (so the projection below never collapses — the degeneracy guard). Then
// t1 = FxNormalize(e_min - FxDot(e_min, n)*n) (project e_min off n, normalize), and t2 = FxCross(n, t1) (n,t1
// orthonormal -> n x t1 is already unit, no second normalize). (n, t1, t2) is a right-handed frame. FIXED
// order, integer -> bit-identical CPU<->Vulkan<->Metal.

#include <cstdint>
#include <vector>

#include "sim/convex.h"   // read-only: FxDot/FxCross (convex.h) + the fpx toolbox (FxVec3/fx/kOne/kFrac,
                          // FxSub/FxScale/FxNormalize/FxLength via fpx). convex.h #includes fpx.h read-only.

namespace hf::sim {
namespace fric {

// Pull the shared Q16.16 scalar + helpers into this namespace (re-use, do NOT redefine).
using convex::fx;
using convex::kOne;
using convex::kFrac;
using convex::FxVec3;
using convex::FxDot;
using convex::FxCross;

// ----- TangentBasis: the two orthonormal tangents spanning the contact tangent plane --------------------
// (t1, t2) span the plane perpendicular to the caller's contact normal n. n is the caller's input, NOT
// stored (FC2+ feeds the manifold normal). std430-packable as 6 x int32 (24 bytes) — the GPU TangentBasis
// mirror the fric_basis.comp memcmp compares against.
struct TangentBasis {
    FxVec3 t1;
    FxVec3 t2;
};

// ----- LeastAlignedAxis(n): the FIXED argmin cardinal-axis choice (the degeneracy guard) ----------------
// Of e0/e1/e2 (each kOne on one component, exposed via n's components since e_i . n == n.axis_i), return the
// index of the axis LEAST aligned with n (smallest |n.axis|). FIXED tie-break: the LOWEST index wins (strict
// < on the running minimum). Because |n.axis| == |FxDot(n, e_axis)|, no products are needed — a plain abs
// compare on n's components. Identical CPU/HLSL (a branch on three abs values).
inline uint32_t LeastAlignedAxis(const FxVec3& n) {
    auto absfx = [](fx v) { return v < 0 ? -v : v; };
    const fx a0 = absfx(n.x), a1 = absfx(n.y), a2 = absfx(n.z);
    uint32_t best = 0;
    fx bestVal = a0;
    if (a1 < bestVal) { bestVal = a1; best = 1; }   // strict-< -> lowest index keeps a tie
    if (a2 < bestVal) { bestVal = a2; best = 2; }
    return best;
}

// ----- CardinalAxis(i): the i-th cardinal axis e_i (kOne on component i, 0 elsewhere) -------------------
inline FxVec3 CardinalAxis(uint32_t i) {
    return FxVec3{(i == 0) ? kOne : (fx)0, (i == 1) ? kOne : (fx)0, (i == 2) ? kOne : (fx)0};
}

// ----- MakeTangentBasis(n): the fixed integer Gram-Schmidt orthonormal tangent basis at a contact -------
// The four PINNED steps (every ordering decision fixed):
//   (1) pick the LEAST-aligned cardinal axis e_min (LeastAlignedAxis(n) — lowest-index tie-break);
//   (2) project e_min off n: r = e_min - FxDot(e_min, n)*n (remove the n component);
//   (3) t1 = FxNormalize(r) (the int64 FxISqrt; r is comfortably non-zero because e_min is least-aligned);
//   (4) t2 = FxCross(n, t1) (n, t1 orthonormal -> n x t1 already unit, no second normalize).
// Pure integer, FIXED order -> bit-identical CPU<->Vulkan<->Metal. The shader copies THIS body VERBATIM.
inline TangentBasis MakeTangentBasis(const FxVec3& n) {
    const uint32_t mi = LeastAlignedAxis(n);
    const FxVec3 e = CardinalAxis(mi);
    const fx eDotN = FxDot(e, n);                          // the n-component of e
    // r = e - (e . n) * n  (project e off n).
    const FxVec3 r = FxVec3{e.x - fpx::fxmul(eDotN, n.x),
                            e.y - fpx::fxmul(eDotN, n.y),
                            e.z - fpx::fxmul(eDotN, n.z)};
    const FxVec3 t1 = fpx::FxNormalize(r);                 // unit tangent (int64 FxISqrt/fxdiv)
    const FxVec3 t2 = FxCross(n, t1);                      // already unit (n, t1 orthonormal)
    return TangentBasis{t1, t2};
}

// ----- BasisMeasure: the deterministic orthonormality summary over a set of contact normals -------------
// normals    = total normals measured; maxDotErr = the max over all normals of |n.t1|, |n.t2|, |t1.t2| (the
// orthogonality residual — 0 for a perfect basis, a small integer for the fixed-point drift); minLen/maxLen
// = the min/max FxLength(t1) AND FxLength(t2) over the set (both ~ kOne for an orthonormal basis). Pure
// integer, fixed order -> deterministic. The showcase prints + asserts maxDotErr / |len - kOne| below eps.
struct BasisMeasure {
    uint32_t normals   = 0;
    fx       maxDotErr = 0;     // max |n.t1| / |n.t2| / |t1.t2|
    fx       minLen    = 0;     // min FxLength over {t1, t2} of all normals (0 if no normals)
    fx       maxLen    = 0;     // max FxLength over {t1, t2} of all normals (0 if no normals)
};

// MeasureBasis(normals): MakeTangentBasis over each normal, accumulate the deterministic orthonormality
// summary. Pure integer, fixed order.
inline BasisMeasure MeasureBasis(const std::vector<FxVec3>& normals) {
    auto absfx = [](fx v) { return v < 0 ? -v : v; };
    BasisMeasure m;
    m.normals = (uint32_t)normals.size();
    bool first = true;
    for (const FxVec3& n : normals) {
        const TangentBasis b = MakeTangentBasis(n);
        const fx d0 = absfx(FxDot(n, b.t1));
        const fx d1 = absfx(FxDot(n, b.t2));
        const fx d2 = absfx(FxDot(b.t1, b.t2));
        if (d0 > m.maxDotErr) m.maxDotErr = d0;
        if (d1 > m.maxDotErr) m.maxDotErr = d1;
        if (d2 > m.maxDotErr) m.maxDotErr = d2;
        const fx l1 = fpx::FxLength(b.t1);
        const fx l2 = fpx::FxLength(b.t2);
        if (first) { m.minLen = l1; m.maxLen = l1; first = false; }
        if (l1 < m.minLen) m.minLen = l1;
        if (l1 > m.maxLen) m.maxLen = l1;
        if (l2 < m.minLen) m.minLen = l2;
        if (l2 > m.maxLen) m.maxLen = l2;
    }
    return m;
}

// =========================================================================================================
// Slice FC2 — THE FRICTION-AUGMENTED MANIFOLD POINT. APPENDED after MeasureBasis (FC1's lines above are
// BYTE-FROZEN). For a box-box pair, take the frozen CX2 contact manifold and pair every contact point with
// its FC1 tangent basis + zeroed (warm-start-ready) impulse accumulators, into a FrictionManifold — the
// per-contact solver STATE FC3's cone-clamped tangent-impulse solver consumes. PURE INTEGER, every order
// PINNED -> bit-identical CPU<->Vulkan<->Metal.
//
// THE int64 REALITY (the FC1/CX1 lesson): the manifold clip (BoxSatStable/BuildManifold) + the tangent-basis
// FxNormalize/FxDot/FxCross products are int64. DXC compiles int64 (Vulkan); glslc cannot. So the GPU shader
// shaders/fric_points.comp is VULKAN-SPIR-V-ONLY (NOT in hf_gen_msl); the Metal --fric-points runs the CPU
// BuildFrictionPoints below -> byte-identical to the Vulkan GPU result BY CONSTRUCTION (the FC1 convention),
// while the Vulkan side carries the GPU==CPU memcmp proof. fric_points.comp copies BuildFrictionPoints's body
// (and the BoxSatStable/BuildManifold/MakeTangentBasis it calls) VERBATIM, so the GPU FrictionPoint[] is
// byte-identical to the CPU reference.

// Pull the convex narrowphase types used by FC2 (re-use, do NOT redefine).
using convex::FxBox;
using convex::SatResult;
using convex::ContactManifold;

// ----- FrictionPoint: one contact point + the A->B normal + the FC1 tangent basis + zeroed accumulators ----
// point = a CX2 contact point; normal = the manifold normal SIGN-CORRECTED to point from A toward B (the
// SolveManifoldImpulse rule, applied ONCE per pair); (t1, t2) = MakeTangentBasis(normal) (the FC1 orthonormal
// tangent basis); normalImpulse / tangentImpulse1 / tangentImpulse2 = the three impulse accumulators, ALL
// ZEROED at build (the warm-start hooks FC3 fills; FC2 establishes the structure, the baseline re-solves from
// zero each tick). std430-packable as 12 x int32 (48 bytes) — the GPU FrictionPoint mirror memcmp's against.
struct FrictionPoint {
    FxVec3 point;
    FxVec3 normal;
    FxVec3 t1;
    FxVec3 t2;
    fx     normalImpulse   = 0;
    fx     tangentImpulse1 = 0;
    fx     tangentImpulse2 = 0;
};

// ----- FrictionManifold: a fixed-capacity count + 4 FrictionPoints (the CX2 ContactManifold capacity) ------
// count = the number of valid friction points (0 if separated, 1 for edge-edge, 1..4 for face). pts[0..count)
// are filled, the rest left default (zeroed).
struct FrictionManifold {
    uint32_t      count = 0;
    FrictionPoint pts[4];
};

// ----- BuildFrictionPoints(bodyA, boxA, bodyB, boxB): the frozen CX2 manifold + the FC1 basis per point -----
// The PINNED steps (every ordering decision fixed; the shader copies THIS body VERBATIM):
//   (1) sat = convex::BoxSatStable(bodyA, boxA, bodyB, boxB) (the CX4 face-preference SAT, frozen);
//   (2) !sat.overlap -> return {count = 0} (separated);
//   (3) m = convex::BuildManifold(bodyA, boxA, bodyB, boxB, sat);
//   (4) nAB = m.normal SIGN-CORRECTED to point A->B ONCE (flip if FxDot(m.normal, bodyB.pos-bodyA.pos) < 0 —
//       exactly the SolveManifoldImpulse rule);
//   (5) (t1, t2) = MakeTangentBasis(nAB) (computed ONCE — the normal is the same for every point of a pair);
//   (6) for each manifold point i in 0..m.count: pts[i] = {point = m.points[i], normal = nAB, t1, t2,
//       accumulators = 0}; count = m.count.
// PURE INTEGER, FIXED order -> bit-identical CPU<->Vulkan<->Metal. FC2 only BUILDS the state — applies NO
// impulse (that is FC3); the accumulators stay zero.
inline FrictionManifold BuildFrictionPoints(const fpx::FxBody& bodyA, const FxBox& boxA,
                                            const fpx::FxBody& bodyB, const FxBox& boxB) {
    FrictionManifold fm;
    const SatResult sat = convex::BoxSatStable(bodyA, boxA, bodyB, boxB);
    if (!sat.overlap) return fm;   // separated -> {count = 0}
    const ContactManifold m = convex::BuildManifold(bodyA, boxA, bodyB, boxB, sat);
    if (m.count == 0) return fm;   // degenerate (no kept points) -> empty
    // The A->B normal sign-correction (the SolveManifoldImpulse rule), applied ONCE per pair.
    FxVec3 nAB = m.normal;
    if (FxDot(nAB, fpx::FxSub(bodyB.pos, bodyA.pos)) < 0) nAB = FxVec3{-nAB.x, -nAB.y, -nAB.z};
    // The FC1 tangent basis at the (shared) A->B normal, computed ONCE.
    const TangentBasis tb = MakeTangentBasis(nAB);
    fm.count = m.count;
    for (uint32_t i = 0; i < m.count; ++i) {
        FrictionPoint& fp = fm.pts[i];
        fp.point  = m.points[i];
        fp.normal = nAB;
        fp.t1     = tb.t1;
        fp.t2     = tb.t2;
        fp.normalImpulse   = 0;   // ZEROED at build (the FC3 warm-start hooks)
        fp.tangentImpulse1 = 0;
        fp.tangentImpulse2 = 0;
    }
    return fm;
}

// ----- FrictionPointMeasure: the deterministic summary over a set of box pairs ----------------------------
// pairs            = total pairs measured; pairsWithContact = pairs reporting >=1 friction point; totalPoints
// = the sum of all friction points; maxDotErr = the max over all points of |n.t1| / |n.t2| / |t1.t2| (the
// basis orthogonality residual — 0 for a perfect basis, a small integer for the fixed-point drift). Pure
// integer, fixed order -> deterministic. The showcase prints + asserts.
struct FrictionPointMeasure {
    uint32_t pairs            = 0;
    uint32_t pairsWithContact = 0;
    uint32_t totalPoints      = 0;
    fx       maxDotErr        = 0;   // max |n.t1| / |n.t2| / |t1.t2| over all points
};

// MeasureFrictionPoints(pairs): BuildFrictionPoints over each pair, accumulate the deterministic summary.
// Pure integer, fixed order.
inline FrictionPointMeasure MeasureFrictionPoints(const std::vector<convex::SatPair>& pairs) {
    auto absfx = [](fx v) { return v < 0 ? -v : v; };
    FrictionPointMeasure pm;
    pm.pairs = (uint32_t)pairs.size();
    for (const convex::SatPair& p : pairs) {
        const FrictionManifold fm = BuildFrictionPoints(p.bodyA, p.boxA, p.bodyB, p.boxB);
        if (fm.count > 0) { ++pm.pairsWithContact; pm.totalPoints += fm.count; }
        for (uint32_t i = 0; i < fm.count; ++i) {
            const FrictionPoint& fp = fm.pts[i];
            const fx d0 = absfx(FxDot(fp.normal, fp.t1));
            const fx d1 = absfx(FxDot(fp.normal, fp.t2));
            const fx d2 = absfx(FxDot(fp.t1, fp.t2));
            if (d0 > pm.maxDotErr) pm.maxDotErr = d0;
            if (d1 > pm.maxDotErr) pm.maxDotErr = d1;
            if (d2 > pm.maxDotErr) pm.maxDotErr = d2;
        }
    }
    return pm;
}

// =========================================================================================================
// Slice FC3 — THE CONE-CLAMPED TANGENT-IMPULSE SOLVER (where friction BITES). APPENDED after
// MeasureFrictionPoints (FC1/FC2's lines above are BYTE-FROZEN). FC1 built the tangent basis, FC2 the
// per-contact FrictionPoint[] state. FC3 adds the COULOMB FRICTION CONE: a tangent impulse at each contact,
// clamped to +-mu*jn (mu x the normal impulse), applied to BOTH linear AND angular velocity through the
// inertia tensor — so a box sliding on a static box has its slide ARRESTED (static cone) / DECELERATED
// (kinetic cone), and the contact drag produces TORQUE. INTEGER bit-exact, every order PINNED.
//
// The NORMAL part of SolveFrictionImpulse REPRODUCES convex::SolveManifoldImpulse (convex.h:651) EXACTLY —
// same order, same int64 math — so mu=0 (zero friction cone) leaves a body BYTE-IDENTICAL to
// convex::ResolveContactPair (the make-or-break control). We do NOT call/modify the frozen
// SolveManifoldImpulse; we reproduce its normal body, then add the cone-clamped tangent sweep.
//
// THE int64 REALITY (the CX3/FC1 lesson): the inertia fxdiv + the FxDot/FxCross/FxMat3MulVec Q16.16 products
// are int64. DXC compiles int64 (Vulkan); glslc cannot. So shaders/fric_solve.comp is VULKAN-SPIR-V-ONLY
// (NOT in hf_gen_msl); the Metal --fric-solve runs the CPU ResolveContactFriction -> byte-identical to the
// Vulkan GPU result BY CONSTRUCTION, while the Vulkan side carries the GPU==CPU memcmp proof. fric_solve.comp
// copies ResolveContactFriction's body (and the BoxSatStable/BuildManifold/MakeTangentBasis/inertia/solve it
// calls) VERBATIM, so the GPU resolved bodies are byte-identical to the CPU reference.

// Pull the convex inertia + solve types FC3 uses (re-use, do NOT redefine).
using convex::FxMat3;
using convex::FxMat3MulVec;
using convex::FxBoxInvInertiaBody;
using convex::WorldInvInertia;
using fpx::FxScale;
using fpx::FxAdd;
using fpx::fxmul;
using fpx::fxdiv;

// ----- FricSolveConfig: the friction-solve parameters (the CX3 ContactSolveConfig shape + mu) ------------
// restitution = the Q16.16 bounce factor (0 = fully inelastic; the FC3 showcase uses 0); mu = the Q16.16
// Coulomb friction coefficient (the cone half-angle: |jt| <= mu*jn; mu=0 -> frictionless == CX3); iters =
// the Gauss-Seidel sweep count over the manifold points.
struct FricSolveConfig {
    fx       restitution = 0;   // Q16.16 bounce factor
    fx       mu          = 0;   // Q16.16 Coulomb friction coefficient (0 = frictionless)
    uint32_t iters       = 4;   // Gauss-Seidel sweeps over the friction points
};

// ----- SolveFrictionImpulse: the combined normal + cone-clamped tangent Gauss-Seidel over a FrictionManifold
// Mutates bodyA/bodyB vel+angVel AND the fm accumulators. The Gauss-Seidel order is PINNED: `iters` sweeps;
// each sweep iterates the friction points 0..count-1 in order; within a point: NORMAL first, then t1, then
// t2; mutating vel/angVel IN PLACE (later points/tangents see the earlier updates). Per point:
//   rA = p - bodyA.pos, rB = p - bodyB.pos (world lever arms); vpA = vA + wA x rA, vpB = vB + wB x rB.
//   (NORMAL — the CX3 convex.h:662-689 form reproduced VERBATIM):
//     vn = (vpB - vpA).n; if vn >= 0 SKIP (separating/resting -> jn := 0, the cone collapses, no tangent).
//     kn = invMa + invMb + n.((Iinv_a*(rA x n)) x rA) + n.((Iinv_b*(rB x n)) x rB); if kn <= 0 skip.
//     jn = -(kOne+restitution)*vn / kn, clamp jn >= 0; J = n*jn; apply (vA -= J*invMa; wA -= Iinv_a*(rA x J);
//          vB += ...; wB += ...).
//   (TANGENT — THE NEW PHYSICS — for t1 THEN t2, recomputing the contact-point velocities AFTER the normal
//    apply, sequential-impulse): vt = (vpB - vpA).t; kt = invMa + invMb + t.((Iinv_a*(rA x t)) x rA) +
//    t.((Iinv_b*(rB x t)) x rB) (the same form as kn with t for n); jt = -vt / kt; CLAMP to the Coulomb cone
//    jt = clamp(jt, -mu*jn, +mu*jn) (mu x THIS sweep's jn — the coupled-iteration approximation); Jt = t*jt;
//    apply (vA -= Jt*invMa; wA -= Iinv_a*(rA x Jt); vB += ...; wB += ...).
//   Store the last jn into fp.normalImpulse, the last jt into fp.tangentImpulse1/2 (diagnostic warm hooks).
// PURE INTEGER, FIXED order -> bit-identical CPU<->Vulkan<->Metal. The shader copies THIS body VERBATIM.
inline void SolveFrictionImpulse(fpx::FxBody& bodyA, fpx::FxBody& bodyB,
                                 const FxMat3& invIaW, const FxMat3& invIbW,
                                 FrictionManifold& fm, fx restitution, fx mu, uint32_t iters) {
    if (fm.count == 0) return;

    const fx invMassA = bodyA.invMass;
    const fx invMassB = bodyB.invMass;

    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t pi = 0; pi < fm.count; ++pi) {
            FrictionPoint& fp = fm.pts[pi];
            const FxVec3 p  = fp.point;
            const FxVec3 n  = fp.normal;   // already A->B (BuildFrictionPoints sign-corrected it ONCE)
            const FxVec3 rA = fpx::FxSub(p, bodyA.pos);
            const FxVec3 rB = fpx::FxSub(p, bodyB.pos);

            // ---- NORMAL impulse (the CX3 SolveManifoldImpulse form reproduced VERBATIM) ----
            const FxVec3 vpA0 = FxAdd(bodyA.vel, FxCross(bodyA.angVel, rA));
            const FxVec3 vpB0 = FxAdd(bodyB.vel, FxCross(bodyB.angVel, rB));
            const fx vn = FxDot(fpx::FxSub(vpB0, vpA0), n);
            fx jn = 0;
            if (vn < 0) {
                const FxVec3 raxn = FxCross(rA, n);
                const FxVec3 rbxn = FxCross(rB, n);
                const fx angA = FxDot(n, FxCross(FxMat3MulVec(invIaW, raxn), rA));
                const fx angB = FxDot(n, FxCross(FxMat3MulVec(invIbW, rbxn), rB));
                const fx kn = invMassA + invMassB + angA + angB;
                if (kn > 0) {
                    jn = fxdiv(-fxmul(kOne + restitution, vn), kn);
                    if (jn < 0) jn = 0;
                    const FxVec3 J = FxScale(n, jn);
                    bodyA.vel = fpx::FxSub(bodyA.vel, FxScale(J, invMassA));
                    bodyA.angVel = fpx::FxSub(bodyA.angVel, FxMat3MulVec(invIaW, FxCross(rA, J)));
                    bodyB.vel = FxAdd(bodyB.vel, FxScale(J, invMassB));
                    bodyB.angVel = FxAdd(bodyB.angVel, FxMat3MulVec(invIbW, FxCross(rB, J)));
                }
            }
            fp.normalImpulse = jn;

            // ---- TANGENT friction impulses (THE NEW PHYSICS), t1 then t2, cone-clamped to +-mu*jn ----
            const fx coneLo = -fxmul(mu, jn);   // -mu*jn
            const fx coneHi =  fxmul(mu, jn);   // +mu*jn (jn >= 0 -> coneHi >= 0 >= coneLo)
            const FxVec3 tangents[2] = {fp.t1, fp.t2};
            fx jtOut[2] = {0, 0};
            for (int ti = 0; ti < 2; ++ti) {
                const FxVec3 t = tangents[ti];
                // recompute the contact-point velocities AFTER the normal (and prior tangent) apply.
                const FxVec3 vpA = FxAdd(bodyA.vel, FxCross(bodyA.angVel, rA));
                const FxVec3 vpB = FxAdd(bodyB.vel, FxCross(bodyB.angVel, rB));
                const fx vt = FxDot(fpx::FxSub(vpB, vpA), t);
                const FxVec3 raxt = FxCross(rA, t);
                const FxVec3 rbxt = FxCross(rB, t);
                const fx angA = FxDot(t, FxCross(FxMat3MulVec(invIaW, raxt), rA));
                const fx angB = FxDot(t, FxCross(FxMat3MulVec(invIbW, rbxt), rB));
                const fx kt = invMassA + invMassB + angA + angB;
                if (kt <= 0) { jtOut[ti] = 0; continue; }
                fx jt = fxdiv(-vt, kt);
                if (jt < coneLo) jt = coneLo;
                else if (jt > coneHi) jt = coneHi;   // CLAMP to the Coulomb cone +-mu*jn
                const FxVec3 Jt = FxScale(t, jt);
                bodyA.vel = fpx::FxSub(bodyA.vel, FxScale(Jt, invMassA));
                bodyA.angVel = fpx::FxSub(bodyA.angVel, FxMat3MulVec(invIaW, FxCross(rA, Jt)));
                bodyB.vel = FxAdd(bodyB.vel, FxScale(Jt, invMassB));
                bodyB.angVel = FxAdd(bodyB.angVel, FxMat3MulVec(invIbW, FxCross(rB, Jt)));
                jtOut[ti] = jt;
            }
            fp.tangentImpulse1 = jtOut[0];
            fp.tangentImpulse2 = jtOut[1];
        }
    }
}

// ----- ResolveContactFriction(bodyA, boxA, bodyB, boxB, cfg): one box-box pair end-to-end ----------------
// BuildFrictionPoints (FC2: BoxSatStable -> BuildManifold -> the A->B basis) -> if count==0 return (no
// overlap / degenerate) -> the world inertias (convex::FxBoxInvInertiaBody + WorldInvInertia, frozen) ->
// SolveFrictionImpulse. VELOCITY-ONLY (NO position de-penetration — that is FC4). The shader copies THIS
// body VERBATIM so the GPU resolved-body result is byte-identical to the CPU.
inline void ResolveContactFriction(fpx::FxBody& bodyA, const FxBox& boxA,
                                   fpx::FxBody& bodyB, const FxBox& boxB,
                                   const FricSolveConfig& cfg) {
    FrictionManifold fm = BuildFrictionPoints(bodyA, boxA, bodyB, boxB);
    if (fm.count == 0) return;   // separated / degenerate -> no-op
    const FxVec3 invIa = FxBoxInvInertiaBody(boxA, bodyA.invMass);
    const FxVec3 invIb = FxBoxInvInertiaBody(boxB, bodyB.invMass);
    const FxMat3 invIaW = WorldInvInertia(bodyA, invIa);
    const FxMat3 invIbW = WorldInvInertia(bodyB, invIb);
    SolveFrictionImpulse(bodyA, bodyB, invIaW, invIbW, fm, cfg.restitution, cfg.mu, cfg.iters);
}

}  // namespace fric
}  // namespace hf::sim
