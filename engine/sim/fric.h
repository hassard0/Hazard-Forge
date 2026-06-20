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

}  // namespace fric
}  // namespace hf::sim
