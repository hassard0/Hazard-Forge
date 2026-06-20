#pragma once
// Slice CX1 — Deterministic Convex Rigid-Body Contacts: THE BOX-BOX SAT OVERLAP TEST (the BEACHHEAD of
// FLAGSHIP #19: DETERMINISTIC CONVEX RIGID-BODY CONTACTS — SAT manifold + inertia tensor + angular
// impulse, hf::sim::convex). The whole deterministic-sim suite (fpx/cloth/fluid/grain/couple*/fract/
// vehicle/joint) shares ONE caveat in SEVEN headers: contacts are SPHERE-SPHERE with NO inertia tensor /
// no torque-from-contact. This flagship lifts that ceiling. CX1 is the narrowphase BEACHHEAD: a
// deterministic INTEGER box-box Separating-Axis Test (SAT) that, for a pair of ORIENTED boxes, computes
// the minimum-penetration axis + depth (or reports separation), bit-exact CPU<->Vulkan<->Metal.
//
// Header-only, namespace hf::sim::convex, #include "sim/fpx.h" READ-ONLY (do NOT modify fpx.h). Reuses the
// fpx Q16.16 toolbox VERBATIM: fx/fxmul/fxdiv/FxVec3/FxAdd/FxSub/FxDot... — wait, FxDot lives here? No:
// FxDot/FxNormalize/FxLength/FxRotate/FxISqrt are all in fpx.h. We add FxMat3 (the CX3 inertia-tensor type,
// minimal here) + FxCross (the fpx::FxRotate internal cross PROMOTED to a public helper — NOT copied from
// fpx.h source-modification; re-expressed here) + FxBox + SatResult + BoxSat + MeasureSat.
//
// THE int64 REALITY (the FPX3/FPX1 lesson, the honest proof-strength call): FxNormalize/FxISqrt + the
// FxDot/FxCross Q16.16 products are int64 (world-scale products overflow int32). DXC compiles int64 (the
// Vulkan path); glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL. So the GPU
// shader shaders/convex_sat.comp.hlsl is VULKAN-SPIR-V-ONLY (in the Vulkan compile list, NOT in the Metal
// hf_gen_msl list); the Metal --convex-sat runs the CPU BoxSat -> byte-identical to the Vulkan GPU result
// BY CONSTRUCTION (the fpx_solve.comp / boids_steer.comp convention), while the Vulkan side carries the
// GPU==CPU memcmp proof. convex_sat.comp copies BoxSat's body VERBATIM, so the GPU exercises the EXACT
// integer ops -> a divergence is exactly what the host GPU==CPU memcmp catches.
//
// FIXED AXIS ORDER (the determinism crux): the 15 candidate separating axes are iterated in ONE fixed
// order — 3 A-face normals (A's local X/Y/Z rotated to world), 3 B-face normals, then the 9 edge-edge
// crosses FxCross(faceAxisA_i, faceAxisB_j) in a fixed (i,j) order — with the SAME FxNormalize/FxDot/
// FxCross int64 ops and the SAME strict-< min-pen tie-break (ties keep the LOWEST axis index). A degenerate
// near-zero edge-cross (FxLength < kEdgeEps; parallel face axes) is SKIPPED deterministically (it cannot
// separate). Per-pair-disjoint (one GPU thread per box pair) -> race-free, two runs byte-identical.

#include <cstdint>
#include <vector>

#include "sim/fpx.h"   // read-only: fx/fxmul/fxdiv/kOne/kFrac, FxVec3/FxAdd/FxSub/FxScale, FxDot? (no —
                       // FxDot is added below; fpx.h has FxLength/FxISqrt/FxNormalize/FxQuat/FxBody/FxRotate)

namespace hf::sim {
namespace convex {

// Pull the fpx Q16.16 scalar + helpers into this namespace (re-use, do NOT redefine the fixed-point format).
using fpx::fx;
using fpx::kOne;
using fpx::kFrac;
using fpx::fxmul;
using fpx::fxdiv;
using fpx::FxVec3;
using fpx::FxBody;
using fpx::FxQuat;

// ----- FxDot: the Q16.16 dot product (int64 intermediate, the FxLength sum-of-squares discipline) --------
// a·b = (ax*bx + ay*by + az*bz) >> kFrac, each product an int64 (world-scale products overflow int32), the
// sum kept in int64 then a single arithmetic right shift -> Q16.16. Deterministic, identical CPU/HLSL.
inline fx FxDot(const FxVec3& a, const FxVec3& b) {
    int64_t d = (int64_t)a.x * (int64_t)b.x + (int64_t)a.y * (int64_t)b.y + (int64_t)a.z * (int64_t)b.z;
    return (fx)(d >> kFrac);
}

// ----- FxCross: the Q16.16 cross product (the fpx::FxRotate internal cross promoted to a public helper) ---
// a×b = (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx), each fxmul an int64 product (the SAME ops the
// FxRotate cross uses ~fpx.h:444). The edge-edge separating axes are FxCross(faceAxisA, faceAxisB).
inline FxVec3 FxCross(const FxVec3& a, const FxVec3& b) {
    return FxVec3{
        fxmul(a.y, b.z) - fxmul(a.z, b.y),
        fxmul(a.z, b.x) - fxmul(a.x, b.z),
        fxmul(a.x, b.y) - fxmul(a.y, b.x),
    };
}

// ----- FxMat3: the Q16.16 row-major 3x3 matrix (the inertia-tensor type, introduced for CX3) -------------
// Minimal here: identity, diagonal, mul-by-vector, transpose. CX3 (the angular-impulse solve) is the full
// consumer; CX1 only introduces the type + these helpers (YAGNI — no FxMat3Mul-mat / inverse until CX3).
// m[r*3 + c] is row r, column c. Q16.16 components.
struct FxMat3 {
    fx m[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
};

// FxMat3Identity(): the Q16.16 identity (diagonal kOne).
inline FxMat3 FxMat3Identity() {
    FxMat3 r;
    r.m[0] = kOne; r.m[4] = kOne; r.m[8] = kOne;
    return r;
}

// FxMat3Diagonal(d): a diagonal matrix with d.x/d.y/d.z on the diagonal (an axis-aligned inertia tensor).
inline FxMat3 FxMat3Diagonal(const FxVec3& d) {
    FxMat3 r;
    r.m[0] = d.x; r.m[4] = d.y; r.m[8] = d.z;
    return r;
}

// FxMat3MulVec(M, v) = M·v in Q16.16 (each row dotted with v via FxDot). int64 in the products.
inline FxVec3 FxMat3MulVec(const FxMat3& M, const FxVec3& v) {
    const FxVec3 r0{M.m[0], M.m[1], M.m[2]};
    const FxVec3 r1{M.m[3], M.m[4], M.m[5]};
    const FxVec3 r2{M.m[6], M.m[7], M.m[8]};
    return FxVec3{FxDot(r0, v), FxDot(r1, v), FxDot(r2, v)};
}

// FxMat3Transpose(M): the transpose (swap m[r*3+c] with m[c*3+r]). Pure integer copy, no products.
inline FxMat3 FxMat3Transpose(const FxMat3& M) {
    FxMat3 r;
    r.m[0] = M.m[0]; r.m[1] = M.m[3]; r.m[2] = M.m[6];
    r.m[3] = M.m[1]; r.m[4] = M.m[4]; r.m[5] = M.m[7];
    r.m[6] = M.m[2]; r.m[7] = M.m[5]; r.m[8] = M.m[8];
    return r;
}

// ----- FxBox: an oriented box collider (the body carries pos + orient; the box carries the half-extents) -
// halfExtents are the per-axis half-widths in Q16.16 (a 2x2x2 box has halfExtents {kOne,kOne,kOne}). The
// box is centered at its FxBody::pos, oriented by its FxBody::orient (the 3 world face axes = FxRotate of
// the local X/Y/Z). std430-packable as 3 x int32 (12 bytes).
struct FxBox {
    FxVec3 halfExtents;
};

// ----- SatResult: the box-box SAT outcome ----------------------------------------------------------------
// overlap=false -> the boxes are SEPARATED (some axis separates them); axisIndex/penetration/axis are
// then unused (0). overlap=true -> axisIndex is the 0..14 min-penetration axis index, penetration is the
// (non-negative) min depth over all axes, axis is the unit min-pen separating axis SIGNED toward B (so
// moving B by +axis*penetration separates the pair). std430-packable: overlap as a uint32, axisIndex
// uint32, penetration fx, axis 3 x int32 -> 6 x int32 (24 bytes) for the GPU mirror.
struct SatResult {
    bool     overlap     = false;
    uint32_t axisIndex   = 0;
    fx       penetration = 0;
    FxVec3   axis;
};

// The near-zero edge-cross epsilon: an edge-edge cross FxCross(faceAxisA_i, faceAxisB_j) whose FxLength is
// below this is DEGENERATE (the two face axes are ~parallel) -> that axis cannot separate -> SKIP it
// deterministically (the SAME threshold on CPU + GPU, byte-identical). ~1/256 of a Q16.16 unit.
inline constexpr fx kEdgeEps = kOne / 256;   // ~0.0039 world units of cross-product length

// BoxAxes(b): the 3 WORLD face axes of an oriented box (its local +X/+Y/+Z rotated by the body orient).
// Unit by construction (a unit orient rotates the unit local axes to unit world axes, up to the fixed-point
// FxRotate drift). The SAT projects onto these (the 3 A-face normals + the 3 B-face normals).
inline void BoxAxes(const FxBody& b, FxVec3 axesOut[3]) {
    axesOut[0] = fpx::FxRotate(b.orient, FxVec3{kOne, 0, 0});
    axesOut[1] = fpx::FxRotate(b.orient, FxVec3{0, kOne, 0});
    axesOut[2] = fpx::FxRotate(b.orient, FxVec3{0, 0, kOne});
}

// ProjectedRadius(L, axes, h): the projected half-width (radius) of a box onto the unit axis L:
// r = |L·axes[0]|·h.x + |L·axes[1]|·h.y + |L·axes[2]|·h.z (the standard OBB extent onto a direction).
// Each |L·axis_i| is a Q16.16 magnitude; the fxmul with the half-extent + the sum stay in Q16.16. int64
// inside FxDot/fxmul. The shader copies THIS body VERBATIM.
inline fx ProjectedRadius(const FxVec3& L, const FxVec3 axes[3], const FxVec3& h) {
    auto absfx = [](fx v) { return v < 0 ? -v : v; };
    const fx d0 = absfx(FxDot(L, axes[0]));
    const fx d1 = absfx(FxDot(L, axes[1]));
    const fx d2 = absfx(FxDot(L, axes[2]));
    return fxmul(d0, h.x) + fxmul(d1, h.y) + fxmul(d2, h.z);
}

// BoxSat(bodyA, boxA, bodyB, boxB): the 15-axis box-box Separating-Axis Test in Q16.16.
//
// The 15 candidate separating axes, in FIXED order:
//   index 0..2  : A's 3 world face normals (axA[0..2])
//   index 3..5  : B's 3 world face normals (axB[0..2])
//   index 6..14 : the 9 edge-edge crosses FxCross(axA[i], axB[j]) for i in 0..2, j in 0..2
//                 (index = 6 + i*3 + j) — a degenerate near-zero cross (FxLength < kEdgeEps) is SKIPPED.
// Per axis L: normalize L (FxNormalize — int64 FxISqrt; the face axes are already ~unit, the edge crosses
// are not), then rA = ProjectedRadius(L, axA, hA), rB = ProjectedRadius(L, axB, hB),
// s = |L·(b.pos - a.pos)|. The axis SEPARATES iff s > rA + rB -> return {overlap=false}. Else the
// penetration on L is pen = (rA + rB) - s; track the MINIMUM pen over all axes (strict-< scan, the LOWEST
// axis index wins ties — bit-reproducible). The result axis is that min-pen L SIGNED toward B (flip if
// L·(b.pos - a.pos) < 0). Pure integer, fixed order -> bit-identical CPU<->Vulkan<->Metal.
inline SatResult BoxSat(const FxBody& bodyA, const FxBox& boxA, const FxBody& bodyB, const FxBox& boxB) {
    FxVec3 axA[3], axB[3];
    BoxAxes(bodyA, axA);
    BoxAxes(bodyB, axB);
    const FxVec3 hA = boxA.halfExtents;
    const FxVec3 hB = boxB.halfExtents;
    const FxVec3 t = fpx::FxSub(bodyB.pos, bodyA.pos);   // center separation A->B

    auto absfx = [](fx v) { return v < 0 ? -v : v; };

    // The running minimum-penetration axis (strict-< scan, lowest-index tie-break).
    bool   found      = false;
    fx     minPen     = 0;
    uint32_t minIndex = 0;
    FxVec3 minAxis;

    // A single axis test: returns false (separated -> caller bails) or true (overlapping -> updates min).
    // skipDegenerate=true for the edge crosses (a near-zero L can't separate -> ignore, do NOT update min).
    auto testAxis = [&](const FxVec3& rawL, uint32_t index, bool skipDegenerate) -> bool {
        // Degenerate edge-cross guard: a near-zero raw cross is skipped (deterministic, FxLength<kEdgeEps).
        if (skipDegenerate) {
            const fx rawLen = fpx::FxLength(rawL);
            if (rawLen < kEdgeEps) return true;   // can't separate on this axis -> ignore it
        }
        const FxVec3 L = fpx::FxNormalize(rawL);   // unit axis (int64 FxISqrt/fxdiv)
        const fx rA = ProjectedRadius(L, axA, hA);
        const fx rB = ProjectedRadius(L, axB, hB);
        const fx s  = absfx(FxDot(L, t));
        const fx sum = rA + rB;
        if (s > sum) return false;                 // SEPARATED on this axis -> the whole test fails
        const fx pen = sum - s;                    // penetration depth along L
        if (!found || pen < minPen) {              // strict-<: lowest index keeps a tie
            found    = true;
            minPen   = pen;
            minIndex = index;
            // Sign the axis toward B: flip if L points away from (b.pos - a.pos).
            minAxis  = (FxDot(L, t) < 0) ? FxVec3{-L.x, -L.y, -L.z} : L;
        }
        return true;
    };

    // (1) A's 3 face normals (indices 0..2) — already unit, NOT skip-degenerate.
    for (uint32_t i = 0; i < 3; ++i)
        if (!testAxis(axA[i], i, false)) return SatResult{};   // separated
    // (2) B's 3 face normals (indices 3..5).
    for (uint32_t j = 0; j < 3; ++j)
        if (!testAxis(axB[j], 3 + j, false)) return SatResult{};
    // (3) the 9 edge-edge crosses (indices 6..14), FIXED (i,j) order, skip-degenerate.
    for (uint32_t i = 0; i < 3; ++i)
        for (uint32_t j = 0; j < 3; ++j) {
            const FxVec3 e = FxCross(axA[i], axB[j]);
            if (!testAxis(e, 6 + i * 3 + j, true)) return SatResult{};
        }

    // No axis separated -> the boxes OVERLAP. Report the min-pen axis + depth.
    SatResult r;
    r.overlap     = true;
    r.axisIndex   = minIndex;
    r.penetration = minPen;
    r.axis        = minAxis;
    return r;
}

// ----- A SAT pair (the showcase's deterministic box-pair array element) ----------------------------------
// A pair of oriented boxes: each an FxBody (pos + orient) + an FxBox (half-extents). The showcase builds a
// fixed array of these spanning separated / deep face-face / edge-edge / touching cases.
struct SatPair {
    FxBody bodyA;
    FxBox  boxA;
    FxBody bodyB;
    FxBox  boxB;
};

// SatMeasure: the deterministic summary of a set of SAT pairs — how many overlap + the mean/min penetration
// over the overlapping ones (pure integer; the showcase prints + cross-checks against truth).
struct SatMeasure {
    uint32_t pairs       = 0;   // total pairs measured
    uint32_t overlapping = 0;   // pairs reporting overlap=true
    fx       meanPen     = 0;   // mean penetration over the overlapping pairs (0 if none)
    fx       minPen      = 0;   // min penetration over the overlapping pairs (0 if none)
};

// MeasureSat(pairs): run BoxSat over each pair, count overlaps, accumulate the mean/min penetration. Pure
// integer, fixed order -> deterministic. (The mean is an integer divide by the overlap count.)
inline SatMeasure MeasureSat(const std::vector<SatPair>& pairs) {
    SatMeasure m;
    m.pairs = (uint32_t)pairs.size();
    int64_t penSum = 0;
    bool firstOverlap = true;
    for (const SatPair& p : pairs) {
        const SatResult r = BoxSat(p.bodyA, p.boxA, p.bodyB, p.boxB);
        if (r.overlap) {
            ++m.overlapping;
            penSum += (int64_t)r.penetration;
            if (firstOverlap || r.penetration < m.minPen) { m.minPen = r.penetration; firstOverlap = false; }
        }
    }
    if (m.overlapping > 0) m.meanPen = (fx)(penSum / (int64_t)m.overlapping);
    return m;
}

}  // namespace convex
}  // namespace hf::sim
